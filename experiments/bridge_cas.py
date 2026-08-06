"""
bridge_cas.py  —  Compress–Add–Smooth continual memory via Bridge Diffusion
=============================================================================

PyTorch implementation (autograd-compatible, GPU-ready).

Classes
-------
GaussianMixture   : K-component GM in R^d
ProtocolGrid      : L+1 GM node states on uniform grid [0,1]
ContinualMemory   : daily CAS loop + readout-time bookkeeping
ForgetMetrics     : raw / normalised / decomposed forgetting metrics

All tensor operations avoid in-place mutation for autograd safety.
The only non-differentiable component is Hungarian matching (scipy,
CPU-side, O(K^3) — negligible for moderate K).

Requires: torch, numpy (thin bridge for scipy), scipy (Hungarian only).
"""

from __future__ import annotations
import math
from typing import Optional, Dict, List, Tuple

import torch
from torch import Tensor
import numpy as np
from scipy.optimize import linear_sum_assignment

# ═══════════════════════════════════════════════════════════════════════
#  GaussianMixture
# ═══════════════════════════════════════════════════════════════════════

class GaussianMixture:
    """K-component Gaussian mixture in R^d.

    Parameters
    ----------
    weights : (K,)          mixture weights, must sum to 1
    means   : (K, d)        component means
    covs    : (K, d, d)     component covariance matrices (SPD)
    """

    def __init__(self, weights: Tensor, means: Tensor, covs: Tensor):
        assert weights.dim() == 1
        assert means.dim() == 2
        assert covs.dim() == 3
        assert weights.shape[0] == means.shape[0] == covs.shape[0]
        assert means.shape[1] == covs.shape[1] == covs.shape[2]
        self.weights = weights          # (K,)
        self.means   = means            # (K, d)
        self.covs    = covs             # (K, d, d)

    # ── properties ──────────────────────────────────────────────────

    @property
    def K(self) -> int:
        return self.weights.shape[0]

    @property
    def d(self) -> int:
        return self.means.shape[1]

    @property
    def device(self) -> torch.device:
        return self.weights.device

    @property
    def dtype(self) -> torch.dtype:
        return self.weights.dtype

    # ── moments ─────────────────────────────────────────────────────

    def overall_mean(self) -> Tensor:
        """Overall mean: μ = Σ_k π_k m_k.  Returns (d,)."""
        return torch.einsum('k, kd -> d', self.weights, self.means)

    def overall_cov(self) -> Tensor:
        """Overall covariance (law of total variance).  Returns (d, d)."""
        mu = self.overall_mean()                          # (d,)
        # within-component: E[Σ_k]
        within = torch.einsum('k, kij -> ij', self.weights, self.covs)
        # between-component: E[m_k m_k^T] - μ μ^T
        dm = self.means - mu.unsqueeze(0)                 # (K, d)
        between = torch.einsum('k, ki, kj -> ij', self.weights, dm, dm)
        return within + between

    # ── density evaluation ──────────────────────────────────────────

    def log_component_densities(self, x: Tensor) -> Tensor:
        """Log N(x; m_k, Σ_k) for each component.

        Parameters
        ----------
        x : (*, d)

        Returns
        -------
        (*, K)
        """
        d = self.d
        # Cholesky for numerical stability
        L_chol = torch.linalg.cholesky(self.covs)        # (K, d, d)
        # log|Σ_k| = 2 Σ log diag(L_k)
        log_det = 2.0 * torch.sum(
            torch.log(torch.diagonal(L_chol, dim1=-2, dim2=-1)), dim=-1
        )                                                 # (K,)

        diff = x.unsqueeze(-2) - self.means              # (*, K, d)
        # solve L_k z = diff  →  z = L_k^{-1} diff
        # L_chol is (K, d, d); diff.unsqueeze(-1) is (*, K, d, 1)
        # PyTorch broadcasts L_chol over the batch dims automatically.
        z = torch.linalg.solve_triangular(
            L_chol, diff.unsqueeze(-1), upper=False
        ).squeeze(-1)                                     # (*, K, d)
        mahal = torch.sum(z * z, dim=-1)                  # (*, K)

        log_norm = -0.5 * (d * math.log(2 * math.pi) + log_det)  # (K,)
        return log_norm - 0.5 * mahal                     # (*, K)

    def log_density(self, x: Tensor) -> Tensor:
        """Log p(x) = log Σ_k π_k N(x; m_k, Σ_k).

        Parameters
        ----------
        x : (*, d)

        Returns
        -------
        (*,)
        """
        log_comp = self.log_component_densities(x)        # (*, K)
        log_w = torch.log(self.weights)                   # (K,)
        return torch.logsumexp(log_comp + log_w, dim=-1)

    def density(self, x: Tensor) -> Tensor:
        """p(x).  Parameters and returns same as log_density."""
        return torch.exp(self.log_density(x))

    # ── sampling ────────────────────────────────────────────────────

    @torch.no_grad()
    def sample(self, n: int) -> Tensor:
        """Draw n i.i.d. samples.  Returns (n, d).  Not differentiable."""
        # choose components
        idx = torch.multinomial(self.weights, n, replacement=True)  # (n,)
        mu  = self.means[idx]                             # (n, d)
        cov = self.covs[idx]                              # (n, d, d)
        L_chol = torch.linalg.cholesky(cov)               # (n, d, d)
        z = torch.randn(n, self.d, device=self.device, dtype=self.dtype)
        return mu + torch.einsum('nij, nj -> ni', L_chol, z)

    # ── interpolation (class method) ───────────────────────────────

    @staticmethod
    def interpolate(gm_a: 'GaussianMixture',
                    gm_b: 'GaussianMixture',
                    alpha: float | Tensor) -> 'GaussianMixture':
        """Piecewise-linear interpolation: (1-α) gm_a + α gm_b.

        Operates component-wise on (weights, means, covs).
        Both GMs must have the same K and d with matched component ordering.
        """
        if isinstance(alpha, (int, float)):
            alpha = torch.tensor(alpha, device=gm_a.device, dtype=gm_a.dtype)
        a = alpha
        w = (1.0 - a) * gm_a.weights + a * gm_b.weights
        m = (1.0 - a) * gm_a.means   + a * gm_b.means
        c = (1.0 - a) * gm_a.covs    + a * gm_b.covs
        return GaussianMixture(w, m, c)

    # ── utilities ───────────────────────────────────────────────────

    def clone(self) -> 'GaussianMixture':
        return GaussianMixture(
            self.weights.clone(), self.means.clone(), self.covs.clone()
        )

    def detach(self) -> 'GaussianMixture':
        return GaussianMixture(
            self.weights.detach(), self.means.detach(), self.covs.detach()
        )

    def to(self, device=None, dtype=None) -> 'GaussianMixture':
        kw = {}
        if device is not None: kw['device'] = device
        if dtype  is not None: kw['dtype']  = dtype
        return GaussianMixture(
            self.weights.to(**kw), self.means.to(**kw), self.covs.to(**kw)
        )

    def __repr__(self):
        return (f"GaussianMixture(K={self.K}, d={self.d}, "
                f"device={self.device}, dtype={self.dtype})")


# ═══════════════════════════════════════════════════════════════════════
#  ProtocolGrid
# ═══════════════════════════════════════════════════════════════════════

class ProtocolGrid:
    """L segments on [0,1] described by L+1 GM node states.

    Node j sits at time t_j = j/L,  j = 0, ..., L.
    Between nodes the density is given by piecewise-linear interpolation
    of the GM parameters (weights, means, covs).
    """

    def __init__(self, nodes: List[GaussianMixture]):
        """
        Parameters
        ----------
        nodes : list of L+1 GaussianMixture objects
        """
        assert len(nodes) >= 2, "Need at least 2 nodes (L >= 1)"
        self.nodes = list(nodes)

    @property
    def L(self) -> int:
        """Number of segments."""
        return len(self.nodes) - 1

    @property
    def K(self) -> int:
        return self.nodes[0].K

    @property
    def d(self) -> int:
        return self.nodes[0].d

    def node_times(self) -> Tensor:
        """Uniform node times: [0, 1/L, 2/L, ..., 1]."""
        L = self.L
        return torch.linspace(0.0, 1.0, L + 1,
                              device=self.nodes[0].device,
                              dtype=self.nodes[0].dtype)

    # ── interpolation / replay ──────────────────────────────────────

    def evaluate_at(self, t: float | Tensor) -> GaussianMixture:
        """Evaluate piecewise-linear interpolant at time t ∈ [0, 1].

        Returns a GaussianMixture whose parameters are the linear blend
        of the two enclosing nodes.
        """
        L = self.L
        if isinstance(t, Tensor):
            t_val = t.item()
        else:
            t_val = float(t)

        # clamp to [0, 1]
        t_val = max(0.0, min(1.0, t_val))

        # find segment index
        j = int(t_val * L)
        j = min(j, L - 1)                     # handle t = 1.0 exactly

        t_j = j / L
        alpha = (t_val - t_j) * L              # ∈ [0, 1]
        alpha = max(0.0, min(1.0, alpha))      # numerical safety

        return GaussianMixture.interpolate(self.nodes[j],
                                           self.nodes[j + 1], alpha)

    # ── CAS operations ──────────────────────────────────────────────

    def compress(self) -> 'ProtocolGrid':
        """Step 1: rescale [0,1] → [0, L/(L+1)].

        The L+1 node GM states are unchanged; only their time labels
        are reinterpreted.  Returns a new ProtocolGrid whose nodes
        correspond to times {0, 1/(L+1), 2/(L+1), ..., L/(L+1)},
        but stored as-is (the grid semantics change in the add step).
        """
        # Node states are identical; this is a conceptual relabelling.
        # We return a copy so the caller can mutate freely.
        return ProtocolGrid([n.clone() for n in self.nodes])

    def add(self, gm_new: GaussianMixture) -> 'ProtocolGrid':
        """Step 2: append new-day target at t = 1.

        Precondition: self was produced by compress() and has L+1 nodes
        at times {0, 1/(L+1), ..., L/(L+1)}.

        Appends gm_new at t = 1, yielding L+2 nodes on a uniform grid
        of L+1 segments with spacing 1/(L+1).
        """
        new_nodes = [n.clone() for n in self.nodes] + [gm_new.clone()]
        return ProtocolGrid(new_nodes)

    def smooth(self, L_target: int) -> 'ProtocolGrid':
        """Step 3: rebin L+1 segments (L+2 nodes) → L_target segments.

        Evaluates the current piecewise-linear interpolant (whose nodes
        sit at the *current* uniform grid) at the L_target+1 target node
        times {0, 1/L_target, ..., 1}.  Since the current grid has L+2
        nodes at {0, 1/(L+1), ..., 1} where L+1 = self.L, the evaluation
        is just a linear blend of adjacent nodes.

        Returns
        -------
        ProtocolGrid with L_target+1 nodes.
        """
        L_aug = self.L                          # current number of segments
        new_nodes = []
        for j in range(L_target + 1):
            t_new = j / L_target if L_target > 0 else 0.0
            # find enclosing segment in augmented grid
            k_float = t_new * L_aug
            k = int(k_float)
            k = min(k, L_aug - 1)
            alpha = k_float - k
            alpha = max(0.0, min(1.0, alpha))
            gm_j = GaussianMixture.interpolate(
                self.nodes[k], self.nodes[k + 1], alpha
            )
            new_nodes.append(gm_j)
        return ProtocolGrid(new_nodes)

    # ── convenience ─────────────────────────────────────────────────

    @staticmethod
    def from_interpolant(gm_0: GaussianMixture,
                         gm_1: GaussianMixture,
                         L: int) -> 'ProtocolGrid':
        """Build initial protocol by linear interpolation from gm_0 to gm_1."""
        nodes = []
        for j in range(L + 1):
            alpha = j / L
            nodes.append(GaussianMixture.interpolate(gm_0, gm_1, alpha))
        return ProtocolGrid(nodes)

    def clone(self) -> 'ProtocolGrid':
        return ProtocolGrid([n.clone() for n in self.nodes])

    def __repr__(self):
        return (f"ProtocolGrid(L={self.L}, K={self.K}, d={self.d}, "
                f"n_nodes={len(self.nodes)})")


# ═══════════════════════════════════════════════════════════════════════
#  Rebinning matrix (for reference / diagnostics — not used in hot path)
# ═══════════════════════════════════════════════════════════════════════

def rebinning_matrix(L: int) -> Tensor:
    """Compute the (L+1) × (L+2) rebinning matrix W.

    W[j, k] is the interpolation weight of augmented node k when
    evaluating the interpolant at target node j.

    The augmented grid has L+2 nodes at {0, 1/(L+1), ..., 1}.
    The target grid has L+1 nodes at {0, 1/L, ..., 1}.
    """
    L_aug = L + 1    # number of augmented segments
    L_tgt = L        # number of target segments
    n_aug = L_aug + 1
    n_tgt = L_tgt + 1

    W = torch.zeros(n_tgt, n_aug, dtype=torch.float64)
    for j in range(n_tgt):
        t_new = j / L_tgt if L_tgt > 0 else 0.0
        k_float = t_new * L_aug
        k = int(k_float)
        k = min(k, L_aug - 1)
        alpha = k_float - k
        W[j, k]     = 1.0 - alpha
        W[j, k + 1] = alpha
    return W


# ═══════════════════════════════════════════════════════════════════════
#  ForgetMetrics
# ═══════════════════════════════════════════════════════════════════════

class ForgetMetrics:
    """Forgetting metrics for Gaussian mixtures (Eqs. 6–12 of the paper)."""

    @staticmethod
    def raw_mismatch(gm_a: GaussianMixture,
                     gm_b: GaussianMixture) -> Tensor:
        """Raw moment-based forgetting: ||Δμ||² + ||ΔΣ||_F².

        Returns a scalar tensor (differentiable w.r.t. GM parameters).
        """
        dmu = gm_a.overall_mean() - gm_b.overall_mean()
        dSig = gm_a.overall_cov() - gm_b.overall_cov()
        return torch.sum(dmu * dmu) + torch.sum(dSig * dSig)

    @staticmethod
    def amnesia_baseline(gm_prior: GaussianMixture,
                         gm_orig: GaussianMixture) -> Tensor:
        """Amnesia baseline: mismatch between prior and original target."""
        return ForgetMetrics.raw_mismatch(gm_prior, gm_orig)

    @staticmethod
    def normalised(raw: Tensor, amnesia: Tensor,
                   eps: float = 1e-15) -> Tensor:
        """Normalised forgetting: raw / amnesia."""
        return raw / (amnesia + eps)

    # ── decomposed metric (K > 1) ──────────────────────────────────

    @staticmethod
    def hungarian_match(means_a: Tensor, means_b: Tensor) -> np.ndarray:
        """Match components by pairwise mean distance.

        Parameters
        ----------
        means_a, means_b : (K, d)

        Returns
        -------
        perm : (K,) int array — perm[i] = index in b matched to component i in a.
        """
        with torch.no_grad():
            a = means_a.detach().cpu().numpy()
            b = means_b.detach().cpu().numpy()
        K = a.shape[0]
        cost = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                cost[i, j] = np.sum((a[i] - b[j]) ** 2)
        _, col_ind = linear_sum_assignment(cost)
        return col_ind

    @staticmethod
    def decomposed(gm_orig: GaussianMixture,
                   gm_replay: GaussianMixture) -> Dict[str, Tensor]:
        """Decomposed forgetting with Hungarian matching (Eq. 12).

        Returns dict with keys:
            F_mean, F_cov, F_weight, F_total  — scalar tensors
            per_comp_mean, per_comp_cov        — (K,) tensors
            perm                               — numpy int array
        """
        perm = ForgetMetrics.hungarian_match(gm_orig.means, gm_replay.means)
        perm_t = torch.tensor(perm, dtype=torch.long,
                              device=gm_replay.device)

        w_o = gm_orig.weights
        m_o = gm_orig.means
        c_o = gm_orig.covs
        w_r = gm_replay.weights[perm_t]
        m_r = gm_replay.means[perm_t]
        c_r = gm_replay.covs[perm_t]

        K = w_o.shape[0]

        # per-component errors
        dm = m_o - m_r                                     # (K, d)
        dc = c_o - c_r                                     # (K, d, d)
        per_comp_mean = torch.sum(dm * dm, dim=-1)         # (K,)
        per_comp_cov  = torch.sum(dc * dc, dim=(-2, -1))   # (K,)

        wbar = torch.maximum(w_o, w_r)                     # (K,)
        F_mean   = torch.sum(wbar * per_comp_mean)
        F_cov    = torch.sum(wbar * per_comp_cov)
        F_weight = torch.sum((w_o - w_r) ** 2)
        F_total  = F_mean + F_cov + F_weight

        return dict(
            F_mean=F_mean, F_cov=F_cov, F_weight=F_weight,
            F_total=F_total,
            per_comp_mean=per_comp_mean, per_comp_cov=per_comp_cov,
            perm=perm,
        )


# ═══════════════════════════════════════════════════════════════════════
#  ContinualMemory
# ═══════════════════════════════════════════════════════════════════════

class ContinualMemory:
    """Orchestrates the daily Compress–Add–Smooth loop.

    Parameters
    ----------
    gm_prior : GaussianMixture
        Prior distribution q^(0) at t = 0.
    L : int
        Segment budget (number of protocol segments; L+1 nodes).
    store_history : bool
        If True, keep original daily targets for metric computation.
    """

    def __init__(self, gm_prior: GaussianMixture, L: int, *,
                 store_history: bool = True):
        self.gm_prior = gm_prior.clone()
        self.L = L
        self.store_history = store_history

        self.protocol: Optional[ProtocolGrid] = None
        self.current_day: int = 0
        self.readout_times: Dict[int, float] = {}

        # original targets for metric evaluation
        self.history: List[GaussianMixture] = []

    # ── main entry point ────────────────────────────────────────────

    def incorporate(self, gm_new: GaussianMixture) -> None:
        """Incorporate one new daily distribution via CAS.

        After calling this method, self.current_day is incremented and
        the protocol is updated.
        """
        self.current_day += 1
        day = self.current_day

        if self.store_history:
            self.history.append(gm_new.detach().clone())

        if self.protocol is None:
            # Day 1: initialise via linear interpolation
            self.protocol = ProtocolGrid.from_interpolant(
                self.gm_prior, gm_new, self.L
            )
            self.readout_times[day] = 1.0
            return

        # ── Step 1: compress ────────────────────────────────────────
        compressed = self.protocol.compress()

        # ── Step 2: add ─────────────────────────────────────────────
        augmented = compressed.add(gm_new)

        # ── Step 3: smooth ──────────────────────────────────────────
        self.protocol = augmented.smooth(self.L)

        # ── update readout times ────────────────────────────────────
        ratio = self.L / (self.L + 1)
        new_rt = {}
        for m, t_old in self.readout_times.items():
            new_rt[m] = t_old * ratio
        new_rt[day] = 1.0
        self.readout_times = new_rt

    # ── replay / query ──────────────────────────────────────────────

    def replay(self, day_m: int) -> GaussianMixture:
        """Replay past day m: evaluate the protocol at readout time t_{m|n}.

        Returns a GaussianMixture (differentiable w.r.t. protocol nodes).
        """
        if day_m not in self.readout_times:
            raise KeyError(f"Day {day_m} not in readout_times "
                           f"(current day = {self.current_day})")
        t = self.readout_times[day_m]
        return self.protocol.evaluate_at(t)

    # ── forgetting metrics ──────────────────────────────────────────

    def forgetting_raw(self, day_m: int) -> Tensor:
        """Raw forgetting F_{m→n} for past day m."""
        gm_replay = self.replay(day_m)
        gm_orig   = self.history[day_m - 1]
        return ForgetMetrics.raw_mismatch(gm_replay, gm_orig)

    def forgetting_amnesia(self, day_m: int) -> Tensor:
        """Amnesia baseline for past day m."""
        gm_orig = self.history[day_m - 1]
        return ForgetMetrics.amnesia_baseline(self.gm_prior, gm_orig)

    def forgetting_normalised(self, day_m: int) -> Tensor:
        """Normalised forgetting F̄_{m→n} for past day m."""
        raw = self.forgetting_raw(day_m)
        amnesia = self.forgetting_amnesia(day_m)
        return ForgetMetrics.normalised(raw, amnesia)

    def forgetting_decomposed(self, day_m: int) -> Dict[str, Tensor]:
        """Decomposed forgetting (K > 1) for past day m."""
        gm_replay = self.replay(day_m)
        gm_orig   = self.history[day_m - 1]
        return ForgetMetrics.decomposed(gm_orig, gm_replay)

    def __repr__(self):
        return (f"ContinualMemory(L={self.L}, day={self.current_day}, "
                f"K={self.protocol.K if self.protocol else '?'}, "
                f"d={self.gm_prior.d})")


# ═══════════════════════════════════════════════════════════════════════
#  Convenience: daily distribution generators
# ═══════════════════════════════════════════════════════════════════════

def make_daily_gaussians_circle(
    n_days: int,
    R: float = 2.0,
    period: int = 50,
    cov_scale: float = 0.5,
    d: int = 2,
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device('cpu'),
) -> Tuple[List[GaussianMixture], Tensor]:
    """K=1 Gaussians with means on a circle.

    Returns (list of n_days GMs, (n_days, d) tensor of means).
    """
    dists = []
    means_list = []
    for m in range(1, n_days + 1):
        angle = 2 * math.pi * m / period
        mu = torch.zeros(d, dtype=dtype, device=device)
        mu[0] = R * math.cos(angle)
        mu[1] = R * math.sin(angle)
        cov = cov_scale * torch.eye(d, dtype=dtype, device=device)
        gm = GaussianMixture(
            weights=torch.ones(1, dtype=dtype, device=device),
            means=mu.unsqueeze(0),
            covs=cov.unsqueeze(0),
        )
        dists.append(gm)
        means_list.append(mu)
    return dists, torch.stack(means_list)


def make_daily_gaussians_linear(
    n_days: int,
    speed: float = 0.15,
    cov_scale: float = 0.5,
    d: int = 2,
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device('cpu'),
) -> Tuple[List[GaussianMixture], Tensor]:
    """K=1 Gaussians with means drifting linearly along x_1.

    Returns (list of n_days GMs, (n_days, d) tensor of means).
    """
    dists = []
    means_list = []
    for m in range(1, n_days + 1):
        mu = torch.zeros(d, dtype=dtype, device=device)
        mu[0] = m * speed
        cov = cov_scale * torch.eye(d, dtype=dtype, device=device)
        gm = GaussianMixture(
            weights=torch.ones(1, dtype=dtype, device=device),
            means=mu.unsqueeze(0),
            covs=cov.unsqueeze(0),
        )
        dists.append(gm)
        means_list.append(mu)
    return dists, torch.stack(means_list)


def make_daily_gmm_circle(
    n_days: int,
    K: int = 3,
    R: float = 2.0,
    r: float = 0.8,
    period: int = 50,
    cov_scale: float = 0.3,
    d: int = 2,
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device('cpu'),
) -> Tuple[List[GaussianMixture], Tensor]:
    """K-component GMM: mixture centre traces a circle, components
    are arranged in a regular polygon of radius r around the centre.

    Returns (list of n_days GMs, (n_days, K, d) tensor of component means).
    """
    dists = []
    all_means = []
    for m in range(1, n_days + 1):
        angle = 2 * math.pi * m / period
        centre = torch.zeros(d, dtype=dtype, device=device)
        centre[0] = R * math.cos(angle)
        centre[1] = R * math.sin(angle)

        comp_means = torch.zeros(K, d, dtype=dtype, device=device)
        for k in range(K):
            phi_k = angle + 2 * math.pi * k / K
            comp_means[k, 0] = centre[0] + r * math.cos(phi_k)
            comp_means[k, 1] = centre[1] + r * math.sin(phi_k)

        comp_covs = cov_scale * torch.eye(d, dtype=dtype, device=device)
        comp_covs = comp_covs.unsqueeze(0).expand(K, -1, -1).clone()

        weights = torch.ones(K, dtype=dtype, device=device) / K

        gm = GaussianMixture(weights=weights, means=comp_means, covs=comp_covs)
        dists.append(gm)
        all_means.append(comp_means)

    return dists, torch.stack(all_means)


# ═══════════════════════════════════════════════════════════════════════
#  Convenience: analysis utilities
# ═══════════════════════════════════════════════════════════════════════

def age_curve(
    mem: ContinualMemory, n_days: int, theta: float = 0.5,
) -> Dict[str, np.ndarray | int | None]:
    """Compute age-averaged normalised forgetting curve.

    Returns dict with:
        ages     : (max_age+1,) int array
        F_mu     : (max_age+1,) mean normalised forgetting at each age
        F_std    : (max_age+1,) std
        counts   : (max_age+1,) number of (m,n) pairs
        half_life: int or None
    """
    max_age = n_days - 1
    F_mu  = np.full(max_age + 1, np.nan)
    F_std = np.full(max_age + 1, np.nan)
    counts = np.zeros(max_age + 1, dtype=int)

    # precompute all normalised forgetting values
    F_norm_matrix = np.full((n_days, n_days), np.nan)

    # We need to replay every past day at every subsequent current day.
    # But ContinualMemory only stores the *current* protocol.
    # So we re-run the loop, computing metrics after each incorporation.
    # This is already done externally in the notebooks; here we compute
    # from the final-day protocol only (replay all stored days at day n).
    for m in range(1, n_days + 1):
        if m in mem.readout_times:
            try:
                raw = mem.forgetting_raw(m).item()
                amnesia = mem.forgetting_amnesia(m).item()
                fnorm = raw / amnesia if amnesia > 1e-15 else 0.0
                age = mem.current_day - m
                F_norm_matrix[m - 1, mem.current_day - 1] = fnorm
            except Exception:
                pass

    # average by age (using final-day column only)
    for a in range(max_age + 1):
        m = n_days - a
        if 1 <= m <= n_days:
            val = F_norm_matrix[m - 1, n_days - 1]
            if not np.isnan(val):
                F_mu[a] = val
                F_std[a] = 0.0
                counts[a] = 1

    # half-life
    half_life = None
    for a in range(len(F_mu)):
        if not np.isnan(F_mu[a]) and F_mu[a] >= theta:
            half_life = a
            break

    return dict(
        ages=np.arange(max_age + 1),
        F_mu=F_mu,
        F_std=F_std,
        counts=counts,
        half_life=half_life,
    )


def run_cl_loop(
    daily_dists: List[GaussianMixture],
    gm_prior: GaussianMixture,
    L: int,
    verbose_every: int = 25,
) -> Tuple[ContinualMemory, np.ndarray, np.ndarray, List[dict]]:
    """Run the full CAS loop and compute metrics at every step.

    Returns
    -------
    mem       : ContinualMemory (at final day)
    F_raw     : (n, n) array — raw forgetting F_{m→n}
    F_norm    : (n, n) array — normalised forgetting F̄_{m→n}
    snapshots : list of per-day dicts with readout_times, metrics
    """
    n = len(daily_dists)
    F_raw  = np.full((n, n), np.nan)
    F_norm = np.full((n, n), np.nan)
    snapshots = []

    mem = ContinualMemory(gm_prior, L, store_history=True)

    for day_idx in range(n):
        day = day_idx + 1
        mem.incorporate(daily_dists[day_idx])

        fr_dict, fn_dict = {}, {}
        for m in range(1, day + 1):
            if m in mem.readout_times:
                try:
                    raw_val = mem.forgetting_raw(m).item()
                    amnesia_val = mem.forgetting_amnesia(m).item()
                    fnorm = raw_val / amnesia_val if amnesia_val > 1e-15 else 0.0
                    fr_dict[m] = raw_val
                    fn_dict[m] = fnorm
                    F_raw[m - 1, day - 1]  = raw_val
                    F_norm[m - 1, day - 1] = fnorm
                except Exception:
                    pass

        snapshots.append(dict(
            day=day,
            readout_times=dict(mem.readout_times),
            f_raw=fr_dict,
            f_norm=fn_dict,
        ))

        if verbose_every and (day % verbose_every == 0 or day == n):
            worst = max(fn_dict.values()) if fn_dict else 0
            print(f"  day {day:3d}:  K={mem.protocol.K}  "
                  f"#mem={len(fn_dict)}  worst F̄={worst:.3f}")

    return mem, F_raw, F_norm, snapshots


def compute_age_curves(
    F_norm: np.ndarray, n: int, theta: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int | None]:
    """Compute age-averaged forgetting from the full F̄(m,n) matrix.

    Returns
    -------
    ages     : (max_age+1,)
    F_mu     : (max_age+1,)
    F_std    : (max_age+1,)
    counts   : (max_age+1,)
    half_life: int or None
    """
    max_age = n - 1
    F_mu  = np.full(max_age + 1, np.nan)
    F_std = np.full(max_age + 1, np.nan)
    counts = np.zeros(max_age + 1, dtype=int)

    for a in range(max_age + 1):
        vals = []
        for m in range(1, n + 1):
            col = m + a - 1
            if col < n and not np.isnan(F_norm[m - 1, col]):
                vals.append(F_norm[m - 1, col])
        if vals:
            F_mu[a]  = np.mean(vals)
            F_std[a] = np.std(vals) if len(vals) > 1 else 0.0
            counts[a] = len(vals)

    half_life = None
    for a in range(len(F_mu)):
        if not np.isnan(F_mu[a]) and F_mu[a] >= theta:
            half_life = a
            break

    return np.arange(max_age + 1), F_mu, F_std, counts, half_life
