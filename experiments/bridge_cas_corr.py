"""
bridge_cas_corr.py — corrected CAS utilities for TMLR revision Days 2--4.

This module preserves the deterministic Compress--Add--Smooth API used by the
submission notebooks and adds two revision-specific pieces:

1. distribution-sensitive, permutation-invariant replay diagnostics
   (sample-based sliced-Wasserstein and held-out log-likelihood), and
2. a corrected one-dimensional time-varying-weight SDE validation helper.

The Appendix A correction affects only the optional SDE realization of a
prescribed density path.  The deterministic CAS recursion and all marginal
protocol evaluations are unchanged.
"""

from __future__ import annotations
import math
from typing import Optional, Dict, List, Tuple, Iterable

import numpy as np
import torch
from torch import Tensor
from scipy.optimize import linear_sum_assignment
from scipy.stats import wasserstein_distance
from scipy.special import erf

# ---------------------------------------------------------------------
# Gaussian mixture
# ---------------------------------------------------------------------

class GaussianMixture:
    """K-component Gaussian mixture in R^d with labeled components."""

    def __init__(self, weights: Tensor, means: Tensor, covs: Tensor):
        assert weights.dim() == 1
        assert means.dim() == 2
        assert covs.dim() == 3
        assert weights.shape[0] == means.shape[0] == covs.shape[0]
        assert means.shape[1] == covs.shape[1] == covs.shape[2]
        self.weights = weights
        self.means = means
        self.covs = covs

    @property
    def K(self) -> int:
        return int(self.weights.shape[0])

    @property
    def d(self) -> int:
        return int(self.means.shape[1])

    @property
    def device(self) -> torch.device:
        return self.weights.device

    @property
    def dtype(self) -> torch.dtype:
        return self.weights.dtype

    def clone(self) -> "GaussianMixture":
        return GaussianMixture(self.weights.clone(), self.means.clone(), self.covs.clone())

    def detach(self) -> "GaussianMixture":
        return GaussianMixture(self.weights.detach(), self.means.detach(), self.covs.detach())

    def to(self, device=None, dtype=None) -> "GaussianMixture":
        kw = {}
        if device is not None:
            kw["device"] = device
        if dtype is not None:
            kw["dtype"] = dtype
        return GaussianMixture(self.weights.to(**kw), self.means.to(**kw), self.covs.to(**kw))

    def overall_mean(self) -> Tensor:
        return torch.einsum("k,kd->d", self.weights, self.means)

    def overall_cov(self) -> Tensor:
        mu = self.overall_mean()
        within = torch.einsum("k,kij->ij", self.weights, self.covs)
        dm = self.means - mu.unsqueeze(0)
        between = torch.einsum("k,ki,kj->ij", self.weights, dm, dm)
        return within + between

    def log_component_densities(self, x: Tensor) -> Tensor:
        """Log N(x; m_k,Sigma_k), returning shape (*,K)."""
        d = self.d
        L = torch.linalg.cholesky(self.covs)
        log_det = 2.0 * torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1)), dim=-1)
        diff = x.unsqueeze(-2) - self.means
        z = torch.linalg.solve_triangular(L, diff.unsqueeze(-1), upper=False).squeeze(-1)
        mahal = torch.sum(z * z, dim=-1)
        log_norm = -0.5 * (d * math.log(2.0 * math.pi) + log_det)
        return log_norm - 0.5 * mahal

    def log_density(self, x: Tensor) -> Tensor:
        log_comp = self.log_component_densities(x)
        log_w = torch.log(torch.clamp(self.weights, min=1e-300))
        return torch.logsumexp(log_comp + log_w, dim=-1)

    def density(self, x: Tensor) -> Tensor:
        return torch.exp(self.log_density(x))

    @torch.no_grad()
    def sample(self, n: int) -> Tensor:
        idx = torch.multinomial(self.weights, n, replacement=True)
        mu = self.means[idx]
        cov = self.covs[idx]
        L = torch.linalg.cholesky(cov)
        z = torch.randn(n, self.d, device=self.device, dtype=self.dtype)
        return mu + torch.einsum("nij,nj->ni", L, z)

    @staticmethod
    def interpolate(gm_a: "GaussianMixture", gm_b: "GaussianMixture", alpha: float | Tensor) -> "GaussianMixture":
        if isinstance(alpha, (int, float)):
            alpha = torch.tensor(float(alpha), device=gm_a.device, dtype=gm_a.dtype)
        w = (1.0 - alpha) * gm_a.weights + alpha * gm_b.weights
        m = (1.0 - alpha) * gm_a.means + alpha * gm_b.means
        c = (1.0 - alpha) * gm_a.covs + alpha * gm_b.covs
        w = w / torch.sum(w)
        return GaussianMixture(w, m, c)

    def __repr__(self) -> str:
        return f"GaussianMixture(K={self.K}, d={self.d}, device={self.device}, dtype={self.dtype})"


class ProtocolGrid:
    """L segments represented by L+1 labeled-GM nodes on [0,1]."""

    def __init__(self, nodes: List[GaussianMixture]):
        assert len(nodes) >= 2
        self.nodes = list(nodes)

    @property
    def L(self) -> int:
        return len(self.nodes) - 1

    @property
    def K(self) -> int:
        return self.nodes[0].K

    @property
    def d(self) -> int:
        return self.nodes[0].d

    def node_times(self) -> Tensor:
        return torch.linspace(0.0, 1.0, self.L + 1, dtype=self.nodes[0].dtype, device=self.nodes[0].device)

    def evaluate_at(self, t: float | Tensor) -> GaussianMixture:
        if isinstance(t, Tensor):
            t = float(t.item())
        else:
            t = float(t)
        t = max(0.0, min(1.0, t))
        j = min(int(t * self.L), self.L - 1)
        alpha = (t - j / self.L) * self.L
        alpha = max(0.0, min(1.0, alpha))
        return GaussianMixture.interpolate(self.nodes[j], self.nodes[j + 1], alpha)

    def compress(self) -> "ProtocolGrid":
        return ProtocolGrid([n.clone() for n in self.nodes])

    def add(self, gm_new: GaussianMixture) -> "ProtocolGrid":
        return ProtocolGrid([n.clone() for n in self.nodes] + [gm_new.clone()])

    def smooth(self, L_target: int) -> "ProtocolGrid":
        L_aug = self.L
        new_nodes = []
        for j in range(L_target + 1):
            t_new = j / L_target if L_target > 0 else 0.0
            k_float = t_new * L_aug
            k = min(int(k_float), L_aug - 1)
            alpha = max(0.0, min(1.0, k_float - k))
            new_nodes.append(GaussianMixture.interpolate(self.nodes[k], self.nodes[k + 1], alpha))
        return ProtocolGrid(new_nodes)

    @staticmethod
    def from_interpolant(gm_0: GaussianMixture, gm_1: GaussianMixture, L: int) -> "ProtocolGrid":
        nodes = [GaussianMixture.interpolate(gm_0, gm_1, j / L) for j in range(L + 1)]
        return ProtocolGrid(nodes)


def rebinning_matrix(L: int) -> Tensor:
    L_aug = L + 1
    n_aug = L_aug + 1
    n_tgt = L + 1
    W = torch.zeros(n_tgt, n_aug, dtype=torch.float64)
    for j in range(n_tgt):
        t_new = j / L if L > 0 else 0.0
        k_float = t_new * L_aug
        k = min(int(k_float), L_aug - 1)
        alpha = k_float - k
        W[j, k] = 1.0 - alpha
        W[j, k + 1] = alpha
    return W


class ForgetMetrics:
    @staticmethod
    def raw_mismatch(gm_a: GaussianMixture, gm_b: GaussianMixture) -> Tensor:
        dmu = gm_a.overall_mean() - gm_b.overall_mean()
        dS = gm_a.overall_cov() - gm_b.overall_cov()
        return torch.sum(dmu * dmu) + torch.sum(dS * dS)

    @staticmethod
    def amnesia_baseline(gm_prior: GaussianMixture, gm_orig: GaussianMixture) -> Tensor:
        return ForgetMetrics.raw_mismatch(gm_prior, gm_orig)

    @staticmethod
    def normalised(raw: Tensor, amnesia: Tensor, eps: float = 1e-15) -> Tensor:
        return raw / (amnesia + eps)

    @staticmethod
    def hungarian_match(means_a: Tensor, means_b: Tensor) -> np.ndarray:
        D = torch.cdist(means_a.detach().cpu(), means_b.detach().cpu()).numpy()
        row_ind, col_ind = linear_sum_assignment(D)
        perm = np.empty_like(col_ind)
        perm[row_ind] = col_ind
        return perm

    @staticmethod
    def decomposed(gm_orig: GaussianMixture, gm_replay: GaussianMixture) -> Dict[str, Tensor]:
        perm = ForgetMetrics.hungarian_match(gm_orig.means, gm_replay.means)
        perm_t = torch.tensor(perm, dtype=torch.long, device=gm_replay.device)
        w_o, m_o, c_o = gm_orig.weights, gm_orig.means, gm_orig.covs
        w_r, m_r, c_r = gm_replay.weights[perm_t], gm_replay.means[perm_t], gm_replay.covs[perm_t]
        dm = m_o - m_r
        dc = c_o - c_r
        per_comp_mean = torch.sum(dm * dm, dim=-1)
        per_comp_cov = torch.sum(dc * dc, dim=(-2, -1))
        wbar = torch.maximum(w_o, w_r)
        F_mean = torch.sum(wbar * per_comp_mean)
        F_cov = torch.sum(wbar * per_comp_cov)
        F_weight = torch.sum((w_o - w_r) ** 2)
        F_total = F_mean + F_cov + F_weight
        return dict(F_mean=F_mean, F_cov=F_cov, F_weight=F_weight, F_total=F_total,
                    per_comp_mean=per_comp_mean, per_comp_cov=per_comp_cov, perm=perm)


class ContinualMemory:
    """Daily CAS loop.  Production readout can be computed from age only."""

    def __init__(self, gm_prior: GaussianMixture, L: int, *, store_history: bool = True,
                 store_readout_times: bool = True):
        self.gm_prior = gm_prior.clone()
        self.L = int(L)
        self.store_history = store_history
        self.store_readout_times = store_readout_times
        self.protocol: Optional[ProtocolGrid] = None
        self.current_day = 0
        self.readout_times: Dict[int, float] = {}
        self.history: List[GaussianMixture] = []

    def readout_time_from_age(self, age: int) -> float:
        return (self.L / (self.L + 1.0)) ** int(age)

    def readout_time(self, day_m: int) -> float:
        age = self.current_day - int(day_m)
        return self.readout_time_from_age(age)

    def incorporate(self, gm_new: GaussianMixture) -> None:
        self.current_day += 1
        day = self.current_day
        if self.store_history:
            self.history.append(gm_new.detach().clone())
        if self.protocol is None:
            self.protocol = ProtocolGrid.from_interpolant(self.gm_prior, gm_new, self.L)
            if self.store_readout_times:
                self.readout_times[day] = 1.0
            return
        self.protocol = self.protocol.compress().add(gm_new).smooth(self.L)
        if self.store_readout_times:
            ratio = self.L / (self.L + 1.0)
            self.readout_times = {m: t * ratio for m, t in self.readout_times.items()}
            self.readout_times[day] = 1.0

    def replay(self, day_m: int) -> GaussianMixture:
        if self.protocol is None:
            raise RuntimeError("No protocol yet")
        # Do not require an O(n) readout dictionary; compute from age.
        t = self.readout_time(day_m)
        return self.protocol.evaluate_at(t)

    def forgetting_raw(self, day_m: int) -> Tensor:
        gm_orig = self.history[day_m - 1]
        gm_rep = self.replay(day_m)
        return ForgetMetrics.raw_mismatch(gm_orig, gm_rep)

    def forgetting_normalised(self, day_m: int) -> Tensor:
        gm_orig = self.history[day_m - 1]
        raw = self.forgetting_raw(day_m)
        base = ForgetMetrics.amnesia_baseline(self.gm_prior, gm_orig)
        return ForgetMetrics.normalised(raw, base)

    def forgetting_decomposed(self, day_m: int) -> Dict[str, Tensor]:
        return ForgetMetrics.decomposed(self.history[day_m - 1], self.replay(day_m))


# ---------------------------------------------------------------------
# Data generators and CAS experiment utilities
# ---------------------------------------------------------------------

def make_daily_gaussians_circle(n_days: int, R: float = 2.0, period: float = 50.0,
                                cov_scale: float = 0.5, d: int = 2) -> Tuple[List[GaussianMixture], Tensor]:
    dists, means = [], []
    for m in range(1, n_days + 1):
        mu = torch.zeros(d, dtype=torch.get_default_dtype())
        mu[0] = R * math.cos(2.0 * math.pi * m / period)
        if d >= 2:
            mu[1] = R * math.sin(2.0 * math.pi * m / period)
        cov = cov_scale * torch.eye(d)
        gm = GaussianMixture(torch.ones(1, dtype=mu.dtype), mu.view(1, d), cov.view(1, d, d))
        dists.append(gm)
        means.append(mu)
    return dists, torch.stack(means, dim=0)


def make_daily_gaussians_linear(n_days: int, R: float = 2.0, cov_scale: float = 0.5, d: int = 2) -> Tuple[List[GaussianMixture], Tensor]:
    dists, means = [], []
    xs = torch.linspace(-R, R, n_days, dtype=torch.get_default_dtype())
    for i in range(n_days):
        mu = torch.zeros(d, dtype=torch.get_default_dtype())
        mu[0] = xs[i]
        cov = cov_scale * torch.eye(d)
        dists.append(GaussianMixture(torch.ones(1, dtype=mu.dtype), mu.view(1, d), cov.view(1, d, d)))
        means.append(mu)
    return dists, torch.stack(means, dim=0)


def make_daily_gmm_circle(n_days: int, K: int = 3, R: float = 2.0, r: float = 0.8,
                          period: float = 50.0, cov_scale: float = 0.3, d: int = 2) -> Tuple[List[GaussianMixture], Tensor]:
    dists, all_means = [], []
    dtype = torch.get_default_dtype()
    for m in range(1, n_days + 1):
        center = torch.zeros(d, dtype=dtype)
        center[0] = R * math.cos(2.0 * math.pi * m / period)
        if d >= 2:
            center[1] = R * math.sin(2.0 * math.pi * m / period)
        means = []
        rot = 2.0 * math.pi * m / period
        for k in range(K):
            theta = rot + 2.0 * math.pi * k / K
            mk = center.clone()
            mk[0] += r * math.cos(theta)
            if d >= 2:
                mk[1] += r * math.sin(theta)
            means.append(mk)
        means = torch.stack(means)
        covs = cov_scale * torch.eye(d, dtype=dtype).unsqueeze(0).expand(K, -1, -1).clone()
        weights = torch.ones(K, dtype=dtype) / K
        dists.append(GaussianMixture(weights, means, covs))
        all_means.append(means)
    return dists, torch.stack(all_means, dim=0)


def run_cl_loop(daily_dists: List[GaussianMixture], gm_prior: GaussianMixture, L: int,
                verbose_every: int = 0, store_snapshots: bool = False):
    n_days = len(daily_dists)
    mem = ContinualMemory(gm_prior, L, store_history=True, store_readout_times=True)
    Fraw = np.full((n_days + 1, n_days + 1), np.nan)
    Fnorm = np.full((n_days + 1, n_days + 1), np.nan)
    snaps = []
    for i, gm in enumerate(daily_dists, start=1):
        mem.incorporate(gm)
        if store_snapshots:
            snaps.append(mem.protocol)
        for m in range(1, i + 1):
            raw = mem.forgetting_raw(m).detach().cpu().item()
            base = ForgetMetrics.amnesia_baseline(gm_prior, mem.history[m - 1]).detach().cpu().item()
            Fraw[m, i] = raw
            Fnorm[m, i] = raw / (base + 1e-15)
        if verbose_every and (i % verbose_every == 0):
            print(f"  day {i}/{n_days}")
    return mem, Fraw, Fnorm, snaps


def compute_age_curves(Fnorm: np.ndarray, n_days: int, theta: float = 0.5):
    ages = np.arange(n_days)
    F_mu = np.full(n_days, np.nan)
    F_std = np.full(n_days, np.nan)
    counts = np.zeros(n_days, dtype=int)
    for a in ages:
        vals = []
        for m in range(1, n_days - a + 1):
            n = m + a
            v = Fnorm[m, n]
            if not np.isnan(v):
                vals.append(v)
        if vals:
            vals = np.asarray(vals, dtype=float)
            F_mu[a] = np.mean(vals)
            F_std[a] = np.std(vals)
            counts[a] = len(vals)
    hl = None
    for a in range(1, n_days):
        if not np.isnan(F_mu[a]) and F_mu[a] >= theta:
            hl = int(a)
            break
    return ages, F_mu, F_std, counts, hl


# ---------------------------------------------------------------------
# Distribution-sensitive diagnostics
# ---------------------------------------------------------------------

def _torch_from_np(x: np.ndarray, dtype: torch.dtype = torch.float64) -> Tensor:
    return torch.as_tensor(x, dtype=dtype)


def random_unit_directions(d: int, n_dirs: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    V = rng.normal(size=(n_dirs, d))
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
    return V


def sliced_wasserstein2_samples(x: np.ndarray, y: np.ndarray, directions: np.ndarray) -> float:
    """Sample estimate of sliced W2^2."""
    px = x @ directions.T
    py = y @ directions.T
    px.sort(axis=0)
    py.sort(axis=0)
    n = min(px.shape[0], py.shape[0])
    return float(np.mean((px[:n] - py[:n]) ** 2))


def heldout_nll_delta(gm_ref: GaussianMixture, gm_model: GaussianMixture,
                      samples_ref: Tensor) -> float:
    """E_{x~ref}[-log model(x) + log ref(x)], estimated on samples_ref."""
    with torch.no_grad():
        nll_model = -gm_model.log_density(samples_ref).mean().item()
        nll_ref = -gm_ref.log_density(samples_ref).mean().item()
    return float(nll_model - nll_ref)


def compute_distribution_metric_matrices(daily_dists: List[GaussianMixture], gm_prior: GaussianMixture,
                                         L: int, *, n_samples: int = 256, n_dirs: int = 32,
                                         seed: int = 0, verbose_every: int = 0):
    """Run CAS and compute normalized SW2 and held-out NLL matrices.

    Normalization uses the corresponding prior-vs-original amnesia baseline.
    These metrics are sample-based and permutation-invariant at the density level.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    n_days = len(daily_dists)
    d = daily_dists[0].d
    dirs = random_unit_directions(d, n_dirs, seed=seed)

    # Fixed held-out/reference samples per day and prior samples for baselines.
    orig_samples_t, orig_samples_np = [], []
    for gm in daily_dists:
        s = gm.sample(n_samples).detach()
        orig_samples_t.append(s)
        orig_samples_np.append(s.cpu().numpy())
    prior_samples_np = gm_prior.sample(n_samples).detach().cpu().numpy()

    sw_base = np.zeros(n_days + 1)
    nll_base = np.zeros(n_days + 1)
    for m, gm_orig in enumerate(daily_dists, start=1):
        sw_base[m] = sliced_wasserstein2_samples(prior_samples_np, orig_samples_np[m - 1], dirs)
        nll_base[m] = max(heldout_nll_delta(gm_orig, gm_prior, orig_samples_t[m - 1]), 1e-12)
        sw_base[m] = max(sw_base[m], 1e-12)

    Fsw = np.full((n_days + 1, n_days + 1), np.nan)
    Fnll = np.full((n_days + 1, n_days + 1), np.nan)
    mem = ContinualMemory(gm_prior, L, store_history=True, store_readout_times=True)
    for n, gm_new in enumerate(daily_dists, start=1):
        mem.incorporate(gm_new)
        for m in range(1, n + 1):
            gm_rep = mem.replay(m)
            rep_np = gm_rep.sample(n_samples).detach().cpu().numpy()
            Fsw[m, n] = sliced_wasserstein2_samples(rep_np, orig_samples_np[m - 1], dirs) / sw_base[m]
            Fnll[m, n] = heldout_nll_delta(daily_dists[m - 1], gm_rep, orig_samples_t[m - 1]) / nll_base[m]
        if verbose_every and n % verbose_every == 0:
            print(f"  distribution metrics day {n}/{n_days}")
    return mem, Fsw, Fnll


# ---------------------------------------------------------------------
# Corrected time-varying-weight SDE validation helper (1D)
# ---------------------------------------------------------------------

def _normal_pdf_np(x, mean, sigma):
    return np.exp(-0.5 * ((x - mean) / sigma) ** 2) / (math.sqrt(2.0 * math.pi) * sigma)


def _normal_cdf_np(x, mean, sigma):
    return 0.5 * (1.0 + erf((x - mean) / (math.sqrt(2.0) * sigma)))


def weight_bridge_params(t: float, amp: float = 0.3, omega: float = 2.0 * math.pi):
    """Two-component 1D bridge with fixed means/covariances and monotone varying weights.

    We keep the unused amp/omega arguments for backward compatibility with earlier
    calls.  The path is pi_1(t)=0.2+0.6t, pi_2(t)=0.8-0.6t, so the density
    changes only through a zero-mass time derivative of the weights.
    """
    p1 = 0.2 + 0.6 * float(t)
    dp1 = 0.6
    return np.array([p1, 1.0 - p1]), np.array([dp1, -dp1])


def corrected_weight_bridge_drift_1d(x: np.ndarray, t: float, *, means=(-2.0, 2.0), sigma: float = 0.7,
                                     amp: float = 0.3, old_wrong_sign: bool = False) -> np.ndarray:
    """Drift for a 1D GM path with time-varying weights.

    Correct current: J_wt(x,t) = - sum_k dot(pi_k) Phi_k(x), which gives
    dJ/dx = - sum_k dot(pi_k) g_k(x) and therefore satisfies the continuity
    equation for weight changes.  The old incorrect Appendix sign/factor is
    included only as a diagnostic option.
    """
    x = np.asarray(x, dtype=float)
    pis, dpis = weight_bridge_params(t, amp=amp)
    means = np.asarray(means, dtype=float)
    pdfs = np.stack([_normal_pdf_np(x, m, sigma) for m in means], axis=0)
    cdfs = np.stack([_normal_cdf_np(x, m, sigma) for m in means], axis=0)
    p = np.sum(pis[:, None] * pdfs, axis=0)
    score_num = np.sum(pis[:, None] * pdfs * (-(x[None, :] - means[:, None]) / sigma ** 2), axis=0)
    score = score_num / np.maximum(p, 1e-300)
    J_corr = -np.sum(dpis[:, None] * cdfs, axis=0)
    if old_wrong_sign:
        J = -0.5 * J_corr
    else:
        J = J_corr
    return J / np.maximum(p, 1e-300) + 0.5 * score


def sample_weight_bridge_target(n: int, t: float, *, means=(-2.0, 2.0), sigma: float = 0.7,
                                amp: float = 0.3, seed: Optional[int] = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pis, _ = weight_bridge_params(t, amp=amp)
    comp = rng.choice(2, size=n, p=pis)
    means = np.asarray(means)
    return means[comp] + sigma * rng.normal(size=n)


def simulate_weight_bridge_sde(n_paths: int = 4000, n_steps: int = 500, *, seed: int = 0,
                               old_wrong_sign: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ts = np.linspace(0.0, 1.0, n_steps + 1)
    dt = 1.0 / n_steps
    x = sample_weight_bridge_target(n_paths, 0.0, seed=seed)
    snapshots = {0.0: x.copy()}
    requested = [0.25, 0.50, 0.75, 1.0]
    save_index = {int(round(st * n_steps)): st for st in requested}
    for i in range(n_steps):
        t = ts[i]
        drift = corrected_weight_bridge_drift_1d(x, t, old_wrong_sign=old_wrong_sign)
        x = x + drift * dt + math.sqrt(dt) * rng.normal(size=n_paths)
        j = i + 1
        if j in save_index:
            snapshots[save_index[j]] = x.copy()
    return ts, snapshots


def validate_weight_bridge_sde(n_paths: int = 4000, n_steps: int = 500, seed: int = 0):
    """Return W1 marginal errors for corrected and old-wrong-sign drifts."""
    _, snaps_corr = simulate_weight_bridge_sde(n_paths, n_steps, seed=seed, old_wrong_sign=False)
    _, snaps_old = simulate_weight_bridge_sde(n_paths, n_steps, seed=seed, old_wrong_sign=True)
    out = []
    for t in [0.0, 0.25, 0.50, 0.75, 1.0]:
        target = sample_weight_bridge_target(n_paths, t, seed=seed + int(1000 * t) + 17)
        w_corr = wasserstein_distance(snaps_corr[t], target)
        w_old = wasserstein_distance(snaps_old[t], target)
        out.append((t, w_corr, w_old))
    return out, snaps_corr, snaps_old
