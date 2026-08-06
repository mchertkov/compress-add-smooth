"""
test_bridge_cas.py — unit tests for the CAS continual memory module
"""

import sys
import torch
import numpy as np
torch.set_default_dtype(torch.float64)

from bridge_cas import (
    GaussianMixture, ProtocolGrid, ContinualMemory, ForgetMetrics,
    rebinning_matrix, make_daily_gaussians_circle, make_daily_gmm_circle,
    run_cl_loop, compute_age_curves,
)

PASS = 0
FAIL = 0

def check(name, condition):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}")

# ═══════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  1. GaussianMixture basics")
print("=" * 60)

gm1 = GaussianMixture(
    weights=torch.tensor([1.0]),
    means=torch.tensor([[1.0, 2.0]]),
    covs=torch.tensor([[[0.5, 0.0], [0.0, 0.5]]]),
)
check("K=1 shape", gm1.K == 1 and gm1.d == 2)

mu = gm1.overall_mean()
check("K=1 mean", torch.allclose(mu, torch.tensor([1.0, 2.0])))

cov = gm1.overall_cov()
check("K=1 cov", torch.allclose(cov, 0.5 * torch.eye(2)))

# K=2 mixture: equal weights, symmetric means
gm2 = GaussianMixture(
    weights=torch.tensor([0.5, 0.5]),
    means=torch.tensor([[1.0, 0.0], [-1.0, 0.0]]),
    covs=torch.stack([0.1 * torch.eye(2), 0.1 * torch.eye(2)]),
)
mu2 = gm2.overall_mean()
check("K=2 symmetric mean ≈ 0", torch.allclose(mu2, torch.zeros(2), atol=1e-14))

cov2 = gm2.overall_cov()
# within: 0.1 I;  between: 0.5*(1,0)(1,0)^T + 0.5*(-1,0)(-1,0)^T = diag(1,0)
expected_cov = torch.tensor([[1.1, 0.0], [0.0, 0.1]])
check("K=2 overall cov", torch.allclose(cov2, expected_cov, atol=1e-14))

# Density evaluation
x = torch.tensor([1.0, 0.0])
logp = gm1.log_density(x)
check("log_density runs", logp.dim() == 0)
check("density > 0", gm1.density(x).item() > 0)

# Batch density
xb = torch.randn(50, 2)
logpb = gm1.log_density(xb)
check("batch log_density shape", logpb.shape == (50,))

# Sampling
samp = gm1.sample(1000)
check("sample shape", samp.shape == (1000, 2))
check("sample mean ≈ (1,2)", torch.allclose(samp.mean(0), torch.tensor([1.0, 2.0]), atol=0.15))

# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  2. Interpolation")
print("=" * 60)

gm_a = GaussianMixture(
    weights=torch.tensor([1.0]),
    means=torch.tensor([[0.0, 0.0]]),
    covs=torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
)
gm_b = GaussianMixture(
    weights=torch.tensor([1.0]),
    means=torch.tensor([[2.0, 4.0]]),
    covs=torch.tensor([[[3.0, 0.0], [0.0, 3.0]]]),
)

gm_half = GaussianMixture.interpolate(gm_a, gm_b, 0.5)
check("interp mean", torch.allclose(gm_half.means, torch.tensor([[1.0, 2.0]])))
check("interp cov", torch.allclose(gm_half.covs, torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])))

gm_zero = GaussianMixture.interpolate(gm_a, gm_b, 0.0)
check("interp α=0", torch.allclose(gm_zero.means, gm_a.means))

gm_one = GaussianMixture.interpolate(gm_a, gm_b, 1.0)
check("interp α=1", torch.allclose(gm_one.means, gm_b.means))

# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  3. ProtocolGrid")
print("=" * 60)

L = 4
pg = ProtocolGrid.from_interpolant(gm_a, gm_b, L)
check("grid L", pg.L == L)
check("grid n_nodes", len(pg.nodes) == L + 1)

# Check endpoints
check("node[0] = gm_a", torch.allclose(pg.nodes[0].means, gm_a.means))
check("node[L] = gm_b", torch.allclose(pg.nodes[L].means, gm_b.means))

# Evaluate at midpoint
gm_mid = pg.evaluate_at(0.5)
check("eval(0.5) mean", torch.allclose(gm_mid.means, torch.tensor([[1.0, 2.0]]), atol=1e-12))

# Evaluate at node times should recover nodes
for j in range(L + 1):
    t_j = j / L
    gm_j = pg.evaluate_at(t_j)
    check(f"eval(t_{j}) = node[{j}]",
          torch.allclose(gm_j.means, pg.nodes[j].means, atol=1e-12))

# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  4. CAS operations")
print("=" * 60)

# Compress: states unchanged
pg_cmp = pg.compress()
check("compress preserves L", pg_cmp.L == pg.L)
for j in range(L + 1):
    check(f"compress node[{j}] unchanged",
          torch.allclose(pg_cmp.nodes[j].means, pg.nodes[j].means, atol=1e-14))

# Add: L+2 nodes
gm_new = GaussianMixture(
    weights=torch.tensor([1.0]),
    means=torch.tensor([[5.0, 5.0]]),
    covs=torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
)
pg_aug = pg_cmp.add(gm_new)
check("add: L+2 nodes", len(pg_aug.nodes) == L + 2)
check("add: last node = gm_new",
      torch.allclose(pg_aug.nodes[-1].means, gm_new.means, atol=1e-14))

# Smooth: back to L+1 nodes
pg_smooth = pg_aug.smooth(L)
check("smooth: L+1 nodes", len(pg_smooth.nodes) == L + 1)
check("smooth: L segments", pg_smooth.L == L)

# Endpoints: t=0 should be close to original (minor rebinning shift)
# t=1 should be close to gm_new
check("smooth: node[0] ≈ gm_a (endpoint preserved)",
      torch.allclose(pg_smooth.nodes[0].means, gm_a.means, atol=1e-12))
check("smooth: node[L] ≈ gm_new",
      torch.allclose(pg_smooth.nodes[L].means, gm_new.means, atol=1e-12))

# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  5. Rebinning matrix")
print("=" * 60)

W = rebinning_matrix(L)
check("W shape", W.shape == (L + 1, L + 2))
check("W rows sum to 1", torch.allclose(W.sum(dim=1), torch.ones(L + 1), atol=1e-14))
check("W non-negative", (W >= -1e-15).all().item())
check("W sparse (≤ 2 nonzeros per row)",
      all((W[j] > 1e-15).sum().item() <= 2 for j in range(L + 1)))

# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  6. ContinualMemory: readout-time decay")
print("=" * 60)

L_test = 10
gm_prior = GaussianMixture(
    weights=torch.tensor([1.0]),
    means=torch.zeros(1, 2),
    covs=torch.eye(2).unsqueeze(0),
)
daily, _ = make_daily_gaussians_circle(20, R=2.0, period=50, d=2)

mem = ContinualMemory(gm_prior, L_test)
for d_idx in range(20):
    mem.incorporate(daily[d_idx])

# Check readout time of day 1 after 20 days: (L/(L+1))^19
ratio = L_test / (L_test + 1)
expected_t1 = ratio ** 19
actual_t1 = mem.readout_times[1]
check(f"readout t_{{1|20}} = {actual_t1:.6f} ≈ {expected_t1:.6f}",
      abs(actual_t1 - expected_t1) < 1e-10)

# Day 20 readout should be 1.0
check("readout t_{20|20} = 1.0", abs(mem.readout_times[20] - 1.0) < 1e-14)

# Forgetting at age 0 should be ≈ 0
f_age0 = mem.forgetting_normalised(20).item()
check(f"forgetting at age 0 = {f_age0:.6e} ≈ 0", f_age0 < 1e-6)

# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  7. ForgetMetrics")
print("=" * 60)

# Identical GMs: F = 0
f_self = ForgetMetrics.raw_mismatch(gm1, gm1)
check("self-mismatch = 0", f_self.item() < 1e-14)

# Decomposed metric
gm_orig = GaussianMixture(
    weights=torch.tensor([0.5, 0.5]),
    means=torch.tensor([[0.0, 0.0], [2.0, 0.0]]),
    covs=torch.stack([0.3 * torch.eye(2), 0.3 * torch.eye(2)]),
)
gm_shifted = GaussianMixture(
    weights=torch.tensor([0.5, 0.5]),
    means=torch.tensor([[0.1, 0.0], [2.1, 0.0]]),   # small shift
    covs=torch.stack([0.3 * torch.eye(2), 0.3 * torch.eye(2)]),
)
dec = ForgetMetrics.decomposed(gm_orig, gm_shifted)
check("decomposed F_cov ≈ 0 (covs unchanged)", dec['F_cov'].item() < 1e-14)
check("decomposed F_weight ≈ 0 (weights unchanged)", dec['F_weight'].item() < 1e-14)
check("decomposed F_mean > 0 (means shifted)", dec['F_mean'].item() > 1e-6)

# Permutation check: swap components in replay
gm_swapped = GaussianMixture(
    weights=torch.tensor([0.5, 0.5]),
    means=torch.tensor([[2.1, 0.0], [0.1, 0.0]]),   # swapped order
    covs=torch.stack([0.3 * torch.eye(2), 0.3 * torch.eye(2)]),
)
dec2 = ForgetMetrics.decomposed(gm_orig, gm_swapped)
check("Hungarian handles swap",
      abs(dec2['F_mean'].item() - dec['F_mean'].item()) < 1e-12)

# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  8. Autograd compatibility")
print("=" * 60)

# Check that forgetting metric is differentiable w.r.t. means
means_param = torch.tensor([[1.0, 2.0]], requires_grad=True)
gm_diff = GaussianMixture(
    weights=torch.tensor([1.0]),
    means=means_param,
    covs=torch.tensor([[[0.5, 0.0], [0.0, 0.5]]]),
)
gm_target = GaussianMixture(
    weights=torch.tensor([1.0]),
    means=torch.tensor([[3.0, 4.0]]),
    covs=torch.tensor([[[0.5, 0.0], [0.0, 0.5]]]),
)
loss = ForgetMetrics.raw_mismatch(gm_diff, gm_target)
loss.backward()
check("autograd: grad exists", means_param.grad is not None)
check("autograd: grad nonzero", means_param.grad.abs().sum().item() > 1e-6)

# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  9. Full CL loop (K=1, n=30)")
print("=" * 60)

daily30, means30 = make_daily_gaussians_circle(30, R=2.0, period=50, d=2)
gm_prior30 = GaussianMixture(
    weights=torch.tensor([1.0]),
    means=torch.zeros(1, 2),
    covs=torch.eye(2).unsqueeze(0),
)
mem30, Fraw30, Fnorm30, snaps30 = run_cl_loop(
    daily30, gm_prior30, L=10, verbose_every=0)

ages, F_mu, F_std, counts, hl = compute_age_curves(Fnorm30, 30)
check("age curve computed", len(ages) == 30)
check("F̄(age=0) ≈ 0", F_mu[0] < 1e-4 if not np.isnan(F_mu[0]) else False)
check("F̄ increases with age (rough)", 
      np.nanmean(F_mu[1:5]) < np.nanmean(F_mu[20:25]) if not np.all(np.isnan(F_mu[20:25])) else True)
print(f"  half-life = {hl}")

# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  10. Full CL loop (K=3, n=30)")
print("=" * 60)

daily_gmm, comp_means = make_daily_gmm_circle(30, K=3, R=2.0, r=0.8, period=50, d=2)
gm_prior_k3 = GaussianMixture(
    weights=torch.ones(3) / 3,
    means=torch.zeros(3, 2),
    covs=torch.eye(2).unsqueeze(0).expand(3, -1, -1).clone(),
)
mem_k3, Fraw_k3, Fnorm_k3, snaps_k3 = run_cl_loop(
    daily_gmm, gm_prior_k3, L=10, verbose_every=0)

check("K=3 loop completed", mem_k3.current_day == 30)

# Decomposed metric at final day for day 1
dec_k3 = mem_k3.forgetting_decomposed(1)
check("K=3 decomposed has keys",
      all(k in dec_k3 for k in ['F_mean', 'F_cov', 'F_weight', 'F_total']))
print(f"  K=3 day-1 decomposed: F_mean={dec_k3['F_mean']:.4f}  "
      f"F_cov={dec_k3['F_cov']:.4f}  F_weight={dec_k3['F_weight']:.6f}")

# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"  RESULTS: {PASS} passed, {FAIL} failed")
print("=" * 60)
sys.exit(0 if FAIL == 0 else 1)
