# %% [markdown]
# # Exp05: Scaling Experiments
#
# Three scaling tests for the CAS continual memory:
# - **A**: Crowding sweep — vary inter-component separation at fixed K
# - **B**: Dimension scaling — 2D signal embedded in d=2,4,8,16
# - **C**: Split-and-merge curriculum — time-varying mixture geometry
#
# Produces paper figures for Section 8.

# %%
# ═══════════════════════════════════════════════════════════════════════
#  Imports + style
# ═══════════════════════════════════════════════════════════════════════
import os, sys, math
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if '__file__' in dir() else os.getcwd())
from bridge_cas import (
    GaussianMixture, ContinualMemory, ForgetMetrics,
    make_daily_gmm_circle, run_cl_loop, compute_age_curves,
)

plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          11,
    "axes.labelsize":     12,
    "axes.titlesize":     13,
    "legend.fontsize":    10,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "lines.linewidth":    1.4,
    "lines.markersize":   4,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
})

FIGS = os.path.join(os.getcwd(), "figs")
os.makedirs(FIGS, exist_ok=True)

C_BLUE   = "#2166ac"
C_ORANGE = "#e08214"
C_GREEN  = "#1a9641"
C_RED    = "#d73027"
C_PURPLE = "#7570b3"
C_GRAY   = "#636363"
SWEEP_COLORS = [C_BLUE, C_ORANGE, C_GREEN, C_RED, C_PURPLE, C_GRAY,
                "#e7298a", "#66a61e"]

N_DAYS = 100
D      = 2
L_DEF  = 10
PERIOD = 50
R_DRIFT = 2.0

def make_prior(K, d):
    return GaussianMixture(
        weights=torch.ones(K) / K,
        means=torch.zeros(K, d),
        covs=torch.eye(d).unsqueeze(0).expand(K, -1, -1).clone(),
    )

print("Imports OK.  torch", torch.__version__)


# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════
# %%
# ═══════════════════════════════════════════════════════════════════════
#  A. Crowding sweep
# ═══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  A: Crowding sweep")
print("=" * 70)

COV_SC = 0.3    # fixed per-component covariance scale
sigma  = math.sqrt(COV_SC)   # component std dev ≈ 0.548

# Crowding ratio: χ = r / σ  where r is inter-component offset from
# the mixture centre and σ = sqrt(cov_scale).
# Small χ → components overlap heavily; large χ → well separated.

r_VALS = [0.15, 0.3, 0.5, 0.8, 1.2, 2.0]
chi_VALS = [rv / sigma for rv in r_VALS]

# ── A1: sweep crowding at fixed K=3 ─────────────────────────────────
print("\n  A1: crowding sweep at K=3")
K_CROWD = 3
res_crowd_k3 = []
for i, rv in enumerate(r_VALS):
    chi = rv / sigma
    print(f"    r={rv:.2f}, χ={chi:.2f} ...", end=" ", flush=True)
    dd, _ = make_daily_gmm_circle(N_DAYS, K=K_CROWD, R=R_DRIFT, r=rv,
                                   period=PERIOD, cov_scale=COV_SC, d=D)
    prior = make_prior(K_CROWD, D)
    mem, _, Fn, _ = run_cl_loop(dd, prior, L=L_DEF, verbose_every=0)
    a, fm, fs, _, hl = compute_age_curves(Fn, N_DAYS)

    # compute mean-share of forgetting at final day
    mean_shares = []
    for m in range(1, N_DAYS + 1):
        if m in mem.readout_times:
            try:
                dec = mem.forgetting_decomposed(m)
                ft = dec['F_total'].item()
                if ft > 1e-15:
                    mean_shares.append(dec['F_mean'].item() / ft)
            except Exception:
                pass
    avg_mean_share = np.mean(mean_shares) if mean_shares else float('nan')

    res_crowd_k3.append((rv, chi, a, fm, fs, hl, avg_mean_share))
    print(f"a₁/₂={hl}, mean_share={avg_mean_share:.2%}")

# ── A2: sweep crowding at multiple K ────────────────────────────────
print("\n  A2: half-life vs crowding for K=2,3,5,8")
K_CROWD_VALS = [2, 3, 5, 8]
res_crowd_K = {}   # K → list of (chi, hl)
for Kv in K_CROWD_VALS:
    res_crowd_K[Kv] = []
    for rv in r_VALS:
        chi = rv / sigma
        print(f"    K={Kv}, r={rv:.2f}, χ={chi:.2f} ...", end=" ", flush=True)
        dd, _ = make_daily_gmm_circle(N_DAYS, K=Kv, R=R_DRIFT, r=rv,
                                       period=PERIOD, cov_scale=COV_SC, d=D)
        prior = make_prior(Kv, D)
        _, _, Fn, _ = run_cl_loop(dd, prior, L=L_DEF, verbose_every=0)
        _, fm, _, _, hl = compute_age_curves(Fn, N_DAYS)
        res_crowd_K[Kv].append((chi, hl))
        print(f"a₁/₂={hl}")

# %%
# ── A: Crowding figures ──────────────────────────────────────────────

# Figure C1: (a) half-life vs χ for multiple K, (b) age curves at K=3,
# (c) mean share vs χ
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# (a) Half-life vs crowding for multiple K
ax = axes[0]
for i_k, Kv in enumerate(K_CROWD_VALS):
    chis = [c for c, _ in res_crowd_K[Kv]]
    hls  = [h if h is not None else N_DAYS for _, h in res_crowd_K[Kv]]
    ax.plot(chis, hls, "o-", ms=4, color=SWEEP_COLORS[i_k],
            label=f"$K={Kv}$")
ax.set_xlabel(r"Crowding ratio $\chi = r/\sigma$")
ax.set_ylabel(r"$a_{1/2}$")
ax.set_title("(a) Half-life vs crowding")
ax.legend(fontsize=8)

# (b) Age curves at K=3 for different crowding
ax = axes[1]
for i, (rv, chi, a, fm, fs, hl, ms_) in enumerate(res_crowd_k3):
    mask = ~np.isnan(fm) & (a > 0)
    lbl = f"$\\chi={chi:.1f}$"
    if hl is not None:
        lbl += f" ($a_{{1/2}}={hl}$)"
    ax.plot(a[mask], fm[mask], "o-", ms=2, color=SWEEP_COLORS[i], label=lbl)
ax.axhline(0.5, ls="--", color="gray", lw=0.5)
ax.axhline(1.0, ls=":", color="gray", lw=0.3, alpha=0.5)
ax.set_xlabel("Age $a$"); ax.set_ylabel(r"$\bar{F}(a)$")
ax.set_title("(b) Age curves ($K=3$)")
ax.legend(fontsize=5.5, loc="upper left")
ax.set_ylim(-0.03, 1.5); ax.set_xlim(0, N_DAYS)

# (c) Mean-share vs χ at K=3
ax = axes[2]
chis_k3    = [chi for _, chi, _, _, _, _, _ in res_crowd_k3]
mshares_k3 = [ms_ for _, _, _, _, _, _, ms_ in res_crowd_k3]
ax.plot(chis_k3, mshares_k3, "s-", ms=5, color=C_RED)
ax.set_xlabel(r"Crowding ratio $\chi$")
ax.set_ylabel("Mean-error share of total $F$")
ax.set_title("(c) Mean share ($K=3$)")
ax.set_ylim(0, 1.05)
ax.axhline(1.0, ls=":", color="gray", lw=0.3)

fig.tight_layout()
fig.savefig(os.path.join(FIGS, "c1_crowding_phase_diagram.pdf"))
fig.savefig(os.path.join(FIGS, "c1_crowding_phase_diagram.png"))
plt.close(fig)
print("  Saved c1_crowding_phase_diagram")

# Representative age curves figure
fig, ax = plt.subplots(figsize=(5.5, 3.8))
# pick weak, medium, strong
pick_idx = [0, 2, 4]  # r=0.15, 0.5, 1.2
labels_regime = ["strong crowding", "medium", "weak"]
for j, idx in enumerate(pick_idx):
    rv, chi, a, fm, fs, hl, _ = res_crowd_k3[idx]
    mask = ~np.isnan(fm) & (a > 0)
    ax.plot(a[mask], fm[mask], "o-", ms=3, color=SWEEP_COLORS[j],
            label=f"{labels_regime[j]} ($\\chi={chi:.1f}$, $a_{{1/2}}={hl}$)")
ax.axhline(0.5, ls="--", color="gray", lw=0.5)
ax.axhline(1.0, ls=":", color="gray", lw=0.3, alpha=0.5)
ax.set_xlabel("Age $a$"); ax.set_ylabel(r"$\bar{F}(a)$")
ax.set_title("Crowding regimes ($K=3$, $L=10$)")
ax.legend(fontsize=8, loc="upper left")
ax.set_ylim(-0.03, 1.5); ax.set_xlim(0, N_DAYS)
fig.savefig(os.path.join(FIGS, "c2_crowding_age_curves.pdf"))
fig.savefig(os.path.join(FIGS, "c2_crowding_age_curves.png"))
plt.close(fig)
print("  Saved c2_crowding_age_curves")


# ═══════════════════════════════════════════════════════════════════════
# %%
# ═══════════════════════════════════════════════════════════════════════
#  B. Dimension scaling — 2D signal in d-dimensional ambient space
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  B: Dimension scaling")
print("=" * 70)

K_DIM   = 3
R_COMP  = 0.8
COV_SC  = 0.3
D_VALS  = [2, 4, 8, 16]

# ── B1: no nuisance drift (extra dims are static at 0) ──────────────
print("\n  B1: no nuisance drift")
res_dim_no_nuisance = []
for dv in D_VALS:
    print(f"    d={dv} ...", end=" ", flush=True)
    dd, _ = make_daily_gmm_circle(N_DAYS, K=K_DIM, R=R_DRIFT, r=R_COMP,
                                   period=PERIOD, cov_scale=COV_SC, d=dv)
    prior = make_prior(K_DIM, dv)
    mem, _, Fn, _ = run_cl_loop(dd, prior, L=L_DEF, verbose_every=0)
    a, fm, fs, _, hl = compute_age_curves(Fn, N_DAYS)

    # mean-error share
    mean_shares = []
    for m in range(1, N_DAYS + 1):
        if m in mem.readout_times:
            try:
                dec = mem.forgetting_decomposed(m)
                ft = dec['F_total'].item()
                if ft > 1e-15:
                    mean_shares.append(dec['F_mean'].item() / ft)
            except Exception:
                pass
    avg_ms = np.mean(mean_shares) if mean_shares else float('nan')

    res_dim_no_nuisance.append((dv, a, fm, fs, hl, avg_ms))
    print(f"a₁/₂={hl}, mean_share={avg_ms:.2%}")

# ── B2: with nuisance drift (extra dims have slow random walk) ───────
print("\n  B2: with nuisance drift")


def make_daily_gmm_circle_nuisance(
    n_days, K=3, R=2.0, r=0.8, period=50, cov_scale=0.3, d=4,
    nuisance_speed=0.1, seed=42,
):
    """Like make_daily_gmm_circle but extra dims (2..d-1) have a slow
    random-walk drift."""
    rng = np.random.RandomState(seed)
    dists = []
    all_means = []
    nuisance_pos = np.zeros(max(d - 2, 0))

    for m in range(1, n_days + 1):
        angle = 2 * math.pi * m / period
        centre = np.zeros(d)
        centre[0] = R * math.cos(angle)
        centre[1] = R * math.sin(angle)

        # nuisance walk
        if d > 2:
            nuisance_pos = nuisance_pos + nuisance_speed * rng.randn(d - 2)
            centre[2:] = nuisance_pos

        comp_means = np.zeros((K, d))
        for k in range(K):
            phi_k = angle + 2 * math.pi * k / K
            comp_means[k, 0] = centre[0] + r * math.cos(phi_k)
            comp_means[k, 1] = centre[1] + r * math.sin(phi_k)
            if d > 2:
                comp_means[k, 2:] = centre[2:]

        comp_covs = cov_scale * np.eye(d)
        comp_covs = np.tile(comp_covs, (K, 1, 1))
        weights = np.ones(K) / K

        gm = GaussianMixture(
            weights=torch.tensor(weights),
            means=torch.tensor(comp_means),
            covs=torch.tensor(comp_covs),
        )
        dists.append(gm)
        all_means.append(comp_means.copy())
    return dists, np.array(all_means)


res_dim_nuisance = []
for dv in D_VALS:
    if dv == 2:
        # same as no-nuisance for d=2
        res_dim_nuisance.append(res_dim_no_nuisance[0])
        print(f"    d={dv} (= no-nuisance)")
        continue
    print(f"    d={dv} ...", end=" ", flush=True)
    dd, _ = make_daily_gmm_circle_nuisance(
        N_DAYS, K=K_DIM, R=R_DRIFT, r=R_COMP, period=PERIOD,
        cov_scale=COV_SC, d=dv, nuisance_speed=0.1)
    prior = make_prior(K_DIM, dv)
    mem, _, Fn, _ = run_cl_loop(dd, prior, L=L_DEF, verbose_every=0)
    a, fm, fs, _, hl = compute_age_curves(Fn, N_DAYS)

    mean_shares = []
    for m in range(1, N_DAYS + 1):
        if m in mem.readout_times:
            try:
                dec = mem.forgetting_decomposed(m)
                ft = dec['F_total'].item()
                if ft > 1e-15:
                    mean_shares.append(dec['F_mean'].item() / ft)
            except Exception:
                pass
    avg_ms = np.mean(mean_shares) if mean_shares else float('nan')

    res_dim_nuisance.append((dv, a, fm, fs, hl, avg_ms))
    print(f"a₁/₂={hl}, mean_share={avg_ms:.2%}")

# %%
# ── B: Dimension figures ─────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# (a) Age curves — no nuisance
ax = axes[0]
for i, (dv, a, fm, fs, hl, _) in enumerate(res_dim_no_nuisance):
    mask = ~np.isnan(fm) & (a > 0)
    lbl = f"$d={dv}$"
    if hl is not None:
        lbl += f" ($a_{{1/2}}={hl}$)"
    ax.plot(a[mask], fm[mask], "o-", ms=2, color=SWEEP_COLORS[i], label=lbl)
ax.axhline(0.5, ls="--", color="gray", lw=0.5)
ax.axhline(1.0, ls=":", color="gray", lw=0.3, alpha=0.5)
ax.set_xlabel("Age $a$"); ax.set_ylabel(r"$\bar{F}(a)$")
ax.set_title("(a) No nuisance drift")
ax.legend(fontsize=8, loc="upper left")
ax.set_ylim(-0.03, 1.5); ax.set_xlim(0, N_DAYS)

# (b) Half-life vs d for both settings
ax = axes[1]
d_nn = [dv for dv, _, _, _, _, _ in res_dim_no_nuisance]
hl_nn = [hl if hl is not None else N_DAYS
         for _, _, _, _, hl, _ in res_dim_no_nuisance]
d_nu = [dv for dv, _, _, _, _, _ in res_dim_nuisance]
hl_nu = [hl if hl is not None else N_DAYS
         for _, _, _, _, hl, _ in res_dim_nuisance]
ax.plot(d_nn, hl_nn, "o-", ms=5, color=C_BLUE, label="no nuisance")
ax.plot(d_nu, hl_nu, "s--", ms=5, color=C_RED, label="nuisance drift")
ax.set_xlabel("Ambient dimension $d$")
ax.set_ylabel(r"$a_{1/2}$")
ax.set_title("(b) Half-life vs $d$")
ax.legend(fontsize=9)
ax.set_xticks(D_VALS)

# (c) Mean-error share vs d
ax = axes[2]
ms_nn = [ms_ for _, _, _, _, _, ms_ in res_dim_no_nuisance]
ms_nu = [ms_ for _, _, _, _, _, ms_ in res_dim_nuisance]
ax.plot(d_nn, ms_nn, "o-", ms=5, color=C_BLUE, label="no nuisance")
ax.plot(d_nu, ms_nu, "s--", ms=5, color=C_RED, label="nuisance drift")
ax.set_xlabel("Ambient dimension $d$")
ax.set_ylabel("Mean-error share")
ax.set_title("(c) Mean share vs $d$")
ax.legend(fontsize=9)
ax.set_xticks(D_VALS)
ax.set_ylim(0, 1.05)

fig.tight_layout()
fig.savefig(os.path.join(FIGS, "d1_signal_vs_nuisance_scaling.pdf"))
fig.savefig(os.path.join(FIGS, "d1_signal_vs_nuisance_scaling.png"))
plt.close(fig)
print("  Saved d1_signal_vs_nuisance_scaling")


# ═══════════════════════════════════════════════════════════════════════
# %%
# ═══════════════════════════════════════════════════════════════════════
#  C. Split-and-merge curriculum
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  C: Split-and-merge curriculum")
print("=" * 70)

# K=3 throughout, but the geometry changes:
#   Days   1- 30: normal rotating triangle (r=0.8)
#   Days  31- 50: two components merge (r_12 → 0.05, r_3 stays)
#   Days  51- 80: split back to rotating triangle (r=0.8)
#   Days  81-100: all three collapse toward centre (r → 0.1)
#
# This tests whether topological changes in the mixture disrupt retention.

K_SM    = 3
COV_SM  = 0.3
R_SM    = 2.0
P_SM    = 50


def make_split_merge_curriculum(n_days, K=3, R=2.0, period=50,
                                cov_scale=0.3, d=2):
    """Generate a split-merge curriculum for K=3."""
    dists = []
    all_means = []
    for m in range(1, n_days + 1):
        angle = 2 * math.pi * m / period
        centre = np.zeros(d)
        centre[0] = R * math.cos(angle)
        centre[1] = R * math.sin(angle)

        # Determine inter-component radius based on curriculum phase
        if m <= 30:
            # Phase 1: normal triangle
            r_eff = 0.8
            offsets = _triangle_offsets(angle, K, r_eff)
        elif m <= 50:
            # Phase 2: components 0,1 merge; component 2 stays out
            # Smooth transition over 5 days
            t_merge = min((m - 30) / 5.0, 1.0)
            r_01 = 0.8 * (1 - t_merge) + 0.05 * t_merge
            r_2  = 0.8
            offsets = np.zeros((K, d))
            for k in range(K):
                phi_k = angle + 2 * math.pi * k / K
                rk = r_01 if k < 2 else r_2
                offsets[k, 0] = rk * math.cos(phi_k)
                offsets[k, 1] = rk * math.sin(phi_k)
        elif m <= 80:
            # Phase 3: split back
            t_split = min((m - 50) / 5.0, 1.0)
            r_eff = 0.05 * (1 - t_split) + 0.8 * t_split
            offsets = _triangle_offsets(angle, K, r_eff)
        else:
            # Phase 4: all collapse
            t_collapse = min((m - 80) / 5.0, 1.0)
            r_eff = 0.8 * (1 - t_collapse) + 0.1 * t_collapse
            offsets = _triangle_offsets(angle, K, r_eff)

        comp_means = np.zeros((K, d))
        for k in range(K):
            comp_means[k] = centre + offsets[k]

        comp_covs = cov_scale * np.eye(d)
        comp_covs = np.tile(comp_covs, (K, 1, 1))
        weights = np.ones(K) / K

        gm = GaussianMixture(
            weights=torch.tensor(weights),
            means=torch.tensor(comp_means),
            covs=torch.tensor(comp_covs),
        )
        dists.append(gm)
        all_means.append(comp_means.copy())

    return dists, np.array(all_means)


def _triangle_offsets(angle, K, r):
    d = 2
    offsets = np.zeros((K, d))
    for k in range(K):
        phi_k = angle + 2 * math.pi * k / K
        offsets[k, 0] = r * math.cos(phi_k)
        offsets[k, 1] = r * math.sin(phi_k)
    return offsets


daily_sm, comp_means_sm = make_split_merge_curriculum(
    N_DAYS, K=K_SM, R=R_SM, period=P_SM, cov_scale=COV_SM, d=D)

print(f"  Generated {N_DAYS}-day split-merge curriculum, K={K_SM}")

# Run CAS
prior_sm = make_prior(K_SM, D)
mem_sm, Fraw_sm, Fnorm_sm, snaps_sm = run_cl_loop(
    daily_sm, prior_sm, L=L_DEF, verbose_every=25)
ages_sm, Fmu_sm, Fsig_sm, cnt_sm, hl_sm = compute_age_curves(Fnorm_sm, N_DAYS)
print(f"  Half-life: a₁/₂ = {hl_sm}")

# %%
# ── C: Split-merge figures ───────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

# (a) Component means at selected times, showing the curriculum
ax = axes[0]
cmap = plt.cm.viridis
show_every = 2
for m_idx in range(0, N_DAYS, show_every):
    c = cmap(m_idx / N_DAYS)
    for k in range(K_SM):
        ax.plot(comp_means_sm[m_idx, k, 0], comp_means_sm[m_idx, k, 1],
                "o", ms=2.5, color=c, alpha=0.5)

# Mark phase boundaries
for phase_day, label in [(1, "normal"), (31, "merge"), (51, "split"),
                          (81, "collapse")]:
    idx = phase_day - 1
    if idx < N_DAYS:
        for k in range(K_SM):
            ax.plot(comp_means_sm[idx, k, 0], comp_means_sm[idx, k, 1],
                    "D", ms=5, color="red", zorder=10)

ax.plot(0, 0, "k*", ms=8, zorder=10)
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
ax.set_title("(a) Component means (curriculum)")
ax.set_aspect("equal")
ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5)

# Add colourbar
sm = plt.cm.ScalarMappable(cmap="viridis",
                            norm=plt.Normalize(1, N_DAYS))
plt.colorbar(sm, ax=ax, shrink=0.8, label="day", pad=0.02)

# (b) Age-forgetting curve
ax = axes[1]
mask = ~np.isnan(Fmu_sm) & (ages_sm > 0)
ax.plot(ages_sm[mask], Fmu_sm[mask], "o-", ms=3, color=C_BLUE)
ax.fill_between(ages_sm[mask],
    np.clip(Fmu_sm[mask] - Fsig_sm[mask], 0, None),
    Fmu_sm[mask] + Fsig_sm[mask], alpha=0.15, color=C_BLUE)
ax.axhline(0.5, ls="--", color="gray", lw=0.7)
ax.axhline(1.0, ls=":", color="gray", lw=0.5, alpha=0.5)
if hl_sm is not None:
    ax.axvline(hl_sm, ls=":", color=C_RED, lw=1,
               label=f"$a_{{1/2}}={hl_sm}$")
ax.set_xlabel("Age $a$"); ax.set_ylabel(r"$\bar{F}(a)$")
ax.set_title("(b) Age–forgetting curve")
ax.legend(fontsize=10, loc="upper left")
ax.set_ylim(-0.03, 1.5); ax.set_xlim(0, N_DAYS)

fig.tight_layout()
fig.savefig(os.path.join(FIGS, "e1_split_merge_curriculum.pdf"))
fig.savefig(os.path.join(FIGS, "e1_split_merge_curriculum.png"))
plt.close(fig)
print("  Saved e1_split_merge_curriculum")

# %%
# ═══════════════════════════════════════════════════════════════════════
#  Summary
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  Exp05 Summary")
print("=" * 70)

print("\n  A: Crowding sweep (K=3, L=10)")
for rv, chi, _, _, _, hl, ms_ in res_crowd_k3:
    print(f"    r={rv:.2f}, χ={chi:.2f}  →  a₁/₂={hl}, "
          f"mean_share={ms_:.1%}")

print("\n  A: Crowding × K matrix (half-lives):")
print(f"    {'χ':>8s}", end="")
for rv in r_VALS:
    print(f"  {rv/sigma:5.2f}", end="")
print()
for Kv in K_CROWD_VALS:
    print(f"    K={Kv:>2d}   ", end="")
    for chi, hl in res_crowd_K[Kv]:
        hl_s = f"{hl:5d}" if hl is not None else "  >99"
        print(f"  {hl_s}", end="")
    print()

print("\n  B: Dimension scaling (K=3, L=10)")
print("    No nuisance:")
for dv, _, _, _, hl, ms_ in res_dim_no_nuisance:
    print(f"      d={dv:2d}  →  a₁/₂={hl}, mean_share={ms_:.1%}")
print("    With nuisance drift:")
for dv, _, _, _, hl, ms_ in res_dim_nuisance:
    print(f"      d={dv:2d}  →  a₁/₂={hl}, mean_share={ms_:.1%}")

print(f"\n  C: Split-merge curriculum (K={K_SM}, L={L_DEF})")
print(f"    a₁/₂ = {hl_sm}")

print(f"\n  Figures in: {FIGS}/")
print(f"    c1_crowding_phase_diagram  — crowding sweep")
print(f"    c2_crowding_age_curves     — representative regimes")
print(f"    d1_signal_vs_nuisance_scaling — dimension scaling")
print(f"    e1_split_merge_curriculum  — curriculum experiment")
print("=" * 70)

# %%
