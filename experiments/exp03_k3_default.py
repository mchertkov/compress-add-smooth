# %% [markdown]
# # Exp03: K=3 Default Run — Decomposed Forgetting
#
# Gaussian-mixture daily targets ($K{=}3$, $d{=}2$), circular drift.
# Produces paper figures 7, 9.
#
# **Sections**
# - A: Imports, style, helpers
# - B: GMM daily distribution generation
# - C: Default CL run (K=3)
# - D: Decomposed forgetting analysis
# - E: K=1 vs K=3 comparison
# - F: Paper figures

# %%
# ═══════════════════════════════════════════════════════════════════════
#  A. Imports + style + helpers
# ═══════════════════════════════════════════════════════════════════════
import os, sys
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
from scipy.optimize import linear_sum_assignment

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if '__file__' in dir() else os.getcwd())
from bridge_cas import (
    GaussianMixture, ContinualMemory, ForgetMetrics,
    make_daily_gaussians_circle, make_daily_gmm_circle,
    run_cl_loop, compute_age_curves,
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
COMP_COLORS = [C_BLUE, C_ORANGE, C_GREEN, C_RED, C_PURPLE]

print("Imports OK.  torch", torch.__version__)


# ── decomposed forgetting history helper ─────────────────────────────

def compute_decomposed_history(mem: ContinualMemory, n_days: int):
    """Compute decomposed metric for all past days at the final day.

    Returns dict mapping day m → decomposed metric dict.
    """
    results = {}
    for m in range(1, n_days + 1):
        if m not in mem.readout_times:
            continue
        try:
            dec = mem.forgetting_decomposed(m)
            dec['day'] = m
            dec['age'] = mem.current_day - m
            results[m] = dec
        except Exception:
            pass
    return results


# %%
# ═══════════════════════════════════════════════════════════════════════
#  B. GMM daily distribution generation
# ═══════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  B: Daily GMM generation")
print("=" * 60)

N_DAYS   = 100
D        = 2
K_GMM    = 3
R_DRIFT  = 2.0
R_COMP   = 0.8     # offset of component means from mixture centre
PERIOD   = 50
COV_SC   = 0.3     # per-component covariance scale

daily_gmm, daily_comp_means_t = make_daily_gmm_circle(
    N_DAYS, K=K_GMM, R=R_DRIFT, r=R_COMP, period=PERIOD,
    cov_scale=COV_SC, d=D)
daily_comp_means = daily_comp_means_t.numpy()   # (n_days, K, d)

print(f"  {N_DAYS} daily GMMs: K={K_GMM}, R={R_DRIFT}, r={R_COMP}, P={PERIOD}")
print(f"  Component cov: {COV_SC}·I, equal weights π=1/{K_GMM}")
print(f"  daily_comp_means shape: {daily_comp_means.shape}")

# Quick visualisation
fig, ax = plt.subplots(figsize=(6, 6))
for m in range(0, N_DAYS, 5):
    for k in range(K_GMM):
        c = plt.cm.viridis(m / N_DAYS)
        ax.plot(daily_comp_means[m, k, 0], daily_comp_means[m, k, 1],
                "o", ms=3, color=c, alpha=0.6)
ax.plot(0, 0, "k*", ms=10, zorder=5)
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
ax.set_title(f"Daily GMM component means (every 5th day, K={K_GMM})")
ax.set_aspect("equal")
fig.savefig(os.path.join(FIGS, "b_daily_gmm_means.png"))
plt.close(fig)
print("  Saved b_daily_gmm_means.png")

# ── priors ───────────────────────────────────────────────────────────
gm_prior_k3 = GaussianMixture(
    weights=torch.ones(K_GMM) / K_GMM,
    means=torch.zeros(K_GMM, D),
    covs=torch.eye(D).unsqueeze(0).expand(K_GMM, -1, -1).clone(),
)

gm_prior_k1 = GaussianMixture(
    weights=torch.ones(1),
    means=torch.zeros(1, D),
    covs=torch.eye(D).unsqueeze(0),
)

# %%
# ═══════════════════════════════════════════════════════════════════════
#  C. Default CL run (K=3)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  C: Default CL run (K=3)")
print("=" * 60)

L_DEF = 10

print(f"  Protocol: L={L_DEF}")
print()

mem3, Fraw3, Fnorm3, snaps3 = run_cl_loop(
    daily_gmm, gm_prior_k3, L=L_DEF, verbose_every=25)

ages3, Fmu3, Fsig3, cnt3, hl3 = compute_age_curves(Fnorm3, N_DAYS)
print(f"\n  K=3 half-life: a₁/₂ = {hl3}")

# %%
# ═══════════════════════════════════════════════════════════════════════
#  D. Decomposed forgetting analysis
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  D: Decomposed forgetting")
print("=" * 60)

decomp3 = compute_decomposed_history(mem3, N_DAYS)

# Collect decomposed metrics vs age
ages_dec, Fm_dec, Fc_dec, Fw_dec, Ft_dec = [], [], [], [], []
pc_mean_dec = []
for m in sorted(decomp3.keys()):
    d = decomp3[m]
    ages_dec.append(d['age'])
    Fm_dec.append(d['F_mean'].item())
    Fc_dec.append(d['F_cov'].item())
    Fw_dec.append(d['F_weight'].item())
    Ft_dec.append(d['F_total'].item())
    pc_mean_dec.append(d['per_comp_mean'].detach().numpy())

ages_dec    = np.array(ages_dec)
Fm_dec      = np.array(Fm_dec)
Fc_dec      = np.array(Fc_dec)
Fw_dec      = np.array(Fw_dec)
Ft_dec      = np.array(Ft_dec)
pc_mean_dec = np.array(pc_mean_dec)   # (n, K)

# Age-average the decomposed metrics
max_age = N_DAYS - 1
Fm_age = np.full(max_age + 1, np.nan)
Fc_age = np.full(max_age + 1, np.nan)
Fw_age = np.full(max_age + 1, np.nan)
for a in range(max_age + 1):
    mask = ages_dec == a
    if mask.sum() > 0:
        Fm_age[a] = np.mean(Fm_dec[mask])
        Fc_age[a] = np.mean(Fc_dec[mask])
        Fw_age[a] = np.mean(Fw_dec[mask])

print(f"  Computed decomposed metric for {len(decomp3)} past days")

for a_show in [5, 15, 25, 40]:
    mask = ages_dec == a_show
    if mask.sum() > 0:
        fm = np.mean(Fm_dec[mask])
        fc = np.mean(Fc_dec[mask])
        fw = np.mean(Fw_dec[mask])
        ft = fm + fc + fw
        if ft > 1e-15:
            print(f"  age={a_show:2d}: F_mean={fm:.4f} ({100*fm/ft:.0f}%)  "
                  f"F_cov={fc:.4f} ({100*fc/ft:.0f}%)  "
                  f"F_weight={fw:.6f} ({100*fw/ft:.1f}%)")

# ── Decomposed metric standalone figure ──────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# (a) Stacked area
ax = axes[0]
a_plot = np.arange(1, max_age + 1)
fm_p = np.nan_to_num(Fm_age[1:])
fc_p = np.nan_to_num(Fc_age[1:])
fw_p = np.nan_to_num(Fw_age[1:])
ax.fill_between(a_plot, 0, fm_p, alpha=0.7, color=C_BLUE,
                label="$F_{\\mathrm{mean}}$")
ax.fill_between(a_plot, fm_p, fm_p + fc_p, alpha=0.7, color=C_ORANGE,
                label="$F_{\\mathrm{cov}}$")
ax.fill_between(a_plot, fm_p + fc_p, fm_p + fc_p + fw_p, alpha=0.7,
                color=C_GREEN, label="$F_{\\mathrm{weight}}$")
ax.set_xlabel("Age $a$"); ax.set_ylabel("Decomposed $F$")
ax.set_title("(a) Forgetting decomposition")
ax.legend(fontsize=9, loc="upper left")
ax.set_xlim(1, N_DAYS - 1)

# (b) Per-component mean error
ax = axes[1]
for k in range(K_GMM):
    pc_age = np.full(max_age + 1, np.nan)
    for a in range(max_age + 1):
        mask = ages_dec == a
        if mask.sum() > 0:
            pc_age[a] = np.mean(pc_mean_dec[mask, k])
    valid = ~np.isnan(pc_age) & (np.arange(max_age + 1) > 0)
    ax.plot(np.arange(max_age + 1)[valid], pc_age[valid], "o-", ms=2,
            color=COMP_COLORS[k], label=f"comp {k+1}")
ax.set_xlabel("Age $a$"); ax.set_ylabel("$\\|\\Delta m_k\\|^2$")
ax.set_title("(b) Per-component mean error")
ax.legend(fontsize=9)
ax.set_xlim(1, N_DAYS - 1)

# (c) Weight drift
ax = axes[2]
dw_age = np.full((max_age + 1, K_GMM), np.nan)
for a in range(max_age + 1):
    mask_ids = [i for i, ag in enumerate(ages_dec) if ag == a]
    if mask_ids:
        dws = np.array([decomp3[sorted(decomp3.keys())[i]]['perm']
                         for i in mask_ids])
        # Compute weight deltas properly
        dw_vals = []
        for i in mask_ids:
            m_key = sorted(decomp3.keys())[i]
            d = decomp3[m_key]
            gm_orig = mem3.history[m_key - 1]
            gm_rep  = mem3.replay(m_key)
            perm_t  = torch.tensor(d['perm'], dtype=torch.long)
            w_o = gm_orig.weights.numpy()
            w_r = gm_rep.weights.detach().numpy()[d['perm']]
            dw_vals.append(w_o - w_r)
        dw_age[a] = np.mean(dw_vals, axis=0)

for k in range(K_GMM):
    valid = ~np.isnan(dw_age[:, k]) & (np.arange(max_age + 1) > 0)
    ax.plot(np.arange(max_age + 1)[valid], dw_age[valid, k], "o-", ms=2,
            color=COMP_COLORS[k], label=f"comp {k+1}")
ax.axhline(0, ls="--", color="gray", lw=0.5)
ax.set_xlabel("Age $a$"); ax.set_ylabel("$\\Delta\\pi_k$")
ax.set_title("(c) Weight drift")
ax.legend(fontsize=9)
ax.set_xlim(1, N_DAYS - 1)

fig.tight_layout()
fig.savefig(os.path.join(FIGS, "d_decomposed_forgetting.pdf"))
fig.savefig(os.path.join(FIGS, "d_decomposed_forgetting.png"))
plt.close(fig)
print("  Saved d_decomposed_forgetting")

# %%
# ═══════════════════════════════════════════════════════════════════════
#  E. K=1 vs K=3 comparison
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  E: K=1 vs K=3 comparison")
print("=" * 60)

daily_k1, means_k1_t = make_daily_gaussians_circle(
    N_DAYS, R=R_DRIFT, period=PERIOD, cov_scale=0.5, d=D)

print("  K=1 run ...", end=" ", flush=True)
mem1, _, Fnorm1, _ = run_cl_loop(
    daily_k1, gm_prior_k1, L=L_DEF, verbose_every=0)
ages1, Fmu1, Fsig1, _, hl1 = compute_age_curves(Fnorm1, N_DAYS)
print(f"a₁/₂ = {hl1}")
print(f"  K=3 half-life: a₁/₂ = {hl3}")

# standalone comparison figure
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

ax = axes[0]
m1 = ~np.isnan(Fmu1) & (ages1 > 0)
m3 = ~np.isnan(Fmu3) & (ages3 > 0)
ax.plot(ages1[m1], Fmu1[m1], "o-", ms=2.5, color=C_BLUE,
        label=f"$K=1$ ($a_{{1/2}}={hl1}$)")
ax.plot(ages3[m3], Fmu3[m3], "s-", ms=2.5, color=C_RED,
        label=f"$K=3$ ($a_{{1/2}}={hl3}$)")
ax.axhline(0.5, ls="--", color="gray", lw=0.6)
ax.axhline(1.0, ls=":", color="gray", lw=0.4, alpha=0.5)
ax.set_xlabel("Age $a$"); ax.set_ylabel(r"$\bar{F}(a)$")
ax.set_title("(a) Age–forgetting: $K=1$ vs $K=3$")
ax.legend(fontsize=10)
ax.set_ylim(-0.03, 1.5); ax.set_xlim(0, N_DAYS)

ax = axes[1]
Fn_clip = np.clip(Fnorm3, 0, 1.0)
im = ax.imshow(Fn_clip, aspect="auto", origin="lower",
               cmap="RdYlGn_r", vmin=0, vmax=1.0,
               extent=[0.5, N_DAYS+0.5, 0.5, N_DAYS+0.5])
ax.plot([0.5, N_DAYS+0.5], [0.5, N_DAYS+0.5], "k-", lw=0.3, alpha=0.3)
ax.set_xlabel("Current day $n$"); ax.set_ylabel("Recalled day $m$")
ax.set_title(r"(b) $\bar{F}(m,n)$ for $K=3$")
plt.colorbar(im, ax=ax, shrink=0.82, label=r"$\bar{F}$", pad=0.02)

fig.tight_layout()
fig.savefig(os.path.join(FIGS, "e_k1_vs_k3.pdf"))
fig.savefig(os.path.join(FIGS, "e_k1_vs_k3.png"))
plt.close(fig)
print("  Saved e_k1_vs_k3")

# %% [markdown]
# ---
# ## F. Paper figures

# %%
# ═══════════════════════════════════════════════════════════════════════
#  PAPER FIGURE 7 (full width):
#  (a) K=1 vs K=3 age curves  (b) Decomposed forgetting (stacked)
# ═══════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(7.0, 3.2))
gs  = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.35)

# (a) K comparison
ax = fig.add_subplot(gs[0])
m1 = ~np.isnan(Fmu1) & (ages1 > 0)
m3 = ~np.isnan(Fmu3) & (ages3 > 0)
ax.plot(ages1[m1], Fmu1[m1], "o-", ms=2.5, color=C_BLUE,
        label=f"$K=1$ ($a_{{1/2}}={hl1}$)")
ax.plot(ages3[m3], Fmu3[m3], "s-", ms=2.5, color=C_RED,
        label=f"$K=3$ ($a_{{1/2}}={hl3}$)")
ax.fill_between(ages3[m3],
    np.clip(Fmu3[m3] - Fsig3[m3], 0, None), Fmu3[m3] + Fsig3[m3],
    alpha=0.1, color=C_RED)
ax.axhline(0.5, ls="--", color="gray", lw=0.6)
ax.axhline(1.0, ls=":", color="gray", lw=0.4, alpha=0.5)
ax.set_xlabel("Age $a$"); ax.set_ylabel(r"$\bar{F}(a)$")
ax.set_title("(a) $K=1$ vs $K=3$", fontsize=11)
ax.legend(fontsize=9, loc="upper left")
ax.set_ylim(-0.03, 1.5); ax.set_xlim(0, N_DAYS)

# (b) Decomposed
ax = fig.add_subplot(gs[1])
a_plot = np.arange(1, max_age + 1)
fm_p = np.nan_to_num(Fm_age[1:])
fc_p = np.nan_to_num(Fc_age[1:])
fw_p = np.nan_to_num(Fw_age[1:])
ax.fill_between(a_plot, 0, fm_p, alpha=0.7, color=C_BLUE,
                label="$F_{\\mathrm{mean}}$")
ax.fill_between(a_plot, fm_p, fm_p + fc_p, alpha=0.7, color=C_ORANGE,
                label="$F_{\\mathrm{cov}}$")
ax.fill_between(a_plot, fm_p + fc_p, fm_p + fc_p + fw_p, alpha=0.7,
                color=C_GREEN, label="$F_{\\mathrm{weight}}$")
ax.set_xlabel("Age $a$"); ax.set_ylabel("Decomposed $F$ (raw)")
ax.set_title("(b) Forgetting decomposition ($K=3$)", fontsize=11)
ax.legend(fontsize=8, loc="upper left")
ax.set_xlim(1, N_DAYS - 1)

fig.savefig(os.path.join(FIGS, "paper_fig7_k1_vs_k3.pdf"))
fig.savefig(os.path.join(FIGS, "paper_fig7_k1_vs_k3.png"))
plt.close(fig)
print("Saved paper_fig7")

# %%
# ═══════════════════════════════════════════════════════════════════════
#  PAPER FIGURE 9 (full width):
#  Component-level trajectory: original vs replay at final day
# ═══════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5))
cmap = plt.cm.viridis

# (a) Original component means
ax = axes[0]
for k in range(K_GMM):
    xs = daily_comp_means[:, k, 0]
    ys = daily_comp_means[:, k, 1]
    ax.scatter(xs, ys, c=np.arange(1, N_DAYS+1), cmap="viridis",
               s=8, alpha=0.5, vmin=1, vmax=N_DAYS, edgecolors="none")
    ax.plot(xs, ys, "-", lw=0.2, color="gray", alpha=0.3)
ax.plot(0, 0, "k*", ms=8, zorder=5)
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
ax.set_title("(a) Original component means", fontsize=11)
ax.set_aspect("equal")
ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5)

# (b) Replay: per-component means at final day
ax = axes[1]
# Background: original faint
for k in range(K_GMM):
    ax.scatter(daily_comp_means[:, k, 0], daily_comp_means[:, k, 1],
               c=np.arange(1, N_DAYS+1), cmap="viridis",
               s=6, alpha=0.15, vmin=1, vmax=N_DAYS, edgecolors="none")

# Replay
for m in range(1, N_DAYS + 1):
    if m not in mem3.readout_times:
        continue
    try:
        gm_rep = mem3.replay(m)
        mu_np = gm_rep.means.detach().numpy()          # (K, d)
        gm_orig = mem3.history[m - 1]
        perm = ForgetMetrics.hungarian_match(gm_orig.means, gm_rep.means)
        c = cmap((m - 1) / max(N_DAYS - 1, 1))
        for k in range(K_GMM):
            ax.plot(mu_np[perm[k], 0], mu_np[perm[k], 1],
                    "x", ms=4, color=c, markeredgewidth=0.8, alpha=0.7)
    except Exception:
        pass

ax.plot(0, 0, "k*", ms=8, zorder=5)
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
ax.set_title("(b) Replay component means (×)", fontsize=11)
ax.set_aspect("equal")
ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5)

fig.tight_layout()
fig.savefig(os.path.join(FIGS, "paper_fig9_k3_trajectories.pdf"))
fig.savefig(os.path.join(FIGS, "paper_fig9_k3_trajectories.png"))
plt.close(fig)
print("Saved paper_fig9")

# %%
# ═══════════════════════════════════════════════════════════════════════
#  Summary
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  Exp03 Summary")
print("=" * 60)
print(f"  Setup: n={N_DAYS}, d={D}, circle drift R={R_DRIFT}, P={PERIOD}")
print(f"  GMM: K={K_GMM}, component offset r={R_COMP}, Σ_k={COV_SC}·I")
print(f"  Protocol: L={L_DEF}")
print(f"\n  Half-lives:")
print(f"    K=1: a₁/₂ = {hl1}")
print(f"    K=3: a₁/₂ = {hl3}")

print(f"\n  Decomposed forgetting (K=3, age=15):")
mask15 = ages_dec == 15
if mask15.sum() > 0:
    fm15 = np.mean(Fm_dec[mask15])
    fc15 = np.mean(Fc_dec[mask15])
    fw15 = np.mean(Fw_dec[mask15])
    ft15 = fm15 + fc15 + fw15
    print(f"    F_mean  = {fm15:.4f}  ({100*fm15/ft15:.0f}%)")
    print(f"    F_cov   = {fc15:.4f}  ({100*fc15/ft15:.0f}%)")
    print(f"    F_weight = {fw15:.6f}  ({100*fw15/ft15:.1f}%)")

print(f"\n  Paper figures in: {FIGS}/")
print(f"    paper_fig7:  K=1 vs K=3 + decomposition")
print(f"    paper_fig9:  component trajectories")
print("=" * 60)

# %%
