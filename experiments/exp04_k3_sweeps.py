# %% [markdown]
# # Exp04: K=3 Sweeps — L sweep, K sweep
#
# Gaussian-mixture daily targets: sweep $L$ at fixed $K{=}3$,
# and sweep $K$ at fixed $L{=}10$.
# Produces paper figures 8, 10.
#
# **Sections**
# - A: Imports, style
# - B: L sweep (K=3)
# - C: K sweep
# - D: Paper figures

# %%
# ═══════════════════════════════════════════════════════════════════════
#  A. Imports + style
# ═══════════════════════════════════════════════════════════════════════
import os, sys
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if '__file__' in dir() else os.getcwd())
from bridge_cas import (
    GaussianMixture,
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
SWEEP_COLORS = [C_BLUE, C_ORANGE, C_GREEN, C_RED, C_PURPLE, "#636363"]

# ── common parameters ────────────────────────────────────────────────
N_DAYS   = 100
D        = 2
R_DRIFT  = 2.0
R_COMP   = 0.8
PERIOD   = 50
COV_SC   = 0.3
L_DEF    = 10
K_DEF    = 3

print("Imports OK.  torch", torch.__version__)


def make_prior(K, D):
    """Standard isotropic prior with K components at origin."""
    return GaussianMixture(
        weights=torch.ones(K) / K,
        means=torch.zeros(K, D),
        covs=torch.eye(D).unsqueeze(0).expand(K, -1, -1).clone(),
    )


# %%
# ═══════════════════════════════════════════════════════════════════════
#  B. L sweep (K=3)
# ═══════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  B: L sweep (K=3)")
print("=" * 60)

daily_gmm, _ = make_daily_gmm_circle(
    N_DAYS, K=K_DEF, R=R_DRIFT, r=R_COMP, period=PERIOD,
    cov_scale=COV_SC, d=D)

L_VALS = [5, 10, 15, 20, 30]
res_L3 = []
for Lv in L_VALS:
    print(f"  L={Lv} ...", end=" ", flush=True)
    prior = make_prior(K_DEF, D)
    _, _, Fn, _ = run_cl_loop(daily_gmm, prior, L=Lv, verbose_every=0)
    a, fm, fs, _, hl = compute_age_curves(Fn, N_DAYS)
    res_L3.append((Lv, a, fm, fs, hl))
    print(f"a₁/₂={hl}")

# standalone figure
fig, ax = plt.subplots(figsize=(5.5, 3.8))
for i, (Lv, a, fm, fs, hl) in enumerate(res_L3):
    m = ~np.isnan(fm) & (a > 0)
    lbl = f"$L={Lv}$"
    if hl is not None:
        lbl += f" ($a_{{1/2}}={hl}$)"
    ax.plot(a[m], fm[m], "o-", ms=2, color=SWEEP_COLORS[i], label=lbl)
ax.axhline(0.5, ls="--", color="gray", lw=0.5)
ax.axhline(1.0, ls=":", color="gray", lw=0.3, alpha=0.5)
ax.set_xlabel("Age $a$"); ax.set_ylabel(r"$\bar{F}(a)$")
ax.set_title(f"$L$ sweep ($K={K_DEF}$, circle drift)")
ax.legend(fontsize=7, loc="upper left")
ax.set_ylim(-0.03, 1.5); ax.set_xlim(0, N_DAYS)
fig.savefig(os.path.join(FIGS, "b_L_sweep_k3.pdf"))
fig.savefig(os.path.join(FIGS, "b_L_sweep_k3.png"))
plt.close(fig)
print("  Saved b_L_sweep_k3")

# %%
# ═══════════════════════════════════════════════════════════════════════
#  C. K sweep (L=10)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  C: K sweep")
print("=" * 60)

K_VALS = [1, 2, 3, 5, 8]
res_K = []
for Kv in K_VALS:
    print(f"  K={Kv} ...", end=" ", flush=True)
    if Kv == 1:
        dd, _ = make_daily_gaussians_circle(
            N_DAYS, R=R_DRIFT, period=PERIOD, cov_scale=0.5, d=D)
    else:
        dd, _ = make_daily_gmm_circle(
            N_DAYS, K=Kv, R=R_DRIFT, r=R_COMP, period=PERIOD,
            cov_scale=COV_SC, d=D)
    prior = make_prior(Kv, D)
    _, _, Fn, _ = run_cl_loop(dd, prior, L=L_DEF, verbose_every=0)
    a, fm, fs, _, hl = compute_age_curves(Fn, N_DAYS)
    res_K.append((Kv, a, fm, fs, hl))
    print(f"a₁/₂={hl}")

# standalone figure: age curves
fig, ax = plt.subplots(figsize=(5.5, 3.8))
for i, (Kv, a, fm, fs, hl) in enumerate(res_K):
    m = ~np.isnan(fm) & (a > 0)
    lbl = f"$K={Kv}$"
    if hl is not None:
        lbl += f" ($a_{{1/2}}={hl}$)"
    ax.plot(a[m], fm[m], "o-", ms=2, color=SWEEP_COLORS[i], label=lbl)
ax.axhline(0.5, ls="--", color="gray", lw=0.5)
ax.axhline(1.0, ls=":", color="gray", lw=0.3, alpha=0.5)
ax.set_xlabel("Age $a$"); ax.set_ylabel(r"$\bar{F}(a)$")
ax.set_title(f"$K$ sweep ($L={L_DEF}$)")
ax.legend(fontsize=7, loc="upper left")
ax.set_ylim(-0.03, 1.5); ax.set_xlim(0, N_DAYS)
fig.savefig(os.path.join(FIGS, "c_K_sweep.pdf"))
fig.savefig(os.path.join(FIGS, "c_K_sweep.png"))
plt.close(fig)
print("  Saved c_K_sweep")

# %% [markdown]
# ---
# ## D. Paper figures

# %%
# ═══════════════════════════════════════════════════════════════════════
#  PAPER FIGURE 8 (full width):
#  (a) L sweep (K=3)  (b) K sweep  (c) half-life vs L and K
# ═══════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.8))

# (a) L sweep
ax = axes[0]
for i, (Lv, a, fm, fs, hl) in enumerate(res_L3):
    m = ~np.isnan(fm) & (a > 0)
    ax.plot(a[m], fm[m], "o-", ms=2, color=SWEEP_COLORS[i],
            label=f"$L={Lv}$")
ax.axhline(0.5, ls="--", color="gray", lw=0.5)
ax.axhline(1.0, ls=":", color="gray", lw=0.3, alpha=0.5)
ax.set_xlabel("Age $a$"); ax.set_ylabel(r"$\bar{F}(a)$")
ax.set_title("(a) $L$ sweep ($K=3$)", fontsize=11)
ax.legend(fontsize=6.5, loc="upper left")
ax.set_ylim(-0.03, 1.5); ax.set_xlim(0, N_DAYS)

# (b) K sweep
ax = axes[1]
for i, (Kv, a, fm, fs, hl) in enumerate(res_K):
    m = ~np.isnan(fm) & (a > 0)
    ax.plot(a[m], fm[m], "o-", ms=2, color=SWEEP_COLORS[i],
            label=f"$K={Kv}$")
ax.axhline(0.5, ls="--", color="gray", lw=0.5)
ax.axhline(1.0, ls=":", color="gray", lw=0.3, alpha=0.5)
ax.set_xlabel("Age $a$"); ax.set_ylabel(r"$\bar{F}(a)$")
ax.set_title("(b) $K$ sweep ($L=10$)", fontsize=11)
ax.legend(fontsize=6.5, loc="upper left")
ax.set_ylim(-0.03, 1.5); ax.set_xlim(0, N_DAYS)

# (c) Half-life summary: grouped bar
ax = axes[2]
# L half-lives
L_labels = [str(lv) for lv, _, _, _, _ in res_L3]
L_hls    = [hl if hl is not None else N_DAYS for _, _, _, _, hl in res_L3]
x_L = np.arange(len(L_labels))
bars_L = ax.bar(x_L - 0.18, L_hls, width=0.32, color=C_BLUE,
                alpha=0.8, label="$L$ sweep")
for i, h in enumerate(L_hls):
    ax.text(x_L[i] - 0.18, h + 0.5, str(h), ha="center",
            va="bottom", fontsize=6, color=C_BLUE)

# K half-lives (on twin axis for independent scale)
ax_r = ax.twinx()
K_labels = [str(kv) for kv, _, _, _, _ in res_K]
K_hls    = [hl if hl is not None else N_DAYS for _, _, _, _, hl in res_K]
x_K = np.arange(len(K_labels))
bars_K = ax_r.bar(x_K + 0.18, K_hls, width=0.32, color=C_ORANGE,
                  alpha=0.8, label="$K$ sweep")
for i, h in enumerate(K_hls):
    ax_r.text(x_K[i] + 0.18, h + 0.5, str(h), ha="center",
              va="bottom", fontsize=6, color=C_ORANGE)

ax.set_xlabel("$L$ (blue) / $K$ (orange)")
ax.set_ylabel(r"$a_{1/2}$ ($L$ sweep)", color=C_BLUE)
ax_r.set_ylabel(r"$a_{1/2}$ ($K$ sweep)", color=C_ORANGE)
ax.tick_params(axis='y', labelcolor=C_BLUE)
ax_r.tick_params(axis='y', labelcolor=C_ORANGE)
ax.set_title("(c) Half-life summary", fontsize=11)

# combine x-tick labels (use whichever is longer)
if len(L_labels) >= len(K_labels):
    ax.set_xticks(x_L)
    ax.set_xticklabels(L_labels, fontsize=7)
else:
    ax.set_xticks(x_K)
    ax.set_xticklabels(K_labels, fontsize=7)

fig.tight_layout()
fig.savefig(os.path.join(FIGS, "paper_fig8_k3_sweeps.pdf"))
fig.savefig(os.path.join(FIGS, "paper_fig8_k3_sweeps.png"))
plt.close(fig)
print("Saved paper_fig8")

# %%
# ═══════════════════════════════════════════════════════════════════════
#  PAPER FIGURE 10 (single column):
#  Half-life vs K — the "phase diagram" question
# ═══════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(4, 3))
K_plot  = [kv for kv, _, _, _, _ in res_K]
hl_plot = [hl if hl is not None else N_DAYS for _, _, _, _, hl in res_K]
ax.bar([str(k) for k in K_plot], hl_plot,
       color=[SWEEP_COLORS[i] for i in range(len(K_plot))], alpha=0.85)
for i, (k, h) in enumerate(zip(K_plot, hl_plot)):
    ax.text(i, h + 0.5, str(h), ha="center", va="bottom", fontsize=9)
ax.set_xlabel("Number of components $K$")
ax.set_ylabel("Retention half-life $a_{1/2}$")
ax.set_title("Half-life vs $K$")
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "paper_fig10_halflife_vs_K.pdf"))
fig.savefig(os.path.join(FIGS, "paper_fig10_halflife_vs_K.png"))
plt.close(fig)
print("Saved paper_fig10")

# %%
# ═══════════════════════════════════════════════════════════════════════
#  Summary
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  Exp04 Summary")
print("=" * 60)

print(f"\n  L sweep (K={K_DEF}):")
for Lv, _, _, _, hl in res_L3:
    print(f"    L={Lv:2d}  →  a₁/₂ = {hl}")

print(f"\n  K sweep (L={L_DEF}):")
for Kv, _, _, _, hl in res_K:
    print(f"    K={Kv}  →  a₁/₂ = {hl}")

print(f"\n  Paper figures in: {FIGS}/")
print(f"    paper_fig8:   L sweep + K sweep + half-life summary")
print(f"    paper_fig10:  half-life vs K")
print("=" * 60)

# %%
