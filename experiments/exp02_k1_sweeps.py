# %% [markdown]
# # Exp02: K=1 Parameter Sweeps
#
# Single-Gaussian ($K{=}1$, $d{=}2$): sweep segment budget $L$,
# drift speed (circle period $P$), and circle vs linear geometry.
# Produces paper figures 3, 4.
#
# **Sections**
# - A: Imports, style
# - B: L sweep
# - C: Drift-speed sweep
# - D: Circle vs linear drift
# - E: Paper figures

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
    make_daily_gaussians_circle, make_daily_gaussians_linear,
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
N_DAYS  = 100
D       = 2
R_DRIFT = 2.0
PERIOD  = 50
COV_SC  = 0.5

gm_prior = GaussianMixture(
    weights=torch.ones(1),
    means=torch.zeros(1, D),
    covs=torch.eye(D).unsqueeze(0),
)

print("Imports OK.  torch", torch.__version__)

# %%
# ═══════════════════════════════════════════════════════════════════════
#  B. L sweep
# ═══════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  B: L sweep")
print("=" * 60)

daily_dists, daily_means_t = make_daily_gaussians_circle(
    N_DAYS, R=R_DRIFT, period=PERIOD, cov_scale=COV_SC, d=D)

L_VALS = [5, 8, 10, 15, 20, 30]
res_L = []
for Lv in L_VALS:
    print(f"  L={Lv} ...", end=" ", flush=True)
    _, _, Fn, _ = run_cl_loop(daily_dists, gm_prior, L=Lv, verbose_every=0)
    a, fm, fs, _, hl = compute_age_curves(Fn, N_DAYS)
    res_L.append((Lv, a, fm, fs, hl))
    print(f"a₁/₂={hl}")

# standalone figure
fig, ax = plt.subplots(figsize=(5.5, 3.8))
for i, (Lv, a, fm, fs, hl) in enumerate(res_L):
    m = ~np.isnan(fm) & (a > 0)
    lbl = f"$L={Lv}$"
    if hl is not None:
        lbl += f" ($a_{{1/2}}={hl}$)"
    ax.plot(a[m], fm[m], "o-", ms=2, color=SWEEP_COLORS[i], label=lbl)
ax.axhline(0.5, ls="--", color="gray", lw=0.5)
ax.axhline(1.0, ls=":", color="gray", lw=0.3, alpha=0.5)
ax.set_xlabel("Age $a$")
ax.set_ylabel(r"$\bar{F}(a)$")
ax.set_title("$L$ sweep ($K=1$, circle drift)")
ax.legend(fontsize=7, loc="upper left")
ax.set_ylim(-0.03, 1.3); ax.set_xlim(0, N_DAYS)
fig.savefig(os.path.join(FIGS, "b_L_sweep.pdf"))
fig.savefig(os.path.join(FIGS, "b_L_sweep.png"))
plt.close(fig)
print("  Saved b_L_sweep")

# %%
# ═══════════════════════════════════════════════════════════════════════
#  C. Drift-speed sweep (circle period)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  C: Drift-speed sweep")
print("=" * 60)

L_DEF = 10
SPEED_PERIODS = [25, 50, 100, 200]
res_speed = []
for P in SPEED_PERIODS:
    print(f"  period={P} ...", end=" ", flush=True)
    dd, _ = make_daily_gaussians_circle(N_DAYS, R=R_DRIFT, period=P,
                                         cov_scale=COV_SC, d=D)
    _, _, Fn, _ = run_cl_loop(dd, gm_prior, L=L_DEF, verbose_every=0)
    a, fm, fs, _, hl = compute_age_curves(Fn, N_DAYS)
    res_speed.append((P, a, fm, fs, hl))
    print(f"a₁/₂={hl}")

fig, ax = plt.subplots(figsize=(5.5, 3.8))
for i, (P, a, fm, fs, hl) in enumerate(res_speed):
    m = ~np.isnan(fm) & (a > 0)
    lbl = f"$P={P}$"
    if hl is not None:
        lbl += f" ($a_{{1/2}}={hl}$)"
    ax.plot(a[m], fm[m], "o-", ms=2, color=SWEEP_COLORS[i], label=lbl)
ax.axhline(0.5, ls="--", color="gray", lw=0.5)
ax.axhline(1.0, ls=":", color="gray", lw=0.3, alpha=0.5)
ax.set_xlabel("Age $a$")
ax.set_ylabel(r"$\bar{F}(a)$")
ax.set_title(f"Drift speed (period $P$), $L={L_DEF}$")
ax.legend(fontsize=7, loc="upper left")
ax.set_ylim(-0.03, 1.3); ax.set_xlim(0, N_DAYS)
fig.savefig(os.path.join(FIGS, "c_speed_sweep.pdf"))
fig.savefig(os.path.join(FIGS, "c_speed_sweep.png"))
plt.close(fig)
print("  Saved c_speed_sweep")

# %%
# ═══════════════════════════════════════════════════════════════════════
#  D. Circle vs linear drift
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  D: Circle vs linear drift")
print("=" * 60)

# circle (default)
print("  circle ...", end=" ", flush=True)
_, _, Fn_circ, _ = run_cl_loop(daily_dists, gm_prior, L=L_DEF, verbose_every=0)
a_circ, fm_circ, fs_circ, _, hl_circ = compute_age_curves(Fn_circ, N_DAYS)
print(f"a₁/₂={hl_circ}")

# linear
print("  linear ...", end=" ", flush=True)
daily_lin, means_lin_t = make_daily_gaussians_linear(
    N_DAYS, speed=0.15, cov_scale=COV_SC, d=D)
_, _, Fn_lin, _ = run_cl_loop(daily_lin, gm_prior, L=L_DEF, verbose_every=0)
a_lin, fm_lin, fs_lin, _, hl_lin = compute_age_curves(Fn_lin, N_DAYS)
print(f"a₁/₂={hl_lin}")

fig, ax = plt.subplots(figsize=(5.5, 3.8))
mc = ~np.isnan(fm_circ) & (a_circ > 0)
ml = ~np.isnan(fm_lin)  & (a_lin > 0)
ax.plot(a_circ[mc], fm_circ[mc], "o-", ms=2, color=C_BLUE,
        label=f"circle ($a_{{1/2}}$={hl_circ})")
ax.plot(a_lin[ml], fm_lin[ml], "s-", ms=2, color=C_ORANGE,
        label=f"linear ($a_{{1/2}}$={hl_lin})")
ax.axhline(0.5, ls="--", color="gray", lw=0.5)
ax.axhline(1.0, ls=":", color="gray", lw=0.3, alpha=0.5)
ax.set_xlabel("Age $a$"); ax.set_ylabel(r"$\bar{F}(a)$")
ax.set_title(f"Circle vs linear drift ($L={L_DEF}$)")
ax.legend(fontsize=8, loc="upper left")
ax.set_ylim(-0.03, 1.3); ax.set_xlim(0, N_DAYS)
fig.savefig(os.path.join(FIGS, "d_circle_vs_linear.pdf"))
fig.savefig(os.path.join(FIGS, "d_circle_vs_linear.png"))
plt.close(fig)
print("  Saved d_circle_vs_linear")

# %% [markdown]
# ---
# ## E. Paper figures

# %%
# ═══════════════════════════════════════════════════════════════════════
#  PAPER FIGURE 3  (full width):
#  (a) L sweep  (b) half-life vs L
# ═══════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))

# ── (a) L sweep curves ──
ax = axes[0]
for i, (Lv, a, fm, fs, hl) in enumerate(res_L):
    m = ~np.isnan(fm) & (a > 0)
    ax.plot(a[m], fm[m], "o-", ms=2, color=SWEEP_COLORS[i],
            label=f"$L={Lv}$")
ax.axhline(0.5, ls="--", color="gray", lw=0.5)
ax.axhline(1.0, ls=":", color="gray", lw=0.3, alpha=0.5)
ax.set_xlabel("Age $a$")
ax.set_ylabel(r"$\bar{F}(a)$")
ax.set_title("(a) $L$ sweep", fontsize=11)
ax.legend(fontsize=7, ncol=2, loc="upper left")
ax.set_ylim(-0.03, 1.3); ax.set_xlim(0, N_DAYS)

# ── (b) half-life vs L ──
ax = axes[1]
L_plot = [lv for lv, _, _, _, _ in res_L]
hl_plot = [hl if hl is not None else N_DAYS for _, _, _, _, hl in res_L]
ax.bar([str(l) for l in L_plot], hl_plot,
       color=[SWEEP_COLORS[i] for i in range(len(L_plot))], alpha=0.85)
for i, (l, h) in enumerate(zip(L_plot, hl_plot)):
    ax.text(i, h + 0.5, str(h), ha="center", va="bottom", fontsize=9)
ax.set_xlabel("Segment budget $L$")
ax.set_ylabel("Retention half-life $a_{1/2}$")
ax.set_title("(b) Half-life vs $L$", fontsize=11)

fig.tight_layout()
fig.savefig(os.path.join(FIGS, "paper_fig3_L_sweep.pdf"))
fig.savefig(os.path.join(FIGS, "paper_fig3_L_sweep.png"))
plt.close(fig)
print("Saved paper_fig3")

# %%
# ═══════════════════════════════════════════════════════════════════════
#  PAPER FIGURE 4  (full width):
#  (a) Drift speed  (b) Circle vs linear  (c) half-life vs P
# ═══════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.8))

# ── (a) drift speed ──
ax = axes[0]
for i, (P, a, fm, fs, hl) in enumerate(res_speed):
    m = ~np.isnan(fm) & (a > 0)
    hl_s = f", $a_{{1/2}}$={hl}" if hl else ""
    ax.plot(a[m], fm[m], "o-", ms=2, color=SWEEP_COLORS[i],
            label=f"$P={P}${hl_s}")
ax.axhline(0.5, ls="--", color="gray", lw=0.5)
ax.axhline(1.0, ls=":", color="gray", lw=0.3, alpha=0.5)
ax.set_xlabel("Age $a$"); ax.set_ylabel(r"$\bar{F}(a)$")
ax.set_title("(a) Drift speed", fontsize=11)
ax.legend(fontsize=6, loc="upper left")
ax.set_ylim(-0.03, 1.3); ax.set_xlim(0, N_DAYS)

# ── (b) circle vs linear ──
ax = axes[1]
mc = ~np.isnan(fm_circ) & (a_circ > 0)
ml = ~np.isnan(fm_lin)  & (a_lin > 0)
ax.plot(a_circ[mc], fm_circ[mc], "o-", ms=2, color=C_BLUE,
        label=f"circle ($a_{{1/2}}$={hl_circ})")
ax.plot(a_lin[ml], fm_lin[ml], "s-", ms=2, color=C_ORANGE,
        label=f"linear ($a_{{1/2}}$={hl_lin})")
ax.axhline(0.5, ls="--", color="gray", lw=0.5)
ax.axhline(1.0, ls=":", color="gray", lw=0.3, alpha=0.5)
ax.set_xlabel("Age $a$"); ax.set_ylabel(r"$\bar{F}(a)$")
ax.set_title("(b) Circle vs linear", fontsize=11)
ax.legend(fontsize=7, loc="upper left")
ax.set_ylim(-0.03, 1.3); ax.set_xlim(0, N_DAYS)

# ── (c) half-life vs P ──
ax = axes[2]
P_plot = [p for p, _, _, _, _ in res_speed]
hl_p = [hl if hl is not None else N_DAYS for _, _, _, _, hl in res_speed]
ax.bar([str(p) for p in P_plot], hl_p,
       color=[SWEEP_COLORS[i] for i in range(len(P_plot))], alpha=0.85)
for i, (p, h) in enumerate(zip(P_plot, hl_p)):
    ax.text(i, h + 0.3, str(h), ha="center", va="bottom", fontsize=8)
ax.set_xlabel("Period $P$")
ax.set_ylabel("$a_{1/2}$")
ax.set_title("(c) Half-life vs speed", fontsize=11)

fig.tight_layout()
fig.savefig(os.path.join(FIGS, "paper_fig4_speed_shape.pdf"))
fig.savefig(os.path.join(FIGS, "paper_fig4_speed_shape.png"))
plt.close(fig)
print("Saved paper_fig4")

# %%
# ═══════════════════════════════════════════════════════════════════════
#  Summary
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  Exp02 Summary")
print("=" * 60)

print(f"\n  L sweep:  {dict(zip(L_VALS, [h for _, _, _, _, h in res_L]))}")
print(f"  Speed:    {dict(zip(SPEED_PERIODS, [h for _, _, _, _, h in res_speed]))}")
print(f"  Circle:   a₁/₂ = {hl_circ}")
print(f"  Linear:   a₁/₂ = {hl_lin}")

print(f"\n  Paper figures in: {FIGS}/")
print(f"    paper_fig3:  L sweep + half-life vs L")
print(f"    paper_fig4:  drift speed + circle vs linear + half-life vs P")
print("=" * 60)

# %%
