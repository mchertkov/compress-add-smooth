# %% [markdown]
# # Exp01: K=1 Default Run — Core Diagnostics
#
# Single-Gaussian daily targets ($K{=}1$, $d{=}2$), circular drift.
# Produces paper figures 1, 2, 6.
#
# **Sections**
# - A: Imports, style
# - B: Default CL run
# - C: Core diagnostics (age curve, heatmap, trajectory, readout, ellipses)
# - D: Paper figures (combined panels)

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
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse

# ── locate bridge_cas ────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if '__file__' in dir() else os.getcwd())
from bridge_cas import (
    GaussianMixture, ContinualMemory,
    make_daily_gaussians_circle, make_daily_gaussians_linear,
    run_cl_loop, compute_age_curves,
)

# ── publication style ────────────────────────────────────────────────
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
    "text.usetex":        False,
})

FIGS = os.path.join(os.getcwd(), "figs")
os.makedirs(FIGS, exist_ok=True)

C_BLUE   = "#2166ac"
C_ORANGE = "#e08214"
C_GREEN  = "#1a9641"
C_RED    = "#d73027"
C_PURPLE = "#7570b3"

print("Imports OK.  torch", torch.__version__)

# %%
# ═══════════════════════════════════════════════════════════════════════
#  B. Default CL run
# ═══════════════════════════════════════════════════════════════════════

N_DAYS  = 100
D       = 2
R_DRIFT = 2.0
PERIOD  = 50
COV_SC  = 0.5
L_DEF   = 10

daily_dists, daily_means_t = make_daily_gaussians_circle(
    N_DAYS, R=R_DRIFT, period=PERIOD, cov_scale=COV_SC, d=D)
daily_means = daily_means_t.numpy()

gm_prior = GaussianMixture(
    weights=torch.ones(1),
    means=torch.zeros(1, D),
    covs=torch.eye(D).unsqueeze(0),
)

print(f"Circle drift: n={N_DAYS}, R={R_DRIFT}, P={PERIOD}, Σ={COV_SC}·I")
print(f"Protocol:     L={L_DEF}")
print()

mem_def, Fraw_def, Fnorm_def, snaps_def = run_cl_loop(
    daily_dists, gm_prior, L=L_DEF, verbose_every=25)

ages_def, Fmu_def, Fsig_def, cnt_def, hl_def = compute_age_curves(Fnorm_def, N_DAYS)
print(f"\nRetention half-life a₁/₂ = {hl_def}")

# ── helpers ──────────────────────────────────────────────────────────

def replay_means_at_final(mem, n):
    """Collect replay means at the final day for all stored past days."""
    ms, ds = [], []
    for m in range(1, n + 1):
        if m in mem.readout_times:
            try:
                gm = mem.replay(m)
                ms.append(gm.overall_mean().detach().numpy())
                ds.append(m)
            except Exception:
                pass
    return (np.array(ms), np.array(ds)) if ms else (None, None)


def cov_ellipse(mu, cov, n_std=2.0, **kwargs):
    """Return a matplotlib Ellipse for a 2D Gaussian."""
    vals, vecs = np.linalg.eigh(cov[:2, :2])
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w, h = 2 * n_std * np.sqrt(np.maximum(vals, 0))
    return Ellipse(xy=mu[:2], width=w, height=h, angle=angle, **kwargs)


# %% [markdown]
# ---
# ## C. Core diagnostics

# %%
# ═══════════════════════════════════════════════════════════════════════
#  C1. Age–forgetting curve
# ═══════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(5.5, 3.8))
mask = ~np.isnan(Fmu_def) & (ages_def > 0)
ax.plot(ages_def[mask], np.clip(Fmu_def[mask], 0, None), "o-",
        ms=3, color=C_BLUE, label=r"$\bar{F}(a)$")
ax.fill_between(ages_def[mask],
    np.clip(Fmu_def[mask] - Fsig_def[mask], 0, None),
    np.clip(Fmu_def[mask] + Fsig_def[mask], 0, None),
    alpha=0.15, color=C_BLUE)
ax.axhline(0.5, ls="--", color="gray", lw=0.7, label=r"$\theta=0.5$")
ax.axhline(1.0, ls=":", color="gray", lw=0.5, alpha=0.5)
if hl_def is not None:
    ax.axvline(hl_def, ls=":", color=C_RED, lw=1,
               label=f"$a_{{1/2}}={hl_def}$")
ax.set_xlabel("Age $a = n - m$")
ax.set_ylabel(r"Normalised forgetting $\bar{F}(a)$")
ax.set_title(f"Age–forgetting curve  ($L={L_DEF}$, circle drift)")
ax.set_ylim(-0.03, 1.35)
ax.set_xlim(0, N_DAYS)
ax.legend(loc="upper left", framealpha=0.9)
fig.savefig(os.path.join(FIGS, "c1_age_curve.pdf"))
fig.savefig(os.path.join(FIGS, "c1_age_curve.png"))
plt.close(fig)
print("Saved c1_age_curve")

# %%
# ═══════════════════════════════════════════════════════════════════════
#  C2. Forgetting heatmap F̄(m, n)
# ═══════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(5.5, 5.0))
Fn_clip = np.clip(Fnorm_def, 0, 1.2)
im = ax.imshow(Fn_clip, aspect="auto", origin="lower",
               cmap="RdYlGn_r", vmin=0, vmax=1.0,
               extent=[0.5, N_DAYS+0.5, 0.5, N_DAYS+0.5])
ax.plot([0.5, N_DAYS+0.5], [0.5, N_DAYS+0.5], "k-", lw=0.5, alpha=0.3)
ax.set_xlabel("Current day $n$")
ax.set_ylabel("Recalled day $m$")
ax.set_title(r"$\bar{F}(m,n)$  ($L=%d$)" % L_DEF)
plt.colorbar(im, ax=ax, shrink=0.85, label=r"$\bar{F}$")
fig.savefig(os.path.join(FIGS, "c2_heatmap.pdf"))
fig.savefig(os.path.join(FIGS, "c2_heatmap.png"))
plt.close(fig)
print("Saved c2_heatmap")

# %%
# ═══════════════════════════════════════════════════════════════════════
#  C3. Mean trajectory: original vs replay
# ═══════════════════════════════════════════════════════════════════════

rep_means, rep_days = replay_means_at_final(mem_def, N_DAYS)

fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.plot(daily_means[:, 0], daily_means[:, 1], "-", lw=0.4,
        color="gray", alpha=0.5, zorder=1)
sc_orig = ax.scatter(daily_means[:, 0], daily_means[:, 1],
    c=np.arange(1, N_DAYS+1), cmap="viridis", s=18, zorder=2,
    edgecolors="none", vmin=1, vmax=N_DAYS, label="original $\\mu$")
if rep_means is not None:
    ax.scatter(rep_means[:, 0], rep_means[:, 1],
        c=rep_days, cmap="viridis", s=50, marker="x", linewidths=1.2,
        zorder=3, vmin=1, vmax=N_DAYS, label="replay $\\mu$")
ax.plot(0, 0, "k*", ms=10, zorder=5, label="$x_0$")
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
ax.set_title("Original vs replay means (day $n=%d$)" % N_DAYS)
ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
ax.set_aspect("equal")
ax.set_xlim(-2.8, 2.8); ax.set_ylim(-2.8, 2.8)
plt.colorbar(sc_orig, ax=ax, shrink=0.75, label="day")
fig.savefig(os.path.join(FIGS, "c3_trajectory.pdf"))
fig.savefig(os.path.join(FIGS, "c3_trajectory.png"))
plt.close(fig)
print("Saved c3_trajectory")

# %%
# ═══════════════════════════════════════════════════════════════════════
#  C4. Readout time decay
# ═══════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(5.5, 3.5))
for m in range(1, N_DAYS + 1):
    ts, ns = [], []
    for snap in snaps_def:
        if m in snap["readout_times"]:
            ts.append(snap["readout_times"][m]); ns.append(snap["day"])
    if ts:
        ax.plot(ns, ts, "-", lw=0.3, color=C_BLUE, alpha=0.25)

ratio = L_DEF / (L_DEF + 1)
for m_ex in [1, 25, 50, 75]:
    n_arr = np.arange(m_ex, N_DAYS + 1)
    t_thy = ratio ** (n_arr - m_ex)
    ax.plot(n_arr, t_thy, "k--", lw=0.6, alpha=0.4)

ax.set_xlabel("Current day $n$")
ax.set_ylabel("Readout time $t_{m|n}$")
ax.set_title("Readout-time decay (blue=actual, dashed=$(L/(L+1))^{n-m}$)")
ax.set_ylim(-0.02, 1.02)
fig.savefig(os.path.join(FIGS, "c4_readout.pdf"))
fig.savefig(os.path.join(FIGS, "c4_readout.png"))
plt.close(fig)
print("Saved c4_readout")

# %%
# ═══════════════════════════════════════════════════════════════════════
#  C5. Confusion ellipses for selected days
# ═══════════════════════════════════════════════════════════════════════

ZOOM_DAYS = [5, 15, 25, 35, 50, 75, 95]

fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.plot(daily_means[:, 0], daily_means[:, 1], "-", lw=0.3,
        color="gray", alpha=0.3, zorder=1)

cmap = plt.cm.viridis
norm = plt.Normalize(1, N_DAYS)

for m in ZOOM_DAYS:
    if m > N_DAYS or m not in mem_def.readout_times:
        continue
    try:
        gm_rep = mem_def.replay(m)
        mu_np  = gm_rep.means[0].detach().numpy()
        cov_np = gm_rep.covs[0].detach().numpy()
        orig_mu = daily_means[m - 1]
        c = cmap(norm(m))

        ax.plot(orig_mu[0], orig_mu[1], "o", ms=7, color=c, zorder=3)
        ax.annotate(f"d{m}", xy=orig_mu[:2], fontsize=7,
                    xytext=(4, 4), textcoords="offset points", color=c)

        ell = cov_ellipse(mu_np, cov_np, n_std=2.0,
                          fill=False, edgecolor=c, lw=1.2, ls="--", zorder=4)
        ax.add_patch(ell)
        ax.plot(mu_np[0], mu_np[1], "x", ms=7, color=c,
                markeredgewidth=1.5, zorder=5)

        ax.annotate("", xy=orig_mu[:2], xytext=mu_np[:2],
                    arrowprops=dict(arrowstyle="->", color=c, lw=0.8,
                                    alpha=0.6))
    except Exception as e:
        print(f"  day {m}: {e}")

ax.plot(0, 0, "k*", ms=10, zorder=6)
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
ax.set_title("Replay (×, dashed ellipse) vs original (dots) for selected days")
ax.set_aspect("equal")
ax.set_xlim(-3.2, 3.2); ax.set_ylim(-3.2, 3.2)
fig.savefig(os.path.join(FIGS, "c5_confusion_ellipses.pdf"))
fig.savefig(os.path.join(FIGS, "c5_confusion_ellipses.png"))
plt.close(fig)
print("Saved c5_confusion_ellipses")

# %% [markdown]
# ---
# ## D. Paper figures

# %%
# ═══════════════════════════════════════════════════════════════════════
#  PAPER FIGURE 1  (full width):
#  (a) Age–forgetting curve  (b) Forgetting heatmap
# ═══════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(7.0, 3.2))
gs  = gridspec.GridSpec(1, 2, width_ratios=[1.1, 1], wspace=0.35)

# ── (a) age curve ──
ax = fig.add_subplot(gs[0])
mask = ~np.isnan(Fmu_def) & (ages_def > 0)
ax.plot(ages_def[mask], Fmu_def[mask], "o-", ms=2.5, color=C_BLUE)
ax.fill_between(ages_def[mask],
    np.clip(Fmu_def[mask] - Fsig_def[mask], 0, None),
    Fmu_def[mask] + Fsig_def[mask],
    alpha=0.12, color=C_BLUE)
ax.axhline(0.5, ls="--", color="gray", lw=0.6)
ax.axhline(1.0, ls=":", color="gray", lw=0.4, alpha=0.5)
if hl_def is not None:
    ax.axvline(hl_def, ls=":", color=C_RED, lw=0.8,
               label=f"$a_{{1/2}}={hl_def}$")
ax.set_xlabel("Age $a = n-m$")
ax.set_ylabel(r"$\bar{F}(a)$")
ax.set_ylim(-0.03, 1.3)
ax.set_xlim(0, N_DAYS)
ax.legend(loc="upper left", fontsize=9)
ax.set_title("(a) Age–forgetting curve", fontsize=11)

# ── (b) heatmap ──
ax2 = fig.add_subplot(gs[1])
Fn_clip = np.clip(Fnorm_def, 0, 1.0)
im = ax2.imshow(Fn_clip, aspect="auto", origin="lower",
                cmap="RdYlGn_r", vmin=0, vmax=1.0,
                extent=[0.5, N_DAYS+0.5, 0.5, N_DAYS+0.5])
ax2.plot([0.5, N_DAYS+0.5], [0.5, N_DAYS+0.5], "k-", lw=0.3, alpha=0.3)
ax2.set_xlabel("Current day $n$")
ax2.set_ylabel("Recalled day $m$")
ax2.set_title(r"(b) $\bar{F}(m,n)$", fontsize=11)
plt.colorbar(im, ax=ax2, shrink=0.82, label=r"$\bar{F}$", pad=0.02)

fig.savefig(os.path.join(FIGS, "paper_fig1_age_and_heatmap.pdf"))
fig.savefig(os.path.join(FIGS, "paper_fig1_age_and_heatmap.png"))
plt.close(fig)
print("Saved paper_fig1")

# %%
# ═══════════════════════════════════════════════════════════════════════
#  PAPER FIGURE 2  (full width):
#  (a) Original vs replay trajectory  (b) Readout-time decay
# ═══════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(7.0, 3.5))
gs  = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.35)

# ── (a) trajectory ──
ax = fig.add_subplot(gs[0])
ax.plot(daily_means[:, 0], daily_means[:, 1], "-", lw=0.3,
        color="gray", alpha=0.5, zorder=1)
sc = ax.scatter(daily_means[:, 0], daily_means[:, 1],
    c=np.arange(1, N_DAYS+1), cmap="viridis", s=14, zorder=2,
    edgecolors="none", vmin=1, vmax=N_DAYS)
rep_m, rep_d = replay_means_at_final(mem_def, N_DAYS)
if rep_m is not None:
    ax.scatter(rep_m[:, 0], rep_m[:, 1],
        c=rep_d, cmap="viridis", s=35, marker="x", linewidths=1.0,
        zorder=3, vmin=1, vmax=N_DAYS)
ax.plot(0, 0, "k*", ms=8, zorder=5)
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
ax.set_title("(a) Original (dots) vs replay (×)", fontsize=11)
ax.set_aspect("equal")
ax.set_xlim(-2.8, 2.8); ax.set_ylim(-2.8, 2.8)
plt.colorbar(sc, ax=ax, shrink=0.78, label="day", pad=0.02)

# ── (b) readout decay ──
ax2 = fig.add_subplot(gs[1])
for m in range(1, N_DAYS + 1):
    ts, ns = [], []
    for snap in snaps_def:
        if m in snap["readout_times"]:
            ts.append(snap["readout_times"][m]); ns.append(snap["day"])
    if ts:
        ax2.plot(ns, ts, "-", lw=0.2, color=C_BLUE, alpha=0.2)
ratio = L_DEF / (L_DEF + 1)
for m_ex in [1, 25, 50, 75]:
    na = np.arange(m_ex, N_DAYS + 1)
    ax2.plot(na, ratio**(na - m_ex), "k--", lw=0.5, alpha=0.4)
ax2.set_xlabel("Current day $n$")
ax2.set_ylabel("Readout time $t_{m|n}$")
ax2.set_title("(b) Readout-time decay", fontsize=11)
ax2.set_ylim(-0.02, 1.02)

fig.savefig(os.path.join(FIGS, "paper_fig2_traj_and_readout.pdf"))
fig.savefig(os.path.join(FIGS, "paper_fig2_traj_and_readout.png"))
plt.close(fig)
print("Saved paper_fig2")

# %%
# ═══════════════════════════════════════════════════════════════════════
#  PAPER FIGURE 6  (single-column):
#  Confusion zoom: replay covariance ellipses for selected days
# ═══════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.plot(daily_means[:, 0], daily_means[:, 1], "-", lw=0.3,
        color="gray", alpha=0.3, zorder=1)
cmap = plt.cm.viridis
norm_c = plt.Normalize(1, N_DAYS)

for m in ZOOM_DAYS:
    if m > N_DAYS or m not in mem_def.readout_times:
        continue
    try:
        gm_rep = mem_def.replay(m)
        mu_np  = gm_rep.means[0].detach().numpy()
        cov_np = gm_rep.covs[0].detach().numpy()
        orig_mu = daily_means[m - 1]
        c = cmap(norm_c(m))
        ax.plot(orig_mu[0], orig_mu[1], "o", ms=7, color=c, zorder=3)
        ax.annotate(f"d{m}", xy=orig_mu[:2], fontsize=7,
                    xytext=(4, 4), textcoords="offset points", color=c)
        ell = cov_ellipse(mu_np, cov_np, n_std=2.0,
                          fill=False, edgecolor=c, lw=1.2, ls="--", zorder=4)
        ax.add_patch(ell)
        ax.plot(mu_np[0], mu_np[1], "x", ms=7, color=c,
                markeredgewidth=1.5, zorder=5)
        ax.annotate("", xy=orig_mu[:2], xytext=mu_np[:2],
                    arrowprops=dict(arrowstyle="->", color=c, lw=0.8, alpha=0.6))
    except Exception as e:
        print(f"  day {m}: {e}")

ax.plot(0, 0, "k*", ms=10, zorder=6)
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
ax.set_title("Replay (×, dashed) vs original (dots)")
ax.set_aspect("equal")
ax.set_xlim(-3.2, 3.2); ax.set_ylim(-3.2, 3.2)
fig.savefig(os.path.join(FIGS, "paper_fig6_confusion_ellipses.pdf"))
fig.savefig(os.path.join(FIGS, "paper_fig6_confusion_ellipses.png"))
plt.close(fig)
print("Saved paper_fig6")

# %%
# ═══════════════════════════════════════════════════════════════════════
#  Summary
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  Exp01 Summary")
print("=" * 60)
print(f"  Setup:  n={N_DAYS}, d={D}, circle R={R_DRIFT}, P={PERIOD}")
print(f"  Prior:  N(0, I)")
print(f"  Protocol: L={L_DEF}")
print(f"  Half-life: a₁/₂ = {hl_def}")
print(f"  Figures in: {FIGS}/")
print("=" * 60)

# %%
