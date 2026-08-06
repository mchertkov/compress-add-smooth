# %% [markdown]
# # Exp06: MNIST Latent-Space Continual Memory
#
# PCA-compress MNIST → fit per-class Gaussians → rotating-dominance
# curriculum → CAS loop → visual forgetting + movie generation.
#
# **Sections**
# - A: Imports, style
# - B: PCA latent space construction
# - C: Daily GM generation (rotating-dominance curriculum)
# - D: CAS loop + forgetting metrics
# - E: Visual forgetting — decoded replay vs originals
# - F: Movie generation (density-level + SDE trajectory)
# - G: Comparison with synthetic K=3
# - H: Dimension sweep (optional)

# %%
# ═══════════════════════════════════════════════════════════════════════
#  A. Imports + style
# ═══════════════════════════════════════════════════════════════════════
import os, sys, math
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if '__file__' in dir() else os.getcwd())
from bridge_cas import (
    GaussianMixture, ContinualMemory, ForgetMetrics,
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
    "image.cmap":         "gray",
})

FIGS = os.path.join(os.getcwd(), "figs")
os.makedirs(FIGS, exist_ok=True)

C_BLUE   = "#2166ac"
C_ORANGE = "#e08214"
C_GREEN  = "#1a9641"
C_RED    = "#d73027"
C_PURPLE = "#7570b3"

print("Imports OK.  torch", torch.__version__)


# ═══════════════════════════════════════════════════════════════════════
# %%
# ═══════════════════════════════════════════════════════════════════════
#  B. PCA latent space construction
# ═══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  B: PCA latent space")
print("=" * 70)

D_PCA = 12          # latent dimension
CLASSES = [0, 3, 8]  # digit classes
K_PER_CLASS = 1      # one Gaussian per class → K=3 total
K_TOTAL = len(CLASSES) * K_PER_CLASS

# ── load MNIST ───────────────────────────────────────────────────────
mnist = datasets.MNIST(root="./data", train=True, download=True,
                       transform=transforms.ToTensor())
images_all = mnist.data.numpy().astype(np.float64) / 255.0  # (60000, 28, 28)
labels_all = mnist.targets.numpy()

# Flatten
N_total, H, W = images_all.shape
images_flat = images_all.reshape(N_total, H * W)  # (60000, 784)
print(f"  MNIST loaded: {N_total} images, {H}×{W} pixels")

# ── filter to selected classes ───────────────────────────────────────
mask_classes = np.isin(labels_all, CLASSES)
images_sel = images_flat[mask_classes]
labels_sel = labels_all[mask_classes]
print(f"  Selected classes {CLASSES}: {len(images_sel)} images")
for c in CLASSES:
    print(f"    class {c}: {(labels_sel == c).sum()} images")

# ── PCA on selected classes ──────────────────────────────────────────
mean_img = images_sel.mean(axis=0)                   # (784,)
X_centered = images_sel - mean_img                   # (N, 784)

# Economy SVD
U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
V_d = Vt[:D_PCA].T                                  # (784, d) PCA basis
explained_var = (S[:D_PCA] ** 2).sum() / (S ** 2).sum()
print(f"  PCA to d={D_PCA}: explained variance = {explained_var:.1%}")

# Project to latent space
Z_sel = X_centered @ V_d                             # (N, d)
print(f"  Latent vectors shape: {Z_sel.shape}")

# ── fit per-class Gaussians ──────────────────────────────────────────
class_stats = {}
for c in CLASSES:
    z_c = Z_sel[labels_sel == c]
    mu_c = z_c.mean(axis=0)
    cov_c = np.cov(z_c, rowvar=False)
    # Regularise covariance for numerical stability
    cov_c = cov_c + 1e-6 * np.eye(D_PCA)
    class_stats[c] = dict(mean=mu_c, cov=cov_c, n=len(z_c))
    print(f"  Class {c}: n={len(z_c)}, "
          f"||μ||={np.linalg.norm(mu_c):.2f}, "
          f"tr(Σ)={np.trace(cov_c):.2f}")

# Inter-class distances → effective crowding ratio
for i, c1 in enumerate(CLASSES):
    for c2 in CLASSES[i+1:]:
        dist = np.linalg.norm(class_stats[c1]['mean'] - class_stats[c2]['mean'])
        avg_std = 0.5 * (np.sqrt(np.trace(class_stats[c1]['cov']) / D_PCA) +
                         np.sqrt(np.trace(class_stats[c2]['cov']) / D_PCA))
        chi = dist / avg_std
        print(f"  Distance {c1}↔{c2}: ||Δμ||={dist:.2f}, "
              f"avg σ={avg_std:.2f}, χ={chi:.2f}")


# ── PCA decode helper ────────────────────────────────────────────────
def pca_decode(z):
    """Decode latent vector(s) z ∈ R^d to pixel space R^784.
    z: (..., d) numpy array → (..., 784) numpy array, clipped to [0,1].
    """
    x = z @ V_d.T + mean_img
    return np.clip(x, 0, 1)


def show_decoded(z, ax, title=""):
    """Decode a single latent vector and display as 28×28 image."""
    img = pca_decode(z).reshape(H, W)
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


# ── visualise class centroids in pixel space ─────────────────────────
fig, axes = plt.subplots(1, len(CLASSES) + 1, figsize=(3 * (len(CLASSES) + 1), 3))
# Mean image
axes[0].imshow(mean_img.reshape(H, W), cmap="gray", vmin=0, vmax=1)
axes[0].set_title("PCA mean", fontsize=10); axes[0].axis("off")
for i, c in enumerate(CLASSES):
    show_decoded(class_stats[c]['mean'], axes[i + 1], f"class {c} centroid")
fig.suptitle(f"PCA centroids (d={D_PCA}, var={explained_var:.0%})", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "mnist_b_centroids.png"))
plt.close(fig)
print("  Saved mnist_b_centroids.png")


# ═══════════════════════════════════════════════════════════════════════
# %%
# ═══════════════════════════════════════════════════════════════════════
#  C. Daily GM generation — rotating-dominance curriculum
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  C: Daily GM generation")
print("=" * 70)

N_DAYS = 100
PERIOD = 30    # rotation period (days)

# Each day: K=3 GM with time-varying weights (rotating dominance)
# and fixed per-class means and covariances.
# Weights: softmax(A * cos(2π m/P + 2πk/3)) with amplitude A


def make_rotating_dominance_gms(n_days, class_stats, classes, period,
                                 amplitude=2.0, d=None):
    """Generate daily GMs with rotating class dominance.

    Returns: list of GaussianMixture, (n_days, K) array of weights
    """
    K = len(classes)
    if d is None:
        d = class_stats[classes[0]]['mean'].shape[0]

    dists = []
    all_weights = []

    # Fixed means and covariances from class fits
    means_np = np.stack([class_stats[c]['mean'] for c in classes])    # (K, d)
    covs_np  = np.stack([class_stats[c]['cov']  for c in classes])    # (K, d, d)
    means_t = torch.tensor(means_np)
    covs_t  = torch.tensor(covs_np)

    for m in range(1, n_days + 1):
        # Rotating weights via softmax
        logits = np.array([amplitude * math.cos(2 * math.pi * m / period
                                                 + 2 * math.pi * k / K)
                           for k in range(K)])
        w = np.exp(logits - logits.max())
        w = w / w.sum()

        gm = GaussianMixture(
            weights=torch.tensor(w),
            means=means_t.clone(),
            covs=covs_t.clone(),
        )
        dists.append(gm)
        all_weights.append(w)

    return dists, np.array(all_weights)


daily_gms, daily_weights = make_rotating_dominance_gms(
    N_DAYS, class_stats, CLASSES, PERIOD, amplitude=2.0, d=D_PCA)

print(f"  Generated {N_DAYS} daily GMs: K={K_TOTAL}, d={D_PCA}, P={PERIOD}")
print(f"  Weight range: min={daily_weights.min():.3f}, max={daily_weights.max():.3f}")

# ── visualise weight evolution ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 3))
for k, c in enumerate(CLASSES):
    ax.plot(range(1, N_DAYS + 1), daily_weights[:, k], "-",
            lw=1.5, label=f"digit {c}")
ax.set_xlabel("Day $m$"); ax.set_ylabel("Weight $\\pi_k$")
ax.set_title("Rotating-dominance curriculum")
ax.legend(fontsize=9)
ax.set_xlim(1, N_DAYS)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "mnist_c_weights.png"))
plt.close(fig)
print("  Saved mnist_c_weights.png")

# ── show representative daily samples in pixel space ─────────────────
fig, axes = plt.subplots(3, 8, figsize=(14, 5.5))
sample_days = [1, 10, 20, 30, 40, 60, 80, 100]
for col, day in enumerate(sample_days):
    gm = daily_gms[day - 1]
    samples = gm.sample(3).numpy()  # (3, d)
    for row in range(3):
        img = pca_decode(samples[row]).reshape(H, W)
        axes[row, col].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[row, col].axis("off")
    axes[0, col].set_title(f"d{day}", fontsize=9)
fig.suptitle("Daily samples (3 per day, decoded from PCA)", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "mnist_c_daily_samples.png"))
plt.close(fig)
print("  Saved mnist_c_daily_samples.png")


# ═══════════════════════════════════════════════════════════════════════
# %%
# ═══════════════════════════════════════════════════════════════════════
#  D. CAS loop + forgetting metrics
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  D: CAS loop")
print("=" * 70)

L_DEF = 10

# Prior: isotropic Gaussian at PCA origin (all classes equally weighted)
gm_prior = GaussianMixture(
    weights=torch.ones(K_TOTAL) / K_TOTAL,
    means=torch.zeros(K_TOTAL, D_PCA),
    covs=torch.eye(D_PCA).unsqueeze(0).expand(K_TOTAL, -1, -1).clone() * 10.0,
    # broad prior — 10× identity covers the PCA range
)

print(f"  Protocol: L={L_DEF}, K={K_TOTAL}, d={D_PCA}")
print()

mem, Fraw, Fnorm, snaps = run_cl_loop(
    daily_gms, gm_prior, L=L_DEF, verbose_every=25)

ages, Fmu, Fsig, cnt, hl = compute_age_curves(Fnorm, N_DAYS)
print(f"\n  MNIST half-life: a₁/₂ = {hl}")

# ── age-forgetting curve ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 3.8))
mask = ~np.isnan(Fmu) & (ages > 0)
ax.plot(ages[mask], Fmu[mask], "o-", ms=3, color=C_BLUE,
        label=r"$\bar{F}(a)$")
ax.fill_between(ages[mask],
    np.clip(Fmu[mask] - Fsig[mask], 0, None),
    Fmu[mask] + Fsig[mask], alpha=0.15, color=C_BLUE)
ax.axhline(0.5, ls="--", color="gray", lw=0.7, label=r"$\theta=0.5$")
ax.axhline(1.0, ls=":", color="gray", lw=0.5, alpha=0.5)
if hl is not None:
    ax.axvline(hl, ls=":", color=C_RED, lw=1,
               label=f"$a_{{1/2}}={hl}$")
ax.set_xlabel("Age $a = n - m$")
ax.set_ylabel(r"$\bar{F}(a)$")
ax.set_title(f"MNIST age–forgetting ($L={L_DEF}$, $d={D_PCA}$, $K={K_TOTAL}$)")
ax.set_ylim(-0.03, 1.5)
ax.set_xlim(0, N_DAYS)
ax.legend(loc="upper left", fontsize=9)
fig.savefig(os.path.join(FIGS, "mnist_d_age_curve.pdf"))
fig.savefig(os.path.join(FIGS, "mnist_d_age_curve.png"))
plt.close(fig)
print("  Saved mnist_d_age_curve")

# ── decomposed forgetting ────────────────────────────────────────────
ages_dec, Fm_dec, Fc_dec, Fw_dec = [], [], [], []
for m in range(1, N_DAYS + 1):
    if m not in mem.readout_times:
        continue
    try:
        dec = mem.forgetting_decomposed(m)
        ages_dec.append(mem.current_day - m)
        Fm_dec.append(dec['F_mean'].item())
        Fc_dec.append(dec['F_cov'].item())
        Fw_dec.append(dec['F_weight'].item())
    except Exception:
        pass

ages_dec = np.array(ages_dec)
Fm_dec = np.array(Fm_dec)
Fc_dec = np.array(Fc_dec)
Fw_dec = np.array(Fw_dec)

# Age-average
max_age = N_DAYS - 1
Fm_age = np.full(max_age + 1, np.nan)
Fc_age = np.full(max_age + 1, np.nan)
Fw_age = np.full(max_age + 1, np.nan)
for a in range(max_age + 1):
    m = ages_dec == a
    if m.sum() > 0:
        Fm_age[a] = np.mean(Fm_dec[m])
        Fc_age[a] = np.mean(Fc_dec[m])
        Fw_age[a] = np.mean(Fw_dec[m])

fig, ax = plt.subplots(figsize=(6, 3.5))
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
ax.set_title("MNIST forgetting decomposition")
ax.legend(fontsize=9, loc="upper left")
ax.set_xlim(1, N_DAYS - 1)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "mnist_d_decomposed.png"))
plt.close(fig)
print("  Saved mnist_d_decomposed")


# ═══════════════════════════════════════════════════════════════════════
# %%
# ═══════════════════════════════════════════════════════════════════════
#  E. Visual forgetting — decoded replay vs originals
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  E: Visual forgetting")
print("=" * 70)

# For selected past days, decode both:
#   - the original daily GM (sample from it)
#   - the replayed GM (evaluate protocol at readout time, sample from it)
# Display side by side.

SHOW_DAYS = [5, 15, 25, 35, 50, 70, 90, 99]
N_SAMPLES_PER = 5

fig, axes = plt.subplots(2 * N_SAMPLES_PER, len(SHOW_DAYS),
                         figsize=(2 * len(SHOW_DAYS), 2 * 2 * N_SAMPLES_PER))

for col, day in enumerate(SHOW_DAYS):
    gm_orig = mem.history[day - 1]

    # Original samples
    orig_samples = gm_orig.sample(N_SAMPLES_PER).numpy()
    for row in range(N_SAMPLES_PER):
        img = pca_decode(orig_samples[row]).reshape(H, W)
        axes[row, col].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[row, col].axis("off")
        if row == 0:
            axes[row, col].set_title(f"d{day} orig", fontsize=8)

    # Replay samples (if available)
    if day in mem.readout_times:
        try:
            gm_rep = mem.replay(day)
            rep_samples = gm_rep.sample(N_SAMPLES_PER).detach().numpy()
            for row in range(N_SAMPLES_PER):
                img = pca_decode(rep_samples[row]).reshape(H, W)
                axes[N_SAMPLES_PER + row, col].imshow(img, cmap="gray",
                                                       vmin=0, vmax=1)
                axes[N_SAMPLES_PER + row, col].axis("off")
                if row == 0:
                    fnorm = Fnorm[day - 1, N_DAYS - 1]
                    fstr = f"{fnorm:.2f}" if not np.isnan(fnorm) else "?"
                    axes[N_SAMPLES_PER + row, col].set_title(
                        f"replay F̄={fstr}", fontsize=7)
        except Exception as e:
            print(f"    day {day}: {e}")
            for row in range(N_SAMPLES_PER):
                axes[N_SAMPLES_PER + row, col].axis("off")
    else:
        for row in range(N_SAMPLES_PER):
            axes[N_SAMPLES_PER + row, col].axis("off")

fig.suptitle("Original (top rows) vs Replay (bottom rows)", fontsize=12, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "mnist_e_visual_forgetting.png"), dpi=200)
plt.close(fig)
print("  Saved mnist_e_visual_forgetting.png")

# ── centroid comparison: decoded replay means vs originals ───────────
fig, axes = plt.subplots(2, len(SHOW_DAYS), figsize=(2 * len(SHOW_DAYS), 4.5))
for col, day in enumerate(SHOW_DAYS):
    # Original: decode overall mean
    gm_orig = mem.history[day - 1]
    mu_orig = gm_orig.overall_mean().numpy()
    img_orig = pca_decode(mu_orig).reshape(H, W)
    axes[0, col].imshow(img_orig, cmap="gray", vmin=0, vmax=1)
    axes[0, col].set_title(f"d{day} orig", fontsize=8)
    axes[0, col].axis("off")

    # Replay: decode overall mean
    if day in mem.readout_times:
        try:
            gm_rep = mem.replay(day)
            mu_rep = gm_rep.overall_mean().detach().numpy()
            img_rep = pca_decode(mu_rep).reshape(H, W)
            axes[1, col].imshow(img_rep, cmap="gray", vmin=0, vmax=1)
            fnorm = Fnorm[day - 1, N_DAYS - 1]
            fstr = f"{fnorm:.2f}" if not np.isnan(fnorm) else "?"
            axes[1, col].set_title(f"replay F̄={fstr}", fontsize=7)
        except Exception:
            pass
    axes[1, col].axis("off")

fig.suptitle("Decoded means: original (top) vs replay (bottom)", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "mnist_e_centroid_comparison.png"), dpi=200)
plt.close(fig)
print("  Saved mnist_e_centroid_comparison.png")

# ── per-component replay: show what each component looks like ────────
fig, axes = plt.subplots(K_TOTAL, len(SHOW_DAYS),
                         figsize=(2 * len(SHOW_DAYS), 2 * K_TOTAL))
for col, day in enumerate(SHOW_DAYS):
    if day not in mem.readout_times:
        for k in range(K_TOTAL):
            axes[k, col].axis("off")
        continue
    try:
        gm_rep = mem.replay(day)
        for k in range(K_TOTAL):
            mu_k = gm_rep.means[k].detach().numpy()
            img = pca_decode(mu_k).reshape(H, W)
            axes[k, col].imshow(img, cmap="gray", vmin=0, vmax=1)
            axes[k, col].axis("off")
            if col == 0:
                axes[k, col].set_ylabel(f"comp {k}\n(digit {CLASSES[k]})",
                                         fontsize=8, rotation=0, labelpad=40)
        axes[0, col].set_title(f"d{day}", fontsize=8)
    except Exception:
        for k in range(K_TOTAL):
            axes[k, col].axis("off")

fig.suptitle("Per-component replay means (decoded)", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "mnist_e_per_component.png"), dpi=200)
plt.close(fig)
print("  Saved mnist_e_per_component.png")


# ═══════════════════════════════════════════════════════════════════════
# %%
# ═══════════════════════════════════════════════════════════════════════
#  F. Movie generation
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  F: Movie generation")
print("=" * 70)

# ── F1: Density-level movie (sample from marginals at each frame) ────
N_FRAMES = 50
frame_times = np.linspace(0, 1, N_FRAMES)

# For each frame time, evaluate the protocol interpolant and sample
fig, axes = plt.subplots(4, N_FRAMES, figsize=(N_FRAMES * 0.7, 4 * 0.9))

for col, t in enumerate(frame_times):
    gm_t = mem.protocol.evaluate_at(t)
    samples = gm_t.sample(4).numpy()
    for row in range(4):
        img = pca_decode(samples[row]).reshape(H, W)
        axes[row, col].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[row, col].axis("off")
    if col % 5 == 0 or col == N_FRAMES - 1:
        axes[0, col].set_title(f"t={t:.2f}", fontsize=5)

fig.suptitle("Density-level movie: samples from $p_t$ at each frame time",
             fontsize=11, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "mnist_f_density_movie.png"), dpi=150)
plt.close(fig)
print("  Saved mnist_f_density_movie.png")

# ── F2: Mean-level movie (decoded overall mean at each frame) ────────
fig, axes = plt.subplots(1, N_FRAMES, figsize=(N_FRAMES * 0.7, 1.2))

for col, t in enumerate(frame_times):
    gm_t = mem.protocol.evaluate_at(t)
    mu_t = gm_t.overall_mean().detach().numpy()
    img = pca_decode(mu_t).reshape(H, W)
    axes[col].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes[col].axis("off")
    if col % 10 == 0 or col == N_FRAMES - 1:
        axes[col].set_title(f"{t:.1f}", fontsize=5)

fig.suptitle("Mean movie: decoded $\\mu(t)$ from $t=0$ (oldest) to $t=1$ (now)",
             fontsize=10, y=1.08)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "mnist_f_mean_movie.png"), dpi=200)
plt.close(fig)
print("  Saved mnist_f_mean_movie.png")

# ── F3: Per-component movie (each component's mean decoded) ──────────
fig, axes = plt.subplots(K_TOTAL, N_FRAMES,
                         figsize=(N_FRAMES * 0.7, K_TOTAL * 0.9))

for col, t in enumerate(frame_times):
    gm_t = mem.protocol.evaluate_at(t)
    for k in range(K_TOTAL):
        mu_k = gm_t.means[k].detach().numpy()
        img = pca_decode(mu_k).reshape(H, W)
        axes[k, col].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[k, col].axis("off")
    if col % 10 == 0 or col == N_FRAMES - 1:
        axes[0, col].set_title(f"{t:.1f}", fontsize=5)

for k in range(K_TOTAL):
    axes[k, 0].set_ylabel(f"d{CLASSES[k]}", fontsize=7, rotation=0, labelpad=15)

fig.suptitle("Component movie: per-class means from $t=0$ to $t=1$",
             fontsize=10, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "mnist_f_component_movie.png"), dpi=150)
plt.close(fig)
print("  Saved mnist_f_component_movie.png")

# ── F4: Weight evolution across the protocol ─────────────────────────
fig, ax = plt.subplots(figsize=(6, 3))
w_trajectory = []
for t in frame_times:
    gm_t = mem.protocol.evaluate_at(t)
    w_trajectory.append(gm_t.weights.detach().numpy())
w_trajectory = np.array(w_trajectory)  # (N_FRAMES, K)

for k, c in enumerate(CLASSES):
    ax.plot(frame_times, w_trajectory[:, k], "-", lw=1.5, label=f"digit {c}")
ax.set_xlabel("Protocol time $t$")
ax.set_ylabel("Weight $\\pi_k(t)$")
ax.set_title("Component weights across protocol")
ax.legend(fontsize=9)
ax.set_xlim(0, 1)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "mnist_f_protocol_weights.png"))
plt.close(fig)
print("  Saved mnist_f_protocol_weights.png")


# ═══════════════════════════════════════════════════════════════════════
# %%
# ═══════════════════════════════════════════════════════════════════════
#  G. Comparison with synthetic K=3
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  G: Comparison with synthetic K=3")
print("=" * 70)

# Run synthetic K=3 with same L=10 for fair comparison
from bridge_cas import make_daily_gmm_circle

daily_synth, _ = make_daily_gmm_circle(
    N_DAYS, K=3, R=2.0, r=0.8, period=PERIOD, cov_scale=0.3, d=2)

gm_prior_synth = GaussianMixture(
    weights=torch.ones(3) / 3,
    means=torch.zeros(3, 2),
    covs=torch.eye(2).unsqueeze(0).expand(3, -1, -1).clone(),
)

print("  Synthetic K=3 run...", end=" ", flush=True)
_, _, Fnorm_synth, _ = run_cl_loop(
    daily_synth, gm_prior_synth, L=L_DEF, verbose_every=0)
ages_s, Fmu_s, Fsig_s, _, hl_s = compute_age_curves(Fnorm_synth, N_DAYS)
print(f"a₁/₂ = {hl_s}")

# Comparison figure
fig, ax = plt.subplots(figsize=(5.5, 3.8))
ms = ~np.isnan(Fmu_s) & (ages_s > 0)
mm = ~np.isnan(Fmu) & (ages > 0)
ax.plot(ages_s[ms], Fmu_s[ms], "o-", ms=2.5, color=C_BLUE,
        label=f"synthetic ($a_{{1/2}}={hl_s}$)")
ax.plot(ages[mm], Fmu[mm], "s-", ms=2.5, color=C_RED,
        label=f"MNIST ($a_{{1/2}}={hl}$)")
ax.axhline(0.5, ls="--", color="gray", lw=0.6)
ax.axhline(1.0, ls=":", color="gray", lw=0.4, alpha=0.5)
ax.set_xlabel("Age $a$"); ax.set_ylabel(r"$\bar{F}(a)$")
ax.set_title(f"Synthetic vs MNIST ($L={L_DEF}$, $K=3$)")
ax.legend(fontsize=9, loc="upper left")
ax.set_ylim(-0.03, 1.5); ax.set_xlim(0, N_DAYS)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "mnist_g_synth_vs_mnist.pdf"))
fig.savefig(os.path.join(FIGS, "mnist_g_synth_vs_mnist.png"))
plt.close(fig)
print("  Saved mnist_g_synth_vs_mnist")


# ═══════════════════════════════════════════════════════════════════════
# %%
# ═══════════════════════════════════════════════════════════════════════
#  H. Dimension sweep (optional)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  H: PCA dimension sweep")
print("=" * 70)

D_SWEEP = [4, 8, 12, 20, 30]
res_d_sweep = []

for d_pca in D_SWEEP:
    print(f"  d={d_pca} ...", end=" ", flush=True)

    # Re-do PCA at this dimension
    V_d_sweep = Vt[:d_pca].T
    Z_sweep = X_centered @ V_d_sweep

    # Fit per-class Gaussians
    cs = {}
    for c in CLASSES:
        z_c = Z_sweep[labels_sel == c]
        mu_c = z_c.mean(axis=0)
        cov_c = np.cov(z_c, rowvar=False) + 1e-6 * np.eye(d_pca)
        cs[c] = dict(mean=mu_c, cov=cov_c)

    # Generate daily GMs at this d
    dd, _ = make_rotating_dominance_gms(
        N_DAYS, cs, CLASSES, PERIOD, amplitude=2.0, d=d_pca)

    prior_d = GaussianMixture(
        weights=torch.ones(K_TOTAL) / K_TOTAL,
        means=torch.zeros(K_TOTAL, d_pca),
        covs=torch.eye(d_pca).unsqueeze(0).expand(K_TOTAL, -1, -1).clone() * 10.0,
    )

    _, _, Fn_d, _ = run_cl_loop(dd, prior_d, L=L_DEF, verbose_every=0)
    a_d, fm_d, fs_d, _, hl_d = compute_age_curves(Fn_d, N_DAYS)
    res_d_sweep.append((d_pca, a_d, fm_d, fs_d, hl_d))
    print(f"a₁/₂={hl_d}")

# Figure
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

SWEEP_COLORS = [C_BLUE, C_ORANGE, C_GREEN, C_RED, C_PURPLE, "#636363"]

ax = axes[0]
for i, (d_pca, a, fm, fs, hl_d) in enumerate(res_d_sweep):
    m = ~np.isnan(fm) & (a > 0)
    lbl = f"$d={d_pca}$"
    if hl_d is not None:
        lbl += f" ($a_{{1/2}}={hl_d}$)"
    ax.plot(a[m], fm[m], "o-", ms=2, color=SWEEP_COLORS[i], label=lbl)
ax.axhline(0.5, ls="--", color="gray", lw=0.5)
ax.axhline(1.0, ls=":", color="gray", lw=0.3, alpha=0.5)
ax.set_xlabel("Age $a$"); ax.set_ylabel(r"$\bar{F}(a)$")
ax.set_title("(a) MNIST age curves vs PCA dimension")
ax.legend(fontsize=7, loc="upper left")
ax.set_ylim(-0.03, 1.5); ax.set_xlim(0, N_DAYS)

ax = axes[1]
d_plot = [dp for dp, _, _, _, _ in res_d_sweep]
hl_plot = [h if h is not None else N_DAYS for _, _, _, _, h in res_d_sweep]
ax.plot(d_plot, hl_plot, "o-", ms=5, color=C_RED)
ax.set_xlabel("PCA dimension $d$")
ax.set_ylabel(r"$a_{1/2}$")
ax.set_title("(b) Half-life vs PCA dimension")
ax.set_xticks(d_plot)

fig.tight_layout()
fig.savefig(os.path.join(FIGS, "mnist_h_dimension_sweep.pdf"))
fig.savefig(os.path.join(FIGS, "mnist_h_dimension_sweep.png"))
plt.close(fig)
print("  Saved mnist_h_dimension_sweep")


# ═══════════════════════════════════════════════════════════════════════
# %%
# ═══════════════════════════════════════════════════════════════════════
#  Summary
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  Exp06 Summary")
print("=" * 70)
print(f"  PCA: d={D_PCA}, explained variance={explained_var:.1%}")
print(f"  Classes: {CLASSES}, K={K_TOTAL}")
print(f"  Curriculum: rotating dominance, period={PERIOD}")
print(f"  Protocol: L={L_DEF}")
print(f"\n  MNIST half-life: a₁/₂ = {hl}")
print(f"  Synthetic K=3 half-life: a₁/₂ = {hl_s}")

print(f"\n  PCA dimension sweep:")
for d_pca, _, _, _, hl_d in res_d_sweep:
    print(f"    d={d_pca:2d}  →  a₁/₂ = {hl_d}")

print(f"\n  Figures in: {FIGS}/")
print(f"    mnist_b_centroids          — PCA class centroids")
print(f"    mnist_c_weights            — rotating-dominance weights")
print(f"    mnist_c_daily_samples      — decoded daily samples")
print(f"    mnist_d_age_curve          — age-forgetting curve")
print(f"    mnist_d_decomposed         — forgetting decomposition")
print(f"    mnist_e_visual_forgetting  — orig vs replay samples")
print(f"    mnist_e_centroid_comparison — decoded means comparison")
print(f"    mnist_e_per_component      — per-component replay means")
print(f"    mnist_f_density_movie      — density-level movie frames")
print(f"    mnist_f_mean_movie         — mean-decoded movie strip")
print(f"    mnist_f_component_movie    — per-component movie strip")
print(f"    mnist_f_protocol_weights   — weight evolution in protocol")
print(f"    mnist_g_synth_vs_mnist     — synthetic vs MNIST comparison")
print(f"    mnist_h_dimension_sweep    — half-life vs PCA d")
print("=" * 70)

# %%
