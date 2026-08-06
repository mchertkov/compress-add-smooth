# %% [markdown]
# # Exp08: MNIST Latent Replay Classification / NLL
#
# Task-level proxy for the CAS temporal-memory paper.
#
# We reuse the MNIST PCA latent-space setup from `exp06_mnist.ipynb` and compare CAS with the memory-matched baselines from `exp07_baselines-corr.ipynb`.
#
# **Question.** If a past day is replayed as a Gaussian-mixture density, how well does the replayed density classify held-out latent MNIST samples from that day?
#
# **Metrics.** For each method and each recalled day, we evaluate:
# - held-out latent classification accuracy using the posterior class probabilities induced by the replayed GM;
# - posterior negative log likelihood (NLL) of the true digit class;
# - density NLL under the replayed GM.
#
# All methods use the same state budget: CAS with $L=10$ stores $L+1=11$ GM states, and every baseline stores $B=11$ GM states.

# %%
# ═══════════════════════════════════════════════════════════════════════
#  A. Imports + style
# ═══════════════════════════════════════════════════════════════════════
import os, sys, math, json
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

from bridge_cas_corr import GaussianMixture, run_cl_loop

plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          11,
    "axes.labelsize":     12,
    "axes.titlesize":     13,
    "legend.fontsize":    9,
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
C_GRAY   = "#636363"
COLORS = {
    "CAS": C_BLUE,
    "FIFO": C_ORANGE,
    "Reservoir": C_GREEN,
    "Log-age": C_PURPLE,
    "Greedy-PL": C_RED,
    "Original": C_GRAY,
}

# ------------------------------------------------------------------
# Self-contained memory-matched baseline utilities.
# These are copied from exp07_baselines-corr / run_days4_6_baselines.py
# so this notebook does not depend on that script being in the working
# directory.
# ------------------------------------------------------------------

def gm_to_state(gm):
    return dict(
        w=gm.weights.detach().cpu().numpy().copy(),
        m=gm.means.detach().cpu().numpy().copy(),
        c=gm.covs.detach().cpu().numpy().copy(),
    )


def clone_state(s):
    return dict(w=s['w'].copy(), m=s['m'].copy(), c=s['c'].copy())


def interp_state(sa, sb, alpha):
    a = float(np.clip(alpha, 0.0, 1.0))
    w = (1.0 - a) * sa['w'] + a * sb['w']
    w = w / np.sum(w)
    return dict(
        w=w,
        m=(1.0 - a) * sa['m'] + a * sb['m'],
        c=(1.0 - a) * sa['c'] + a * sb['c'],
    )


def state_moments(s):
    w, m, c = s['w'], s['m'], s['c']
    mu = np.einsum('k,kd->d', w, m)
    within = np.einsum('k,kij->ij', w, c)
    dm = m - mu[None, :]
    between = np.einsum('k,ki,kj->ij', w, dm, dm)
    return mu, within + between


def raw_mismatch_states(sa, sb):
    mua, ca = state_moments(sa)
    mub, cb = state_moments(sb)
    return float(np.sum((mua - mub) ** 2) + np.sum((ca - cb) ** 2))


def replay_from_knots(knots, day):
    if not knots:
        raise RuntimeError('empty knot set')
    ks = sorted(knots, key=lambda z: z[0])
    if day <= ks[0][0]:
        return clone_state(ks[0][1])
    if day >= ks[-1][0]:
        return clone_state(ks[-1][1])
    for i, (di, si) in enumerate(ks):
        if day == di:
            return clone_state(si)
        if di > day:
            d0, s0 = ks[i - 1]
            d1, s1 = ks[i]
            alpha = (float(day) - float(d0)) / (float(d1) - float(d0))
            return interp_state(s0, s1, alpha)
    return clone_state(ks[-1][1])


class FIFOBuffer:
    name = 'FIFO'
    def __init__(self, B):
        self.B, self.items = B, []
    def update(self, day, s):
        self.items.append((int(day), clone_state(s)))
        if len(self.items) > self.B:
            self.items.pop(0)
    def replay(self, day):
        return replay_from_knots(self.items, day)
    def n_states(self):
        return len(self.items)


class ReservoirBuffer:
    name = 'Reservoir'
    def __init__(self, B, seed=0):
        self.B, self.rng = B, np.random.default_rng(seed)
        self.current, self.past, self.n_past_seen = None, [], 0
    def _add_to_reservoir(self, item):
        self.n_past_seen += 1
        cap = max(self.B - 1, 0)
        if cap <= 0:
            return
        if len(self.past) < cap:
            self.past.append((int(item[0]), clone_state(item[1])))
        else:
            j = self.rng.integers(1, self.n_past_seen + 1)
            if j <= cap:
                self.past[j - 1] = (int(item[0]), clone_state(item[1]))
    def update(self, day, s):
        if self.current is not None:
            self._add_to_reservoir(self.current)
        self.current = (int(day), clone_state(s))
    def knots(self):
        return self.past + ([self.current] if self.current is not None else [])
    def replay(self, day):
        return replay_from_knots(self.knots(), day)
    def n_states(self):
        return len(self.knots())


class LogAgeBuffer:
    name = 'Log-age'
    def __init__(self, B):
        self.B, self.current, self.past, self.day = B, None, [], 0
    def update(self, day, s):
        self.day = int(day)
        if self.current is not None:
            self.past.append((int(self.current[0]), clone_state(self.current[1])))
        self.current = (int(day), clone_state(s))
        cap = max(self.B - 1, 0)
        while len(self.past) > cap:
            self._drop_one_past()
    def _drop_one_past(self):
        items = sorted(self.past, key=lambda z: z[0])
        if len(items) <= 2:
            items.pop(0)
            self.past = items
            return
        ages = np.array([max(self.day - d, 0) for d, _ in items], dtype=float)
        u = np.log1p(ages)
        best_i, best_score = 1, float('inf')
        for i in range(1, len(items) - 1):
            score = abs(u[i-1] - u[i]) + abs(u[i] - u[i+1])
            if score < best_score:
                best_i, best_score = i, score
        del items[best_i]
        self.past = items
    def knots(self):
        return self.past + ([self.current] if self.current is not None else [])
    def replay(self, day):
        return replay_from_knots(self.knots(), day)
    def n_states(self):
        return len(self.knots())


class GreedyPLBuffer:
    name = 'Greedy-PL'
    def __init__(self, B):
        self.B, self.items = B, []
    def update(self, day, s):
        self.items.append((int(day), clone_state(s)))
        self.items = sorted(self.items, key=lambda z: z[0])
        while len(self.items) > self.B:
            self._drop_min_local_error()
    def _drop_min_local_error(self):
        if len(self.items) <= 2:
            self.items.pop(0)
            return
        best_i, best_err = 1, float('inf')
        for i in range(1, len(self.items) - 1):
            day_i, s_i = self.items[i]
            d0, s0 = self.items[i - 1]
            d1, s1 = self.items[i + 1]
            pred = interp_state(s0, s1, (day_i - d0) / (d1 - d0))
            err = raw_mismatch_states(s_i, pred)
            if err < best_err:
                best_i, best_err = i, err
        del self.items[best_i]
    def replay(self, day):
        return replay_from_knots(self.items, day)
    def n_states(self):
        return len(self.items)

print("Imports OK. torch", torch.__version__)



# %%
# ═══════════════════════════════════════════════════════════════════════
#  B. PCA latent-space construction from MNIST train/test splits
# ═══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  B: MNIST PCA latent space")
print("=" * 70)

D_PCA = 12
CLASSES = [0, 3, 8]
K_TOTAL = len(CLASSES)

# The notebook assumes that MNIST can be downloaded or is already present
# under ./data.  In a no-internet environment, run exp06_mnist.ipynb once
# locally or place the standard MNIST files under ./data/MNIST/raw.
train_ds = datasets.MNIST(root="./data", train=True, download=True,
                          transform=transforms.ToTensor())
test_ds = datasets.MNIST(root="./data", train=False, download=True,
                         transform=transforms.ToTensor())

Xtr_all = train_ds.data.numpy().astype(np.float64) / 255.0
Ytr_all = train_ds.targets.numpy()
Xte_all = test_ds.data.numpy().astype(np.float64) / 255.0
Yte_all = test_ds.targets.numpy()

Ntr, H, W = Xtr_all.shape
Nte = Xte_all.shape[0]
Xtr_all = Xtr_all.reshape(Ntr, H * W)
Xte_all = Xte_all.reshape(Nte, H * W)

mask_tr = np.isin(Ytr_all, CLASSES)
mask_te = np.isin(Yte_all, CLASSES)
Xtr = Xtr_all[mask_tr]
Ytr = Ytr_all[mask_tr]
Xte = Xte_all[mask_te]
Yte = Yte_all[mask_te]

print(f"  train selected: {len(Xtr)} images; test selected: {len(Xte)} images")
for c in CLASSES:
    print(f"    digit {c}: train={(Ytr == c).sum()}, test={(Yte == c).sum()}")

# PCA on selected training classes
mean_img = Xtr.mean(axis=0)
Xc = Xtr - mean_img
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
V_d = Vt[:D_PCA].T
explained = (S[:D_PCA] ** 2).sum() / (S ** 2).sum()
print(f"  PCA dimension d={D_PCA}; explained variance={explained:.1%}")

Ztr = (Xtr - mean_img) @ V_d
Zte = (Xte - mean_img) @ V_d

# Per-class Gaussian fits from training split, test pools for evaluation
class_stats = {}
test_pools = {}
for kk, c in enumerate(CLASSES):
    zc = Ztr[Ytr == c]
    mu = zc.mean(axis=0)
    cov = np.cov(zc, rowvar=False) + 1e-6 * np.eye(D_PCA)
    class_stats[c] = dict(mean=mu, cov=cov, n=len(zc), index=kk)
    test_pools[kk] = Zte[Yte == c]
    print(f"  digit {c}: ||mu||={np.linalg.norm(mu):.2f}, tr(cov)={np.trace(cov):.2f}, test_pool={len(test_pools[kk])}")

means_np = np.stack([class_stats[c]['mean'] for c in CLASSES])
covs_np = np.stack([class_stats[c]['cov'] for c in CLASSES])
means_t = torch.tensor(means_np)
covs_t = torch.tensor(covs_np)

# Decode helper, useful for quick sanity plots if desired.
def pca_decode(z):
    x = z @ V_d.T + mean_img
    return np.clip(x, 0, 1)


# %%
# ═══════════════════════════════════════════════════════════════════════
#  C. Daily GM generation: rotating dominance curriculum
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  C: Daily GM generation")
print("=" * 70)

N_DAYS = 100
PERIOD = 30
AMPLITUDE = 2.0
L_DEF = 10
B_STATES = L_DEF + 1


def rotating_weights(day, K=K_TOTAL, period=PERIOD, amplitude=AMPLITUDE):
    logits = np.array([
        amplitude * math.cos(2 * math.pi * day / period + 2 * math.pi * k / K)
        for k in range(K)
    ])
    w = np.exp(logits - logits.max())
    return w / w.sum()


def make_daily_mnist_gms(n_days=N_DAYS):
    dists, weights = [], []
    for m in range(1, n_days + 1):
        w = rotating_weights(m)
        gm = GaussianMixture(
            weights=torch.tensor(w),
            means=means_t.clone(),
            covs=covs_t.clone(),
        )
        dists.append(gm)
        weights.append(w)
    return dists, np.array(weights)


daily_gms, daily_weights = make_daily_mnist_gms(N_DAYS)

# Prior matches exp06_mnist.py: broad isotropic Gaussian at PCA origin.
gm_prior = GaussianMixture(
    weights=torch.ones(K_TOTAL) / K_TOTAL,
    means=torch.zeros(K_TOTAL, D_PCA),
    covs=10.0 * torch.eye(D_PCA).unsqueeze(0).expand(K_TOTAL, -1, -1).clone(),
)

print(f"  Generated {N_DAYS} daily GMs: K={K_TOTAL}, d={D_PCA}, period={PERIOD}")
print(f"  Memory budget: CAS L={L_DEF} -> B={B_STATES} stored GM states for all methods")
print(f"  Weight range: {daily_weights.min():.3f} to {daily_weights.max():.3f}")


# %%
# ═══════════════════════════════════════════════════════════════════════
#  D. Run CAS and memory-matched baselines to the final day
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  D: Run memories")
print("=" * 70)

# CAS memory
mem_cas, _, _, _ = run_cl_loop(daily_gms, gm_prior, L=L_DEF, verbose_every=25)

# Baseline states
states = [gm_to_state(gm) for gm in daily_gms]


def state_to_gm(s):
    return GaussianMixture(
        weights=torch.tensor(s['w']),
        means=torch.tensor(s['m']),
        covs=torch.tensor(s['c']),
    )


def fit_baseline(baseline):
    for day, state in enumerate(states, start=1):
        baseline.update(day, state)
    return baseline

baseline_objects = {
    "FIFO": fit_baseline(FIFOBuffer(B_STATES)),
    "Log-age": fit_baseline(LogAgeBuffer(B_STATES)),
    "Greedy-PL": fit_baseline(GreedyPLBuffer(B_STATES)),
}
# Reservoir will be averaged over seeds during evaluation.

print("  CAS and deterministic baselines fitted to final day.")
for name, b in baseline_objects.items():
    print(f"  {name}: stored states={b.n_states()}")


# %%
# ═══════════════════════════════════════════════════════════════════════
#  E. Held-out latent classification / NLL evaluation helpers
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  E: Evaluation helpers")
print("=" * 70)

N_EVAL_PER_DAY = 300
RNG_SEED = 123
rng = np.random.default_rng(RNG_SEED)


def sample_heldout_day(day, n=N_EVAL_PER_DAY, rng=rng):
    """Sample held-out latent examples from the empirical MNIST test pools
    according to the original day-specific class weights.

    Returns X: (n,d), y: integer class index in {0,1,2}.
    """
    w = daily_weights[day - 1]
    y = rng.choice(K_TOTAL, size=n, p=w)
    X = np.zeros((n, D_PCA), dtype=np.float64)
    for k in range(K_TOTAL):
        idx = np.where(y == k)[0]
        if len(idx) == 0:
            continue
        pool = test_pools[k]
        pick = rng.integers(0, len(pool), size=len(idx))
        X[idx] = pool[pick]
    return X, y

# Fixed evaluation samples per day for all methods.
eval_sets = {day: sample_heldout_day(day) for day in range(1, N_DAYS + 1)}


def gm_component_logpdf_np(gm, X):
    """Return log N_k(X) for all components: shape (n,K)."""
    w = gm.weights.detach().cpu().numpy()
    means = gm.means.detach().cpu().numpy()
    covs = gm.covs.detach().cpu().numpy()
    n, d = X.shape
    out = np.empty((n, len(w)), dtype=np.float64)
    for k in range(len(w)):
        L = np.linalg.cholesky(covs[k])
        diff = X - means[k][None, :]
        sol = np.linalg.solve(L, diff.T).T
        maha = np.sum(sol * sol, axis=1)
        logdet = 2.0 * np.sum(np.log(np.diag(L)))
        out[:, k] = -0.5 * (d * np.log(2 * np.pi) + logdet + maha)
    return out


def logsumexp_np(A, axis=1):
    m = np.max(A, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(A - m), axis=axis, keepdims=True))).squeeze(axis)


def replay_task_metrics(gm, X, y):
    """Classification and NLL metrics induced by replayed GM.

    The class posterior is p(k|x) proportional to pi_k N_k(x).
    Returns accuracy, posterior NLL, density NLL.
    """
    w = gm.weights.detach().cpu().numpy()
    log_comp = gm_component_logpdf_np(gm, X)
    log_joint = log_comp + np.log(w[None, :] + 1e-300)
    log_mix = logsumexp_np(log_joint, axis=1)
    log_post = log_joint - log_mix[:, None]
    pred = np.argmax(log_post, axis=1)
    acc = float(np.mean(pred == y))
    post_nll = float(-np.mean(log_post[np.arange(len(y)), y]))
    dens_nll = float(-np.mean(log_mix))
    return dict(acc=acc, post_nll=post_nll, dens_nll=dens_nll)


def cas_replay(day):
    return mem_cas.replay(day)


def baseline_replay(baseline, day):
    return state_to_gm(baseline.replay(day))

print(f"  Fixed held-out evaluation sets: {N_DAYS} days × {N_EVAL_PER_DAY} samples/day")


# %%
# ═══════════════════════════════════════════════════════════════════════
#  F. Evaluate CAS, baselines, and original daily GM oracle
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  F: Evaluate methods")
print("=" * 70)

METHODS = ["Original", "CAS", "FIFO", "Reservoir", "Log-age", "Greedy-PL"]


def evaluate_method(get_gm):
    ages, acc, post_nll, dens_nll = [], [], [], []
    for day in range(1, N_DAYS + 1):
        X, y = eval_sets[day]
        gm = get_gm(day)
        met = replay_task_metrics(gm, X, y)
        ages.append(N_DAYS - day)
        acc.append(met['acc'])
        post_nll.append(met['post_nll'])
        dens_nll.append(met['dens_nll'])
    order = np.argsort(ages)
    return dict(
        ages=np.array(ages)[order],
        acc=np.array(acc)[order],
        post_nll=np.array(post_nll)[order],
        dens_nll=np.array(dens_nll)[order],
    )

results = {}
results["Original"] = evaluate_method(lambda day: daily_gms[day - 1])
results["CAS"] = evaluate_method(cas_replay)
results["FIFO"] = evaluate_method(lambda day: baseline_replay(baseline_objects["FIFO"], day))
results["Log-age"] = evaluate_method(lambda day: baseline_replay(baseline_objects["Log-age"], day))
results["Greedy-PL"] = evaluate_method(lambda day: baseline_replay(baseline_objects["Greedy-PL"], day))

# Reservoir: average metrics over several seeds.  We keep the current state
# and B-1 uniformly sampled past states, matching exp07.
reservoir_runs = []
for seed in range(5):
    b = fit_baseline(ReservoirBuffer(B_STATES, seed=seed))
    reservoir_runs.append(evaluate_method(lambda day, bb=b: baseline_replay(bb, day)))

ages = reservoir_runs[0]['ages']
results["Reservoir"] = dict(
    ages=ages,
    acc=np.mean([r['acc'] for r in reservoir_runs], axis=0),
    acc_std=np.std([r['acc'] for r in reservoir_runs], axis=0),
    post_nll=np.mean([r['post_nll'] for r in reservoir_runs], axis=0),
    post_nll_std=np.std([r['post_nll'] for r in reservoir_runs], axis=0),
    dens_nll=np.mean([r['dens_nll'] for r in reservoir_runs], axis=0),
    dens_nll_std=np.std([r['dens_nll'] for r in reservoir_runs], axis=0),
)

# Compact scalar summary: all ages and old memories only.
OLD_AGE = 30
summary = {}
for name in METHODS:
    r = results[name]
    old = r['ages'] >= OLD_AGE
    summary[name] = {
        'states': B_STATES if name != 'Original' else 0,
        'acc_all': float(np.mean(r['acc'])),
        'acc_old': float(np.mean(r['acc'][old])),
        'post_nll_all': float(np.mean(r['post_nll'])),
        'post_nll_old': float(np.mean(r['post_nll'][old])),
        'dens_nll_all': float(np.mean(r['dens_nll'])),
        'dens_nll_old': float(np.mean(r['dens_nll'][old])),
    }

print(json.dumps(summary, indent=2))
with open(os.path.join(FIGS, 'days6_7_mnist_replay_task_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print("  Saved", os.path.join(FIGS, 'days6_7_mnist_replay_task_summary.json'))


# %%
# ═══════════════════════════════════════════════════════════════════════
#  G. Manuscript figures
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  G: Figures")
print("=" * 70)

# Accuracy and posterior NLL as functions of age.
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

ax = axes[0]
for name in METHODS:
    if name == "Original":
        ls, marker, alpha = '--', None, 0.85
    else:
        ls, marker, alpha = '-', 'o', 1.0
    r = results[name]
    ax.plot(r['ages'], r['acc'], linestyle=ls, marker=marker, ms=2,
            color=COLORS[name], alpha=alpha, label=name)
ax.set_xlabel(r"Age $a=N-m$")
ax.set_ylabel("held-out accuracy")
ax.set_title("(a) Latent classification")
ax.set_xlim(0, N_DAYS)
ax.set_ylim(0.0, 1.03)
ax.legend(fontsize=6.5, loc="lower left", framealpha=0.85)

ax = axes[1]
for name in METHODS:
    if name == "Original":
        ls, marker, alpha = '--', None, 0.85
    else:
        ls, marker, alpha = '-', 'o', 1.0
    r = results[name]
    ax.plot(r['ages'], r['post_nll'], linestyle=ls, marker=marker, ms=2,
            color=COLORS[name], alpha=alpha, label=name)
ax.set_xlabel(r"Age $a=N-m$")
ax.set_ylabel("posterior NLL")
ax.set_title("(b) Class-posterior NLL")
ax.set_xlim(0, N_DAYS)

fig.tight_layout()
fig.savefig(os.path.join(FIGS, "mnist_replay_classification.pdf"))
fig.savefig(os.path.join(FIGS, "mnist_replay_classification.png"))
plt.close(fig)
print("  Saved mnist_replay_classification")

# Compact table as a figure.
methods_for_table = ["Original", "CAS", "FIFO", "Reservoir", "Log-age", "Greedy-PL"]
col_labels = ["method", "states", "acc\nall", f"acc\nage≥{OLD_AGE}", "NLL\nall", f"NLL\nage≥{OLD_AGE}"]
cell_text = []
for name in methods_for_table:
    s = summary[name]
    cell_text.append([
        name,
        str(s['states']),
        f"{s['acc_all']:.3f}",
        f"{s['acc_old']:.3f}",
        f"{s['post_nll_all']:.3f}",
        f"{s['post_nll_old']:.3f}",
    ])

fig, ax = plt.subplots(figsize=(7.0, 2.2))
ax.axis('off')
tab = ax.table(cellText=cell_text, colLabels=col_labels, cellLoc='center', loc='center')
tab.auto_set_font_size(False)
tab.set_fontsize(8)
tab.scale(1.0, 1.25)
ax.set_title(f"MNIST latent replay task, B={B_STATES} GM states", fontsize=11, pad=8)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "mnist_replay_task_table.pdf"))
fig.savefig(os.path.join(FIGS, "mnist_replay_task_table.png"))
plt.close(fig)
print("  Saved mnist_replay_task_table")


# %% [markdown]
# ---
#
# ## Notes for the manuscript
#
# - `mnist_replay_classification.pdf` is the main figure for the paper.
# - `mnist_replay_task_table.pdf` gives compact scalar values for the response matrix or appendix.
# - The posterior-NLL panel is usually more informative than accuracy because the three class-conditional Gaussian components remain fairly separable in the PCA space; accuracy can remain high even when the recalled class weights are distorted.
