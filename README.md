# compress-add-smooth
This repository contains the Python/Jupyter implementation and experiments for  Compress--Add--Smooth: Fixed-Budget Temporal Compression of Density-Valued StreamsMichael (Misha) Chertkov, Transactions on Machine Learning Research, 08/2026.  OpenReview page: https://openreview.net/forum?id=wjoixYG0mC

## What is in this repository?

The code implements the Compress--Add--Smooth (CAS) recursion for fixed-budget temporal memory of distribution-valued streams.  In the Gaussian-mixture instantiation used in the paper, the memory is a piecewise-linear protocol on a replay interval `[0,1]`.  Each node stores a labeled Gaussian mixture.  The daily update performs:

1. **Compress:** rescale the old protocol from `[0,1]` to `[0,L/(L+1)]`.
2. **Add:** append the new day distribution at `t=1`.
3. **Smooth:** rebin the resulting `(L+1)`-segment protocol back to the fixed `L`-segment grid.

The repository also includes diagnostics for moment-based forgetting, sliced-Wasserstein and held-out likelihood checks, memory-matched streaming baselines, MNIST latent-space experiments, and the optional SDE marginal-validation test.

## Which files reproduce which parts of the paper?

| Paper component | Main file(s) | Main outputs |
|---|---|---|
| CAS implementation and tests | `bridge_cas.py`, `bridge_cas_corr.py`, `test_bridge_cas-corr.ipynb` | SDE validation figures and unit-style checks |
| Single-Gaussian default experiment | `exp01_k1_default-corr.ipynb` | age curve, forgetting matrix, replay geometry |
| Single-Gaussian parameter sweeps | `exp02_k1_sweeps.ipynb` | `L` sweep, drift-speed sweep, circle vs. linear drift |
| Gaussian-mixture default experiment | `exp03_k3_default-corr.ipynb` | `K=1` vs. `K=3`, decomposed forgetting, component trajectories |
| `L` and `K` sweeps | `exp04_k3_sweeps-corr.ipynb` | half-life vs. `L`, half-life vs. `K` |
| Scaling experiments | `exp05_scaling.ipynb` | crowding, ambient dimension, split--merge curriculum |
| MNIST latent protocol | `exp06_mnist-corr.ipynb` | PCA centroids, MNIST forgetting, protocol movie, dimension sweep |
| Memory-matched baselines | `exp07_baselines-corr.ipynb` | FIFO, reservoir, log-age, greedy piecewise-linear comparison |
| MNIST latent replay proxy task | `exp08_mnist_replay_task-corr.ipynb` | classification accuracy and class-posterior NLL |

The notebooks with suffix `-corr` correspond to the accepted-paper experiments added or updated during revision.  The unsuffixed notebooks/scripts are retained for traceability and for reproducing the original experiment sequence.

## Installation

A minimal environment is:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

or, with conda/mamba:

```bash
mamba env create -f environment.yml
mamba activate cas-memory
```

The code was developed with Python 3.10+ and uses `numpy`, `scipy`, `torch`, `torchvision`, `matplotlib`, and `jupyter`.

## Running the experiments

From the repository root:

```bash
cd experiments
jupyter lab
```

Then execute the notebooks in numerical order.  Synthetic experiments do not require external data.  MNIST experiments use `torchvision.datasets.MNIST(root="./data", train=True, download=True, ...)` and will download MNIST locally if it is not already present.

A quick smoke test is:

```bash
cd experiments
python test_bridge_cas.py
```

For the accepted-paper SDE validation, execute:

```text
test_bridge_cas-corr.ipynb
```

For the final baseline and MNIST proxy experiments, execute:

```text
exp07_baselines-corr.ipynb
exp08_mnist_replay_task-corr.ipynb
```

All figures are written to `experiments/figs/`.

## Notes on reproducibility

The synthetic experiments are deterministic up to standard numerical variation.  Some diagnostics use Monte Carlo sampling or random directions for sliced-Wasserstein estimates; seeds are set inside the notebooks where needed.  Exact numerical values may vary slightly across hardware, PyTorch versions, and random seeds.

The `figs/` directory contain generated outputs used in the paper.  Regenerating the full figure set may overwrite these files.

## Citation

If you use this code, please cite the paper:

```bibtex
@article{chertkov2026cas,
  title   = {Compress--Add--Smooth: Fixed-Budget Temporal Compression of Density-Valued Streams},
  author  = {Chertkov, Michael (Misha)},
  journal = {Transactions on Machine Learning Research},
  year    = {2026},
  note    = {Reviewed on OpenReview: https://openreview.net/forum?id=wjoixYG0mC}
}
```

## License

Code in this repository is released under the MIT License.  The accepted TMLR paper is distributed under TMLR's publication terms.  MNIST is an external dataset and is not redistributed here.
