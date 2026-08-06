# compress-add-smooth
This repository contains the Python/Jupyter implementation and experiments for  Compress--Add--Smooth: Fixed-Budget Temporal Compression of Density-Valued StreamsMichael (Misha) Chertkov, Transactions on Machine Learning Research, 08/2026.  OpenReview page: https://openreview.net/forum?id=wjoixYG0mC

## What is in this repository?

The code implements the Compress--Add--Smooth (CAS) recursion for fixed-budget temporal memory of distribution-valued streams.  In the Gaussian-mixture instantiation used in the paper, the memory is a piecewise-linear protocol on a replay interval `[0,1]`.  Each node stores a labeled Gaussian mixture.  The daily update performs:

1. **Compress:** rescale the old protocol from `[0,1]` to `[0,L/(L+1)]`.
2. **Add:** append the new day distribution at `t=1`.
3. **Smooth:** rebin the resulting `(L+1)`-segment protocol back to the fixed `L`-segment grid.

The repository also includes diagnostics for moment-based forgetting, sliced-Wasserstein and held-out likelihood checks, memory-matched streaming baselines, MNIST latent-space experiments, and the optional SDE marginal-validation test.
