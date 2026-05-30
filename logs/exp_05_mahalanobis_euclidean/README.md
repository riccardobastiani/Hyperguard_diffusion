# exp_05: PCA + Ledoit-Wolf Mahalanobis Euclidean Baseline

## Goal

Evaluate a stronger covariance-aware Euclidean baseline on the held-out HF features from exp_04.

## Definition

Plain Euclidean distance uses a spherical safe region:

```text
d_E(x) = ||x - mu_safe||_2
```

Mahalanobis distance uses the shape of the safe distribution:

```text
d_M(x) = sqrt((x - mu_safe)^T Sigma^{-1} (x - mu_safe))
```

Because hidden states are high-dimensional and the safe training cache has few samples, this experiment uses PCA before covariance fitting:

```text
z = W_k^T (x - mu_pca)
```

Then Ledoit-Wolf shrinkage estimates a stable covariance:

```text
Sigma_lw = (1 - lambda) S + lambda tau I
```

The detector score is:

```text
d_M(z) = sqrt((z - mu_safe)^T Sigma_lw^{-1} (z - mu_safe))
```

The radius is calibrated on train/cache safe scores:

```text
R = quantile_q({d_M(z_i_safe_train)})
block if d_M(z_test) > R
```

## Run

This experiment reuses held-out features from exp_04. Run exp_04 first if `heldout_safe_features.npz` and `heldout_unsafe_features.npz` are missing.

```bash
cd /media/hdd/huseynov/Hyperguard_diffusion
source .venv312/bin/activate

PYTHON=.venv312/bin/python \
PCA_DIM=32 \
./run_mahalanobis_euclidean.sh
```

## Outputs

- `mahalanobis_heldout.log`
- `mahalanobis_heldout_summary.csv`
- `mahalanobis_heldout_rates.png`
- `ANALYSIS.md`
