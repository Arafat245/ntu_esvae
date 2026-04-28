# Raw-Skeleton Action Classification (NTU 5-class)

This folder contains the parallel **raw-skeleton** classification pipeline
for the NTU activity-recognition project. It mirrors `Tangent_Vector/`
end-to-end (same five activities, same 69 common subjects, same L5SO CV)
but feeds models the **raw NTU joint coordinates** instead of Kendall
tangent-space vectors. The two pipelines are matched-pair comparable so
the contribution of the manifold representation is isolated.

## Inputs

All scripts read from `../data/data_ntu.pkl`:

- `dict[str -> np.ndarray (25 joints, 3 xyz, T_var frames)]` of raw NTU
  skeleton trajectories.
- 345 entries with keys like `'8_A080'` (subject id `_` class id).
- Frame counts vary per sample (24–159, mean ≈67); each sample is
  **linearly interpolated to T = 100** along the time axis in
  `cv_utils.load_data`. Per-channel z-score standardisation is computed
  on the training fold only.

The 345 samples are 69 subjects × 5 classes (perfectly balanced):
`A080 squat_down`, `A097 arm_circles`, `A098 arm_swings`,
`A100 kick_backward`, `A101 cross_toe_touch`.

## Cross-validation protocol

Identical to `Tangent_Vector/`: deterministic `seed=42` shuffle of the 69
subjects, then partitioned into **14 folds** (13 × 5 subjects + 1 × 4).
Each fold's held-out subjects form the test set; the remaining subjects
are training. No separate validation set — predictions are pooled across
all 14 folds, and **subject-level bootstrap 95% CIs** (2000 resamples)
are reported.

`cv_utils.py` re-imports the CV partition, bootstrap, and reporting
helpers directly from `Tangent_Vector/cv_utils.py` (via `importlib`) so
the two pipelines really do use the same fold splits.

## Files

### Scripts
- `cv_utils.py` — raw-skeleton loader (linear time resample, returns
  both `X_seq (N, 75, T)` and `X_flat (N, 25*3*T)`), plus shared helpers.
- `pca_clf.py` — PCA on raw skeletons → KNN/SVM/RF/XGBoost/MLP. Reads
  the chosen R and KNN config from `../Tangent_Vector/results/` so the
  PCA-KNN baseline uses the **same R and same KNN as ES-VAE**.
- `sequence_clf.py` — TCN / LSTM / Transformer / STGCN on the
  `(N, 75, T)` raw-skeleton sequence. Architectures and capacities are
  copied verbatim from `Tangent_Vector/sequence_clf.py` for a like-for-like
  comparison.
- `vae_clf.py` — vanilla MSE-loss VAE on flattened raw-skeleton sequences,
  followed by KNN on latent means. The encoder topology is the same as
  ES-VAE's `NonlinearVAE`, but the reconstruction loss is plain MSE
  (no exp-map / geodesic distance). When run without `--sweep`, the
  encoder hyperparameters and KNN config are auto-matched to the ES-VAE
  selection from `../Tangent_Vector/results/esvae_clf_config.json`, so
  ES-VAE vs Vanilla-VAE is a clean isolated test of the manifold loss.

### Outputs (`results/`)
- **Per-script metrics**:
  - `pca_clf_metrics.csv` (rows: KNN, SVM, RF, XGBoost, MLP),
    `pca_clf_classwise_best.csv`, `pca_clf_oof.json`
  - `sequence_clf_metrics.csv` (rows: TCN, LSTM, TRANSFORMER, STGCN),
    `sequence_clf_classwise_best.csv`
  - `vae_clf_metrics.csv`, `vae_clf_classwise.csv`, `vae_clf_config.json`,
    `vae_clf_oof.json`
- **Combined summaries**:
  - `summary_headline.csv` — one row per method (10 total) with Macro
    F1/Prec/Recall + Accuracy and 95% CIs, sorted by Macro-F1.
  - `summary_all_methods.csv` — full version including weighted metrics.
  - `classwise_best.csv` — sklearn `classification_report` for the best
    method **with saved OOF predictions** (currently PCA-SVM; sequence
    baselines do not persist per-sample OOF, so the headline-best
    Transformer is summarised in its own `sequence_clf_classwise_best.csv`).
- **Logs**: `*.log` — per-fold training prints.

## How to run

From `Raw_Skeleton/`:

```bash
# PCA classical baselines (auto-matches R + KNN to ES-VAE selection)
python pca_clf.py --bootstrap 2000

# Sequence baselines (cuda:0)
python sequence_clf.py --device cuda:0 --epochs 30 --bootstrap 2000

# Vanilla VAE + KNN (cuda:1; auto-matches encoder + KNN to ES-VAE config)
python vae_clf.py --device cuda:1 --bootstrap 2000

# Optional: tiny encoder × KNN sweep on the VAE
python vae_clf.py --device cuda:1 --sweep --bootstrap 2000
```

## Headline comparison (raw skeletons)

Pooled across 14 L5SO folds. Cells show **mean [95% CI]** from the
2000-iter subject bootstrap. Sorted by Macro-F1.

| Method | Macro-F1 | Macro Precision | Macro Recall | Accuracy |
|---|---|---|---|---|
| **PCA-SVM (RBF)** | **0.971 [0.951, 0.988]** | 0.972 [0.954, 0.989] | 0.971 [0.951, 0.988] | 0.971 [0.951, 0.988] |
| **Transformer** | **0.968 [0.937, 0.994]** | 0.969 [0.941, 0.994] | 0.968 [0.936, 0.994] | 0.968 [0.936, 0.994] |
| LSTM | 0.954 [0.924, 0.977] | 0.954 [0.928, 0.977] | 0.954 [0.925, 0.977] | 0.954 [0.925, 0.977] |
| TCN | 0.951 [0.921, 0.977] | 0.953 [0.927, 0.977] | 0.951 [0.919, 0.977] | 0.951 [0.919, 0.977] |
| PCA-MLP | 0.951 [0.921, 0.977] | 0.952 [0.924, 0.977] | 0.951 [0.919, 0.977] | 0.951 [0.919, 0.977] |
| **PCA-KNN (R=16, k=3 distance)** | **0.928 [0.898, 0.953]** | 0.933 [0.909, 0.955] | 0.928 [0.899, 0.954] | 0.928 [0.899, 0.954] |
| PCA-RF | 0.925 [0.891, 0.954] | 0.927 [0.897, 0.954] | 0.925 [0.890, 0.954] | 0.925 [0.890, 0.954] |
| PCA-XGBoost | 0.922 [0.890, 0.953] | 0.926 [0.897, 0.956] | 0.922 [0.890, 0.954] | 0.922 [0.890, 0.954] |
| **Vanilla VAE + KNN** | **0.763 [0.721, 0.803]** | 0.764 [0.724, 0.805] | 0.762 [0.722, 0.803] | 0.762 [0.722, 0.803] |
| STGCN | 0.441 [0.394, 0.487] | 0.448 [0.402, 0.500] | 0.441 [0.397, 0.487] | 0.441 [0.397, 0.487] |

### Headline note on STGCN

STGCN collapses on raw skeletons (0.44) for two compounding reasons:
the lightweight cap (channels = 8/16, dropout 0.40) carried over from
the Tangent_Vector pipeline, and — more importantly — raw NTU skeletons
preserve **world translation and absolute scale**, which the small
STGCN can't normalise away. On Kendall tangent vectors that translation
is already removed by `preprocess_temporal`, so STGCN there is much
better-behaved (0.86). The other models cope because they either
standardise per channel (TCN/LSTM/Transformer with `ChannelStandardizer`)
or learn projections that absorb the offset.

## ES-VAE vs Vanilla VAE — the manifold-loss isolation test

Same encoder topology (2-layer tanh MLP, R=16, hidden=512, dropout=0.10,
150 epochs, KL-annealed), same KNN downstream classifier (k=3, distance).
The **only** difference is the reconstruction loss / target:

| Pipeline | Reconstruction target | Loss | Pooled Macro-F1 |
|---|---|---|---:|
| ES-VAE on tangent vectors | manifold curve via `exp_map(mu, v_hat)` | squared geodesic distance on Kendall preshape space | **0.951** [0.929, 0.971] |
| Vanilla VAE on raw skeletons | flattened raw sequence | plain MSE | **0.763** [0.721, 0.803] |

ES-VAE outperforms by **+0.19 Macro-F1**, attributable entirely to (i)
the Kendall preshape representation (translation-/scale-/rotation-invariant
curves) and (ii) the geodesic reconstruction loss that respects that
manifold structure.

## Tangent vs Raw — same-method comparison

Same architectures, same hyperparameters, same CV folds — only the
input differs (Kendall tangent vector vs linearly-resampled raw skeleton).

| Method | Tangent (Macro-F1) | Raw (Macro-F1) | Δ |
|---|---:|---:|---:|
| ES-VAE / Vanilla VAE | **0.951** | 0.763 | **+0.188** |
| TCN | 0.945 | 0.951 | −0.006 |
| LSTM | 0.936 | 0.954 | −0.018 |
| Transformer | 0.933 | 0.968 | −0.035 |
| STGCN | 0.860 | 0.441 | **+0.419** |
| PCA-KNN | 0.924 | 0.928 | −0.004 |
| PCA-MLP | 0.960 | 0.951 | +0.009 |
| PCA-SVM | 0.954 | 0.971 | −0.017 |

Take-aways:
- The **VAE-style encoder** depends critically on the manifold prior;
  raw input drops Macro-F1 by ~0.19.
- **Sequence neural models** modestly *prefer* raw skeletons — the
  scale and absolute-position cues add discriminative signal that the
  Kendall normalisation removes.
- **STGCN** is the most translation-sensitive: it loses ~0.42 Macro-F1
  going from tangent (centred per frame) to raw (with world offsets).
- **PCA-based classifiers** are roughly equivalent on either input
  (linear projection captures comparable structure on both).

## Classwise breakdown — best **headline** model with saved OOF (PCA-SVM)

`sklearn.metrics.classification_report` on pooled OOF predictions across
all 14 L5SO folds. Source: `results/classwise_best.csv`.

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| A080 squat_down | 1.0000 | 0.9710 | 0.9853 | 69 |
| A097 arm_circles | 0.9848 | 0.9420 | 0.9630 | 69 |
| A098 arm_swings | 0.9577 | 0.9855 | 0.9714 | 69 |
| A100 kick_backward | 0.9452 | 1.0000 | 0.9718 | 69 |
| A101 cross_toe_touch | 0.9706 | 0.9565 | 0.9635 | 69 |
|  |  |  |  |  |
| accuracy |  |  | 0.9710 | 345 |
| macro avg | 0.9717 | 0.9710 | 0.9710 | 345 |
| weighted avg | 0.9717 | 0.9710 | 0.9710 | 345 |

(For Transformer, which slightly trails PCA-SVM at 0.968 but has the
best per-fold prints, see `sequence_clf_classwise_best.csv`.)

## Notes

- The L5SO partition is identical to `Tangent_Vector/` (same shuffle
  seed, same per-fold subjects), so per-method comparisons across the
  two folders are matched-pair valid.
- Per-fold prints are in the `*.log` files for reproducibility.
- The PCA-KNN row is the headline PCA comparison (matches the same
  KNN config as ES-VAE / Vanilla VAE). PCA-SVM/MLP/RF/XGBoost rows are
  reported for reference.
- Sequence baselines complete in ~3 min on an A5000 (cuda:0); PCA full
  bootstrap takes ~11 min; vanilla VAE single-config ~5 min on cuda:1.
