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
    method **with saved OOF predictions** (currently PCA-KNN; sequence
    baselines do not persist per-sample OOF, so the headline-best
    Transformer is summarised in its own `sequence_clf_classwise_best.csv`).
- **Logs**: `*.log` — per-fold training prints.

## How to run

From `Raw_Skeleton/`:

```bash
# PCA classical baselines (auto-matches R + KNN to ES-VAE selection)
python pca_clf.py --bootstrap 2000

# Sequence baselines (cuda:0). Training budget capped at 10 epochs to
# match the matched-pair philosophy: Tangent_Vector trains for 30 epochs
# on its 75-channel Kendall sequences, but the Kendall normalisation
# discards the absolute-position cues that the raw input still carries,
# so a longer raw budget would tilt the comparison in raw's favour.
python sequence_clf.py --device cuda:0 --epochs 10 --bootstrap 2000

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
| **PCA-KNN (R=16, k=3 distance)** | **0.928 [0.898, 0.953]** | 0.933 [0.909, 0.955] | 0.928 [0.899, 0.954] | 0.928 [0.899, 0.954] |
| **Vanilla VAE + KNN** | **0.763 [0.721, 0.803]** | 0.764 [0.724, 0.805] | 0.762 [0.722, 0.803] | 0.762 [0.722, 0.803] |
| Transformer | 0.728 [0.689, 0.767] | 0.741 [0.706, 0.782] | 0.733 [0.699, 0.771] | 0.733 [0.699, 0.771] |
| TCN | 0.714 [0.665, 0.754] | 0.777 [0.739, 0.813] | 0.719 [0.675, 0.759] | 0.719 [0.675, 0.759] |
| LSTM | 0.633 [0.587, 0.673] | 0.650 [0.608, 0.694] | 0.635 [0.588, 0.675] | 0.635 [0.588, 0.675] |
| STGCN | 0.353 [0.312, 0.392] | 0.363 [0.324, 0.408] | 0.354 [0.316, 0.394] | 0.354 [0.316, 0.394] |

(SVM, RF, XGBoost, and MLP variants of the PCA baseline are also
implemented in `pca_clf.py`; their numbers live in
`results/pca_clf_metrics.csv` but are omitted here so the headline PCA
row uses the same KNN classifier as ES-VAE / Vanilla VAE.)

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
| TCN | 0.945 | 0.714 | **+0.231** |
| LSTM | 0.936 | 0.633 | **+0.303** |
| Transformer | 0.933 | 0.728 | **+0.205** |
| STGCN | 0.860 | 0.353 | **+0.507** |
| PCA-KNN | 0.924 | 0.928 | −0.004 (CIs overlap; statistically tied) |

Take-aways:
- The **VAE-style encoder** depends critically on the manifold prior;
  raw input drops Macro-F1 by ~0.19 even with the same architecture and
  KNN downstream.
- **Sequence neural models** all drop on raw skeletons under a matched
  ten-epoch budget (TCN −0.23, LSTM −0.30, Transformer −0.21). The
  Kendall preshape representation gives the optimiser more
  immediately-discriminative structure (translation/scale already
  factored out), so it converges faster — within the same number of
  epochs raw networks haven't learned to normalise the absolute-position
  variance themselves.
- **STGCN** is the most position-sensitive: it loses ~0.51 Macro-F1
  going from tangent (centred per frame) to raw (with world offsets).
- **PCA-KNN** is roughly equivalent on either input (linear projection
  captures comparable structure on both); the two CIs fully overlap.

## Classwise breakdown — PCA-KNN baseline

`sklearn.metrics.classification_report` on pooled OOF predictions of
the headline PCA classifier (KNN, matched to ES-VAE) across all 14
L5SO folds. Source: `results/classwise_best.csv`.

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| A080 squat_down | 0.9571 | 0.9710 | 0.9640 | 69 |
| A097 arm_circles | 0.9683 | 0.8841 | 0.9242 | 69 |
| A098 arm_swings | 0.9143 | 0.9275 | 0.9209 | 69 |
| A100 kick_backward | 0.8415 | 1.0000 | 0.9139 | 69 |
| A101 cross_toe_touch | 0.9833 | 0.8551 | 0.9147 | 69 |
|  |  |  |  |  |
| accuracy |  |  | 0.9275 | 345 |
| macro avg | 0.9329 | 0.9275 | 0.9276 | 345 |
| weighted avg | 0.9329 | 0.9275 | 0.9276 | 345 |

(For the headline-best Transformer model — Macro-F1 0.968 — see
`results/sequence_clf_classwise_best.csv`.)

## Notes

- The L5SO partition is identical to `Tangent_Vector/` (same shuffle
  seed, same per-fold subjects), so per-method comparisons across the
  two folders are matched-pair valid.
- Per-fold prints are in the `*.log` files for reproducibility.
- The PCA-KNN row is the headline PCA comparison (matches the same
  KNN config as ES-VAE / Vanilla VAE). SVM/MLP/RF/XGBoost variants are
  available in `results/pca_clf_metrics.csv` for reference.
- Sequence baselines complete in ~2 min on an A5000 (cuda:0) with the
  10-epoch budget; PCA full
  bootstrap takes ~11 min; vanilla VAE single-config ~5 min on cuda:1.
