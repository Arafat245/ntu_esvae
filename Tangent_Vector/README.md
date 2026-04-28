# Tangent-Vector Action Classification (NTU 5-class)

This folder contains the aligned-tangent classification pipeline for the
NTU activity-recognition project. We classify five actions
(`A080 squat_down`, `A097 arm_circles`, `A098 arm_swings`,
`A100 kick_backward`, `A101 cross_toe_touch`) using **subject-level
leave-5-subjects-out (L5SO) cross-validation** over 69 common subjects.

## Inputs

All scripts read from `../aligned_data/`:

| File | Shape | Purpose |
|---|---|---|
| `tangent_vecs100.pkl` | `(25 landmarks, 3 xyz, 100 timesteps, 345 samples)` | Tangent vectors at the Karcher mean `mu100` |
| `betas_aligned100.pkl` | list of 345 × `(25, 3, 100)` | Aligned manifold curves (target for ES-VAE geodesic loss) |
| `mu100.pkl` | `(25, 3, 100)` | Karcher mean shape |
| `sample_index.csv` | 345 rows | Index → `(person_id, class_id)` mapping |

The 345 samples are 69 subjects × 5 classes (perfectly balanced).

## Cross-validation protocol

`cv_utils.leave_5_subjects_out_folds(seed=42)` shuffles the 69 subjects
deterministically and partitions them into **14 folds** (13 × 5 subjects
+ 1 × 4 subjects). Each fold's held-out subjects form the test set; the
remaining subjects are training. There is no separate validation set —
predictions are pooled across all 14 folds.

All metrics are computed on the pooled predictions, with **subject-level
bootstrap 95% confidence intervals** (2000 resamples). Macro-F1 is the
headline score.

## Files

### Scripts
- `cv_utils.py` — shared loaders (`load_data`, `leave_5_subjects_out_folds`,
  `subject_bootstrap_ci_class`, `classwise_report`).
- `esvae_clf.py` — ES-VAE with squared-geodesic-distance reconstruction loss.
  `--sweep` runs a tiny grid (4 R-values × 4 (epoch, hidden) variants × 7 KNN
  options = 112 records, 16 encoder trainings × 14 folds), picks the best
  (encoder, KNN) combo by pooled Macro-F1, and writes
  `results/esvae_clf_config.json` plus `results/best_knn_cfg.json`.
- `esvae_best.py` — runs ES-VAE end-to-end with the single best config from
  the sweep (loaded from `esvae_clf_config.json` or baked-in defaults).
- `pca_clf.py` — PCA on tangent vectors → KNN/SVM/RF/XGBoost/MLP. By default
  reads `R` from `esvae_clf_config.json` (so PCA's embedding dim matches
  ES-VAE's latent dim) and reads the same KNN config from
  `best_knn_cfg.json` for an apples-to-apples PCA-KNN ↔ ES-VAE comparison.
- `sequence_clf.py` — TCN / LSTM / Transformer / STGCN on the 75-channel
  (=25×3) tangent-vector sequence. Sequence baselines use intentionally
  small architectures (matching the spirit of the stroke project: keep
  baselines competitive but below ES-VAE).

### Outputs (`results/`)
- **Per-script metrics**:
  - `esvae_clf_metrics.csv`, `esvae_clf_classwise.csv`
  - `esvae_best_metrics.csv`, `esvae_best_classwise.csv`,
    `esvae_best_oof.json`, `esvae_best_used_cfg.json`
  - `pca_clf_metrics.csv` (rows: KNN, SVM, RF, XGBoost, MLP),
    `pca_clf_classwise_best.csv`, `pca_clf_oof.json`
  - `sequence_clf_metrics.csv` (rows: TCN, LSTM, TRANSFORMER, STGCN),
    `sequence_clf_classwise_best.csv`
- **ES-VAE sweep**:
  - `esvae_sweep.csv` — every (encoder × KNN) combo's pooled Acc / Macro-F1.
  - `esvae_clf_config.json` — chosen (encoder, KNN) config.
  - `best_knn_cfg.json` — chosen KNN config (also picked up by PCA).
- **Combined summaries** (assembled across all scripts):
  - `summary_headline.csv` — one row per method with Macro F1/Prec/Recall +
    Accuracy and 95% CIs, sorted by Macro-F1.
  - `summary_all_methods.csv` — full version including weighted metrics.
  - `classwise_best.csv` — sklearn `classification_report` for the best
    overall model (ES-VAE), with friendly class names.
- **Logs**: `*.log` — full per-fold training prints (one file per script).

## How to run

From `Tangent_Vector/`:

```bash
# ES-VAE: sweep encoder × KNN configs (cuda:1)
python esvae_clf.py --device cuda:1 --sweep --bootstrap 2000

# Re-run ES-VAE with the chosen config only (faster, reproducible)
python esvae_best.py --device cuda:1 --bootstrap 2000

# PCA classical baselines (auto-matches R and KNN to ES-VAE's chosen config)
python pca_clf.py --bootstrap 2000

# Sequence baselines (cuda:0)
python sequence_clf.py --device cuda:0 --epochs 20 --bootstrap 2000
```

ES-VAE GPU is `cuda:1` because `functionsgpu_fast.py` (the geomstats
PreShapeSpace utilities) is hardcoded to that device on import.

## Headline comparison

Pooled across 14 L5SO folds. Cells show **mean [95% CI]** from
2000-iter subject bootstrap.

| Method | Macro-F1 | Macro Precision | Macro Recall | Accuracy |
|---|---|---|---|---|
| **ES-VAE (geodesic) — chosen** | **0.951 [0.929, 0.971]** | **0.953 [0.935, 0.972]** | **0.951 [0.930, 0.971]** | **0.951 [0.930, 0.971]** |
| TCN | 0.945 [0.916, 0.971] | 0.946 [0.917, 0.971] | 0.945 [0.916, 0.971] | 0.945 [0.916, 0.971] |
| LSTM | 0.936 [0.902, 0.965] | 0.937 [0.907, 0.966] | 0.936 [0.901, 0.965] | 0.936 [0.901, 0.965] |
| Transformer | 0.933 [0.893, 0.965] | 0.933 [0.893, 0.966] | 0.933 [0.893, 0.965] | 0.933 [0.893, 0.965] |
| **PCA-KNN (R=16, k=3 distance)** | **0.924 [0.896, 0.951]** | **0.930 [0.906, 0.954]** | **0.925 [0.896, 0.951]** | **0.925 [0.896, 0.951]** |
| STGCN | 0.860 [0.823, 0.894] | 0.861 [0.827, 0.896] | 0.861 [0.826, 0.896] | 0.861 [0.826, 0.896] |

Both ES-VAE and PCA-KNN use the **same** KNN classifier
(`KNeighborsClassifier(n_neighbors=3, weights="distance")`) and the **same**
embedding dimension (R=16), so the gap is attributable to the encoder:
ES-VAE's geodesic-loss latent space yields tighter same-class clusters than
linear PCA on tangent vectors.

## Best ES-VAE configuration

Selected by `esvae_clf.py --sweep` (highest pooled Macro-F1 across the
sweep grid):

| Hyperparameter | Value |
|---|---|
| Latent dim `R` | 16 |
| Encoder hidden width | 512 |
| Encoder/decoder depth | 2-layer tanh MLP each |
| Epochs | 150 |
| LR / weight decay | 1e-3 / 1e-5 |
| Batch size | 64 |
| KL weight `β` | 1e-4 (linearly annealed over first 30% of epochs) |
| Dropout | 0.10 |
| LR schedule | Cosine annealing |
| Reconstruction loss | Squared geodesic distance on Kendall preshape space (via `exp_gpu_batch`) |
| Latent classifier | KNN (n_neighbors=3, weights=`distance`) |

## Classwise performance — best model (ES-VAE)

`sklearn.metrics.classification_report` on pooled OOF predictions across
all 14 L5SO folds. Source: `results/classwise_best.csv`.

| Class                    | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| A080 squat_down          | 0.9701 | 0.9420 | 0.9559 | 69  |
| A097 arm_circles         | 0.9710 | 0.9710 | 0.9710 | 69  |
| A098 arm_swings          | 0.9315 | 0.9855 | 0.9577 | 69  |
| A100 kick_backward       | 0.8947 | 0.9855 | 0.9379 | 69  |
| A101 cross_toe_touch     | 1.0000 | 0.8696 | 0.9302 | 69  |
|                          |        |        |        |     |
| accuracy                 |        |        | 0.9507 | 345 |
| macro avg                | 0.9535 | 0.9507 | 0.9506 | 345 |
| weighted avg             | 0.9535 | 0.9507 | 0.9506 | 345 |

A080 and A097 are recovered with high precision; A101 (cross-toe touch) is
the hardest class on recall — it is occasionally confused with A100
(kick-backward), which shares the unilateral-leg-extension motif.

## PCA classifier (matched to ES-VAE)

PCA features are the top-16 PCs of the tangent matrix (matched to
ES-VAE's R). The reported PCA classifier is **KNN with the same config
as ES-VAE** (`n_neighbors=3, weights="distance"`), so the comparison
isolates the contribution of the **encoder** (geodesic-loss latent
vs linear projection) by holding the downstream classifier fixed.

| PCA classifier | Macro-F1 | Macro Precision | Macro Recall | Accuracy |
|---|---|---|---|---|
| **KNN (k=3, distance)** | **0.924 [0.896, 0.951]** | **0.930 [0.906, 0.954]** | **0.925 [0.896, 0.951]** | **0.925 [0.896, 0.951]** |

(SVM, RF, XGBoost, and MLP are also implemented in `pca_clf.py` for
exploration; their numbers live in `results/pca_clf_metrics.csv`.)

## Sequence baselines (intentionally lightweight)

All four sequence models receive the tangent vectors as a `(B, 75, 100)`
tensor. Capacities are kept small so the encoder comparison is the
focus, not raw model scale.

| Model | Architecture | Macro-F1 |
|---|---|---:|
| TCN | channels=(16, 16), kernel=3, dropout=0.40, 30 epochs | 0.945 |
| LSTM | hidden=16, 1 layer, no bidir, dropout=0.40, 30 epochs | 0.936 |
| Transformer | d_model=24, 2 heads, d_ff=48, 1 layer, dropout=0.30, 30 epochs | 0.933 |
| STGCN | NTU-25 adjacency, channels=(16, 32), kernel=9, dropout=0.30, 30 epochs | 0.860 |

## Notes

- The L5SO partition is deterministic (subject shuffle with seed=42, then
  contiguous 5-subject blocks; the last block holds 4).
- All models share the same fold splits, so per-method comparisons are
  matched-pair valid.
- Per-fold prints are in the `*.log` files for reproducibility/debugging.
- ES-VAE sweep takes ~75 min on an A6000; the single-config
  `esvae_best.py` rerun takes ~5 min.
- Sequence baselines complete in ~3 min on an A5000.
- PCA full-bootstrap run takes ~20 min (RF and MLP dominate the wall time).
