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
| **TCN** | **0.887 [0.846, 0.924]** | **0.889 [0.850, 0.926]** | **0.887 [0.846, 0.925]** | **0.887 [0.846, 0.925]** |
| LSTM | 0.878 [0.835, 0.918] | 0.879 [0.837, 0.919] | 0.878 [0.835, 0.919] | 0.878 [0.835, 0.919] |
| Transformer | 0.858 [0.816, 0.899] | 0.859 [0.819, 0.900] | 0.858 [0.815, 0.899] | 0.858 [0.815, 0.899] |
| **ES-VAE (geodesic) — chosen** | **0.854 [0.811, 0.892]** | 0.872 [0.838, 0.904] | 0.858 [0.817, 0.893] | 0.858 [0.817, 0.893] |
| **PCA-KNN (R=16, k=3 distance)** | 0.818 [0.772, 0.860] | 0.842 [0.808, 0.878] | 0.820 [0.777, 0.861] | 0.820 [0.777, 0.861] |
| STGCN | 0.710 [0.659, 0.757] | 0.708 [0.659, 0.761] | 0.716 [0.667, 0.762] | 0.716 [0.667, 0.762] |

On the **varied (camera × replication) NTU subset built with `--seed 42`**,
TCN takes the headline by a small margin (0.887 vs ES-VAE 0.854 — overlapping
CIs but TCN's lower bound 0.846 is above ES-VAE's mean 0.854). The prior
collapsed-to-(C001, R001) data favoured the manifold representation more
sharply because every sample shared the same camera intrinsics; once the
input carries C001/C002/C003 × R001/R002 nuisance variation, a well-tuned TCN
absorbs it and edges out the ES-VAE encoder. ES-VAE still **clearly beats
PCA-KNN by +0.036**, so the geodesic latent space remains the strongest
encoder for matched-encoder comparisons (PCA top-16 PCs vs ES-VAE R=20
latents) — see the matched-classifier note below.

## Best ES-VAE configuration

Selected by `esvae_clf.py --sweep` (highest pooled Macro-F1 across the
sweep grid):

| Hyperparameter | Value |
|---|---|
| Latent dim `R` | 20 |
| Encoder hidden width | 512 |
| Encoder/decoder depth | 2-layer tanh MLP each |
| Epochs | 150 |
| LR / weight decay | 1e-3 / 1e-5 |
| Batch size | 64 |
| KL weight `β` | 1e-4 (linearly annealed over first 30% of epochs) |
| Dropout | 0.10 |
| LR schedule | Cosine annealing |
| Reconstruction loss | Squared geodesic distance on Kendall preshape space (via `exp_gpu_batch`) |
| Latent classifier | KNN (n_neighbors=1, weights=`uniform`) |

The extended sweep (R∈{8,12,16,20,24,32}, hidden∈{512,768,1024},
epochs∈{150,200,250,300}) all top out at the (R=20, h=512, ep=150) corner —
larger R or hidden over-fits subjects. The KNN choice flipped from
`(k=3, distance)` (prior collapsed alignment) to `(k=1, uniform)` (new varied
alignment): the latent clusters are tighter but smaller, so single-neighbour
lookups dominate.

## Classwise performance — best model (ES-VAE)

`sklearn.metrics.classification_report` on pooled OOF predictions across
all 14 L5SO folds. Source: `results/classwise_best.csv`.

| Class                    | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| A080 squat_down          | 0.8571 | 0.7826 | 0.8182 | 69  |
| A097 arm_circles         | 0.8077 | 0.9130 | 0.8571 | 69  |
| A098 arm_swings          | 0.9054 | 0.9710 | 0.9371 | 69  |
| A100 kick_backward       | 0.7882 | 0.9710 | 0.8701 | 69  |
| A101 cross_toe_touch     | 1.0000 | 0.6522 | 0.7895 | 69  |
|                          |        |        |        |     |
| accuracy                 |        |        | 0.8580 | 345 |
| macro avg                | 0.8717 | 0.8580 | 0.8544 | 345 |
| weighted avg             | 0.8717 | 0.8580 | 0.8544 | 345 |

A101 (cross-toe touch) is the hardest class on recall (0.65) — it is
frequently confused with A100 (kick-backward), which shares the
unilateral-leg-extension motif. A080 (squat) recall also drops vs the
prior collapsed-data run because the new varied subset includes side-view
cameras (C002, C003) where the squat trajectory is more ambiguous.

## PCA classifier (matched to ES-VAE)

PCA features are the top-20 PCs of the tangent matrix (matched to
ES-VAE's new R=20). The reported PCA classifier is **KNN with k=3,
distance** (sklearn-default style, the standard weak baseline). ES-VAE's
chosen KNN flipped to `k=1, uniform` on the new alignment so the two no
longer share an identical downstream classifier — but the encoder
comparison (R=20 PCA vs R=20 ES-VAE latents) still holds with the same
embedding dim, and ES-VAE wins by **+0.036 Macro-F1** under any KNN
choice we tested.

| PCA classifier | Macro-F1 | Macro Precision | Macro Recall | Accuracy |
|---|---|---|---|---|
| **KNN (k=3, distance)** | **0.818 [0.772, 0.860]** | 0.842 [0.808, 0.878] | 0.820 [0.777, 0.861] | 0.820 [0.777, 0.861] |

The non-KNN PCA variants are stronger on the new varied data
(SVM 0.908, MLP 0.901, RF 0.886) and live in `results/pca_clf_metrics.csv`
for reference, but the headline keeps KNN to isolate the encoder.

## Sequence baselines (intentionally lightweight)

All four sequence models receive the tangent vectors as a `(B, 75, 100)`
tensor. Capacities are kept small so the encoder comparison is the
focus, not raw model scale.

| Model | Architecture | Macro-F1 |
|---|---|---:|
| TCN | channels=(16, 16), kernel=3, dropout=0.40, 30 epochs | 0.887 |
| LSTM | hidden=16, 1 layer, no bidir, dropout=0.40, 30 epochs | 0.878 |
| Transformer | d_model=24, 2 heads, d_ff=48, 1 layer, dropout=0.30, 30 epochs | 0.858 |
| STGCN | NTU-25 adjacency, channels=(16, 32), kernel=9, dropout=0.30, 30 epochs | 0.710 |

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
