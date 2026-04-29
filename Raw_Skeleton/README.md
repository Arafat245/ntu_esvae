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
- `pca_clf.py` — PCA on raw skeletons → KNN. Reads the chosen R and KNN
  config from `../Tangent_Vector/results/` so the PCA-KNN baseline uses
  the **same R and same KNN as ES-VAE / Vanilla VAE**.
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
  - `pca_clf_metrics.csv` (KNN only — same `(k=3, distance)` as ES-VAE
    / Vanilla VAE), `pca_clf_classwise_best.csv`, `pca_clf_oof.json`
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

# Sequence baselines (cuda:0). 30-epoch budget matches Tangent_Vector
# exactly so the architecture/optimiser comparison is apples-to-apples;
# the only difference is the input representation (raw vs Kendall).
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
| **Transformer** | **0.852 [0.816, 0.885]** | 0.852 [0.818, 0.887] | 0.852 [0.815, 0.884] | 0.852 [0.815, 0.884] |
| **PCA-KNN (R=16, k=3 distance)** | **0.822 [0.780, 0.857]** | 0.831 [0.793, 0.866] | 0.823 [0.783, 0.858] | 0.823 [0.783, 0.858] |
| TCN | 0.800 [0.758, 0.839] | 0.813 [0.773, 0.854] | 0.800 [0.757, 0.841] | 0.800 [0.757, 0.841] |
| LSTM | 0.768 [0.719, 0.812] | 0.769 [0.723, 0.814] | 0.768 [0.719, 0.812] | 0.768 [0.719, 0.812] |
| **Vanilla VAE + KNN** | **0.592 [0.543, 0.637]** | 0.594 [0.546, 0.643] | 0.594 [0.548, 0.641] | 0.594 [0.548, 0.641] |
| STGCN | 0.454 [0.398, 0.509] | 0.467 [0.413, 0.526] | 0.455 [0.403, 0.510] | 0.455 [0.403, 0.510] |

The PCA baseline runs **only** the KNN classifier (same `n_neighbors=3,
weights="distance"` as ES-VAE / Vanilla VAE) so the comparison isolates
the encoder, not the downstream model. The training data here is the
**varied** NTU subset built with `--seed 42`, which spreads each
(subject, class) selection across all available cameras (C001/C002/C003)
and replications (R001/R002) instead of collapsing to C001/R001 — this
re-introduces the world-translation, scale, and rotation nuisance
factors that the Kendall preshape representation in `Tangent_Vector/`
explicitly removes.

### Headline note on STGCN

STGCN collapses on raw skeletons (0.45) for two compounding reasons:
the lightweight cap (channels = 16/32, dropout 0.30) carried over from
the Tangent_Vector pipeline, and — more importantly — raw NTU skeletons
preserve **world translation and absolute scale** (and now camera/replication
variation from the `--seed 42` curation), which the small STGCN can't
normalise away. On Kendall tangent vectors that translation is already
removed by `preprocess_temporal`, so STGCN there is much better-behaved
(0.86). The other models cope because they either standardise per
channel (TCN/LSTM/Transformer with `ChannelStandardizer`) or learn
projections that absorb the offset.

## ES-VAE vs Vanilla VAE — the manifold-loss isolation test

Same encoder topology (2-layer tanh MLP, R=16, hidden=512, dropout=0.10,
150 epochs, KL-annealed), same KNN downstream classifier (k=3, distance).
The **only** difference is the reconstruction loss / target:

| Pipeline | Reconstruction target | Loss | Pooled Macro-F1 |
|---|---|---|---:|
| ES-VAE on tangent vectors | manifold curve via `exp_map(mu, v_hat)` | squared geodesic distance on Kendall preshape space | **0.854** [0.811, 0.892] |
| Vanilla VAE on raw skeletons | flattened raw sequence | plain MSE | **0.592** [0.543, 0.637] |

ES-VAE outperforms by **+0.26 Macro-F1**, attributable entirely to (i)
the Kendall preshape representation (translation-/scale-/rotation-invariant
curves) and (ii) the geodesic reconstruction loss that respects that
manifold structure. Both pipelines run on the **same varied-camera NTU
subset** (C001/C002/C003 × R001/R002) — the unconstrained VAE encoder
cannot absorb the nuisance variation that the manifold pipeline factors
out analytically.

## Tangent vs Raw — same-method comparison

Same architectures, same hyperparameters, same CV folds — only the
input differs (Kendall tangent vector vs linearly-resampled raw skeleton).

| Method | Tangent (Macro-F1) | Raw (Macro-F1) | Δ |
|---|---:|---:|---:|
| ES-VAE / Vanilla VAE | **0.854** | 0.592 | **+0.262** |
| TCN | 0.887 | 0.800 | **+0.087** |
| LSTM | 0.878 | 0.768 | **+0.110** |
| Transformer | 0.858 | 0.852 | **+0.006** |
| STGCN | 0.710 | 0.454 | **+0.256** |
| PCA-KNN | 0.818 | 0.822 | −0.004 (CIs overlap; tied) |

Take-aways:
- The **VAE-style encoder** depends critically on the manifold prior;
  raw input drops Macro-F1 by ~0.26 even with the same architecture and
  KNN downstream.
- **Sequence neural models** mostly drop on raw skeletons under the same
  30-epoch budget (TCN −0.09, LSTM −0.11, Transformer −0.01). Transformer
  comes closest to closing the gap because its attention can absorb the
  nuisance variation given enough capacity; TCN/LSTM still benefit
  clearly from the Kendall normalisation.
- **STGCN** is the most position-sensitive: it loses ~0.26 Macro-F1
  going from tangent (centred per frame) to raw (with world offsets).
- **PCA-KNN** is statistically tied (0.818 vs 0.822); the linear
  projection captures comparable structure on either input — neither
  representation gives the linear baseline a meaningful edge.

## Classwise breakdown — PCA-KNN baseline

`sklearn.metrics.classification_report` on pooled OOF predictions of
the headline PCA classifier (KNN, matched to ES-VAE) across all 14
L5SO folds. Source: `results/classwise_best.csv`.

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| A080 squat_down | 0.8696 | 0.8696 | 0.8696 | 69 |
| A097 arm_circles | 0.8361 | 0.7391 | 0.7846 | 69 |
| A098 arm_swings | 0.7969 | 0.7391 | 0.7669 | 69 |
| A100 kick_backward | 0.7391 | 0.9855 | 0.8447 | 69 |
| A101 cross_toe_touch | 0.9153 | 0.7826 | 0.8437 | 69 |
|  |  |  |  |  |
| accuracy |  |  | 0.8232 | 345 |
| macro avg | 0.8314 | 0.8232 | 0.8219 | 345 |
| weighted avg | 0.8314 | 0.8232 | 0.8219 | 345 |

(For the headline-best Transformer model — Macro-F1 0.852 — see
`results/sequence_clf_classwise_best.csv`.)

## Notes

- The L5SO partition is identical to `Tangent_Vector/` (same shuffle
  seed, same per-fold subjects), so per-method comparisons across the
  two folders are matched-pair valid.
- Per-fold prints are in the `*.log` files for reproducibility.
- The PCA baseline runs only the matched KNN classifier; non-KNN
  variants were removed so the comparison is purely about the encoder.
- Sequence baselines complete in ~5 min on an A5000 (cuda:0) at the
  30-epoch budget; PCA-KNN full bootstrap takes ~2 min; vanilla VAE
  single-config ~5 min on cuda:1.
