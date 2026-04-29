# Raw-Skeleton Action Classification (NTU 10-class top10)

This folder is the parallel **raw-skeleton** classification pipeline. It
mirrors `Tangent_Vector/` end-to-end (same 10 actions, same 40 subjects,
same fold splits) but feeds models the **raw NTU joint coordinates**
instead of Kendall tangent-space vectors. The two pipelines are
matched-pair comparable so the contribution of the manifold representation
is isolated.

## Inputs

All scripts read from `../data/data_ntu.pkl`:

- `dict[str -> np.ndarray (25 joints, 3 xyz, T_var frames)]` of raw NTU
  skeleton trajectories (single-person trials only).
- 400 entries with keys like `'5_A001'` (subject id `_` class id).
- Frame counts vary per sample; each is **linearly interpolated to T = 100**
  along the time axis in `cv_utils.load_data`. Per-channel z-score
  standardisation is computed on the training fold only.

400 samples = 40 subjects × 10 classes (perfectly balanced):
A001 drink water, A002 eat meal, A003 brush teeth, A004 brush hair,
A008 sitting down, A009 standing up, A023 hand waving, A028 phone call,
A029 play with phone, A031 pointing to something.

## Cross-validation protocols

Identical to `Tangent_Vector/` — 8 L5SO subject folds, 3 cross-view
(leave-one-camera-out) folds, and 17 cross-setup (leave-one-NTU-setup-out)
folds. `cv_utils.py` re-imports the partition / bootstrap / reporting
helpers directly from `../Tangent_Vector/cv_utils.py` (via `importlib`)
so the two pipelines really do use the same fold splits. Reported CIs
use 2000-iter subject bootstrap.

## Files

### Scripts
- `cv_utils.py` — raw-skeleton loader (linear time resample, returns
  `(N, 75, T)` and `(N, 25*3*T)` flat), plus shared helpers re-exported
  from the Tangent_Vector module.
- `pca_clf.py` — PCA on raw skeletons → KNN. Reads `R` from
  `../Tangent_Vector/results/esvae_clf_config.json` and KNN from
  `../Tangent_Vector/results/best_knn_cfg.json` so the PCA-KNN baseline
  uses the **same R and same KNN as ES-VAE / Vanilla VAE** in each CV mode.
- `sequence_clf.py` — TCN / LSTM / Transformer / STGCN on the
  `(N, 75, T)` raw-skeleton sequence. Architectures and capacities are
  copied verbatim from `../Tangent_Vector/sequence_clf.py` for a like-for-like
  comparison.
- `vae_clf.py` — Vanilla MSE-loss VAE on flattened raw-skeleton sequences,
  followed by KNN on latent means. The encoder topology is **identical to
  ES-VAE's `NonlinearVAE`** (R=48, hidden=768, 2-layer tanh MLP, dropout=0.10,
  150 epochs, KL-annealed) — the **only** difference is the reconstruction
  loss: plain MSE on raw coords vs squared geodesic distance on Kendall
  preshape space. When run without `--sweep`, the encoder hyperparameters
  and KNN config are auto-matched to the ES-VAE selection.

### Outputs (`results/`)
- Per-CV-mode metrics with suffix `{,_xview,_xsetup}`:
  - `pca_clf_metrics*.csv`, `pca_clf_classwise_best*.csv`, `pca_clf_oof*.json`
  - `sequence_clf_metrics*.csv`, `sequence_clf_classwise_best*.csv`
  - `vae_clf_metrics*.csv`, `vae_clf_classwise*.csv`, `vae_clf_config*.json`
- Logs: `*.log`.

## How to run

From `Raw_Skeleton/` (after Phase-1 sweeps in `../Tangent_Vector/` have
locked the ES-VAE config):

```bash
# PCA classical baselines (auto-matches R + KNN to ES-VAE selection)
python pca_clf.py --cv-mode subject     # then --cv-mode view, setup

# Vanilla VAE + KNN (auto-matches encoder + KNN to ES-VAE config)
python vae_clf.py --cv-mode subject

# Sequence baselines
python sequence_clf.py --cv-mode subject
```

## Headline comparison (cross-subject, L5SO, 8 folds)

Pooled across 8 L5SO folds. Cells show **mean [95% CI]** from the
2000-iter subject bootstrap. Sorted by Macro-F1.

| Method | Macro-F1 | Macro Precision | Macro Recall |
|---|---|---|---|
| PCA-KNN (R=48, k=5 distance) | **0.483 [0.438, 0.525]** | 0.545 [0.491, 0.604] | 0.495 [0.453, 0.537] |
| Transformer | 0.333 [0.286, 0.379] | 0.339 [0.284, 0.404] | 0.360 [0.318, 0.405] |
| Vanilla VAE + KNN | 0.265 [0.223, 0.308] | 0.262 [0.221, 0.311] | 0.270 [0.230, 0.312] |
| TCN | 0.210 [0.169, 0.251] | 0.227 [0.178, 0.285] | 0.228 [0.188, 0.268] |
| LSTM | 0.197 [0.163, 0.231] | 0.209 [0.162, 0.258] | 0.215 [0.185, 0.245] |
| STGCN | 0.105 [0.079, 0.129] | 0.132 [0.082, 0.179] | 0.110 [0.085, 0.135] |

PCA-KNN takes the headline because the linear baseline is the only model
that gets reasonable mileage out of raw 3D coordinates. Transformer is
the best NN; STGCN collapses (raw skeletons preserve world translation
which the small ST-GCN cannot normalise away). The Vanilla VAE on raw
3D coords sits well below all of these — exactly the contrast we use to
isolate the manifold prior in the ES-VAE comparison below.

## ES-VAE vs Vanilla VAE — the manifold-loss isolation test

Same encoder topology (2-layer tanh MLP, R=48, hidden=768, dropout=0.10,
150 epochs, KL-annealed), same KNN downstream classifier (k=5, distance).
The **only** difference is the reconstruction loss / target:

| Pipeline | Reconstruction target | Loss | Pooled Macro-F1 |
|---|---|---|---:|
| ES-VAE on tangent vectors | manifold curve via `exp_map(mu, v_hat)` | squared geodesic distance on Kendall preshape space | **0.557 [0.516, 0.598]** |
| Vanilla VAE on raw skeletons | flattened raw sequence | plain MSE | **0.265 [0.223, 0.308]** |

ES-VAE outperforms by **+0.292 Macro-F1** under matched architecture,
matched optimizer, matched KNN — attributable entirely to (i) the
Kendall preshape representation (translation-/scale-/rotation-invariant
curves, time-warp-normalised by SRVF) and (ii) the geodesic reconstruction
loss that respects that manifold structure. The unconstrained MSE-on-raw
VAE cannot absorb the nuisance variation that the manifold pipeline
factors out analytically.

## Tangent vs Raw — same-method comparison (subject CV)

Same architectures, same fold splits — only the input differs (Kendall
tangent vector vs linearly-resampled raw skeleton).

| Method | Tangent (Macro-F1) | Raw (Macro-F1) | Δ |
|---|---:|---:|---:|
| ES-VAE / Vanilla VAE | 0.557 | 0.265 | **+0.292** |
| PCA-KNN | 0.498 | 0.483 | +0.015 |
| TCN | 0.390 | 0.210 | +0.180 |
| LSTM | 0.379 | 0.197 | +0.182 |
| Transformer | 0.442 | 0.333 | +0.109 |
| STGCN | 0.411 | 0.105 | **+0.306** |

Take-aways:
- The **VAE-style encoder** depends critically on the manifold prior — raw
  input drops Macro-F1 by **0.29** even with identical architecture and KNN.
- **STGCN** is the most position-sensitive: it loses **0.31** going from
  tangent (centred per frame) to raw (with world offsets). On tangent
  vectors STGCN is competitive with the other sequence models; on raw
  it collapses.
- **TCN/LSTM** drop ~0.18; **Transformer** absorbs more of the nuisance
  variation through attention but still leaves ~0.11 on the table.
- **PCA-KNN** is statistically tied (the linear projection captures
  comparable structure on either input) — neither representation gives
  the linear baseline a meaningful edge.

The same Δ pattern holds under cross-view and cross-setup CV (every TV
method beats its RS counterpart in all three protocols, 18/18 head-to-head
wins). See `../results_tables_top10.md` for the full subject and view
tables in NeurIPS-paper format.

## Classwise breakdown — Vanilla VAE (manifold-loss ablation)

`sklearn.metrics.classification_report` on pooled OOF predictions of the
Vanilla VAE (the matched-architecture baseline for ES-VAE) across all 8
L5SO folds. Source: `results/vae_clf_classwise.csv`.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| A001 drink water | 0.242 | 0.200 | 0.219 | 40 |
| A002 eat meal | 0.114 | 0.100 | 0.107 | 40 |
| A003 brush teeth | 0.265 | 0.225 | 0.243 | 40 |
| A004 brush hair | 0.160 | 0.200 | 0.178 | 40 |
| A028 phone call | 0.158 | 0.150 | 0.154 | 40 |
| A029 play with phone | 0.146 | 0.150 | 0.148 | 40 |
| A008 sitting down | 0.510 | 0.650 | 0.571 | 40 |
| A009 standing up | 0.442 | 0.475 | 0.458 | 40 |
| A023 hand waving | 0.270 | 0.250 | 0.260 | 40 |
| A031 pointing | 0.316 | 0.300 | 0.308 | 40 |
| accuracy |  |  | 0.270 | 400 |
| macro avg | 0.262 | 0.270 | 0.265 | 400 |

Even on raw skeletons, the 4 whole-body anchors get a small lift
(F1 ≈ 0.40); the 6 hand-to-face classes are essentially at chance
(F1 ≈ 0.18 vs 10-class chance of 0.10). The manifold prior is doing
the discriminative work for trajectory-shape classes.

## Notes

- The L5SO / view / setup partitions are identical to `Tangent_Vector/`,
  so per-method comparisons across the two folders are matched-pair valid.
- Per-fold prints are in `*.log` files.
- Sequence baselines complete in ~3–5 min on an A5000 (cuda:0); PCA-KNN
  full bootstrap takes ~1 min; Vanilla VAE single-config ~3 min on cuda:1.
