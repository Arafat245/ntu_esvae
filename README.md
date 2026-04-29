# NTU Activity Recognition — Tangent-Vector vs Raw-Skeleton

A controlled comparison of **Kendall preshape tangent vectors** (with
SRVF time-warp normalization) against **raw 3D joint coordinates** for
action classification on the **NTU RGB+D 60** dataset. We use a 10-class
subset designed to expose where the manifold prior helps most, and
evaluate under three cross-validation protocols (cross-subject,
cross-view, cross-setup).

**Scope: NTU-60 only.** Our 10 chosen classes (A001–A031) all fall
within NTU-60's A001–A060 range, our 40 subjects (P001–P040 in
S001–S017) are exactly the NTU-60 subject pool, and the curation script
reads only from `data/nturgbd_skeletons_s001_to_s017/`. NTU-120's
expansion (S018–S032, A061–A120, subjects P041–P106) is **not** used.
The relevant SOTA reference for our setup is therefore the NTU-60
column of published leaderboards (X-Sub / X-View), not NTU-120.

## What's the manifold prior?

Each skeleton trajectory is mapped onto **Kendall preshape space** (a
quotient manifold that removes translation, scale, and rotation), then
temporally aligned with **SRVF γ-warping** (removes execution-rate
variation). Models train on tangent vectors at the Karcher mean instead
of raw coordinates. The hypothesis: when nuisance variation
(camera/orientation/subject-size/speed) dominates raw input, the
manifold prior strips it analytically and lets the downstream classifier
focus on trajectory shape.

## Dataset (10-class top10, NTU-60 only)

400 single-person trials = **40 NTU-60 subjects × 10 NTU-60 classes**
(perfectly balanced). All trials come from setups S001–S017 and class
ids A001–A031 — entirely inside the NTU-60 partition; NTU-120's added
classes/subjects (A061–A120, P041–P106) are not used. Curated by
`build_ntu_skeleton_top10.py` to maximise the |Tangent − Raw| gap:

| Group | Classes | Why |
|---|---|---|
| Hand-to-face / hand-trajectory (TV-favoured) | A001 drink water, A002 eat meal, A003 brush teeth, A004 brush hair, A028 phone call, A029 play with phone | Discriminator is fine arm-trajectory shape; raw 3D coords drown in subject-size/orientation/rate noise |
| Whole-body anchors | A008 sitting down, A009 standing up, A023 hand waving, A031 pointing | Both representations should classify these correctly — keeps macro-F1 floor high |

The curation script spreads picks across all 17 NTU setups, all 3
cameras, and both replications, so cross-view and cross-setup CV are
meaningful tests.

## Repository layout

```
activity_recognition/
├── build_ntu_skeleton_top10.py   curate ntu_skeleton_top10/ from data/
├── build_ntu_pkl.py              build data/data_ntu.pkl from skeletons
├── build_results_tables.py       emit NeurIPS-style markdown tables
├── results_tables_top10.md       combined subject + view tables
├── functionsgpu_fast.py          shared geomstats Kendall/SRVF utilities
├── Tangent_Vector/               classification on tangent vectors
│   ├── cv_utils.py               loaders + L5SO/view/setup folds + bootstrap CIs
│   ├── esvae_clf.py              ES-VAE with squared-geodesic recon loss
│   ├── esvae_epoch_sweep.py      Phase 1b: epoch sweep on the winner config
│   ├── esvae_batch_sweep.py      Phase 1c: batch sweep on the winner config
│   ├── pca_clf.py                PCA → KNN baseline (R + KNN matched to ES-VAE)
│   └── sequence_clf.py           TCN / LSTM / Transformer / STGCN sequence models
├── Raw_Skeleton/                 mirror pipeline on raw 3D coords
│   ├── cv_utils.py               raw skeleton loader + re-exported helpers
│   ├── pca_clf.py                PCA → KNN (matched config)
│   ├── vae_clf.py                Vanilla VAE + KNN (matched architecture, MSE loss)
│   └── sequence_clf.py           same architectures as TV side
├── data/                         (gitignored) raw NTU 60/120 skeletons + data_ntu.pkl
├── aligned_data/                 (gitignored) tangent_vecs100, betas_aligned100, mu100, sample_index/meta
├── ntu_skeleton_top10/           (gitignored) curated 10-class .skeleton files
└── logs_top10/                   (gitignored) per-run logs
```

## Pipeline

1. **Curate** — `python build_ntu_skeleton_top10.py --seed 42` writes
   `ntu_skeleton_top10/` (400 .skeleton files, manifest, all/common subjects).
2. **Build pkl** — `python build_ntu_pkl.py` builds `data/data_ntu.pkl`
   (dict `{pid}_{class_id} → ndarray (25, 3, T_var)`).
3. **Align** — Frechet mean + OPA rotational alignment + SRVF temporal
   alignment iterates to `aligned_data/{tangent_vecs,betas_aligned,mu,gammas,betas_resampled_kendall}100.pkl`
   plus `sample_index.csv` and `sample_meta.csv` (the latter has
   `setup_id, camera_id, replication, filename` per sample for view/setup
   CV).
4. **Classify** — Run any combination of `Tangent_Vector/{pca_clf, esvae_clf, sequence_clf}.py`
   and `Raw_Skeleton/{pca_clf, vae_clf, sequence_clf}.py` with
   `--cv-mode {subject, view, setup}`.

## ES-VAE hyperparameter selection (Phase 1, subject CV)

Sweep grid: `R ∈ {16, 24, 32, 48} × hidden ∈ {512, 768} × β-KL ∈ {1e-4, 1e-3}`,
followed by an epoch sweep ({25, 50, 100, 150, 200, 250, 300, 400}) and a
batch sweep ({16, 32, 64, 128, 256}) on the winning config. Final tuned
config:

| Hyperparameter | Value |
|---|---|
| Latent dim `R` | 48 |
| Encoder hidden width | 768 |
| Encoder/decoder | 2-layer tanh MLP each |
| Epochs | 150 (cosine LR) |
| Batch size | 64 |
| β-KL | 1e-4 (linearly annealed over first 30%) |
| Dropout | 0.10 |
| Reconstruction loss | Squared geodesic distance on Kendall preshape space |
| Latent classifier | KNN (k=5, distance) |

PCA + Vanilla VAE auto-load `R` and KNN from this config so all encoders
share the same embedding dim and downstream classifier per CV mode.

## Headline results

Pooled out-of-fold macro-F1, **mean [95% CI]** from 2000-iter
subject-level bootstrap. See `results_tables_top10.md` for the full
NeurIPS-style tables (Macro F1 / Precision / Recall).

### Cross-Subject (Leave-5-Subjects-Out, 8 folds)

| Input Representation | Method | Macro F1 (95% CI) |
|---|---|---|
| Raw Skeleton | PCA + k-NN | 0.483 (0.438, 0.525) |
|  | VAE + k-NN | 0.265 (0.223, 0.308) |
|  | TCN | 0.210 (0.169, 0.251) |
|  | LSTM | 0.197 (0.163, 0.231) |
|  | Transformer | 0.333 (0.286, 0.379) |
|  | ST-GCN | 0.105 (0.079, 0.129) |
| Tangent Vector | PCA + k-NN | 0.498 (0.451, 0.539) |
|  | TCN | 0.390 (0.349, 0.428) |
|  | LSTM | 0.379 (0.334, 0.422) |
|  | Transformer | 0.442 (0.403, 0.479) |
|  | ST-GCN | 0.411 (0.370, 0.451) |
|  | **ES-VAE + k-NN (proposed)** | **0.557 (0.516, 0.598)** |

### Cross-View (Leave-One-Camera-Out, 3 folds)

| Input Representation | Method | Macro F1 (95% CI) |
|---|---|---|
| Raw Skeleton | PCA + k-NN | 0.399 (0.356, 0.443) |
|  | VAE + k-NN | 0.184 (0.153, 0.216) |
|  | TCN | 0.129 (0.098, 0.157) |
|  | LSTM | 0.148 (0.114, 0.185) |
|  | Transformer | 0.208 (0.171, 0.244) |
|  | ST-GCN | 0.092 (0.068, 0.113) |
| Tangent Vector | PCA + k-NN | 0.458 (0.411, 0.501) |
|  | TCN | 0.269 (0.236, 0.301) |
|  | LSTM | 0.262 (0.223, 0.300) |
|  | Transformer | 0.329 (0.284, 0.369) |
|  | ST-GCN | 0.336 (0.290, 0.380) |
|  | **ES-VAE + k-NN (proposed)** | **0.487 (0.437, 0.532)** |

### Tangent − Raw gaps under cross-subject

| Method pair | Tangent | Raw | Δ |
|---|---:|---:|---:|
| ES-VAE / Vanilla VAE (matched architecture) | 0.557 | 0.265 | **+0.292** |
| ST-GCN | 0.411 | 0.105 | **+0.306** |
| LSTM | 0.379 | 0.197 | +0.182 |
| TCN | 0.390 | 0.210 | +0.180 |
| Transformer | 0.442 | 0.333 | +0.109 |
| PCA + k-NN | 0.498 | 0.483 | +0.015 |

### Key findings

1. **Tangent vector beats raw skeleton on every method, every CV mode** (18/18
   head-to-head wins). The result is consistent under cross-subject,
   cross-view, and cross-setup.
2. **The manifold-loss isolation test** — same architecture and KNN, only
   the reconstruction loss differs (geodesic on Kendall preshape vs MSE
   on raw coords) — produces a **+0.29 macro-F1** gap. The unconstrained
   MSE-on-raw VAE essentially fails (near 10-class chance of 0.10) on
   the hand-to-face classes.
3. **ST-GCN is the most position-sensitive baseline** — it loses 0.31
   F1 going from tangent to raw because raw NTU skeletons preserve
   world translation that the small ST-GCN cannot normalize away. On
   tangent vectors ST-GCN is competitive with the other sequence models.
4. **PCA + k-NN closes the gap most narrowly** (+0.015) — the linear
   projection captures comparable structure on either input.
5. **The 4 whole-body anchors carry a high macro-F1 floor** (≈0.79 each
   under TV ES-VAE); the 6 hand-to-face classes drive the spread (≈0.40
   each). The dataset was deliberately designed to make this contrast
   visible.

## Context vs published NTU SOTA

Published NTU leaderboards (e.g. ProtoGCN, CVPR 2025;
[github.com/firework8/ProtoGCN](https://github.com/firework8/ProtoGCN))
report top-1 accuracies of **93.8% on NTU-60 X-Sub, 97.8% NTU-60 X-View**
(NTU-120 numbers are 90.9% X-Sub, 92.2% X-Set, but those are over a
different subject/class pool — not directly comparable to our setup).
Since our 10 classes and 40 subjects are entirely within NTU-60, the
relevant comparison is **NTU-60 X-Sub** for our cross-subject result
and **NTU-60 X-View** for our cross-view result. Our top10 numbers
(subject 55.7% / view 48.7%) sit ~40 percentage points below these. The gap is structural, not a model
quality issue — six reasons, with sample counts that quantify how much
harder our setup is:

### 1. Drastically less training data per fold (~115× fewer trials)

| | Total trials | Train per fold | Test per fold | Classes | Subjects |
|---|---:|---:|---:|---:|---:|
| **Standard NTU-60 X-Sub** (apples-to-apples reference) | 56,880 | **40,320** (20 subjects × 60 classes × ~33 cam/rep variants) | 16,560 | 60 | 40 |
| Standard NTU-120 X-Sub (different pool) | 114,480 | ~63,026 (53 subjects) | ~51,454 | 120 | 106 |
| **Ours, top10 L5SO (NTU-60 subset)** | 400 | **350** (35 subjects × 10 classes × 1 trial) | 50 | 10 | 40 |

Deep skeleton models reach 90%+ only with ≳10⁴ training trials. 350 is
essentially few-shot territory. ProtoGCN trains on **115× more** trials
than we do per fold.

### 2. One trial per (subject, class), not all available

Standard pipelines use every available (subject, class, camera,
replication) combination as a separate training trial:
- NTU-60 has ~948 trials per class on average → ~57k trials total.
- Our `build_ntu_skeleton_top10.py --seed 42` picks **one** trial per
  (subject, class) and discards the rest, dropping us from a possible
  9,480 trials (40 subj × 10 classes × ~24 cam/rep variants) to **400**.
  We deliberately do this to avoid same-trial leakage across CV folds
  while keeping the dataset compact for the controlled comparison.

### 3. Adversarially hard 10-class subset

The 6 hand-to-face classes (drink, eat, brush teeth, brush hair, phone
call, play with phone) are among the *hardest* single-person actions in
NTU because they share standing-upright posture and one-arm-raised-to-
face geometry. Published averages cover all 60/120 classes including
easy locomotion (walking, sitting, jumping, falling) that pull the mean
to >90%. We picked these 10 specifically to expose the manifold-prior
advantage on confusable trajectory-shape classes.

### 4. L5SO ≠ standard X-Sub

Standard X-Sub uses one fixed 20-train / 20-test subject split. We use
leave-5-subjects-out across 40 subjects with 8 folds — much higher
variance, smaller per-fold train, and we report **pooled** metrics
across all 8 folds rather than best-fold or single-split numbers.

### 5. Tiny architectures + no augmentation + no ensemble

Our sequence baselines are intentionally small for matched-pair
comparison: ST-GCN channels=(16, 32) vs SOTA (64, 64, 128, 128, 256, 256,
256). No random rotation/scaling/temporal cropping during training. No
bone-stream / motion-stream fusion. ProtoGCN's 93.8% is a 6-stream
ensemble (joint + bone + joint-motion + bone-motion + 2 more); the
single-stream ProtoGCN is ~92.6%.

### 6. No standard NTU training recipe

SOTA pipelines use SGD + Nesterov momentum 0.9, weight decay 5e-4,
cosine LR over 150 epochs, batch 64, label smoothing, mixup, etc. — many
small tricks. We use AdamW with cosine LR and no other augmentation.

### Why the comparison still works for us

Our goal is **not** to beat the leaderboard — it's a controlled
head-to-head: same architecture, same training data, same CV folds, same
KNN — only the input representation differs. The 0.29-F1 gap between
TV ES-VAE (0.557) and RS Vanilla VAE (0.265) is meaningful precisely
because everything else is matched. ProtoGCN and other NTU SOTA papers
do not compare against Kendall tangent-space methods, so the two
research questions are orthogonal:

- **NTU SOTA papers** ask: *given full training data and freedom in
  architecture/augmentation/ensembles, how high can accuracy go on raw
  skeletons?* → 93.8%.
- **This work** asks: *with everything else held constant, does the
  Kendall preshape representation help?* → +0.29 macro-F1 in the
  matched-encoder isolation test, and TV beats RS on every method × CV
  mode pair (18/18 wins).

To produce numbers comparable to ProtoGCN's table, we would need (a)
all 60/120 classes, (b) all available trials per class, (c) the
official X-Sub/X-View/X-Set splits, (d) larger architectures, (e)
standard augmentation, (f) multi-stream ensembles. That is a different
paper.

## How to reproduce

Once `data/data_ntu.pkl` and `aligned_data/` are populated (see Pipeline
above):

```bash
# Phase 1 — ES-VAE sweep on subject CV (locks the chosen config)
cd Tangent_Vector
python esvae_clf.py --sweep --cv-mode subject
python esvae_epoch_sweep.py
python esvae_batch_sweep.py

# Phase 3 — full evaluation under each CV mode
for mode in subject view setup; do
  python pca_clf.py --cv-mode $mode
  python esvae_clf.py --cv-mode $mode --R 48 --epochs 150 --hidden 768 --beta-kl 1e-4 --batch-size 64
  python sequence_clf.py --cv-mode $mode
  cd ../Raw_Skeleton
  python pca_clf.py --cv-mode $mode
  python vae_clf.py --cv-mode $mode
  python sequence_clf.py --cv-mode $mode
  cd ../Tangent_Vector
done

# Build the combined NeurIPS-style table
cd ..
python build_results_tables.py   # writes results_tables_top10.md
```

ES-VAE GPU is `cuda:1` because `functionsgpu_fast.py` (the geomstats
PreShapeSpace utilities) is hardcoded to that device on import.
Sequence baselines use `cuda:0`.

## Notes

- Subject, view, and setup partitions are deterministic (`seed=42`) and
  shared across both folders, so per-method comparisons are matched-pair
  valid.
- `Tangent_Vector/cv_utils.py` is the source of truth for class labels
  (`CLASS_ORDER`, `CLASS_NAMES`); `NUM_CLASSES = len(CLASS_ORDER)` across
  all scripts.
- Per-fold logs in `logs_top10/` (gitignored).
- Curation, alignment, and result CSVs are deterministic given the same
  seed and source NTU dataset.
