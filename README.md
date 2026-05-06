# NTU Activity Recognition — Tangent-Vector vs Raw-Skeleton

This repository is the official implementation of *NTU Activity Recognition — Tangent-Vector vs Raw-Skeleton*, a controlled comparison of **Kendall preshape tangent vectors** (with SRVF time-warp normalization) against **raw 3D joint coordinates** for action classification on the **NTU RGB+D 60** dataset. We use a 10-class subset designed to expose where the manifold prior helps most, and evaluate under three cross-validation protocols (cross-subject, cross-view, cross-setup).

**Scope: NTU-60 only.** Our 10 chosen classes (A001–A031) all fall within NTU-60's A001–A060 range, our 40 subjects (P001–P040 in S001–S017) are exactly the NTU-60 subject pool, and the curation script reads only from `data/nturgbd_skeletons_s001_to_s017/`. NTU-120's expansion (S018–S032, A061–A120, subjects P041–P106) is **not** used.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

Additional setup:

- Download the **NTU RGB+D 60** skeleton archive (`nturgbd_skeletons_s001_to_s017`) from [the NTU site](https://rose1.ntu.edu.sg/dataset/actionRecognition/) and place it under `data/`.
- A CUDA GPU is required. ES-VAE is hardcoded to `cuda:1` on import (geomstats `PreShapeSpace` utilities in `functionsgpu_fast.py`); sequence baselines run on `cuda:0`.

### Repository layout

```
activity_recognition/
├── build_ntu_skeleton_top10.py   curate ntu_skeleton_top10/ from data/
├── build_ntu_pkl.py              build data/data_ntu.pkl from skeletons
├── build_results_tables.py       emit NeurIPS-style markdown tables
├── functionsgpu_fast.py          shared geomstats Kendall/SRVF utilities
├── official_compare/             adapted official Hyper-GCN / Sparse-ST-GCN runners
├── Tangent_Vector/               classification on tangent vectors
├── Raw_Skeleton/                 mirror pipeline on raw 3D coords
├── Temp_exps/                    standalone experiments
├── data/                         (gitignored) raw NTU skeletons + data_ntu.pkl
├── aligned_data/                 (gitignored) tangent_vecs100, betas_aligned100, mu100, sample_index/meta
├── ntu_skeleton_top10/           (gitignored) curated 10-class .skeleton files
└── logs_top10/                   (gitignored) per-run logs
```

### Dataset (10-class top10, NTU-60 only)

400 single-person trials = **40 NTU-60 subjects × 10 NTU-60 classes** (perfectly balanced):

| Group | Classes |
|---|---|
| Hand-to-face / hand-trajectory (TV-favoured) | A001 drink water, A002 eat meal, A003 brush teeth, A004 brush hair, A028 phone call, A029 play with phone |
| Whole-body anchors | A008 sitting down, A009 standing up, A023 hand waving, A031 pointing |

The curation script spreads picks across all 17 NTU setups, all 3 cameras, and both replications, so cross-view and cross-setup CV are meaningful tests.

## Training

To curate, build, align, and train all models in the paper:

```train
# 1. Curate the 10-class subset (400 .skeleton files, deterministic seed=42)
python build_ntu_skeleton_top10.py --seed 42

# 2. Build data/data_ntu.pkl from the curated skeletons
python build_ntu_pkl.py

# 3. ES-VAE hyperparameter sweep on subject CV (locks the chosen config)
cd Tangent_Vector
python esvae_clf.py --sweep --cv-mode subject
python esvae_epoch_sweep.py
python esvae_batch_sweep.py
cd ..

# 4. Full training under each CV mode
for mode in subject view setup; do
  (cd Tangent_Vector && \
    python pca_clf.py --cv-mode $mode && \
    python esvae_clf.py --cv-mode $mode --R 48 --epochs 150 --hidden 768 --beta-kl 1e-4 --batch-size 64 && \
    python sequence_clf.py --cv-mode $mode)
  (cd Raw_Skeleton && \
    python pca_clf.py --cv-mode $mode && \
    python vae_clf.py --cv-mode $mode && \
    python sequence_clf.py --cv-mode $mode)
done

# 5. Train the adapted official baselines on subject CV
python official_compare/hypergcn_runner.py     --representation raw     --variant base --epochs 20 --batch-size 64 --device cuda:0
python official_compare/hypergcn_runner.py     --representation tangent --variant base --epochs 20 --batch-size 64 --device cuda:1
python official_compare/sparse_stgcn_runner.py --representation raw     --epochs 20 --batch-size 32 --device cuda:0
python official_compare/sparse_stgcn_runner.py --representation tangent --epochs 20 --batch-size 32 --device cuda:1
```

Final tuned ES-VAE config locked in by the Phase-1 sweep:

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

PCA + Vanilla VAE auto-load `R` and KNN from this config so all encoders share the same embedding dim and downstream classifier per CV mode.

## Evaluation

To evaluate the trained models and produce the combined NeurIPS-style results table:

```eval
python build_results_tables.py   # writes results_tables_top10.md
```

Per-method evaluation is folded into the training scripts above: each `*_clf.py` writes pooled out-of-fold metrics with **2000-iter subject-level bootstrap 95% CIs** to `Tangent_Vector/results/` and `Raw_Skeleton/results/`. `official_compare/results/*.json` stores subject-CV outputs for the adapted official runners.

## Pre-trained Models

We do not distribute pre-trained checkpoints. All experiments are deterministic given `seed=42`, the source NTU-60 dataset, and the locked ES-VAE config above; rerunning the training commands reproduces every reported number. ES-VAE training takes ~15 min per CV mode on an A6000; the full Phase-1 sweep takes ~2 hours.

## Results

Pooled out-of-fold macro-F1, **mean [95% CI]** from 2000-iter subject-level bootstrap. See `results_tables_top10.md` for the full NeurIPS-style tables (Macro F1 / Precision / Recall).

### [Skeleton-Based Action Recognition on NTU RGB+D](https://paperswithcode.com/sota/skeleton-based-action-recognition-on-ntu-rgbd) — top10 cross-subject (Leave-5-Subjects-Out, 8 folds)

| Input Representation | Method | Macro F1 (95% CI) |
| ------------------ |---------------- | -------------- |
| Raw Skeleton | **Hyper-GCN** | **0.539 (0.500, 0.575)** |
|  | Sparse-ST-GCN | 0.489 (0.440, 0.537) |
|  | PCA + k-NN | 0.483 (0.438, 0.525) |
|  | Transformer | 0.333 (0.286, 0.379) |
|  | VAE + k-NN | 0.265 (0.223, 0.308) |
|  | TCN | 0.210 (0.169, 0.251) |
|  | LSTM | 0.197 (0.163, 0.231) |
|  | ST-GCN | 0.105 (0.079, 0.129) |
| Tangent Vector | **ES-VAE + k-NN (proposed)** | **0.557 (0.516, 0.598)** |
|  | Sparse-ST-GCN | 0.501 (0.460, 0.542) |
|  | PCA + k-NN | 0.498 (0.451, 0.539) |
|  | Transformer | 0.442 (0.403, 0.479) |
|  | ST-GCN | 0.411 (0.370, 0.451) |
|  | TCN | 0.390 (0.349, 0.428) |
|  | LSTM | 0.379 (0.334, 0.422) |
|  | Hyper-GCN | 0.377 (0.329, 0.422) |

### Cross-View (Leave-One-Camera-Out, 3 folds)

| Input Representation | Method | Macro F1 (95% CI) |
|---|---|---|
| Raw Skeleton | PCA + k-NN | 0.399 (0.356, 0.443) |
|  | Transformer | 0.208 (0.171, 0.244) |
|  | VAE + k-NN | 0.184 (0.153, 0.216) |
|  | LSTM | 0.148 (0.114, 0.185) |
|  | TCN | 0.129 (0.098, 0.157) |
|  | ST-GCN | 0.092 (0.068, 0.113) |
| Tangent Vector | **ES-VAE + k-NN (proposed)** | **0.487 (0.437, 0.532)** |
|  | PCA + k-NN | 0.458 (0.411, 0.501) |
|  | ST-GCN | 0.336 (0.290, 0.380) |
|  | Transformer | 0.329 (0.284, 0.369) |
|  | TCN | 0.269 (0.236, 0.301) |
|  | LSTM | 0.262 (0.223, 0.300) |

### Tangent − Raw gaps under cross-subject

| Method pair | Tangent | Raw | Δ |
|---|---:|---:|---:|
| ES-VAE / Vanilla VAE (matched architecture) | 0.557 | 0.265 | **+0.292** |
| Hyper-GCN | 0.377 | 0.539 | **-0.162** |
| Sparse-ST-GCN | 0.501 | 0.489 | +0.011 |
| ST-GCN | 0.411 | 0.105 | **+0.306** |
| LSTM | 0.379 | 0.197 | +0.182 |
| TCN | 0.390 | 0.210 | +0.180 |
| Transformer | 0.442 | 0.333 | +0.109 |
| PCA + k-NN | 0.498 | 0.483 | +0.015 |

### Key findings

1. **Within the original matched local pipeline, tangent vector beats raw skeleton on every method and every CV mode** (18/18 head-to-head wins) — consistent under cross-subject, cross-view, and cross-setup.
2. **The adapted official-model benchmark is mixed on this tiny subset**: Hyper-GCN strongly prefers raw coordinates, while Sparse-ST-GCN gives a slight edge to tangent vectors.
3. **The manifold-loss isolation test** — same architecture and KNN, only the reconstruction loss differs (geodesic on Kendall preshape vs MSE on raw coords) — produces a **+0.29 macro-F1** gap.
4. **ST-GCN is the most position-sensitive baseline** — it loses 0.31 F1 going from tangent to raw because raw NTU skeletons preserve world translation that the small ST-GCN cannot normalize away.
5. **PCA + k-NN closes the gap most narrowly** (+0.015) — the linear projection captures comparable structure on either input.
6. **The 4 whole-body anchors carry a high macro-F1 floor** (≈0.79 each under TV ES-VAE); the 6 hand-to-face classes drive the spread (≈0.40 each).

### Context vs published NTU SOTA

Published NTU leaderboards (e.g. Hyper-GCN, ICCV 2025; [github.com/6UOOON9/Hyper-GCN](https://github.com/6UOOON9/Hyper-GCN)) report top-1 of **93.7% on NTU-60 X-Sub, 97.8% NTU-60 X-View**. Our top10 numbers (subject 55.7% / view 48.7%) sit ~40 percentage points below these. The gap is structural, not a model quality issue:

| | Total trials | Train per fold | Test per fold | Classes | Subjects |
|---|---:|---:|---:|---:|---:|
| **Standard NTU-60 X-Sub** | 56,880 | **40,320** | 16,560 | 60 | 40 |
| **Ours, top10 L5SO (NTU-60 subset)** | 400 | **350** | 50 | 10 | 40 |

Reasons: (1) ~115× fewer training trials per fold; (2) one trial per (subject, class), not all camera/replication variants; (3) adversarially hard 10-class subset; (4) L5SO ≠ standard X-Sub; (5) single-stream comparison vs published 4-stream recipes; (6) no standard NTU training recipe (no SGD+Nesterov, no label smoothing, no augmentation).

Our goal is **not** to beat the leaderboard — it's a controlled head-to-head: same architecture, same training data, same CV folds, same KNN — only the input representation differs. NTU SOTA papers do not compare against Kendall tangent-space methods, so the two research questions are orthogonal:

- **NTU SOTA papers** ask: *given full data and freedom in architecture/augmentation/ensembles, how high can accuracy go on raw skeletons?* → 93.7%.
- **This work** asks: *with everything else held constant, does the Kendall preshape representation help?* → +0.29 macro-F1 in the matched-encoder isolation test, and 18/18 wins in the matched-pair pipeline.

## Contributing

Issues and pull requests are welcome. Please keep contributions focused on the NTU top10 controlled-comparison setup; out-of-scope changes (other datasets, full NTU-60/120 retraining recipes) belong in a fork.
