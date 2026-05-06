# Official-Model Comparison on the NTU Top10 Subset

This folder adapts the **official GitHub implementations** of two recent NTU-specialised backbones to the local **400-sample NTU-60 top10 subset** used elsewhere in this repo:

- **Hyper-GCN** (ICCV 2025)
- **Sparse-ST-GCN** (CVPR 2025)

What is preserved from the official repos: backbone architecture, graph definition, head / loss logic, clip sampling format, single-stream NTU joint-input setting.

What is adapted locally: dataset loader reads `../data/data_ntu.pkl` (raw skeletons) and `../aligned_data/tangent_vecs100.pkl` (tangent vectors); class count is reduced from `60` to `10`; folds reuse the repo's deterministic **8 leave-5-subjects-out** partition from `../Tangent_Vector/cv_utils.py`; training harness is lightweight PyTorch instead of the original PYSKL/MMCV experiment stack.

## Requirements

Install dependencies from the repo root:

```setup
pip install -r ../requirements.txt
```

Inputs (gitignored, populated by the alignment pipeline at the repo root):

- `../data/data_ntu.pkl` — raw skeletons (400 single-person trials, dict `{pid}_{class_id} → ndarray (25, 3, T_var)`).
- `../aligned_data/tangent_vecs100.pkl` — Kendall tangent vectors at the Karcher mean.

### Files

- `common.py` — data loading, sampling, metrics, graph helper
- `hypergcn_runner.py` — adapted Hyper-GCN subject-CV runner
- `sparse_stgcn_runner.py` — adapted Sparse-ST-GCN subject-CV runner
- `results/*.json` — final subject-CV outputs for raw and tangent runs

## Training

From repo root:

```train
python official_compare/hypergcn_runner.py     --representation raw     --variant base --epochs 20 --batch-size 64 --device cuda:0
python official_compare/hypergcn_runner.py     --representation tangent --variant base --epochs 20 --batch-size 64 --device cuda:1
python official_compare/sparse_stgcn_runner.py --representation raw     --epochs 20 --batch-size 32 --device cuda:0
python official_compare/sparse_stgcn_runner.py --representation tangent --epochs 20 --batch-size 32 --device cuda:1
```

## Evaluation

Both runners perform subject-CV evaluation in-line and write pooled-OOF metrics (top-1, macro-F1, 95% CIs) to `results/*.json`:

```eval
ls official_compare/results/
```

## Pre-trained Models

No checkpoints are distributed. All runs are deterministic given `seed=42` and the published official model definitions (with the local subset and harness adaptations described above).

## Results

### Final subject-CV results

| Model | Input | Top-1 | Macro-F1 | 95% CI |
|---|---|---:|---:|---:|
| Hyper-GCN | Raw skeleton | **0.5475** | **0.5389** | [0.4999, 0.5754] |
| Hyper-GCN | Tangent vector | 0.3800 | 0.3768 | [0.3290, 0.4216] |
| Sparse-ST-GCN | Raw skeleton | 0.5075 | 0.4893 | [0.4397, 0.5367] |
| Sparse-ST-GCN | Tangent vector | **0.5150** | **0.5006** | [0.4595, 0.5421] |

The result is mixed: Hyper-GCN is a useful recent raw-skeleton baseline but drops sharply on tangent vectors on this tiny subset, while Sparse-ST-GCN gives a slight edge to tangent vectors.

## Contributing

These runners deliberately stay close to the upstream architecture/loss code so that comparisons remain meaningful. Architectural changes belong upstream (see [Hyper-GCN](https://github.com/6UOOON9/Hyper-GCN) and the Sparse-ST-GCN repo). PRs that improve the local data adapter, sampling, or fold reuse are welcome.
