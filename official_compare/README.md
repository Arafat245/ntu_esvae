# Official-Model Comparison on the NTU Top10 Subset

This folder adapts the **official GitHub implementations** of:

- **ProtoGCN** (CVPR 2025)
- **Sparse-ST-GCN** (CVPR 2025)

to the local **400-sample NTU-60 top10 subset** used elsewhere in this
repo.

## What is preserved from the official repos

- backbone architecture
- graph definition
- head / loss logic
- clip sampling format
- single-stream NTU joint-input setting

## What is adapted locally

- dataset loader reads `../data/data_ntu.pkl` for raw skeletons and
  `../aligned_data/tangent_vecs100.pkl` for tangent vectors
- class count is reduced from `60` to `10`
- folds reuse the repo's deterministic **8 leave-5-subjects-out**
  partition from `Tangent_Vector/cv_utils.py`
- training harness is lightweight PyTorch instead of the original
  PYSKL/MMCV experiment stack

## Files

- `common.py` — data loading, sampling, metrics, graph helper
- `protogcn_runner.py` — adapted ProtoGCN subject-CV runner
- `sparse_stgcn_runner.py` — adapted Sparse-ST-GCN subject-CV runner
- `results/*.json` — final subject-CV outputs for raw and tangent runs

## Reproduction

From repo root:

```bash
python official_compare/protogcn_runner.py --representation raw     --epochs 20 --batch-size 32 --device cuda:0
python official_compare/protogcn_runner.py --representation tangent --epochs 20 --batch-size 32 --device cuda:0
python official_compare/sparse_stgcn_runner.py --representation raw     --epochs 20 --batch-size 32 --device cuda:1
python official_compare/sparse_stgcn_runner.py --representation tangent --epochs 20 --batch-size 32 --device cuda:1
```

## Final subject-CV results

| Model | Input | Top-1 | Macro-F1 | 95% CI |
|---|---|---:|---:|---:|
| ProtoGCN | Raw skeleton | **0.6275** | **0.6167** | [0.5732, 0.6592] |
| ProtoGCN | Tangent vector | 0.5525 | 0.5510 | [0.5024, 0.5941] |
| Sparse-ST-GCN | Raw skeleton | 0.5075 | 0.4893 | [0.4397, 0.5367] |
| Sparse-ST-GCN | Tangent vector | **0.5150** | **0.5006** | [0.4595, 0.5421] |

The result is mixed: ProtoGCN prefers raw coordinates on this tiny
subset, while Sparse-ST-GCN gives a slight edge to tangent vectors.
