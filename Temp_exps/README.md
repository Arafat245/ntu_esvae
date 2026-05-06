# Temp Experiments

This folder holds standalone experiments that should not disturb the main `../Tangent_Vector/` and `../Raw_Skeleton/` pipelines. Current experiment: a TCN-intermediate-feature + KNN probe under the same fold protocol used in the main repo.

## Requirements

Install dependencies from the repo root:

```setup
pip install -r ../requirements.txt
```

The probe reads from `../aligned_data/tangent_vecs100.pkl` (tangent representation) or `../data/data_ntu.pkl` (raw representation), and reuses the L5SO subject folds from `../Tangent_Vector/cv_utils.py`.

## Training

`tcn_intermediate_knn.py` trains the existing small TCN under the same fold protocol used in the repo, extracts pooled features after each temporal block, then fits KNN on those intermediate features:

```train
python3 Temp_exps/tcn_intermediate_knn.py --representation tangent --cv-mode subject
python3 Temp_exps/tcn_intermediate_knn.py --representation raw     --cv-mode subject
```

Run from the repo root.

## Evaluation

The script reports pooled OOF metrics with the same subject-level bootstrap helper used elsewhere in the repo. Results are written to `Temp_exps/results/`:

```eval
ls Temp_exps/results/
```

## Pre-trained Models

No checkpoints are distributed. The probe is small and re-runs end-to-end in a few minutes per representation.

## Results

Outputs land in `Temp_exps/results/` per run. Numbers are not pinned in this README because the experiment is exploratory and the configuration may change; consult the result CSVs for the latest values.

## Contributing

Add new experiments here as self-contained scripts that reuse `../Tangent_Vector/cv_utils.py` for fold splits and the bootstrap helper, so results stay matched-pair comparable with the main pipelines.
