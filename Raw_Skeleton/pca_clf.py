#!/usr/bin/env python
"""PCA + classical classifier baseline on RAW NTU skeletons (resampled to
T=100). Subject-level leave-5-subjects-out CV, identical to Tangent_Vector.

Reads:  ../data/data_ntu.pkl
Writes: results/pca_clf_metrics.csv (rows: KNN/SVM/RF/XGBoost/MLP)
        results/pca_clf_classwise_best.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

from cv_utils import (
    classwise_report,
    fold_indices,
    leave_5_subjects_out_folds,
    load_data,
    metrics_summary_df,
    subject_bootstrap_ci_class,
    CLASS_ORDER,
)

SEED = 42
TV_RESULTS = Path(__file__).resolve().parents[1] / "Tangent_Vector" / "results"


def fpca_project(X_train_flat, X_other_flat, R: int):
    """SVD on (D x N_train); return (N_train, R), (N_other, R)."""
    U, s, Vt = np.linalg.svd(X_train_flat, full_matrices=False)
    Ur = U[:, :R]
    return (Ur.T @ X_train_flat).T, (Ur.T @ X_other_flat).T


def get_models(seed: int, knn_cfg: dict | None = None):
    knn_cfg = knn_cfg or {}
    return {
        "KNN": KNeighborsClassifier(**knn_cfg),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--R", type=int, default=16, help="PCA components")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--output-dir", type=str,
                    default=str(Path(__file__).resolve().parent / "results"))
    ap.add_argument("--knn-cfg-file", type=str, default=None,
                    help="Path to JSON with KNN params. Defaults: "
                         "Tangent_Vector/results/best_knn_cfg.json then sklearn defaults.")
    ap.add_argument("--esvae-cfg-file", type=str, default=None,
                    help="Path to esvae_clf_config.json. Defaults to "
                         "Tangent_Vector/results/esvae_clf_config.json.")
    ap.add_argument("--match-esvae-R", action="store_true", default=True,
                    help="If true (default), set R from the ES-VAE config file.")
    args = ap.parse_args()

    np.random.seed(args.seed)

    knn_cfg = None
    cands = []
    if args.knn_cfg_file:
        cands.append(Path(args.knn_cfg_file))
    cands.append(TV_RESULTS / "best_knn_cfg.json")
    for p in cands:
        if p.exists():
            with open(p) as fh:
                knn_cfg = json.load(fh)
            print(f"Loaded KNN config from {p}: {knn_cfg}")
            break
    if knn_cfg is None:
        print("Using sklearn-default KNN config.")

    if args.match_esvae_R:
        cands = [Path(args.esvae_cfg_file)] if args.esvae_cfg_file else []
        cands.append(TV_RESULTS / "esvae_clf_config.json")
        for p in cands:
            if p.exists():
                with open(p) as fh:
                    cfg = json.load(fh)
                R_match = int(cfg.get("encoder", {}).get("R", args.R))
                print(f"Matching PCA R to ES-VAE encoder R from {p}: {R_match}")
                args.R = R_match
                break

    raw, _, X_flat, y, subj, _ = load_data(args.T)
    K, M, T = 25, 3, args.T
    # SVD expects (D, N) like the Tangent_Vector pipeline.
    flat_DN = X_flat.T.astype(np.float32)
    print(f"flat shape: {flat_DN.shape}, y: {y.shape}, R={args.R}")

    folds = leave_5_subjects_out_folds(subj, seed=args.seed)
    print(f"Folds: {len(folds)} (sizes={[len(f) for f in folds]})")

    models = get_models(args.seed, knn_cfg=knn_cfg)
    print(f"KNN model: {models['KNN']}")
    pooled = {name: {"targets": [], "preds": [], "subjects": []} for name in models}

    for k, test_subjects in enumerate(tqdm(folds, desc="L5SO folds")):
        tr_idx, te_idx = fold_indices(subj, test_subjects)
        Xtr, Xte = fpca_project(flat_DN[:, tr_idx], flat_DN[:, te_idx], R=args.R)
        scaler = StandardScaler().fit(Xtr)
        Xtr_s = scaler.transform(Xtr)
        Xte_s = scaler.transform(Xte)
        ytr, yte = y[tr_idx], y[te_idx]
        subjects_te = subj[te_idx]

        for name in models:
            m = type(models[name])(**models[name].get_params())
            if name in ("RF", "XGBoost"):
                m.fit(Xtr, ytr)
                preds = m.predict(Xte)
            else:
                m.fit(Xtr_s, ytr)
                preds = m.predict(Xte_s)
            pooled[name]["targets"].extend(yte.tolist())
            pooled[name]["preds"].extend(preds.tolist())
            pooled[name]["subjects"].extend(subjects_te.tolist())

    rows = []
    ci_results = {}
    for name, d in pooled.items():
        ci = subject_bootstrap_ci_class(
            d["targets"], d["preds"], d["subjects"],
            n_bootstrap=args.bootstrap, random_state=args.seed,
        )
        ci_results[name] = ci
        rows.append(metrics_summary_df(name, ci))
        f1m = f1_score(d["targets"], d["preds"], average="macro", zero_division=0)
        acc = accuracy_score(d["targets"], d["preds"])
        print(f"{name:10s}  Acc={acc:.3f}  Macro-F1={f1m:.3f}")

    summary = pd.concat(rows, ignore_index=True)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "pca_clf_metrics.csv", index=False)

    best_name = max(pooled.keys(), key=lambda n: ci_results[n]["F1 (macro)"]["mean"])
    cw = classwise_report(pooled[best_name]["targets"], pooled[best_name]["preds"], CLASS_ORDER)
    cw["model"] = best_name
    cw.to_csv(out_dir / "pca_clf_classwise_best.csv", index=False)
    print(f"Best PCA model: {best_name}")
    print(cw.to_string(index=False))

    raw_oof = {name: {k: list(map(int, v)) for k, v in d.items()}
               for name, d in pooled.items()}
    with open(out_dir / "pca_clf_oof.json", "w") as fh:
        json.dump(raw_oof, fh)


if __name__ == "__main__":
    main()
