#!/usr/bin/env python
"""PCA + classical classifier baseline for NTU 5-class action classification
on tangent vectors. Subject-level leave-5-subjects-out CV.

Reads:  ../aligned_data/tangent_vecs100.pkl, ../aligned_data/sample_index.csv
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


def fpca_project(X_train_flat, X_other_flat, R: int):
    """SVD on tangent_flat (D x N_train); project both train and other."""
    U, s, Vt = np.linalg.svd(X_train_flat, full_matrices=False)
    Ur = U[:, :R]
    return (Ur.T @ X_train_flat).T, (Ur.T @ X_other_flat).T


def get_models(seed: int, knn_cfg: dict | None = None):
    knn_cfg = knn_cfg or {}
    return {
        "KNN": KNeighborsClassifier(**knn_cfg),
        "SVM": SVC(kernel="rbf", C=4.0, gamma="scale", random_state=seed),
        "RF": RandomForestClassifier(n_estimators=400, random_state=seed, n_jobs=-1),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            objective="multi:softprob", num_class=5, eval_metric="mlogloss",
            random_state=seed, tree_method="hist", n_jobs=-1, verbosity=0,
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(128, 64), max_iter=2000,
            early_stopping=False, random_state=seed,
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tslen", type=int, default=100)
    ap.add_argument("--R", type=int, default=32, help="PCA components")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--output-dir", type=str,
                    default=str(Path(__file__).resolve().parent / "results"))
    ap.add_argument("--knn-cfg-file", type=str, default=None,
                    help="Path to JSON with KNN params (n_neighbors, weights). "
                         "If omitted, falls back to results/best_knn_cfg.json then sklearn defaults.")
    ap.add_argument("--esvae-cfg-file", type=str, default=None,
                    help="Path to esvae_clf_config.json. If present, the PCA "
                         "embedding dim R is set to encoder.R for an apples-to-apples "
                         "comparison with ES-VAE's KNN-on-latents result.")
    ap.add_argument("--match-esvae-R", action="store_true", default=True,
                    help="If true (default), read R from esvae_clf_config.json when present.")
    args = ap.parse_args()

    np.random.seed(args.seed)

    knn_cfg = None
    candidates = []
    if args.knn_cfg_file:
        candidates.append(Path(args.knn_cfg_file))
    candidates.append(Path(args.output_dir) / "best_knn_cfg.json")
    for p in candidates:
        if p.exists():
            with open(p) as fh:
                knn_cfg = json.load(fh)
            print(f"Loaded KNN config from {p}: {knn_cfg}")
            break
    if knn_cfg is None:
        print("Using sklearn-default KNN config.")

    if args.match_esvae_R:
        cand = [Path(args.esvae_cfg_file)] if args.esvae_cfg_file else []
        cand.append(Path(args.output_dir) / "esvae_clf_config.json")
        for p in cand:
            if p.exists():
                with open(p) as fh:
                    esvae_cfg = json.load(fh)
                R_match = int(esvae_cfg.get("encoder", {}).get("R", args.R))
                print(f"Matching PCA R to ES-VAE encoder R from {p}: {R_match}")
                args.R = R_match
                break

    tan, _, _, _, y, subj, classes = load_data(args.tslen)
    K, M, T, N = tan.shape
    tangent_flat = tan.reshape(K * M * T, N).astype(np.float32)
    print(f"tangent_flat: {tangent_flat.shape}, y: {y.shape}, R={args.R}")

    folds = leave_5_subjects_out_folds(subj, seed=args.seed)
    print(f"Folds: {len(folds)} (sizes={[len(f) for f in folds]})")

    models = get_models(args.seed, knn_cfg=knn_cfg)
    print(f"KNN model details: {models['KNN']}")
    pooled = {name: {"targets": [], "preds": [], "subjects": []} for name in models}

    for k, test_subjects in enumerate(tqdm(folds, desc="L5SO folds")):
        train_idx, test_idx = fold_indices(subj, test_subjects)
        Xtr, Xte = fpca_project(tangent_flat[:, train_idx], tangent_flat[:, test_idx], R=args.R)

        # Standardize PCA features (helps SVM/MLP)
        scaler = StandardScaler().fit(Xtr)
        Xtr_s = scaler.transform(Xtr)
        Xte_s = scaler.transform(Xte)

        ytr, yte = y[train_idx], y[test_idx]
        subjects_te = subj[test_idx]

        for name in models:
            m = type(models[name])(**models[name].get_params())
            # Tree models don't need scaling; sklearn is fine either way
            if name in ("RF", "XGBoost"):
                m.fit(Xtr, ytr)
                preds = m.predict(Xte)
            else:
                m.fit(Xtr_s, ytr)
                preds = m.predict(Xte_s)
            pooled[name]["targets"].extend(yte.tolist())
            pooled[name]["preds"].extend(preds.tolist())
            pooled[name]["subjects"].extend(subjects_te.tolist())

    # Aggregate
    rows = []
    ci_results = {}
    for name, d in pooled.items():
        ci = subject_bootstrap_ci_class(
            d["targets"], d["preds"], d["subjects"],
            n_bootstrap=args.bootstrap, random_state=args.seed,
        )
        ci_results[name] = ci
        rows.append(metrics_summary_df(name, ci))
        acc = accuracy_score(d["targets"], d["preds"])
        f1m = f1_score(d["targets"], d["preds"], average="macro", zero_division=0)
        print(f"{name:10s}  Acc={acc:.3f}  Macro-F1={f1m:.3f}")

    summary = pd.concat(rows, ignore_index=True)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "pca_clf_metrics.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved {summary_path}")

    # Best model -> classwise
    best_name = max(pooled.keys(), key=lambda n: ci_results[n]["F1 (macro)"]["mean"])
    cw = classwise_report(pooled[best_name]["targets"], pooled[best_name]["preds"], CLASS_ORDER)
    cw["model"] = best_name
    cw_path = out_dir / "pca_clf_classwise_best.csv"
    cw.to_csv(cw_path, index=False)
    print(f"Best PCA model: {best_name}")
    print(cw.to_string(index=False))

    # Save raw OOF predictions for downstream use
    raw = {name: {k: list(map(int, v)) for k, v in d.items()} for name, d in pooled.items()}
    with open(out_dir / "pca_clf_oof.json", "w") as fh:
        json.dump(raw, fh)


if __name__ == "__main__":
    main()
