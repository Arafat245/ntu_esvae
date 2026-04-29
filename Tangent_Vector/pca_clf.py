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
    get_folds_and_axis,
    leave_5_subjects_out_folds,
    load_data,
    metrics_summary_df,
    subject_bootstrap_ci_class,
    CLASS_ORDER,
)

SEED = 42


def fpca_project(X_train_flat, X_other_flat, R: int):
    """Centered PCA via SVD on tangent_flat (D x N_train).

    Subtracts the per-dimension training-fold mean from both train and test
    before SVD so the components capture variance, not the mean direction.
    """
    mean_d = X_train_flat.mean(axis=1, keepdims=True)
    Xtr_c = X_train_flat - mean_d
    Xte_c = X_other_flat - mean_d
    U, s, Vt = np.linalg.svd(Xtr_c, full_matrices=False)
    Ur = U[:, :R]
    return (Ur.T @ Xtr_c).T, (Ur.T @ Xte_c).T


def get_models(seed: int, knn_cfg: dict | None = None):
    knn_cfg = knn_cfg or {}
    return {
        "KNN": KNeighborsClassifier(**knn_cfg),
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
    ap.add_argument("--cv-mode", choices=["subject", "view", "setup"], default="subject",
                    help="subject = 14 L5SO folds; view = 3 leave-one-camera-out folds.")
    args = ap.parse_args()

    np.random.seed(args.seed)

    # PCA mirrors the ES-VAE winner's KNN config (same k, same weights) for a
    # fair comparison. Falls back to k=3 distance if no winner config is found.
    knn_cfg = {"n_neighbors": 3, "weights": "distance"}
    knn_candidates: list[Path] = []
    if args.knn_cfg_file:
        knn_candidates.append(Path(args.knn_cfg_file))
    knn_candidates.append(Path(args.output_dir) / "best_knn_cfg.json")
    for p in knn_candidates:
        if p.exists():
            with open(p) as fh:
                knn_cfg = json.load(fh)
            print(f"PCA KNN matched to ES-VAE winner from {p}: {knn_cfg}")
            break
    else:
        print(f"PCA KNN (fallback default): {knn_cfg}")

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

    folds, fold_axis, mode_label = get_folds_and_axis(args.cv_mode, subj, seed=args.seed)
    print(f"CV mode: {mode_label}; folds: {len(folds)} (sizes={[len(f) for f in folds]})")

    models = get_models(args.seed, knn_cfg=knn_cfg)
    print(f"KNN model details: {models['KNN']}")
    pooled = {name: {"targets": [], "preds": [], "subjects": []} for name in models}

    for k, test_subjects in enumerate(tqdm(folds, desc=mode_label)):
        train_idx, test_idx = fold_indices(fold_axis, test_subjects)
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

    suffix = {"subject": "", "view": "_xview", "setup": "_xsetup"}[args.cv_mode]
    summary = pd.concat(rows, ignore_index=True)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"pca_clf_metrics{suffix}.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved {summary_path}")

    # Best model -> classwise
    best_name = max(pooled.keys(), key=lambda n: ci_results[n]["F1 (macro)"]["mean"])
    cw = classwise_report(pooled[best_name]["targets"], pooled[best_name]["preds"], CLASS_ORDER)
    cw["model"] = best_name
    cw_path = out_dir / f"pca_clf_classwise_best{suffix}.csv"
    cw.to_csv(cw_path, index=False)
    print(f"Best PCA model: {best_name}")
    print(cw.to_string(index=False))

    # Save raw OOF predictions for downstream use
    raw = {name: {k: list(map(int, v)) for k, v in d.items()} for name, d in pooled.items()}
    with open(out_dir / f"pca_clf_oof{suffix}.json", "w") as fh:
        json.dump(raw, fh)


if __name__ == "__main__":
    main()
