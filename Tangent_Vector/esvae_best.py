#!/usr/bin/env python
"""Run ES-VAE classification with the best (encoder, KNN) config selected by
`esvae_clf.py --sweep`.

By default the chosen config is read from `results/esvae_clf_config.json`
(written by the sweep). Defaults below are also kept in sync with the latest
sweep result so the script stays self-contained even if the JSON is missing.

Usage (from Tangent_Vector/):
    python esvae_best.py                  # use config from JSON if present
    python esvae_best.py --device cuda:0  # override device
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("GEOMSTATS_BACKEND", "pytorch")

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
TV_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(TV_DIR))

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score

from cv_utils import (
    classwise_report,
    fold_indices,
    leave_5_subjects_out_folds,
    load_data,
    metrics_summary_df,
    subject_bootstrap_ci_class,
    CLASS_NAMES,
    CLASS_ORDER,
)
from esvae_clf import (
    encode_batch,
    set_deterministic,
    train_esvae_fold,
)

# Defaults baked from sweep result (overridden by JSON file when present).
# Selected by `esvae_clf.py --sweep`: pooled Macro-F1 = 0.951.
DEFAULT_ENCODER = {
    "seed": 42,
    "R": 16,
    "epochs": 150,
    "lr": 1e-3,
    "batch_size": 64,
    "beta_kl": 1e-4,
    "dropout": 0.10,
    "hidden": 512,
}
DEFAULT_KNN = {"n_neighbors": 3, "weights": "distance"}


def load_chosen_cfg(results_dir: Path):
    cfg_path = results_dir / "esvae_clf_config.json"
    if cfg_path.exists():
        with open(cfg_path) as fh:
            payload = json.load(fh)
        enc = {**DEFAULT_ENCODER, **payload.get("encoder", {})}
        knn = {**DEFAULT_KNN, **payload.get("knn", {})}
        return enc, knn, str(cfg_path)
    return dict(DEFAULT_ENCODER), dict(DEFAULT_KNN), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tslen", type=int, default=100)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", type=str,
                    default=str(Path(__file__).resolve().parent / "results"))
    ap.add_argument("--config-file", type=str, default=None,
                    help="Path to esvae_clf_config.json. Defaults to "
                         "results/esvae_clf_config.json.")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.config_file:
        with open(args.config_file) as fh:
            payload = json.load(fh)
        enc_cfg = {**DEFAULT_ENCODER, **payload.get("encoder", {})}
        knn_cfg = {**DEFAULT_KNN, **payload.get("knn", {})}
        cfg_src = args.config_file
    else:
        enc_cfg, knn_cfg, cfg_src = load_chosen_cfg(out_dir)

    if cfg_src:
        print(f"Loaded config from {cfg_src}")
    else:
        print("Config file not found; falling back to baked-in defaults.")
    print(f"Encoder: {enc_cfg}")
    print(f"KNN:     {knn_cfg}")

    device = torch.device(args.device or ("cuda:1" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float32
    print(f"Device: {device}")

    set_deterministic(args.seed)
    tangent, _, mu_arr, X_man_np, y, subj, _ = load_data(args.tslen)
    K, M, T, N = tangent.shape
    D = K * M * T

    mu_shape = torch.from_numpy(mu_arr.reshape(-1).astype(np.float32)).to(
        device=device, dtype=dtype
    )
    X_tan_full = torch.from_numpy(
        tangent.transpose(3, 0, 1, 2).reshape(-1, D).astype(np.float32)
    ).to(device=device, dtype=dtype)
    X_man_full = torch.from_numpy(X_man_np.astype(np.float32)).to(
        device=device, dtype=dtype
    )

    folds = leave_5_subjects_out_folds(subj, seed=args.seed)
    print(f"Folds: {len(folds)}  sizes={[len(f) for f in folds]}")

    pooled = {"targets": [], "preds": [], "subjects": []}
    for k, test_subjects in enumerate(folds):
        tr_idx, te_idx = fold_indices(subj, test_subjects)
        seed = enc_cfg["seed"] + k
        model = train_esvae_fold(
            X_tan_full[tr_idx], X_man_full[tr_idx], mu_shape, K, M, T,
            R=enc_cfg["R"], num_epochs=enc_cfg["epochs"], lr=enc_cfg["lr"],
            batch_size=min(enc_cfg["batch_size"], len(tr_idx)),
            beta_kl=enc_cfg["beta_kl"], dropout=enc_cfg["dropout"],
            hidden=enc_cfg["hidden"],
            device=device, dtype=dtype, seed=seed,
        )
        Z_tr = encode_batch(model, X_tan_full[tr_idx], device, dtype)
        Z_te = encode_batch(model, X_tan_full[te_idx], device, dtype)
        knn = KNeighborsClassifier(**knn_cfg)
        knn.fit(Z_tr, y[tr_idx])
        preds = knn.predict(Z_te)

        pooled["targets"].extend(y[te_idx].tolist())
        pooled["preds"].extend(preds.tolist())
        pooled["subjects"].extend(subj[te_idx].tolist())

        f1m = f1_score(y[te_idx], preds, average="macro", zero_division=0)
        acc = accuracy_score(y[te_idx], preds)
        print(f"  fold {k+1:02d}/{len(folds)}  acc={acc:.3f}  macroF1={f1m:.3f}",
              flush=True)

    pooled_acc = accuracy_score(pooled["targets"], pooled["preds"])
    pooled_f1 = f1_score(pooled["targets"], pooled["preds"], average="macro",
                         zero_division=0)
    print(f"\nPooled  acc={pooled_acc:.3f}  macroF1={pooled_f1:.3f}")

    ci = subject_bootstrap_ci_class(
        pooled["targets"], pooled["preds"], pooled["subjects"],
        n_bootstrap=args.bootstrap, random_state=args.seed,
    )
    summary = metrics_summary_df("ES-VAE (best)", ci)
    summary.to_csv(out_dir / "esvae_best_metrics.csv", index=False)
    print(summary.to_string(index=False))

    # Persist pooled OOF predictions
    with open(out_dir / "esvae_best_oof.json", "w") as fh:
        json.dump({"targets": [int(x) for x in pooled["targets"]],
                   "preds":   [int(x) for x in pooled["preds"]],
                   "subjects":[int(x) for x in pooled["subjects"]],
                   "class_order": CLASS_ORDER},
                  fh)

    # sklearn classification_report (per-class + macro/weighted avg + accuracy).
    # Use friendly target names (e.g. "A080 squat_down").
    target_names = [f"{c} {CLASS_NAMES[c]}" for c in CLASS_ORDER]
    rep_dict = classification_report(
        pooled["targets"], pooled["preds"],
        labels=list(range(len(CLASS_ORDER))),
        target_names=target_names,
        digits=4, zero_division=0, output_dict=True,
    )
    rep_text = classification_report(
        pooled["targets"], pooled["preds"],
        labels=list(range(len(CLASS_ORDER))),
        target_names=target_names,
        digits=4, zero_division=0,
    )
    print("\nsklearn classification_report:")
    print(rep_text)

    # Convert dict to a tidy DataFrame, mirroring sklearn's print order.
    rows = []
    for label in target_names + ["accuracy", "macro avg", "weighted avg"]:
        d = rep_dict.get(label)
        if d is None:
            continue
        if label == "accuracy":
            rows.append({
                "class": "accuracy",
                "precision": "",
                "recall": "",
                "f1-score": round(float(d), 4),
                "support": int(rep_dict["macro avg"]["support"]),
            })
        else:
            rows.append({
                "class": label,
                "precision": round(float(d["precision"]), 4),
                "recall":    round(float(d["recall"]), 4),
                "f1-score":  round(float(d["f1-score"]), 4),
                "support":   int(d["support"]),
            })
    cw_df = pd.DataFrame(rows)
    cw_df["model"] = "ES-VAE (best)"
    cw_df.to_csv(out_dir / "esvae_best_classwise.csv", index=False)
    # Also overwrite the project-wide classwise_best.csv used by README.
    cw_df.to_csv(out_dir / "classwise_best.csv", index=False)

    with open(out_dir / "esvae_best_used_cfg.json", "w") as fh:
        json.dump({"encoder": enc_cfg, "knn": knn_cfg,
                   "pooled_macroF1": pooled_f1, "pooled_acc": pooled_acc},
                  fh, indent=2)
    print(f"\nSaved: esvae_best_metrics.csv, esvae_best_classwise.csv, "
          f"classwise_best.csv (sklearn report), esvae_best_oof.json, "
          f"esvae_best_used_cfg.json")


if __name__ == "__main__":
    main()
