#!/usr/bin/env python
"""Phase 1b: epoch sweep on the winning (R, hidden, beta_kl) config.

Reads the main-sweep winner from results/esvae_clf_config.json and re-runs
that exact encoder + KNN config at multiple epoch counts on subject CV.
Writes results/esvae_epoch_sweep.csv.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT.as_posix()))
sys.path.insert(0, str((REPO_ROOT / "Tangent_Vector").as_posix()))

from cv_utils import (  # noqa: E402
    fold_indices, get_folds_and_axis, load_data,
)
from esvae_clf import (  # noqa: E402
    encode_batch, set_deterministic, train_esvae_fold,
)

EPOCHS_GRID = [25, 50, 100, 150, 200, 250, 300, 400]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tslen", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    ap.add_argument("--cfg-file", default=str(REPO_ROOT / "Tangent_Vector" / "results" / "esvae_clf_config.json"))
    ap.add_argument("--output-dir", default=str(REPO_ROOT / "Tangent_Vector" / "results"))
    args = ap.parse_args()

    cfg = json.loads(Path(args.cfg_file).read_text())
    enc = cfg["encoder"]
    knn_cfg = cfg["knn"]
    print(f"Loaded winner: enc={enc}  knn={knn_cfg}")

    device = torch.device(args.device or ("cuda:1" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float32
    set_deterministic(args.seed)

    tangent, betas, mu_arr, X_man_np, y, subj, _ = load_data(args.tslen)
    K, M, T, N = tangent.shape
    D = K * M * T

    X_tan_full = torch.from_numpy(
        tangent.transpose(3, 0, 1, 2).reshape(-1, D).astype(np.float32)
    ).to(device=device, dtype=dtype)
    X_man_full = torch.from_numpy(X_man_np.astype(np.float32)).to(device=device, dtype=dtype)
    mu_shape = torch.from_numpy(mu_arr.reshape(-1).astype(np.float32)).to(device=device, dtype=dtype)

    folds, fold_axis, _ = get_folds_and_axis("subject", subj, seed=args.seed)
    print(f"L5SO folds: {len(folds)}  sizes={[len(f) for f in folds]}")

    rows = []
    for ep in EPOCHS_GRID:
        pooled = {"targets": [], "preds": [], "subjects": []}
        for k, test_subjects in enumerate(folds):
            tr_idx, te_idx = fold_indices(fold_axis, test_subjects)
            seed = enc["seed"] + k
            model = train_esvae_fold(
                X_tan_full[tr_idx], X_man_full[tr_idx], mu_shape, K, M, T,
                R=enc["R"], num_epochs=ep, lr=enc["lr"],
                batch_size=min(enc["batch_size"], len(tr_idx)),
                beta_kl=enc["beta_kl"], dropout=enc["dropout"], hidden=enc["hidden"],
                device=device, dtype=dtype, seed=seed,
            )
            Z_tr = encode_batch(model, X_tan_full[tr_idx], device, dtype)
            Z_te = encode_batch(model, X_tan_full[te_idx], device, dtype)
            ytr, yte = y[tr_idx], y[te_idx]

            knn = KNeighborsClassifier(**knn_cfg)
            knn.fit(Z_tr, ytr)
            preds = knn.predict(Z_te)
            pooled["targets"].extend(yte.tolist())
            pooled["preds"].extend(preds.tolist())
            pooled["subjects"].extend(subj[te_idx].tolist())

        f1m = f1_score(pooled["targets"], pooled["preds"], average="macro", zero_division=0)
        acc = accuracy_score(pooled["targets"], pooled["preds"])
        print(f"epochs={ep:4d}  acc={acc:.4f}  macroF1={f1m:.4f}", flush=True)
        rows.append({"epochs": ep, "pooled_acc": acc, "pooled_macroF1": f1m})

    out = Path(args.output_dir) / "esvae_epoch_sweep.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved {out}")
    best = max(rows, key=lambda r: r["pooled_macroF1"])
    print(f"Best epochs={best['epochs']}  macroF1={best['pooled_macroF1']:.4f}")


if __name__ == "__main__":
    main()
