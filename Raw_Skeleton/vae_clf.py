#!/usr/bin/env python
"""Vanilla VAE + KNN baseline on RAW NTU skeletons (resampled to T=100).

Architectural twin of Tangent_Vector/esvae_clf.py's NonlinearVAE, but
trained with **plain MSE reconstruction loss** (no manifold structure,
no exp/log map). The latent means are then classified with KNN.

Subject-level leave-5-subjects-out CV (identical partition to
Tangent_Vector). The default config and KNN are aligned to the ES-VAE
selection so the comparison is matched-pair fair (same R, same KNN);
`--sweep` runs a small encoder × KNN grid like esvae_clf.py.

Reads:  ../data/data_ntu.pkl
Writes: results/vae_clf_metrics.csv, results/vae_clf_classwise.csv,
        results/vae_clf_config.json, results/vae_sweep.csv (when --sweep),
        results/vae_clf_oof.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader, TensorDataset

from cv_utils import (
    classwise_report,
    fold_indices,
    get_folds_and_axis,
    leave_5_subjects_out_folds,
    load_data,
    metrics_summary_df,
    subject_bootstrap_ci_class,
    CLASS_NAMES,
    CLASS_ORDER,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TV_RESULTS = REPO_ROOT / "Tangent_Vector" / "results"

NUM_CLASSES = len(CLASS_ORDER)
SEED = 42


def set_deterministic(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Model — same MLP topology as ES-VAE's NonlinearVAE
# ---------------------------------------------------------------------------
class NonlinearVAE(nn.Module):
    def __init__(self, D: int, R: int, H: int = 512, dropout: float = 0.10):
        super().__init__()
        self.enc1 = nn.Linear(D, H, bias=False)
        self.enc2 = nn.Linear(H, H, bias=False)
        self.mu_head = nn.Linear(H, R, bias=False)
        self.lv_head = nn.Linear(H, R)
        self.dec1 = nn.Linear(R, H, bias=False)
        self.dec2 = nn.Linear(H, D, bias=False)
        self.dropout = nn.Dropout(dropout)

    def encode(self, x):
        h = torch.tanh(self.enc1(x))
        h = self.dropout(h)
        h = torch.tanh(self.enc2(h))
        return self.mu_head(h), self.lv_head(h)

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        h = torch.tanh(self.dec1(z))
        return self.dec2(h)

    def forward(self, x):
        mu, lv = self.encode(x)
        z = self.reparam(mu, lv)
        x_hat = self.decode(z)
        return x_hat, mu, lv, z


def vae_loss(x, x_hat, mu, lv, beta: float = 1e-4):
    # MSE summed over features then averaged over batch (matches the
    # geodesic-loss-mean scale used by ES-VAE's recon term).
    recon = ((x - x_hat) ** 2).sum(dim=1).mean()
    kl = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp(), dim=1).mean()
    return recon + beta * kl, recon, kl


def train_vae_fold(X_tr, R, num_epochs, lr, batch_size, beta_kl, dropout,
                   hidden, device, dtype, seed, weight_decay=1e-5,
                   kl_warmup_frac=0.30):
    set_deterministic(seed)
    D = X_tr.shape[1]
    model = NonlinearVAE(D, R, H=hidden, dropout=dropout).to(device=device, dtype=dtype)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(num_epochs, 1))

    ds = TensorDataset(X_tr)
    # DataLoader's sampler always lives on CPU even when tensors are on GPU.
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False, generator=g)

    warmup = max(int(num_epochs * kl_warmup_frac), 1)
    model.train()
    for ep in range(num_epochs):
        kl_w = beta_kl * min(1.0, (ep + 1) / warmup)
        for (xb,) in loader:
            xb = xb.to(device=device, dtype=dtype, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            x_hat, mu, lv, z = model(xb)
            loss, _, _ = vae_loss(xb, x_hat, mu, lv, beta=kl_w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()

    model.eval()
    return model


@torch.no_grad()
def encode(model, X, device, dtype):
    model.eval()
    X = X.to(device=device, dtype=dtype)
    mu, _ = model.encode(X)
    return mu.cpu().numpy()


KNN_GRID = [
    {"n_neighbors": 1, "weights": "uniform"},
    {"n_neighbors": 3, "weights": "uniform"},
    {"n_neighbors": 5, "weights": "uniform"},
    {"n_neighbors": 7, "weights": "uniform"},
    {"n_neighbors": 3, "weights": "distance"},
    {"n_neighbors": 5, "weights": "distance"},
    {"n_neighbors": 7, "weights": "distance"},
]


def _knn_key(c):
    return f"k={c['n_neighbors']},{c['weights']}"


def standardize_train_apply(Xtr, *others):
    mean = Xtr.mean(axis=0, keepdims=True).astype(np.float32)
    std = Xtr.std(axis=0, keepdims=True).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    out = [((Xtr - mean) / std).astype(np.float32)]
    for x in others:
        out.append(((x - mean) / std).astype(np.float32))
    return out


def run_cv(enc_cfg, knn_grid, X_flat, y, subj, folds, device, dtype, fold_axis=None):
    pooled = {_knn_key(c): {"targets": [], "preds": [], "subjects": [], "cfg": c}
              for c in knn_grid}
    fold_axis_arr = subj if fold_axis is None else fold_axis

    for k, test_subjects in enumerate(folds):
        tr_idx, te_idx = fold_indices(fold_axis_arr, test_subjects)
        Xtr_s, Xte_s = standardize_train_apply(X_flat[tr_idx], X_flat[te_idx])
        Xtr_t = torch.from_numpy(Xtr_s).to(device=device, dtype=dtype)
        Xte_t = torch.from_numpy(Xte_s).to(device=device, dtype=dtype)

        seed = enc_cfg["seed"] + k
        model = train_vae_fold(
            Xtr_t,
            R=enc_cfg["R"], num_epochs=enc_cfg["epochs"], lr=enc_cfg["lr"],
            batch_size=min(enc_cfg["batch_size"], len(tr_idx)),
            beta_kl=enc_cfg["beta_kl"], dropout=enc_cfg["dropout"],
            hidden=enc_cfg["hidden"],
            device=device, dtype=dtype, seed=seed,
        )
        Z_tr = encode(model, Xtr_t, device, dtype)
        Z_te = encode(model, Xte_t, device, dtype)

        ytr, yte = y[tr_idx], y[te_idx]
        subjects_te = subj[te_idx].tolist()

        line = []
        for c in knn_grid:
            knn = KNeighborsClassifier(**c)
            knn.fit(Z_tr, ytr)
            preds = knn.predict(Z_te)
            entry = pooled[_knn_key(c)]
            entry["targets"].extend(yte.tolist())
            entry["preds"].extend(preds.tolist())
            entry["subjects"].extend(subjects_te)
            line.append(f"{_knn_key(c)}:{f1_score(yte, preds, average='macro', zero_division=0):.3f}")
        print(f"  fold {k+1:02d}/{len(folds)}  " + "  ".join(line), flush=True)

    summary = {}
    for c in knn_grid:
        e = pooled[_knn_key(c)]
        summary[_knn_key(c)] = {
            "acc": accuracy_score(e["targets"], e["preds"]),
            "macroF1": f1_score(e["targets"], e["preds"], average="macro", zero_division=0),
            "knn_cfg": c,
        }
    return pooled, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--R", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--beta-kl", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--sweep", action="store_true",
                    help="Tiny grid over (R, epochs) × KNN options.")
    ap.add_argument("--match-esvae-config", action="store_true", default=True,
                    help="If set (default), align R / epochs / hidden / dropout / "
                         "and KNN to ES-VAE's chosen config (read from "
                         "../Tangent_Vector/results/esvae_clf_config.json) when present "
                         "and --sweep is not used.")
    ap.add_argument("--cv-mode", choices=["subject", "view", "setup"], default="subject")
    ap.add_argument("--output-dir", type=str,
                    default=str(Path(__file__).resolve().parent / "results"))
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda:1" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float32
    print(f"Device: {device}")

    set_deterministic(args.seed)
    _, _, X_flat, y, subj, _ = load_data(args.T)
    print(f"X_flat: {X_flat.shape}, y: {y.shape}")

    folds, fold_axis, mode_label = get_folds_and_axis(args.cv_mode, subj, seed=args.seed)
    print(f"CV mode: {mode_label}; folds: {len(folds)}  sizes={[len(f) for f in folds]}")

    base_cfg = dict(seed=args.seed, R=args.R, epochs=args.epochs, lr=args.lr,
                    batch_size=args.batch_size, beta_kl=args.beta_kl,
                    dropout=args.dropout, hidden=args.hidden)

    chosen_knn = None
    if not args.sweep and args.match_esvae_config:
        cfg_path = TV_RESULTS / "esvae_clf_config.json"
        if cfg_path.exists():
            with open(cfg_path) as fh:
                payload = json.load(fh)
            enc = payload.get("encoder", {})
            for fld in ("R", "epochs", "lr", "batch_size", "beta_kl", "dropout", "hidden"):
                if fld in enc:
                    base_cfg[fld] = enc[fld]
            chosen_knn = payload.get("knn", None)
            print(f"Matched encoder + KNN to ES-VAE config from {cfg_path}: "
                  f"enc={base_cfg}, knn={chosen_knn}")
        else:
            print("ES-VAE config file not found; using CLI defaults.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.sweep:
        enc_grid = []
        for R in [8, 12, 16, 24]:
            for ep, hidden in [(150, 512), (200, 512), (200, 768)]:
                enc_grid.append({**base_cfg, "R": R, "epochs": ep, "hidden": hidden})
        all_records = []
        best = None
        for i, ec in enumerate(enc_grid):
            print(f"\n[Encoder {i+1}/{len(enc_grid)}] cfg={ec}")
            pooled_dict, summary = run_cv(ec, KNN_GRID, X_flat, y, subj, folds, device, dtype, fold_axis=fold_axis)
            for kk, s in summary.items():
                print(f"  -> {kk:20s}  acc={s['acc']:.3f} macroF1={s['macroF1']:.3f}")
                all_records.append({**ec, "knn_n_neighbors": s["knn_cfg"]["n_neighbors"],
                                    "knn_weights": s["knn_cfg"]["weights"],
                                    "pooled_acc": s["acc"], "pooled_macroF1": s["macroF1"]})
                if best is None or s["macroF1"] > best["macroF1"]:
                    best = {"enc_cfg": ec, "knn_cfg": s["knn_cfg"],
                            "pooled": pooled_dict[kk], "macroF1": s["macroF1"],
                            "acc": s["acc"]}
        _sweep_suffix = {"subject": "", "view": "_xview", "setup": "_xsetup"}[args.cv_mode]
        pd.DataFrame(all_records).to_csv(out_dir / f"vae_sweep{_sweep_suffix}.csv", index=False)
        pooled = best["pooled"]
        cfg = best["enc_cfg"]
        knn_cfg = best["knn_cfg"]
        print(f"\nBest combo -> enc={cfg}  knn={knn_cfg}  macroF1={best['macroF1']:.3f}")
    else:
        knn_to_use = chosen_knn or {"n_neighbors": 5, "weights": "uniform"}
        print(f"\nSingle config: enc={base_cfg}  knn={knn_to_use}")
        pooled_dict, summary = run_cv(base_cfg, [knn_to_use], X_flat, y, subj, folds, device, dtype, fold_axis=fold_axis)
        only_key = list(pooled_dict.keys())[0]
        pooled = pooled_dict[only_key]
        cfg = base_cfg
        knn_cfg = knn_to_use
        print(f"\nPooled  acc={summary[only_key]['acc']:.3f}  "
              f"macroF1={summary[only_key]['macroF1']:.3f}")

    ci = subject_bootstrap_ci_class(
        pooled["targets"], pooled["preds"], pooled["subjects"],
        n_bootstrap=args.bootstrap, random_state=args.seed,
    )
    suffix = {"subject": "", "view": "_xview", "setup": "_xsetup"}[args.cv_mode]
    summary_df = metrics_summary_df("Vanilla VAE", ci)
    summary_df.to_csv(out_dir / f"vae_clf_metrics{suffix}.csv", index=False)
    print(summary_df.to_string(index=False))

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
    rows = []
    for label in target_names + ["accuracy", "macro avg", "weighted avg"]:
        e = rep_dict.get(label)
        if e is None:
            continue
        if label == "accuracy":
            rows.append({"class": "accuracy", "precision": "", "recall": "",
                         "f1-score": round(float(e), 4),
                         "support": int(rep_dict["macro avg"]["support"])})
        else:
            rows.append({"class": label,
                         "precision": round(float(e["precision"]), 4),
                         "recall":    round(float(e["recall"]), 4),
                         "f1-score":  round(float(e["f1-score"]), 4),
                         "support":   int(e["support"])})
    cw_df = pd.DataFrame(rows)
    cw_df["model"] = "Vanilla VAE"
    cw_df.to_csv(out_dir / f"vae_clf_classwise{suffix}.csv", index=False)

    with open(out_dir / f"vae_clf_config{suffix}.json", "w") as fh:
        json.dump({"encoder": cfg, "knn": knn_cfg,
                   "pooled_macroF1": float(ci["F1 (macro)"]["mean"])},
                  fh, indent=2)
    with open(out_dir / f"vae_clf_oof{suffix}.json", "w") as fh:
        json.dump({"targets": [int(x) for x in pooled["targets"]],
                   "preds":   [int(x) for x in pooled["preds"]],
                   "subjects":[int(x) for x in pooled["subjects"]],
                   "class_order": CLASS_ORDER}, fh)


if __name__ == "__main__":
    main()
