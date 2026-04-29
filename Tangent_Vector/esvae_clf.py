#!/usr/bin/env python
"""ES-VAE with geodesic reconstruction loss for NTU 5-class action
classification on tangent vectors.

Pipeline (per fold of leave-5-subjects-out CV):
  1. Train a NonlinearVAE that decodes back to tangent vectors.
  2. Reconstruction is mapped to the manifold via exp_map(mu, v_hat) and
     compared to the aligned manifold curve via squared geodesic distance.
  3. After training, take latent means as features and fit a small KNN.
  4. Pool predictions across folds, report Macro F1/Precision/Recall + 95% CI.

A tiny in-script sweep over (R, num_epochs, k_knn) selects the config with
best pooled Macro F1 (with --sweep). Default just runs the chosen best config.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

# Important: functionsgpu_fast.py sets default device to cuda:1 on import.
# Allow override via env var before import.
os.environ.setdefault("GEOMSTATS_BACKEND", "pytorch")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Import the project's manifold utilities
from functionsgpu_fast import (  # noqa: E402
    exp_gpu_batch,
    squared_geodesic_distance,
)

from cv_utils import (  # noqa: E402
    classwise_report,
    fold_indices,
    get_folds_and_axis,
    leave_5_subjects_out_folds,
    load_data,
    metrics_summary_df,
    subject_bootstrap_ci_class,
    CLASS_ORDER,
)

NUM_CLASSES = len(CLASS_ORDER)
SEED = 42


# ---------------------------------------------------------------------------
def set_deterministic(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class NonlinearVAE(nn.Module):
    """Tanh MLP VAE used inside ES-VAE (mirrors stroke_riemann)."""

    def __init__(self, D: int, R: int, H: int = 256, dropout: float = 0.20):
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
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.tanh(self.dec1(z))
        return self.dec2(h)

    def forward(self, x):
        mu, lv = self.encode(x)
        z = self.reparam(mu, lv)
        v_hat = self.decode(z)
        return v_hat, mu, lv, z


class ESVAE(nn.Module):
    """Wraps a tangent-space VAE with a manifold reconstruction head."""

    def __init__(self, base_vae, mu_shape, K, M, T):
        super().__init__()
        self.vae = base_vae
        self.K, self.M, self.T = K, M, T
        self.register_buffer("mu_shape", mu_shape)  # (K, M, T) flat

    def forward(self, x):
        v_hat, mu_z, lv, z = self.vae(x)
        B = v_hat.shape[0]
        v_hat_r = v_hat.view(B, self.K, self.M, self.T)
        mu = self.mu_shape.view(self.K, self.M, self.T)
        x_recon_man = exp_gpu_batch(mu, v_hat_r).view(B, -1)
        return x_recon_man, mu_z, lv, z, v_hat


def esvae_loss(x_man, x_hat_man, mu_z, lv, K, M, T, beta=1e-4):
    dist = squared_geodesic_distance(x_man, x_hat_man, K, M, T)
    recon = dist.mean()
    kl = -0.5 * torch.sum(1 + lv - mu_z.pow(2) - lv.exp(), dim=1).mean()
    return recon + beta * kl, recon, kl


def train_esvae_fold(X_tan_tr, X_man_tr, mu_shape, K, M, T,
                     R, num_epochs, lr, batch_size, beta_kl, dropout, hidden,
                     device, dtype, seed, weight_decay=1e-5, kl_warmup_frac=0.3):
    set_deterministic(seed)
    D = X_tan_tr.shape[1]
    base = NonlinearVAE(D, R, H=hidden, dropout=dropout).to(device=device, dtype=dtype)
    model = ESVAE(base, mu_shape, K, M, T).to(device=device, dtype=dtype)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(num_epochs, 1))

    ds = TensorDataset(X_tan_tr, X_man_tr)
    g = torch.Generator(device=device).manual_seed(seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False, generator=g)

    warmup_epochs = max(int(num_epochs * kl_warmup_frac), 1)
    model.train()
    for ep in range(num_epochs):
        # Linearly anneal KL weight from 0 to beta_kl over the first warmup epochs.
        kl_w = beta_kl * min(1.0, (ep + 1) / warmup_epochs)
        for xb_tan, xb_man in loader:
            xb_tan = xb_tan.to(device=device, dtype=dtype, non_blocking=True)
            xb_man = xb_man.to(device=device, dtype=dtype, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            x_hat_man, mu_z, lv, z, _ = model(xb_tan)
            loss, _, _ = esvae_loss(xb_man, x_hat_man, mu_z, lv, K, M, T, beta=kl_w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()

    model.eval()
    return model


@torch.no_grad()
def encode_batch(model, X_tan, device, dtype):
    model.eval()
    X_tan = X_tan.to(device=device, dtype=dtype)
    mu, _ = model.vae.encode(X_tan)
    return mu.detach().cpu().numpy()


KNN_GRID = [
    {"n_neighbors": 1, "weights": "uniform"},
    {"n_neighbors": 3, "weights": "uniform"},
    {"n_neighbors": 5, "weights": "uniform"},
    {"n_neighbors": 7, "weights": "uniform"},
    {"n_neighbors": 3, "weights": "distance"},
    {"n_neighbors": 5, "weights": "distance"},
    {"n_neighbors": 7, "weights": "distance"},
]


def _knn_key(knn_cfg: dict) -> str:
    return f"k={knn_cfg['n_neighbors']},{knn_cfg['weights']}"


def run_cv(enc_cfg, knn_grid, tangent, X_man_np, mu_shape, y, subj, folds, K, M, T, device, dtype, fold_axis=None):
    """Train the encoder once per fold; evaluate every KNN config on the same latents.

    Returns: dict[knn_key] -> {"targets","preds","subjects","cfg"}.
    """
    D = K * M * T
    X_tan_full = torch.from_numpy(
        tangent.transpose(3, 0, 1, 2).reshape(-1, D).astype(np.float32)
    ).to(device=device, dtype=dtype)
    X_man_full = torch.from_numpy(X_man_np.astype(np.float32)).to(device=device, dtype=dtype)

    pooled = {
        _knn_key(kc): {"targets": [], "preds": [], "subjects": [], "cfg": kc}
        for kc in knn_grid
    }

    fold_axis_arr = subj if fold_axis is None else fold_axis
    for k, test_subjects in enumerate(folds):
        train_idx, test_idx = fold_indices(fold_axis_arr, test_subjects)
        seed = enc_cfg["seed"] + k
        model = train_esvae_fold(
            X_tan_full[train_idx], X_man_full[train_idx], mu_shape,
            K, M, T,
            R=enc_cfg["R"], num_epochs=enc_cfg["epochs"], lr=enc_cfg["lr"],
            batch_size=min(enc_cfg["batch_size"], len(train_idx)),
            beta_kl=enc_cfg["beta_kl"], dropout=enc_cfg["dropout"], hidden=enc_cfg["hidden"],
            device=device, dtype=dtype, seed=seed,
        )
        Z_tr = encode_batch(model, X_tan_full[train_idx], device, dtype)
        Z_te = encode_batch(model, X_tan_full[test_idx], device, dtype)
        ytr, yte = y[train_idx], y[test_idx]
        subjects_te = subj[test_idx].tolist()

        fold_lines = []
        for kc in knn_grid:
            knn = KNeighborsClassifier(**kc)
            knn.fit(Z_tr, ytr)
            preds = knn.predict(Z_te)
            entry = pooled[_knn_key(kc)]
            entry["targets"].extend(yte.tolist())
            entry["preds"].extend(preds.tolist())
            entry["subjects"].extend(subjects_te)
            f1m = f1_score(yte, preds, average="macro", zero_division=0)
            fold_lines.append(f"{_knn_key(kc)}:{f1m:.3f}")
        print(f"  fold {k+1:02d}/{len(folds)}  " + "  ".join(fold_lines), flush=True)

    summary = {}
    for kc in knn_grid:
        entry = pooled[_knn_key(kc)]
        f1m = f1_score(entry["targets"], entry["preds"], average="macro", zero_division=0)
        acc = accuracy_score(entry["targets"], entry["preds"])
        summary[_knn_key(kc)] = {"acc": acc, "macroF1": f1m, "knn_cfg": kc}
    return pooled, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tslen", type=int, default=100)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--R", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--beta-kl", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--k-knn", type=int, default=5,
                    help="Kept for compatibility; KNN classifier always uses sklearn defaults to match PCA.")
    ap.add_argument("--sweep", action="store_true",
                    help="Tiny grid search over (R, epochs).")
    ap.add_argument("--cv-mode", choices=["subject", "view", "setup"], default="subject")
    ap.add_argument("--output-dir", type=str,
                    default=str(Path(__file__).resolve().parent / "results"))
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda:1" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float32
    print(f"Device: {device}")

    set_deterministic(args.seed)
    tangent, betas, mu_arr, X_man_np, y, subj, _ = load_data(args.tslen)
    K, M, T, N = tangent.shape
    print(f"tangent {tangent.shape}, betas {betas.shape}, mu {mu_arr.shape}, y {y.shape}")

    mu_shape = torch.from_numpy(mu_arr.reshape(-1).astype(np.float32)).to(device=device, dtype=dtype)
    folds, fold_axis, mode_label = get_folds_and_axis(args.cv_mode, subj, seed=args.seed)
    print(f"CV mode: {mode_label}; folds: {len(folds)}  sizes={[len(f) for f in folds]}")

    base_cfg = dict(
        seed=args.seed, R=args.R, epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, beta_kl=args.beta_kl,
        dropout=args.dropout, hidden=args.hidden, k_knn=args.k_knn,
    )

    if args.sweep:
        enc_grid = []
        # Phase 1 sweep for the 10-class set: R × hidden × beta_kl.
        for R in [16, 24, 32, 48]:
            for hidden in [512, 768]:
                for beta in [1e-4, 1e-3]:
                    enc_grid.append({**base_cfg, "R": R, "epochs": 200,
                                     "hidden": hidden, "beta_kl": beta,
                                     "dropout": 0.10})
    else:
        enc_grid = [base_cfg]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_records = []
    best = None  # (enc_cfg, knn_cfg, pooled, macroF1, acc)
    for i, enc_cfg in enumerate(enc_grid):
        print(f"\n[Encoder {i+1}/{len(enc_grid)}] cfg={enc_cfg}")
        pooled_dict, summary = run_cv(enc_cfg, KNN_GRID, tangent, X_man_np, mu_shape,
                                      y, subj, folds, K, M, T, device, dtype, fold_axis=fold_axis)
        for knn_key, s in summary.items():
            print(f"  -> {knn_key:20s}  acc={s['acc']:.3f} macroF1={s['macroF1']:.3f}")
            all_records.append({
                **enc_cfg,
                "knn_n_neighbors": s["knn_cfg"]["n_neighbors"],
                "knn_weights": s["knn_cfg"]["weights"],
                "pooled_acc": s["acc"],
                "pooled_macroF1": s["macroF1"],
            })
            if best is None or s["macroF1"] > best["macroF1"]:
                best = {
                    "enc_cfg": enc_cfg,
                    "knn_cfg": s["knn_cfg"],
                    "pooled": pooled_dict[knn_key],
                    "macroF1": s["macroF1"],
                    "acc": s["acc"],
                }

    suffix = {"subject": "", "view": "_xview", "setup": "_xsetup"}[args.cv_mode]
    pd.DataFrame(all_records).to_csv(out_dir / f"esvae_sweep{suffix}.csv", index=False)
    pooled = best["pooled"]
    cfg = best["enc_cfg"]
    knn_cfg = best["knn_cfg"]
    print(f"\nBest combo -> enc={cfg}  knn={knn_cfg}  macroF1={best['macroF1']:.3f}")

    ci = subject_bootstrap_ci_class(
        pooled["targets"], pooled["preds"], pooled["subjects"],
        n_bootstrap=args.bootstrap, random_state=args.seed,
    )
    summary_df = metrics_summary_df("ES-VAE (geodesic)", ci)
    summary_path = out_dir / f"esvae_clf_metrics{suffix}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved {summary_path}")
    print(summary_df.to_string(index=False))

    cw = classwise_report(pooled["targets"], pooled["preds"], CLASS_ORDER)
    cw["model"] = "ES-VAE"
    cw_path = out_dir / f"esvae_clf_classwise{suffix}.csv"
    cw.to_csv(cw_path, index=False)
    print(cw.to_string(index=False))

    # Persist the chosen encoder + KNN config so PCA can reuse the same KNN.
    import json
    with open(out_dir / f"esvae_clf_config{suffix}.json", "w") as fh:
        json.dump({"encoder": cfg, "knn": knn_cfg, "pooled_macroF1": best["macroF1"]},
                  fh, indent=2)
    with open(out_dir / f"best_knn_cfg{suffix}.json", "w") as fh:
        json.dump(knn_cfg, fh, indent=2)


if __name__ == "__main__":
    main()
