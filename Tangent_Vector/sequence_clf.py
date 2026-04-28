#!/usr/bin/env python
"""Sequence baselines (TCN / LSTM / Transformer / STGCN) for NTU 5-class
action classification on tangent vectors. Subject-level leave-5-subjects-out CV.

Each tangent sample is treated as a single sequence of length T=100 with
75 channels (= 25 joints x 3 coords). No sliding window.

Reads:  ../aligned_data/tangent_vecs100.pkl, ../aligned_data/sample_index.csv
Writes: results/sequence_clf_metrics.csv (rows: TCN/LSTM/TRANSFORMER/STGCN)
        results/sequence_clf_classwise_best.csv
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.nn.utils import weight_norm
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

NUM_CLASSES = 5
NUM_JOINTS = 25
COORDS = 3
SEED = 42


# ---------------------------------------------------------------------------
# Determinism + standardization
# ---------------------------------------------------------------------------
def set_deterministic(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ChannelStandardizer:
    """Per-channel z-score over (samples, time) computed on training set."""

    def fit(self, x):
        # x: (N, C, T)
        self.mean = x.mean(axis=(0, 2), keepdims=True).astype(np.float32)
        self.std = x.std(axis=(0, 2), keepdims=True).astype(np.float32)
        self.std = np.where(self.std < 1e-6, 1.0, self.std)
        return self

    def transform(self, x):
        return ((x - self.mean) / self.std).astype(np.float32)


# ---------------------------------------------------------------------------
# TCN
# ---------------------------------------------------------------------------
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_in, n_out, kernel, dilation, dropout):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.conv1 = weight_norm(nn.Conv1d(n_in, n_out, kernel, padding=pad, dilation=dilation))
        self.chomp1 = Chomp1d(pad)
        self.bn1 = nn.BatchNorm1d(n_out)
        self.conv2 = weight_norm(nn.Conv1d(n_out, n_out, kernel, padding=pad, dilation=dilation))
        self.chomp2 = Chomp1d(pad)
        self.bn2 = nn.BatchNorm1d(n_out)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.down = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else None

    def forward(self, x):
        out = self.drop(self.act(self.bn1(self.chomp1(self.conv1(x)))))
        out = self.drop(self.act(self.bn2(self.chomp2(self.conv2(out)))))
        res = x if self.down is None else self.down(x)
        return self.act(out + res)


class TCNClassifier(nn.Module):
    def __init__(self, in_dim, channels=(16, 16), kernel=3, dropout=0.40, hidden=24):
        super().__init__()
        blocks = []
        prev = in_dim
        for i, c in enumerate(channels):
            blocks.append(TemporalBlock(prev, c, kernel, 2 ** i, dropout))
            prev = c
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.LayerNorm(channels[-1] * 2),
            nn.Linear(channels[-1] * 2, hidden),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, NUM_CLASSES),
        )

    def forward(self, x):  # x: (B, C, T)
        feat = self.tcn(x)
        pooled = torch.cat([feat.mean(dim=-1), feat.max(dim=-1).values], dim=1)
        return self.head(pooled)


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------
class LSTMClassifier(nn.Module):
    def __init__(self, in_dim, hidden=16, layers=1, dropout=0.40, bidir=False):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim, hidden_size=hidden, num_layers=layers,
            batch_first=True, bidirectional=bidir,
            dropout=dropout if layers > 1 else 0.0,
        )
        h_out = hidden * (2 if bidir else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(h_out * 2),
            nn.Linear(h_out * 2, hidden),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, NUM_CLASSES),
        )

    def forward(self, x):  # x: (B, C, T) -> (B, T, C)
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        feat = torch.cat([out.mean(dim=1), out.max(dim=1).values], dim=1)
        return self.head(feat)


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerClassifier(nn.Module):
    def __init__(self, in_dim, d_model=24, n_heads=2, d_ff=48, n_layers=1, dropout=0.30):
        super().__init__()
        self.embed = nn.Linear(in_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, NUM_CLASSES),
        )

    def forward(self, x):
        x = x.transpose(1, 2)               # (B, T, C)
        z = self.pos(self.embed(x))
        z = self.encoder(z)
        z = z.mean(dim=1)
        return self.head(z)


# ---------------------------------------------------------------------------
# STGCN with NTU-25 adjacency
# ---------------------------------------------------------------------------
def ntu25_adjacency():
    """Symmetric normalized adjacency for NTU 25-joint kinematic skeleton."""
    edges = [
        (0, 1), (1, 20), (20, 2), (2, 3),
        (20, 4), (4, 5), (5, 6), (6, 7), (7, 21), (7, 22),
        (20, 8), (8, 9), (9, 10), (10, 11), (11, 23), (11, 24),
        (0, 12), (12, 13), (13, 14), (14, 15),
        (0, 16), (16, 17), (17, 18), (18, 19),
    ]
    n = NUM_JOINTS
    A = torch.eye(n)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    D = A.sum(1)
    D_inv = torch.pow(D + 1e-6, -0.5)
    return D_inv.unsqueeze(1) * A * D_inv.unsqueeze(0)


class STBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel=9, dropout=0.2):
        super().__init__()
        self.spatial = nn.Linear(in_c, out_c)
        self.spatial_bn = nn.BatchNorm2d(out_c)
        pad = (kernel - 1) // 2
        self.temporal = nn.Conv2d(out_c, out_c, (kernel, 1), padding=(pad, 0))
        self.temporal_bn = nn.BatchNorm2d(out_c)
        self.drop = nn.Dropout(dropout)
        self.down = nn.Linear(in_c, out_c) if in_c != out_c else None

    def forward(self, x, A):
        # x: (B, T, V, C)
        res = x
        out = torch.einsum("ij,btjc->btic", A, x)
        out = self.spatial(out)              # (B, T, V, out_c)
        out = out.permute(0, 3, 1, 2)        # (B, out_c, T, V)
        out = F.relu(self.spatial_bn(out))
        out = self.temporal(out)
        out = self.temporal_bn(out)
        out = self.drop(out)
        out = out.permute(0, 2, 3, 1)        # (B, T, V, out_c)
        if self.down is not None:
            res = self.down(res)
        return F.relu(out + res)


class STGCNClassifier(nn.Module):
    def __init__(self, channels=(16, 32), kernel=9, dropout=0.30):
        super().__init__()
        self.register_buffer("A", ntu25_adjacency())
        self.in_proj = nn.Linear(COORDS, channels[0])
        self.blocks = nn.ModuleList()
        prev = channels[0]
        for c in channels[1:]:
            self.blocks.append(STBlock(prev, c, kernel=kernel, dropout=dropout))
            prev = c
        self.head = nn.Sequential(
            nn.LayerNorm(channels[-1]),
            nn.Linear(channels[-1], channels[-1]),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(channels[-1], NUM_CLASSES),
        )

    def forward(self, x):  # x: (B, C=75, T)
        b, c, t = x.shape
        x = x.view(b, NUM_JOINTS, COORDS, t).permute(0, 3, 1, 2)  # (B, T, V, 3)
        x = self.in_proj(x)
        for blk in self.blocks:
            x = blk(x, self.A)
        x = x.mean(dim=(1, 2))  # global avg over time and joints
        return self.head(x)


# ---------------------------------------------------------------------------
# Training / eval
# ---------------------------------------------------------------------------
MODEL_BUILDERS = {
    "TCN":         lambda C: TCNClassifier(C),
    "LSTM":        lambda C: LSTMClassifier(C),
    "TRANSFORMER": lambda C: TransformerClassifier(C),
    "STGCN":       lambda C: STGCNClassifier(),
}


def train_one(model, Xtr, ytr, device, epochs, lr, batch_size, weight_decay, label_smoothing, seed):
    set_deterministic(seed)
    model = model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(epochs, 1))
    crit = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    Xtr_t = torch.from_numpy(Xtr).to(device)
    ytr_t = torch.from_numpy(ytr).long().to(device)
    n = Xtr_t.shape[0]

    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = Xtr_t[idx], ytr_t[idx]
            logits = model(xb)
            loss = crit(logits, yb)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
        sched.step()
    return model


@torch.no_grad()
def predict(model, X, device, batch_size=64):
    model.eval()
    X_t = torch.from_numpy(X).to(device)
    out = []
    for i in range(0, X_t.shape[0], batch_size):
        logits = model(X_t[i : i + batch_size])
        out.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tslen", type=int, default=100)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--models", type=str, default="TCN,LSTM,TRANSFORMER,STGCN",
                    help="Comma-separated subset of MODEL_BUILDERS keys.")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--output-dir", type=str,
                    default=str(Path(__file__).resolve().parent / "results"))
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    set_deterministic(args.seed)
    tan, _, _, _, y, subj, _ = load_data(args.tslen)
    K, M, T, N = tan.shape
    # (B, C, T) with C = K*M = 75
    X = tan.transpose(3, 0, 1, 2).reshape(N, K * M, T).astype(np.float32)
    print(f"X: {X.shape}, y: {y.shape}, classes: {np.bincount(y)}")

    folds = leave_5_subjects_out_folds(subj, seed=args.seed)
    print(f"Folds: {len(folds)}  sizes={[len(f) for f in folds]}")

    chosen = [m.strip().upper() for m in args.models.split(",") if m.strip()]
    pooled = {name: {"targets": [], "preds": [], "subjects": []} for name in chosen}

    for k, test_subjects in enumerate(folds):
        train_idx, test_idx = fold_indices(subj, test_subjects)
        scaler = ChannelStandardizer().fit(X[train_idx])
        Xtr = scaler.transform(X[train_idx])
        Xte = scaler.transform(X[test_idx])
        ytr, yte = y[train_idx], y[test_idx]

        for name in chosen:
            fold_seed = args.seed + k * 1000
            model = MODEL_BUILDERS[name](X.shape[1])
            model = train_one(
                model, Xtr, ytr, device,
                epochs=args.epochs, lr=args.lr,
                batch_size=args.batch_size,
                weight_decay=args.weight_decay,
                label_smoothing=args.label_smoothing,
                seed=fold_seed,
            )
            preds = predict(model, Xte, device)
            pooled[name]["targets"].extend(yte.tolist())
            pooled[name]["preds"].extend(preds.tolist())
            pooled[name]["subjects"].extend(subj[test_idx].tolist())

            f1m = f1_score(yte, preds, average="macro", zero_division=0)
            acc = accuracy_score(yte, preds)
            print(f"Fold {k+1:02d}/{len(folds)} | {name:11s} | acc={acc:.3f} macroF1={f1m:.3f}")

    # Aggregate
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    ci_results = {}
    for name in chosen:
        d = pooled[name]
        ci = subject_bootstrap_ci_class(
            d["targets"], d["preds"], d["subjects"],
            n_bootstrap=args.bootstrap, random_state=args.seed,
        )
        ci_results[name] = ci
        rows.append(metrics_summary_df(name, ci))
        print(f"{name:11s} pooled  Acc={ci['Accuracy']['mean']:.3f}  "
              f"MacroF1={ci['F1 (macro)']['mean']:.3f}  "
              f"MacroPrec={ci['Precision (macro)']['mean']:.3f}  "
              f"MacroRec={ci['Recall (macro)']['mean']:.3f}")

    summary = pd.concat(rows, ignore_index=True)
    summary_path = out_dir / "sequence_clf_metrics.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved {summary_path}")

    best = max(chosen, key=lambda n: ci_results[n]["F1 (macro)"]["mean"])
    cw = classwise_report(pooled[best]["targets"], pooled[best]["preds"], CLASS_ORDER)
    cw["model"] = best
    cw_path = out_dir / "sequence_clf_classwise_best.csv"
    cw.to_csv(cw_path, index=False)
    print(f"Best sequence model: {best}")
    print(cw.to_string(index=False))


if __name__ == "__main__":
    main()
