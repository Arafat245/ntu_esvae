#!/usr/bin/env python
"""Build an ES-VAE alignment-ablation CSV across the NTU and stroke repos.

Rows are fixed to the alignment ablation stages:
  1. No alignment (raw coordinates)
  2. Translation removal (centering)
  3. Translation + scale removal (preshape)
  4. Translation + scale + rotation removal (Kendall shape)
  5. Full alignment (Kendall + TSRVF)

Outputs one CSV with:
  - NTU subject-CV Macro F1
  - stroke subject-CV RMSE
  - 95% CI bounds for both metrics

Important note about model usage:
  - The full-alignment row reuses the already-reported ES-VAE results from the
    repos, as requested.
  - For the non-full rows, the script reuses the ES-VAE encoder family and
    downstream KNN/KNN-regressor protocol, but trains with Euclidean
    reconstruction on the stage-specific coordinates. This keeps the encoder
    family fixed while letting the representation change across rows.

By default the script only emits the CSV from cached results plus the existing
full-alignment row. Pass --compute-missing to train the missing rows.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from torch.utils.data import DataLoader, TensorDataset


ROW_LABELS = [
    "No alignment (raw coordinates)",
    "Translation removal (centering)",
    "Translation + scale removal (preshape)",
    "Translation + scale + rotation removal (Kendall shape)",
    "Full alignment (Kendall + TSRVF)",
]

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_STROKE_ROOT = REPO_ROOT.parent / "stroke_riemann"
CACHE_DIR_NAME = "ablation_cache"

NTU_CLASS_ORDER = [
    "A001",
    "A002",
    "A003",
    "A004",
    "A008",
    "A009",
    "A023",
    "A028",
    "A029",
    "A031",
]


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def bootstrap_interval(values: list[float], ci: float = 95.0) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    alpha = (100.0 - ci) / 2.0
    lo, hi = np.percentile(values, [alpha, 100.0 - alpha])
    return float(lo), float(hi)


def subject_bootstrap_ci_class(
    targets: np.ndarray,
    preds: np.ndarray,
    subject_ids: np.ndarray,
    n_bootstrap: int = 2000,
    random_state: int = 42,
) -> dict[str, float]:
    rng = np.random.default_rng(random_state)
    unique_subjects = np.unique(subject_ids)
    boots = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(unique_subjects, size=len(unique_subjects), replace=True)
        idx = np.concatenate([np.where(subject_ids == s)[0] for s in sampled])
        t = targets[idx]
        p = preds[idx]
        if len(np.unique(t)) < 2:
            continue
        boots.append(f1_score(t, p, average="macro", zero_division=0))
    mean = f1_score(targets, preds, average="macro", zero_division=0)
    lo, hi = bootstrap_interval(boots)
    return {"mean": float(mean), "ci_low": lo, "ci_high": hi}


def subject_bootstrap_ci_regression(
    targets: np.ndarray,
    preds: np.ndarray,
    subject_ids: np.ndarray,
    n_bootstrap: int = 2000,
    random_state: int = 42,
) -> dict[str, float]:
    rng = np.random.default_rng(random_state)
    unique_subjects = np.unique(subject_ids)
    boots = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(unique_subjects, size=len(unique_subjects), replace=True)
        idx = np.isin(subject_ids, sampled)
        t = targets[idx]
        p = preds[idx]
        if len(np.unique(t)) < 2:
            continue
        boots.append(float(np.sqrt(mean_squared_error(t, p))))
    mean = float(np.sqrt(mean_squared_error(targets, preds)))
    lo, hi = bootstrap_interval(boots)
    return {"mean": float(mean), "ci_low": lo, "ci_high": hi}


def center_frame(frame: np.ndarray) -> np.ndarray:
    return frame - frame.mean(axis=0, keepdims=True)


def center_temporal(curve: np.ndarray) -> np.ndarray:
    out = np.array(curve, dtype=np.float32, copy=True)
    for t in range(out.shape[-1]):
        out[:, :, t] = center_frame(out[:, :, t])
    return out


def preshape_temporal(curve: np.ndarray) -> np.ndarray:
    out = center_temporal(curve)
    for t in range(out.shape[-1]):
        norm = np.linalg.norm(out[:, :, t], ord="fro")
        if norm < 1e-8:
            continue
        out[:, :, t] /= norm
    return out


def resample_curve_euclidean(curve: np.ndarray, target_len: int) -> np.ndarray:
    src_len = curve.shape[-1]
    if src_len == target_len:
        return np.array(curve, dtype=np.float32, copy=True)
    src_t = np.linspace(0.0, 1.0, src_len, dtype=np.float32)
    dst_t = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
    flat = curve.reshape(-1, src_len)
    flat_out = np.vstack([np.interp(dst_t, src_t, row).astype(np.float32) for row in flat])
    return flat_out.reshape(curve.shape[0], curve.shape[1], target_len)


def align_frame_to_reference(frame: np.ndarray, ref: np.ndarray) -> np.ndarray:
    # Solve min_R ||frame R - ref||_F with R orthogonal.
    m = frame.T @ ref
    u, _, vt = np.linalg.svd(m, full_matrices=False)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1.0
        r = u @ vt
    aligned = frame @ r
    return aligned.astype(np.float32, copy=False)


def align_curve_to_reference(curve: np.ndarray, ref: np.ndarray) -> np.ndarray:
    out = np.empty_like(curve, dtype=np.float32)
    for t in range(curve.shape[-1]):
        out[:, :, t] = align_frame_to_reference(curve[:, :, t], ref[:, :, t])
    return out


def rotation_only_procrustes_mean(
    curves: np.ndarray,
    max_iters: int = 10,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    mu = np.array(curves[0], dtype=np.float32, copy=True)
    mu = preshape_temporal(mu)
    prev = float("inf")
    aligned = None
    for _ in range(max_iters):
        aligned = np.stack([align_curve_to_reference(curve, mu) for curve in curves], axis=0).astype(np.float32)
        mu_new = preshape_temporal(aligned.mean(axis=0))
        delta = float(np.linalg.norm(mu_new - mu))
        mu = mu_new
        if abs(prev - delta) < tol:
            break
        prev = delta
    assert aligned is not None
    aligned = np.stack([align_curve_to_reference(curve, mu) for curve in curves], axis=0).astype(np.float32)
    return mu, aligned


def standardize_train_apply(train_x: np.ndarray, *others: np.ndarray) -> list[np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True).astype(np.float32)
    std = train_x.std(axis=0, keepdims=True).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    out = [((train_x - mean) / std).astype(np.float32)]
    for x in others:
        out.append(((x - mean) / std).astype(np.float32))
    return out


class NTUVAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden: int = 768, dropout: float = 0.10):
        super().__init__()
        self.enc1 = nn.Linear(input_dim, hidden, bias=False)
        self.enc2 = nn.Linear(hidden, hidden, bias=False)
        self.mu_head = nn.Linear(hidden, latent_dim, bias=False)
        self.lv_head = nn.Linear(hidden, latent_dim)
        self.dec1 = nn.Linear(latent_dim, hidden, bias=False)
        self.dec2 = nn.Linear(hidden, input_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.tanh(self.enc1(x))
        h = self.dropout(h)
        h = torch.tanh(self.enc2(h))
        return self.mu_head(h), self.lv_head(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.dec1(z))
        return self.dec2(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, lv = self.encode(x)
        std = torch.exp(0.5 * lv)
        z = mu + std * torch.randn_like(std)
        x_hat = self.decode(z)
        return x_hat, mu, lv


class StrokeVAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden: int = 128, decoder_hidden: int = 16, dropout: float = 0.10):
        super().__init__()
        self.enc1 = nn.Linear(input_dim, hidden, bias=False)
        self.mu_head = nn.Linear(hidden, latent_dim, bias=False)
        self.lv_head = nn.Linear(hidden, latent_dim)
        self.dec1 = nn.Linear(latent_dim, decoder_hidden, bias=False)
        self.dec2 = nn.Linear(decoder_hidden, input_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.tanh(self.enc1(x))
        h = self.dropout(h)
        return self.mu_head(h), self.lv_head(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.dec1(z))
        return self.dec2(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, lv = self.encode(x)
        std = torch.exp(0.5 * lv)
        z = mu + std * torch.randn_like(std)
        x_hat = self.decode(z)
        return x_hat, mu, lv


def vae_loss(x: torch.Tensor, x_hat: torch.Tensor, mu: torch.Tensor, lv: torch.Tensor, beta: float) -> torch.Tensor:
    recon = ((x - x_hat) ** 2).sum(dim=1).mean()
    kl = -0.5 * torch.sum(1.0 + lv - mu.pow(2) - lv.exp(), dim=1).mean()
    return recon + beta * kl


@dataclass
class MetricResult:
    mean: float
    ci_low: float
    ci_high: float
    source: str

    def as_json(self) -> dict[str, float | str]:
        return {
            "mean": self.mean,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "source": self.source,
        }


def load_json_result(path: Path) -> MetricResult:
    data = json.loads(path.read_text())
    return MetricResult(
        mean=float(data["mean"]),
        ci_low=float(data["ci_low"]),
        ci_high=float(data["ci_high"]),
        source=str(data.get("source", "cache")),
    )


def save_json_result(path: Path, result: MetricResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.as_json(), indent=2))


def get_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def make_loader(x: torch.Tensor, batch_size: int, seed: int) -> DataLoader:
    ds = TensorDataset(x)
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False, generator=g)


def train_ntu_vae(train_x: np.ndarray, device: torch.device, seed: int, epochs: int) -> NTUVAE:
    cfg = json.loads((REPO_ROOT / "Tangent_Vector" / "results" / "esvae_clf_config.json").read_text())
    enc = cfg["encoder"]
    set_deterministic(seed)
    x = torch.from_numpy(train_x).to(device=device, dtype=torch.float32)
    model = NTUVAE(x.shape[1], enc["R"], hidden=enc["hidden"], dropout=enc["dropout"]).to(device=device, dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=enc["lr"], weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    loader = make_loader(x, min(enc["batch_size"], len(train_x)), seed)
    warmup = max(int(epochs * 0.30), 1)
    model.train()
    for ep in range(epochs):
        beta = enc["beta_kl"] * min(1.0, (ep + 1) / warmup)
        for (xb,) in loader:
            xb = xb.to(device=device, dtype=torch.float32, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            x_hat, mu, lv = model(xb)
            loss = vae_loss(xb, x_hat, mu, lv, beta=beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
    model.eval()
    return model


def train_stroke_vae(train_x: np.ndarray, device: torch.device, seed: int, epochs: int) -> StrokeVAE:
    set_deterministic(seed)
    x = torch.from_numpy(train_x).to(device=device, dtype=torch.float32)
    model = StrokeVAE(x.shape[1], latent_dim=38, hidden=128, decoder_hidden=16, dropout=0.10).to(device=device, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        x_hat, mu, lv = model(x)
        loss = vae_loss(x, x_hat, mu, lv, beta=2 ** (-3))
        loss.backward()
        opt.step()
    model.eval()
    return model


@torch.no_grad()
def encode_latents(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    tensor = torch.from_numpy(x).to(device=device, dtype=torch.float32)
    mu, _ = model.encode(tensor)
    return mu.detach().cpu().numpy()


def leave_5_subjects_out_folds(subject_ids: np.ndarray, seed: int = 42, fold_size: int = 5) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    unique = np.array(sorted(np.unique(subject_ids).tolist()), dtype=np.int64)
    perm = rng.permutation(unique)
    return [perm[i : i + fold_size] for i in range(0, len(perm), fold_size)]


def fold_indices(subject_ids: np.ndarray, test_subjects: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    test_mask = np.isin(subject_ids, test_subjects)
    return np.where(~test_mask)[0], np.where(test_mask)[0]


def stroke_val_test(participant_ids: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    step_val = 5
    step_test = 5
    block_size = step_val + step_test
    n_blocks = len(participant_ids) // block_size
    block_start = (k % n_blocks) * block_size
    is_last_block = (block_start == (n_blocks - 1) * block_size) and (len(participant_ids) % block_size != 0)
    block_end = len(participant_ids) if is_last_block else block_start + block_size
    if k < n_blocks:
        if is_last_block:
            validation = participant_ids[block_start : block_start + step_val]
            test = participant_ids[block_start + step_val : block_end]
        else:
            validation = participant_ids[block_start : block_start + step_val]
            test = participant_ids[block_start + step_val : block_start + block_size]
    else:
        if is_last_block:
            validation = participant_ids[block_start + step_val : block_end]
            test = participant_ids[block_start : block_start + step_val]
        else:
            validation = participant_ids[block_start + step_val : block_start + block_size]
            test = participant_ids[block_start : block_start + step_val]
    return np.asarray(validation), np.asarray(test)


def evaluate_ntu_stage(curves: np.ndarray, subject_ids: np.ndarray, y: np.ndarray, device: torch.device, epochs: int, max_folds: int | None = None) -> MetricResult:
    folds = leave_5_subjects_out_folds(subject_ids, seed=42)
    if max_folds is not None:
        folds = folds[:max_folds]
    flat = curves.reshape(curves.shape[0], -1).astype(np.float32)
    cfg = json.loads((REPO_ROOT / "Tangent_Vector" / "results" / "esvae_clf_config.json").read_text())
    knn_cfg = cfg["knn"]
    pooled_targets: list[int] = []
    pooled_preds: list[int] = []
    pooled_subjects: list[int] = []
    for fold_idx, test_subjects in enumerate(folds):
        train_idx, test_idx = fold_indices(subject_ids, test_subjects)
        xtr, xte = standardize_train_apply(flat[train_idx], flat[test_idx])
        model = train_ntu_vae(xtr, device=device, seed=42 + fold_idx, epochs=epochs)
        ztr = encode_latents(model, xtr, device)
        zte = encode_latents(model, xte, device)
        knn = KNeighborsClassifier(**knn_cfg)
        knn.fit(ztr, y[train_idx])
        preds = knn.predict(zte)
        pooled_targets.extend(y[test_idx].tolist())
        pooled_preds.extend(preds.tolist())
        pooled_subjects.extend(subject_ids[test_idx].tolist())
    targets_np = np.asarray(pooled_targets, dtype=np.int64)
    preds_np = np.asarray(pooled_preds, dtype=np.int64)
    subj_np = np.asarray(pooled_subjects, dtype=np.int64)
    ci = subject_bootstrap_ci_class(targets_np, preds_np, subj_np)
    return MetricResult(mean=ci["mean"], ci_low=ci["ci_low"], ci_high=ci["ci_high"], source="computed")


def evaluate_stroke_stage(curves: np.ndarray, participant_ids: np.ndarray, y: np.ndarray, device: torch.device, epochs: int, max_folds: int | None = None) -> MetricResult:
    n_folds = 30 if max_folds is None else max_folds
    flat = curves.reshape(curves.shape[0], -1).astype(np.float32)
    pooled_targets: list[float] = []
    pooled_preds: list[float] = []
    pooled_subjects: list[int] = []
    for fold_idx in range(n_folds):
        val_pids, test_pids = stroke_val_test(participant_ids, fold_idx)
        train_pids = np.setdiff1d(participant_ids, np.concatenate([val_pids, test_pids]))
        train_idx = np.array([i for i, pid in enumerate(participant_ids) if pid in train_pids], dtype=np.int64)
        test_idx = np.array([i for i, pid in enumerate(participant_ids) if pid in test_pids], dtype=np.int64)
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        xtr, xte = standardize_train_apply(flat[train_idx], flat[test_idx])
        model = train_stroke_vae(xtr, device=device, seed=42 + fold_idx, epochs=epochs)
        ztr = encode_latents(model, xtr, device)
        zte = encode_latents(model, xte, device)
        knr = KNeighborsRegressor()
        knr.fit(ztr, y[train_idx])
        preds = knr.predict(zte)
        pooled_targets.extend(y[test_idx].tolist())
        pooled_preds.extend(preds.tolist())
        pooled_subjects.extend(participant_ids[test_idx].astype(int).tolist())
    targets_np = np.asarray(pooled_targets, dtype=np.float32)
    preds_np = np.asarray(pooled_preds, dtype=np.float32)
    subj_np = np.asarray(pooled_subjects, dtype=np.int64)
    ci = subject_bootstrap_ci_regression(targets_np, preds_np, subj_np)
    return MetricResult(mean=ci["mean"], ci_low=ci["ci_low"], ci_high=ci["ci_high"], source="computed")


def load_ntu_metadata() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    df = pd.read_csv(REPO_ROOT / "aligned_data" / "sample_index.csv")
    df = df.sort_values("sample_index").reset_index(drop=True)
    class_to_int = {name: idx for idx, name in enumerate(NTU_CLASS_ORDER)}
    y = df["class_id"].map(class_to_int).to_numpy(dtype=np.int64)
    subjects = df["person_id"].to_numpy(dtype=np.int64)
    return y, subjects, df


def load_ntu_centering_curves(cache_dir: Path) -> np.ndarray:
    cache_path = cache_dir / "ntu_centering_curves100.npy"
    if cache_path.exists():
        return np.load(cache_path)
    with open(REPO_ROOT / "data" / "data_ntu.pkl", "rb") as fh:
        raw_dict = pickle.load(fh)
    _, _, df = load_ntu_metadata()
    curves = []
    for row in df.itertuples(index=False):
        key = f"{int(row.person_id)}_{row.class_id}"
        curve = np.asarray(raw_dict[key], dtype=np.float32)
        curve = center_temporal(curve)
        curve = resample_curve_euclidean(curve, target_len=100)
        curves.append(curve)
    arr = np.stack(curves, axis=0).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, arr)
    return arr


def load_ntu_raw_curves(cache_dir: Path) -> np.ndarray:
    cache_path = cache_dir / "ntu_raw_curves100.npy"
    if cache_path.exists():
        return np.load(cache_path)
    with open(REPO_ROOT / "data" / "data_ntu.pkl", "rb") as fh:
        raw_dict = pickle.load(fh)
    _, _, df = load_ntu_metadata()
    curves = []
    for row in df.itertuples(index=False):
        key = f"{int(row.person_id)}_{row.class_id}"
        curve = np.asarray(raw_dict[key], dtype=np.float32)
        curve = resample_curve_euclidean(curve, target_len=100)
        curves.append(curve)
    arr = np.stack(curves, axis=0).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, arr)
    return arr


def load_ntu_preshape_curves() -> np.ndarray:
    with open(REPO_ROOT / "aligned_data" / "betas_resampled_kendall100.pkl", "rb") as fh:
        curves = pickle.load(fh)
    return np.stack(curves, axis=0).astype(np.float32)


def load_or_build_ntu_kendall_shape(cache_dir: Path) -> np.ndarray:
    cache_path = cache_dir / "ntu_kendall_shape_curves100.npy"
    if cache_path.exists():
        return np.load(cache_path)
    preshape = load_ntu_preshape_curves()
    _, aligned = rotation_only_procrustes_mean(preshape, max_iters=10)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, aligned)
    return aligned


def parse_stroke_csv_curve(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    gait_cycles = df.iloc[:, :-1].to_numpy(dtype=np.float32)
    n_rows = gait_cycles.shape[0]
    return gait_cycles.reshape(n_rows, 32, 3).transpose(1, 2, 0).astype(np.float32)


def load_stroke_metadata(stroke_root: Path) -> tuple[np.ndarray, np.ndarray, dict[int, Path]]:
    participant_ids = np.loadtxt(stroke_root / "labels_data" / "pids.txt").astype(int)
    y_poma = np.loadtxt(stroke_root / "labels_data" / "y_poma.txt").astype(np.float32)
    file_map = {}
    for path in sorted((stroke_root / "csv_r").glob("ID*_*.csv")):
        m = re.match(r"ID(\d+)_", path.name)
        if m:
            file_map[int(m.group(1))] = path
    return participant_ids, y_poma, file_map


def load_stroke_centering_curves(stroke_root: Path, cache_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache_path = cache_dir / "stroke_centering_curves200.npy"
    pids, y_poma, file_map = load_stroke_metadata(stroke_root)
    if cache_path.exists():
        return np.load(cache_path), pids, y_poma
    curves = []
    for pid in pids:
        curve = parse_stroke_csv_curve(file_map[int(pid)])
        curve = center_temporal(curve)
        curve = resample_curve_euclidean(curve, target_len=200)
        curves.append(curve)
    arr = np.stack(curves, axis=0).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, arr)
    return arr, pids, y_poma


def load_stroke_raw_curves(stroke_root: Path, cache_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache_path = cache_dir / "stroke_raw_curves200.npy"
    pids, y_poma, file_map = load_stroke_metadata(stroke_root)
    if cache_path.exists():
        return np.load(cache_path), pids, y_poma
    curves = []
    for pid in pids:
        curve = parse_stroke_csv_curve(file_map[int(pid)])
        curve = resample_curve_euclidean(curve, target_len=200)
        curves.append(curve)
    arr = np.stack(curves, axis=0).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, arr)
    return arr, pids, y_poma


def load_stroke_preshape_curves(stroke_root: Path, cache_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache_path = cache_dir / "stroke_preshape_curves200.npy"
    pids, y_poma, file_map = load_stroke_metadata(stroke_root)
    if cache_path.exists():
        return np.load(cache_path), pids, y_poma
    curves = []
    for pid in pids:
        curve = parse_stroke_csv_curve(file_map[int(pid)])
        curve = preshape_temporal(curve)
        curve = resample_curve_euclidean(curve, target_len=200)
        curves.append(curve)
    arr = np.stack(curves, axis=0).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, arr)
    return arr, pids, y_poma


def load_or_build_stroke_kendall_shape(stroke_root: Path, cache_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache_path = cache_dir / "stroke_kendall_shape_curves200.npy"
    pids, y_poma, _ = load_stroke_metadata(stroke_root)
    if cache_path.exists():
        return np.load(cache_path), pids, y_poma
    preshape, _, _ = load_stroke_preshape_curves(stroke_root, cache_dir)
    _, aligned = rotation_only_procrustes_mean(preshape, max_iters=10)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, aligned)
    return aligned, pids, y_poma


def load_cached_ntu_full_result() -> MetricResult:
    df = pd.read_csv(REPO_ROOT / "Tangent_Vector" / "results" / "esvae_clf_metrics.csv")
    row = df.iloc[0]
    return MetricResult(
        mean=float(row["F1 (macro) mean"]),
        ci_low=float(row["F1 (macro) ci_low"]),
        ci_high=float(row["F1 (macro) ci_high"]),
        source="cached_full_alignment",
    )


def load_cached_stroke_full_result(stroke_root: Path) -> MetricResult:
    readme = (stroke_root / "README.md").read_text()
    pattern = re.compile(
        r"\| Tangent Vector \| \*\*ES-VAE \+ k-NN \(proposed\)\*\* \| "
        r"\*\*[^|]+\*\* \| "
        r"\*\*(?P<mean>\d+\.\d+) \((?P<lo>\d+\.\d+), (?P<hi>\d+\.\d+)\)\*\* \|"
    )
    match = pattern.search(readme)
    if not match:
        return MetricResult(mean=2.82, ci_low=2.29, ci_high=3.21, source="cached_full_alignment_fallback")
    return MetricResult(
        mean=float(match.group("mean")),
        ci_low=float(match.group("lo")),
        ci_high=float(match.group("hi")),
        source="cached_full_alignment",
    )


def build_rows(
    ntu_results: dict[str, MetricResult | None],
    stroke_results: dict[str, MetricResult | None],
) -> pd.DataFrame:
    rows = []
    for label in ROW_LABELS:
        ntu = ntu_results.get(label)
        stroke = stroke_results.get(label)
        rows.append(
            {
                "alignment_stage": label,
                "ntu_macro_f1": None if ntu is None else ntu.mean,
                "ntu_macro_f1_ci_low": None if ntu is None else ntu.ci_low,
                "ntu_macro_f1_ci_high": None if ntu is None else ntu.ci_high,
                "stroke_rmse": None if stroke is None else stroke.mean,
                "stroke_rmse_ci_low": None if stroke is None else stroke.ci_low,
                "stroke_rmse_ci_high": None if stroke is None else stroke.ci_high,
                "ntu_source": None if ntu is None else ntu.source,
                "stroke_source": None if stroke is None else stroke.source,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stroke-root", type=Path, default=DEFAULT_STROKE_ROOT)
    ap.add_argument("--output-csv", type=Path, default=REPO_ROOT / "alignment_ablation_esvae.csv")
    ap.add_argument("--cache-dir", type=Path, default=REPO_ROOT / CACHE_DIR_NAME)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--compute-missing", action="store_true", help="Train the non-full rows if their cache JSONs do not exist.")
    ap.add_argument("--force-recompute", action="store_true")
    ap.add_argument("--max-folds", type=int, default=None, help="Optional quick-run cap for CV folds.")
    ap.add_argument("--ntu-epochs", type=int, default=150)
    ap.add_argument("--stroke-epochs", type=int, default=25)
    args = ap.parse_args()

    cache_dir = args.cache_dir
    device = get_device(args.device)

    ntu_results: dict[str, MetricResult | None] = {}
    stroke_results: dict[str, MetricResult | None] = {}

    # Always include the already-available full-alignment rows.
    ntu_results["Full alignment (Kendall + TSRVF)"] = load_cached_ntu_full_result()
    stroke_results["Full alignment (Kendall + TSRVF)"] = load_cached_stroke_full_result(args.stroke_root)

    stage_specs = [
        (
            "No alignment (raw coordinates)",
            cache_dir / "ntu_raw_subject_macro_f1.json",
            cache_dir / "stroke_raw_subject_rmse.json",
            lambda: evaluate_ntu_stage(*load_ntu_raw_bundle(cache_dir), device=device, epochs=args.ntu_epochs, max_folds=args.max_folds),
            lambda: evaluate_stroke_stage(*load_stroke_raw_curves(args.stroke_root, cache_dir), device=device, epochs=args.stroke_epochs, max_folds=args.max_folds),
        ),
        (
            "Translation removal (centering)",
            cache_dir / "ntu_centering_subject_macro_f1.json",
            cache_dir / "stroke_centering_subject_rmse.json",
            lambda: evaluate_ntu_stage(*load_ntu_centering_bundle(cache_dir), device=device, epochs=args.ntu_epochs, max_folds=args.max_folds),
            lambda: evaluate_stroke_stage(*load_stroke_centering_curves(args.stroke_root, cache_dir), device=device, epochs=args.stroke_epochs, max_folds=args.max_folds),
        ),
        (
            "Translation + scale removal (preshape)",
            cache_dir / "ntu_preshape_subject_macro_f1.json",
            cache_dir / "stroke_preshape_subject_rmse.json",
            lambda: evaluate_ntu_stage(*load_ntu_preshape_bundle(), device=device, epochs=args.ntu_epochs, max_folds=args.max_folds),
            lambda: evaluate_stroke_stage(*load_stroke_preshape_curves(args.stroke_root, cache_dir), device=device, epochs=args.stroke_epochs, max_folds=args.max_folds),
        ),
        (
            "Translation + scale + rotation removal (Kendall shape)",
            cache_dir / "ntu_kendall_shape_subject_macro_f1.json",
            cache_dir / "stroke_kendall_shape_subject_rmse.json",
            lambda: evaluate_ntu_stage(*load_ntu_kendall_bundle(cache_dir), device=device, epochs=args.ntu_epochs, max_folds=args.max_folds),
            lambda: evaluate_stroke_stage(*load_or_build_stroke_kendall_shape(args.stroke_root, cache_dir), device=device, epochs=args.stroke_epochs, max_folds=args.max_folds),
        ),
    ]

    for label, ntu_json, stroke_json, ntu_fn, stroke_fn in stage_specs:
        if ntu_json.exists() and not args.force_recompute:
            ntu_results[label] = load_json_result(ntu_json)
        elif args.compute_missing:
            result = ntu_fn()
            save_json_result(ntu_json, result)
            ntu_results[label] = result
        else:
            ntu_results[label] = None

        if stroke_json.exists() and not args.force_recompute:
            stroke_results[label] = load_json_result(stroke_json)
        elif args.compute_missing:
            result = stroke_fn()
            save_json_result(stroke_json, result)
            stroke_results[label] = result
        else:
            stroke_results[label] = None

    df = build_rows(ntu_results, stroke_results)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False, float_format="%.2f")
    print(f"Wrote {args.output_csv}")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))


def load_ntu_centering_bundle(cache_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    curves = load_ntu_centering_curves(cache_dir)
    y, subjects, _ = load_ntu_metadata()
    return curves, subjects, y


def load_ntu_raw_bundle(cache_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    curves = load_ntu_raw_curves(cache_dir)
    y, subjects, _ = load_ntu_metadata()
    return curves, subjects, y


def load_ntu_preshape_bundle() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    curves = load_ntu_preshape_curves()
    y, subjects, _ = load_ntu_metadata()
    return curves, subjects, y


def load_ntu_kendall_bundle(cache_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    curves = load_or_build_ntu_kendall_shape(cache_dir)
    y, subjects, _ = load_ntu_metadata()
    return curves, subjects, y


if __name__ == "__main__":
    main()
