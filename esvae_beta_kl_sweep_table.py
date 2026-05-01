#!/usr/bin/env python
"""Sweep ES-VAE beta_kl on NTU and stroke, then write CSV tables.

Outputs:
  - beta_kl_sweep/esvae_beta_kl_sweep_long.csv
  - beta_kl_sweep/esvae_beta_kl_sweep_table.csv
  - beta_kl_sweep/cache/*.json

The display CSV uses two metric rows plus two beta-label rows because the
requested NTU and stroke beta grids are different and CSV has only one header.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor


REPO_ROOT = Path(__file__).resolve().parent
STROKE_ROOT = Path("/mnt/sdb/arafat/stroke_riemann")
OUT_DIR = REPO_ROOT / "beta_kl_sweep"
CACHE_DIR = OUT_DIR / "cache"
LATEX_OUT = REPO_ROOT / "latex_results_table.txt"

NTU_BETAS = [
    ("10^-5", 1e-5),
    ("10^-4", 1e-4),
    ("10^-3", 1e-3),
    ("10^-2", 1e-2),
    ("10^-1", 1e-1),
]
STROKE_BETAS = [
    ("2^-5", 2 ** (-5)),
    ("2^-4", 2 ** (-4)),
    ("2^-3", 2 ** (-3)),
    ("2^-2", 2 ** (-2)),
    ("2^-1", 2 ** (-1)),
]

SEED = 42


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def import_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def round_metric(value: float) -> float:
    if not np.isfinite(value):
        return float("nan")
    return round(float(value), 2)


def latex_num(value: float) -> str:
    if not np.isfinite(value):
        return "--"
    return f"{float(value):.2f}"


def latex_beta(label: str) -> str:
    if "^" not in label:
        return label
    base, exponent = label.split("^", 1)
    return f"{base}^{{{exponent}}}"


def write_latex_table(ntu_scores: dict[str, float], stroke_scores: dict[str, float]) -> None:
    ntu_metric_row = " & ".join(latex_num(ntu_scores.get(label, np.nan)) for label, _beta in NTU_BETAS)
    stroke_metric_row = " & ".join(latex_num(stroke_scores.get(label, np.nan)) for label, _beta in STROKE_BETAS)
    ntu_label_row = " & ".join(f"${latex_beta(label)}$" for label, _beta in NTU_BETAS)
    stroke_label_row = " & ".join(f"${latex_beta(label)}$" for label, _beta in STROKE_BETAS)

    latex = "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Effect of the KL-weight $\beta$ on ES-VAE performance for NTU-60 action recognition and the stroke dataset. NTU-60 is reported as Macro F1 and stroke is reported as $R^2$.}",
            r"\label{tab:esvae_beta_sweep}",
            r"\begin{tabular}{lccccc}",
            r"\hline",
            r"Metric & $\beta_1$ & $\beta_2$ & $\beta_3$ & $\beta_4$ & $\beta_5$ \\",
            r"\hline",
            f"NTU beta labels & {ntu_label_row} \\\\",
            f"NTU Macro F1 & {ntu_metric_row} \\\\",
            f"Stroke beta labels & {stroke_label_row} \\\\",
            f"Stroke $R^2$ & {stroke_metric_row} \\\\",
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    LATEX_OUT.write_text(latex + "\n")


def write_csvs(ntu_scores: dict[str, float], stroke_scores: dict[str, float]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    long_rows = []
    for label, beta in NTU_BETAS:
        long_rows.append(
            {
                "dataset": "NTU-60",
                "metric": "Macro F1",
                "beta_label": label,
                "beta_value": beta,
                "value": round_metric(ntu_scores.get(label, np.nan)),
            }
        )
    for label, beta in STROKE_BETAS:
        long_rows.append(
            {
                "dataset": "Stroke",
                "metric": "R2",
                "beta_label": label,
                "beta_value": beta,
                "value": round_metric(stroke_scores.get(label, np.nan)),
            }
        )
    pd.DataFrame(long_rows).to_csv(OUT_DIR / "esvae_beta_kl_sweep_long.csv", index=False)

    display_rows = [
        {
            "row": "NTU beta labels",
            "beta_1": NTU_BETAS[0][0],
            "beta_2": NTU_BETAS[1][0],
            "beta_3": NTU_BETAS[2][0],
            "beta_4": NTU_BETAS[3][0],
            "beta_5": NTU_BETAS[4][0],
        },
        {
            "row": "NTU Macro F1",
            "beta_1": round_metric(ntu_scores.get(NTU_BETAS[0][0], np.nan)),
            "beta_2": round_metric(ntu_scores.get(NTU_BETAS[1][0], np.nan)),
            "beta_3": round_metric(ntu_scores.get(NTU_BETAS[2][0], np.nan)),
            "beta_4": round_metric(ntu_scores.get(NTU_BETAS[3][0], np.nan)),
            "beta_5": round_metric(ntu_scores.get(NTU_BETAS[4][0], np.nan)),
        },
        {
            "row": "Stroke beta labels",
            "beta_1": STROKE_BETAS[0][0],
            "beta_2": STROKE_BETAS[1][0],
            "beta_3": STROKE_BETAS[2][0],
            "beta_4": STROKE_BETAS[3][0],
            "beta_5": STROKE_BETAS[4][0],
        },
        {
            "row": "Stroke R2",
            "beta_1": round_metric(stroke_scores.get(STROKE_BETAS[0][0], np.nan)),
            "beta_2": round_metric(stroke_scores.get(STROKE_BETAS[1][0], np.nan)),
            "beta_3": round_metric(stroke_scores.get(STROKE_BETAS[2][0], np.nan)),
            "beta_4": round_metric(stroke_scores.get(STROKE_BETAS[3][0], np.nan)),
            "beta_5": round_metric(stroke_scores.get(STROKE_BETAS[4][0], np.nan)),
        },
    ]
    pd.DataFrame(display_rows).to_csv(OUT_DIR / "esvae_beta_kl_sweep_table.csv", index=False)
    write_latex_table(ntu_scores, stroke_scores)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def load_cached_ntu_scores() -> dict[str, float]:
    scores: dict[str, float] = {}
    for label, _beta in NTU_BETAS:
        path = CACHE_DIR / f"ntu_beta_{label.replace('^', '').replace('-', 'm')}.json"
        if not path.exists():
            continue
        data = load_json(path)
        if "macro_f1" in data:
            scores[label] = float(data["macro_f1"])
    return scores


def load_cached_stroke_scores() -> dict[str, float]:
    scores: dict[str, float] = {}
    for label, _beta in STROKE_BETAS:
        path = CACHE_DIR / f"stroke_beta_{label.replace('^', '').replace('-', 'm')}.json"
        if not path.exists():
            continue
        data = load_json(path)
        if "r2" in data:
            scores[label] = float(data["r2"])
    return scores


@dataclass
class NTUContext:
    mod: object
    tangent: np.ndarray
    X_man_np: np.ndarray
    y: np.ndarray
    subj: np.ndarray
    folds: list[np.ndarray]
    fold_axis: np.ndarray
    mu_shape: torch.Tensor
    K: int
    M: int
    T: int
    device: torch.device
    dtype: torch.dtype
    enc_cfg: dict
    knn_cfg: dict


def build_ntu_context(device_str: str) -> NTUContext:
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "Tangent_Vector"))
    ntu_mod = import_module_from_path("ntu_esvae_clf_sweep", REPO_ROOT / "Tangent_Vector" / "esvae_clf.py")

    cfg = load_json(REPO_ROOT / "Tangent_Vector" / "results" / "esvae_clf_config.json")
    knn_cfg = load_json(REPO_ROOT / "Tangent_Vector" / "results" / "best_knn_cfg.json")
    tangent, _betas, mu_arr, X_man_np, y, subj, _view = ntu_mod.load_data(100)
    folds, fold_axis, _label = ntu_mod.get_folds_and_axis("subject", subj)

    K, M, T, _ = tangent.shape
    device = torch.device(device_str)
    dtype = torch.float32
    mu_shape = torch.from_numpy(mu_arr.reshape(-1).astype(np.float32)).to(device=device, dtype=dtype)
    return NTUContext(
        mod=ntu_mod,
        tangent=tangent,
        X_man_np=X_man_np,
        y=y,
        subj=subj,
        folds=folds,
        fold_axis=fold_axis,
        mu_shape=mu_shape,
        K=K,
        M=M,
        T=T,
        device=device,
        dtype=dtype,
        enc_cfg=cfg["encoder"],
        knn_cfg=knn_cfg,
    )


def run_ntu_beta(ctx: NTUContext, beta_label: str, beta_value: float, force: bool = False) -> float:
    cache_path = CACHE_DIR / f"ntu_beta_{beta_label.replace('^', '').replace('-', 'm')}.json"
    if cache_path.exists() and not force:
        return float(load_json(cache_path)["macro_f1"])

    enc_cfg = dict(ctx.enc_cfg)
    enc_cfg["beta_kl"] = beta_value

    pooled, summary = ctx.mod.run_cv(
        enc_cfg=enc_cfg,
        knn_grid=[ctx.knn_cfg],
        tangent=ctx.tangent,
        X_man_np=ctx.X_man_np,
        mu_shape=ctx.mu_shape,
        y=ctx.y,
        subj=ctx.subj,
        folds=ctx.folds,
        K=ctx.K,
        M=ctx.M,
        T=ctx.T,
        device=ctx.device,
        dtype=ctx.dtype,
        fold_axis=ctx.fold_axis,
    )
    key = ctx.mod._knn_key(ctx.knn_cfg)
    macro_f1 = float(summary[key]["macroF1"])
    save_json(
        cache_path,
        {
            "dataset": "NTU-60",
            "metric": "Macro F1",
            "beta_label": beta_label,
            "beta_value": beta_value,
            "macro_f1": macro_f1,
            "knn_cfg": ctx.knn_cfg,
            "encoder_cfg": enc_cfg,
            "n_predictions": len(pooled[key]["preds"]),
        },
    )
    return macro_f1


@dataclass
class StrokeContext:
    fg: object
    val_test_mod: object
    tangent_flat: np.ndarray
    betas_flat: np.ndarray
    y_poma: np.ndarray
    participant_ids: np.ndarray
    mu_shape: torch.Tensor
    K: int
    M: int
    T: int
    device: torch.device
    dtype: torch.dtype


class StrokeNonlinearVAE(nn.Module):
    def __init__(self, D: int, R: int, H: int = 128, dropout: float = 0.1):
        super().__init__()
        self.W1 = nn.Linear(D, H, bias=False)
        self.W2_mu = nn.Linear(H, R, bias=False)
        self.W2_logvar = nn.Linear(H, R)
        self.dropout = nn.Dropout(p=dropout)
        self.dec1 = nn.Linear(R, 16, bias=False)
        self.dec2 = nn.Linear(16, D, bias=False)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.tanh(self.W1(x))
        h = self.dropout(h)
        mu = self.W2_mu(h)
        logvar = self.W2_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h_recon = torch.tanh(self.dec1(z))
        return self.dec2(h_recon)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar, z


class StrokeESVAE(nn.Module):
    def __init__(self, base_vae: StrokeNonlinearVAE, mu_shape: torch.Tensor, exp_map_fn):
        super().__init__()
        self.vae = base_vae
        self.exp_map_fn = exp_map_fn
        self.register_buffer("mu_shape", mu_shape)

    def forward(self, x: torch.Tensor):
        x_hat, mu, logvar, z = self.vae(x)
        v_hat = x_hat.view(x.shape[0], 32, 3, 200)
        x_hat_man = self.exp_map_fn(self.mu_shape.view(32, 3, 200), v_hat).view(x.shape[0], -1)
        return x_hat_man, mu, logvar, z, v_hat


def stroke_esvae_loss(fg, x_man: torch.Tensor, x_hat_man: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, K: int, M: int, T: int, beta: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dist = fg.squared_geodesic_distance(x_man, x_hat_man, K, M, T)
    recon = torch.mean(dist.sum(dim=1))
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    avg_kl = kl.mean()
    return recon + beta * avg_kl, recon, avg_kl


def build_stroke_context(device_str: str) -> StrokeContext:
    fg = import_module_from_path("stroke_functionsgpu_fast_sweep", STROKE_ROOT / "Tangent_Vector" / "functionsgpu_fast.py")
    val_test_mod = import_module_from_path("stroke_val_test_sweep", STROKE_ROOT / "Tangent_Vector" / "val_test.py")

    with open(STROKE_ROOT / "aligned_data" / "betas_aligned200.pkl", "rb") as f:
        betas_aligned = pickle.load(f)
    with open(STROKE_ROOT / "aligned_data" / "mu200.pkl", "rb") as f:
        mu_all_t = pickle.load(f)
    with open(STROKE_ROOT / "aligned_data" / "tangent_vecs200.pkl", "rb") as f:
        tangent_vec_all = pickle.load(f)

    y_poma = np.loadtxt(STROKE_ROOT / "labels_data" / "y_poma.txt").astype(np.float32)
    participant_ids = np.loadtxt(STROKE_ROOT / "labels_data" / "pids.txt").astype(int)

    betas = np.array(betas_aligned).astype(np.float32)
    K, M, T = betas.shape[1:]
    tangent_flat = tangent_vec_all.reshape((K * M * T, len(participant_ids))).T.astype(np.float32)
    betas_flat = betas.reshape(betas.shape[0], -1).astype(np.float32)

    device = torch.device(device_str)
    dtype = torch.float32
    mu_shape = torch.from_numpy(mu_all_t.reshape(-1).astype(np.float32)).to(device=device, dtype=dtype)
    return StrokeContext(
        fg=fg,
        val_test_mod=val_test_mod,
        tangent_flat=tangent_flat,
        betas_flat=betas_flat,
        y_poma=y_poma,
        participant_ids=participant_ids,
        mu_shape=mu_shape,
        K=K,
        M=M,
        T=T,
        device=device,
        dtype=dtype,
    )


def train_stroke_esvae_fold(ctx: StrokeContext, X_tan_train: torch.Tensor, X_man_train: torch.Tensor, beta_kl: float, seed: int, R: int = 38, num_epochs: int = 25, lr: float = 1e-3) -> StrokeESVAE:
    set_deterministic(seed)
    D = X_tan_train.shape[1]
    base_vae = StrokeNonlinearVAE(D, R).to(device=ctx.device, dtype=ctx.dtype)
    model = StrokeESVAE(base_vae, ctx.mu_shape, ctx.fg.exp_gpu_batch).to(device=ctx.device, dtype=ctx.dtype)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(num_epochs):
        opt.zero_grad(set_to_none=True)
        x_hat_man, mu, logvar, _z, _v_hat = model(X_tan_train)
        loss, _recon, _kl = stroke_esvae_loss(ctx.fg, X_man_train, x_hat_man, mu, logvar, ctx.K, ctx.M, ctx.T, beta_kl)
        loss.backward()
        opt.step()
    model.eval()
    return model


def run_stroke_beta(ctx: StrokeContext, beta_label: str, beta_value: float, force: bool = False) -> float:
    cache_path = CACHE_DIR / f"stroke_beta_{beta_label.replace('^', '').replace('-', 'm')}.json"
    if cache_path.exists() and not force:
        data = load_json(cache_path)
        if "r2" in data:
            return float(data["r2"])

    X_tan = torch.from_numpy(ctx.tangent_flat).to(device=ctx.device, dtype=ctx.dtype)
    X_man = torch.from_numpy(ctx.betas_flat).to(device=ctx.device, dtype=ctx.dtype)

    pooled_targets: list[float] = []
    pooled_preds: list[float] = []
    for fold_idx in range(30):
        val_pids, test_pids = ctx.val_test_mod.val_test(ctx.participant_ids, fold_idx)
        validation_pids = set(np.asarray(val_pids).tolist())
        test_pids = set(np.asarray(test_pids).tolist())
        train_pids = set(ctx.participant_ids.tolist()) - validation_pids - test_pids

        train_idx = np.array([j for j in range(len(ctx.y_poma)) if int(ctx.participant_ids[j]) in train_pids], dtype=np.int64)
        test_idx = np.array([j for j in range(len(ctx.y_poma)) if int(ctx.participant_ids[j]) in test_pids], dtype=np.int64)
        validation_idx = np.array([j for j in range(len(ctx.y_poma)) if int(ctx.participant_ids[j]) in validation_pids], dtype=np.int64)
        if len(train_idx) == 0 or len(test_idx) == 0 or len(validation_idx) == 0:
            continue

        fold_seed = SEED + fold_idx
        model = train_stroke_esvae_fold(
            ctx,
            X_tan_train=X_tan[train_idx],
            X_man_train=X_man[train_idx],
            beta_kl=beta_value,
            seed=fold_seed,
        )

        with torch.no_grad():
            mu_train, _ = model.vae.encode(X_tan[train_idx])
            mu_test, _ = model.vae.encode(X_tan[test_idx])
        Z_train = mu_train.detach().cpu().numpy()
        Z_test = mu_test.detach().cpu().numpy()

        knr = KNeighborsRegressor()
        knr.fit(Z_train, ctx.y_poma[train_idx])
        preds = knr.predict(Z_test)
        pooled_targets.extend(ctx.y_poma[test_idx].tolist())
        pooled_preds.extend(preds.tolist())
        print(f"  stroke fold {fold_idx + 1:02d}/30  beta={beta_label}  fold_rmse={np.sqrt(mean_squared_error(ctx.y_poma[test_idx], preds)):.3f}", flush=True)

    targets_np = np.asarray(pooled_targets, dtype=np.float32)
    preds_np = np.asarray(pooled_preds, dtype=np.float32)
    rmse = float(np.sqrt(mean_squared_error(targets_np, preds_np)))
    r2 = float(r2_score(targets_np, preds_np))
    save_json(
        cache_path,
        {
            "dataset": "Stroke",
            "metric": "R2",
            "beta_label": beta_label,
            "beta_value": beta_value,
            "rmse": rmse,
            "r2": r2,
            "epochs": 25,
            "R": 38,
            "lr": 1e-3,
            "n_predictions": len(pooled_preds),
        },
    )
    return r2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ntu-device", default="cuda:1")
    ap.add_argument("--stroke-device", default="cuda:1")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-ntu", action="store_true")
    ap.add_argument("--skip-stroke", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    ntu_scores: dict[str, float] = load_cached_ntu_scores()
    stroke_scores: dict[str, float] = load_cached_stroke_scores()

    if not args.skip_ntu:
        print(f"Building NTU context on {args.ntu_device}", flush=True)
        ntu_ctx = build_ntu_context(args.ntu_device)
        for label, beta in NTU_BETAS:
            print(f"Running NTU beta sweep: beta={label} ({beta})", flush=True)
            ntu_scores[label] = run_ntu_beta(ntu_ctx, label, beta, force=args.force)
            write_csvs(ntu_scores, stroke_scores)

    if not args.skip_stroke:
        print(f"Building stroke context on {args.stroke_device}", flush=True)
        stroke_ctx = build_stroke_context(args.stroke_device)
        for label, beta in STROKE_BETAS:
            print(f"Running stroke beta sweep: beta={label} ({beta})", flush=True)
            stroke_scores[label] = run_stroke_beta(stroke_ctx, label, beta, force=args.force)
            write_csvs(ntu_scores, stroke_scores)

    write_csvs(ntu_scores, stroke_scores)
    print(f"Wrote {OUT_DIR / 'esvae_beta_kl_sweep_table.csv'}", flush=True)
    print(f"Wrote {OUT_DIR / 'esvae_beta_kl_sweep_long.csv'}", flush=True)


if __name__ == "__main__":
    main()
