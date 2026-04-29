"""Shared utilities for NTU 5-class action classification on tangent vectors.

- load_data(): returns (tangent_vec_all, betas_aligned, mu, X_man, y, subject_ids,
              class_names) using `aligned_data/tangent_vecs100.pkl`,
              `aligned_data/betas_aligned100.pkl`, `aligned_data/mu100.pkl`,
              and `aligned_data/sample_index.csv`.
- leave_5_subjects_out_folds(): deterministic 14-fold L5SO over 69 subjects
  (folds 0-12 hold 5 subjects each, fold 13 holds 4).
- subject_bootstrap_ci_class(): subject-level bootstrap CI for classification
  metrics (mirrors the helper in stroke_riemann/Tangent_Vector/ci_class.py).
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ALIGNED_DIR = REPO_ROOT / "aligned_data"

CLASS_ORDER = ["A001", "A002", "A003", "A004", "A008",
               "A009", "A023", "A028", "A029", "A031"]
CLASS_NAMES = {
    "A001": "drink_water",
    "A002": "eat_meal",
    "A003": "brush_teeth",
    "A004": "brush_hair",
    "A008": "sitting_down",
    "A009": "standing_up",
    "A023": "hand_waving",
    "A028": "phone_call",
    "A029": "play_with_phone",
    "A031": "pointing_to_something",
}


def load_data(tslen: int = 100):
    """Load tangent vectors, aligned manifold curves, mean shape and labels.

    Returns
    -------
    tangent : np.ndarray (K, M, T, N)  — N=345 samples on the last axis
    betas   : np.ndarray (N, K, M, T)  — aligned curves
    mu      : np.ndarray (K, M, T)
    X_man   : np.ndarray (N, K*M*T)    — flattened manifold curves
    y       : np.ndarray (N,) int64    — class index in [0, 5)
    subjects: np.ndarray (N,) int64    — NTU subject id (e.g. 8, 41, ...)
    classes : list[str] CLASS_ORDER
    """
    with open(ALIGNED_DIR / f"tangent_vecs{tslen}.pkl", "rb") as fh:
        tangent = np.asarray(pickle.load(fh), dtype=np.float32)
    with open(ALIGNED_DIR / f"betas_aligned{tslen}.pkl", "rb") as fh:
        betas_list = pickle.load(fh)
    with open(ALIGNED_DIR / f"mu{tslen}.pkl", "rb") as fh:
        mu = np.asarray(pickle.load(fh), dtype=np.float32)

    betas = np.stack(betas_list, axis=0).astype(np.float32)  # (N, K, M, T)

    idx_df = pd.read_csv(ALIGNED_DIR / "sample_index.csv")
    idx_df = idx_df.sort_values("sample_index").reset_index(drop=True)
    class_to_int = {c: i for i, c in enumerate(CLASS_ORDER)}
    y = idx_df["class_id"].map(class_to_int).to_numpy(dtype=np.int64)
    subjects = idx_df["person_id"].to_numpy(dtype=np.int64)

    N = len(y)
    assert tangent.shape[-1] == N == betas.shape[0], (
        f"shape mismatch tangent={tangent.shape}, betas={betas.shape}, N={N}"
    )

    X_man = betas.reshape(N, -1).astype(np.float32)
    return tangent, betas, mu, X_man, y, subjects, CLASS_ORDER


def leave_5_subjects_out_folds(subject_ids: np.ndarray, seed: int = 42, fold_size: int = 5):
    """Deterministic L5SO partition.

    Returns
    -------
    folds : list of np.ndarray, each holding the subject IDs in that test fold.
    """
    rng = np.random.default_rng(seed)
    unique = np.array(sorted(np.unique(subject_ids).tolist()), dtype=np.int64)
    perm = rng.permutation(unique)
    folds = []
    for start in range(0, len(perm), fold_size):
        folds.append(perm[start : start + fold_size])
    return folds


def fold_indices(subject_ids: np.ndarray, test_subjects: np.ndarray):
    """Return (train_idx, test_idx) for a given test-subject set."""
    test_mask = np.isin(subject_ids, test_subjects)
    test_idx = np.where(test_mask)[0]
    train_idx = np.where(~test_mask)[0]
    return train_idx, test_idx


def load_camera_ids() -> np.ndarray:
    """Per-sample camera id (1/2/3) in the canonical 345-row order."""
    df = pd.read_csv(ALIGNED_DIR / "sample_meta.csv")
    df = df.sort_values("sample_index").reset_index(drop=True)
    return df["camera_id"].to_numpy(dtype=np.int64)


def cross_view_folds(camera_ids: np.ndarray):
    """Leave-one-camera-out: 3 folds. Each fold's 'test set' is one camera id."""
    return [np.array([c], dtype=np.int64) for c in sorted(np.unique(camera_ids).tolist())]


def load_setup_ids() -> np.ndarray:
    """Per-sample NTU setup id (S001-S032) in canonical 345-row order."""
    df = pd.read_csv(ALIGNED_DIR / "sample_meta.csv")
    df = df.sort_values("sample_index").reset_index(drop=True)
    return df["setup_id"].to_numpy(dtype=np.int64)


def cross_setup_folds(setup_ids: np.ndarray):
    """Leave-one-setup-out: one fold per distinct NTU setup that appears in
    the curated subset. NTU setups differ in room geometry / camera placement
    / lighting, so this is a stronger generalisation test than cross-view."""
    return [np.array([s], dtype=np.int64) for s in sorted(np.unique(setup_ids).tolist())]


def get_folds_and_axis(mode: str, subj: np.ndarray, seed: int = 42):
    """Returns (folds, fold_axis_array, label_for_logging).

    mode: 'subject' -> 14 L5SO folds; fold_axis_array is `subj`.
    mode: 'view'    -> 3 leave-one-camera-out folds; fold_axis_array is camera_ids.
    mode: 'setup'   -> N leave-one-setup-out folds (N = distinct setups present);
                       fold_axis_array is setup_ids.
    """
    if mode == "subject":
        return leave_5_subjects_out_folds(subj, seed=seed), subj, "subject (L5SO)"
    if mode == "view":
        cameras = load_camera_ids()
        if len(cameras) != len(subj):
            raise RuntimeError(f"camera_ids length {len(cameras)} != subj length {len(subj)}")
        return cross_view_folds(cameras), cameras, "view (leave-one-camera-out)"
    if mode == "setup":
        setups = load_setup_ids()
        if len(setups) != len(subj):
            raise RuntimeError(f"setup_ids length {len(setups)} != subj length {len(subj)}")
        return cross_setup_folds(setups), setups, "setup (leave-one-setup-out)"
    raise ValueError(f"unknown cv mode {mode!r}")


def subject_bootstrap_ci_class(
    targets,
    preds,
    subject_ids,
    n_bootstrap: int = 2000,
    ci: int = 95,
    random_state: int = 42,
):
    """Subject-level bootstrap CI for pooled OOF classification predictions."""
    rng = np.random.default_rng(random_state)
    targets = np.asarray(targets)
    preds = np.asarray(preds)
    subject_ids = np.asarray(subject_ids)
    unique_subjects = np.unique(subject_ids)
    n_subjects = len(unique_subjects)

    boots = {k: [] for k in [
        "Accuracy", "F1 (weighted)", "F1 (macro)",
        "Precision (weighted)", "Precision (macro)",
        "Recall (weighted)", "Recall (macro)",
    ]}

    for _ in range(n_bootstrap):
        sampled = rng.choice(unique_subjects, size=n_subjects, replace=True)
        # gather every sample whose subject is in the resample (with multiplicity)
        idx_lists = [np.where(subject_ids == s)[0] for s in sampled]
        if not idx_lists:
            continue
        idx = np.concatenate(idx_lists)
        t, p = targets[idx], preds[idx]
        if len(t) < 2 or len(np.unique(t)) < 2:
            continue
        boots["Accuracy"].append(accuracy_score(t, p))
        boots["F1 (weighted)"].append(f1_score(t, p, average="weighted", zero_division=0))
        boots["F1 (macro)"].append(f1_score(t, p, average="macro", zero_division=0))
        boots["Precision (weighted)"].append(precision_score(t, p, average="weighted", zero_division=0))
        boots["Precision (macro)"].append(precision_score(t, p, average="macro", zero_division=0))
        boots["Recall (weighted)"].append(recall_score(t, p, average="weighted", zero_division=0))
        boots["Recall (macro)"].append(recall_score(t, p, average="macro", zero_division=0))

    alpha = (100 - ci) / 2

    def interval(x):
        if not x:
            return np.array([np.nan, np.nan])
        return np.round(np.percentile(x, [alpha, 100 - alpha]), 4)

    point = {
        "Accuracy": accuracy_score(targets, preds),
        "F1 (weighted)": f1_score(targets, preds, average="weighted", zero_division=0),
        "F1 (macro)": f1_score(targets, preds, average="macro", zero_division=0),
        "Precision (weighted)": precision_score(targets, preds, average="weighted", zero_division=0),
        "Precision (macro)": precision_score(targets, preds, average="macro", zero_division=0),
        "Recall (weighted)": recall_score(targets, preds, average="weighted", zero_division=0),
        "Recall (macro)": recall_score(targets, preds, average="macro", zero_division=0),
    }
    return {k: {"mean": float(point[k]), "ci": interval(boots[k])} for k in point}


def classwise_report(targets, preds, class_names):
    """Per-class precision/recall/F1 (and support) as a DataFrame."""
    targets = np.asarray(targets)
    preds = np.asarray(preds)
    rows = []
    for k, name in enumerate(class_names):
        mask = targets == k
        support = int(mask.sum())
        tp = int(((preds == k) & mask).sum())
        fp = int(((preds == k) & ~mask).sum())
        fn = int(((preds != k) & mask).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        rows.append({
            "class": name,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": support,
        })
    return pd.DataFrame(rows)


def format_metric(mean: float, lo: float, hi: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def metrics_summary_df(method_name: str, ci_dict: dict) -> pd.DataFrame:
    """Flatten a CI dict into a tidy 1-row DataFrame for combining results."""
    flat = {"method": method_name}
    for metric, vals in ci_dict.items():
        m = vals["mean"]
        lo, hi = vals["ci"]
        flat[f"{metric} mean"] = round(float(m), 4)
        flat[f"{metric} ci_low"] = round(float(lo), 4)
        flat[f"{metric} ci_high"] = round(float(hi), 4)
    return pd.DataFrame([flat])


__all__ = [
    "load_data",
    "leave_5_subjects_out_folds",
    "fold_indices",
    "subject_bootstrap_ci_class",
    "classwise_report",
    "metrics_summary_df",
    "format_metric",
    "CLASS_ORDER",
    "CLASS_NAMES",
]
