"""Shared utilities for raw-skeleton NTU 5-class action classification.

- load_data(): reads `data/data_ntu.pkl` (dict[str -> (25, 3, T_var)] of
  raw NTU joint positions), linearly resamples each sample to a common
  T (default 100) along the time axis, and returns:
    raw      : (N, 25, 3, T) float32
    X_seq    : (N, 75, T)    float32 — sequence input for TCN/LSTM/etc.
    X_flat   : (N, 25*3*T)   float32 — flat input for VAE / PCA
    y        : (N,) int64    — class index in [0, 5)
    subjects : (N,) int64    — NTU subject id
    classes  : list[str] CLASS_ORDER
- leave_5_subjects_out_folds(): identical partition to Tangent_Vector
  (deterministic shuffle with seed=42, 14 folds: 13×5 + 1×4 subjects).
- subject_bootstrap_ci_class(), classwise_report(), metrics_summary_df()
  re-exported from Tangent_Vector/cv_utils so both pipelines share helpers.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "data_ntu.pkl"

# Reuse helpers from the Tangent_Vector pipeline (same CV partition,
# bootstrap, and reporting). We load the file under an explicit module
# name to avoid colliding with this very module.
import importlib.util as _ilu  # noqa: E402

_TV_CV_PATH = REPO_ROOT / "Tangent_Vector" / "cv_utils.py"
_spec = _ilu.spec_from_file_location("tv_cv_utils", _TV_CV_PATH)
_tv_cv = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_tv_cv)

CLASS_NAMES = _tv_cv.CLASS_NAMES
CLASS_ORDER = _tv_cv.CLASS_ORDER
classwise_report = _tv_cv.classwise_report
fold_indices = _tv_cv.fold_indices
leave_5_subjects_out_folds = _tv_cv.leave_5_subjects_out_folds
metrics_summary_df = _tv_cv.metrics_summary_df
subject_bootstrap_ci_class = _tv_cv.subject_bootstrap_ci_class

CLASS_TO_INT = {c: i for i, c in enumerate(CLASS_ORDER)}


def linear_resample_time(arr: np.ndarray, T: int) -> np.ndarray:
    """Linearly interpolate `arr` (shape (..., T_orig)) to length T along
    the last axis. Operates per-channel; vectorised over leading dims."""
    arr = np.asarray(arr, dtype=np.float32)
    T_orig = arr.shape[-1]
    if T_orig == T:
        return arr.copy()
    src = np.linspace(0.0, 1.0, T_orig, dtype=np.float64)
    dst = np.linspace(0.0, 1.0, T,      dtype=np.float64)
    flat = arr.reshape(-1, T_orig).astype(np.float64)
    # piecewise-linear interp via vectorised np.interp loop (T_orig small)
    out = np.empty((flat.shape[0], T), dtype=np.float32)
    for i in range(flat.shape[0]):
        out[i] = np.interp(dst, src, flat[i]).astype(np.float32)
    return out.reshape(*arr.shape[:-1], T)


def load_data(T: int = 100):
    """Load raw NTU skeletons and produce (raw, X_seq, X_flat, y, subj, classes)."""
    with open(DATA_PATH, "rb") as fh:
        d = pickle.load(fh)

    keys = list(d.keys())
    N = len(keys)
    raw = np.empty((N, 25, 3, T), dtype=np.float32)
    y = np.empty(N, dtype=np.int64)
    subj = np.empty(N, dtype=np.int64)

    for i, k in enumerate(keys):
        pid_str, cls = k.split("_")
        subj[i] = int(pid_str)
        y[i] = CLASS_TO_INT[cls]
        raw[i] = linear_resample_time(d[k], T)

    X_seq = raw.reshape(N, 25 * 3, T).astype(np.float32)
    X_flat = raw.reshape(N, -1).astype(np.float32)
    return raw, X_seq, X_flat, y, subj, CLASS_ORDER


__all__ = [
    "load_data",
    "linear_resample_time",
    "leave_5_subjects_out_folds",
    "fold_indices",
    "subject_bootstrap_ci_class",
    "classwise_report",
    "metrics_summary_df",
    "CLASS_NAMES",
    "CLASS_ORDER",
]
