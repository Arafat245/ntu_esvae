from __future__ import annotations

import json
import math
import pickle
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from Tangent_Vector.cv_utils import (
    CLASS_ORDER,
    fold_indices,
    leave_5_subjects_out_folds,
    subject_bootstrap_ci_class,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ALIGNED_DIR = REPO_ROOT / "aligned_data"
DATA_PATH = REPO_ROOT / "data" / "data_ntu.pkl"


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def top_k_accuracy(scores: np.ndarray, labels: np.ndarray, ks=(1,)) -> list[float]:
    max_k = max(ks)
    topk = np.argsort(scores, axis=1)[:, ::-1][:, :max_k]
    out = []
    for k in ks:
        correct = (topk[:, :k] == labels[:, None]).any(axis=1)
        out.append(float(correct.mean()))
    return out


def cosine_lr(base_lr: float, epoch: int, total_epochs: int) -> float:
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * epoch / total_epochs))


class Graph:
    """Official PYSKL/ProtoGCN graph utility, reduced to the NTU case."""

    def __init__(
        self,
        layout: str = "nturgb+d",
        mode: str = "random",
        max_hop: int = 1,
        nx_node: int = 1,
        num_filter: int = 3,
        init_std: float = 0.02,
        init_off: float = 0.04,
    ):
        self.max_hop = max_hop
        self.layout = layout
        self.mode = mode
        self.num_filter = num_filter
        self.init_std = init_std
        self.init_off = init_off
        self.nx_node = nx_node
        self.get_layout(layout)
        self.A = getattr(self, mode)()

    def get_layout(self, layout: str) -> None:
        if layout != "nturgb+d":
            raise ValueError(f"Unsupported layout: {layout}")
        self.num_node = 25
        neighbor_base = [
            (1, 2),
            (2, 21),
            (3, 21),
            (4, 3),
            (5, 21),
            (6, 5),
            (7, 6),
            (8, 7),
            (9, 21),
            (10, 9),
            (11, 10),
            (12, 11),
            (13, 1),
            (14, 13),
            (15, 14),
            (16, 15),
            (17, 1),
            (18, 17),
            (19, 18),
            (20, 19),
            (22, 8),
            (23, 8),
            (24, 12),
            (25, 12),
        ]
        self.inward = [(i - 1, j - 1) for (i, j) in neighbor_base]

    def random(self) -> np.ndarray:
        num_node = self.num_node * self.nx_node
        return (
            np.random.randn(self.num_filter, num_node, num_node) * self.init_std
            + self.init_off
        ).astype(np.float32)


def _load_index_df() -> pd.DataFrame:
    df = pd.read_csv(ALIGNED_DIR / "sample_index.csv")
    df = df.sort_values("sample_index").reset_index(drop=True)
    return df


def _raw_key(pid: int, class_id: str) -> str:
    return f"{pid}_{class_id}"


def load_raw_sequences() -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    with open(DATA_PATH, "rb") as fh:
        raw_dict = pickle.load(fh)
    df = _load_index_df()
    sequences: list[np.ndarray] = []
    labels = []
    subjects = []
    class_to_int = {c: i for i, c in enumerate(CLASS_ORDER)}
    for row in df.itertuples(index=False):
        key = _raw_key(int(row.person_id), row.class_id)
        arr = np.asarray(raw_dict[key], dtype=np.float32)  # (25, 3, T)
        seq = arr.transpose(2, 0, 1)[None]  # (1, T, V, C)
        sequences.append(seq)
        labels.append(class_to_int[row.class_id])
        subjects.append(int(row.person_id))
    return sequences, np.asarray(labels, dtype=np.int64), np.asarray(subjects, dtype=np.int64)


def load_tangent_sequences() -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    with open(ALIGNED_DIR / "tangent_vecs100.pkl", "rb") as fh:
        tangent = np.asarray(pickle.load(fh), dtype=np.float32)  # (25, 3, 100, N)
    df = _load_index_df()
    labels = []
    subjects = []
    class_to_int = {c: i for i, c in enumerate(CLASS_ORDER)}
    moved = np.moveaxis(tangent, -1, 0)  # (N, 25, 3, 100)
    sequences = [sample.transpose(2, 0, 1)[None].astype(np.float32) for sample in moved]
    for row in df.itertuples(index=False):
        labels.append(class_to_int[row.class_id])
        subjects.append(int(row.person_id))
    return sequences, np.asarray(labels, dtype=np.int64), np.asarray(subjects, dtype=np.int64)


def _unit_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < 1e-6:
        return vector
    return vector / norm


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    if np.abs(v1).sum() < 1e-6 or np.abs(v2).sum() < 1e-6:
        return 0.0
    v1_u = _unit_vector(v1)
    v2_u = _unit_vector(v2)
    return float(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))


def _rotation_matrix(axis: np.ndarray, theta: float) -> np.ndarray:
    if np.abs(axis).sum() < 1e-6 or abs(theta) < 1e-6:
        return np.eye(3, dtype=np.float32)
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
        ],
        dtype=np.float32,
    )


def pre_normalize_3d(
    keypoint: np.ndarray,
    align_spine: bool = False,
    align_center: bool = True,
    zaxis=(0, 1),
    xaxis=(8, 4),
) -> np.ndarray:
    skeleton = keypoint.copy()
    m, t, v, c = skeleton.shape
    if skeleton.sum() == 0:
        return skeleton
    index0 = [i for i in range(t) if not np.all(np.isclose(skeleton[0, i], 0))]
    if not index0:
        return skeleton
    skeleton = skeleton[:, np.array(index0)]
    if align_center:
        main_body_center = skeleton[0, 0, 1].copy()
        mask = ((skeleton != 0).sum(-1) > 0)[..., None]
        skeleton = (skeleton - main_body_center) * mask
    if align_spine:
        joint_bottom = skeleton[0, 0, zaxis[0]]
        joint_top = skeleton[0, 0, zaxis[1]]
        axis = np.cross(joint_top - joint_bottom, [0, 0, 1])
        angle = _angle_between(joint_top - joint_bottom, [0, 0, 1])
        matrix_z = _rotation_matrix(axis, angle)
        skeleton = np.einsum("mtvc,dc->mtvd", skeleton, matrix_z)
        joint_rshoulder = skeleton[0, 0, xaxis[0]]
        joint_lshoulder = skeleton[0, 0, xaxis[1]]
        axis = np.cross(joint_rshoulder - joint_lshoulder, [1, 0, 0])
        angle = _angle_between(joint_rshoulder - joint_lshoulder, [1, 0, 0])
        matrix_x = _rotation_matrix(axis, angle)
        skeleton = np.einsum("mtvc,dc->mtvd", skeleton, matrix_x)
    return skeleton.astype(np.float32)


def random_rot(keypoint: np.ndarray, theta: float = 0.2) -> np.ndarray:
    rot_theta = np.random.uniform(-theta, theta, size=3)
    cos, sin = np.cos(rot_theta), np.sin(rot_theta)
    rx = np.array([[1, 0, 0], [0, cos[0], sin[0]], [0, -sin[0], cos[0]]], dtype=np.float32)
    ry = np.array([[cos[1], 0, -sin[1]], [0, 1, 0], [sin[1], 0, cos[1]]], dtype=np.float32)
    rz = np.array([[cos[2], sin[2], 0], [-sin[2], cos[2], 0], [0, 0, 1]], dtype=np.float32)
    rot = rz @ (ry @ rx)
    return np.einsum("ab,mtvb->mtva", rot, keypoint).astype(np.float32)


def sample_uniform_clips(
    keypoint: np.ndarray,
    clip_len: int,
    num_clips: int,
    test_mode: bool,
    seed: int = 255,
) -> np.ndarray:
    if test_mode:
        np.random.seed(seed)
    m, t, v, c = keypoint.shape
    clips = []
    for clip_idx in range(num_clips):
        num_frames = t
        if num_frames < clip_len:
            if test_mode:
                start = clip_idx if num_frames < num_clips else clip_idx * num_frames // num_clips
            else:
                start = np.random.randint(0, num_frames)
            inds = np.arange(start, start + clip_len) % num_frames
        elif clip_len <= num_frames < 2 * clip_len:
            basic = np.arange(clip_len)
            inds = np.random.choice(clip_len + 1, num_frames - clip_len, replace=False)
            offset = np.zeros(clip_len + 1, dtype=np.int64)
            offset[inds] = 1
            inds = basic + np.cumsum(offset)[:-1]
        else:
            bids = np.array([i * num_frames // clip_len for i in range(clip_len + 1)])
            bsize = np.diff(bids)
            bst = bids[:clip_len]
            offset = np.random.randint(bsize)
            inds = bst + offset
        clips.append(keypoint[:, inds].copy())
    return np.concatenate(clips, axis=1).astype(np.float32)


def format_gcn_input(keypoint: np.ndarray, num_person: int = 2) -> np.ndarray:
    if keypoint.shape[0] < num_person:
        pad_dim = num_person - keypoint.shape[0]
        pad = np.zeros((pad_dim,) + keypoint.shape[1:], dtype=keypoint.dtype)
        keypoint = np.concatenate((keypoint, pad), axis=0)
    elif keypoint.shape[0] > num_person:
        keypoint = keypoint[:num_person]
    m, t, v, c = keypoint.shape
    nc = 1
    if t % 10 == 0 and t > 100:
        pass
    keypoint = keypoint.reshape((m, 1, t, v, c)).transpose(1, 0, 2, 3, 4)
    return np.ascontiguousarray(keypoint)


def format_gcn_input_with_clips(keypoint: np.ndarray, num_clips: int, num_person: int = 2) -> np.ndarray:
    if keypoint.shape[0] < num_person:
        pad_dim = num_person - keypoint.shape[0]
        pad = np.zeros((pad_dim,) + keypoint.shape[1:], dtype=keypoint.dtype)
        keypoint = np.concatenate((keypoint, pad), axis=0)
    elif keypoint.shape[0] > num_person:
        keypoint = keypoint[:num_person]
    m, t, v, c = keypoint.shape
    assert t % num_clips == 0
    keypoint = keypoint.reshape((m, num_clips, t // num_clips, v, c)).transpose(1, 0, 2, 3, 4)
    return np.ascontiguousarray(keypoint)


@dataclass
class DatasetConfig:
    clip_len: int
    train_num_clips: int = 1
    eval_num_clips: int = 10
    pre_normalize: bool = True
    random_rotation: bool = True


class PoseSequenceDataset(Dataset):
    def __init__(
        self,
        sequences: list[np.ndarray],
        labels: np.ndarray,
        indices: np.ndarray,
        config: DatasetConfig,
        train: bool,
    ):
        self.sequences = sequences
        self.labels = labels
        self.indices = np.asarray(indices, dtype=np.int64)
        self.config = config
        self.train = train

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        idx = int(self.indices[item])
        kp = self.sequences[idx].copy()
        if self.config.pre_normalize:
            kp = pre_normalize_3d(kp, align_spine=False, align_center=True)
        if self.train and self.config.random_rotation:
            kp = random_rot(kp, theta=0.2)
        num_clips = self.config.train_num_clips if self.train else self.config.eval_num_clips
        kp = sample_uniform_clips(
            kp,
            clip_len=self.config.clip_len,
            num_clips=num_clips,
            test_mode=not self.train,
        )
        kp = format_gcn_input_with_clips(kp, num_clips=num_clips, num_person=2)
        return torch.from_numpy(kp), torch.tensor(self.labels[idx], dtype=torch.long), idx


def make_loaders(
    sequences: list[np.ndarray],
    labels: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    config: DatasetConfig,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    train_ds = PoseSequenceDataset(sequences, labels, train_idx, config, train=True)
    test_ds = PoseSequenceDataset(sequences, labels, test_idx, config, train=False)
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, test_loader


def load_representation(representation: str) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    if representation == "raw":
        return load_raw_sequences()
    if representation == "tangent":
        return load_tangent_sequences()
    raise ValueError(f"Unknown representation: {representation}")


def subject_folds(subject_ids: np.ndarray, seed: int = 42):
    return leave_5_subjects_out_folds(subject_ids, seed=seed)


def evaluate_from_oof(
    labels: np.ndarray,
    probs: np.ndarray,
    subjects: np.ndarray,
) -> dict:
    preds = probs.argmax(axis=1)
    ci = subject_bootstrap_ci_class(labels, preds, subjects, n_bootstrap=2000, random_state=42)
    top1, top5 = top_k_accuracy(probs, labels, ks=(1, min(5, probs.shape[1])))
    return {
        "metrics": ci,
        "top1_acc": top1,
        "top5_acc": top5,
        "preds": preds.tolist(),
        "probs": probs.tolist(),
        "labels": labels.tolist(),
    }


def save_json(path: Path, payload: dict) -> None:
    def _default(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=_default)
