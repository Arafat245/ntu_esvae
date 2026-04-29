from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
HYPER_ROOT = REPO_ROOT / "external" / "Hyper-GCN"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(HYPER_ROOT) not in sys.path:
    sys.path.insert(0, str(HYPER_ROOT))

from feeders.tools import random_rot, valid_crop_resize  # type: ignore  # noqa: E402

from official_compare.common import (  # noqa: E402
    evaluate_from_oof,
    load_representation,
    save_json,
    set_deterministic,
    subject_folds,
)


RESULTS_DIR = Path(__file__).resolve().parent / "results"


class DivergenceLoss(nn.Module):
    """Official Hyper-GCN divergence regularizer on virtual hyper-joints."""

    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, hyper_joints: list[torch.Tensor]) -> torch.Tensor:
        vertex_num, _ = hyper_joints[0].size()
        loss = 0.0
        for item in hyper_joints:
            norm = torch.norm(item, dim=-1, keepdim=True, p=2)
            norm = norm @ norm.T
            loss_i = item @ item.T
            loss_i = loss_i / (norm + 1e-8)
            loss_p = self.relu(loss_i)
            loss_p = (loss_p.sum() - vertex_num) / (vertex_num * (vertex_num - 1))
            loss = loss + loss_p
        return loss / len(hyper_joints)


class HyperGCNDataset(Dataset):
    def __init__(
        self,
        sequences: list[np.ndarray],
        labels: np.ndarray,
        indices: np.ndarray,
        representation: str,
        train: bool,
        window_size: int = 64,
    ):
        self.sequences = sequences
        self.labels = labels
        self.indices = np.asarray(indices, dtype=np.int64)
        self.representation = representation
        self.train = train
        self.window_size = window_size

    def __len__(self) -> int:
        return len(self.indices)

    def _valid_frame_num(self, data_numpy: np.ndarray) -> int:
        valid = np.any(np.abs(data_numpy) > 1e-8, axis=(0, 2, 3))
        count = int(valid.sum())
        return count if count > 0 else data_numpy.shape[1]

    def __getitem__(self, item: int):
        idx = int(self.indices[item])
        kp = self.sequences[idx].copy()  # (1, T, V, C)
        data_numpy = kp.transpose(3, 1, 2, 0).astype(np.float32)  # (C, T, V, M)
        valid_frame_num = self._valid_frame_num(data_numpy)
        p_interval = [0.5, 1.0] if self.train else [0.95]
        data_numpy = valid_crop_resize(data_numpy, valid_frame_num, p_interval, self.window_size)

        if self.train and self.representation == "raw":
            data_numpy = random_rot(data_numpy)

        # Mirror the official joint-modality transform on raw coordinates:
        # subtract the center joint trajectory while preserving that joint.
        if self.representation == "raw":
            trajectory = data_numpy[:, :, 20].copy()
            data_numpy = data_numpy - data_numpy[:, :, 20:21]
            data_numpy[:, :, 20] = trajectory

        # Official Hyper-GCN is parameterized with num_person=2.
        # Our subset stores a single skeleton per sample, so pad a zero person
        # to preserve the official input contract instead of changing the model.
        num_person = data_numpy.shape[-1]
        if num_person < 2:
            pad = np.zeros((*data_numpy.shape[:-1], 2 - num_person), dtype=data_numpy.dtype)
            data_numpy = np.concatenate([data_numpy, pad], axis=-1)
        elif num_person > 2:
            data_numpy = data_numpy[..., :2]

        return (
            torch.from_numpy(np.ascontiguousarray(data_numpy)),
            torch.tensor(self.labels[idx], dtype=torch.long),
            idx,
        )


def make_loaders(
    sequences: list[np.ndarray],
    labels: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    representation: str,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    train_ds = HyperGCNDataset(sequences, labels, train_idx, representation=representation, train=True)
    test_ds = HyperGCNDataset(sequences, labels, test_idx, representation=representation, train=False)
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        generator=generator,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    return train_loader, test_loader


def fold_indices(subject_ids: np.ndarray, test_subjects: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    test_mask = np.isin(subject_ids, test_subjects)
    test_idx = np.where(test_mask)[0]
    train_idx = np.where(~test_mask)[0]
    return train_idx, test_idx


def build_model(variant: str, num_classes: int) -> nn.Module:
    if variant == "base":
        from model.hypergcn_base import Model as HyperModel  # type: ignore
    elif variant == "large":
        from model.hypergcn_large import Model as HyperModel  # type: ignore
    else:
        raise ValueError(f"Unknown Hyper-GCN variant: {variant}")

    return HyperModel(
        num_class=num_classes,
        num_point=25,
        num_person=2,
        hyper_joints=3,
        graph="graph.ntu_rgb_d.Graph",
        graph_args={"labeling_mode": "virtual_ensemble"},
    )


def adjust_learning_rate(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    base_lr: float,
    step: list[int],
    decay_rate: float,
    warm_up_epoch: int,
) -> float:
    if epoch < warm_up_epoch:
        lr = base_lr * (epoch + 1) / warm_up_epoch
    else:
        lr = base_lr * (decay_rate ** np.sum(epoch >= np.asarray(step)))
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def run_subject_cv(args: argparse.Namespace) -> Path:
    set_deterministic(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.set_device(device)

    sequences, labels, subjects = load_representation(args.representation)
    folds = subject_folds(subjects, seed=args.seed)
    oof_probs = np.zeros((len(labels), len(np.unique(labels))), dtype=np.float32)
    per_fold = []

    for fold_id, test_subjects in enumerate(folds):
        if args.max_folds is not None and fold_id >= args.max_folds:
            break
        train_idx, test_idx = fold_indices(subjects, test_subjects)
        train_loader, test_loader = make_loaders(
            sequences,
            labels,
            train_idx,
            test_idx,
            representation=args.representation,
            batch_size=args.batch_size,
            seed=args.seed + fold_id,
        )

        model = build_model(args.variant, num_classes=len(np.unique(labels))).to(device)
        ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
        h_loss = DivergenceLoss().to(device)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.05,
            momentum=0.9,
            nesterov=True,
            weight_decay=4e-4,
        )

        for epoch in range(args.epochs):
            adjust_learning_rate(
                optimizer,
                epoch=epoch,
                base_lr=0.05,
                step=[110, 120],
                decay_rate=0.1,
                warm_up_epoch=5,
            )
            model.train()
            for x, y, _ in train_loader:
                x = x.to(device=device, dtype=torch.float32)
                y = y.to(device=device)
                optimizer.zero_grad()
                logits, hyper_joints = model(x)
                loss = ce_loss(logits, y) + h_loss(hyper_joints)
                loss.backward()
                optimizer.step()

        model.eval()
        fold_probs = []
        fold_indices_seen = []
        with torch.no_grad():
            for x, _, sample_idx in test_loader:
                logits, _ = model(x.to(device=device, dtype=torch.float32))
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                fold_probs.append(probs)
                fold_indices_seen.extend(sample_idx.tolist())

        fold_probs = np.concatenate(fold_probs, axis=0)
        fold_idx_arr = np.asarray(fold_indices_seen, dtype=np.int64)
        oof_probs[fold_idx_arr] = fold_probs
        fold_pred = fold_probs.argmax(axis=1)
        fold_acc = float((fold_pred == labels[fold_idx_arr]).mean())
        print(f"[Hyper-GCN/{args.variant}][{args.representation}] fold={fold_id} acc={fold_acc:.4f}")
        per_fold.append(
            {
                "fold": fold_id,
                "test_subjects": test_subjects.tolist(),
                "epochs": args.epochs,
            }
        )

    payload = evaluate_from_oof(labels, oof_probs, subjects)
    payload["folds"] = per_fold
    payload["config"] = {
        "representation": args.representation,
        "variant": args.variant,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "model": "Hyper-GCN",
    }
    out_path = RESULTS_DIR / f"hypergcn_{args.variant}_{args.representation}_subject.json"
    save_json(out_path, payload)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--representation", choices=["raw", "tangent"], required=True)
    parser.add_argument("--variant", choices=["base", "large"], default="base")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    out_path = run_subject_cv(args)
    print(out_path)


if __name__ == "__main__":
    main()
