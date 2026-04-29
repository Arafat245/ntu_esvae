from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from official_compare.common import (
    DatasetConfig,
    Graph,
    cosine_lr,
    evaluate_from_oof,
    fold_indices,
    load_representation,
    make_loaders,
    save_json,
    set_deterministic,
    subject_folds,
)


RESULTS_DIR = Path(__file__).resolve().parent / "results"
EPS = 1e-4


class GetSubnet(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, threshold, zeros, ones):
        return torch.where(scores < threshold, zeros.to(scores.device), ones.to(scores.device))

    @staticmethod
    def backward(ctx, g):
        return g, None, None, None


class SparseConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        conv_sparsity: float = 0.0,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.sparsity = conv_sparsity
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        self.weight.is_mask = True
        self.weight_score = nn.Parameter(torch.empty_like(self.weight))
        self.weight_score.is_score = True
        self.weight_score.sparsity = conv_sparsity
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.weight_score, nonlinearity="relu")
        self.register_buffer("zeros", torch.zeros_like(self.weight_score))
        self.register_buffer("ones", torch.ones_like(self.weight_score))

    def forward(self, x: torch.Tensor, threshold: float) -> torch.Tensor:
        subnet = GetSubnet.apply(self.weight_score, threshold, self.zeros, self.ones)
        return F.conv2d(
            x,
            self.weight * subnet,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class UnitGCNSparse(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, A: torch.Tensor, sparse_ratio: float = 0.6):
        super().__init__()
        self.num_subsets = A.size(0)
        self.A = nn.Parameter(A.clone())
        self.conv = SparseConv2d(in_channels, out_channels * A.size(0), 1, sparse_ratio)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor, threshold: float) -> torch.Tensor:
        n, c, t, v = x.shape
        x = self.conv(x, threshold).view(n, self.num_subsets, -1, t, v)
        x = torch.einsum("nkctv,kvw->nctw", x, self.A).contiguous()
        return self.act(self.bn(x))


class UnitTCNSparse(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size=9, stride=1, conv_sparsity=0.6):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = SparseConv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            conv_sparsity=conv_sparsity,
            padding=(pad, 0),
            stride=(stride, 1),
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor, threshold: float) -> torch.Tensor:
        return self.drop(self.bn(self.conv(x, threshold)))


class STGCNBlockSparse(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, A: torch.Tensor, stride: int = 1, residual: bool = True, sparse_ratio: float = 0.6):
        super().__init__()
        self.gcn = UnitGCNSparse(in_channels, out_channels, A, sparse_ratio=sparse_ratio)
        self.tcn = UnitTCNSparse(out_channels, out_channels, stride=stride, conv_sparsity=sparse_ratio)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x, threshold: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x, threshold: x
        else:
            self.residual = UnitTCNSparse(in_channels, out_channels, kernel_size=1, stride=stride, conv_sparsity=sparse_ratio)

    def forward(self, x: torch.Tensor, threshold: float) -> torch.Tensor:
        res = self.residual(x, threshold) if isinstance(self.residual, nn.Module) else self.residual(x, threshold)
        return self.relu(self.tcn(self.gcn(x, threshold), threshold) + res)


class STGCNSparseBackbone(nn.Module):
    def __init__(self, sparse_ratio: float = 0.6, warm_up: int = 60):
        super().__init__()
        graph = Graph(layout="nturgb+d", mode="random", num_filter=3, init_off=0.04, init_std=0.02)
        A = torch.tensor(graph.A, dtype=torch.float32)
        self.data_bn = nn.BatchNorm1d(3 * A.size(1))
        self.linear_sparsity = sparse_ratio
        self.warm_up = warm_up
        self.base_channels = 64
        modules = [STGCNBlockSparse(3, 64, A.clone(), 1, residual=False, sparse_ratio=sparse_ratio)]
        inflate_times = 0
        current = 64
        for i in range(2, 11):
            stride = 2 if i in [5, 8] else 1
            if i in [5, 8]:
                inflate_times += 1
            out_channels = int(self.base_channels * 2 ** inflate_times + EPS)
            modules.append(STGCNBlockSparse(current, out_channels, A.clone(), stride=stride, sparse_ratio=sparse_ratio))
            current = out_channels
        self.gcn = nn.ModuleList(modules)
        self.out_channels = current

    def percentile(self, t: torch.Tensor, q: float) -> float:
        k = 1 + round(0.01 * float(q) * (t.numel() - 1))
        return t.view(-1).kthvalue(k).values.item()

    def get_threshold(self, sparsity: float) -> float:
        score_params = []
        for p in self.gcn.parameters():
            if hasattr(p, "is_score") and p.is_score and p.sparsity == self.linear_sparsity:
                score_params.append(p.detach().flatten())
        joined = torch.cat(score_params)
        return self.percentile(joined, sparsity * 100)

    def forward(self, x: torch.Tensor, current_epoch: int, max_epoch: int) -> torch.Tensor:
        sparsity = 0.0 if current_epoch < self.warm_up else self.linear_sparsity
        n, m, t, v, c = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous().float()
        x = self.data_bn(x.view(n * m, v * c, t))
        x = x.view(n, m, v, c, t).permute(0, 1, 3, 4, 2).contiguous().view(n * m, c, t, v)
        threshold = self.get_threshold(sparsity)
        for block in self.gcn:
            x = block(x, threshold)
        return x.reshape((n, m) + x.shape[1:])


class GCNHead(nn.Module):
    def __init__(self, num_classes: int, in_channels: int):
        super().__init__()
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, m, c, t, v = x.shape
        pooled = x.view(n * m, c, t, v).mean(dim=(2, 3)).view(n, m, c).mean(dim=1)
        return self.fc(pooled)


class SparseSTGCNClassifier(nn.Module):
    def __init__(self, num_classes: int = 10, warm_up: int = 60):
        super().__init__()
        self.backbone = STGCNSparseBackbone(sparse_ratio=0.6, warm_up=warm_up)
        self.head = GCNHead(num_classes=num_classes, in_channels=self.backbone.out_channels)
        self.ce = nn.CrossEntropyLoss()

    def forward_train(self, x: torch.Tensor, y: torch.Tensor, epoch: int, total_epochs: int):
        x = x[:, 0]
        feat = self.backbone(x, current_epoch=epoch + 1, max_epoch=total_epochs)
        logits = self.head(feat)
        return self.ce(logits, y), logits

    @torch.no_grad()
    def forward_eval(self, x: torch.Tensor, total_epochs: int) -> torch.Tensor:
        bsz, num_clips = x.shape[:2]
        x = x.reshape(bsz * num_clips, *x.shape[2:])
        feat = self.backbone(x, current_epoch=total_epochs + 1, max_epoch=total_epochs)
        logits = self.head(feat)
        return torch.softmax(logits, dim=-1).view(bsz, num_clips, -1).mean(dim=1)


def run_subject_cv(args: argparse.Namespace) -> Path:
    set_deterministic(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.set_device(device)
    sequences, labels, subjects = load_representation(args.representation)
    folds = subject_folds(subjects, seed=args.seed)
    oof_probs = np.zeros((len(labels), len(np.unique(labels))), dtype=np.float32)
    per_fold = []
    ds_cfg = DatasetConfig(
        clip_len=60,
        train_num_clips=1,
        eval_num_clips=10,
        pre_normalize=(args.representation == "raw"),
        random_rotation=(args.representation == "raw"),
    )
    warm_up = args.warmup if args.warmup is not None else max(1, round(args.epochs * 0.6))
    for fold_id, test_subjects in enumerate(folds):
        if args.max_folds is not None and fold_id >= args.max_folds:
            break
        train_idx, test_idx = fold_indices(subjects, test_subjects)
        train_loader, test_loader = make_loaders(
            sequences,
            labels,
            train_idx,
            test_idx,
            ds_cfg,
            batch_size=args.batch_size,
            seed=args.seed + fold_id,
        )
        model = SparseSTGCNClassifier(num_classes=len(np.unique(labels)), warm_up=warm_up).to(device)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.1,
            momentum=0.9,
            weight_decay=5e-4,
            nesterov=True,
        )
        for epoch in range(args.epochs):
            lr = cosine_lr(0.1, epoch, args.epochs)
            optimizer.param_groups[0]["lr"] = lr
            model.train()
            for x, y, _ in train_loader:
                x = x.to(device=device, dtype=torch.float32)
                y = y.to(device=device)
                optimizer.zero_grad()
                loss, _ = model.forward_train(x, y, epoch=epoch, total_epochs=args.epochs)
                loss.backward()
                optimizer.step()
        model.eval()
        fold_probs = []
        fold_indices_seen = []
        with torch.no_grad():
            for x, _, sample_idx in test_loader:
                probs = model.forward_eval(x.to(device=device, dtype=torch.float32), total_epochs=args.epochs).cpu().numpy()
                fold_probs.append(probs)
                fold_indices_seen.extend(sample_idx.tolist())
        fold_probs = np.concatenate(fold_probs, axis=0)
        oof_probs[np.asarray(fold_indices_seen, dtype=np.int64)] = fold_probs
        per_fold.append(
            {
                "fold": fold_id,
                "test_subjects": test_subjects.tolist(),
                "epochs": args.epochs,
            }
        )
        fold_pred = fold_probs.argmax(axis=1)
        fold_acc = float((fold_pred == labels[np.asarray(fold_indices_seen, dtype=np.int64)]).mean())
        print(f"[Sparse-ST-GCN][{args.representation}] fold={fold_id} acc={fold_acc:.4f}")
    payload = evaluate_from_oof(labels, oof_probs, subjects)
    payload["folds"] = per_fold
    payload["config"] = {
        "representation": args.representation,
        "epochs": args.epochs,
        "warm_up": warm_up,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "model": "Sparse-ST-GCN",
    }
    out_path = RESULTS_DIR / f"sparse_stgcn_{args.representation}_subject.json"
    save_json(out_path, payload)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--representation", choices=["raw", "tangent"], required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    args = parser.parse_args()
    out_path = run_subject_cv(args)
    print(out_path)


if __name__ == "__main__":
    main()
