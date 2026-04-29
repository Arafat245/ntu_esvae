from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn

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


class UnitGCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A: torch.Tensor,
        ratio: float = 0.125,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.num_subsets = A.size(0)
        self.mid_channels = int(ratio * out_channels)
        self.A = nn.Parameter(A.clone())
        self.pre = nn.Sequential(
            nn.Conv2d(in_channels, self.mid_channels * self.num_subsets, 1),
            nn.BatchNorm2d(self.mid_channels * self.num_subsets),
            nn.ReLU(),
        )
        self.post = nn.Conv2d(self.mid_channels * self.num_subsets, out_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(self.num_subsets))
        self.beta = nn.Parameter(torch.zeros(self.num_subsets))
        self.conv1 = nn.Conv2d(in_channels, self.mid_channels * self.num_subsets, 1)
        self.conv2 = nn.Conv2d(in_channels, self.mid_channels * self.num_subsets, 1)
        self.down = (
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
            if in_channels != out_channels
            else nn.Identity()
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(-2)

    def forward(self, x: torch.Tensor):
        n, c, t, v = x.shape
        res = self.down(x)
        A = self.A[None, :, None, None]
        pre_x = self.pre(x).reshape(n, self.num_subsets, self.mid_channels, t, v)
        x1 = self.conv1(x).reshape(n, self.num_subsets, self.mid_channels, -1, v).mean(dim=-2, keepdim=True)
        x2 = self.conv2(x).reshape(n, self.num_subsets, self.mid_channels, -1, v).mean(dim=-2, keepdim=True)
        inter_graph = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2)) * self.alpha[0]
        A = inter_graph + A
        intra_graph = self.softmax(torch.einsum("nkctv,nkctw->nktvw", x1, x2)[:, :, None]) * self.beta[0]
        A = (A + intra_graph).squeeze(3)
        x = torch.einsum("nkctv,nkcvw->nkctw", pre_x, A).contiguous().reshape(n, -1, t, v)
        x = self.post(x)
        get_graph = (inter_graph + intra_graph).squeeze(3).reshape(n, -1, v, v)
        return self.act(self.bn(x) + res), get_graph


class UnitTCN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size=9, stride=1, dilation=1, dropout=0.0):
        super().__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1),
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.bn(self.conv(x)))


class MSTCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_joints: int = 25,
        dropout: float = 0.0,
        ms_cfg=((3, 1), (3, 2), (3, 3), (3, 4), ("max", 3), "1x1"),
        stride: int = 1,
    ):
        super().__init__()
        num_branches = len(ms_cfg)
        mid_channels = out_channels // num_branches
        rem_mid_channels = out_channels - mid_channels * (num_branches - 1)
        self.add_coeff = nn.Parameter(torch.zeros(num_joints))
        self.act = nn.ReLU()
        self.branches = nn.ModuleList()
        for i, cfg in enumerate(ms_cfg):
            branch_c = rem_mid_channels if i == 0 else mid_channels
            if cfg == "1x1":
                self.branches.append(nn.Conv2d(in_channels, branch_c, kernel_size=1, stride=(stride, 1)))
            elif cfg[0] == "max":
                self.branches.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, branch_c, kernel_size=1),
                        nn.BatchNorm2d(branch_c),
                        self.act,
                        nn.MaxPool2d(kernel_size=(cfg[1], 1), stride=(stride, 1), padding=(1, 0)),
                    )
                )
            else:
                self.branches.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, branch_c, kernel_size=1),
                        nn.BatchNorm2d(branch_c),
                        self.act,
                        UnitTCN(branch_c, branch_c, kernel_size=cfg[0], stride=stride, dilation=cfg[1]),
                    )
                )
        tin_channels = mid_channels * (num_branches - 1) + rem_mid_channels
        self.transform = nn.Sequential(
            nn.BatchNorm2d(tin_channels),
            self.act,
            nn.Conv2d(tin_channels, out_channels, kernel_size=1),
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, t, v = x.shape
        x = torch.cat([x, x.mean(-1, keepdim=True)], -1)
        outs = [branch(x) for branch in self.branches]
        out = torch.cat(outs, dim=1)
        local_feat = out[..., :v]
        global_feat = torch.einsum("nct,v->nctv", out[..., v], self.add_coeff[:v])
        feat = self.transform(local_feat + global_feat)
        return self.drop(self.bn(feat))


class GCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, A: torch.Tensor, stride: int = 1, residual: bool = True):
        super().__init__()
        self.gcn = UnitGCN(in_channels, out_channels, A)
        self.tcn = MSTCN(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = UnitTCN(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x: torch.Tensor):
        res = self.residual(x)
        x, graph = self.gcn(x)
        x = self.tcn(x) + res
        return self.relu(x), graph


class PrototypeReconstructionNetwork(nn.Module):
    def __init__(self, dim: int, n_prototype: int = 400, dropout: float = 0.1):
        super().__init__()
        self.query_matrix = nn.Linear(dim, n_prototype, bias=False)
        self.memory_matrix = nn.Linear(n_prototype, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = torch.softmax(self.query_matrix(x), dim=-1)
        return self.dropout(self.memory_matrix(query))


class ProtoGCNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        graph = Graph(layout="nturgb+d", mode="random", num_filter=8, init_off=0.04, init_std=0.02)
        A = torch.tensor(graph.A, dtype=torch.float32)
        self.data_bn = nn.BatchNorm1d(3 * A.size(1))
        self.gcn = nn.ModuleList()
        base_channels = 96
        num_stages = 10
        modules = [GCNBlock(3, base_channels, A.clone(), 1, residual=False)]
        inflate_times = 0
        current = base_channels
        for i in range(2, num_stages + 1):
            stride = 2 if i in [5, 8] else 1
            if i in [5, 8]:
                inflate_times += 1
            out_channels = int(96 * 2 ** inflate_times + EPS)
            modules.append(GCNBlock(current, out_channels, A.clone(), stride=stride))
            current = out_channels
        self.gcn = nn.ModuleList(modules)
        self.post = nn.Conv2d(current, current, 1)
        self.bn = nn.BatchNorm2d(current)
        self.relu = nn.ReLU()
        self.prn = PrototypeReconstructionNetwork(dim=384, n_prototype=400)
        self.out_channels = current

    def forward(self, x: torch.Tensor):
        n, m, t, v, c = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = self.data_bn(x.view(n * m, v * c, t))
        x = x.view(n, m, v, c, t).permute(0, 1, 3, 4, 2).contiguous().view(n * m, c, t, v)
        graphs = []
        for block in self.gcn:
            x, g = block(x)
            graphs.append(g)
        x = x.reshape((n, m) + x.shape[1:])
        graph = graphs[-1]
        c_graph = x.size(2)
        graph = graph.view(n, m, c_graph, v, v).mean(1).view(n, c_graph, v * v)
        recon = []
        for i in range(n):
            the_graph = graph[i].permute(1, 0)
            the_graph = self.prn(the_graph)
            the_graph = the_graph.permute(1, 0).view(c_graph, v, v)
            recon.append(the_graph)
        re_graph = torch.stack(recon, dim=0)
        reconstructed_graph = self.relu(self.bn(self.post(re_graph))).mean(1).view(n, -1)
        return x, reconstructed_graph


class ClassSpecificContrastiveLoss(nn.Module):
    def __init__(self, n_class: int, n_channel: int = 625, h_channel: int = 256, tmp: float = 0.125, mom: float = 0.9):
        super().__init__()
        self.n_class = n_class
        self.tmp = tmp
        self.mom = mom
        self.register_buffer("avg_f", torch.randn(h_channel, n_class))
        self.cl_fc = nn.Linear(n_channel, h_channel)
        self.loss = nn.CrossEntropyLoss(reduction="none")

    def onehot(self, label: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.one_hot(label.long(), num_classes=self.n_class).float()

    def forward(self, feature: torch.Tensor, lbl: torch.Tensor, logit: torch.Tensor) -> torch.Tensor:
        feature = self.cl_fc(feature)
        pred = logit.max(1)[1]
        pred_one = self.onehot(pred)
        lbl_one = self.onehot(lbl)
        prob = torch.softmax(logit, 1)
        mask = lbl_one * pred_one * (prob > 0.0).float()
        f = feature.permute(1, 0)
        mask_sum = mask.sum(0, keepdim=True)
        f_mask = torch.matmul(f, mask) / (mask_sum + 1e-12)
        has_object = (mask_sum > 1e-8).float()
        has_object[has_object > 0.1] = self.mom
        has_object[has_object <= 0.1] = 1.0
        f_mem = self.avg_f * has_object + (1 - has_object) * f_mask
        self.avg_f = f_mem.detach()
        feature = feature / (torch.norm(feature, p=2, dim=1, keepdim=True) + 1e-12)
        f_mem = f_mem.permute(1, 0)
        f_mem = f_mem / (torch.norm(f_mem, p=2, dim=-1, keepdim=True) + 1e-12)
        score_cl = torch.matmul(f_mem, feature.permute(1, 0)) / self.tmp
        return self.loss(score_cl.permute(1, 0).contiguous(), lbl).mean()


class ProtoGCNClassifier(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.backbone = ProtoGCNBackbone()
        self.fc = nn.Linear(self.backbone.out_channels, num_classes)
        self.ce = nn.CrossEntropyLoss()
        self.csc = ClassSpecificContrastiveLoss(num_classes, n_channel=625)

    def _pool_logits(self, feat: torch.Tensor) -> torch.Tensor:
        n, m, c, t, v = feat.shape
        pooled = feat.view(n * m, c, t, v).mean(dim=(2, 3)).view(n, m, c).mean(dim=1)
        return self.fc(pooled)

    def forward_train(self, x: torch.Tensor, y: torch.Tensor):
        x = x[:, 0]
        feat, graph = self.backbone(x)
        logits = self._pool_logits(feat)
        loss = self.ce(logits, y) + 0.2 * self.csc(graph, y, logits.detach())
        return loss, logits

    @torch.no_grad()
    def forward_eval(self, x: torch.Tensor) -> torch.Tensor:
        bsz, num_clips = x.shape[:2]
        x = x.reshape(bsz * num_clips, *x.shape[2:])
        feat, _ = self.backbone(x)
        logits = self._pool_logits(feat)
        probs = torch.softmax(logits, dim=-1).view(bsz, num_clips, -1).mean(dim=1)
        return probs


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
        clip_len=100,
        train_num_clips=1,
        eval_num_clips=10,
        pre_normalize=(args.representation == "raw"),
        random_rotation=(args.representation == "raw"),
    )
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
        model = ProtoGCNClassifier(num_classes=len(np.unique(labels))).to(device)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.025,
            momentum=0.9,
            weight_decay=5e-4,
            nesterov=True,
        )
        for epoch in range(args.epochs):
            lr = cosine_lr(0.025, epoch, args.epochs)
            optimizer.param_groups[0]["lr"] = lr
            model.train()
            for x, y, _ in train_loader:
                x = x.to(device=device, dtype=torch.float32)
                y = y.to(device=device)
                optimizer.zero_grad()
                loss, _ = model.forward_train(x, y)
                loss.backward()
                optimizer.step()
        model.eval()
        fold_probs = []
        fold_indices_seen = []
        with torch.no_grad():
            for x, _, sample_idx in test_loader:
                probs = model.forward_eval(x.to(device=device, dtype=torch.float32)).cpu().numpy()
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
        print(f"[ProtoGCN][{args.representation}] fold={fold_id} acc={fold_acc:.4f}")
    payload = evaluate_from_oof(labels, oof_probs, subjects)
    payload["folds"] = per_fold
    payload["config"] = {
        "representation": args.representation,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "model": "ProtoGCN",
    }
    out_path = RESULTS_DIR / f"protogcn_{args.representation}_subject.json"
    save_json(out_path, payload)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--representation", choices=["raw", "tangent"], required=True)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    out_path = run_subject_cv(args)
    print(out_path)


if __name__ == "__main__":
    main()
