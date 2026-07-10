from __future__ import annotations

import argparse
import csv
import json
import math
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_pipeline import load_dataset
from goal6plus_vit_experiment import (
    DEVICE,
    evaluate,
    make_train_args,
    prepare_forward,
    preprocess_batch,
    train_teacher,
)
from train_logic_vit_tiny import build_model, set_deterministic


DEFAULTS = {
    "dataset": "cifar-10",
    "seed": 0,
    "data_encoding": "real-input",
    "augment": True,
    "batch_size": 128,
    "num_workers": 2,
    "valid_set_size": 0.1,
    "img_size": 32,
    "patch_size": 4,
    "embed_dim": 96,
    "depth": 2,
    "num_heads": 3,
    "drop_path_rate": 0.0,
    "ffn_type": "logic",
    "logic_ffn_layers": 2,
    "num_forest_layers": 2,
    "logic_connections": "random",
    "logic_n_thresholds": 7,
    "logic_use_thermometer": True,
    "logic_encoding_temperature": 10.0,
    "logic_act_fn": "SIN01",
    "logic_weight_init": "ri",
    "logic_weight_init_sigma": 0.5,
    "logic_shift_init": False,
    "logic_shift_init_type": "ri",
    "logic_shift_init_shift": 1.2,
    "logic_shift_init_direction": "0101",
    "logic_resconnection_init": False,
    "logic_res_connect_fraction": 0.0,
    "grad_factor": 1.0,
    "logic_connectivity": "fixed",
    "learnable_conn_k": 64,
    "learnable_conn_use_skip_bias": True,
    "logic_conn_lr_multiplier": 0.2,
    "post_discretize_finetune_iters": 0,
    "post_discretize_finetune_lr": 0.0,
    "head_type": "linear",
    "signed_head_topk": 32,
    "signed_head_use_topk_mask": False,
    "teacher_iters": 8000,
    "student_iters": 0,
    "learning_rate": 1e-3,
    "student_lr": 3e-4,
    "weight_decay": 0.01,
    "label_smoothing": 0.0,
    "temp_start": 1.0,
    "temp_end": 0.3,
    "temp_warmup_ratio": 0.02,
    "temp_cooldown_ratio": 0.05,
    "distill_alpha": 0.0,
    "distill_tau": 2.0,
    "student_scope": "head",
    "student_hard_thermometer": False,
    "eval_split": "test",
    "eval_max_batches": -1,
    "out_dir": "runs/frozen_feature_probe",
}


class MLPProbe(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int], out_dim: int, dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for width in hidden:
            layers.append(nn.Linear(prev, width))
            layers.append(nn.BatchNorm1d(width))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = width
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen CLS-feature probe for ViT-LGN.")
    parser.add_argument("--config-json", type=str, default="")
    parser.add_argument("--out-dir", type=str, default="runs/frozen_feature_probe")
    parser.add_argument("--teacher-iters", type=int, default=None)
    parser.add_argument("--eval-split", choices=["valid", "test"], default=None)
    parser.add_argument("--eval-max-batches", type=int, default=None)
    parser.add_argument("--probe-epochs", type=int, default=80)
    parser.add_argument("--probe-batch-size", type=int, default=1024)
    parser.add_argument("--mlp-hidden", type=str, default="1024,1024")
    parser.add_argument("--mlp-lr", type=float, default=1e-3)
    parser.add_argument("--mlp-weight-decay", type=float, default=1e-4)
    parser.add_argument("--mlp-dropout", type=float, default=0.1)
    parser.add_argument("--tree-max-train", type=int, default=45000)
    parser.add_argument("--svm-max-train", type=int, default=12000)
    parser.add_argument("--knn-max-train", type=int, default=20000)
    parser.add_argument("--skip-sklearn", action="store_true")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> SimpleNamespace:
    cfg = dict(DEFAULTS)
    if args.config_json:
        cfg.update(json.loads(Path(args.config_json).read_text(encoding="utf-8")))
    if args.teacher_iters is not None:
        cfg["teacher_iters"] = args.teacher_iters
    if args.eval_split is not None:
        cfg["eval_split"] = args.eval_split
    if args.eval_max_batches is not None:
        cfg["eval_max_batches"] = args.eval_max_batches
    cfg["out_dir"] = args.out_dir
    for key, value in DEFAULTS.items():
        cfg.setdefault(key, value)
    return SimpleNamespace(**cfg)


def cls_features(model: nn.Module, images: torch.Tensor, transform) -> torch.Tensor:
    images = preprocess_batch(images, transform)
    x = model.patch_embed(images)
    batch = x.shape[0]
    cls_tokens = model.cls_token.expand(batch, -1, -1)
    x = torch.cat([cls_tokens, x], dim=1)
    x = x + model.pos_embed
    x = model.pos_drop(x)
    for block in model.blocks:
        x = block(x)
    x = model.norm(x)
    return x[:, 0]


@torch.no_grad()
def extract_features(model: nn.Module, loader, transform, max_batches: int | None) -> tuple[np.ndarray, np.ndarray]:
    prepare_forward(model, "relaxed_eval")
    feats: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for batch_idx, (images, targets) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        feat = cls_features(model, images, transform)
        feats.append(feat.detach().cpu())
        labels.append(targets.detach().cpu())
    return torch.cat(feats).numpy(), torch.cat(labels).numpy()


def standardize(train_x: np.ndarray, eval_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.maximum(std, 1e-6)
    return (train_x - mean) / std, (eval_x - mean) / std, mean, std


def accuracy_np(pred: np.ndarray, target: np.ndarray) -> float:
    return float((pred == target).mean()) if target.size else 0.0


def train_mlp_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    valid_x: np.ndarray,
    valid_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, float | str]:
    hidden = [int(x) for x in args.mlp_hidden.split(",") if x.strip()]
    train_t = torch.tensor(train_x, dtype=torch.float32)
    train_y_t = torch.tensor(train_y, dtype=torch.long)
    valid_t = torch.tensor(valid_x, dtype=torch.float32)
    valid_y_t = torch.tensor(valid_y, dtype=torch.long)
    test_t = torch.tensor(test_x, dtype=torch.float32)
    test_y_t = torch.tensor(test_y, dtype=torch.long)
    num_classes = int(max(train_y.max(), valid_y.max(), test_y.max()) + 1)
    model = MLPProbe(train_x.shape[1], hidden, num_classes, args.mlp_dropout).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=args.mlp_lr, weight_decay=args.mlp_weight_decay)
    gen = torch.Generator().manual_seed(12345)
    start = time.time()
    best_valid = -1.0
    best_state = None

    def predict_acc(x: torch.Tensor, y: torch.Tensor) -> float:
        pred = []
        model.eval()
        with torch.no_grad():
            for start_idx in range(0, x.shape[0], args.probe_batch_size):
                logits = model(x[start_idx : start_idx + args.probe_batch_size].to(DEVICE))
                pred.append(logits.argmax(dim=-1).cpu())
        return float((torch.cat(pred) == y).float().mean().item())

    for _epoch in range(args.probe_epochs):
        perm = torch.randperm(train_t.shape[0], generator=gen)
        model.train()
        for start_idx in range(0, train_t.shape[0], args.probe_batch_size):
            idx = perm[start_idx : start_idx + args.probe_batch_size]
            xb = train_t[idx].to(DEVICE)
            yb = train_y_t[idx].to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
        valid_acc = predict_acc(valid_t, valid_y_t)
        if valid_acc > best_valid:
            best_valid = valid_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    train_acc = predict_acc(train_t, train_y_t)
    valid_acc = predict_acc(valid_t, valid_y_t)
    test_acc = predict_acc(test_t, test_y_t)
    return {
        "method": "dense_mlp",
        "train_acc": train_acc,
        "valid_acc": valid_acc,
        "test_acc": test_acc,
        "fit_time": time.time() - start,
        "details": f"hidden={hidden},epochs={args.probe_epochs},selected_by=valid",
    }


def subsample(x: np.ndarray, y: np.ndarray, max_n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if max_n <= 0 or x.shape[0] <= max_n:
        return x, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.shape[0], size=max_n, replace=False)
    return x[idx], y[idx]


def run_sklearn_probes(
    train_x: np.ndarray,
    train_y: np.ndarray,
    valid_x: np.ndarray,
    valid_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    args: argparse.Namespace,
) -> list[dict[str, float | str]]:
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    probes: list[tuple[str, object, int]] = [
        ("logistic_regression", LogisticRegression(max_iter=1000, C=1.0, n_jobs=-1), 0),
        ("knn_5", KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1), args.knn_max_train),
        ("random_forest", RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=0), args.tree_max_train),
        ("hist_gbdt", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=63, random_state=0), args.tree_max_train),
        ("decision_tree_lut", DecisionTreeClassifier(random_state=0, min_samples_leaf=1), args.tree_max_train),
        ("rbf_svm", SVC(C=10.0, gamma="scale"), args.svm_max_train),
    ]
    rows: list[dict[str, float | str]] = []
    for name, clf, max_train in probes:
        x_sub, y_sub = subsample(train_x, train_y, max_train, seed=100 + len(rows))
        start = time.time()
        clf.fit(x_sub, y_sub)
        pred_train = clf.predict(x_sub)
        pred_valid = clf.predict(valid_x)
        pred_test = clf.predict(test_x)
        rows.append(
            {
                "method": name,
                "train_acc": accuracy_np(pred_train, y_sub),
                "valid_acc": accuracy_np(pred_valid, valid_y),
                "test_acc": accuracy_np(pred_test, test_y),
                "fit_time": time.time() - start,
                "details": f"train_n={x_sub.shape[0]}",
            }
        )
    return rows


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    probe_args = parse_args()
    cfg = load_config(probe_args)
    set_deterministic(cfg.seed)
    out_root = Path(probe_args.out_dir)
    out_dir = out_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(cfg), indent=2, sort_keys=True), encoding="utf-8")

    train_args = make_train_args(cfg)
    train_loader, valid_loader, test_loader, train_eval_loader, _num_batches, transform = load_dataset(train_args)
    if valid_loader is None:
        raise RuntimeError("Frozen-feature probe requires a validation split for model selection.")
    eval_max_batches = None if cfg.eval_max_batches < 0 else cfg.eval_max_batches

    teacher = build_model(train_args).to(DEVICE)
    teacher_time = train_teacher(teacher, train_loader, transform, cfg, train_args)
    teacher_valid_acc, teacher_valid_loss = evaluate(teacher, valid_loader, transform, "relaxed_eval", eval_max_batches)
    teacher_test_acc, teacher_test_loss = evaluate(teacher, test_loader, transform, "relaxed_eval", eval_max_batches)

    feature_start = time.time()
    train_x, train_y = extract_features(teacher, train_eval_loader, transform, None)
    valid_x, valid_y = extract_features(teacher, valid_loader, transform, eval_max_batches)
    test_x, test_y = extract_features(teacher, test_loader, transform, eval_max_batches)
    train_x_std, valid_x_std, _mean, _std = standardize(train_x, valid_x)
    test_x_std = (test_x - _mean) / _std
    feature_time = time.time() - feature_start

    rows: list[dict[str, object]] = [
        {
            "method": "original_linear_head",
            "train_acc": math.nan,
            "valid_acc": teacher_valid_acc,
            "test_acc": teacher_test_acc,
            "fit_time": 0.0,
            "details": f"valid_loss={teacher_valid_loss:.6g};test_loss={teacher_test_loss:.6g}",
        }
    ]
    rows.append(train_mlp_probe(train_x_std, train_y, valid_x_std, valid_y, test_x_std, test_y, probe_args))
    if not probe_args.skip_sklearn:
        rows.extend(run_sklearn_probes(train_x_std, train_y, valid_x_std, valid_y, test_x_std, test_y, probe_args))

    probe_rows = [row for row in rows if row["method"] != "original_linear_head"]
    best_valid_probe = max(probe_rows, key=lambda row: float(row["valid_acc"]))
    best_test_observed = max(probe_rows, key=lambda row: float(row["test_acc"]))
    summary = {
        "out_dir": str(out_dir),
        "device": str(DEVICE),
        "teacher_train_time": teacher_time,
        "feature_extract_time": feature_time,
        "train_feature_shape": list(train_x.shape),
        "valid_feature_shape": list(valid_x.shape),
        "test_feature_shape": list(test_x.shape),
        "teacher_relaxed_valid_acc": teacher_valid_acc,
        "teacher_relaxed_valid_loss": teacher_valid_loss,
        "teacher_relaxed_test_acc": teacher_test_acc,
        "teacher_relaxed_test_loss": teacher_test_loss,
        "best_valid_probe_method": best_valid_probe["method"],
        "best_valid_probe_valid_acc": best_valid_probe["valid_acc"],
        "best_valid_probe_test_acc": best_valid_probe["test_acc"],
        "best_valid_probe_test_delta_vs_original": float(best_valid_probe["test_acc"]) - teacher_test_acc,
        "best_test_observed_method": best_test_observed["method"],
        "best_test_observed_test_acc": best_test_observed["test_acc"],
        "best_test_observed_delta_vs_original": float(best_test_observed["test_acc"]) - teacher_test_acc,
        "rows": rows,
    }
    write_rows(out_dir / "probe_results.csv", rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
