#!/usr/bin/env python3
"""Run reproducible Gumbel-ST hyperparameter sweeps."""

from __future__ import annotations

import argparse
import csv
import itertools
import subprocess
import sys
from pathlib import Path


def parse_float_list(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--method", default="gumbel_st", choices=["gumbel_st", "gumbel_soft"])
    parser.add_argument("--datasets", nargs="+", default=["parity8", "majority9", "random_sparse10"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--epochs-grid", default="80")
    parser.add_argument("--lr-grid", default="0.005,0.01,0.02")
    parser.add_argument("--temp-start-grid", default="1.0,1.5,2.0")
    parser.add_argument("--temp-end-grid", default="0.3,0.6,1.0")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--group-tau", type=float, default=1.0)
    parser.add_argument("--data-dir", default="/home/spco/data")
    parser.add_argument("--download-data", action="store_true")
    parser.add_argument("--threshold-levels", type=int, default=1)
    parser.add_argument("--image-max-train", type=int, default=4000)
    parser.add_argument("--image-max-test", type=int, default=1000)
    parser.add_argument("--target-acc-override", type=float)
    parser.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--benchmark", default="hard_lgn_benchmark.py")
    parser.add_argument("--max-trials", type=int, default=0, help="0 means run the full grid.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    lrs = parse_float_list(args.lr_grid)
    starts = parse_float_list(args.temp_start_grid)
    ends = parse_float_list(args.temp_end_grid)
    epochs_values = parse_int_list(args.epochs_grid)
    grid = list(itertools.product(epochs_values, lrs, starts, ends))
    if args.max_trials > 0:
        grid = grid[: args.max_trials]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    command_rows: list[dict[str, object]] = []

    for trial_id, (epochs, lr, temp_start, temp_end) in enumerate(grid):
        trial_dir = args.out_dir / "trials" / f"trial_{trial_id:03d}"
        cmd = [
            args.python,
            args.benchmark,
            "--datasets",
            *args.datasets,
            "--methods",
            args.method,
            "--epochs",
            str(epochs),
            "--width",
            str(args.width),
            "--layers",
            str(args.layers),
            "--seeds",
            *[str(seed) for seed in args.seeds],
            "--batch-size",
            str(args.batch_size),
            "--eval-batch-size",
            str(args.eval_batch_size),
            "--lr",
            str(lr),
            "--weight-decay",
            str(args.weight_decay),
            "--group-tau",
            str(args.group_tau),
            "--gumbel-temp-start",
            str(temp_start),
            "--gumbel-temp-end",
            str(temp_end),
            "--data-dir",
            args.data_dir,
            "--threshold-levels",
            str(args.threshold_levels),
            "--image-max-train",
            str(args.image_max_train),
            "--image-max-test",
            str(args.image_max_test),
            "--device",
            args.device,
            "--out-dir",
            str(trial_dir),
        ]
        if args.download_data:
            cmd.append("--download-data")
        if args.target_acc_override is not None:
            cmd.extend(["--target-acc-override", str(args.target_acc_override)])
        command_rows.append(
            {
                "trial": trial_id,
                "epochs": epochs,
                "lr": lr,
                "gumbel_temp_start": temp_start,
                "gumbel_temp_end": temp_end,
                "out_dir": str(trial_dir),
                "command": " ".join(cmd),
            }
        )
        print(f"trial={trial_id} epochs={epochs} lr={lr} temp={temp_start}->{temp_end}", flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True)
            for row in read_rows(trial_dir / "results.csv"):
                row.update(
                    {
                        "trial": trial_id,
                        "epochs_cfg": epochs,
                        "lr_cfg": lr,
                        "gumbel_temp_start_cfg": temp_start,
                        "gumbel_temp_end_cfg": temp_end,
                        "trial_out_dir": str(trial_dir),
                    }
                )
                all_rows.append(row)
            write_rows(args.out_dir / "all_results.partial.csv", all_rows)
        write_rows(args.out_dir / "commands.csv", command_rows)

    if args.dry_run:
        return

    write_rows(args.out_dir / "all_results.csv", all_rows)
    best_rows: list[dict[str, object]] = []
    keys = sorted({(str(row["dataset"]), str(row["seed"])) for row in all_rows})
    for dataset, seed in keys:
        group = [row for row in all_rows if row["dataset"] == dataset and row["seed"] == seed]
        best = sorted(
            group,
            key=lambda row: (
                -float(row["discrete_acc"]),
                float(row["acc_gap"]),
                float(row["discrete_loss"]),
                float(row["train_time"]),
            ),
        )[0]
        best_rows.append(best)
    write_rows(args.out_dir / "best_by_dataset_seed.csv", best_rows)

    best_dataset_rows: list[dict[str, object]] = []
    for dataset in sorted({str(row["dataset"]) for row in all_rows}):
        group = [row for row in all_rows if row["dataset"] == dataset]
        best = sorted(
            group,
            key=lambda row: (
                -float(row["discrete_acc"]),
                float(row["acc_gap"]),
                float(row["discrete_loss"]),
                float(row["train_time"]),
            ),
        )[0]
        best_dataset_rows.append(best)
    write_rows(args.out_dir / "best_by_dataset.csv", best_dataset_rows)


if __name__ == "__main__":
    main()
