#!/usr/bin/env python3
"""Launch the full 120-permutation public reproduction grid."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from data import FORMAL_SEEDS, all_task_orders, load_split_mnist
from run_experiment import run_single


METHODS = ["no_replay", "er100", "er500", "ewc"]
FIELDNAMES = [
    "seed",
    "order_id",
    "task_order",
    "method",
    "memory_budget",
    "method_cell",
    "status",
    "ewc_lambda",
    "selected_ewc_lambda",
    "epochs",
    "batch_size",
    "optimizer_steps_total",
    "optimizer_steps_per_task",
    "processed_examples_total",
    "processed_examples_per_task",
    "effective_epoch_passes_per_task",
    "A_after_mean",
    "A_after_median",
    "A_after_min",
    "A_after_le_0_001_count",
    "final_average_accuracy",
    "average_forgetting",
    "AF",
    "elapsed_seconds",
    "elapsed_time_seconds",
    "device",
    "train_cap_per_digit",
    "test_cap_per_digit",
    "result_kind",
    "task_0_A_after",
    "task_1_A_after",
    "task_2_A_after",
    "task_3_A_after",
    "task_4_A_after",
    "task_0_A_end",
    "task_1_A_end",
    "task_2_A_end",
    "task_3_A_end",
    "task_4_A_end",
    "task_0_final_accuracy",
    "task_1_final_accuracy",
    "task_2_final_accuracy",
    "task_3_final_accuracy",
    "task_4_final_accuracy",
]


def append_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        if "AF" not in row:
            row["AF"] = row.get("average_forgetting", 0.0)
        if "elapsed_time_seconds" not in row:
            row["elapsed_time_seconds"] = row.get("elapsed_seconds", 0.0)
        row.setdefault("train_cap_per_digit", 1000)
        row.setdefault("test_cap_per_digit", 500)
        row.setdefault("result_kind", "public_rerun")
        writer.writerow(row)


def existing_keys(path: Path) -> set[tuple[str, int, str]]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        return {(row["order_id"], int(row["seed"]), row["method_cell"]) for row in csv.DictReader(handle)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/results_120perm_4methods.csv"))
    parser.add_argument("--matched-output", type=Path, default=Path("results/results_compute_matched.csv"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/torchvision_data"))
    parser.add_argument("--download-mnist", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--train-cap-per-digit", type=int, default=1000)
    parser.add_argument("--test-cap-per-digit", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit("CUDA is unavailable; pass --allow-cpu for CPU reproduction.")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = load_split_mnist(
        args.data_dir,
        train_cap_per_digit=args.train_cap_per_digit,
        test_cap_per_digit=args.test_cap_per_digit,
        download=args.download_mnist,
    )
    orders = [f"perm_{idx:03d}" for idx, _order in enumerate(all_task_orders())]

    full_seen = existing_keys(args.output)
    for order_id in orders:
        for seed in FORMAL_SEEDS:
            for method in METHODS:
                key = (order_id, seed, method)
                if key in full_seen:
                    continue
                row = run_single(data=data, seed=seed, order_id=order_id, method=method, device=device, batch_size=args.batch_size, epochs=args.epochs)
                row.pop("_task_rows")
                append_row(args.output, row)
                full_seen.add(key)

    matched_seen = existing_keys(args.matched_output)
    for order_id in orders:
        for seed in FORMAL_SEEDS:
            key = (order_id, seed, "er500_matched")
            if key in matched_seen:
                continue
            row = run_single(data=data, seed=seed, order_id=order_id, method="er500_matched", device=device, batch_size=args.batch_size, epochs=args.epochs)
            row.pop("_task_rows")
            append_row(args.matched_output, row)
            matched_seen.add(key)


if __name__ == "__main__":
    main()
