#!/usr/bin/env python3
"""Revision experiment: replay memory budget and task-order interaction."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


def _add_main_source_path() -> None:
    here = Path(__file__).resolve()
    candidates = [
        here.parent,
        here.parent.parent,
        here.parent.parent / "01_experiment" / "source_code",
        here.parent.parent.parent / "01_experiment" / "source_code",
    ]
    for candidate in candidates:
        if (candidate / "run_split_mnist_replay_pipeline.py").exists():
            sys.path.insert(0, str(candidate))
            return
    raise RuntimeError("run_split_mnist_replay_pipeline.py not found")


_add_main_source_path()
from run_split_mnist_replay_pipeline import FORMAL_ORDER_IDS, FORMAL_SEEDS, load_split_mnist, run_one, stable_seed, write_csv, write_json  # noqa: E402


BUDGETS = [0, 50, 100, 200, 500, 1000]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("../raw_data/torchvision_data"))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train-cap-per-digit", type=int, default=1000)
    parser.add_argument("--test-cap-per-digit", type=int, default=500)
    parser.add_argument("--torch-threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.torch_threads)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("GPU is required for the formal revision experiments")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_split_mnist(args.data_dir, args.train_cap_per_digit, args.test_cap_per_digit, seed=stable_seed("revision_budget_data"))
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    total = len(BUDGETS) * len(FORMAL_SEEDS) * len(FORMAL_ORDER_IDS)
    run_index = 0
    for budget in BUDGETS:
        for seed in FORMAL_SEEDS:
            for order_id in FORMAL_ORDER_IDS:
                run_index += 1
                result = run_one(
                    data=data,
                    seed=seed,
                    order_id=order_id,
                    method_cfg={"method": "experience_replay", "memory_budget": budget, "ewc_lambda": 0.0, "use_replay": budget > 0, "use_ewc": False},
                    device=device,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    fisher_batches=4,
                    heartbeat=None,
                    run_index=run_index,
                    total_runs=total,
                )
                rows.append(
                    {
                        "seed": seed,
                        "order_id": order_id,
                        "task_order": result["task_order"],
                        "method": "experience_replay",
                        "memory_budget": budget,
                        "final_average_accuracy": f"{float(result['final_average_accuracy']):.6f}",
                        "average_forgetting": f"{float(result['average_forgetting']):.6f}",
                        **{f"task_{i}_final_accuracy": f"{float(result[f'task_{i}_final_accuracy']):.6f}" for i in range(5)},
                    }
                )
                write_csv(args.output_dir / "extended_budget_order.csv", rows)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["memory_budget"])].append(row)
    budget_summary = []
    order_effect = []
    for budget in sorted(grouped):
        items = grouped[budget]
        accuracies = torch.tensor([float(row["final_average_accuracy"]) for row in items], dtype=torch.float64)
        forgetting = torch.tensor([float(row["average_forgetting"]) for row in items], dtype=torch.float64)
        by_order: dict[str, list[float]] = defaultdict(list)
        for row in items:
            by_order[str(row["order_id"])].append(float(row["final_average_accuracy"]))
        order_means = {order: sum(values) / len(values) for order, values in by_order.items()}
        order_values = torch.tensor(list(order_means.values()), dtype=torch.float64)
        best_order = max(order_means, key=order_means.get)
        worst_order = min(order_means, key=order_means.get)
        gap = order_means[best_order] - order_means[worst_order]
        budget_summary.append(
            {
                "memory_budget": budget,
                "mean_accuracy": float(f"{float(accuracies.mean().item()):.6f}"),
                "mean_forgetting": float(f"{float(forgetting.mean().item()):.6f}"),
                "n_runs": len(items),
                "order_std": float(f"{float(order_values.std(unbiased=True).item()):.6f}"),
                "std_accuracy": float(f"{float(accuracies.std(unbiased=True).item()):.6f}"),
                "std_forgetting": float(f"{float(forgetting.std(unbiased=True).item()):.6f}"),
            }
        )
        order_effect.append(
            {
                "best_order_accuracy": float(f"{order_means[best_order]:.6f}"),
                "best_order_id": best_order,
                "gap": float(f"{gap:.6f}"),
                "memory_budget": budget,
                "order_effect_eliminated": bool(gap < 0.02),
                "worst_order_accuracy": float(f"{order_means[worst_order]:.6f}"),
                "worst_order_id": worst_order,
            }
        )
    write_json(args.output_dir / "budget_scaling_summary.json", budget_summary)
    write_json(args.output_dir / "order_effect_by_budget.json", order_effect)
    print(json.dumps({"script": Path(__file__).name, "status": "PASS", "rows": len(rows), "output_dir": str(args.output_dir), "elapsed_seconds": round(time.perf_counter() - started, 3)}, sort_keys=True))


if __name__ == "__main__":
    main()
