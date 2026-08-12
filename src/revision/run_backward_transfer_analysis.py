#!/usr/bin/env python3
"""Revision experiment: separate retention and backward transfer."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

TINY_BASELINE_THRESHOLD = 0.001


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
from run_split_mnist_replay_pipeline import (  # noqa: E402
    FORMAL_ORDER_IDS,
    FORMAL_SEEDS,
    METHOD_BUDGETS,
    MLP,
    SplitMnistData,
    estimate_fisher,
    evaluate,
    ewc_penalty,
    load_split_mnist,
    make_loader,
    select_replay_indices,
    stable_seed,
    task_order,
    update_memory,
    write_csv,
    write_json,
)


def rounded(value: float, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def mean_or_none(values: list[float], digits: int) -> float | None:
    if not values:
        return None
    return float(f"{sum(values) / len(values):.{digits}f}")


def median_or_none(values: list[float], digits: int) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        median = ordered[midpoint]
    else:
        median = (ordered[midpoint - 1] + ordered[midpoint]) / 2.0
    return float(f"{median:.{digits}f}")


def summarize_backward_transfer(detail_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in detail_rows:
        grouped[(str(row["method"]), int(row["memory_budget"]))].append(row)
    summary = []
    for method, budget in sorted(grouped):
        items = grouped[(method, budget)]
        bwts = [float(row["BWT"]) for row in items]
        retentions = [float(row["retention"]) for row in items]
        valid_retentions = [
            float(row["retention"])
            for row in items
            if float(row["acc_just_after"]) > TINY_BASELINE_THRESHOLD
        ]
        tiny_baseline_count = sum(
            1 for row in items if float(row["acc_just_after"]) <= TINY_BASELINE_THRESHOLD
        )
        tiny_baseline_ratio = tiny_baseline_count / len(items)
        summary.append(
            {
                "mean_BWT": float(f"{sum(bwts) / len(bwts):.6f}"),
                "mean_retention": float(f"{sum(retentions) / len(retentions):.4f}"),
                "mean_retention_excluding_tiny": mean_or_none(valid_retentions, 4),
                "median_retention_excluding_tiny": median_or_none(valid_retentions, 4),
                "memory_budget": budget,
                "method": method,
                "n_observations": len(items),
                "positive_BWT_ratio": float(f"{sum(1 for value in bwts if value > 0) / len(bwts):.4f}"),
                "retention_unstable": tiny_baseline_ratio > 0.05,
                "std_BWT": float(f"{torch.tensor(bwts).std(unbiased=True).item():.6f}"),
                "tiny_baseline_count": tiny_baseline_count,
                "tiny_baseline_ratio": float(f"{tiny_baseline_ratio:.4f}"),
                "tiny_baseline_threshold": TINY_BASELINE_THRESHOLD,
                "valid_retention_count": len(valid_retentions),
            }
        )
    return summary


def run_one_with_histories(
    data: SplitMnistData,
    seed: int,
    order_id: str,
    method_cfg: dict[str, Any],
    device: torch.device,
    batch_size: int,
    epochs: int,
    fisher_batches: int,
) -> dict[int, list[float]]:
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    criterion = nn.CrossEntropyLoss()
    memory: list[int] = []
    anchors: list[dict[str, dict[str, torch.Tensor]]] = []
    histories: dict[int, list[float]] = {task_id: [] for task_id in range(5)}
    order = task_order(order_id)
    for position, task_id in enumerate(order):
        current = data.train_by_task[task_id]
        replay = select_replay_indices(memory, int(method_cfg["memory_budget"]), seed=stable_seed(seed, order_id, method_cfg["method"], task_id))
        train_indices = torch.unique(torch.cat([current, replay])) if replay.numel() else current
        for epoch in range(epochs):
            loader = make_loader(data.train_x, data.train_y, train_indices, batch_size, seed=stable_seed(seed, order_id, task_id, epoch))
            model.train()
            for batch_x, batch_y in loader:
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_x.to(device))
                loss = criterion(logits, batch_y.to(device))
                if method_cfg["use_ewc"] and anchors:
                    loss = loss + float(method_cfg["ewc_lambda"]) * ewc_penalty(model, anchors, device)
                loss.backward()
                optimizer.step()
        for seen_task in order[: position + 1]:
            histories[seen_task].append(evaluate(model, data.test_x, data.test_y, data.test_by_task[seen_task], device, batch_size))
        if method_cfg["use_replay"]:
            memory = update_memory(memory, current, int(method_cfg["memory_budget"]), seed=stable_seed("memory", seed, order_id, task_id))
        if method_cfg["use_ewc"]:
            anchors.append(
                {
                    "params": {name: param.detach().cpu().clone() for name, param in model.named_parameters() if param.requires_grad},
                    "fisher": estimate_fisher(model, data.train_x, data.train_y, current, device, batch_size, fisher_batches),
                }
            )
    return histories


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("../raw_data/torchvision_data"))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train-cap-per-digit", type=int, default=1000)
    parser.add_argument("--test-cap-per-digit", type=int, default=500)
    parser.add_argument("--fisher-batches", type=int, default=4)
    parser.add_argument("--torch-threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.torch_threads)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("GPU is required for the formal revision experiments")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_split_mnist(args.data_dir, args.train_cap_per_digit, args.test_cap_per_digit, seed=stable_seed("revision_bwt_data"))
    detail_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for seed in FORMAL_SEEDS:
        for order_id in FORMAL_ORDER_IDS:
            for method_cfg in METHOD_BUDGETS:
                histories = run_one_with_histories(data, seed, order_id, method_cfg, device, args.batch_size, args.epochs, args.fisher_batches)
                for task_id in range(5):
                    acc_just_after = histories[task_id][0]
                    acc_final = histories[task_id][-1]
                    bwt = acc_final - acc_just_after
                    retention = acc_final / max(acc_just_after, 1.0e-8)
                    detail_rows.append(
                        {
                            "seed": seed,
                            "order_id": order_id,
                            "method": method_cfg["method"],
                            "memory_budget": int(method_cfg["memory_budget"]),
                            "task_id": task_id,
                            "acc_just_after": rounded(acc_just_after),
                            "acc_final": rounded(acc_final),
                            "BWT": rounded(bwt),
                            "retention": rounded(retention, 4),
                        }
                    )
                write_csv(args.output_dir / "backward_transfer_detail.csv", detail_rows)
    summary = summarize_backward_transfer(detail_rows)
    write_json(args.output_dir / "backward_transfer_summary.json", summary)
    print(json.dumps({"script": Path(__file__).name, "status": "PASS", "rows": len(detail_rows), "output_dir": str(args.output_dir), "elapsed_seconds": round(time.perf_counter() - started, 3)}, sort_keys=True))


if __name__ == "__main__":
    main()
