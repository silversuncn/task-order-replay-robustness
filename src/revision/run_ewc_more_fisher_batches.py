#!/usr/bin/env python3
"""Revision experiment: EWC sweep over Fisher mini-batch counts."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import nn


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
    MLP,
    ewc_penalty,
    evaluate,
    load_split_mnist,
    make_loader,
    stable_seed,
    task_order,
    write_csv,
    write_json,
)


FISHER_BATCHES = [4, 8, 16, 32, 64]
SEEDS = [1, 2, 3, 4, 5]


def estimate_fisher_repeated(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    indices: torch.Tensor,
    device: torch.device,
    batch_size: int,
    max_batches: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Estimate Fisher with repeated shuffled passes so 8+ batches are real."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    per_batch: list[dict[str, torch.Tensor]] = []
    passes = 0
    while len(per_batch) < max_batches:
        loader = make_loader(x, y, indices, batch_size, seed=stable_seed("ewc_sweep_fisher", seed, passes, int(indices.numel())))
        for batch_x, batch_y in loader:
            if len(per_batch) >= max_batches:
                break
            model.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x.to(device)), batch_y.to(device))
            loss.backward()
            fisher_this = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    fisher_this[name] = param.grad.detach().cpu().pow(2)
            per_batch.append(fisher_this)
        passes += 1
    return {name: torch.stack([batch[name] for batch in per_batch], dim=0).mean(dim=0) for name in per_batch[0]}


def run_ewc_one(
    *,
    data: Any,
    seed: int,
    order_id: str,
    fisher_batches: int,
    device: torch.device,
    batch_size: int,
    epochs: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    criterion = nn.CrossEntropyLoss()
    anchors: list[dict[str, dict[str, torch.Tensor]]] = []
    histories: dict[int, list[float]] = {task_id: [] for task_id in range(5)}
    order = task_order(order_id)
    for position, task_id in enumerate(order):
        current = data.train_by_task[task_id]
        for epoch in range(epochs):
            loader = make_loader(data.train_x, data.train_y, current, batch_size, seed=stable_seed(seed, order_id, task_id, epoch))
            model.train()
            for batch_x, batch_y in loader:
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_x.to(device))
                loss = criterion(logits, batch_y.to(device))
                if anchors:
                    loss = loss + 25.0 * ewc_penalty(model, anchors, device)
                loss.backward()
                optimizer.step()
        for seen_task in order[: position + 1]:
            histories[seen_task].append(evaluate(model, data.test_x, data.test_y, data.test_by_task[seen_task], device, batch_size))
        anchors.append(
            {
                "params": {name: param.detach().cpu().clone() for name, param in model.named_parameters() if param.requires_grad},
                "fisher": estimate_fisher_repeated(model, data.train_x, data.train_y, current, device, batch_size, fisher_batches, seed=stable_seed(seed, order_id, task_id, fisher_batches)),
            }
        )
    final_task_acc = {f"task_{task_id}_final_accuracy": (histories[task_id][-1] if histories[task_id] else 0.0) for task_id in range(5)}
    forgetting_values = []
    for values in histories.values():
        if len(values) > 1:
            forgetting_values.append(max(values[:-1]) - values[-1])
    final_average_accuracy = sum(final_task_acc.values()) / 5
    average_forgetting = sum(forgetting_values) / len(forgetting_values) if forgetting_values else 0.0
    return {"final_average_accuracy": final_average_accuracy, "average_forgetting": average_forgetting, **final_task_acc}


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
    data = load_split_mnist(args.data_dir, args.train_cap_per_digit, args.test_cap_per_digit, seed=stable_seed("revision_ewc_sweep_data"))
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    total = len(FISHER_BATCHES) * len(SEEDS) * len(FORMAL_ORDER_IDS)
    run_index = 0
    for fisher_batches in FISHER_BATCHES:
        for seed in SEEDS:
            for order_id in FORMAL_ORDER_IDS:
                run_index += 1
                result = run_ewc_one(
                    data=data,
                    seed=seed,
                    order_id=order_id,
                    fisher_batches=fisher_batches,
                    device=device,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                )
                rows.append(
                    {
                        "seed": seed,
                        "order_id": order_id,
                        "fisher_batches": fisher_batches,
                        "final_average_accuracy": f"{float(result['final_average_accuracy']):.6f}",
                        "average_forgetting": f"{float(result['average_forgetting']):.6f}",
                        **{f"task_{i}_final_accuracy": f"{float(result[f'task_{i}_final_accuracy']):.6f}" for i in range(5)},
                    }
                )
                write_csv(args.output_dir / "ewc_fisher_batch_sweep.csv", rows)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["fisher_batches"])].append(row)
    baseline = sum(float(row["final_average_accuracy"]) for row in grouped[4]) / len(grouped[4])
    summary = []
    for fisher_batches in sorted(grouped):
        items = grouped[fisher_batches]
        accuracies = torch.tensor([float(row["final_average_accuracy"]) for row in items], dtype=torch.float64)
        forgetting = torch.tensor([float(row["average_forgetting"]) for row in items], dtype=torch.float64)
        mean_acc = float(accuracies.mean().item())
        summary.append(
            {
                "fisher_batches": fisher_batches,
                "improvement_over_4batch": float(f"{mean_acc - baseline:.6f}"),
                "mean_accuracy": float(f"{mean_acc:.6f}"),
                "mean_forgetting": float(f"{float(forgetting.mean().item()):.6f}"),
                "n_runs": len(items),
                "std_accuracy": float(f"{float(accuracies.std(unbiased=True).item()):.6f}"),
                "std_forgetting": float(f"{float(forgetting.std(unbiased=True).item()):.6f}"),
            }
        )
    write_json(args.output_dir / "ewc_fisher_batch_summary.json", summary)
    print(json.dumps({"script": Path(__file__).name, "status": "PASS", "rows": len(rows), "output_dir": str(args.output_dir), "elapsed_seconds": round(time.perf_counter() - started, 3)}, sort_keys=True))


if __name__ == "__main__":
    main()
