#!/usr/bin/env python3
"""Run one Split-MNIST task order and replay experiment."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from data import SplitMnistData, evaluate, load_split_mnist, make_loader, stable_seed, task_order
from methods import er, ewc, no_replay
from model import MLP


def method_config(name: str) -> dict[str, Any]:
    if name == "no_replay":
        return no_replay.config()
    if name == "er100":
        return er.config(100)
    if name == "er500":
        return er.config(500)
    if name == "ewc":
        return ewc.config(100.0)
    if name == "er500_matched":
        return er.config(500, matched=True)
    raise ValueError(f"unknown method: {name}")


def train_task(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    data: SplitMnistData,
    train_indices: torch.Tensor,
    device: torch.device,
    batch_size: int,
    loader_seed_parts: tuple[Any, ...],
    epochs: int,
    step_cap: int | None,
    anchors: list[dict[str, dict[str, torch.Tensor]]],
    cfg: dict[str, Any],
) -> tuple[int, int, int]:
    optimizer_steps = 0
    processed_examples = 0
    effective_epochs = 0
    epoch = 0
    while True:
        if step_cap is not None and optimizer_steps >= step_cap:
            break
        if step_cap is None and epoch >= epochs:
            break
        loader = make_loader(data.train_x, data.train_y, train_indices, batch_size, seed=stable_seed(*loader_seed_parts, epoch))
        model.train()
        took_step = False
        for batch_x, batch_y in loader:
            if step_cap is not None and optimizer_steps >= step_cap:
                break
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x.to(device))
            loss = criterion(logits, batch_y.to(device))
            if cfg["use_ewc"] and anchors:
                loss = loss + float(cfg["ewc_lambda"]) * ewc.penalty(model, anchors, device)
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
            processed_examples += int(batch_y.numel())
            took_step = True
        if took_step:
            effective_epochs += 1
        epoch += 1
    return optimizer_steps, processed_examples, effective_epochs


def run_single(
    *,
    data: SplitMnistData,
    seed: int,
    order_id: str,
    method: str,
    device: torch.device,
    batch_size: int = 128,
    epochs: int = 10,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cfg = method_config(method)
    model = MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    criterion = nn.CrossEntropyLoss()
    order = task_order(order_id)
    memory: list[int] = []
    anchors: list[dict[str, dict[str, torch.Tensor]]] = []
    histories: dict[int, list[float]] = {task_id: [] for task_id in range(5)}
    a_after_by_task: dict[int, float] = {}
    optimizer_steps_per_task: list[int] = []
    processed_examples_per_task: list[int] = []
    effective_epoch_passes: list[int] = []
    task_rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    no_replay_step_cap = None
    if cfg["matched"]:
        no_replay_step_cap = epochs * ((int(data.train_by_task[0].numel()) + batch_size - 1) // batch_size)

    for position, task_id in enumerate(order):
        current = data.train_by_task[task_id]
        replay_seed_method = "experience_replay" if cfg["matched"] else cfg["method"]
        replay = er.select_replay_indices(memory, int(cfg["memory_budget"]), seed=stable_seed(seed, order_id, replay_seed_method, task_id))
        train_indices = torch.unique(torch.cat([current, replay])) if replay.numel() else current
        task_steps, task_examples, effective_epochs = train_task(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            data=data,
            train_indices=train_indices,
            device=device,
            batch_size=batch_size,
            loader_seed_parts=(seed, order_id, task_id),
            epochs=epochs,
            step_cap=no_replay_step_cap,
            anchors=anchors,
            cfg=cfg,
        )
        a_after = evaluate(model, data.test_x, data.test_y, data.test_by_task[task_id], device, batch_size)
        a_after_by_task[task_id] = a_after
        optimizer_steps_per_task.append(task_steps)
        processed_examples_per_task.append(task_examples)
        effective_epoch_passes.append(effective_epochs)

        for seen_task in order[: position + 1]:
            histories[seen_task].append(evaluate(model, data.test_x, data.test_y, data.test_by_task[seen_task], device, batch_size))

        task_rows.append(
            {
                "seed": seed,
                "order_id": order_id,
                "task_order": "-".join(str(item) for item in order),
                "method": cfg["method"],
                "memory_budget": int(cfg["memory_budget"]),
                "method_cell": cfg["method_cell"],
                "task_id": task_id,
                "task_position": position,
                "A_after": a_after,
                "A_final": 0.0,
                "BWT": 0.0,
                "forgetting": 0.0,
                "optimizer_steps": task_steps,
                "processed_examples": task_examples,
                "current_examples": int(current.numel()),
                "replay_examples": int(replay.numel()),
                "train_examples_total": int(train_indices.numel()),
            }
        )

        if cfg["use_replay"]:
            memory = er.update_memory(memory, current, int(cfg["memory_budget"]), seed=stable_seed("memory", seed, order_id, task_id))
        if cfg["use_ewc"] and position < len(order) - 1:
            anchors.append(
                {
                    "params": {name: param.detach().cpu().clone() for name, param in model.named_parameters() if param.requires_grad},
                    "fisher": ewc.estimate_fisher(model, data.train_x, data.train_y, current, device, batch_size),
                }
            )

    final_task_acc = {task_id: (histories[task_id][-1] if histories[task_id] else 0.0) for task_id in range(5)}
    for task_row in task_rows:
        final_value = final_task_acc[int(task_row["task_id"])]
        task_row["A_final"] = final_value
        task_row["BWT"] = final_value - float(task_row["A_after"])
        task_row["forgetting"] = float(task_row["A_after"]) - final_value

    forgetting_values = [max(values[:-1]) - values[-1] for values in histories.values() if len(values) > 1]
    a_after_values = [a_after_by_task[task_id] for task_id in order]
    row = {
        "seed": seed,
        "order_id": order_id,
        "task_order": "-".join(str(item) for item in order),
        "method": cfg["method"],
        "memory_budget": int(cfg["memory_budget"]),
        "method_cell": cfg["method_cell"],
        "status": "PASS",
        "ewc_lambda": float(cfg["ewc_lambda"]),
        "selected_ewc_lambda": float(cfg["ewc_lambda"]) if cfg["use_ewc"] else 0.0,
        "epochs": epochs,
        "batch_size": batch_size,
        "optimizer_steps_total": sum(optimizer_steps_per_task),
        "optimizer_steps_per_task": ";".join(str(value) for value in optimizer_steps_per_task),
        "processed_examples_total": sum(processed_examples_per_task),
        "processed_examples_per_task": ";".join(str(value) for value in processed_examples_per_task),
        "effective_epoch_passes_per_task": ";".join(str(value) for value in effective_epoch_passes),
        "A_after_mean": sum(a_after_values) / len(a_after_values),
        "A_after_median": statistics.median(a_after_values),
        "A_after_min": min(a_after_values),
        "A_after_le_0_001_count": sum(1 for value in a_after_values if value <= 0.001),
        "final_average_accuracy": sum(final_task_acc.values()) / 5,
        "average_forgetting": sum(forgetting_values) / len(forgetting_values) if forgetting_values else 0.0,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "device": str(device),
        **{f"task_{task_id}_final_accuracy": final_task_acc[task_id] for task_id in range(5)},
        "_task_rows": task_rows,
    }
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["no_replay", "er100", "er500", "ewc", "er500_matched"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--order-id", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/torchvision_data"))
    parser.add_argument("--download-mnist", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train-cap-per-digit", type=int, default=1000)
    parser.add_argument("--test-cap-per-digit", type=int, default=500)
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
    row = run_single(
        data=data,
        seed=args.seed,
        order_id=args.order_id,
        method=args.method,
        device=device,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )
    task_rows = row.pop("_task_rows")
    print(json.dumps({"run": row, "task_rows": task_rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
