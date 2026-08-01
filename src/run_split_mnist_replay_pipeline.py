#!/usr/bin/env python3
"""Split-MNIST replay/EWC experiment pipeline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


TASKS = ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9))
PILOT_SEEDS = [1, 2]
FORMAL_SEEDS = list(range(1, 11))
PILOT_ORDER_IDS = ["canonical", "shuffle_0"]
FORMAL_ORDER_IDS = ["canonical", "shuffle_0", "shuffle_1", "shuffle_2", "shuffle_3"]
METHOD_BUDGETS = [
    {"method": "no_replay", "memory_budget": 0, "ewc_lambda": 0.0, "use_replay": False, "use_ewc": False},
    {"method": "experience_replay", "memory_budget": 100, "ewc_lambda": 0.0, "use_replay": True, "use_ewc": False},
    {"method": "experience_replay", "memory_budget": 500, "ewc_lambda": 0.0, "use_replay": True, "use_ewc": False},
    {"method": "ewc", "memory_budget": 0, "ewc_lambda": 25.0, "use_replay": False, "use_ewc": True},
]


def stable_seed(*parts: Any) -> int:
    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def task_order(order_id: str) -> list[int]:
    order = list(range(len(TASKS)))
    if order_id == "canonical":
        return order
    if not order_id.startswith("shuffle_"):
        raise ValueError(f"unknown order_id: {order_id}")
    shuffle_idx = int(order_id.split("_", 1)[1])
    rng = random.Random(stable_seed("split_mnist_order", shuffle_idx))
    rng.shuffle(order)
    return order


def expected_row_count(seeds: list[int], order_ids: list[str]) -> int:
    return len(seeds) * len(order_ids) * len(METHOD_BUDGETS)


def matrix_for_mode(mode: str) -> tuple[list[int], list[str]]:
    if mode == "pilot":
        return PILOT_SEEDS, PILOT_ORDER_IDS
    if mode == "formal":
        return FORMAL_SEEDS, FORMAL_ORDER_IDS
    raise ValueError(f"unknown mode: {mode}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def update_heartbeat(path: Path | None, **payload: Any) -> None:
    if path is None:
        return
    payload = {"updated_at_epoch": round(time.time(), 3), **payload}
    write_json(path, payload)


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class SplitMnistData:
    train_x: torch.Tensor
    train_y: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor
    train_by_task: dict[int, torch.Tensor]
    test_by_task: dict[int, torch.Tensor]


def load_split_mnist(data_dir: Path, train_cap_per_digit: int, test_cap_per_digit: int, seed: int) -> SplitMnistData:
    transform = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST(str(data_dir), train=True, download=True, transform=transform)
    test = datasets.MNIST(str(data_dir), train=False, download=True, transform=transform)

    def tensorize(ds: datasets.MNIST, cap_per_digit: int) -> tuple[torch.Tensor, torch.Tensor]:
        by_digit: dict[int, list[int]] = {digit: [] for digit in range(10)}
        labels = [int(y) for y in ds.targets.tolist()]
        for idx, label in enumerate(labels):
            by_digit[label].append(idx)
        xs: list[torch.Tensor] = []
        ys: list[int] = []
        rng = random.Random(seed)
        for digit in range(10):
            indices = list(by_digit[digit])
            rng.shuffle(indices)
            for idx in sorted(indices[:cap_per_digit]):
                x, y = ds[idx]
                xs.append(x.view(-1))
                ys.append(int(y))
        return torch.stack(xs).float(), torch.tensor(ys, dtype=torch.long)

    train_x, train_y = tensorize(train, train_cap_per_digit)
    test_x, test_y = tensorize(test, test_cap_per_digit)
    train_by_task = {}
    test_by_task = {}
    for task_id, digits in enumerate(TASKS):
        train_mask = torch.isin(train_y, torch.tensor(digits))
        test_mask = torch.isin(test_y, torch.tensor(digits))
        train_by_task[task_id] = train_mask.nonzero(as_tuple=False).view(-1)
        test_by_task[task_id] = test_mask.nonzero(as_tuple=False).view(-1)
    return SplitMnistData(train_x, train_y, test_x, test_y, train_by_task, test_by_task)


def make_loader(x: torch.Tensor, y: torch.Tensor, indices: torch.Tensor, batch_size: int, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset = TensorDataset(x[indices], y[indices])
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0)


def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor, indices: torch.Tensor, device: torch.device, batch_size: int) -> float:
    if indices.numel() == 0:
        return 0.0
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for start in range(0, indices.numel(), batch_size):
            batch_idx = indices[start : start + batch_size]
            logits = model(x[batch_idx].to(device))
            pred = logits.argmax(dim=1).cpu()
            truth = y[batch_idx]
            correct += int((pred == truth).sum().item())
            total += int(truth.numel())
    return float(correct / total) if total else 0.0


def select_replay_indices(memory: list[int], budget: int, seed: int) -> torch.Tensor:
    if budget <= 0 or not memory:
        return torch.empty(0, dtype=torch.long)
    rng = random.Random(seed)
    values = list(memory)
    rng.shuffle(values)
    return torch.tensor(sorted(values[:budget]), dtype=torch.long)


def update_memory(memory: list[int], candidates: torch.Tensor, budget: int, seed: int) -> list[int]:
    if budget <= 0:
        return []
    merged = list(set(memory + [int(i) for i in candidates.tolist()]))
    rng = random.Random(seed)
    rng.shuffle(merged)
    return sorted(merged[:budget])


def estimate_fisher(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    indices: torch.Tensor,
    device: torch.device,
    batch_size: int,
    max_batches: int,
) -> dict[str, torch.Tensor]:
    model.eval()
    fisher = {name: torch.zeros_like(param, device="cpu") for name, param in model.named_parameters() if param.requires_grad}
    criterion = nn.CrossEntropyLoss()
    loader = make_loader(x, y, indices, batch_size, seed=stable_seed("fisher", int(indices.numel())))
    used = 0
    for used, (batch_x, batch_y) in enumerate(loader, start=1):
        if used > max_batches:
            break
        model.zero_grad(set_to_none=True)
        loss = criterion(model(batch_x.to(device)), batch_y.to(device))
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher[name] += param.grad.detach().cpu().pow(2)
    denom = max(1, min(used, max_batches))
    return {name: value / denom for name, value in fisher.items()}


def ewc_penalty(model: nn.Module, anchors: list[dict[str, dict[str, torch.Tensor]]], device: torch.device) -> torch.Tensor:
    penalty = torch.tensor(0.0, device=device)
    for anchor in anchors:
        for name, param in model.named_parameters():
            if name in anchor["params"]:
                old_param = anchor["params"][name].to(device)
                fisher = anchor["fisher"][name].to(device)
                penalty = penalty + (fisher * (param - old_param).pow(2)).sum()
    return penalty


def run_one(
    *,
    data: SplitMnistData,
    seed: int,
    order_id: str,
    method_cfg: dict[str, Any],
    device: torch.device,
    batch_size: int,
    epochs: int,
    fisher_batches: int,
    heartbeat: Path | None,
    run_index: int,
    total_runs: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    criterion = nn.CrossEntropyLoss()
    memory: list[int] = []
    anchors: list[dict[str, dict[str, torch.Tensor]]] = []
    histories: dict[int, list[float]] = {task_id: [] for task_id in range(len(TASKS))}
    order = task_order(order_id)
    started = time.perf_counter()

    for position, task_id in enumerate(order):
        current = data.train_by_task[task_id]
        replay = select_replay_indices(memory, int(method_cfg["memory_budget"]), seed=stable_seed(seed, order_id, method_cfg["method"], task_id))
        train_indices = torch.unique(torch.cat([current, replay])) if replay.numel() else current
        for epoch in range(epochs):
            update_heartbeat(
                heartbeat,
                status="RUNNING",
                stage="train",
                run_index=run_index,
                total_runs=total_runs,
                seed=seed,
                order_id=order_id,
                method=method_cfg["method"],
                memory_budget=method_cfg["memory_budget"],
                task_id=task_id,
                task_position=position,
                epoch=epoch + 1,
                device=str(device),
                gpu_memory_allocated_mb=round(torch.cuda.memory_allocated(device) / 1048576, 2) if device.type == "cuda" else 0.0,
            )
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
        seen = order[: position + 1]
        for seen_task in seen:
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

    final_task_acc = {f"task_{task_id}_final_accuracy": (histories[task_id][-1] if histories[task_id] else 0.0) for task_id in range(len(TASKS))}
    forgetting_values = []
    for task_id, values in histories.items():
        if len(values) > 1:
            forgetting_values.append(max(values[:-1]) - values[-1])
    final_average_accuracy = sum(final_task_acc.values()) / len(TASKS)
    average_forgetting = sum(forgetting_values) / len(forgetting_values) if forgetting_values else 0.0
    return {
        "seed": seed,
        "order_id": order_id,
        "task_order": "-".join(str(i) for i in order),
        "method": method_cfg["method"],
        "memory_budget": int(method_cfg["memory_budget"]),
        "ewc_lambda": float(method_cfg["ewc_lambda"]),
        "epochs": epochs,
        "batch_size": batch_size,
        "final_average_accuracy": final_average_accuracy,
        "average_forgetting": average_forgetting,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "device": str(device),
        **final_task_acc,
    }


def validate_rows(rows: list[dict[str, Any]], expected: int) -> dict[str, Any]:
    numeric_keys = ["final_average_accuracy", "average_forgetting"] + [f"task_{i}_final_accuracy" for i in range(len(TASKS))]
    bad = []
    keys = set()
    duplicates = []
    for row in rows:
        key = (row["seed"], row["order_id"], row["method"], row["memory_budget"])
        if key in keys:
            duplicates.append(key)
        keys.add(key)
        for metric in numeric_keys:
            value = float(row[metric])
            if math.isnan(value) or math.isinf(value):
                bad.append((key, metric, value))
    return {
        "expected_rows": expected,
        "actual_rows": len(rows),
        "duplicate_key_count": len(duplicates),
        "nan_or_inf_count": len(bad),
        "mean_final_average_accuracy": sum(float(r["final_average_accuracy"]) for r in rows) / len(rows) if rows else 0.0,
        "min_final_average_accuracy": min(float(r["final_average_accuracy"]) for r in rows) if rows else 0.0,
        "status": "PASS" if len(rows) == expected and not duplicates and not bad else "FAIL",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pilot", "formal"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--heartbeat", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("../raw_data/torchvision_data"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train-cap-per-digit", type=int, default=1000)
    parser.add_argument("--test-cap-per-digit", type=int, default=500)
    parser.add_argument("--fisher-batches", type=int, default=4)
    parser.add_argument("--torch-threads", type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(args.torch_threads)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seeds, order_ids = matrix_for_mode(args.mode)
    expected = expected_row_count(seeds, order_ids)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    update_heartbeat(args.heartbeat, status="STARTING", stage="load_data", mode=args.mode, expected_rows=expected, device=str(device))
    data = load_split_mnist(args.data_dir, args.train_cap_per_digit, args.test_cap_per_digit, seed=stable_seed(args.mode, "data"))

    rows: list[dict[str, Any]] = []
    total = expected
    run_index = 0
    started = time.perf_counter()
    for seed in seeds:
        for order_id in order_ids:
            for method_cfg in METHOD_BUDGETS:
                run_index += 1
                rows.append(
                    run_one(
                        data=data,
                        seed=seed,
                        order_id=order_id,
                        method_cfg=method_cfg,
                        device=device,
                        batch_size=args.batch_size,
                        epochs=args.epochs,
                        fisher_batches=args.fisher_batches,
                        heartbeat=args.heartbeat,
                        run_index=run_index,
                        total_runs=total,
                    )
                )
                write_csv(args.output_dir / "metrics.csv", rows)
    summary = validate_rows(rows, expected)
    summary.update(
        {
            "mode": args.mode,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "train_cap_per_digit": args.train_cap_per_digit,
            "test_cap_per_digit": args.test_cap_per_digit,
            "seeds": seeds,
            "order_ids": order_ids,
            "method_budgets": METHOD_BUDGETS,
        }
    )
    write_json(args.output_dir / f"{args.mode}_summary.json", summary)
    config = {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()}
    write_json(args.output_dir / "config.json", config | {"device": str(device)})
    update_heartbeat(args.heartbeat, status=summary["status"], stage="complete", mode=args.mode, expected_rows=expected, actual_rows=len(rows))
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
