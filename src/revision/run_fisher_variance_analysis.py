#!/usr/bin/env python3
"""Revision experiment: Fisher diagonal variance by layer and batch count."""

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


SEEDS = [1, 2, 3]
FISHER_BATCHES = [4, 16, 32]


def estimate_fisher_per_batch(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    indices: torch.Tensor,
    device: torch.device,
    batch_size: int,
    max_batches: int,
    seed: int,
) -> list[dict[str, torch.Tensor]]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    per_batch: list[dict[str, torch.Tensor]] = []
    passes = 0
    while len(per_batch) < max_batches:
        loader = make_loader(x, y, indices, batch_size, seed=stable_seed("fisher_variance", seed, passes, int(indices.numel())))
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
    return per_batch


def summarize_per_layer(per_batch: list[dict[str, torch.Tensor]], seed: int, task_id: int, fisher_batches: int) -> list[dict[str, Any]]:
    rows = []
    names = sorted(per_batch[0])
    eps = 1.0e-12
    for name in names:
        stacked = torch.stack([batch[name].reshape(-1).float() for batch in per_batch], dim=0)
        means_per_param = stacked.mean(dim=0)
        stds_per_param = stacked.std(dim=0, unbiased=True)
        cv_values = stds_per_param / torch.clamp(means_per_param.abs(), min=eps)
        rows.append(
            {
                "seed": seed,
                "task_id": task_id,
                "fisher_batches": fisher_batches,
                "layer_name": name,
                "param_count": int(stacked.shape[1]),
                "mean_fisher": f"{float(stacked.mean().item()):.6f}",
                "std_fisher": f"{float(stacked.std(unbiased=True).item()):.6f}",
                "cv": f"{float(cv_values.mean().item()):.6f}",
            }
        )
    return rows


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
    data = load_split_mnist(args.data_dir, args.train_cap_per_digit, args.test_cap_per_digit, seed=stable_seed("revision_fisher_data"))
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for seed in SEEDS:
        torch.manual_seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model = MLP().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
        criterion = nn.CrossEntropyLoss()
        anchors: list[dict[str, dict[str, torch.Tensor]]] = []
        for task_id in task_order("canonical"):
            current = data.train_by_task[task_id]
            for epoch in range(args.epochs):
                loader = make_loader(data.train_x, data.train_y, current, args.batch_size, seed=stable_seed(seed, task_id, epoch))
                model.train()
                for batch_x, batch_y in loader:
                    optimizer.zero_grad(set_to_none=True)
                    loss = criterion(model(batch_x.to(device)), batch_y.to(device))
                    if anchors:
                        loss = loss + 25.0 * ewc_penalty(model, anchors, device)
                    loss.backward()
                    optimizer.step()
            for fisher_batches in FISHER_BATCHES:
                per_batch = estimate_fisher_per_batch(model, data.train_x, data.train_y, current, device, args.batch_size, fisher_batches, seed=stable_seed(seed, task_id, fisher_batches))
                rows.extend(summarize_per_layer(per_batch, seed, task_id, fisher_batches))
            anchor_batches = estimate_fisher_per_batch(model, data.train_x, data.train_y, current, device, args.batch_size, 4, seed=stable_seed("anchor", seed, task_id))
            fisher = {name: torch.stack([batch[name] for batch in anchor_batches], dim=0).mean(dim=0) for name in anchor_batches[0]}
            anchors.append({"params": {name: param.detach().cpu().clone() for name, param in model.named_parameters() if param.requires_grad}, "fisher": fisher})
            write_csv(args.output_dir / "fisher_variance_per_layer.csv", rows)
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["fisher_batches"]), int(row["task_id"]))].append(row)
    summary = []
    for fisher_batches, task_id in sorted(grouped):
        cvs = torch.tensor([float(row["cv"]) for row in grouped[(fisher_batches, task_id)]], dtype=torch.float64)
        summary.append(
            {
                "fisher_batches": fisher_batches,
                "max_cv": float(f"{float(cvs.max().item()):.6f}"),
                "median_cv": float(f"{float(cvs.median().item()):.6f}"),
                "n_layers": int(cvs.numel()),
                "p90_cv": float(f"{float(torch.quantile(cvs, 0.90).item()):.6f}"),
                "p95_cv": float(f"{float(torch.quantile(cvs, 0.95).item()):.6f}"),
                "stable": bool(float(cvs.median().item()) < 0.5),
                "task_id": task_id,
            }
        )
    write_json(args.output_dir / "fisher_variance_summary.json", summary)
    print(json.dumps({"script": Path(__file__).name, "status": "PASS", "rows": len(rows), "output_dir": str(args.output_dir), "elapsed_seconds": round(time.perf_counter() - started, 3)}, sort_keys=True))


if __name__ == "__main__":
    main()
