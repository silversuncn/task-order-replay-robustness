"""Split-MNIST data utilities and deterministic task order definitions."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


TASKS = ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9))
FORMAL_SEEDS = list(range(1, 11))


@dataclass
class SplitMnistData:
    train_x: torch.Tensor
    train_y: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor
    train_by_task: dict[int, torch.Tensor]
    test_by_task: dict[int, torch.Tensor]


def stable_seed(*parts: Any) -> int:
    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def all_task_orders() -> list[tuple[int, ...]]:
    return list(permutations(range(len(TASKS))))


def task_order(order_id: str) -> list[int]:
    if order_id.startswith("perm_"):
        index = int(order_id.split("_", 1)[1])
        orders = all_task_orders()
        if not 0 <= index < len(orders):
            raise ValueError(f"unknown permutation index: {order_id}")
        return list(orders[index])
    if order_id == "canonical":
        return list(range(len(TASKS)))
    if order_id.startswith("shuffle_"):
        shuffle_idx = int(order_id.split("_", 1)[1])
        order = list(range(len(TASKS)))
        rng = random.Random(stable_seed("split_mnist_order", shuffle_idx))
        rng.shuffle(order)
        return order
    raise ValueError(f"unknown order_id: {order_id}")


def load_split_mnist(
    data_dir: Path,
    train_cap_per_digit: int = 1000,
    test_cap_per_digit: int = 500,
    seed: int = stable_seed("revision_v2", "data"),
    download: bool = False,
) -> SplitMnistData:
    transform = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST(str(data_dir), train=True, download=download, transform=transform)
    test = datasets.MNIST(str(data_dir), train=False, download=download, transform=transform)

    def tensorize(ds: datasets.MNIST, cap_per_digit: int) -> tuple[torch.Tensor, torch.Tensor]:
        by_digit: dict[int, list[int]] = {digit: [] for digit in range(10)}
        labels = [int(label) for label in ds.targets.tolist()]
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
    train_by_task: dict[int, torch.Tensor] = {}
    test_by_task: dict[int, torch.Tensor] = {}
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
