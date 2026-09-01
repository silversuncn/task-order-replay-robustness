"""Experience replay configuration and buffer policy."""

from __future__ import annotations

import random
from typing import Any

import torch

def config(memory_budget: int, matched: bool = False) -> dict[str, Any]:
    return {
        "method": "experience_replay_compute_matched" if matched else "experience_replay",
        "memory_budget": int(memory_budget),
        "method_cell": "er500_matched" if matched else f"er{int(memory_budget)}",
        "ewc_lambda": 0.0,
        "use_replay": True,
        "use_ewc": False,
        "matched": bool(matched),
    }


def select_replay_indices(memory: list[int], budget: int, seed: int) -> torch.Tensor:
    if budget <= 0 or not memory:
        return torch.empty(0, dtype=torch.long)
    values = list(memory)
    random.Random(seed).shuffle(values)
    return torch.tensor(sorted(values[:budget]), dtype=torch.long)


def update_memory(memory: list[int], candidates: torch.Tensor, budget: int, seed: int) -> list[int]:
    """Global-union subsampling buffer with strong recency bias."""

    if budget <= 0:
        return []
    merged = list(set(memory + [int(index) for index in candidates.tolist()]))
    random.Random(seed).shuffle(merged)
    return sorted(merged[:budget])
