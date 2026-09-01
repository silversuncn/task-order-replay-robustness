"""NoReplay configuration."""

from __future__ import annotations

from typing import Any


def config() -> dict[str, Any]:
    return {
        "method": "no_replay",
        "memory_budget": 0,
        "method_cell": "no_replay",
        "ewc_lambda": 0.0,
        "use_replay": False,
        "use_ewc": False,
        "matched": False,
    }
