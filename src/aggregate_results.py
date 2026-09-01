#!/usr/bin/env python3
"""Aggregate run-level public CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def method_summary(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["method_cell"]]["final_average_accuracy"].append(float(row["final_average_accuracy"]))
        grouped[row["method_cell"]]["average_forgetting"].append(float(row["AF"] if "AF" in row else row["average_forgetting"]))
    return {
        method: {
            "FAA_mean": mean_std(values["final_average_accuracy"])["mean"],
            "FAA_std": mean_std(values["final_average_accuracy"])["std"],
            "AF_mean": mean_std(values["average_forgetting"])["mean"],
            "AF_std": mean_std(values["average_forgetting"])["std"],
            "n_rows": len(values["final_average_accuracy"]),
        }
        for method, values in sorted(grouped.items())
    }


def build_summary(data_dir: Path) -> dict[str, object]:
    full = read_rows(data_dir / "results_120perm_4methods.csv")
    matched = read_rows(data_dir / "results_compute_matched.csv")
    return {
        "status": "PASS",
        "full_grid": method_summary(full),
        "compute_matched": method_summary(matched),
        "row_counts": {
            "results_120perm_4methods": len(full),
            "results_compute_matched": len(matched),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    print(json.dumps(build_summary(args.data_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
