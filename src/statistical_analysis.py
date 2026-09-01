#!/usr/bin/env python3
"""Seed-level paired tests, Holm correction, and order-regret diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from scipy import stats


CONTRASTS = [
    ("er500", "no_replay"),
    ("er100", "no_replay"),
    ("ewc", "no_replay"),
    ("er500", "er100"),
    ("ewc", "er500"),
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [1.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order, start=1):
        running = max(running, min(1.0, (len(p_values) - rank + 1) * p_values[index]))
        adjusted[index] = running
    return adjusted


def seed_method_means(rows: list[dict[str, str]]) -> dict[tuple[int, str], float]:
    grouped: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["seed"]), row["method_cell"])].append(float(row["final_average_accuracy"]))
    return {key: statistics.mean(values) for key, values in grouped.items()}


def paired_contrasts(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    means = seed_method_means(rows)
    seed_ids = sorted({seed for seed, _method in means})
    results = []
    raw_p = []
    for candidate, baseline in CONTRASTS:
        diffs = [means[(seed, candidate)] - means[(seed, baseline)] for seed in seed_ids]
        delta = statistics.mean(diffs)
        sd = statistics.stdev(diffs)
        sem = sd / math.sqrt(len(diffs))
        test = stats.ttest_1samp(diffs, popmean=0.0)
        ci_low, ci_high = stats.t.interval(0.95, len(diffs) - 1, loc=delta, scale=sem)
        raw_p.append(float(test.pvalue))
        results.append(
            {
                "comparison": f"{candidate}_minus_{baseline}",
                "mean_delta": delta,
                "dz": delta / sd if sd else 0.0,
                "raw_p": float(test.pvalue),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "n_independent_units": len(diffs),
            }
        )
    for result, holm_p in zip(results, holm_adjust(raw_p)):
        result["holm_p"] = holm_p
    return results


def order_regret(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["method_cell"], row["task_order"])].append(float(row["final_average_accuracy"]))
    by_method: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for (method, order), values in grouped.items():
        by_method[method].append((order, statistics.mean(values)))
    output = []
    for method, values in sorted(by_method.items()):
        values = sorted(values, key=lambda item: item[1])
        worst_order, worst_value = values[0]
        best_order, best_value = values[-1]
        output.append(
            {
                "method_cell": method,
                "best_order": best_order,
                "worst_order": worst_order,
                "regret": best_value - worst_value,
                "best_faa": best_value,
                "worst_faa": worst_value,
            }
        )
    return output


def build_analysis(data_dir: Path) -> dict[str, object]:
    rows = read_rows(data_dir / "results_120perm_4methods.csv")
    return {
        "status": "PASS",
        "paired_contrasts": paired_contrasts(rows),
        "order_regret": order_regret(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    print(json.dumps(build_analysis(args.data_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
