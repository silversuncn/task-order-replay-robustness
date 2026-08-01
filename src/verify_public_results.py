#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

EXPECTED_SEEDS = set(range(1, 11))
EXPECTED_ORDERS = {"canonical", "shuffle_0", "shuffle_1", "shuffle_2", "shuffle_3"}
EXPECTED_METHOD_BUDGETS = {
    ("no_replay", 0),
    ("experience_replay", 100),
    ("experience_replay", 500),
    ("ewc", 0),
}


def rows(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(name: str) -> dict[str, object]:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def assert_close(actual: float, expected: float, label: str, tol: float = 1e-12) -> None:
    if not math.isclose(actual, expected, rel_tol=tol, abs_tol=tol):
        raise AssertionError(f"{label}: {actual!r} != {expected!r}")


def summarize_method_budget(formal: list[dict[str, str]]) -> dict[tuple[str, int], dict[str, float]]:
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in formal:
        grouped[(row["method"], int(row["memory_budget"]))].append(row)
    out: dict[tuple[str, int], dict[str, float]] = {}
    for key, vals in grouped.items():
        acc = [float(v["final_average_accuracy"]) for v in vals]
        forgetting = [float(v["average_forgetting"]) for v in vals]
        out[key] = {
            "n": float(len(vals)),
            "mean_final_average_accuracy": mean(acc),
            "std_final_average_accuracy": stdev(acc),
            "mean_average_forgetting": mean(forgetting),
            "std_average_forgetting": stdev(forgetting),
        }
    return out


def summarize_order_regret(formal: list[dict[str, str]]) -> dict[tuple[str, int], dict[str, object]]:
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in formal:
        grouped[(row["method"], int(row["memory_budget"]))].append(row)
    out: dict[tuple[str, int], dict[str, object]] = {}
    for key, vals in grouped.items():
        by_order: dict[str, list[float]] = defaultdict(list)
        for row in vals:
            by_order[row["order_id"]].append(float(row["final_average_accuracy"]))
        order_means = {order: mean(values) for order, values in by_order.items()}
        best = max(order_means, key=order_means.get)
        worst = min(order_means, key=order_means.get)
        out[key] = {
            "best_order": best,
            "worst_order": worst,
            "order_regret": order_means[best] - order_means[worst],
        }
    return out


def build_report() -> dict[str, object]:
    public_summary = read_json("public_summary.json")
    formal_summary = read_json("formal_summary.json")
    phase4 = read_json("phase4_rebuild_analysis_result.json")
    formal = rows("formal_metrics.csv")
    method_summary = rows("summary_by_method_budget.csv")
    order_summary = rows("order_sensitivity.csv")

    if len(formal) != 200:
        raise AssertionError(f"formal rows {len(formal)} != 200")
    if formal_summary["actual_rows"] != 200 or formal_summary["expected_rows"] != 200:
        raise AssertionError("formal summary row-count mismatch")
    if public_summary["row_counts"]["formal_metrics_rows"] != 200:
        raise AssertionError("public summary row-count mismatch")
    if phase4["status"] != "PASS_SCIENTIFIC_GATE_ADEQUATE":
        raise AssertionError("phase4 scientific gate did not pass")

    seeds = {int(row["seed"]) for row in formal}
    orders = {row["order_id"] for row in formal}
    method_budgets = {(row["method"], int(row["memory_budget"])) for row in formal}
    if seeds != EXPECTED_SEEDS:
        raise AssertionError(sorted(seeds))
    if orders != EXPECTED_ORDERS:
        raise AssertionError(sorted(orders))
    if method_budgets != EXPECTED_METHOD_BUDGETS:
        raise AssertionError(sorted(method_budgets))

    keys = [(row["seed"], row["order_id"], row["method"], row["memory_budget"]) for row in formal]
    duplicate_keys = len(keys) - len(set(keys))
    if duplicate_keys:
        raise AssertionError(f"duplicate keys {duplicate_keys}")

    numeric_fields = [
        "final_average_accuracy",
        "average_forgetting",
        "elapsed_seconds",
        "task_0_final_accuracy",
        "task_1_final_accuracy",
        "task_2_final_accuracy",
        "task_3_final_accuracy",
        "task_4_final_accuracy",
    ]
    bad_numeric = []
    for row in formal:
        for field in numeric_fields:
            value = float(row[field])
            if not math.isfinite(value):
                bad_numeric.append((row["seed"], row["order_id"], row["method"], field))
    if bad_numeric:
        raise AssertionError(f"non-finite values: {bad_numeric[:3]}")

    recomputed = summarize_method_budget(formal)
    for row in method_summary:
        key = (row["method"], int(row["memory_budget"]))
        stats = recomputed[key]
        assert_close(stats["n"], float(row["n"]), f"{key} n")
        for field in [
            "mean_final_average_accuracy",
            "std_final_average_accuracy",
            "mean_average_forgetting",
            "std_average_forgetting",
        ]:
            assert_close(stats[field], float(row[field]), f"{key} {field}")

    recomputed_order = summarize_order_regret(formal)
    for row in order_summary:
        key = (row["method"], int(row["memory_budget"]))
        stats = recomputed_order[key]
        if stats["best_order"] != row["best_order"] or stats["worst_order"] != row["worst_order"]:
            raise AssertionError(f"{key} best/worst order mismatch")
        assert_close(float(stats["order_regret"]), float(row["order_regret"]), f"{key} order_regret")

    return {
        "status": "PASS",
        "formal_metrics_rows": len(formal),
        "seeds": sorted(seeds),
        "orders": sorted(orders),
        "method_budgets": sorted([list(item) for item in method_budgets]),
        "best_method": phase4["best_method"],
        "best_memory_budget": phase4["best_memory_budget"],
        "best_mean_final_average_accuracy": phase4["best_mean_final_average_accuracy"],
        "largest_order_regret": public_summary["headline_values"]["largest_order_regret"],
    }


if __name__ == "__main__":
    print(json.dumps(build_report(), indent=2, sort_keys=True))
