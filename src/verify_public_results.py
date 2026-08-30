#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"

EXPECTED_SEEDS = set(range(1, 11))
EXPECTED_ORDERS = {"canonical", "shuffle_0", "shuffle_1", "shuffle_2", "shuffle_3"}
EXPECTED_METHOD_BUDGETS = {
    ("no_replay", 0),
    ("experience_replay", 100),
    ("experience_replay", 500),
    ("ewc", 0),
}
EXPECTED_METHOD_MEANS = {
    "er500": 0.591408,
    "er100": 0.318216,
    "ewc": 0.196428,
    "no_replay": 0.195824,
}
EXPECTED_FIGURES = {
    "fig1_cell_faa_v2.png",
    "fig2_faa_boxplot_v2.png",
    "fig3_order_heatmap_v2.png",
}
PRIVATE_TOKENS = ("/home/", "/Users/", "silver", "miniconda")


def rows(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(name: str) -> dict[str, object]:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def assert_close(actual: float, expected: float, label: str, tol: float = 1e-12) -> None:
    if not math.isclose(actual, expected, rel_tol=tol, abs_tol=tol):
        raise AssertionError(f"{label}: {actual!r} != {expected!r}")


def assert_finite(rows_in: list[dict[str, str]], fields: list[str], label: str) -> None:
    bad = []
    for row in rows_in:
        for field in fields:
            value = float(row[field])
            if not math.isfinite(value):
                bad.append((row.get("seed"), row.get("order_id"), field))
    if bad:
        raise AssertionError(f"{label} non-finite values: {bad[:3]}")


def assert_no_duplicate_keys(rows_in: list[dict[str, str]], fields: list[str], label: str) -> None:
    keys = [tuple(row[field] for field in fields) for row in rows_in]
    duplicate_count = len(keys) - len(set(keys))
    if duplicate_count:
        raise AssertionError(f"{label} duplicate keys: {duplicate_count}")


def summarize_method_cells(formal: list[dict[str, str]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in formal:
        grouped[row["method_cell"]].append(float(row["final_average_accuracy"]))
    return {key: mean(values) for key, values in grouped.items()}


def assert_public_environment(env: dict[str, object]) -> None:
    if "executable" in env:
        raise AssertionError("environment snapshot exposes executable")
    text = json.dumps(env, sort_keys=True)
    hits = [token for token in PRIVATE_TOKENS if token in text]
    if hits:
        raise AssertionError(f"environment snapshot exposes private tokens: {hits}")


def build_report() -> dict[str, object]:
    public_summary = read_json("public_summary.json")
    config = read_json("formal_results_v2_config.json")
    lambda_selection = read_json("ewc_lambda_selection.json")
    environment = read_json("environment_snapshot.json")
    formal = rows("formal_results_v2.csv")
    task_level = rows("bwt_task_level_v2.csv")
    extended = rows("extended_budget_v2.csv")
    lambda_rows = rows("ewc_lambda_search.csv")

    if len(formal) != 200:
        raise AssertionError(f"formal rows {len(formal)} != 200")
    if len(task_level) != 1000:
        raise AssertionError(f"task-level rows {len(task_level)} != 1000")
    if len(extended) != 300:
        raise AssertionError(f"extended-budget rows {len(extended)} != 300")
    if len(lambda_rows) != 35:
        raise AssertionError(f"lambda-search rows {len(lambda_rows)} != 35")

    row_counts = public_summary["row_counts"]
    expected_counts = {
        "formal_results_v2_rows": 200,
        "bwt_task_level_v2_rows": 1000,
        "extended_budget_v2_rows": 300,
        "ewc_lambda_search_rows": 35,
    }
    for key, expected in expected_counts.items():
        if row_counts[key] != expected:
            raise AssertionError(f"{key}: {row_counts[key]} != {expected}")

    if config["selected_epochs"] != 10:
        raise AssertionError("selected_epochs mismatch")
    if config["selected_batch_size"] != 128:
        raise AssertionError("selected_batch_size mismatch")
    if lambda_selection["selected_ewc_lambda"] != 100.0:
        raise AssertionError("selected_ewc_lambda mismatch")
    if public_summary["protocol"]["selected_epochs"] != 10:
        raise AssertionError("public summary selected_epochs mismatch")
    if public_summary["protocol"]["selected_batch_size"] != 128:
        raise AssertionError("public summary selected_batch_size mismatch")

    seeds = {int(row["seed"]) for row in formal}
    orders = {row["order_id"] for row in formal}
    method_budgets = {(row["method"], int(row["memory_budget"])) for row in formal}
    if seeds != EXPECTED_SEEDS:
        raise AssertionError(sorted(seeds))
    if orders != EXPECTED_ORDERS:
        raise AssertionError(sorted(orders))
    if method_budgets != EXPECTED_METHOD_BUDGETS:
        raise AssertionError(sorted(method_budgets))

    assert_no_duplicate_keys(
        formal,
        ["seed", "order_id", "method", "memory_budget"],
        "formal_results_v2.csv",
    )
    assert_no_duplicate_keys(
        task_level,
        ["seed", "order_id", "method", "memory_budget", "task_id"],
        "bwt_task_level_v2.csv",
    )
    assert_no_duplicate_keys(
        extended,
        ["seed", "order_id", "method", "memory_budget"],
        "extended_budget_v2.csv",
    )
    assert_no_duplicate_keys(
        lambda_rows,
        ["seed", "order_id", "ewc_lambda"],
        "ewc_lambda_search.csv",
    )

    formal_numeric_fields = [
        "final_average_accuracy",
        "average_forgetting",
        "elapsed_seconds",
        "A_after_mean",
        "A_after_median",
        "A_after_min",
        "task_0_final_accuracy",
        "task_1_final_accuracy",
        "task_2_final_accuracy",
        "task_3_final_accuracy",
        "task_4_final_accuracy",
    ]
    assert_finite(formal, formal_numeric_fields, "formal_results_v2.csv")
    assert_finite(
        task_level,
        ["A_after", "A_final", "BWT", "forgetting", "processed_examples"],
        "bwt_task_level_v2.csv",
    )
    assert_finite(extended, formal_numeric_fields, "extended_budget_v2.csv")
    assert_finite(lambda_rows, formal_numeric_fields, "ewc_lambda_search.csv")

    method_means = summarize_method_cells(formal)
    for method_cell, expected in EXPECTED_METHOD_MEANS.items():
        assert_close(round(method_means[method_cell], 6), expected, method_cell)
        summary_value = public_summary["method_ranking"][method_cell]["mean_final_average_accuracy"]
        assert_close(round(float(summary_value), 6), expected, f"summary {method_cell}")

    figure_names = {path.name for path in FIGURES.iterdir() if path.is_file()}
    if figure_names != EXPECTED_FIGURES:
        raise AssertionError(f"figure set mismatch: {sorted(figure_names)}")
    for figure in EXPECTED_FIGURES:
        with (FIGURES / figure).open("rb") as handle:
            if handle.read(8) != b"\x89PNG\r\n\x1a\n":
                raise AssertionError(f"{figure} is not a PNG")

    assert_public_environment(environment)

    return {
        "status": "PASS",
        "formal_results_v2_rows": len(formal),
        "bwt_task_level_v2_rows": len(task_level),
        "extended_budget_v2_rows": len(extended),
        "ewc_lambda_search_rows": len(lambda_rows),
        "selected_epochs": config["selected_epochs"],
        "selected_batch_size": config["selected_batch_size"],
        "selected_ewc_lambda": lambda_selection["selected_ewc_lambda"],
        "seeds": sorted(seeds),
        "orders": sorted(orders),
        "method_budgets": sorted([list(item) for item in method_budgets]),
        "method_means": {key: round(value, 6) for key, value in sorted(method_means.items())},
        "figures": sorted(figure_names),
    }


if __name__ == "__main__":
    print(json.dumps(build_report(), indent=2, sort_keys=True))
