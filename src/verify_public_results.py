#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"

EXPECTED_SEEDS = set(range(1, 11))
EXPECTED_ORDERS = {f"perm_{index:03d}" for index in range(120)}
EXPECTED_FULL_METHODS = {"no_replay", "er100", "er500", "ewc"}
EXPECTED_MATCHED_METHODS = {"er500_matched"}
EXPECTED_LEGACY_STATUS = {"PASS_EXACT", "PASS_EXACT_LEGACY_MATCH"}
EXPECTED_FIGURES = {
    "fig1_cell_faa_v2.png",
    "fig2_faa_boxplot_v2.png",
    "fig3_order_heatmap_v2.png",
}
PRIVATE_TOKENS = ("/home/", "/Users/", "silver", "miniconda")

TABLE_II = {
    "no_replay": {"faa_mean": 0.197, "faa_std": 0.002, "af_mean": 0.987, "af_std": 0.007},
    "er100": {"faa_mean": 0.339, "faa_std": 0.049, "af_mean": 0.810, "af_std": 0.061},
    "er500": {"faa_mean": 0.611, "faa_std": 0.050, "af_mean": 0.464, "af_std": 0.062},
    "ewc": {"faa_mean": 0.198, "faa_std": 0.002, "af_mean": 0.989, "af_std": 0.003},
}
CONTRAST_ANCHORS = {
    "er500_minus_no_replay": {"mean_delta": 0.4133, "dz": 122.09, "holm_p": 1.07e-19, "ci_low": 0.410849108285644, "ci_high": 0.41569189171435594},
    "er100_minus_no_replay": {"mean_delta": 0.1412, "dz": 46.96, "holm_p": 2.90e-16, "ci_low": 0.13906563130296629, "ci_high": 0.14336836869703368},
    "ewc_minus_no_replay": {"mean_delta": 0.0004, "dz": 3.18, "holm_p": 3.41e-6, "ci_low": 0.00029622628616753, "ci_high": 0.00046810704716580354},
    "er500_minus_er100": {"mean_delta": 0.2721, "dz": 142.17, "holm_p": 3.39e-20, "ci_low": 0.27068461526827026, "ci_high": 0.27342238473172986},
    "ewc_minus_er500": {"mean_delta": -0.4129, "dz": -120.54, "holm_p": 1.07e-19, "ci_low": -0.41533874958666617, "ci_high": -0.41043791708000055},
}
REGRET_ANCHORS = {
    "no_replay": {"best_order": "4-2-1-3-0", "worst_order": "3-2-4-0-1", "regret": 0.007},
    "er100": {"best_order": "4-1-2-3-0", "worst_order": "1-3-0-2-4", "regret": 0.196},
    "er500": {"best_order": "4-3-2-1-0", "worst_order": "0-1-2-3-4", "regret": 0.173},
    "ewc": {"best_order": "3-2-1-4-0", "worst_order": "0-3-4-2-1", "regret": 0.007},
}


def rows(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(name: str) -> dict[str, object]:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def assert_close(actual: float, expected: float, label: str, tol: float = 1e-12) -> None:
    if not math.isclose(actual, expected, rel_tol=tol, abs_tol=tol):
        raise AssertionError(f"{label}: {actual!r} != {expected!r}")


def assert_rounded(actual: float, expected: float, digits: int, label: str) -> None:
    rounded = round(actual, digits)
    if rounded != expected:
        raise AssertionError(f"{label}: round({actual!r}, {digits}) = {rounded!r}, expected {expected!r}")


def assert_finite(rows_in: Iterable[dict[str, str]], fields: list[str], label: str) -> None:
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


def assert_no_blank_or_nan_tokens(rows_in: list[dict[str, str]], label: str) -> None:
    bad = []
    for index, row in enumerate(rows_in):
        for field, value in row.items():
            normalized = str(value).strip().lower()
            if normalized in {"", "nan", "inf", "-inf"}:
                bad.append((index, field, value))
    if bad:
        raise AssertionError(f"{label} blank/non-finite tokens: {bad[:5]}")


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values)


def method_summary(full_rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in full_rows:
        grouped[row["method_cell"]]["faa"].append(float(row["final_average_accuracy"]))
        grouped[row["method_cell"]]["af"].append(float(row["AF"]))
    summary = {}
    for method, values in grouped.items():
        faa_mean, faa_std = mean_std(values["faa"])
        af_mean, af_std = mean_std(values["af"])
        summary[method] = {"faa_mean": faa_mean, "faa_std": faa_std, "af_mean": af_mean, "af_std": af_std, "n_rows": len(values["faa"])}
    return summary


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [1.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order, start=1):
        running = max(running, min(1.0, (len(p_values) - rank + 1) * p_values[index]))
        adjusted[index] = running
    return adjusted


def paired_contrasts(full_rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    grouped: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in full_rows:
        grouped[(int(row["seed"]), row["method_cell"])].append(float(row["final_average_accuracy"]))
    seed_means = {key: statistics.mean(values) for key, values in grouped.items()}
    seed_ids = sorted({seed for seed, _method in seed_means})
    comparisons = [
        ("er500", "no_replay"),
        ("er100", "no_replay"),
        ("ewc", "no_replay"),
        ("er500", "er100"),
        ("ewc", "er500"),
    ]
    output = {}
    raw_p = []
    labels = []
    for candidate, baseline in comparisons:
        diffs = [seed_means[(seed, candidate)] - seed_means[(seed, baseline)] for seed in seed_ids]
        delta = statistics.mean(diffs)
        sd = statistics.stdev(diffs)
        sem = sd / math.sqrt(len(diffs))
        test = stats.ttest_1samp(diffs, popmean=0.0)
        ci_low, ci_high = stats.t.interval(0.95, len(diffs) - 1, loc=delta, scale=sem)
        label = f"{candidate}_minus_{baseline}"
        labels.append(label)
        raw_p.append(float(test.pvalue))
        output[label] = {
            "mean_delta": delta,
            "dz": delta / sd if sd else 0.0,
            "raw_p": float(test.pvalue),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
        }
    for label, holm_p in zip(labels, holm_adjust(raw_p)):
        output[label]["holm_p"] = holm_p
    return output


def order_regret(full_rows: list[dict[str, str]]) -> dict[str, dict[str, float | str]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in full_rows:
        grouped[(row["method_cell"], row["task_order"])].append(float(row["final_average_accuracy"]))
    by_method: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for (method, order), values in grouped.items():
        by_method[method].append((order, statistics.mean(values)))
    output = {}
    for method, values in by_method.items():
        ordered = sorted(values, key=lambda item: item[1])
        worst_order, worst_value = ordered[0]
        best_order, best_value = ordered[-1]
        output[method] = {
            "best_order": best_order,
            "worst_order": worst_order,
            "regret": best_value - worst_value,
        }
    return output


def assert_public_environment(env: dict[str, object]) -> None:
    if "executable" in env:
        raise AssertionError("environment snapshot exposes executable")
    text = json.dumps(env, sort_keys=True)
    hits = [token for token in PRIVATE_TOKENS if token in text]
    if hits:
        raise AssertionError(f"environment snapshot exposes private tokens: {hits}")


def assert_figures() -> list[str]:
    figure_names = {path.name for path in FIGURES.iterdir() if path.is_file()}
    if figure_names != EXPECTED_FIGURES:
        raise AssertionError(f"figure set mismatch: {sorted(figure_names)}")
    for figure in EXPECTED_FIGURES:
        with (FIGURES / figure).open("rb") as handle:
            if handle.read(8) != b"\x89PNG\r\n\x1a\n":
                raise AssertionError(f"{figure} is not a PNG")
    return sorted(figure_names)


def build_report() -> dict[str, object]:
    public_summary = read_json("public_summary.json")
    environment = read_json("environment_snapshot.json")
    validation = read_json("validation_summary_task1.json")
    formal_reference = rows("formal_results_v2.csv")
    full = rows("results_120perm_4methods.csv")
    matched = rows("results_compute_matched.csv")
    buffer_rows = rows("buffer_composition_log.csv")
    legacy = rows("legacy_formal_comparison.csv")

    if len(formal_reference) != 200:
        raise AssertionError(f"formal reference rows {len(formal_reference)} != 200")
    if len(full) != 4800:
        raise AssertionError(f"120-permutation rows {len(full)} != 4800")
    if len(matched) != 1200:
        raise AssertionError(f"compute-matched rows {len(matched)} != 1200")
    if len(buffer_rows) != 18000:
        raise AssertionError(f"buffer rows {len(buffer_rows)} != 18000")
    if len(legacy) != 200:
        raise AssertionError(f"legacy comparison rows {len(legacy)} != 200")

    assert_no_duplicate_keys(full, ["order_id", "seed", "method_cell"], "results_120perm_4methods.csv")
    assert_no_duplicate_keys(matched, ["order_id", "seed", "method_cell"], "results_compute_matched.csv")
    assert_no_blank_or_nan_tokens(full, "results_120perm_4methods.csv")
    assert_no_blank_or_nan_tokens(matched, "results_compute_matched.csv")
    assert_no_blank_or_nan_tokens(buffer_rows, "buffer_composition_log.csv")
    assert_no_blank_or_nan_tokens(legacy, "legacy_formal_comparison.csv")

    if {int(row["seed"]) for row in full} != EXPECTED_SEEDS:
        raise AssertionError("unexpected seeds in full grid")
    if {row["order_id"] for row in full} != EXPECTED_ORDERS:
        raise AssertionError("unexpected permutation ids in full grid")
    if {row["method_cell"] for row in full} != EXPECTED_FULL_METHODS:
        raise AssertionError("unexpected method cells in full grid")
    if {row["method_cell"] for row in matched} != EXPECTED_MATCHED_METHODS:
        raise AssertionError("unexpected method cells in compute-matched data")

    numeric_fields = [
        "A_after_mean",
        "A_after_median",
        "A_after_min",
        "final_average_accuracy",
        "average_forgetting",
        "AF",
        "task_0_final_accuracy",
        "task_1_final_accuracy",
        "task_2_final_accuracy",
        "task_3_final_accuracy",
        "task_4_final_accuracy",
    ]
    assert_finite(full, numeric_fields, "results_120perm_4methods.csv")
    assert_finite(matched, numeric_fields, "results_compute_matched.csv")
    if not all(0.0 <= float(row["AF"]) <= 1.0 for row in full + matched):
        raise AssertionError("AF outside [0, 1]")
    if {int(float(row["optimizer_steps_total"])) for row in matched} != {800}:
        raise AssertionError("compute-matched optimizer steps are not fixed at 800")

    if validation["status"] != "PASS":
        raise AssertionError("task1 validation summary is not PASS")
    if validation["legacy_comparison"]["status"] != "PASS_EXACT_LEGACY_MATCH":
        raise AssertionError("legacy validation did not match exactly")
    if {row["status"] for row in legacy} - EXPECTED_LEGACY_STATUS:
        raise AssertionError("legacy comparison contains non-pass rows")

    summary = method_summary(full)
    for method, anchors in TABLE_II.items():
        assert_rounded(summary[method]["faa_mean"], anchors["faa_mean"], 3, f"{method} FAA mean")
        assert_rounded(summary[method]["faa_std"], anchors["faa_std"], 3, f"{method} FAA std")
        assert_rounded(summary[method]["af_mean"], anchors["af_mean"], 3, f"{method} AF mean")
        assert_rounded(summary[method]["af_std"], anchors["af_std"], 3, f"{method} AF std")
        if summary[method]["n_rows"] != 1200:
            raise AssertionError(f"{method} row count mismatch")

    matched_seed_means = defaultdict(list)
    for row in matched:
        matched_seed_means[int(row["seed"])].append(float(row["final_average_accuracy"]))
    matched_faa_mean = statistics.mean(float(row["final_average_accuracy"]) for row in matched)
    matched_faa_seed_std = statistics.stdev(statistics.mean(values) for values in matched_seed_means.values())
    assert_rounded(matched_faa_mean, 0.583, 3, "ER500-matched FAA mean")
    assert_rounded(matched_faa_seed_std, 0.004, 3, "ER500-matched seed-level FAA std")

    contrasts = paired_contrasts(full)
    for label, anchors in CONTRAST_ANCHORS.items():
        result = contrasts[label]
        assert_rounded(result["mean_delta"], anchors["mean_delta"], 4, f"{label} delta")
        assert_rounded(result["dz"], anchors["dz"], 2, f"{label} dz")
        assert_close(result["ci_low"], anchors["ci_low"], f"{label} CI low", tol=1e-12)
        assert_close(result["ci_high"], anchors["ci_high"], f"{label} CI high", tol=1e-12)
        assert_close(float(f"{result['holm_p']:.2e}"), anchors["holm_p"], f"{label} Holm p", tol=1e-22)

    regrets = order_regret(full)
    for method, anchors in REGRET_ANCHORS.items():
        result = regrets[method]
        if result["best_order"] != anchors["best_order"] or result["worst_order"] != anchors["worst_order"]:
            raise AssertionError(f"{method} order regret extrema mismatch: {result}")
        assert_rounded(float(result["regret"]), float(anchors["regret"]), 3, f"{method} regret")

    if public_summary["row_counts"]["results_120perm_4methods_rows"] != 4800:
        raise AssertionError("public summary full-grid row count mismatch")
    if public_summary["row_counts"]["results_compute_matched_rows"] != 1200:
        raise AssertionError("public summary compute-matched row count mismatch")
    assert_public_environment(environment)
    figure_names = assert_figures()

    return {
        "status": "PASS",
        "formal_results_v2_reference_rows": len(formal_reference),
        "results_120perm_4methods_rows": len(full),
        "results_compute_matched_rows": len(matched),
        "buffer_composition_rows": len(buffer_rows),
        "legacy_comparison_rows": len(legacy),
        "method_summary": {method: {key: round(value, 6) if isinstance(value, float) else value for key, value in values.items()} for method, values in sorted(summary.items())},
        "compute_matched": {"FAA_mean": round(matched_faa_mean, 6), "FAA_seed_std": round(matched_faa_seed_std, 6)},
        "paired_contrasts": {label: {key: round(value, 8) if isinstance(value, float) else value for key, value in values.items()} for label, values in contrasts.items()},
        "order_regret": regrets,
        "figures": figure_names,
    }


if __name__ == "__main__":
    print(json.dumps(build_report(), indent=2, sort_keys=True))
