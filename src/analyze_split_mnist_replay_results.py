#!/usr/bin/env python3
"""Analyze Split-MNIST task-order replay results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def f(row: dict[str, Any], key: str) -> float:
    return float(row[key])


def grouped_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["method"], int(row["memory_budget"]))].append(row)
    out = []
    for (method, budget), vals in sorted(groups.items()):
        acc = [f(v, "final_average_accuracy") for v in vals]
        forget = [f(v, "average_forgetting") for v in vals]
        out.append(
            {
                "method": method,
                "memory_budget": budget,
                "n": len(vals),
                "mean_final_average_accuracy": mean(acc),
                "std_final_average_accuracy": stdev(acc) if len(acc) > 1 else 0.0,
                "mean_average_forgetting": mean(forget),
                "std_average_forgetting": stdev(forget) if len(forget) > 1 else 0.0,
            }
        )
    return out


def paired_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(r["seed"], r["order_id"], r["method"], int(r["memory_budget"])): r for r in rows}
    out = []
    for row in rows:
        if row["method"] == "no_replay":
            continue
        base = by_key.get((row["seed"], row["order_id"], "no_replay", 0))
        if not base:
            continue
        out.append(
            {
                "seed": row["seed"],
                "order_id": row["order_id"],
                "method": row["method"],
                "memory_budget": int(row["memory_budget"]),
                "delta_final_average_accuracy": f(row, "final_average_accuracy") - f(base, "final_average_accuracy"),
                "forgetting_reduction": f(base, "average_forgetting") - f(row, "average_forgetting"),
            }
        )
    return out


def order_sensitivity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["method"], int(row["memory_budget"]))].append(row)
    out = []
    for (method, budget), vals in sorted(groups.items()):
        by_order: dict[str, list[float]] = defaultdict(list)
        for row in vals:
            by_order[row["order_id"]].append(f(row, "final_average_accuracy"))
        means = {order: mean(values) for order, values in by_order.items()}
        out.append(
            {
                "method": method,
                "memory_budget": budget,
                "best_order": max(means, key=means.get),
                "worst_order": min(means, key=means.get),
                "order_regret": max(means.values()) - min(means.values()),
                **{f"order_mean_{k}": v for k, v in means.items()},
            }
        )
    return out


def make_figures(summary: list[dict[str, Any]], order_rows: list[dict[str, Any]], figure_dir: Path) -> None:
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    labels = [f"{r['method']}\n{r['memory_budget']}" for r in summary]
    acc = [float(r["mean_final_average_accuracy"]) for r in summary]
    forget = [float(r["mean_average_forgetting"]) for r in summary]

    plt.figure(figsize=(7, 4))
    plt.bar(labels, acc, color=["#6B7280", "#2563EB", "#1D4ED8", "#DC2626"])
    plt.ylabel("Final average accuracy")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(figure_dir / "method_final_accuracy.png", dpi=200)
    plt.savefig(figure_dir / "method_final_accuracy.pdf")
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.bar(labels, forget, color=["#6B7280", "#2563EB", "#1D4ED8", "#DC2626"])
    plt.ylabel("Average forgetting")
    plt.ylim(0, max(0.05, max(forget) * 1.2))
    plt.tight_layout()
    plt.savefig(figure_dir / "method_average_forgetting.png", dpi=200)
    plt.savefig(figure_dir / "method_average_forgetting.pdf")
    plt.close()

    plt.figure(figsize=(7, 3.5))
    plt.bar([f"{r['method']}\n{r['memory_budget']}" for r in order_rows], [float(r["order_regret"]) for r in order_rows], color="#059669")
    plt.ylabel("Order regret")
    plt.tight_layout()
    plt.savefig(figure_dir / "order_sensitivity.png", dpi=200)
    plt.savefig(figure_dir / "order_sensitivity.pdf")
    plt.close()


def decide_gate(rows: list[dict[str, Any]], summary: list[dict[str, Any]], deltas: list[dict[str, Any]], expected_rows: int) -> dict[str, Any]:
    numeric_bad = 0
    for row in rows:
        for key, value in row.items():
            if key.endswith("accuracy") or key == "average_forgetting":
                value_f = float(value)
                numeric_bad += int(math.isnan(value_f) or math.isinf(value_f))
    actual_rows = len(rows)
    baseline = next(r for r in summary if r["method"] == "no_replay" and int(r["memory_budget"]) == 0)
    best = max(summary, key=lambda r: float(r["mean_final_average_accuracy"]))
    best_delta = float(best["mean_final_average_accuracy"]) - float(baseline["mean_final_average_accuracy"])
    best_forgetting = min(summary, key=lambda r: float(r["mean_average_forgetting"]))
    forgetting_reduction = float(baseline["mean_average_forgetting"]) - float(best_forgetting["mean_average_forgetting"])
    adequate = actual_rows == expected_rows and numeric_bad == 0 and (best_delta >= 0.03 or forgetting_reduction >= 0.03)
    return {
        "status": "PASS_SCIENTIFIC_GATE_ADEQUATE" if adequate else "FORMAL_ANALYSIS_READY_NEEDS_MANAGER_DECISION",
        "actual_rows": actual_rows,
        "expected_rows": expected_rows,
        "numeric_bad_count": numeric_bad,
        "baseline_mean_final_average_accuracy": baseline["mean_final_average_accuracy"],
        "best_method": best["method"],
        "best_memory_budget": best["memory_budget"],
        "best_mean_final_average_accuracy": best["mean_final_average_accuracy"],
        "best_delta_vs_no_replay": best_delta,
        "best_forgetting_method": best_forgetting["method"],
        "best_forgetting_memory_budget": best_forgetting["memory_budget"],
        "forgetting_reduction_vs_no_replay": forgetting_reduction,
        "claim_boundary": "Split-MNIST finite grid with compact MLP, 10 seeds, five deterministic task orders, no universal CL claim.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=200)
    args = parser.parse_args()
    rows = read_csv(args.formal_dir / "metrics.csv")
    summary = grouped_summary(rows)
    deltas = paired_deltas(rows)
    orders = order_sensitivity(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "summary_by_method_budget.csv", summary)
    write_csv(args.output_dir / "paired_deltas_vs_no_replay.csv", deltas)
    write_csv(args.output_dir / "order_sensitivity.csv", orders)
    make_figures(summary, orders, args.figure_dir)
    gate = decide_gate(rows, summary, deltas, args.expected_rows)
    write_json(args.output_dir / "phase4_rebuild_analysis_result.json", gate)


if __name__ == "__main__":
    main()
