#!/usr/bin/env python3
"""Regenerate public figures from the 120-permutation CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DISPLAY = {
    "no_replay": "NoReplay",
    "er100": "ER100",
    "er500": "ER500",
    "ewc": "EWC",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/results_120perm_4methods.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.data)
    order = ["no_replay", "er100", "er500", "ewc"]

    means = df.groupby("method_cell")["final_average_accuracy"].mean().reindex(order)
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.bar([DISPLAY[item] for item in order], means.values, color=["#4c78a8", "#59a14f", "#f28e2b", "#e15759"])
    ax.set_ylabel("Final Average Accuracy")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(args.output_dir / "fig1_cell_faa_v2.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.boxplot([df.loc[df["method_cell"] == item, "final_average_accuracy"].to_numpy() for item in order], tick_labels=[DISPLAY[item] for item in order])
    ax.set_ylabel("Final Average Accuracy")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(args.output_dir / "fig2_faa_boxplot_v2.png", dpi=200)
    plt.close(fig)

    er500_order = df.loc[df["method_cell"] == "er500"].groupby("task_order")["final_average_accuracy"].mean().sort_values()
    selected_orders = list(er500_order.head(5).index) + list(er500_order.iloc[57:62].index) + list(er500_order.tail(5).index)
    matrix = [
        [df.loc[(df["task_order"] == task_order) & (df["method_cell"] == method), "final_average_accuracy"].mean() for method in order]
        for task_order in selected_orders
    ]
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    image = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([DISPLAY[item] for item in order])
    ax.set_yticks(range(len(selected_orders)))
    ax.set_yticklabels(selected_orders)
    fig.colorbar(image, ax=ax, label="FAA")
    fig.tight_layout()
    fig.savefig(args.output_dir / "fig3_order_heatmap_v2.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
