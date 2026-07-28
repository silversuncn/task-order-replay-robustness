#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def rows(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_report() -> dict[str, object]:
    summary = json.loads((DATA / "public_summary.json").read_text(encoding="utf-8"))
    formal = rows("formal_results.csv")
    datasets = {row["dataset"] for row in formal}
    replay = {row["replay_name"] for row in formal}
    seeds = {int(row["order_seed"]) for row in formal}
    duplicate_keys = len(formal) - len({(row["dataset"], row["replay_name"], row["order_seed"]) for row in formal})
    if len(formal) != 60:
        raise AssertionError(f"formal rows {len(formal)} != 60")
    if datasets != {"digits", "wine", "iris"}:
        raise AssertionError(sorted(datasets))
    if replay != {"no_replay", "replay_16", "replay_32", "replay_64"}:
        raise AssertionError(sorted(replay))
    if seeds != {601, 733, 887, 941, 1013}:
        raise AssertionError(sorted(seeds))
    if duplicate_keys:
        raise AssertionError(f"duplicate keys {duplicate_keys}")
    if any(row["status"] != "PASS" for row in formal):
        raise AssertionError("non-PASS rows found")
    if summary["row_counts"]["formal_results_rows"] != 60:
        raise AssertionError("public summary row count mismatch")
    return {"status": "PASS", "formal_results_rows": len(formal), "datasets": sorted(datasets), "replay_settings": sorted(replay), "order_seeds": sorted(seeds)}


if __name__ == "__main__":
    print(json.dumps(build_report(), indent=2, sort_keys=True))
