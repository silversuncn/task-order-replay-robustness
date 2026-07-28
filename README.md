# Task-Order Robustness of Lightweight Continual Classifiers under Tiny Replay Budgets

> **Task-Order Robustness of Lightweight Continual Classifiers under Tiny Replay Budgets**  
> Yaowen Sun

## Overview

This repository contains a sanitized reproduction bundle for a lightweight
continual-classification measurement study. It reports how final quality,
forgetting, and task-order regret vary under tiny replay memory budgets.

Repository URL: https://github.com/silversuncn/task-order-replay-robustness

## Repository Structure

```text
.
├── README.md
├── CITATION.cff
├── LICENSE
├── requirements.txt
├── data/
│   ├── public_summary.json
│   ├── formal_results.csv
│   ├── formal_replay_effects.json
│   └── formal_order_regret.json
├── figures/
│   └── formal_replay_effects_20260729.svg
├── src/
│   └── verify_public_results.py
└── tests/
    └── test_public_results.py
```

## Experimental Setup

| Dimension | Values | Count |
| --- | --- | ---: |
| Datasets | `digits`, `wine`, `iris` | 3 |
| Replay settings | `no_replay`, `replay_16`, `replay_32`, `replay_64` | 4 |
| Order seeds | 601, 733, 887, 941, 1013 | 5 |
| Classifier | fixed linear logistic-loss SGD classifier | 1 |

Row-count check:

```text
3 datasets x 4 replay settings x 5 order seeds = 60 runs
```

## Hardware & Environment

The matrix is CPU-only and uses Python, NumPy, SciPy, and scikit-learn. Runtime
columns are local diagnostic measurements, not portable deployment benchmarks.

## Key Results

- All 12 dataset-by-replay strata cross the planned task-order sensitivity threshold.
- None of the 9 paired replay-vs-no-replay comparisons crosses the planned replay threshold.
- Tiny replay has mixed dataset-dependent effects in this fixed lightweight setting.

## Requirements

The verification script uses only the Python standard library.

```bash
python src/verify_public_results.py
python -m unittest discover -s tests -q
```

## Citation

```bibtex
@article{sun2026taskorderreplayrobustness,
  title = {Task-Order Robustness of Lightweight Continual Classifiers under Tiny Replay Budgets},
  author = {Sun, Yaowen},
  year = {2026}
}
```
