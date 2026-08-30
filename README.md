# Task-Order Replay Robustness on Split-MNIST with Lightweight Continual MLPs

> **Task-Order Replay Robustness on Split-MNIST with Lightweight Continual MLPs**  
> Yaowen Sun; Qian Zhang; Xin Zhang

## Overview

This repository contains a public verification bundle for a bounded continual-learning measurement study. The study evaluates how deterministic task order and replay memory budget affect a compact class-incremental MLP on Split-MNIST. It does not introduce a new continual-learning algorithm and does not make state-of-the-art claims.

## Repository Structure

```text
.
|-- README.md
|-- LICENSE
|-- MANIFEST.txt
|-- SHA256SUMS
|-- requirements.txt
|-- data/
|   |-- README.md
|   |-- bwt_task_level_v2.csv
|   |-- environment_snapshot.json
|   |-- ewc_lambda_search.csv
|   |-- ewc_lambda_selection.json
|   |-- extended_budget_v2.csv
|   |-- fisher_empirical_vs_batch_mean_square.json
|   |-- formal_results_v2.csv
|   |-- formal_results_v2_config.json
|   |-- pilot_learning_adequacy.json
|   |-- public_summary.json
|   `-- statistical_analysis_v2.json
|-- figures/
|   |-- fig1_cell_faa_v2.png
|   |-- fig2_faa_boxplot_v2.png
|   `-- fig3_order_heatmap_v2.png
|-- src/
|   `-- verify_public_results.py
`-- tests/
    `-- test_public_results.py
```

## Experimental Setup

| Dimension | Values | Count |
| --- | --- | ---: |
| Benchmark | Split-MNIST digit-pair tasks `0/1`, `2/3`, `4/5`, `6/7`, `8/9` | 5 tasks |
| Seeds | `1` through `10` | 10 |
| Task orders | `canonical`, `shuffle_0`, `shuffle_1`, `shuffle_2`, `shuffle_3` | 5 |
| Method-budget cells | `no_replay/0`, `experience_replay/100`, `experience_replay/500`, `ewc/0` | 4 |
| Model | MLP `784 -> 256 -> 128 -> 10` | 1 |

Row-count check:

```text
10 seeds x 5 task orders x 4 method-budget cells = 200 runs
```

The formal grid caps MNIST at 1000 training examples and 500 test examples per digit. Training uses 10 epochs per task, batch size 128, Adam with learning rate `1e-3`, and a single-head 10-class output. The EWC penalty uses corrected empirical per-sample squared gradients with selected lambda `100.0`.

## Hardware and Environment

The recorded grid used CUDA device `cuda:0` with an NVIDIA RTX PRO 6000 Blackwell Workstation Edition GPU. The recorded software environment was Python `3.11.15`, PyTorch `2.11.0+cu128`, torchvision `0.26.0+cu128`, and CUDA `12.8`. The bundled `data/environment_snapshot.json` removes private executable paths.

CPU execution is sufficient for the verification script and unit tests because they read and check bundled CSV/JSON files only.

## Key Results

- The formal grid completed all 200 expected run rows with zero duplicate keys and zero non-finite metric values.
- The task-level backward-transfer table contains 1,000 rows, the extended-budget diagnostic contains 300 rows, and the EWC lambda-search table contains 35 rows.
- Mean final average accuracy by method-budget cell is ER500 `0.591408`, ER100 `0.318216`, EWC `0.196428`, and NoReplay `0.195824`.
- ER500 improves over NoReplay by `0.395584`; ER100 improves over NoReplay by `0.122392`.
- EWC with selected lambda `100.0` is statistically positive but numerically close to NoReplay in this grid.
- The largest task-order regret is `0.134` for ER500, smaller than the ER500-NoReplay method benefit in this measured grid.

These are finite-grid claims about this Split-MNIST compact-MLP setup only.

## Verify

Verify the bundled public results using only the Python standard library:

```bash
python3 src/verify_public_results.py
python3 -m unittest discover -s tests -q
```

The verifier checks row counts, protocol values, selected lambda, finite metrics, duplicate keys, method means, figure presence, and removal of private local paths from the bundled environment snapshot.

## Requirements

The bundled verifier uses the Python standard library. Re-running training or
figure generation requires the packages listed in `requirements.txt`, including
PyTorch and torchvision.

## Citation

```bibtex
@misc{sun2026taskorderreplayrobustness,
  title = {Task-Order Replay Robustness on Split-MNIST with Lightweight Continual MLPs},
  author = {Sun, Yaowen and Zhang, Qian and Zhang, Xin},
  year = {2026}
}
```

## License

This public verification bundle is released under the license included in `LICENSE`.
