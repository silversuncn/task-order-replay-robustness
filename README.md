# Task-Order Replay Robustness on Split-MNIST with Lightweight Continual MLPs

> Yaowen Sun

## Overview

This repository contains a sanitized reproduction bundle for a bounded continual-learning measurement study. The study evaluates how task order and replay memory budget affect a compact class-incremental MLP on Split-MNIST. It does not introduce a new continual-learning algorithm and does not make state-of-the-art claims.

Repository URL: https://github.com/silversuncn/task-order-replay-robustness

## Repository Structure

```text
.
|-- README.md
|-- CITATION.cff
|-- LICENSE
|-- requirements.txt
|-- data/
|   |-- public_summary.json
|   |-- formal_metrics.csv
|   |-- formal_summary.json
|   |-- summary_by_method_budget.csv
|   |-- order_sensitivity.csv
|   `-- phase4_rebuild_analysis_result.json
|-- figures/
|   |-- method_final_accuracy.png
|   |-- method_average_forgetting.png
|   `-- order_sensitivity.png
|-- src/
|   |-- run_split_mnist_replay_pipeline.py
|   |-- analyze_split_mnist_replay_results.py
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

The formal run caps MNIST at 1000 training examples and 500 test examples per digit. Training uses two epochs per task, batch size 512, Adam with learning rate `1e-3`, and a single-head 10-class output.

## Hardware and Environment

The recorded formal matrix ran on CUDA device `cuda:0` with an NVIDIA RTX PRO 6000 Blackwell Workstation Edition GPU. The recorded software environment was PyTorch `2.11.0+cu128` with CUDA `12.8`.

CPU execution is sufficient for the verification script and unit tests. Re-running the full training pipeline requires installing PyTorch and torchvision and downloading MNIST through torchvision.

## Key Results

- The formal matrix completed all 200 expected rows with zero duplicate keys and zero NaN or infinite metric values.
- Experience replay with memory budget 500 achieved the highest mean final average accuracy: `0.233184`.
- The gain of experience replay with memory budget 500 over no replay was `0.058940`.
- EWC had nearly the same mean final average accuracy as no replay: `0.174440` versus `0.174244`.
- EWC reduced mean average forgetting by only `0.000670` versus no replay.
- The largest task-order regret was `0.110680` for experience replay with memory budget 500.

These are finite-grid claims about this Split-MNIST compact-MLP setup only.

## Reproduce and Verify

Verify the bundled public results using only the Python standard library:

```bash
python3 src/verify_public_results.py
python3 -m unittest discover -s tests -q
```

To rerun the training pipeline, install the requirements and run the formal mode:

```bash
python3 src/run_split_mnist_replay_pipeline.py --mode formal --output-dir outputs/formal --data-dir torchvision_data
python3 src/analyze_split_mnist_replay_results.py --metrics outputs/formal/metrics.csv --output-dir outputs/formal/analysis --figures-dir figures
```

The bundled `data/` files are the archived formal-run artifacts used by the manuscript.

## Citation

```bibtex
@article{sun2026taskorderreplayrobustness,
  title = {Task-Order Replay Robustness on Split-MNIST with Lightweight Continual MLPs},
  author = {Sun, Yaowen},
  year = {2026}
}
```

## License

This reproduction bundle is released under the license included in `LICENSE`.
