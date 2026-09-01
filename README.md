# Task-Order and Replay-Buffer Robustness of Continual Learning in a Two-Layer MLP

> Yaowen Sun; Qian Zhang; Xin Zhang

## Overview

This repository contains the public reproduction bundle for a bounded continual-learning measurement study on Split-MNIST. The study measures how deterministic task order and replay memory budget vary jointly for a compact class-incremental MLP. It does not introduce a new continual-learning algorithm and does not make state-of-the-art claims.

## Repository Structure

```text
.
|-- README.md
|-- LICENSE
|-- MANIFEST.txt
|-- SHA256SUMS
|-- checksums_sha256.txt
|-- requirements.txt
|-- data/
|   |-- README.md
|   |-- formal_results_v2.csv
|   |-- results_120perm_4methods.csv
|   |-- results_compute_matched.csv
|   |-- buffer_composition_log.csv
|   |-- legacy_formal_comparison.csv
|   |-- environment_snapshot.json
|   `-- public_summary.json
|-- figures/
|   |-- fig1_cell_faa_v2.png
|   |-- fig2_faa_boxplot_v2.png
|   `-- fig3_order_heatmap_v2.png
|-- scripts/
|   `-- generate_figures.py
|-- src/
|   |-- model.py
|   |-- data.py
|   |-- run_experiment.py
|   |-- launch_grid.py
|   |-- aggregate_results.py
|   |-- statistical_analysis.py
|   |-- verify_public_results.py
|   `-- methods/
|       |-- no_replay.py
|       |-- er.py
|       `-- ewc.py
`-- tests/
    `-- test_public_results.py
```

## Experimental Setup

| Dimension | Values | Count |
| --- | --- | ---: |
| Benchmark | Split-MNIST digit-pair tasks `0/1`, `2/3`, `4/5`, `6/7`, `8/9` | 5 tasks |
| Seeds | `1` through `10` | 10 |
| Task orders | all permutations of five tasks, `perm_000` through `perm_119` | 120 |
| Main method-budget cells | `no_replay/0`, `er100/100`, `er500/500`, `ewc/0` | 4 |
| Compute-matched diagnostic | `er500_matched/500` | 1 |
| Model | MLP `784 -> 256 -> 128 -> 10` | 1 |

Row-count checks:

```text
120 task orders x 10 seeds x 4 main cells = 4,800 main rows
120 task orders x 10 seeds x 1 compute-matched cell = 1,200 diagnostic rows
```

The grid caps MNIST at 1000 training examples and 500 test examples per digit. Training uses 10 epochs per task, batch size 128, Adam with learning rate `1e-3`, and a single-head 10-class output. The EWC penalty uses corrected empirical per-sample squared gradients with selected lambda `100.0`. Replay uses global-union subsampling with strong recency bias.

## Key Results

Mean final average accuracy and average forgetting are recomputed from `data/results_120perm_4methods.csv`:

| Method | Memory | FAA mean | FAA SD | AF mean | AF SD |
| --- | ---: | ---: | ---: | ---: | ---: |
| NoReplay | 0 | 0.197 | 0.002 | 0.987 | 0.007 |
| ER | 100 | 0.339 | 0.049 | 0.810 | 0.061 |
| ER | 500 | 0.611 | 0.050 | 0.464 | 0.062 |
| EWC | 0 | 0.198 | 0.002 | 0.989 | 0.003 |

Seed-level paired contrasts average each seed over all 120 task orders before testing. The EWC contrast is conditional on the selected-lambda procedure and should be interpreted as exploratory. The compute-matched ER500 diagnostic has FAA mean `0.583` with seed-level SD `0.004`.

## Verify Bundled Results

The verifier recomputes row counts, finite-value checks, method summaries, paired t tests with Holm correction, order regret, compute-matched summaries, figure presence, and public-safe environment fields:

```bash
python src/verify_public_results.py
python -m unittest discover -s tests -q
```

## Re-run the Pipeline

Install the recorded dependencies from `requirements.txt`. If the MNIST files are not already present, pass `--download-mnist`; otherwise the scripts use the local torchvision cache under `data/torchvision_data`.

Run one cell:

```bash
python src/run_experiment.py --method er500 --seed 1 --order-id perm_000 --allow-cpu
```

Run the full public grid:

```bash
python src/launch_grid.py --output results/results_120perm_4methods.csv --matched-output results/results_compute_matched.csv --allow-cpu
```

Aggregate and analyze:

```bash
python src/aggregate_results.py --data-dir data
python src/statistical_analysis.py --data-dir data
python scripts/generate_figures.py --data data/results_120perm_4methods.csv --output-dir figures
```

GPU execution is recommended for the full grid. CPU execution is practical for bundled-result verification and small smoke runs.

## Citation

```bibtex
@misc{sun2026taskorderreplayrobustness,
  title = {Task-Order and Replay-Buffer Robustness of Continual Learning in a Two-Layer MLP},
  author = {Sun, Yaowen and Zhang, Qian and Zhang, Xin},
  year = {2026}
}
```

## License

This public reproduction bundle is released under the license included in `LICENSE`.
