# Data Files

The files contain public numeric artifacts for checking the reported
Split-MNIST task-order and replay-budget grid.

- `formal_results_v2.csv`: run-level metrics for the 200-run grid.
- `bwt_task_level_v2.csv`: task-level backward-transfer diagnostics with 1,000
  rows.
- `extended_budget_v2.csv`: extended replay-budget diagnostics with 300 rows.
- `ewc_lambda_search.csv`: EWC lambda-search rows over seven lambda values and
  five task orders.
- `formal_results_v2_config.json`: dataset, task-order, method, and training
  protocol values.
- `ewc_lambda_selection.json`: lambda-search summary and selected lambda
  `100.0`.
- `statistical_analysis_v2.json`: seed-level paired contrasts with Holm
  correction.
- `fisher_empirical_vs_batch_mean_square.json`: diagnostic comparing the old and
  corrected Fisher estimators.
- `pilot_learning_adequacy.json`: pilot gate showing 10 epochs and batch size
  128 were selected.
- `environment_snapshot.json`: public-safe software and hardware environment
  snapshot with private executable paths removed.
- `public_summary.json`: compact file list, row counts, protocol values, and
  headline metrics.
