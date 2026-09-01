# Data Files

The files contain public numeric artifacts for checking and reproducing the reported Split-MNIST task order and replay-budget grid.

- `results_120perm_4methods.csv`: main 120-permutation grid, 4,800 rows.
- `results_compute_matched.csv`: ER500 compute-matched diagnostic, 1,200 rows.
- `buffer_composition_log.csv`: replay-buffer composition trace, 18,000 rows.
- `legacy_formal_comparison.csv`: exact comparison between the retained 200-row reference grid and the corresponding rows in the 120-permutation grid.
- `formal_results_v2.csv`: retained 200-row reference grid used before the full 120-permutation expansion.
- `bwt_task_level_v2.csv`: per-task backward-transfer diagnostics for the retained reference grid.
- `extended_budget_v2.csv`: extended replay-budget diagnostics for the retained reference grid.
- `ewc_lambda_search.csv`: EWC lambda-search rows over seven lambda values and five task orders.
- `formal_results_v2_config.json`: dataset, task order, method, and training protocol values for the retained reference grid.
- `ewc_lambda_selection.json`: lambda-search summary and selected lambda `100.0`.
- `statistical_analysis_v2.json`: seed-level paired contrasts for the retained reference grid.
- `fisher_empirical_vs_batch_mean_square.json`: diagnostic comparing Fisher-estimator variants.
- `pilot_learning_adequacy.json`: pilot gate showing 10 epochs and batch size 128 were selected.
- `validation_summary_task1.json`: validation summary for the full-grid and compute-matched rerun.
- `environment_snapshot.json`: public-safe software and hardware environment snapshot with private executable paths removed.
- `public_summary.json`: compact file list, row counts, protocol values, headline metrics, contrasts, and order-regret anchors.
