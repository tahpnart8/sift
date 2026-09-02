# Where every number comes from

[Back to README](../README.md)

One row per table or claim in the paper, naming the file and the columns. Unless
stated otherwise every number is `metric == "macro_f1"`, seeds averaged within a
cell before anything else, cut-points never pooled.

A cell is one cut-point with one model. There are 20. Files with 100 rows carry
the seed dimension and must be collapsed to 20 before use; files with 20 rows
have already been collapsed.

## Tables

### Table 1, the three kinds of shift

Conceptual, from the literature. No result file.

### Table 2, data and experimental setup

`results/evidence/dataset_summary.parquet`, long format with columns `section`,
`key`, `value`, `note`.

- Raw and final sample counts, feature and family counts: `section ==
  "provenance"` and `section == "overall"`.
- Per-family sizes, the 20 to 119 range and the median of 34.5: `section ==
  "family_size"` and `family_size_stats`.
- Per-cut training and test sizes on both time axes, and the unseen-family
  fractions quoted in Section 3.5: `section == "cut_primary"` for the correct
  axis, `cut_secondary` for the compile-timestamp axis. The key
  `cutNNNN_share_test_rows_unseen` gives the 0.652 against 0.413 comparison at
  cut 2017 and the exact zero at cut 2023.
- How far the two axes disagree, the raw material of control B2: `section ==
  "axis_disagreement"`.

### Table 3, Delta at the baseline configuration

`results/metrics.parquet`, filtered to the baseline configuration, meaning all
four flags `a1_prior`, `a2_labels`, `b1_fs`, `b2_axis` false. Average the five
seeds, then the four models.

- Delta: `M(random_fully_matched) - M(temporal)`.
- Reported gap: `M(random) - M(temporal)`.
- Sample-size gap: `M(random) - M(random_matched)`.
- Class composition: `M(random_matched) - M(random_fully_matched)`.
- Range over the four models: `results/shapley_2024/model_summary_macro_f1.parquet`,
  column `delta_empty`, min and max over the four rows of each cut.

### Table 4, the decomposition by cut

`results/shapley_2024/decomposition_as_reported_macro_f1.parquet`, 100 rows,
5 cuts by 4 models by 5 seeds. Columns `delta_empty`, `phi_a1_prior`,
`phi_a2_labels`, `phi_b1_fs`, `phi_b2_axis`, `residual`.

`v(N)` has no column of its own. It is the sum of the four `phi_*`, equivalently
`delta_empty - residual`.

`results/decomp_2024.parquet` is the same quantity already collapsed to 20 cells
and is the convenient starting point.

### Table 5, the residual and the input-shift test

Two files, joined on `cut`.

- `results/shapley_2024/residual_ci.parquet`, 20 rows. Columns `R`,
  `half_width`, `excludes_zero`, `passes_gate`. The Largest half-width column is
  the maximum of `half_width` over the four models of that cut; the Excludes 0
  column counts `excludes_zero` over the same four.
- `results/shapley_2024/domain_classifier.parquet`, 5 rows. Columns `accuracy`,
  `p_value`, `n_heldout`, `interpretable`. The pre-registered threshold of 100
  held-out samples is the column `min_interpretable_heldout`, not a constant in
  the prose.

**The two R columns are not the same computation.** Table 4 R is lattice
arithmetic, `delta_empty` minus the sum of the four contributions. Table 5 R is a
bootstrap over the stored prediction table. The paper states they differ by at
most 0.040. Do not mix them inside one claim, and do not source Table 5 R from
`model_summary_macro_f1.parquet`.

## Section 3.5, task selection

The binary preliminary check, three parts of one finding:

- `results/evidence/item2_binary_gap.parquet`, the 0.0039 PR-AUC gap.
- `results/evidence/item2_source_shortcut.parquet`, collection source predicted
  at ROC-AUC 0.96 to 0.99, including within the ransomware class.
- `results/evidence/item2_binary_stress_test.parquet`.

This is the only place PR-AUC appears. Everywhere else is macro-F1.

## Section 4.2, the shape of the decomposition

| Claim | File | Columns |
|---|---|---|
| 25 of 80 contributions contain zero under seed intervals, 58 of 80 under bootstrap | `phi_ci_coverage_counts.parquet` | `interval`, `n_cells`, `n_covering_zero` |
| The two interval constructions themselves | `phi_ci_seed.parquet`, `phi_ci_bootstrap.parquet` | `ci_low`, `ci_high`, `covers_zero`, `n_draws` |
| Median protocol-side share of 32.1% | `owen_protocol_share.parquet` | `protocol_share_median`, global row `scope == "all_cells"` |
| Owen values per cell | `owen_tiers.parquet` | `owen_*`, `protocol_share` |

All four live in `results/shapley_2024/`.

## Section 4.3, interaction and sequential ablation

- Harsanyi dividends: `results/shapley_2024/dividends_macro_f1.parquet`, 100
  rows. The A2 with B2 interaction is column `a2_labels+b2_axis`; the A2
  singleton is column `a2_labels`. Confusing the two is the error the paper
  warns about. `results/dividends_2024.parquet` is the same collapsed to 20
  cells.
- All 24 switch-off orders: `results/evidence/sequential_vs_lattice.parquet`.
  Columns `shapley`, `seq_min`, `seq_max`, `seq_width`, `order_at_min`,
  `order_at_max`. The width of 0.1675 for B2 against its Shapley value of 0.0817
  reads off `seq_width`. The sign-change count and the 57.5% agreement rate
  aggregate over the `block` column.

Regenerate with `python scripts/checks/sequential_vs_lattice.py`.

## Section 4.4, the residual diagnosis

| Claim | File |
|---|---|
| Three-step conclusion per cut | `shapley_2024/residual_verdict.parquet` |
| Step two rerun inside a single collection source | `evidence/f4_m4_within_source_domain.parquet` |
| Source is almost a function of time | `evidence/f4_m1_source_by_year.parquet`, `f4_m2_source_predictability.parquet`, `f4_m3_split_source_tv.parquet` |
| Step three, the counterexample refuting the conditional test | `evidence/item1_conditional_counterexample.parquet` |
| The conditional-structure statistic itself | `shapley_2024/conditional_structure.parquet` |
| Importance-weighted risk, the valid replacement | `evidence/iwrisk_family_panel.parquet` |
| Per-row weights behind it | `evidence/iwrisk_weights.parquet` |
| Null result, R against drift magnitude | `shapley_2024/residual_drift_correlation.parquet` with `drift_magnitude.parquet` |

The effective sample sizes of 3.1, 3.5 and 2.4% are columns `ess_clipped` and
`ess_fraction_clipped`. The 10% threshold is
`sift.iwrisk.DEFAULT_MIN_ESS_FRACTION`, not a number in the prose.

`item1_conditional_counterexample.parquet` carries its own generating parameters
as columns, `n_triggers`, `n_satellites`, `p_on`, `p_background`, `seed`,
`n_repeats`, because the first run did not record them.

## Section 4.5, validation and robustness

| Claim | File |
|---|---|
| Recovery Error and False Attribution Rate | `evidence/validation_metrics.parquet`, columns `rec_err_abs`, `v_full`, `phi_sum_injected` |
| The recovery panels behind them | `recovery/recovery.parquet`, filter `metric == "macro_f1"` |
| A2 with B2 reproducing on the double-injection panel | `recovery/dividends.parquet`, coalition `a2_labels+b2_axis` |
| Raw fits, 576 rows, 96 per panel | `recovery/fits.parquet` |
| The three A1 injection configurations | `recovery/a1_hypotheses.parquet`, columns `a1_align`, `a1_skew`, `phi`, `covers_zero` |
| Drift injection scale, R monotone over five levels | `driftsim/residual.parquet`, columns `residual`, `residual_lo`, `residual_hi` |
| Zero false positives on that scale | `driftsim/shapley.parquet` |
| Five controls, explained share 55.5% | `c1/decomp_c1.parquet`, extra column `phi_c1_dedup` |
| Protocol-side share moving to 50.2% | `c1/owen_c1.parquet` |
| Frozen basis: sign agreement 0.80 to 1.00, rank correlation 0.73 to 0.91, mean absolute difference 0.1042 for R | `shapley_2024/basis_compare_summary.parquet` |
| The 20 cells behind it | `shapley_2024/basis_compare_20cells.parquet`, suffix `_rep` reported basis, `_frz` frozen basis |
| The frozen-basis decomposition itself | `shapley_2024/decomposition_baseline_labels_macro_f1.parquet` |

`results/c1/metrics.parquet` holds 6,800 raw rows. `config_id` is 5 bits there,
and `config_id % 16` recovers the four-control index.

## Cross-checks

Neither is cited in the paper as a result. They exist so a reader can verify the
arithmetic without trusting the production path.

- `results/evidence/shapley_independent_check.parquet`. Every Shapley value
  recomputed from `results/metrics.parquet` by an independent implementation.
  Columns `phi_reference`, `phi_independent`, `abs_diff`, `efficiency_error`.
- `results/evidence/sequential_vs_lattice.parquet`, described under Section 4.3.

## Supporting evidence not cited in the main text

These back statements in [appendix-results.md](appendix-results.md).

| Finding | File | Script |
|---|---|---|
| Duplicate samples, overlap and how the gap moves | `item5_dedup_overlap.parquet`, `item5_dedup_gap_move.parquet` | `evidence/duplicate_overlap.py` |
| The `labels` argument of macro-F1 can hide a confound | `item3_labels_default.parquet` | `evidence/metric_label_set.py` |
| AUT can hide a difference in trajectory | `item4_aut.parquet`, `item4_aut_fits.parquet`, `item4_aut_summary.parquet` | `evidence/aut_trajectory.py` |
| Transfer between the two windows | `item6_transfer_cut2015.parquet`, column `comparison` | `evidence/window_transfer.py` |
| Prior deviation against per-class recall | `item7_a1_prior_recall_correlation.parquet`, column `pearson_r` | `evidence/prior_recall_correlation.py` |
| Collection source confounded with time | `f4_m1` to `f4_m4` | `evidence/source_confound.py` |

All files are under `results/evidence/`, all scripts under `scripts/`.

## Script names changed

The scripts were renamed when they moved into `scripts/`. Output filenames were
deliberately left alone, so `item1` through `item7` still appear in
`results/evidence/`.

| Was | Is now |
|---|---|
| `run_lattice_2024.py` | `scripts/run/lattice_2024.py` |
| `run_lattice_c1.py` | `scripts/run/lattice_c1.py` |
| `run_iwrisk_mlran.py` | `scripts/run/iwrisk_mlran.py` |
| `regen_shapley_2024.py` | `scripts/analysis/shapley_2024.py` |
| `regen_dataset_summary.py` | `scripts/analysis/dataset_summary.py` |
| `regen_validation_metrics.py` | `scripts/analysis/recovery_metrics.py` |
| `regen_item1_counterexample.py` | `scripts/evidence/conditional_counterexample.py` |
| `regen_item2_binary.py` | `scripts/evidence/binary_task_gap.py` |
| `regen_item3_labels.py` | `scripts/evidence/metric_label_set.py` |
| `regen_item4_aut.py` | `scripts/evidence/aut_trajectory.py` |
| `regen_item5_dedup.py` | `scripts/evidence/duplicate_overlap.py` |
| `regen_item6_transfer.py` | `scripts/evidence/window_transfer.py` |
| `regen_item7_a1_corr.py` | `scripts/evidence/prior_recall_correlation.py` |
| `f4_source_confound.py` | `scripts/evidence/source_confound.py` |
| `check_shapley_independent.py` | `scripts/checks/shapley_independent.py` |
| `check_sequential_vs_lattice.py` | `scripts/checks/sequential_vs_lattice.py` |
| no runner existed | `scripts/run/recovery.py` |
| no runner existed | `scripts/analysis/basis_compare.py` |
