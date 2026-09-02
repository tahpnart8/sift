# Appendix: results the page limit pushed out of the paper

[Back to README](../README.md)

The paper was compressed to 12 pages for the conference, so several secondary
results and most detailed tables live here instead. Every number in this document
is read from parquet under `results/`, never from the paper's prose.
[results-map.md](results-map.md) names the file behind each one.

## A.1. Split accounting per cut, on both time axes

This is the material behind control B2. Regenerate with `sift.data.build_panel`
under the default `PanelSpec()`, then `sift.splits.temporal_split` with each of
the two `time_column` values.

The correct axis, `first_submission_date_year`, the date a sample was first
submitted:

| Cut | Train samples | Earliest year | Train classes | Test samples | Test classes |
|---|---|---|---|---|---|
| 2015 | 225 | 2012 | 8 | 294 | 17 |
| 2017 | 475 | 2012 | 15 | 184 | 17 |
| 2019 | 595 | 2012 | 20 | 649 | 24 |
| 2021 | 785 | 2012 | 29 | 630 | 23 |
| 2023 | 1,342 | 2012 | 34 | 83 | 10 |

The compile timestamp axis, `Year`. This is also the baseline configuration,
since B2 is off there:

| Cut | Train samples | Earliest year | Train classes | Test samples | Test classes |
|---|---|---|---|---|---|
| 2015 | 640 | 1970 | 22 | 290 | 19 |
| 2017 | 866 | 1970 | 26 | 247 | 17 |
| 2019 | 1,033 | 1970 | 29 | 296 | 21 |
| 2021 | 1,281 | 1970 | 33 | 143 | 14 |
| 2023 | 1,394 | 1970 | 34 | 31 | 6 |

Reading the two side by side is the most direct way to see what B2 does. At cut
2015 the correct axis gives 225 training samples starting in 2012 across 8
classes; the compile axis gives 640 starting in 1970 across 22, nearly three
times the samples. The difference comes from 856 samples carrying the year 1992
and 39 carrying 1970 in the raw frame, that is samples actually submitted years
later but stamped with a date the malware author controls. The cut 2015 training
set on the wrong axis therefore already contains samples from its own future.

The baseline test window series, 290, 247, 296, 143 and 31 samples, is what
Section 3.5 of the paper refers to when it says cut 2023 is down to 31 samples.

## A.2. The decomposition per model, all 20 cells

The paper reports the number pooled over the four models plus one sentence saying
all four give the same source ordering. The full numbers are in
`results/shapley_2024/model_summary_macro_f1.parquet`, 20 rows, each carrying the
half-width of its seed interval in the `_half` columns and a `_covers_zero` flag.

Pooled by model, macro-F1:

| Model | Delta(∅) | φ_A1 | φ_A2 | φ_B1 | φ_B2 | R |
|---|---|---|---|---|---|---|
| logreg | 0.4593 | 0.0216 | 0.1144 | −0.0121 | 0.0797 | 0.2557 |
| random_forest | 0.4763 | 0.0295 | 0.1198 | −0.0038 | 0.0846 | 0.2461 |
| lightgbm | 0.4324 | 0.0249 | 0.1359 | −0.0167 | 0.0896 | 0.1988 |
| mlp | 0.4322 | 0.0242 | 0.1209 | −0.0092 | 0.0726 | 0.2236 |

The ordering A2 above B2 above A1 above B1 holds in all four models, as the paper
states. The residual is less uniform: it runs from 0.199 with LightGBM to 0.256
with Logistic Regression, so the spread across models in the residual is wider
than the spread in any single contribution other than A2. The paper does not
mention this.

The accuracy versions are `model_summary_accuracy.parquet` and
`decomposition_as_reported_accuracy.parquet`.

## A.3. All 16 lattice cells for one representative cell

This is the primary evidence, independent of any aggregation assumption: the
performance of both arms, Delta(S) and v(S), for each of the 16 coalitions. Read
it from `results/metrics.parquet`, filtering on `cut`, `model` and `config_id`
from 0 to 15, with `design` in `temporal` and `random_fully_matched`.

Anyone who wants to check the decomposition without trusting the Shapley step
should start here. Delta(∅) is the row with `config_id` 0, and R is the row with
`config_id` 15.

## A.4. The residual with its interval, per cell

`results/shapley_2024/residual_ci.parquet`, 20 rows. The paper collapses this to
five rows by cut plus one sentence saying the interval excludes zero in 17 of 20
cells. The full table shows that the three cells which do not exclude zero all
belong to cut 2017, and the `passes_gate` column shows that the pre-registered
precision gate, half-width below 0.10, is met in only 3 of 20 cells.

## A.5. The three A1 injection configurations, and why A1 failed at first

The paper keeps the conclusion, that total variation is the wrong summary of
prior shift when the metric is class-averaged, but drops the three-row table.
Source: `results/recovery/a1_hypotheses.parquet`, filtered to `control ==
"a1_prior"` and `metric == "macro_f1"`.

| Configuration | TV between arms | φ_A1 | Excludes zero |
|---|---|---|---|
| Deviation orthogonal to difficulty, original `a1_only` panel, `a1_align == "name"` | 0.2440 | +0.0061 | no |
| Deviation tracking difficulty, ×3, `h2_size_aligned_x3` | 0.2596 | +0.0107 | yes |
| Deviation tracking difficulty, ×12 | 0.4696 | +0.0433 | yes |

The mechanism. The weights of control A1 depend only on the true class, so
per-class recall is exactly invariant under A1. The only quantity A1 can move is
the class-wise sum of prior deviation times that class's recall, which is a
covariance between deviation and difficulty, not a distance. On the first panel
the Pearson correlation between the prior deviation ratio and per-class recall is
only +0.0092 over the 34 classes, recorded in
`results/evidence/item7_a1_prior_recall_correlation.parquet` on the row with
`deviation_definition == "window_share_over_corpus_share"`. The deviation is
nearly orthogonal to difficulty, so A1 has nothing to pick up even at a total
variation of 0.244.

One false positive to record on `h2_size_aligned_x3`: φ_B1, an uninjected
control, came out at −0.0247 with an interval excluding zero.

## A.6. Metric choice can hide a confound

Source: `results/shapley_2024/metric_comparison.parquet`.

| Quantity | accuracy | macro-F1 | Ratio |
|---|---|---|---|
| Delta(∅) | 0.4821 | 0.4500 | 1.07 |
| φ_A1 | 0.0779 | 0.0269 | 2.90 |
| φ_A2 | 0.1522 | 0.1227 | 1.24 |
| φ_B1 | 0.0220 | 0.0216 | 1.02 |
| φ_B2 | 0.1126 | 0.0863 | 1.30 |
| Residual | 0.1600 | 0.2310 | 0.69 |

Choosing macro-F1 as the primary metric shrinks φ_A1 to barely a third of what
accuracy gives, while the other three contributions barely move. The reason is
mechanical rather than statistical. A1 can only act through the covariance
between prior deviation and per-class difficulty, and macro-F1 already balances
the classes, which closes most of that path. The consequence is that a study
reporting only macro-F1 would conclude prior shift is negligible, while the same
data under accuracy gives it nearly three times as much. The residual moves the
other way: under accuracy it is markedly smaller, 0.160 against 0.231.

Read with the caveat from Section 4.5 of the paper: this 2.90 advantage does not
reproduce on the validation panels, where accuracy gives intervals 1.74 to 2.55
times wider and fails to exclude zero at all three injection levels. Treat the
table as an observation about metric sensitivity, not as a validated measurement.

## A.7. AUT can hide a difference in trajectory

Source: `results/evidence/item4_aut.parquet` and `item4_aut_summary.parquet`,
generated by `python scripts/evidence/aut_trajectory.py`.

AUT is not computable over the paper's five overlapping three-year cuts, because
the trapezoidal average requires disjoint slots of equal width;
`sift.metrics.assert_equal_disjoint_slots` rejects that series. This finding
therefore uses four separate disjoint two-year windows, 2016-2017, 2018-2019,
2020-2021 and 2022-2023, run on both of the time axes control B2 chooses between.

The result: the AUT distance between the temporal arm and the reference arm
barely distinguishes the two axes. Pooled over the four models with the `random`
reference, the distance is 0.5070 on the correct axis and 0.4799 on the compile
axis, a difference of −0.0272. On AUT alone the conclusion would be that the
choice of axis does not matter.

Per slot the picture is different. In the first slot, 2016-2017, temporal
performance pooled over four models is 0.3573 on the compile axis and 0.2063 on
the correct axis, a drop of 0.1510 on switching to the correct axis. For Logistic
Regression alone it is 0.3722 down to 0.2054. Later slots compensate, so the
difference disappears from the AUT figure.

The lesson: AUT is a summary statistic and it summarises away exactly what a
drift analysis is interested in, namely where along the time axis the model
fails. This is also why SIFT does not use AUT as its value function.

## A.8. Duplicate samples, full numbers

The paper quotes a range, 0.4 to 28.7 percent on the family panel. Per cut, from
`results/evidence/item5_dedup_overlap.parquet` with `panel_variant ==
"without_dedup"` and `task == "family"`:

| Cut | `first_submission_date_year` axis | `Year` axis |
|---|---|---|
| 2015 | 0.99% | 0.98% |
| 2017 | 0.40% | 0.00% |
| 2019 | 1.22% | 1.02% |
| 2021 | 4.61% | 5.75% |
| 2023 | 28.72% | 20.55% |

The same file keeps the binary panel, `task == "binary"`, the ransomware against
goodware task the paper dropped in Section 3.5. Those numbers are markedly
higher:

| Cut | `first_submission_date_year` axis | `Year` axis |
|---|---|---|
| 2015 | 9.32% | 9.15% |
| 2017 | 13.51% | 10.24% |
| 2019 | 7.74% | 10.86% |
| 2021 | 11.84% | 14.38% |
| 2023 | 36.71% | 31.44% |

The binary panel range is 7.74 to 36.71 percent against 0.40 to 28.72 for the
family panel. Do not mix the two: the binary panel has more samples and a
different class grouping, so its overlap rates are not directly comparable, and
every decomposition result in the paper runs on the family panel.

Under `panel_variant == "with_dedup"` the rate is zero at every cut on both
panels, as the deduplication step defines. The effect on the reported gap is in
`results/evidence/item5_dedup_gap_move.parquet`, column `gap_move`, read directly
off the five-control lattice in `results/c1/metrics.parquet`: 0.0694 pooled, and
per cut 0.0721 / 0.0264 / 0.1095 / 0.1168 / 0.0222.

In the code and the result files this control is identified as `c1_dedup`, not
`b3`, for a historical reason: it sat inside the fixed preprocessing chain before
being pulled out as a fifth control. The paper calls it B3 to match the two-tier
taxonomy. The backward-compatibility constraint is recorded in
`sift/CONTRACT.md`: `ControlFlags`, `CONTROL_NAMES`, `LATTICE_CELLS` and the
four-flag `config_id` encoding in `results/metrics.parquet` do not change, the
four-control lattice remains the default, and the five-control lattice runs only
when `ExtendedControlFlags`, `build_panel_variants`, `build_jobs_c1` and
`run_lattice_c1` are called by name.

## A.9. The cache, two cache-key bugs, and the two-run comparison

The cache key is a content hash over the panel fingerprint, the configuration and
the source hash of every module on the training path; see `sift/cache.py`. Two
cache-key bugs were found during internal audit. The first left `PanelSpec` out
of the key, so two different panels could share an entry. The second memoised the
panel fingerprint so that it went stale, and a changed panel still returned the
old fingerprint. Every result produced before those two were closed was discarded
and recomputed; none was reused. `tests/test_cache_key.py` pins these invariants.

The analysis frame also changed between two runs, from 2012-2023 to 2012-2024.
The reason recorded in the old code was too few samples outside the frame; that
holds for the lower bound but not the upper, since 2024 has 263 raw samples, more
than 2018 or 2019. The panel went from 1,377 samples and 32 classes to 1,425 and
34; the two families crossing the threshold are `8base` and `nemty`.

| Quantity | 2023 panel | 2024 panel |
|---|---|---|
| Delta(∅) | 0.4557 | 0.4500 |
| φ_A2 | 0.1279 | 0.1227 |
| φ_B2 | 0.0753 | 0.0817 |
| φ_A1 | 0.0288 | 0.0250 |
| φ_B1 | −0.0136 | −0.0104 |
| v(N) over Delta | 47.9% | 48.7% |
| R | 0.2372 | 0.2310 |
| A2 with B2 dividend | 0.1440 | 0.1425 |

The source ordering is unchanged. The automated comparison is
`results/shapley_2024/comparison_old_vs_new.parquet`. The old run's own output
directory is not part of this repository, and that comparison file has no runner;
see [limitations.md](limitations.md).

## A.10. The seed dimension

The seed governs exactly two things: the draws in the three random designs, and
the stochastic part of the learner, meaning per-node feature sampling in Random
Forest and weight initialisation in the MLP. The temporal arm splits by year and
is therefore deterministic, so the seed there measures learner noise only. On the
stored results the standard deviation across the five seeds in the temporal arm
is exactly zero for Logistic Regression and for LightGBM, and averages 0.021 for
the MLP and 0.019 for Random Forest. All five seeds are still run for every
design, so that a single temporal point is never compared against a random point
already averaged five times.

The consequence for intervals: for the two deterministic models a seed-based
interval has zero width and is meaningless, so their intervals come from a
bootstrap over the prediction table. This is also why the scoring-basis
comparison must aggregate to 20 cells rather than to 100 groups including the
seed dimension: for Logistic Regression and LightGBM the five seeds are five
copies of the same point, and counting them separately inflates every agreement
statistic. `scripts/analysis/basis_compare.py` does the aggregation correctly.

## A.11. A preprocessing detail that can shift the panel

The deduplication step keeps the sample with the earliest timestamp within each
group of identical feature vectors. Two groups tie on timestamp, so the sort must
be stable and must carry a secondary key. The library default sort produces a
different panel. Anyone rerunning and not getting exactly 1,425 samples should
check here first.

## A.12. A scoring basis that was rejected, and why

The robustness check in Section 4.5 of the paper freezes the class set a priori,
taking it from the family support of the baseline window. One alternative was
considered and rejected: taking the class set as the intersection of the windows
across all 16 coalitions. That is wrong because the intersection sits entirely
inside the window with A2 already on, meaning the A2 intervention has been
applied to every cell in advance, and A2 then has nothing left to contribute. It
also retains only 52 of 100 groups, the rest falling below the shared sample
threshold. The rejected basis is
`results/shapley_2024/decomposition_common_macro_f1.parquet`. The lesson, kept in
the paper: a robustness check is only worth anything when it is defined
independently of the intervention it is checking.

## A.13. Three repetition dimensions and the aggregation rule

This table was in the paper before compression; the current version carries the
rule in prose only.

| Dimension | Levels | Nature | Handling |
|---|---|---|---|
| seed | 5 | pure noise | averaged, dispersion carried into the error |
| model | 4 | robustness check | not pooled, except for RQ1, and then always with the range |
| cut-point | 5 | a substantive variable, the time axis | never pooled |

The three dimensions give 100 combinations, but the base unit of analysis is a
cell, meaning (cut, model), 20 in total, because the seed is always averaged
first. Both levels are directly checkable:
`results/shapley_2024/decomposition_as_reported_macro_f1.parquet` has exactly 100
rows with `cut`, `model` and `seed`, while `results/decomp_2024.parquet` has
exactly 20 rows and no `seed` column. Why the seed must be averaged before
counting anything is in A.10.

The step two and step three shift tests run per cut rather than per cell, so
`results/shapley_2024/domain_classifier.parquet` and `conditional_structure.parquet`
have 5 rows each while `residual_ci.parquet` has 20. Do not compare counts across
the two.

## A.14. Columns cut from Tables 3, 5 and from the dividend discussion

**The Remainder column of Table 3.** The paper drops it with a note that it
equals the Delta column by construction. Read back from
`results/metrics.parquet`, filtering to the baseline configuration, averaging 5
seeds then 4 models, macro-F1:

| Cut | Reported gap | Sample-size gap | Class composition | Remainder | Sum of three |
|---|---|---|---|---|---|
| 2015 | 0.5511 | 0.0381 | 0.0621 | 0.4509 | 0.5511 |
| 2017 | 0.5305 | 0.0334 | −0.0275 | 0.5247 | 0.5305 |
| 2019 | 0.5540 | 0.0134 | 0.0150 | 0.5256 | 0.5540 |
| 2021 | 0.4743 | −0.0115 | 0.0646 | 0.4211 | 0.4743 |
| 2023 | 0.3177 | 0.0219 | −0.0321 | 0.3279 | 0.3177 |

The Remainder column matches Table 3's Delta column to four digits at all five
cuts, and the sum of the three components matches the reported gap to 0.0. This
is a self-check of the split rather than a new result, which is why the paper
drops the column.

**The small dividend terms omitted from Section 4.3.** Source:
`results/shapley_2024/dividends_macro_f1.parquet`, 100 rows, aggregated to 20
cells then averaged:

| Term | Value |
|---|---|
| B1 × B2 | +0.0092 |
| the four-way, A1 × A2 × B1 × B2 | +0.0074 |
| the remaining nine terms | at most 0.0063 in absolute value |

The largest of the remaining nine is A2 × B1 × B2 at −0.0063. The bound quoted in
the paper is always the largest absolute value among the terms it does not list,
which is why it moves when a term is added to or removed from the listed set.

**The domain classifier p column and the k out of 500 column of Table 5.**
Source: `results/shapley_2024/domain_classifier.parquet` column `p_value`, and
`conditional_structure.parquet` column `n_within_at_or_below_across` over
`n_within_draws` of 500:

| Cut | Domain classifier | p | Held-out set | k of 500 | empirical p |
|---|---|---|---|---|---|
| 2015 | 0.800 | 6.0e-14 | 150 | 0 | 0.0020 |
| 2017 | 0.906 | 9.0e-12 | 64 | 3 | 0.0080 |
| 2019 | 0.805 | 1.2e-19 | 210 | 0 | 0.0020 |
| 2021 | 0.827 | 2.0e-36 | 346 | 0 | 0.0020 |
| 2023 | 0.762 | 1.6e-06 | 84 | 0 | 0.0020 |

Cuts 2017 and 2023 carry `in_scope == False` in both files, because the held-out
set is below the pre-registered threshold of 100, which is why the paper leaves
those two cells blank. The numbers are still computed and recorded, and are
printed here for completeness, but must not be read as shift magnitudes.

## A.15. Importance-weighted risk, full numbers

The paper keeps these numbers in prose in Section 4.4 but drops the table.
Source: `results/evidence/iwrisk_family_panel.parquet`, 3 rows, one per cut that
could run. Metric is macro-F1.

| Cut | Raw gap | After weighting and clipping | 95% CI | Effective sample size | Fraction of early window | Domain AUC | Verdict |
|---|---|---|---|---|---|---|---|
| 2015 | +0.3343 | −0.0609 | −0.0989 to +0.0722 | 6.88 | 3.06% | 0.8756 | inconclusive |
| 2019 | +0.4056 | +0.1472 | +0.0421 to +0.2163 | 20.63 | 3.47% | 0.9380 | inconclusive |
| 2021 | +0.4323 | +0.0232 | −0.0625 to +0.1147 | 18.93 | 2.41% | 0.9061 | inconclusive |

The corresponding columns are `raw_gap`, `conditional_gap_clipped`,
`gap_ci_low_clipped` with `gap_ci_high_clipped`, `ess_clipped`,
`ess_fraction_clipped`, `domain_auc` and `verdict`. All three rows carry
`interpretable == False`, and the intervals come from 1,000 bootstrap resamples,
column `n_boot`.

Three supporting numbers to read alongside. The two window sizes, columns
`n_early` and `n_late`, are 225 with 150, 595 with 210, and 785 with 346. On the
held-out half of the early window the effective sample size is lower still:
column `ess_heldout_clipped` gives 2.25 / 9.05 / 9.14 against `n_heldout` of
113 / 298 / 393, and this is the basis for the paper's statement that the
interval at cut 2019 rests on about 9 effective rows. The accuracy of the domain
classifier used to estimate the density ratio, column `domain_accuracy`, is
0.8187 / 0.8745 / 0.8382; this differs from the domain classifier column of Table
5 because it runs on the separate split inside `sift.iwrisk` and is not the same
scope, so the two are not directly comparable.

One place where scope matters. The paper's statement that at every cut a single
row carries more than a quarter of the total weight is true of the unclipped
weights: read `results/evidence/iwrisk_weights.parquet` with `variant ==
"unclipped"` and the largest share is 0.9862 / 0.6282 / 0.9716. Under `variant ==
"clipped"`, which is what actually produces the table above, the largest share
falls to 0.2076 / 0.0761 / 0.0729. These are two different quantities, and
clipping is exactly what pulls it down.

## A.16. Cells containing zero, split by control

The paper gives two pooled counts, 25 of 80 and 58 of 80, and moves the
per-control split here. Source:
`results/shapley_2024/phi_ci_coverage_counts.parquet`, column `n_covering_zero`
over `n_cells`, `metric == "macro_f1"`.

| Control | Seed basis, `interval == "seed_t"` | Bootstrap basis, `interval == "prediction_bootstrap"` |
|---|---|---|
| A1 | 2 of 20 | 20 of 20 |
| A2 | 4 of 20 | 6 of 20 |
| B1 | 12 of 20 | 20 of 20 |
| B2 | 7 of 20 | 12 of 20 |
| **Total** | **25 of 80** | **58 of 80** |

The unit here is the cell, 20 per control, so the total of 80 is four controls
times 20 cells, not 80 groups including the seed dimension. Per-cell intervals
are in `phi_ci_seed.parquet`, 160 rows because it keeps both metrics, and
`phi_ci_bootstrap.parquet`, 80 rows for macro-F1 only.

This is where the two bases disagree most sharply. A1 almost always separates
from zero on the seed basis but contains zero in all 20 cells on the bootstrap;
B1 contains zero in more than half the cells on the seed basis and in all 20 on
the bootstrap. Only A2 survives both. The bootstrap basis is more conservative
because it accounts for the sampling noise of the test window as well, whereas
the seed basis captures only learner noise, and learner noise is exactly zero for
two of the four models; see A.10.

## A.17. Design documents

`sift/CONTRACT.md` is the contract that fixes function signatures, column names
and constants across the package. It is the document to read before changing
anything in `sift/`, and `tests/` asserts most of what it states.

The internal design notes that motivated the fifth control and the
importance-weighted risk check are not part of this repository. What survives of
them is in `sift/CONTRACT.md` and in the module docstrings of `sift/iwrisk.py`
and `sift/recovery.py`. `scripts/run/iwrisk_mlran.py` calls
`sift.iwrisk.importance_weighted_risk` directly and adds no analysis choice of
its own beyond selecting the three cuts and the output location.
