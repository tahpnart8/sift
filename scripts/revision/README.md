# Revision analyses (camera-ready, October 2026)

Analyses added in response to the reviewers. None retrains a model: each reads the stored
lattice and prediction tables in `results/`, or runs on synthetic data. Run from the repository
root with `PYTHONPATH=.`; outputs go to `results/revision/`.

| Script | Reviewer point | Output |
|---|---|---|
| `r12_metric_robustness.py` | R1.2, decomposition under four stored metrics | `r12_metric_robustness.csv` |
| `r13_taxonomy_sensitivity.py` | R1.3, eleven control taxonomies (leave-one-out on the 4- and 5-control frames) | `r13_taxonomy_sensitivity.csv` |
| `r14_bootstrap_dependence.py` | R1.4, Holm correction over the 20 cells and a paired bootstrap across models | `r14_per_cell_holm.csv`, `r14_paired_bootstrap.csv` |
| `r15b_iwrisk_clean.py` | R1.5, importance-weighted risk under pure covariate shift and injected concept drift | `r15b_iwrisk_clean.csv` |
| `r15_iwrisk_validation.py` | R1.5, the same test in a design matching the real feature count, and under shrinking support overlap | `r15_iwrisk_validation.csv` |

`r14` first reproduces `results/shapley_2024/residual_ci.parquet` exactly before applying the new
procedures. In `r15`, the case with disjoint classes between windows passes the effective-sample-size
guard and still implicates the conditional; the paper reports this as a limitation of the guard.
