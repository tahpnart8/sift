# Reproducing the results

[Back to README](../README.md)

## Before anything

```bash
pip install -e ".[dev,notebook]"
pytest
```

The suite is 397 tests. Those needing raw MLRan skip when it is absent, so a
green run without the dataset is expected and still meaningful: the invariants
the paper depends on are asserted on synthetic panels.

Obtain MLRan and place it as [data.md](data.md) describes. Nothing under
"Full chain" runs without it. Everything under "Without refitting" runs from the
parquet files already in this repository.

## Hardware and timing

Everything was measured on one CPU machine, no GPU at any point. Wall-clock
times below are what we observed; they scale with core count.

| Stage | Scale | Time |
|---|---|---|
| Main lattice | 3,400 fits | 1.60 h |
| Five-control lattice | 6,800 fits | 3.09 h |
| Recovery check | 576 fits | 25 min |
| Drift injection scale | 480 fits | 81 s |
| Everything else | no refitting | minutes |

Fitted models are cached under `**/fit_cache/`, keyed by configuration. The
cache is regenerated on demand and is deliberately not in this repository: it is
about 15 MB of pickles that reproduce from the code. `SIFT_CACHE_DIR` moves it.

## Full chain

Run from the repository root, in this order. Steps 1 and 2 are independent of
each other; everything after step 2 depends on step 1.

| # | Command | Writes |
|---|---|---|
| 1 | `python scripts/run/lattice_2024.py` | `results/metrics.parquet`, `results/predictions.parquet`, `results/run_manifest.json` |
| 2 | `python scripts/run/lattice_c1.py` | `results/c1/` |
| 3 | `python scripts/analysis/shapley_2024.py` | `results/shapley_2024/`, 21 files |
| 4 | `python scripts/analysis/basis_compare.py` | `results/shapley_2024/basis_compare_*.parquet` |
| 5 | `python scripts/analysis/dataset_summary.py` | `results/panel.parquet`, `results/evidence/dataset_summary.parquet` |
| 6 | `python scripts/run/recovery.py` | `results/recovery/` |
| 7 | `python scripts/analysis/recovery_metrics.py` | `results/evidence/validation_metrics.parquet` |
| 8 | `python scripts/run/iwrisk_mlran.py` | `results/evidence/iwrisk_*.parquet` |
| 9 | `python -m sift.driftsim` | `results/driftsim/` |
| 10 | `python scripts/evidence/*.py` | `results/evidence/`, one finding per script |
| 11 | `python scripts/checks/*.py` | `results/evidence/`, cross-checks |

`make all` runs exactly this order.

Step 3 splits into four independent stages, each writing its own parquet so a
failure late does not lose the work already done:

```bash
python scripts/analysis/shapley_2024.py decomp
python scripts/analysis/shapley_2024.py bootstrap
python scripts/analysis/shapley_2024.py drift
python scripts/analysis/shapley_2024.py baseline
```

With no argument it runs all four. `bootstrap` recomputes confidence intervals
from the stored prediction table and refits nothing. `drift` needs
`residual_ci.parquet`, which `bootstrap` produces.

Step 4 must follow the `baseline` stage of step 3: it compares the two scoring
bases and needs both decomposition tables.

## Without refitting

These derive every reported table from artefacts already in the repository. This
is the useful path for checking the paper's arithmetic. `make analysis` runs
them.

```bash
python scripts/analysis/basis_compare.py       # the two scoring bases, 20 cells
python scripts/analysis/recovery_metrics.py    # Recovery Error, False Attribution Rate
python scripts/checks/shapley_independent.py   # Shapley recomputed from scratch
python scripts/checks/sequential_vs_lattice.py # all 24 switch-off orders
```

`shapley_independent.py` recomputes every Shapley value directly from
`results/metrics.parquet` through an independent implementation and compares it
against the reported decomposition. `sequential_vs_lattice.py` measures how far
sequential ablation would have landed from the full-lattice answer, under all 24
orders. Both write into `results/evidence/`.

## Notebooks

Three notebooks narrate the same package and need the `notebook` extra.

- `00_data_preparation.ipynb` builds the analysis panel from raw MLRan and
  prints the provenance chain and content fingerprints.
- `03_shapley_attribution.ipynb` runs the decomposition over the 16-cell lattice.
- `04_residual_diagnosis.ipynb` runs the residual diagnosis block.

They are narrative, not the source of any number in the paper. The scripts above
are.

## If a number does not match

Check in this order.

1. The library versions. `pyproject.toml` pins them exactly for this reason.
2. Whether you are pooling the seed dimension. Five seeds of one (cut, model)
   cell are five draws of one quantity and are averaged before anything else.
   Treating them as five independent groups inflates every agreement statistic.
3. Whether you are pooling across cut-points. The five cuts are five independent
   experiments, never five points on a curve.
4. [results-map.md](results-map.md), which names the exact file and column for
   every number, and flags the places where two files hold the same quantity by
   different computation paths.
