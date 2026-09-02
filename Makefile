.PHONY: all test analysis evidence checks lattice recovery clean-cache

PYTHON ?= python

## all: the full chain, in dependency order. Hours; needs MLRan.
all: lattice recovery analysis evidence checks

## test: the test suite. Runs without MLRan; those tests skip.
test:
	pytest

## lattice: the two training runs. About 4.7 hours.
lattice:
	$(PYTHON) scripts/run/lattice_2024.py
	$(PYTHON) scripts/run/lattice_c1.py

## recovery: semi-synthetic recovery panels. About 25 minutes.
recovery:
	$(PYTHON) scripts/run/recovery.py

## analysis: re-derive every reported table. No refitting.
analysis:
	$(PYTHON) scripts/analysis/shapley_2024.py
	$(PYTHON) scripts/analysis/basis_compare.py
	$(PYTHON) scripts/analysis/dataset_summary.py
	$(PYTHON) scripts/analysis/recovery_metrics.py

## evidence: supporting findings, one script per finding.
evidence:
	$(PYTHON) scripts/run/iwrisk_mlran.py
	$(PYTHON) -m sift.driftsim
	$(PYTHON) scripts/evidence/conditional_counterexample.py
	$(PYTHON) scripts/evidence/binary_task_gap.py
	$(PYTHON) scripts/evidence/metric_label_set.py
	$(PYTHON) scripts/evidence/aut_trajectory.py
	$(PYTHON) scripts/evidence/duplicate_overlap.py
	$(PYTHON) scripts/evidence/window_transfer.py
	$(PYTHON) scripts/evidence/prior_recall_correlation.py
	$(PYTHON) scripts/evidence/source_confound.py

## checks: recompute the reported numbers by an independent route.
checks:
	$(PYTHON) scripts/checks/shapley_independent.py
	$(PYTHON) scripts/checks/sequential_vs_lattice.py

## clean-cache: drop the refittable model cache.
clean-cache:
	$(PYTHON) -c "import shutil,pathlib; [shutil.rmtree(p) for p in pathlib.Path('results').rglob('fit_cache')]"
