"""Re-run the full 16-coalition lattice on the 2024-extended panel.

The panel changed from 1377 samples / 32 families to 1425 / 34 when the
analysis window was extended to the end of the MLRan collection, so every
stored metric predates the current PanelSpec and must be recomputed.
"""

from __future__ import annotations

import time

from sift.config import ControlFlags, ExperimentConfig, PanelSpec, SplitSpec
from sift.data import build_panel, load_mlran
from sift.experiment import run_lattice
from sift.paths import RESULTS_DIR

MODELS = ("logreg", "random_forest", "lightgbm", "mlp")
SEEDS = (0, 1, 2, 3, 4)
CUTS = (2015, 2017, 2019, 2021, 2023)


def main() -> None:
    spec = PanelSpec()
    panel, provenance = build_panel(load_mlran(), spec)
    print(f"panel: {len(panel)} rows, window {spec.year_min}-{spec.year_max}", flush=True)
    for step, count in provenance.items() if hasattr(provenance, "items") else []:
        print(f"  {step}: {count}", flush=True)

    # Template only; run_lattice overrides split, flags, model and seed per cell.
    base = ExperimentConfig(
        panel=spec,
        split=SplitSpec(design="temporal", cut_year=CUTS[0]),
        flags=ControlFlags(),
        model_name=MODELS[0],
        seed=SEEDS[0],
    )

    started = time.perf_counter()
    metrics = run_lattice(
        panel,
        base=base,
        model_names=MODELS,
        seeds=SEEDS,
        cut_years=CUTS,
        results_dir=RESULTS_DIR,
    )
    elapsed = time.perf_counter() - started
    print(f"done: {len(metrics)} rows in {elapsed / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
