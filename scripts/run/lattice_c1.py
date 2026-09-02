"""Run the 32-coalition lattice that adds deduplication as a fifth control.

Half of the 6800 fits are identical to the four-control run and come from the
cache, so only the C1-off half is computed. Output goes to ``results/c1/`` and
never touches the four-control artefacts the paper's main tables read.
"""

from __future__ import annotations

import time

from sift.config import ControlFlags, ExperimentConfig, PanelSpec, SplitSpec
from sift.data import build_panel_variants, load_mlran
from sift.experiment import run_lattice_c1
from sift.paths import RESULTS_DIR

MODELS = ("logreg", "random_forest", "lightgbm", "mlp")
SEEDS = (0, 1, 2, 3, 4)
CUTS = (2015, 2017, 2019, 2021, 2023)


def main() -> None:
    spec = PanelSpec()
    panels = build_panel_variants(load_mlran(), spec)
    print("panel variants built", flush=True)

    base = ExperimentConfig(
        panel=spec,
        split=SplitSpec(design="temporal", cut_year=CUTS[0]),
        flags=ControlFlags(),
        model_name=MODELS[0],
        seed=SEEDS[0],
    )

    started = time.perf_counter()
    metrics = run_lattice_c1(
        panels,
        base=base,
        model_names=MODELS,
        seeds=SEEDS,
        cut_years=CUTS,
        results_dir=RESULTS_DIR / "c1",
    )
    print(f"done: {len(metrics)} rows in {(time.perf_counter() - started) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
