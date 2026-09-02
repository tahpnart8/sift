"""Item 4: AUT on four disjoint two-year slots, on both time axes.

Control B2 selects the axis: off is the compile timestamp ``Year``, on is
``first_submission_date_year``. Nothing else is enabled, so the cells are the
baseline coalition (config 0) and the B2-only coalition (config 8).

``sift.metrics.assert_equal_disjoint_slots`` is called on the slot series before
any AUT is taken, which is what makes the trapezoid legitimate here and what
refuses the five overlapping three-year cuts used elsewhere in the paper.
"""

from __future__ import annotations

import os
import sys
import time

from sift.paths import RESULTS_DIR

os.environ.setdefault("SIFT_CACHE_DIR", str(RESULTS_DIR / "evidence" / "fit_cache"))

import numpy as np
import pandas as pd

from sift import (
    AUT_CUT_YEARS,
    AUT_TEST_WINDOW,
    MODEL_NAMES,
    PanelSpec,
    SplitSpec,
    assert_equal_disjoint_slots,
    aut,
    aut_label,
    build_panel,
    load_mlran,
)
from sift.config import ControlFlags, ExperimentConfig
from sift.experiment import panel_fingerprint, run_cell

OUT_FITS = RESULTS_DIR / "evidence" / "item4_aut_fits.parquet"
OUT_AUT = RESULTS_DIR / "evidence" / "item4_aut.parquet"
OUT_SUMMARY = RESULTS_DIR / "evidence" / "item4_aut_summary.parquet"

# ``random`` ignores the time axis entirely, so it gives one reference AUT for
# both axes. ``random_fully_matched`` matches the temporal window's class set
# and size and therefore moves with the axis, which is what a per-axis
# reference AUT needs.
DESIGNS = ("temporal", "random", "random_fully_matched")
AXES = {8: "first_submission_date_year", 0: "Year"}
SEEDS = (0, 1, 2)
METRIC = "macro_f1"


def main() -> None:
    seeds = SEEDS
    if len(sys.argv) > 1:
        seeds = tuple(int(s) for s in sys.argv[1].split(","))

    assert_equal_disjoint_slots(AUT_CUT_YEARS, AUT_TEST_WINDOW)

    raw = load_mlran()
    spec = PanelSpec(task="family")
    panel, _ = build_panel(raw, spec)
    digest = panel_fingerprint(panel)

    rows = []
    started = time.perf_counter()
    for config_id, axis in AXES.items():
        for design in DESIGNS:
            for cut in AUT_CUT_YEARS:
                for model in MODEL_NAMES:
                    for seed in seeds:
                        cfg = ExperimentConfig(
                            panel=spec,
                            split=SplitSpec(
                                design=design,
                                cut_year=cut,
                                test_window=AUT_TEST_WINDOW,
                            ),
                            flags=ControlFlags.from_index(config_id),
                            model_name=model,
                            seed=seed,
                            target=spec.target_column,
                            n_features=200,
                        )
                        rec = run_cell(panel, cfg, panel_digest=digest)
                        rows.append(
                            {
                                "item": "item4_aut",
                                "time_axis": axis,
                                "config_id": config_id,
                                "b2_axis": config_id == 8,
                                "design": design,
                                "cut": cut,
                                "slot": f"{cut}-{cut + AUT_TEST_WINDOW - 1}",
                                "model": model,
                                "seed": seed,
                                "n_train": int(rec["n_train"]),
                                "n_test": int(rec["n_test"]),
                                "k_test": int(rec["k_test"]),
                                "macro_f1": float(rec["macro_f1"]),
                                "accuracy": float(rec["accuracy"]),
                            }
                        )
                print(
                    f"{axis} {design} {cut} done "
                    f"{time.perf_counter() - started:.0f}s",
                    flush=True,
                )
    fits = pd.DataFrame(rows)
    fits.to_parquet(OUT_FITS, index=False)

    # AUT per (axis, design, model), then pooled over models.
    per = (
        fits.groupby(["time_axis", "design", "model", "cut"], as_index=False)[METRIC]
        .mean()
        .sort_values(["time_axis", "design", "model", "cut"])
    )
    out = []
    for (axis, design, model), g in per.groupby(["time_axis", "design", "model"]):
        assert_equal_disjoint_slots(tuple(g["cut"]), AUT_TEST_WINDOW)
        out.append(
            {
                "item": "item4_aut",
                "time_axis": axis,
                "design": design,
                "model": model,
                "metric": METRIC,
                "label": aut_label(METRIC, AUT_CUT_YEARS, AUT_TEST_WINDOW),
                "n_slots": len(g),
                "aut": aut(list(g[METRIC])),
                **{f"slot_{c}": float(v) for c, v in zip(g["cut"], g[METRIC])},
            }
        )
    pooled = (
        fits.groupby(["time_axis", "design", "cut"], as_index=False)[METRIC].mean()
    )
    for (axis, design), g in pooled.groupby(["time_axis", "design"]):
        g = g.sort_values("cut")
        assert_equal_disjoint_slots(tuple(g["cut"]), AUT_TEST_WINDOW)
        out.append(
            {
                "item": "item4_aut",
                "time_axis": axis,
                "design": design,
                "model": "all_four_pooled",
                "metric": METRIC,
                "label": aut_label(METRIC, AUT_CUT_YEARS, AUT_TEST_WINDOW),
                "n_slots": len(g),
                "aut": aut(list(g[METRIC])),
                **{f"slot_{c}": float(v) for c, v in zip(g["cut"], g[METRIC])},
            }
        )
    frame = pd.DataFrame(out)
    frame.to_parquet(OUT_AUT, index=False)

    # The two things the paper claims: the aggregate AUT distance between the
    # two axes, and the first-slot drop when the axis is corrected.
    summary = []
    for reference in ("random", "random_fully_matched"):
        for model in list(MODEL_NAMES) + ["all_four_pooled"]:
            sel = frame[frame.model == model]
            row = {"item": "item4_aut_summary", "reference_design": reference,
                   "model": model, "metric": METRIC}
            for axis in AXES.values():
                t = float(sel[(sel.time_axis == axis) & (sel.design == "temporal")]["aut"].iloc[0])
                r = float(sel[(sel.time_axis == axis) & (sel.design == reference)]["aut"].iloc[0])
                key = "true_axis" if axis.startswith("first") else "compile_axis"
                row[f"aut_temporal_{key}"] = t
                row[f"aut_reference_{key}"] = r
                row[f"distance_{key}"] = r - t
            row["aggregate_distance_difference"] = (
                row["distance_compile_axis"] - row["distance_true_axis"]
            )
            first = AUT_CUT_YEARS[0]
            row["first_slot"] = f"{first}-{first + AUT_TEST_WINDOW - 1}"
            row["first_slot_compile_axis"] = float(
                sel[(sel.time_axis == "Year") & (sel.design == "temporal")][f"slot_{first}"].iloc[0]
            )
            row["first_slot_true_axis"] = float(
                sel[(sel.time_axis == "first_submission_date_year") & (sel.design == "temporal")][f"slot_{first}"].iloc[0]
            )
            row["first_slot_drop_switching_to_true_axis"] = (
                row["first_slot_compile_axis"] - row["first_slot_true_axis"]
            )
            summary.append(row)
    summary = pd.DataFrame(summary)
    summary.to_parquet(OUT_SUMMARY, index=False)
    print(frame.to_string(index=False))
    print()
    print(summary[["reference_design", "model", "aut_temporal_true_axis",
                   "aut_reference_true_axis", "distance_true_axis",
                   "aut_temporal_compile_axis", "aut_reference_compile_axis",
                   "distance_compile_axis", "aggregate_distance_difference",
                   "first_slot_drop_switching_to_true_axis"]].to_string(index=False))
    print("wrote", OUT_FITS, fits.shape, OUT_AUT, frame.shape, OUT_SUMMARY, summary.shape)


if __name__ == "__main__":
    main()
