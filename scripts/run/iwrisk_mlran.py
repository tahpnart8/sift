"""Run the importance-weighted risk estimator on the MLRan family panel.

Nothing here is new analysis. The window construction is copied from notebook
04 cell 9 (``apply_controls`` under the B2 coalition, temporal split, logreg,
seed 0) and handed to :func:`sift.iwrisk.importance_weighted_risk` exactly as
its contract in ``.draft/iwrisk.md`` describes. The only choices made here are
which cuts to run (the three in scope) and where to write the output.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from sift.config import ControlFlags, ExperimentConfig, PanelSpec, SplitSpec
from sift.controls import apply_controls
from sift.data import build_panel, load_mlran
from sift.features import feature_columns
from sift.iwrisk import importance_weighted_risk
from sift.paths import RESULTS_DIR
from sift.seeding import derive_seed

warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

CUTS = (2015, 2019, 2021)
TARGET = "ransomware_family"
COALITION = ControlFlags(a1_prior=True, a2_labels=True, b1_fs=False, b2_axis=True)
N_BOOT = 1000
OUT = RESULTS_DIR / "evidence" / "iwrisk_family_panel.parquet"
OUT_WEIGHTS = RESULTS_DIR / "evidence" / "iwrisk_weights.parquet"


def main() -> None:
    panel, _ = build_panel(load_mlran(), PanelSpec())
    columns = feature_columns(panel)
    print(f"panel {panel.shape}  features {len(columns)}  "
          f"classes {panel[TARGET].nunique()}", flush=True)

    rows, weight_rows = [], []
    for cut in CUTS:
        cfg = ExperimentConfig(panel=PanelSpec(),
                               split=SplitSpec(design="temporal", cut_year=cut),
                               flags=COALITION, model_name="logreg", seed=0)
        controlled = apply_controls(panel, cfg)
        x_early = controlled.train[columns].to_numpy(dtype=np.float64)
        y_early = controlled.train[TARGET].to_numpy()
        x_late = controlled.test[columns].to_numpy(dtype=np.float64)
        y_late = controlled.test[TARGET].to_numpy()
        print(f"cut {cut}: early {len(y_early)} late {len(y_late)}  "
              f"classes early {controlled.train[TARGET].nunique()} "
              f"late {controlled.test[TARGET].nunique()}", flush=True)

        result = importance_weighted_risk(
            x_early, y_early, x_late, y_late,
            seed=derive_seed(0, "iwrisk", cut),
            n_boot=N_BOOT,
        )
        summary = result.summary()
        low_c, high_c = summary.pop("gap_ci_clipped")
        low_u, high_u = summary.pop("gap_ci_unclipped")
        rows.append({
            "cut": cut,
            "coalition_index": COALITION.to_index(),
            **summary,
            "gap_ci_low_clipped": low_c, "gap_ci_high_clipped": high_c,
            "gap_ci_width_clipped": result.clipped.gap_ci_width,
            "gap_ci_low_unclipped": low_u, "gap_ci_high_unclipped": high_u,
            "gap_ci_width_unclipped": result.unclipped.gap_ci_width,
            "direction_clipped": result.clipped.direction,
            "direction_unclipped": result.unclipped.direction,
            "reading_clipped": result.clipped.reading(result.max_decisive_width),
            "reading_unclipped": result.unclipped.reading(result.max_decisive_width),
            "risk_source_weighted_clipped": result.clipped.risk_source_weighted,
            "risk_target_clipped": result.clipped.risk_target,
            "risk_source_unweighted": result.risk_source_unweighted,
            "risk_target_unweighted_fit": result.risk_target_unweighted_fit,
            "ess_heldout_clipped": result.clipped.ess_heldout,
            "ess_fraction_heldout_clipped": result.clipped.ess_fraction_heldout,
            "ess_heldout_unclipped": result.unclipped.ess_heldout,
            "ess_fit_clipped": result.clipped.ess_fit,
            "n_fit": result.n_fit, "n_heldout": result.n_heldout,
            "n_clipped_rows": result.clipped.n_clipped,
            "clip_threshold": result.ratio_clipped.clip_threshold,
            "max_weight_clipped": result.clipped.max_weight,
            "mean_raw_weight": result.ratio_unclipped.mean_raw_weight,
            "n_late_rows_unseen_class": result.n_late_rows_unseen_class,
            "max_decisive_width": result.max_decisive_width,
            "n_boot": N_BOOT, "seed": derive_seed(0, "iwrisk", cut),
        })
        for name, ratio in (("unclipped", result.ratio_unclipped),
                            ("clipped", result.ratio_clipped)):
            weight_rows.append(pd.DataFrame({
                "cut": cut, "variant": name,
                "row": np.arange(ratio.raw_weights.size),
                "raw_weight": ratio.raw_weights,
                "weight": ratio.weights,
            }))

        print(f"  verdict {result.verdict()}  interpretable {result.interpretable}",
              flush=True)
        for reason in result.reasons:
            print(f"  REASON: {reason}", flush=True)
        for note in result.notes:
            print(f"  NOTE:   {note}", flush=True)

    frame = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUT, index=False)
    pd.concat(weight_rows, ignore_index=True).to_parquet(OUT_WEIGHTS, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 80):
        print(frame.drop(columns=["reasons", "notes"]).to_string(index=False))
    print("wrote", OUT, frame.shape)
    print("wrote", OUT_WEIGHTS)


if __name__ == "__main__":
    main()
