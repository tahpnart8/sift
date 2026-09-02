"""Compare the two scoring bases at cell level.

The lattice is scored twice. The reported basis re-derives the label set inside
each configuration; the frozen basis pins the label set to the baseline
configuration. If the choice of basis moved the attribution, the decomposition
would be an artefact of scoring rather than of the controls.

Both inputs carry a seed dimension. Section 3.6 of the paper fixes the
aggregation rule: the five seeds of one (cut, model) cell are five draws of the
same quantity, so they are averaged away before anything is compared. Comparing
over (cut, model, seed) instead would count each cell five times and inflate
every agreement statistic. That is why this runs as its own script rather than
inside the ``baseline`` stage that produces the frozen-basis decomposition.

Reads two decomposition tables, writes one row per cell, twenty rows.
No refitting; both inputs already exist after ``shapley_2024.py``.

    python scripts/analysis/basis_compare.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sift.paths import RESULTS_DIR

OUT_DIR = RESULTS_DIR / "shapley_2024"
AS_REPORTED = OUT_DIR / "decomposition_as_reported_macro_f1.parquet"
FROZEN = OUT_DIR / "decomposition_baseline_labels_macro_f1.parquet"
OUT = OUT_DIR / "basis_compare_20cells.parquet"
SUMMARY = OUT_DIR / "basis_compare_summary.parquet"

CELL = ["cut", "model"]
PHI = ["phi_a1_prior", "phi_a2_labels", "phi_b1_fs", "phi_b2_axis"]
QUANTITIES = PHI + ["residual", "delta_empty"]


def _cells(path: str, suffix: str) -> pd.DataFrame:
    """Average the seed dimension away, one row per (cut, model)."""
    frame = pd.read_parquet(path)
    missing = set(CELL + QUANTITIES) - set(frame.columns)
    if missing:
        raise AssertionError(f"{path} is missing {sorted(missing)}")
    return frame.groupby(CELL)[QUANTITIES].mean().add_suffix(suffix)


def main() -> None:
    rep = _cells(AS_REPORTED, "_rep")
    frz = _cells(FROZEN, "_frz")
    cells = rep.join(frz, how="inner").reset_index()
    if len(cells) != 20:
        raise AssertionError(f"expected 20 cells, got {len(cells)}")

    # v(N) is what the four controls jointly explain: the gap minus the residual.
    cells["vN"] = cells["delta_empty_rep"] - cells["residual_rep"]
    cells = cells.rename(columns={"residual_rep": "R_rep", "residual_frz": "R_frz"})

    order = (
        CELL
        + ["delta_empty_rep"]
        + [c + "_rep" for c in PHI]
        + ["vN", "R_rep"]
        + [c + "_frz" for c in PHI]
        + ["R_frz", "delta_empty_frz"]
    )
    cells = cells[order]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cells.to_parquet(OUT, index=False)

    rows = []
    for quantity in PHI + ["R"]:
        a = cells[quantity + "_rep"]
        b = cells[quantity + "_frz"]
        rows.append(
            {
                "quantity": quantity,
                "n_cells_compared": int(len(cells)),
                "as_reported_mean": float(a.mean()),
                "frozen_labels_mean": float(b.mean()),
                "mean_abs_diff": float((a - b).abs().mean()),
                "max_abs_diff": float((a - b).abs().max()),
                "sign_agreement": float((np.sign(a) == np.sign(b)).mean()),
                "rank_correlation": float(a.corr(b, method="spearman")),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_parquet(SUMMARY, index=False)

    print(summary.to_string(index=False), flush=True)
    print(f"\nwrote {OUT}  {cells.shape}", flush=True)
    print(f"wrote {SUMMARY}  {summary.shape}", flush=True)


if __name__ == "__main__":
    main()
