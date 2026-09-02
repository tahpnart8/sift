"""Recovery error and false attribution rate for the synthetic-injection study.

Reads ``results/recovery/recovery.parquet`` only. Nothing is fitted here: every
number below is an arithmetic restatement of quantities already stored by the
recovery run.

Two quantities are produced.

``RecErr_abs``
    ``|sum_{i injected} phi_i - v_full|`` for each constructed panel and each
    metric. On a panel where the injection is the only thing separating the
    temporal arm from the matched arm, the Shapley shares of the injected
    controls should exhaust the full-coalition value, so the difference is the
    part of the recovery the decomposition failed to route to the right player.

``FAR`` (false attribution rate)
    Share of (panel, control) pairs at which the control was *not* injected yet
    its bootstrap interval excludes zero. Three denominators are reported; see
    :data:`FAR_DEFINITIONS`.

Usage::

    python regen_validation_metrics.py
"""

from __future__ import annotations


import pandas as pd
from sift.paths import RESULTS_DIR

RECOVERY = RESULTS_DIR / "recovery" / "recovery.parquet"
OUT_DIR = RESULTS_DIR / "evidence"
OUT = OUT_DIR / "validation_metrics.parquet"

#: The three false-attribution denominators reported side by side.
FAR_DEFINITIONS: dict[str, str] = {
    "strict": (
        "every (panel, metric, control) pair at which the control was not "
        "injected"
    ),
    "drop_a1_when_a2_injected": (
        "the strict set minus a1_prior pairs on panels where a2_labels was "
        "injected; restricting the label space to families seen in training "
        "moves the class prior of the test window, so a non-zero phi_a1 there "
        "is a real prior shift rather than a spurious one"
    ),
    "drop_all_a1": (
        "the strict set minus every non-injected a1_prior pair, on the view "
        "that any change to the test pool moves the prior"
    ),
}


def recovery_error(recovery: pd.DataFrame) -> pd.DataFrame:
    """Return ``|sum phi over injected controls - v_full|`` per panel and metric."""
    rows = []
    for (panel, metric), group in recovery.groupby(["panel", "metric"], sort=True):
        injected = group[group["injected"]]
        phi_sum = float(injected["phi"].sum())
        v_full = float(group["v_full"].unique()[0])
        v_lo = float(group["v_full_lo"].unique()[0])
        v_hi = float(group["v_full_hi"].unique()[0])
        rows.append(
            {
                "panel": panel,
                "metric": metric,
                "n_injected": int(len(injected)),
                "injected_controls": "+".join(sorted(injected["control"])) or "none",
                "phi_sum_injected": phi_sum,
                "v_full": v_full,
                "v_full_lo": v_lo,
                "v_full_hi": v_hi,
                "v_full_covers_zero": bool(v_lo <= 0.0 <= v_hi),
                "rec_err_abs": abs(phi_sum - v_full),
                # A relative error is only meaningful where v_full is bounded
                # away from zero; it is reported but flagged, never used alone.
                "rec_err_rel": (abs(phi_sum - v_full) / abs(v_full)) if v_full else float("nan"),
                "residual": float(group["residual"].unique()[0]),
                "checks_passed": bool(group["checks_passed"].all()),
            }
        )
    return pd.DataFrame(rows)


def false_attribution(recovery: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the per-pair table and the three rate summaries."""
    a2_panels = set(
        recovery.loc[recovery["injected"] & (recovery["control"] == "a2_labels"), "panel"]
    )
    pairs = recovery[~recovery["injected"]].copy()
    pairs["excludes_zero"] = ~pairs["covers_zero"].astype(bool)
    pairs["a1_on_a2_panel"] = (pairs["control"] == "a1_prior") & pairs["panel"].isin(a2_panels)
    pairs["is_a1"] = pairs["control"] == "a1_prior"
    # A control whose flag never moves on this panel has phi identically zero by
    # construction; such a pair can never raise a false alarm. Counted, and also
    # reported separately so the denominator is not read as 36 live tests.
    pairs["degenerate"] = (
        (pairs["phi"] == 0.0) & (pairs["phi_lo"] == 0.0) & (pairs["phi_hi"] == 0.0)
    )

    masks = {
        "strict": pd.Series(True, index=pairs.index),
        "drop_a1_when_a2_injected": ~pairs["a1_on_a2_panel"],
        "drop_all_a1": ~pairs["is_a1"],
        "strict_non_degenerate": ~pairs["degenerate"],
    }
    rows = []
    for name, mask in masks.items():
        subset = pairs[mask]
        n = int(len(subset))
        k = int(subset["excludes_zero"].sum())
        rows.append(
            {
                "definition": name,
                "n_pairs": n,
                "n_false_attributions": k,
                "far": (k / n) if n else float("nan"),
                "offenders": ";".join(
                    f"{r.panel}/{r.metric}/{r.control}"
                    for r in subset[subset["excludes_zero"]].itertuples()
                ),
                "note": FAR_DEFINITIONS.get(name, "strict set minus structurally zero pairs"),
            }
        )
    return pairs, pd.DataFrame(rows)


def main() -> None:
    recovery = pd.read_parquet(RECOVERY)
    rec = recovery_error(recovery)
    pairs, far = false_attribution(recovery)

    frames = []
    block = rec.copy()
    block.insert(0, "block", "recovery_error")
    frames.append(block)
    block = far.copy()
    block.insert(0, "block", "false_attribution_rate")
    frames.append(block)
    block = pairs[
        [
            "panel",
            "metric",
            "control",
            "phi",
            "phi_lo",
            "phi_hi",
            "excludes_zero",
            "degenerate",
            "a1_on_a2_panel",
            "v_full",
        ]
    ].copy()
    block.insert(0, "block", "non_injected_pairs")
    frames.append(block)

    out = pd.concat(frames, ignore_index=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)

    pd.set_option("display.width", 240)
    print(rec.to_string(index=False))
    print()
    print(far[["definition", "n_pairs", "n_false_attributions", "far"]].to_string(index=False))
    print()
    for row in far.itertuples():
        print(f"{row.definition}: {row.offenders}")
    print(f"\nwrote {OUT}  {out.shape}")


if __name__ == "__main__":
    main()
