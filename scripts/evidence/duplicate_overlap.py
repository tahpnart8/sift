"""Item 5: duplicate leakage across the temporal boundary, and the 0.114 gap move.

Part A -- share of test-window rows whose exact feature vector also occurs in
the training window, measured on the *non-deduplicated* panel (the C1-off
state), per cut. No model is fitted.

Part B -- the effect of the dedup step on the reported gap, read straight off
the five-control C1 lattice at ``results/c1/metrics.parquet``. The gap is
Delta = reference(random_fully_matched) - temporal, at the baseline coalition
(no A/B control on), taken with C1 off minus C1 on.
"""

from __future__ import annotations

import pandas as pd

from sift import PanelSpec, SplitSpec, build_panel_variants, load_mlran, temporal_split
from sift.paths import RESULTS_DIR

CUTS = (2015, 2017, 2019, 2021, 2023)
OUT_A = RESULTS_DIR / "evidence" / "item5_dedup_overlap.parquet"
OUT_B = RESULTS_DIR / "evidence" / "item5_dedup_gap_move.parquet"


def part_a() -> None:
    raw = load_mlran()
    rows = []
    for task in ("family", "binary"):
        spec = PanelSpec(task=task)
        variants = build_panel_variants(raw, spec)
        for dedup, panel in (
            ("with_dedup", variants.with_dedup),
            ("without_dedup", variants.without_dedup),
        ):
            for cut in CUTS:
                for axis in (spec.primary_axis, spec.secondary_axis):
                    split = SplitSpec(design="temporal", cut_year=cut, test_window=3)
                    train_idx, test_idx = temporal_split(panel, split, axis)
                    train_groups = set(panel.loc[train_idx, "grp_id"])
                    test_groups = panel.loc[test_idx, "grp_id"]
                    n_test = int(len(test_groups))
                    n_dup = int(test_groups.isin(train_groups).sum())
                    rows.append(
                    {
                        "item": "item5_dedup_overlap",
                        "task": task,
                        "panel_variant": dedup,
                        "time_axis": axis,
                        "cut": cut,
                        "metric": "share_of_test_rows_duplicated_in_train",
                        "n_train": int(len(train_idx)),
                        "n_test": n_test,
                        "n_test_with_train_duplicate": n_dup,
                        "share": (n_dup / n_test) if n_test else float("nan"),
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_parquet(OUT_A, index=False)
    for task in ("family", "binary"):
        sub = frame[
            (frame.task == task)
            & (frame.panel_variant == "without_dedup")
            & (frame.time_axis == "first_submission_date_year")
        ]
        print(sub[["task", "cut", "n_train", "n_test", "n_test_with_train_duplicate", "share"]].to_string(index=False))
        print("  %s range %.4f to %.4f" % (task, sub.share.min(), sub.share.max()))
    print("wrote", OUT_A, frame.shape)


def part_b() -> None:
    m = pd.read_parquet(RESULTS_DIR / "c1" / "metrics.parquet")
    controls = ["a1_prior", "a2_labels", "b1_fs", "b2_axis"]
    base = m[~m[controls].any(axis=1)]
    per = (
        base.groupby(["c1_dedup", "design", "cut", "model"], as_index=False)["macro_f1"]
        .mean()
    )
    wide = per.pivot_table(
        index=["c1_dedup", "cut", "model"], columns="design", values="macro_f1"
    ).reset_index()
    wide["gap"] = wide["random_fully_matched"] - wide["temporal"]

    pooled = wide.groupby("c1_dedup", as_index=False)["gap"].mean()
    off = float(pooled.loc[pooled.c1_dedup == False, "gap"].iloc[0])
    on = float(pooled.loc[pooled.c1_dedup == True, "gap"].iloc[0])

    per_cut = wide.groupby(["c1_dedup", "cut"], as_index=False)["gap"].mean()
    pc = per_cut.pivot(index="cut", columns="c1_dedup", values="gap")
    pc.columns = ["gap_c1_off_dupes_kept", "gap_c1_on_deduped"]
    pc = pc.reset_index()
    pc["gap_move"] = pc["gap_c1_off_dupes_kept"] - pc["gap_c1_on_deduped"]

    rows = [
        {
            "item": "item5_dedup_gap_move",
            "scope": "pooled over 5 cuts x 4 models x seeds",
            "cut": -1,
            "metric": "macro_f1_gap_random_fully_matched_minus_temporal",
            "gap_c1_off_dupes_kept": off,
            "gap_c1_on_deduped": on,
            "gap_move": off - on,
            "n_fits": int(len(base)),
        }
    ]
    for _, r in pc.iterrows():
        rows.append(
            {
                "item": "item5_dedup_gap_move",
                "scope": "per cut, mean over 4 models x seeds",
                "cut": int(r["cut"]),
                "metric": "macro_f1_gap_random_fully_matched_minus_temporal",
                "gap_c1_off_dupes_kept": float(r["gap_c1_off_dupes_kept"]),
                "gap_c1_on_deduped": float(r["gap_c1_on_deduped"]),
                "gap_move": float(r["gap_move"]),
                "n_fits": int(len(base[base.cut == r["cut"]])),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_parquet(OUT_B, index=False)
    print(frame.to_string(index=False))
    print("wrote", OUT_B, frame.shape)


if __name__ == "__main__":
    part_a()
    print()
    part_b()
