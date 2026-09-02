"""A1: dataset summary from the panel. No fitting."""
from __future__ import annotations
import pandas as pd

from sift.config import PanelSpec
from sift.data import build_panel, load_mlran
from sift.features import feature_columns
from sift.paths import RESULTS_DIR

CUTS = (2015, 2017, 2019, 2021, 2023)
TEST_WINDOW = 3



def main() -> None:
    spec = PanelSpec()
    raw = load_mlran()
    panel, prov = build_panel(raw, spec)
    print("PROVENANCE:", prov)

    panel_path = RESULTS_DIR / "panel.parquet"
    if not panel_path.exists():
        panel.to_parquet(panel_path, index=False)
        print("wrote", panel_path)
    else:
        print("exists", panel_path)

    feats = feature_columns(panel)
    axis = spec.primary_axis
    tgt = "ransomware_family"
    rows = []

    def add(section, key, value, note=""):
        rows.append({"section": section, "key": str(key), "value": float(value), "note": note})

    # --- overall
    add("overall", "n_samples", len(panel))
    add("overall", "n_families", panel[tgt].nunique())
    add("overall", "n_features", len(feats))
    add("overall", "year_min", panel[axis].min())
    add("overall", "year_max", panel[axis].max())
    add("overall", "n_active_median", panel["n_active"].median())

    # --- provenance chain
    for k, v in prov.items():
        add("provenance", k, v)

    # --- class distribution
    sizes = panel[tgt].value_counts().sort_values(ascending=False)
    for fam, n in sizes.items():
        add("family_size", fam, n)
    add("family_size_stats", "min", sizes.min())
    add("family_size_stats", "median", sizes.median())
    add("family_size_stats", "max", sizes.max())
    add("family_size_stats", "mean", sizes.mean())
    add("family_size_stats", "n_families", len(sizes))

    # --- year distribution (primary axis)
    by_year = panel[axis].value_counts().sort_index()
    for y, n in by_year.items():
        add("year_count_primary", int(y), n)
    by_year2 = panel[spec.secondary_axis].value_counts().sort_index()
    for y, n in by_year2.items():
        add("year_count_secondary", int(y), n)

    # --- per-cut split accounting, on BOTH temporal axes.
    # axis='primary'  = first_submission_date_year  -> the corrected axis (b2_axis=True)
    # axis='secondary'= Year (compile timestamp)    -> the naive axis (b2_axis=False)
    for axis_name, axis_col in (("primary", spec.primary_axis), ("secondary", spec.secondary_axis)):
        years = panel[axis_col]
        for cut in CUTS:
            last = min(cut + TEST_WINDOW - 1, spec.year_max)
            tr = panel[years < cut]
            te = panel[years.between(cut, last)]
            ftr = set(tr[tgt].unique())
            fte = set(te[tgt].unique())
            unseen = fte - ftr
            n_unseen_rows = int(te[tgt].isin(unseen).sum())
            p = f"cut{cut}"
            sec = f"cut_{axis_name}"
            add(sec, f"{p}_test_last_year", last)
            add(sec, f"{p}_n_train", len(tr))
            add(sec, f"{p}_n_test", len(te))
            add(sec, f"{p}_k_train", len(ftr))
            add(sec, f"{p}_k_test", len(fte))
            add(sec, f"{p}_k_unseen", len(unseen), note=";".join(sorted(unseen)))
            add(sec, f"{p}_n_test_rows_unseen", n_unseen_rows)
            add(sec, f"{p}_share_test_rows_unseen",
                (n_unseen_rows / len(te)) if len(te) else float("nan"))

    # --- how far the two axes disagree (the raw material of B2)
    disagree = (panel[spec.primary_axis] != panel[spec.secondary_axis])
    add("axis_disagreement", "n_rows_axes_differ", int(disagree.sum()))
    add("axis_disagreement", "share_rows_axes_differ", float(disagree.mean()))
    add("axis_disagreement", "median_primary_minus_secondary",
        float((panel[spec.primary_axis] - panel[spec.secondary_axis]).median()))

    out = pd.DataFrame(rows)
    ev = RESULTS_DIR / "evidence"
    ev.mkdir(parents=True, exist_ok=True)
    out.to_parquet(ev / "dataset_summary.parquet", index=False)
    print("wrote", ev / "dataset_summary.parquet", out.shape)



if __name__ == "__main__":
    main()
