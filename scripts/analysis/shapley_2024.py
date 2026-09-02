"""Regenerate every reported quantity on the 2024-extended panel.

Same code paths, same definitions as notebooks 03 and 04. Nothing is redefined
here: feature selection is :func:`sift.features.feature_columns`, the early and
late windows come from :func:`sift.controls.apply_controls` under the coalition
notebook 04 fixes, and every estimator is the production function.

Stages are independent and each writes its own parquet into
``results/shapley_2024/`` so a later stage failing does not lose an earlier one.

Usage::

    python regen_shapley_2024.py [stage ...]

with stages ``decomp``, ``bootstrap``, ``drift``, ``baseline``. Default: all.
"""

from __future__ import annotations

import json
import sys
import time
import warnings

import numpy as np
import pandas as pd

from sift.config import CONTROL_NAMES, ControlFlags, ExperimentConfig, PanelSpec, SplitSpec
from sift.controls import apply_controls
from sift.data import build_panel, load_mlran
from sift.drift import (
    MIN_INTERPRETABLE_HELDOUT,
    bootstrap_gap_distribution,
    conditional_structure_agreement,
    correlate_residual_with_drift,
    domain_classifier_test,
)
from sift.experiment import lattice_cells
from sift.features import feature_columns
from sift.paths import RESULTS_DIR
from sift.reporting import (
    build_agreement_report,
    build_coalition_values,
    build_gap_table,
    build_model_summary,
    class_prior_reference,
    compare_metric_decompositions,
    decompose_groups,
    panel_matches_run,
    recompute_a1_weights,
    rescore_predictions,
)
from sift.seeding import derive_seed
from sift.shapley import exact_shapley

warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

OUT = RESULTS_DIR / "shapley_2024"
PROTECTED = {"metrics.parquet", "predictions.parquet", "run_manifest.json"}

TIERS = [("a1_prior", "a2_labels"), ("b1_fs", "b2_axis")]
TIER_NAMES = ("data_side", "protocol_side")
PHI_COLUMNS = [f"phi_{name}" for name in CONTROL_NAMES]
PRIMARY_METRIC = "macro_f1"
METRICS_TO_DECOMPOSE = ("macro_f1", "accuracy")
ALPHA = 0.05
N_BOOT = 1000
GATE_HALF_WIDTH = 0.10
FULL_COALITION = 15
N_COND_REPEATS = 500
TARGET = "ransomware_family"
# notebook 04, section 3: prior fixed, label space restricted, time axis correct,
# feature selection left alone.
DRIFT_COALITION = ControlFlags(a1_prior=True, a2_labels=True, b1_fs=False, b2_axis=True)


def _write(frame: pd.DataFrame, name: str) -> None:
    assert name not in PROTECTED, name
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUT / name, index=False)
    print(f"  wrote {name}  {frame.shape}", flush=True)


def load_context():
    metrics = pd.read_parquet(RESULTS_DIR / "metrics.parquet")
    predictions = pd.read_parquet(RESULTS_DIR / "predictions.parquet")
    panel, _ = build_panel(load_mlran(), PanelSpec())
    prior_reference = class_prior_reference(panel)
    manifest = json.loads((RESULTS_DIR / "run_manifest.json").read_text(encoding="utf-8"))
    check = panel_matches_run(panel, manifest)
    print(check["detail"], flush=True)
    if not check["matches"]:
        raise AssertionError("panel does not match the run; nothing below can be trusted")
    print(
        f"metrics {metrics.shape}  predictions {predictions.shape}  panel {panel.shape}  "
        f"features {len(feature_columns(panel))}  families {panel[TARGET].nunique()}",
        flush=True,
    )
    return metrics, predictions, panel, prior_reference


# --------------------------------------------------------------------------
# Stage 1: the decomposition itself, on both metrics, plus the seed interval.
# --------------------------------------------------------------------------
def stage_decomp(metrics, predictions, prior_reference):
    as_reported = rescore_predictions(
        predictions, metrics, basis="as_reported", prior_reference=prior_reference
    )
    agreement = build_agreement_report(as_reported, metrics)
    _write(agreement, "agreement_report.parquet")
    bad = int(agreement.loc[agreement["reconstructable"], "n_mismatch"].sum())
    if bad:
        raise AssertionError(f"{bad} fits disagree between metrics and predictions")

    roles = metrics[["fit_id", "role"]]
    cells = as_reported.merge(roles, on="fit_id", validate="one_to_one")
    cells = cells[cells["role"] == "lattice"]

    frames, values_by_metric = {}, {}
    for metric_name in METRICS_TO_DECOMPOSE:
        gaps = build_gap_table(cells, metric_column=metric_name)
        vals = build_coalition_values(gaps)
        decs, frame, skipped = decompose_groups(
            vals, TIERS, TIER_NAMES, on_incomplete="raise"
        )
        assert len(skipped) == 0
        frames[metric_name] = frame
        values_by_metric[metric_name] = vals
        recomposed = frame[PHI_COLUMNS].sum(axis=1) + frame["residual"]
        worst = float(np.max(np.abs(recomposed - frame["delta_empty"])))
        print(
            f"  {metric_name}: {len(decs)} instances, checks_passed="
            f"{bool(frame['checks_passed'].all())}, max |identity error| = {worst:.3e}",
            flush=True,
        )
        frame_out = frame.copy()
        frame_out.insert(0, "metric", metric_name)
        _write(frame_out, f"decomposition_as_reported_{metric_name}.parquet")

        summary = build_model_summary(frame)
        summary.insert(0, "metric", metric_name)
        _write(summary, f"model_summary_{metric_name}.parquet")

        # Harsanyi dividends, seed-averaged per (cut, model).
        rows = []
        for (cut, model, seed), result in decs.items():
            record = {"cut": cut, "model": model, "seed": seed}
            for coalition, value in result.dividends.items():
                label = "+".join(sorted(coalition)) if coalition else "empty"
                record[label] = float(value)
            rows.append(record)
        dividends = pd.DataFrame(rows)
        dividends.insert(0, "metric", metric_name)
        _write(dividends, f"dividends_{metric_name}.parquet")

    # ---- quantity 3: Owen tiers -----------------------------------------
    owen_rows = []
    for metric_name in METRICS_TO_DECOMPOSE:
        summary = build_model_summary(frames[metric_name])
        tier_cols = ["tier_data_side", "tier_protocol_side"]
        total = summary[tier_cols].sum(axis=1)
        for row, share in zip(summary.itertuples(), (summary["tier_protocol_side"] / total)):
            owen_rows.append(
                {
                    "metric": metric_name,
                    "cut": int(row.cut),
                    "model": row.model,
                    "tier_data_side": float(row.tier_data_side),
                    "tier_protocol_side": float(row.tier_protocol_side),
                    "v_N": float(row.tier_data_side + row.tier_protocol_side),
                    "protocol_share": float(share),
                    **{
                        f"owen_{name}": float(getattr(row, f"owen_{name}"))
                        for name in CONTROL_NAMES
                    },
                }
            )
    owen = pd.DataFrame(owen_rows)
    _write(owen, "owen_tiers.parquet")

    owen_summary = []
    for metric_name, block in owen.groupby("metric"):
        for cut, sub in block.groupby("cut"):
            owen_summary.append(
                {
                    "metric": metric_name,
                    "scope": f"cut_{int(cut)}",
                    "protocol_share_mean": float(sub["protocol_share"].mean()),
                    "protocol_share_median": float(sub["protocol_share"].median()),
                    "n_cells": int(len(sub)),
                }
            )
        owen_summary.append(
            {
                "metric": metric_name,
                "scope": "all_cells",
                "protocol_share_mean": float(block["protocol_share"].mean()),
                "protocol_share_median": float(block["protocol_share"].median()),
                "n_cells": int(len(block)),
            }
        )
    _write(pd.DataFrame(owen_summary), "owen_protocol_share.parquet")

    # ---- quantity 6: metric comparison ----------------------------------
    comparison = compare_metric_decompositions(frames, controls=CONTROL_NAMES)
    pivot = comparison.pivot(index="quantity", columns="metric", values="mean_abs")
    pivot["ratio_accuracy_over_macro_f1"] = pivot["accuracy"] / pivot["macro_f1"].where(
        pivot["macro_f1"] > 1e-12
    )
    pivot = pivot.reset_index()
    _write(pivot, "metric_comparison.parquet")
    print(pivot.to_string(index=False), flush=True)

    # ---- quantity 2a: seed-based phi interval ---------------------------
    rows = []
    for metric_name in METRICS_TO_DECOMPOSE:
        summary = build_model_summary(frames[metric_name])
        for row in summary.itertuples():
            for control in CONTROL_NAMES:
                mean = float(getattr(row, f"phi_{control}"))
                half = float(getattr(row, f"phi_{control}_half"))
                rows.append(
                    {
                        "interval": "seed_t",
                        "metric": metric_name,
                        "cut": int(row.cut),
                        "model": row.model,
                        "control": control,
                        "phi": mean,
                        "ci_low": mean - half,
                        "ci_high": mean + half,
                        "half_width": half,
                        "covers_zero": bool(getattr(row, f"phi_{control}_covers_zero")),
                    }
                )
    seed_ci = pd.DataFrame(rows)
    _write(seed_ci, "phi_ci_seed.parquet")
    for metric_name, block in seed_ci.groupby("metric"):
        print(
            f"  seed interval, {metric_name}: {int(block['covers_zero'].sum())} of "
            f"{len(block)} cover zero",
            flush=True,
        )
    return frames


# --------------------------------------------------------------------------
# Stage 2: the prediction bootstrap, for R and for the four phi.
# --------------------------------------------------------------------------
def _arms(block, seed, config_id, by_fit, prior_reference):
    out = {}
    for design in ("random_fully_matched", "temporal"):
        row = block[
            (block["seed"] == seed)
            & (block["design"] == design)
            & (block["config_id"] == config_id)
        ].iloc[0]
        frame = by_fit[row["fit_id"]]
        y_true = frame["y_true"].to_numpy()
        weight = (
            recompute_a1_weights(y_true, prior_reference) if bool(row["a1_prior"]) else None
        )
        out[design] = (y_true, frame["y_pred"].to_numpy(), weight)
    return out


def stage_bootstrap(metrics, predictions, prior_reference):
    by_fit = {name: block for name, block in predictions.groupby("fit_id", sort=False)}
    lattice = lattice_cells(metrics)

    # ---- quantity 1: bootstrap CI for R, exactly the notebook 04 loop ----
    full = lattice[lattice["config_id"] == FULL_COALITION]
    boot_rows = []
    for (cut, model), block in full.groupby(["cut", "model"], sort=True):
        draws = []
        for seed in sorted(block["seed"].unique()):
            arms = _arms(block, seed, FULL_COALITION, by_fit, prior_reference)
            draws.append(
                bootstrap_gap_distribution(
                    arms["random_fully_matched"],
                    arms["temporal"],
                    n_boot=N_BOOT,
                    seed=derive_seed(0, "boot_R", int(cut), model, int(seed)),
                )
            )
        pooled = np.concatenate(draws)
        low, high = np.percentile(pooled, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
        boot_rows.append(
            {
                "cut": int(cut),
                "model": model,
                "R": float(pooled.mean()),
                "ci_low": float(low),
                "ci_high": float(high),
                "half_width": float((high - low) / 2),
                "excludes_zero": bool(low > 0 or high < 0),
                "passes_gate": bool((high - low) / 2 < GATE_HALF_WIDTH),
                "n_draws": int(pooled.size),
            }
        )
    residual_ci = pd.DataFrame(boot_rows)
    _write(residual_ci, "residual_ci.parquet")
    print(
        f"  R: {int(residual_ci['excludes_zero'].sum())} of {len(residual_ci)} exclude "
        f"zero; max half-width {residual_ci['half_width'].max():.4f}; "
        f"{int(residual_ci['passes_gate'].sum())} of {len(residual_ci)} pass the "
        f"{GATE_HALF_WIDTH} gate",
        flush=True,
    )

    # ---- quantity 2b: bootstrap CI for the four phi ---------------------
    # Same primitive, applied at every coalition. Each coalition is resampled
    # with its own derived seed; replicate b of every coalition is then read as
    # one draw of the whole characteristic function, and exact_shapley is run on
    # it. Pooling the five model seeds matches the R interval above.
    started = time.perf_counter()
    phi_rows = []
    for (cut, model), block in lattice.groupby(["cut", "model"], sort=True):
        pooled = {name: [] for name in CONTROL_NAMES}
        pooled_r = []
        for seed in sorted(block["seed"].unique()):
            per_config = {}
            for config_id in range(16):
                arms = _arms(block, seed, config_id, by_fit, prior_reference)
                per_config[config_id] = bootstrap_gap_distribution(
                    arms["random_fully_matched"],
                    arms["temporal"],
                    n_boot=N_BOOT,
                    seed=derive_seed(
                        0, "boot_phi", int(cut), model, int(seed), int(config_id)
                    ),
                )
            flags = {
                int(row.config_id): frozenset(
                    name for name in CONTROL_NAMES if bool(getattr(row, name))
                )
                for row in block.drop_duplicates("config_id").itertuples()
            }
            delta_empty = per_config[0]
            for b in range(N_BOOT):
                values = {
                    flags[config_id]: float(delta_empty[b] - per_config[config_id][b])
                    for config_id in range(16)
                }
                phi = exact_shapley(values)
                for name in CONTROL_NAMES:
                    pooled[name].append(phi[name])
                pooled_r.append(float(per_config[FULL_COALITION][b]))
        for name in CONTROL_NAMES:
            draws = np.asarray(pooled[name])
            low, high = np.percentile(draws, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
            phi_rows.append(
                {
                    "interval": "prediction_bootstrap",
                    "metric": PRIMARY_METRIC,
                    "cut": int(cut),
                    "model": model,
                    "control": name,
                    "phi": float(draws.mean()),
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "half_width": float((high - low) / 2),
                    "covers_zero": bool(low <= 0.0 <= high),
                    "n_draws": int(draws.size),
                }
            )
        print(
            f"  bootstrapped phi at cut {int(cut)} / {model} "
            f"({time.perf_counter() - started:.0f}s elapsed)",
            flush=True,
        )
    phi_ci = pd.DataFrame(phi_rows)
    _write(phi_ci, "phi_ci_bootstrap.parquet")

    counts = []
    for source, frame in (
        ("prediction_bootstrap", phi_ci),
        ("seed_t", pd.read_parquet(OUT / "phi_ci_seed.parquet")),
    ):
        block = frame[frame["metric"] == PRIMARY_METRIC]
        for control, sub in block.groupby("control"):
            counts.append(
                {
                    "interval": source,
                    "metric": PRIMARY_METRIC,
                    "control": control,
                    "n_cells": int(len(sub)),
                    "n_covering_zero": int(sub["covers_zero"].sum()),
                }
            )
        counts.append(
            {
                "interval": source,
                "metric": PRIMARY_METRIC,
                "control": "TOTAL",
                "n_cells": int(len(block)),
                "n_covering_zero": int(block["covers_zero"].sum()),
            }
        )
    coverage = pd.DataFrame(counts)
    _write(coverage, "phi_ci_coverage_counts.parquet")
    print(coverage.to_string(index=False), flush=True)


# --------------------------------------------------------------------------
# Stage 3: drift diagnostics.
# --------------------------------------------------------------------------
def stage_drift(metrics, panel):
    residual_ci = pd.read_parquet(OUT / "residual_ci.parquet")
    by_cut = residual_ci.groupby("cut").agg(
        n_models=("model", "size"), n_excluding_zero=("excludes_zero", "sum")
    )
    surviving = sorted(by_cut.index[by_cut["n_excluding_zero"] == by_cut["n_models"]])
    print(f"  cuts proceeding to the drift tests: {surviving}", flush=True)

    columns = feature_columns(panel)
    windows = {}
    for cut in sorted(metrics["cut"].unique()):
        cfg = ExperimentConfig(
            panel=PanelSpec(),
            split=SplitSpec(design="temporal", cut_year=int(cut)),
            flags=DRIFT_COALITION,
            model_name="logreg",
            seed=0,
        )
        controlled = apply_controls(panel, cfg)
        windows[int(cut)] = {
            "x_early": controlled.train[columns].to_numpy(dtype=np.float64),
            "y_early": controlled.train[TARGET].to_numpy(),
            "x_late": controlled.test[columns].to_numpy(dtype=np.float64),
            "y_late": controlled.test[TARGET].to_numpy(),
        }
        print(
            f"  cut {cut}: early {len(controlled.train)} x {len(columns)}  "
            f"late {len(controlled.test)}",
            flush=True,
        )

    domain_rows = []
    for cut, window in windows.items():
        result = domain_classifier_test(
            window["x_early"], window["x_late"], seed=derive_seed(0, "domain", cut)
        )
        domain_rows.append(
            {
                "cut": cut,
                "in_scope": cut in surviving,
                "accuracy": result.accuracy,
                "excess": result.excess_accuracy,
                "n_heldout": result.n_heldout,
                "p_value": result.p_value,
                "interpretable": result.interpretable,
                "min_interpretable_heldout": MIN_INTERPRETABLE_HELDOUT,
                "verdict": result.verdict(ALPHA),
            }
        )
    domain = pd.DataFrame(domain_rows)
    _write(domain, "domain_classifier.parquet")
    print(domain.to_string(index=False), flush=True)

    cond_rows = []
    for cut, window in windows.items():
        try:
            outcome = conditional_structure_agreement(
                window["x_early"],
                window["y_early"],
                window["x_late"],
                window["y_late"],
                seed=derive_seed(0, "conditional", cut),
                n_repeats=N_COND_REPEATS,
            )
        except ValueError as error:
            cond_rows.append(
                {
                    "cut": cut,
                    "in_scope": cut in surviving,
                    "n_repeats_requested": N_COND_REPEATS,
                    "note": str(error)[:120],
                }
            )
            continue
        cond_rows.append(
            {
                "cut": cut,
                "in_scope": cut in surviving,
                **outcome,
                "n_repeats_requested": N_COND_REPEATS,
                "note": "",
            }
        )
    conditional = pd.DataFrame(cond_rows)
    _write(conditional, "conditional_structure.parquet")
    print(conditional.to_string(index=False), flush=True)

    # ---- quantity 7: residual against drift magnitude -------------------
    distances = []
    for cut, window in windows.items():
        pooled = np.vstack([window["x_early"], window["x_late"]])
        scale = pooled.std(axis=0)
        scale[scale == 0.0] = np.inf
        gap = (
            np.abs(window["x_late"].mean(axis=0) - window["x_early"].mean(axis=0)) / scale
        )
        distances.append({"cut": cut, "mean_abs_std_diff": float(gap.mean())})
    distances = pd.DataFrame(distances).merge(domain[["cut", "excess"]], on="cut")
    _write(distances, "drift_magnitude.parquet")

    merged = residual_ci.merge(distances, on="cut")
    rows = []
    for column in ("excess", "mean_abs_std_diff"):
        outcome = correlate_residual_with_drift(
            merged["R"].to_numpy(), merged[column].to_numpy()
        )
        rows.append({"statistic_name": column, **outcome,
                     "n_distinct_cuts": int(merged["cut"].nunique())})
        print(
            f"  R vs {column}: rho={outcome['statistic']:+.3f} p={outcome['p_value']:.4f} "
            f"n={outcome['n_cells']}",
            flush=True,
        )
    _write(pd.DataFrame(rows), "residual_drift_correlation.parquet")

    # ---- the verdict table, same rule as notebook 04 --------------------
    verdicts = []
    for cut in sorted(windows):
        if cut not in surviving:
            verdicts.append(
                {
                    "cut": cut,
                    "input_shift": "not tested",
                    "conditional_shift": "not tested",
                    "association_changed": None,
                    "conclusion": "residual not distinguishable from zero",
                }
            )
            continue
        d = domain[domain["cut"] == cut].iloc[0]
        c = conditional[conditional["cut"] == cut].iloc[0]
        input_shift = {"shift": "yes", "no_shift": "no", "undetermined": "undetermined"}[
            d.verdict
        ]
        matched_below = (
            c.get("across_matched") == c.get("across_matched")
            and c.get("across_matched") < c.get("within_lo")
        )
        association_changed = bool(c.get("below_within_null")) and bool(matched_below)
        conditional_shift = "no evidence"
        if input_shift == "yes":
            conclusion = "covariate shift, NOT concept drift"
        elif input_shift == "no":
            conclusion = "unexplained residual, source outside the taxonomy"
        else:
            conclusion = "unexplained residual, evidence not clear"
        verdicts.append(
            {
                "cut": cut,
                "input_shift": input_shift,
                "conditional_shift": conditional_shift,
                "association_changed": association_changed,
                "conclusion": conclusion,
            }
        )
    _write(pd.DataFrame(verdicts), "residual_verdict.parquet")


# --------------------------------------------------------------------------
# Stage 4: the baseline_labels rescoring.
# --------------------------------------------------------------------------
def stage_baseline(metrics, predictions, prior_reference):
    scored = rescore_predictions(
        predictions,
        metrics,
        basis="baseline_labels",
        prior_reference=prior_reference,
    )
    roles = metrics[["fit_id", "role"]]
    cells = scored.merge(roles, on="fit_id", validate="one_to_one")
    cells = cells[cells["role"] == "lattice"]
    _write(
        cells.drop(columns=["frozen_labels"]).assign(
            frozen_labels_size=cells["n_frozen_labels"]
        ),
        "baseline_labels_cells.parquet",
    )

    gaps = build_gap_table(cells, metric_column=PRIMARY_METRIC)
    values = build_coalition_values(gaps)
    decs, frame, skipped = decompose_groups(
        values, TIERS, TIER_NAMES, on_incomplete="skip"
    )
    frame_out = frame.copy()
    frame_out.insert(0, "basis", "baseline_labels")
    _write(frame_out, "decomposition_baseline_labels_macro_f1.parquet")
    n_groups = int(cells.groupby(["cut", "model", "seed"]).ngroups)
    print(
        f"  baseline_labels: {len(frame)} of {n_groups} groups survived, "
        f"{len(skipped)} skipped",
        flush=True,
    )
    if len(skipped):
        _write(skipped, "baseline_labels_skipped.parquet")

    # The two scoring bases are compared by scripts/analysis/basis_compare.py,
    # which averages the seed dimension away first. Comparing them here over
    # (cut, model, seed) would treat the five seeds of a cell as five
    # independent groups, which is the aggregation the paper rules out.


def main() -> None:
    stages = sys.argv[1:] or ["decomp", "bootstrap", "drift", "baseline"]
    metrics, predictions, panel, prior_reference = load_context()
    if "decomp" in stages:
        print("== stage decomp ==", flush=True)
        stage_decomp(metrics, predictions, prior_reference)
    if "bootstrap" in stages:
        print("== stage bootstrap ==", flush=True)
        stage_bootstrap(metrics, predictions, prior_reference)
    if "drift" in stages:
        print("== stage drift ==", flush=True)
        stage_drift(metrics, panel)
    if "baseline" in stages:
        print("== stage baseline ==", flush=True)
        stage_baseline(metrics, predictions, prior_reference)
    print("done", flush=True)


if __name__ == "__main__":
    main()
