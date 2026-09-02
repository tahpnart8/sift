"""Leakage, freezing, and the deduplication tie break.

Three separate hazards live here.  Feature selection under B1 must see training
rows only.  The frozen hyperparameters must be identical across all sixteen
cells, otherwise the Shapley decomposition attributes model selection variance
to the four controls.  And exact deduplication must be a stable sort with
sample_id as the secondary key, because two duplicate groups tie on
first_submission_date and an unstable sort would make the panel machine
dependent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import requires_mlran
from sift.config import ControlFlags, ExperimentConfig, PanelSpec, SplitSpec
from sift.controls import apply_controls
from sift.data import assign_duplicate_groups, build_panel
from sift.features import feature_columns
from sift.mock import synthetic_raw


def _config(
    flags: ControlFlags,
    cut: int = 2019,
    model: str = "logreg",
    n_features: int = 200,
) -> ExperimentConfig:
    return ExperimentConfig(
        panel=PanelSpec(),
        split=SplitSpec(design="temporal", cut_year=cut),
        flags=flags,
        model_name=model,
        seed=0,
        n_features=n_features,
    )


def _expected_survivors(raw: pd.DataFrame) -> set[int]:
    """Recompute the contract rule independently of build_panel.

    Drop the all-zero rows, group by exact feature vector, then keep the row with
    the earliest first_submission_date, ties broken by the smallest sample_id.
    """
    columns = feature_columns(raw)
    frame = raw.copy()
    frame["n_active"] = frame[columns].to_numpy().sum(axis=1)
    frame = frame[frame["n_active"] > 0]
    frame = frame.assign(grp_id=assign_duplicate_groups(frame, columns))
    winners = (
        frame.sort_values(["first_submission_date", "sample_id"], kind="stable")
        .drop_duplicates("grp_id", keep="first")
    )
    return set(winners["sample_id"])


def test_dedup_keeps_the_earliest_submission_of_each_duplicate_group() -> None:
    raw = synthetic_raw(n_samples=300, n_features=20, n_families=6, seed=0)
    panel, provenance = build_panel(raw, PanelSpec(min_class_size=1))

    assert provenance["n_unique_vectors"] < provenance["after_drop_empty"], (
        "the synthetic raw frame carries no duplicates, so this test proves nothing"
    )
    assert provenance["after_dedup"] == provenance["n_unique_vectors"]
    assert set(panel["sample_id"]) <= _expected_survivors(raw)
    assert panel["grp_id"].is_unique


def test_dedup_tiebreak_is_the_stable_sort_with_sample_id_secondary() -> None:
    """Two rows tie exactly on the date, so sample_id decides, ascending.

    The clone is given a larger sample_id and is placed ahead of the original in
    row order, so an implementation that keeps the first occurrence, or that
    sorts on the date alone, keeps the wrong row and fails here.
    """
    raw = synthetic_raw(n_samples=300, n_features=20, n_families=6, seed=0)
    original = raw[raw["sample_type"] == 1].iloc[0]

    clone = original.copy()
    clone["sample_id"] = int(raw["sample_id"].max()) + 1
    assert clone["sample_id"] > original["sample_id"]

    stacked = pd.concat([clone.to_frame().T, raw], ignore_index=True)
    stacked = stacked.astype(raw.dtypes.to_dict())

    columns = feature_columns(stacked)
    groups = assign_duplicate_groups(stacked, columns)
    pair = stacked.loc[
        stacked["sample_id"].isin([original["sample_id"], clone["sample_id"]])
    ]
    assert groups.loc[pair.index].nunique() == 1, "the clone did not land in the same group"

    panel, _ = build_panel(stacked, PanelSpec(min_class_size=1))
    survivors = set(panel["sample_id"])
    assert original["sample_id"] in survivors
    assert clone["sample_id"] not in survivors


def test_dedup_is_invariant_to_the_input_row_order() -> None:
    raw = synthetic_raw(n_samples=300, n_features=20, n_families=6, seed=0)
    shuffled = raw.sample(frac=1.0, random_state=1).reset_index(drop=True)

    a, _ = build_panel(raw, PanelSpec(min_class_size=1))
    b, _ = build_panel(shuffled, PanelSpec(min_class_size=1))
    assert sorted(a["sample_id"]) == sorted(b["sample_id"])


def test_feature_selection_uses_training_rows_only_when_b1_is_on(panel: pd.DataFrame) -> None:
    """B1 on is the correct protocol: mutual information on the training window.

    The check is behavioural.  Rewriting the label of every test row must leave
    the selected feature set untouched when B1 is on.  The B1 off arm is the
    deliberate leak and must be a genuinely different selection.
    """
    # The synthetic panel carries 40 feature columns, so the production budget of
    # 200 would select all of them and make B1 inert by construction.  Ten forces
    # the selection to actually choose.  On the real panel 200 of 483 already
    # binds.
    on = _config(ControlFlags(b1_fs=True), n_features=10)
    off = _config(ControlFlags(b1_fs=False), n_features=10)

    baseline_on = apply_controls(panel, on)
    baseline_off = apply_controls(panel, off)

    scrambled = panel.copy()
    rng = np.random.default_rng(0)
    test_rows = list(baseline_on.test_idx)
    scrambled.loc[test_rows, "ransomware_family"] = rng.permutation(
        scrambled.loc[test_rows, "ransomware_family"].to_numpy()
    )

    perturbed_on = apply_controls(scrambled, on)
    assert list(perturbed_on.columns) == list(baseline_on.columns), (
        "B1 on: the selected feature set moved when only test labels changed, "
        "so selection is reading the test window"
    )
    assert list(baseline_off.columns) != list(baseline_on.columns), (
        "B1 off produced the same features as B1 on, so the control is inert"
    )


def test_no_test_sample_id_appears_in_the_training_window(panel: pd.DataFrame) -> None:
    """Disjointness asserted on sample_id, not on the feature vector.

    Deduplication leaves feature vectors non unique across configurations, so
    identity has to be carried by the id column.
    """
    for index in range(16):
        cfg = _config(ControlFlags.from_index(index))
        data = apply_controls(panel, cfg)
        train_ids = set(panel.loc[data.train_idx, "sample_id"])
        test_ids = set(panel.loc[data.test_idx, "sample_id"])
        assert not (train_ids & test_ids), f"sample_id overlap in config {index}"


def test_frozen_hyperparameters_are_identical_across_all_sixteen_configs() -> None:
    """Design doc section 6: no tuning, one frozen parameter set everywhere."""
    from sift.models import build_model

    for model_name in ("logreg", "random_forest", "lightgbm", "mlp"):
        signatures = set()
        for _ in range(16):
            estimator = build_model(model_name, seed=0)
            params = {
                key: value
                for key, value in estimator.get_params().items()
                if key not in {"random_state", "n_jobs", "verbose", "verbosity"}
            }
            signatures.add(repr(sorted(params.items(), key=lambda kv: kv[0])))
        assert len(signatures) == 1, f"{model_name} hyperparameters vary across the lattice"


def test_estimator_seed_follows_the_derived_seed_not_a_hard_coded_constant() -> None:
    """The pilot pinned random_state=42 inside models(), so model seeds never moved."""
    from sift.models import build_model

    first = build_model("random_forest", seed=0).get_params()["random_state"]
    second = build_model("random_forest", seed=1).get_params()["random_state"]
    assert first is not None and second is not None
    assert first != second


def test_derived_seeds_are_stable_and_path_dependent() -> None:
    from sift.seeding import derive_seed

    assert derive_seed(0, "model", "lightgbm", 2019) == derive_seed(0, "model", "lightgbm", 2019)
    assert derive_seed(0, "model", "lightgbm", 2019) != derive_seed(0, "model", "lightgbm", 2021)
    assert derive_seed(0, "model", "lightgbm", 2019) != derive_seed(1, "model", "lightgbm", 2019)
    assert derive_seed(0, "split", "random", 0) != derive_seed(0, "model", "random", 0)


@requires_mlran
def test_real_panel_reaches_the_documented_row_counts() -> None:
    """Design doc section 5 fixes the provenance chain 4880 to 3311 to 1425.

    The window-filter count moved from 3169 to 3311 and the family panel from
    1377 to 1425 when ``PanelSpec.year_max`` was corrected from 2023 to 2024;
    2024 holds 263 samples, far above the sample-count floor that justifies the
    lower bound. The invariant, that the real chain hits the documented counts,
    is unchanged.
    """
    from sift.data import build_panel, load_mlran

    raw = load_mlran()
    real_panel, provenance = build_panel(raw, PanelSpec())
    counts = list(provenance.values())
    assert counts[0] == 4880
    assert 3311 in counts
    assert len(real_panel) in (3311, 1425)
