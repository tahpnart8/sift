"""Control semantics, lattice totality at run level, and determinism.

CONTRACT.md section 1 states that the sixteen cells must all exist and that an
undersized cell has to raise loudly rather than be skipped, because exact
Shapley needs all sixteen values.  It also fixes what each control does when it
is off and when it is on; those definitions are asserted here behaviourally
rather than trusted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import run_in_subprocess
from sift.config import ControlFlags, ExperimentConfig, PanelSpec, SplitSpec
from sift.controls import CONTROL_ORDER, apply_controls
from sift.experiment import run_cell, run_lattice

CONTRACT_COLUMNS = {
    "fit_id",
    "run_id",
    "config_id",
    "a1_prior",
    "a2_labels",
    "b1_fs",
    "b2_axis",
    "design",
    "cut",
    "seed",
    "model",
    "n_train",
    "n_test",
    "k_train",
    "k_test",
    "macro_f1",
    "accuracy",
    "balanced_accuracy",
    "mcc",
    "fit_seconds",
}


def _config(flags: ControlFlags, cut: int = 2019, model: str = "logreg") -> ExperimentConfig:
    return ExperimentConfig(
        panel=PanelSpec(),
        split=SplitSpec(design="temporal", cut_year=cut),
        flags=flags,
        model_name=model,
        seed=0,
    )


def test_run_lattice_returns_all_sixteen_config_ids(panel: pd.DataFrame, tmp_path) -> None:
    frame = run_lattice(
        panel,
        base=_config(ControlFlags()),
        model_names=["logreg"],
        seeds=[0],
        cut_years=[2019],
        results_dir=tmp_path,
    )
    assert sorted(frame["config_id"].unique().tolist()) == list(range(16))
    # Two roles, two shapes. The lattice arms carry all sixteen coalitions each,
    # because the gap-based value function moves on both. The reference arms are
    # baseline-only measurements about the reported gap and must never be
    # expanded to sixteen. Counts are derived from the role tuples rather than
    # hard coded so that adding a design does not falsify this invariant.
    from sift.experiment import (
        BASELINE_CONFIG_ID,
        LATTICE_DESIGNS,
        REFERENCE_DESIGNS,
    )

    assert set(frame["design"]) == set(LATTICE_DESIGNS) | set(REFERENCE_DESIGNS)

    lattice = frame.loc[frame["design"].isin(LATTICE_DESIGNS)]
    assert lattice.groupby("design")["config_id"].nunique().to_dict() == {
        design: 16 for design in LATTICE_DESIGNS
    }

    reference = frame.loc[frame["design"].isin(REFERENCE_DESIGNS)]
    assert reference.groupby("design")["config_id"].nunique().to_dict() == {
        design: 1 for design in REFERENCE_DESIGNS
    }
    assert set(reference["config_id"]) == {BASELINE_CONFIG_ID}

    assert len(frame) == 16 * len(LATTICE_DESIGNS) + len(REFERENCE_DESIGNS)


def test_run_lattice_emits_every_contract_column(panel: pd.DataFrame, tmp_path) -> None:
    frame = run_lattice(
        panel,
        base=_config(ControlFlags()),
        model_names=["logreg"],
        seeds=[0],
        cut_years=[2019],
        results_dir=tmp_path,
    )
    missing = CONTRACT_COLUMNS - set(frame.columns)
    assert not missing, f"metrics schema is missing {sorted(missing)}"


def test_every_lattice_cell_reports_a_finite_metric(panel: pd.DataFrame, tmp_path) -> None:
    """A cell that quietly produced nan would break exact Shapley silently."""
    frame = run_lattice(
        panel,
        base=_config(ControlFlags()),
        model_names=["logreg"],
        seeds=[0],
        cut_years=[2019],
        results_dir=tmp_path,
    )
    assert frame["macro_f1"].notna().all()
    assert np.isfinite(frame["macro_f1"].to_numpy()).all()
    assert (frame["k_test"] > 0).all()


def test_run_lattice_raises_instead_of_dropping_an_undersized_cell(panel: pd.DataFrame, tmp_path) -> None:
    """CONTRACT.md section 1 forbids skipping a cell, however small it gets.

    A cut placed past the end of the panel starves at least one cell.  The
    correct behaviour is a loud exception, not a shorter table.
    """
    impossible_cut = int(panel["first_submission_date_year"].max()) + 5
    with pytest.raises(Exception) as excinfo:
        run_lattice(
            panel,
            base=_config(ControlFlags(), cut=impossible_cut),
            model_names=["logreg"],
            seeds=[0],
            cut_years=[impossible_cut],
            results_dir=tmp_path,
        )
    assert not isinstance(excinfo.value, KeyboardInterrupt)


def test_b2_axis_off_uses_the_compile_year_and_on_uses_the_submission_date() -> None:
    """The B2 definition in CONTRACT.md section 1, asserted rather than trusted."""
    from sift.mock import synthetic_panel

    local_panel = synthetic_panel(n_samples=600, n_features=40, n_families=8, seed=0)
    off = apply_controls(local_panel, _config(ControlFlags(b2_axis=False)))
    on = apply_controls(local_panel, _config(ControlFlags(b2_axis=True)))
    assert off.time_column == "Year"
    assert on.time_column == "first_submission_date_year"
    # Different axes must actually move samples, otherwise B2 measures nothing.
    assert set(off.test_idx) != set(on.test_idx)


def test_a2_labels_on_removes_families_absent_from_training(
    panel_with_novel_family: pd.DataFrame,
) -> None:
    """Runs on the novel-family fixture, not the plain one.

    On the plain synthetic panel every family spans the whole time range, so
    this assertion held vacuously in R1: A2 removed nothing and the subset check
    was trivially true.  The strict inequality below is what makes it bite.
    """
    local = panel_with_novel_family
    off = apply_controls(local, _config(ControlFlags(a2_labels=False)))
    on = apply_controls(local, _config(ControlFlags(a2_labels=True)))

    train_families_on = set(local.loc[on.train_idx, "ransomware_family"])
    test_families_on = set(local.loc[on.test_idx, "ransomware_family"])
    assert test_families_on <= train_families_on

    assert len(on.test_idx) < len(off.test_idx), "A2 removed nothing"


def test_a1_prior_reweights_without_changing_the_sample_set(panel: pd.DataFrame) -> None:
    """Design doc section 3 chose reweighting precisely so no sample is dropped."""
    off = apply_controls(panel, _config(ControlFlags(a1_prior=False)))
    on = apply_controls(panel, _config(ControlFlags(a1_prior=True)))
    assert set(off.test_idx) == set(on.test_idx)
    assert set(off.train_idx) == set(on.train_idx)
    assert on.test_weights is not None
    assert len(on.test_weights) == len(on.test_idx)
    assert np.all(np.asarray(on.test_weights) > 0)


def test_apply_controls_is_idempotent(panel: pd.DataFrame) -> None:
    """The same input must give the same ControlledData every time."""
    for index in range(16):
        cfg = _config(ControlFlags.from_index(index))
        first = apply_controls(panel, cfg)
        second = apply_controls(panel, cfg)
        assert list(first.train_idx) == list(second.train_idx)
        assert list(first.test_idx) == list(second.test_idx)
        assert list(first.columns) == list(second.columns)


def test_apply_controls_does_not_mutate_the_panel(panel: pd.DataFrame) -> None:
    """A control that edits the shared panel in place poisons later cells."""
    before = pd.util.hash_pandas_object(panel, index=True).sum()
    for index in range(16):
        apply_controls(panel, _config(ControlFlags.from_index(index)))
    after = pd.util.hash_pandas_object(panel, index=True).sum()
    assert before == after


def test_control_order_is_exposed_and_places_the_axis_first() -> None:
    """b2_axis decides which rows land where, so it has to run first."""
    assert CONTROL_ORDER[0] == "b2_axis"
    assert CONTROL_ORDER[-1] == "b1_fs"


def test_run_cell_is_deterministic_across_two_separate_processes() -> None:
    """Same process repeats cannot see hash seeding or thread scheduling effects."""
    source = """
import json
from sift.config import ControlFlags, ExperimentConfig, PanelSpec, SplitSpec
from sift.experiment import run_cell
from sift.mock import synthetic_panel

panel = synthetic_panel(n_samples=600, n_features=40, n_families=8, seed=0)
cfg = ExperimentConfig(
    panel=PanelSpec(),
    split=SplitSpec(design="temporal", cut_year=2019),
    flags=ControlFlags.from_index(11),
    model_name="random_forest",
    seed=0,
)
out = run_cell(panel, cfg)
print(json.dumps({k: out[k] for k in ("macro_f1", "accuracy", "balanced_accuracy", "mcc")}))
"""
    first = run_in_subprocess(source)
    second = run_in_subprocess(source)
    assert first == second, "run_cell is not reproducible across processes"


def test_experiment_config_fingerprint_separates_configurations(panel: pd.DataFrame) -> None:
    """The cache key must not collide between two different cells."""
    fingerprints = {_config(ControlFlags.from_index(i)).fingerprint() for i in range(16)}
    assert len(fingerprints) == 16
    same = _config(ControlFlags.from_index(3)).fingerprint()
    assert same == _config(ControlFlags.from_index(3)).fingerprint()
