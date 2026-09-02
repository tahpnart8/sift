"""The gap-based value function.

The definition, from ``sift/experiment.py``::

    Delta(S) = M_reference(S) - M_temporal(S)
    v(S)     = Delta(empty) - Delta(S)
    R        = Delta(N)

so ``v(empty) = 0`` by construction and efficiency gives the reported identity
``Delta(empty) = sum(phi_i) + R``.

Both arms move with the coalition: ``b2_axis`` changes which families the
temporal window contains, and ``random_fully_matched`` restricts its test pool to
exactly that family set.  The last test in this module is the one that justifies
the change of definition, by exhibiting a case where the moving reference makes
the gap-based and temporal-only forms disagree.

``ASSUMED API``: no function yet turns a metrics table into a value function.
The suite expects ``sift.shapley.gap_values(metrics, model=..., cut=...,
seed=..., metric_column=...)`` returning ``dict[frozenset[str], float]``, and it
must refuse a table carrying reference rows.  Tests that fail on that name today
are the signal, not a defect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import build_metrics, metrics_row
from sift.config import CONTROL_NAMES, ControlFlags
from sift.experiment import (
    BASELINE_CONFIG_ID,
    LATTICE_DESIGNS,
    REFERENCE_DESIGNS,
    REFERENCE_ROLE,
    IncompleteLatticeError,
    assert_design_partition,
    assert_lattice_complete,
    lattice_cells,
    reference_cells,
)
from sift.shapley import decompose, exact_shapley

TOL = 1e-9


def _coalition(config_id: int) -> frozenset[str]:
    return ControlFlags.from_index(config_id).active()


def reference_gap_values(
    temporal_scores: dict[int, float], reference_scores: dict[int, float]
) -> tuple[dict[frozenset[str], float], float]:
    """The definition written out, used to check the production path against.

    Returns the value function and ``Delta(empty)``.
    """
    gaps = {
        _coalition(config_id): reference_scores[config_id] - temporal_scores[config_id]
        for config_id in range(16)
    }
    delta_empty = gaps[frozenset()]
    return {s: delta_empty - g for s, g in gaps.items()}, delta_empty


# --------------------------------------------------------------------------
# The two score maps used throughout. The reference arm deliberately moves.
# --------------------------------------------------------------------------
def _score_maps() -> tuple[dict[int, float], dict[int, float]]:
    rng = np.random.default_rng(11)
    temporal = {c: float(0.30 + 0.02 * c + 0.01 * rng.random()) for c in range(16)}
    reference = {c: float(0.80 + 0.005 * c - 0.01 * rng.random()) for c in range(16)}
    return temporal, reference


def test_v_of_the_empty_coalition_is_exactly_zero() -> None:
    """Exactly zero, not approximately: it is a subtraction of a number from itself."""
    temporal, reference = _score_maps()
    values, _ = reference_gap_values(temporal, reference)
    assert values[frozenset()] == 0.0
    assert repr(values[frozenset()]) in ("0.0", "-0.0")


def test_the_reported_identity_holds_on_a_full_lattice_table() -> None:
    """Delta(empty) == sum(phi) + R, to 1e-9."""
    temporal, reference = _score_maps()
    values, delta_empty = reference_gap_values(temporal, reference)

    result = decompose(delta_empty, values)
    assert sum(result.shapley.values()) + result.residual == pytest.approx(
        delta_empty, abs=TOL
    )
    assert result.all_checks_passed


def test_the_residual_equals_delta_of_the_grand_coalition() -> None:
    """R = Delta(N), which is the whole reason v is written as a gap difference."""
    temporal, reference = _score_maps()
    values, delta_empty = reference_gap_values(temporal, reference)
    delta_full = reference[15] - temporal[15]

    result = decompose(delta_empty, values)
    assert result.residual == pytest.approx(delta_full, abs=TOL)


def test_shapley_values_sum_to_v_of_the_grand_coalition() -> None:
    temporal, reference = _score_maps()
    values, _ = reference_gap_values(temporal, reference)
    phi = exact_shapley(values)
    assert sum(phi.values()) == pytest.approx(values[frozenset(CONTROL_NAMES)], abs=TOL)


# --------------------------------------------------------------------------
# Guards: reference rows and incomplete arms must never be decomposed
# --------------------------------------------------------------------------
def test_design_roles_partition_the_declared_designs() -> None:
    assert_design_partition()
    assert set(LATTICE_DESIGNS) & set(REFERENCE_DESIGNS) == set()


def test_lattice_cells_excludes_the_reference_measurements() -> None:
    temporal, reference = _score_maps()
    metrics = build_metrics(temporal, reference)
    extra = pd.DataFrame(
        [
            metrics_row(design=design, role=REFERENCE_ROLE, config_id=BASELINE_CONFIG_ID)
            for design in REFERENCE_DESIGNS
        ],
        columns=metrics.columns,
    )
    full = pd.concat([metrics, extra], ignore_index=True)

    kept = lattice_cells(full)
    assert set(kept["design"]) == set(LATTICE_DESIGNS)
    assert len(kept) == 32
    assert set(reference_cells(full)["design"]) == set(REFERENCE_DESIGNS)


def test_assert_lattice_complete_rejects_a_reference_arm_off_the_baseline() -> None:
    """A reference design carrying a non-baseline coalition is a contract breach."""
    temporal, reference = _score_maps()
    metrics = build_metrics(temporal, reference)
    stray = pd.DataFrame(
        [metrics_row(design="random_matched", role=REFERENCE_ROLE, config_id=7)],
        columns=metrics.columns,
    )
    with pytest.raises(IncompleteLatticeError):
        assert_lattice_complete(pd.concat([metrics, stray], ignore_index=True))


@pytest.mark.parametrize("design", list(LATTICE_DESIGNS))
def test_assert_lattice_complete_rejects_a_missing_coalition(design: str) -> None:
    """Checked on both arms: exact Shapley needs all 16 on each."""
    temporal, reference = _score_maps()
    metrics = build_metrics(temporal, reference)
    dropped = metrics.loc[~((metrics["design"] == design) & (metrics["config_id"] == 9))]
    with pytest.raises(IncompleteLatticeError):
        assert_lattice_complete(dropped)


def test_exact_shapley_still_refuses_an_incomplete_mapping() -> None:
    temporal, reference = _score_maps()
    values, _ = reference_gap_values(temporal, reference)
    del values[frozenset({"b1_fs", "b2_axis"})]
    with pytest.raises((KeyError, ValueError)):
        exact_shapley(values)


def test_gap_values_refuses_a_table_carrying_reference_rows() -> None:
    """The production builder must not silently average reference rows in."""
    from sift.shapley import gap_values

    temporal, reference = _score_maps()
    metrics = build_metrics(temporal, reference)
    polluted = pd.concat(
        [
            metrics,
            pd.DataFrame(
                [metrics_row(design="random", role=REFERENCE_ROLE)], columns=metrics.columns
            ),
        ],
        ignore_index=True,
    )
    with pytest.raises(Exception):
        gap_values(polluted, model="logreg", cut=2019, seed=0)


def test_gap_values_matches_the_written_definition() -> None:
    from sift.shapley import gap_values

    temporal, reference = _score_maps()
    metrics = build_metrics(temporal, reference)
    expected, _ = reference_gap_values(temporal, reference)

    produced = gap_values(metrics, model="logreg", cut=2019, seed=0)
    assert set(produced) == set(expected)
    for coalition, value in expected.items():
        assert produced[coalition] == pytest.approx(value, abs=TOL)


# --------------------------------------------------------------------------
# The test that justifies the change of definition
# --------------------------------------------------------------------------
def _temporal_only_values(temporal_scores: dict[int, float]) -> dict[frozenset[str], float]:
    """The superseded form: v(S) = M_temporal(S) - M_temporal(baseline)."""
    baseline = temporal_scores[0]
    return {_coalition(c): temporal_scores[c] - baseline for c in range(16)}


def test_the_two_forms_agree_when_the_reference_arm_is_constant() -> None:
    """With a fixed reference the gap-based form reduces to the temporal-only one.

    Delta(empty) - Delta(S) = (M_ref - M_t(empty)) - (M_ref - M_t(S))
                            = M_t(S) - M_t(empty)

    so the change of definition is a genuine generalisation and not a different
    quantity dressed up.
    """
    temporal, _ = _score_maps()
    constant_reference = {c: 0.82 for c in range(16)}

    gap_form, _ = reference_gap_values(temporal, constant_reference)
    temporal_form = _temporal_only_values(temporal)

    for coalition in gap_form:
        assert gap_form[coalition] == pytest.approx(temporal_form[coalition], abs=TOL)


def test_the_two_forms_disagree_when_the_reference_arm_moves() -> None:
    """This is the empirical justification for the gap-based definition.

    ``b2_axis`` changes which families the temporal test window holds, and
    ``random_fully_matched`` restricts its own test pool to that family set, so
    the reference arm moves with the coalition.  The fixture below moves the
    reference only on coalitions containing ``b2_axis``, which is exactly the
    real mechanism, and the two forms then give different Shapley attributions.
    """
    temporal, _ = _score_maps()
    moving_reference = {
        c: 0.82 + (0.06 if "b2_axis" in _coalition(c) else 0.0) for c in range(16)
    }

    gap_form, _ = reference_gap_values(temporal, moving_reference)
    temporal_form = _temporal_only_values(temporal)

    differing = [c for c in gap_form if abs(gap_form[c] - temporal_form[c]) > 1e-6]
    assert differing, "the fixture failed to move the reference arm"

    gap_phi = exact_shapley(gap_form)
    temporal_phi = exact_shapley(temporal_form)
    assert abs(gap_phi["b2_axis"] - temporal_phi["b2_axis"]) > 1e-6, (
        "the moving reference left phi_b2_axis unchanged, so this fixture does "
        "not exercise the reason the definition was changed"
    )
    # The other three controls are untouched by the reference movement here, so
    # the difference must be concentrated on b2_axis rather than smeared.
    for name in ("a1_prior", "a2_labels", "b1_fs"):
        assert gap_phi[name] == pytest.approx(temporal_phi[name], abs=1e-6)
