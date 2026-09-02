"""The size-matched random design.

``random_matched`` draws a random train set of exactly the temporal split's
``n_train`` and a random test set of exactly its ``n_test``, at the same cut and
under the same control flags.  Its whole purpose is to separate the
training-set-size effect from the time-order effect, so a size that is merely
close is not good enough: an implementation that quietly shrank to whatever the
panel could supply would report a Delta that still carried the size effect and
would be worse than useless, because it would look like it had been controlled.

Assumed API, to be confirmed by E1:

* ``"random_matched"`` joins ``sift.experiment.DESIGNS``;
* ``make_split`` dispatches on it, taking the target sizes from the temporal
  split at the same ``cut_year``;
* it raises when the panel cannot supply the requested sizes.

Tests that fail with ImportError or AttributeError today are the intended
signal.  Nothing here is stubbed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sift.config import ControlFlags, ExperimentConfig, PanelSpec, SplitSpec
from sift.controls import apply_controls
from sift.splits import make_split

CUTS = (2015, 2017, 2019, 2021, 2023)
TIME_COLUMN = "first_submission_date_year"
MATCHED = "random_matched"


def _spec(design: str, cut: int) -> SplitSpec:
    return SplitSpec(design=design, cut_year=cut)


def _sizes(panel: pd.DataFrame, design: str, cut: int, seed: int = 0) -> tuple[int, int]:
    train_idx, test_idx = make_split(
        panel,
        _spec(design, cut),
        TIME_COLUMN,
        seed,
        stratify_labels=panel["ransomware_family"],
    )
    return len(train_idx), len(test_idx)


def _config(flags: ControlFlags, design: str, cut: int) -> ExperimentConfig:
    return ExperimentConfig(
        panel=PanelSpec(),
        split=SplitSpec(design=design, cut_year=cut),
        flags=flags,
        model_name="logreg",
        seed=0,
        n_features=20,
    )


def test_random_matched_is_a_registered_design_with_a_role() -> None:
    """Registered in SplitSpec and classified into exactly one role.

    R2 found a stale second copy of the design tuple in experiment.py that drove
    the job builder and silently excluded this design. It is now expressed as a
    role partition, which assert_design_partition enforces.
    """
    from sift.config import SplitSpec
    from sift.experiment import (
        LATTICE_DESIGNS,
        REFERENCE_DESIGNS,
        assert_design_partition,
    )

    assert MATCHED in SplitSpec.DESIGNS
    assert_design_partition()
    # random_matched is a reference measurement, run at the baseline only; the
    # fully matched design is the one that carries all sixteen coalitions.
    assert MATCHED in REFERENCE_DESIGNS
    assert MATCHED not in LATTICE_DESIGNS
    assert "random_fully_matched" in LATTICE_DESIGNS


def test_make_split_does_not_fall_through_to_the_plain_random_design(panel) -> None:
    """The current dispatcher returns random_split for anything not temporal.

    That default is the hazard this test exists for: an unrecognised design name
    would silently produce an ordinary test_size=0.25 split rather than a
    size-matched one, and every downstream number would look plausible.
    """
    plain = _sizes(panel, "random", 2019)
    matched = _sizes(panel, MATCHED, 2019)
    assert matched != plain, (
        "random_matched produced exactly the plain random split sizes, so the "
        "dispatcher is falling through instead of matching"
    )


@pytest.mark.parametrize("cut", CUTS)
def test_matched_sizes_equal_the_temporal_sizes_exactly(panel, cut: int) -> None:
    temporal = _sizes(panel, "temporal", cut)
    matched = _sizes(panel, MATCHED, cut)
    assert matched == temporal, (
        "at cut {0} the matched design gave n_train={1} n_test={2} against the "
        "temporal n_train={3} n_test={4}".format(cut, matched[0], matched[1], *temporal)
    )


def test_matched_design_never_shrinks_below_the_temporal_reference(panel) -> None:
    """Restated as a one-sided invariant, in case E1 chooses a different API.

    Equality is what is wanted, but silent shrinkage is the failure that would
    corrupt the published Delta, so it is asserted separately and loudly.
    """
    for cut in CUTS:
        n_train_t, n_test_t = _sizes(panel, "temporal", cut)
        n_train_m, n_test_m = _sizes(panel, MATCHED, cut)
        assert n_train_m >= n_train_t, "matched train set shrank at cut {0}".format(cut)
        assert n_test_m >= n_test_t, "matched test set shrank at cut {0}".format(cut)


def test_matched_draw_is_reproducible_from_the_seed(panel) -> None:
    first = make_split(panel, _spec(MATCHED, 2019), TIME_COLUMN, 3,
                       stratify_labels=panel["ransomware_family"])
    second = make_split(panel, _spec(MATCHED, 2019), TIME_COLUMN, 3,
                        stratify_labels=panel["ransomware_family"])
    assert list(first[0]) == list(second[0])
    assert list(first[1]) == list(second[1])


def test_matched_draw_responds_to_the_seed(panel) -> None:
    """Otherwise the five model seeds would all evaluate one single draw."""
    first = make_split(panel, _spec(MATCHED, 2019), TIME_COLUMN, 0,
                       stratify_labels=panel["ransomware_family"])
    second = make_split(panel, _spec(MATCHED, 2019), TIME_COLUMN, 1,
                        stratify_labels=panel["ransomware_family"])
    assert list(first[0]) != list(second[0])
    # The sizes must not move with the seed, only the membership.
    assert (len(first[0]), len(first[1])) == (len(second[0]), len(second[1]))


def test_matched_train_and_test_are_disjoint(panel) -> None:
    for cut in CUTS:
        train_idx, test_idx = make_split(
            panel, _spec(MATCHED, cut), TIME_COLUMN, 0,
            stratify_labels=panel["ransomware_family"],
        )
        assert len(set(train_idx) & set(test_idx)) == 0
        assert set(train_idx) <= set(panel.index)
        assert set(test_idx) <= set(panel.index)


def test_matched_design_ignores_time_order(panel) -> None:
    """It is a random draw, so it must not respect the C1 ordering.

    If it did, it would be a second temporal split and would control nothing.
    """
    train_idx, test_idx = make_split(
        panel, _spec(MATCHED, 2019), TIME_COLUMN, 0,
        stratify_labels=panel["ransomware_family"],
    )
    latest_train = panel.loc[train_idx, TIME_COLUMN].max()
    earliest_test = panel.loc[test_idx, TIME_COLUMN].min()
    assert latest_train >= earliest_test, (
        "the matched design produced a time-ordered split, so it is not "
        "isolating the size effect from the order effect"
    )


def test_a2_filter_applies_to_the_matched_design_as_it_does_to_temporal(panel) -> None:
    """The seen-classes filter must be applied the same way in both designs.

    If A2 were skipped under random_matched, the two designs would differ in two
    respects at once and the size effect could not be read off the difference.
    """
    for design in ("temporal", MATCHED):
        controlled = apply_controls(panel, _config(ControlFlags(a2_labels=True), design, 2019))
        train_families = set(controlled.train["ransomware_family"])
        test_families = set(controlled.test["ransomware_family"])
        assert test_families <= train_families, (
            "design {0} left a family in test that is absent from train".format(design)
        )

    unfiltered = apply_controls(panel, _config(ControlFlags(a2_labels=False), MATCHED, 2019))
    filtered = apply_controls(panel, _config(ControlFlags(a2_labels=True), MATCHED, 2019))
    assert len(filtered.test_idx) <= len(unfiltered.test_idx)


def test_matched_design_raises_when_the_panel_cannot_supply_the_sizes(panel) -> None:
    """Loud failure, never a short draw.

    The panel is truncated to fewer rows than the temporal split at this cut
    needs, while the target sizes stay pinned to the full-panel temporal split.
    A correct implementation cannot satisfy the request and must say so.
    """
    n_train, n_test = _sizes(panel, "temporal", 2019)
    keep = max(n_train, n_test) // 2
    starved = panel.iloc[:keep]

    with pytest.raises(Exception) as excinfo:
        make_split(
            starved,
            SplitSpec(design=MATCHED, cut_year=2019, n_train=n_train, n_test=n_test),
            TIME_COLUMN,
            0,
            stratify_labels=starved["ransomware_family"],
        )
    assert not isinstance(excinfo.value, KeyboardInterrupt)


def test_matched_design_reaches_run_cell_and_is_labelled_correctly(panel) -> None:
    """The metrics row must carry the design name, or Delta cannot be split three ways."""
    from sift.experiment import run_cell

    record = run_cell(panel, _config(ControlFlags(), MATCHED, 2019))
    assert record["design"] == MATCHED
    assert np.isfinite(record["macro_f1"])

    temporal = run_cell(panel, _config(ControlFlags(), "temporal", 2019))
    assert record["n_train"] == temporal["n_train"]
    assert record["n_test"] == temporal["n_test"]
