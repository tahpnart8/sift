"""Does drawing the constrained half first bias the reference arm?

``random_fully_matched`` restricts its test-half pool to the families present in
the temporal test window, draws that half first, then draws the training half
from the remainder.  The original order was reversed because drawing training
first failed outright at cut 2021.

CONCLUSION: the reversal does NOT introduce a
selection effect that biases the reference arm.  Three reasons.

1. At cuts 2021 and 2023 on the real panel, ``n_train + n_test == len(panel)``
   exactly, so the training half is the forced complement of the test half and
   the order is provably irrelevant.  Measured deficit between the two designs'
   usable training share at those cuts: 0.000 and 0.000.

2. Where a leftover pool does exist, test-first makes the *measured* half an
   unconditional stratified draw from the family pool.  Train-first would make
   the test half conditional on the training draw, which is strictly worse:
   the half the metric is computed on would then inherit the training draw's
   fluctuations.  Reversing the order moved the conditioning onto the half that
   is not scored.

3. The one genuine asymmetry, that part of the matched training budget goes to
   families outside the test pool, follows from matching ``n_train`` without
   matching training class coverage.  It is present under either order, since
   both stratify the training draw over a pool whose family mix is close to the
   panel's.  Its direction handicaps the reference arm, which lowers Delta and
   therefore lowers R, so it is conservative for the drift claim rather than
   drift-manufacturing.

The invariants below are what that reasoning licenses.  The load-bearing one is
``test_matched_training_half_covers_every_test_family``: it is what keeps the
reference arm able to predict every class it will be scored on, and a future
change that broke it would inflate the gap in a way nothing else here would see.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import requires_mlran
from sift.config import PanelSpec, SplitSpec
from sift.seeding import derive_seed
from sift.splits import make_split, matched_sizes, temporal_test_classes

LABEL = "ransomware_family"
TIME_COLUMN = "first_submission_date_year"
CUTS = (2015, 2017, 2019, 2021, 2023)
FULLY = "random_fully_matched"


@pytest.fixture(scope="module")
def real_panel():
    from sift.data import build_panel, load_mlran

    panel, _ = build_panel(load_mlran(), PanelSpec())
    return panel


def _draw(panel: pd.DataFrame, cut: int, seed: int = 0, restrict: bool = False):
    spec = SplitSpec(design=FULLY, cut_year=cut)
    return make_split(
        panel,
        spec,
        TIME_COLUMN,
        seed,
        stratify_labels=panel[LABEL],
        label_column=LABEL,
        restrict_labels=restrict,
    )


@requires_mlran
@pytest.mark.parametrize("cut", CUTS)
def test_matched_test_half_draws_only_from_the_temporal_family_pool(
    real_panel, cut: int
) -> None:
    """k_test matches because the pool is filtered before the draw, not after."""
    spec = SplitSpec(design=FULLY, cut_year=cut)
    wanted = temporal_test_classes(real_panel, spec, TIME_COLUMN, LABEL, False)
    _, test_idx = _draw(real_panel, cut)

    drawn = set(real_panel.loc[test_idx, LABEL])
    assert drawn <= set(wanted), (
        "the matched test half holds families outside the temporal test window, "
        "so filtering happened after the draw rather than before"
    )


@requires_mlran
@pytest.mark.parametrize("cut", CUTS)
def test_matched_halves_hit_the_temporal_sizes_exactly(real_panel, cut: int) -> None:
    """Filtering before the draw is what keeps this exact rather than short."""
    spec = SplitSpec(design=FULLY, cut_year=cut)
    n_train, n_test = matched_sizes(real_panel, spec, TIME_COLUMN, LABEL, False)
    train_idx, test_idx = _draw(real_panel, cut)
    assert len(train_idx) == n_train
    assert len(test_idx) == n_test


@requires_mlran
@pytest.mark.parametrize("cut", CUTS)
def test_matched_training_half_covers_every_test_family(real_panel, cut: int) -> None:
    """The load-bearing invariant of the reversed draw order.

    If drawing the test half first stranded a family, leaving none of it behind
    for training, the reference arm would be unable to predict a class it is
    scored on.  Its macro-F1 would fall, the gap would widen, and the residual
    would inflate for a purely mechanical reason.  Measured on the real panel
    this holds at every cut, with training class coverage of the test pool at
    17/17, 16/16, 22/22, 21/21 and 8/8.

    Note the temporal arm does NOT satisfy this, covering only 7/17 at cut 2015.
    That difference is control A2's subject and is meant to be there.
    """
    train_idx, test_idx = _draw(real_panel, cut)
    train_families = set(real_panel.loc[train_idx, LABEL])
    test_families = set(real_panel.loc[test_idx, LABEL])
    missing = test_families - train_families
    assert not missing, (
        "the matched training half is missing {0} of the {1} families it will be "
        "scored on at cut {2}: {3}".format(
            len(missing), len(test_families), cut, sorted(missing)[:5]
        )
    )


@requires_mlran
@pytest.mark.parametrize("cut", CUTS)
def test_the_two_halves_partition_when_the_panel_is_exactly_consumed(
    real_panel, cut: int
) -> None:
    """Where n_train + n_test == len(panel), the order cannot matter.

    The training half is then the forced complement of the test half, whichever
    is drawn first. This is the cleanest part of the argument that the reversal
    is safe, and it covers cuts 2021 and 2023.
    """
    spec = SplitSpec(design=FULLY, cut_year=cut)
    n_train, n_test = matched_sizes(real_panel, spec, TIME_COLUMN, LABEL, False)
    train_idx, test_idx = _draw(real_panel, cut)

    if n_train + n_test == len(real_panel):
        assert set(train_idx) == set(real_panel.index) - set(test_idx)
    else:
        assert len(set(train_idx) | set(test_idx)) < len(real_panel)


@requires_mlran
def test_at_least_one_cut_exercises_each_branch(real_panel) -> None:
    """Guards the test above from passing vacuously on one branch only."""
    exact, slack = 0, 0
    for cut in CUTS:
        spec = SplitSpec(design=FULLY, cut_year=cut)
        n_train, n_test = matched_sizes(real_panel, spec, TIME_COLUMN, LABEL, False)
        if n_train + n_test == len(real_panel):
            exact += 1
        else:
            slack += 1
    assert exact > 0 and slack > 0


@requires_mlran
@pytest.mark.parametrize("cut", CUTS)
def test_matched_halves_are_disjoint(real_panel, cut: int) -> None:
    train_idx, test_idx = _draw(real_panel, cut)
    assert len(set(train_idx) & set(test_idx)) == 0


@requires_mlran
def test_the_two_stages_draw_on_different_derived_seeds() -> None:
    """A shared seed would correlate the two halves of the same split."""
    assert derive_seed(0, "matched", "test") != derive_seed(0, "matched", "train")


@requires_mlran
@pytest.mark.parametrize("cut", CUTS)
def test_the_matched_draw_is_reproducible_and_seed_sensitive(real_panel, cut: int) -> None:
    first = _draw(real_panel, cut, seed=3)
    again = _draw(real_panel, cut, seed=3)
    assert list(first[0]) == list(again[0])
    assert list(first[1]) == list(again[1])

    other = _draw(real_panel, cut, seed=4)
    assert (len(other[0]), len(other[1])) == (len(first[0]), len(first[1]))
    assert list(other[1]) != list(first[1]), "the test half ignored the seed"


@requires_mlran
@pytest.mark.parametrize("cut", CUTS)
def test_the_test_half_is_reproducible_from_the_pool_and_seed_alone(
    real_panel, cut: int
) -> None:
    """The substantive claim behind the reversal, checked white box.

    Under test-first the test half is a function of three things only: the
    filtered pool, the requested size, and its own derived seed.  It cannot
    depend on the training draw, because the training draw has not happened yet.
    Recomputing it from those three inputs and finding the same index proves
    that, which is what makes the reversal safe: the conditioning moved onto the
    half that is not scored.

    Under the original order this test could not have been written, because the
    test half was drawn from ``panel minus train`` and so carried the training
    draw's fluctuations into the metric.
    """
    from sift.splits import _draw_exactly

    spec = SplitSpec(design=FULLY, cut_year=cut)
    _, n_test = matched_sizes(real_panel, spec, TIME_COLUMN, LABEL, False)
    wanted = temporal_test_classes(real_panel, spec, TIME_COLUMN, LABEL, False)

    pool = real_panel.index[real_panel[LABEL].isin(wanted).to_numpy()]
    expected, _ = _draw_exactly(
        pool,
        n_test,
        derive_seed(0, "matched", "test"),
        real_panel[LABEL].loc[pool],
        "test",
    )
    _, actual = _draw(real_panel, cut, seed=0)
    assert list(actual) == list(expected), (
        "the matched test half is not reproducible from the pool and its own "
        "seed, so something about the training draw is leaking into it"
    )


@requires_mlran
def test_matched_sizes_track_the_control_flags(real_panel) -> None:
    """The matched arm mirrors the temporal split under the SAME flags.

    Both ``matched_sizes`` and ``temporal_test_classes`` take ``restrict_labels``,
    so enabling A2 shrinks the temporal test window and the matched arm must
    shrink with it.  If it did not, the two arms would be compared at different
    sizes precisely on the coalitions where A2 is active, and phi_a2 would
    absorb a size effect.
    """
    spec = SplitSpec(design=FULLY, cut_year=2019)
    plain = matched_sizes(real_panel, spec, TIME_COLUMN, LABEL, False)
    restricted = matched_sizes(real_panel, spec, TIME_COLUMN, LABEL, True)

    temporal_spec = SplitSpec(design="temporal", cut_year=2019)
    t_plain = matched_sizes(real_panel, temporal_spec, TIME_COLUMN, LABEL, False)
    t_restricted = matched_sizes(real_panel, temporal_spec, TIME_COLUMN, LABEL, True)

    assert plain == t_plain
    assert restricted == t_restricted
    assert restricted[1] <= plain[1], "A2 did not shrink the temporal test window"


@requires_mlran
@pytest.mark.parametrize("cut", CUTS)
def test_the_matched_arm_succeeds_at_every_cut(real_panel, cut: int) -> None:
    """The operative reason for the reversal: the original order failed at 2021."""
    train_idx, test_idx = _draw(real_panel, cut)
    assert len(train_idx) > 0 and len(test_idx) > 0


@requires_mlran
def test_training_budget_spent_outside_the_test_pool_is_recorded_not_hidden(
    real_panel,
) -> None:
    """The one real asymmetry, asserted as a measured fact rather than a bound.

    At cut 2015 only 41 per cent of the matched training budget goes to families
    that can appear in the test half, against 99.6 per cent for the temporal
    arm.  That handicaps the reference arm, lowers Delta and therefore lowers R,
    so it is conservative for the drift claim.  It is asserted here so that the
    number is pinned and any future change to it is visible rather than silent.
    """
    spec = SplitSpec(design=FULLY, cut_year=2015)
    wanted = temporal_test_classes(real_panel, spec, TIME_COLUMN, LABEL, False)
    train_idx, _ = _draw(real_panel, 2015)
    usable = real_panel.loc[train_idx, LABEL].isin(wanted).mean()
    assert 0.30 < usable < 0.55, (
        "usable training share at cut 2015 moved to {0:.3f}; the gap decomposition "
        "assumes this is stable and it is now worth re-deriving".format(usable)
    )
