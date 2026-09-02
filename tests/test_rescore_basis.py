"""The a priori frozen basis of ``sift.reporting.rescore_predictions``.

Design section 8.5 scores every coalition a second time on a frozen basis to
show that v(S) is comparable across coalitions.  The ``"common"`` basis freezes
by intersecting the sixteen realised test windows, and that intersection is a
subset of the A2-on window by construction: control A2 works by deleting test
samples whose family was never seen in training, so every sample it deletes is
absent from the intersection too.  A2 has therefore already been applied to the
frozen set before phi_A2 is measured on it, and a phi_A2 near zero is then an
artefact of the construction rather than evidence about the control.  Control B2
moves the window as well and is compromised the same way.

The ``"baseline_labels"`` basis freezes instead on the family support of the
group's baseline window, the cell with all four controls off.  The property that
makes it usable is stated once here and pinned by every test in this file: the
frozen support is *not a function of which controls are on*.  It is identical
for all sixteen coalitions of a group, and switching A2 on cannot shrink it.

The fixture is built so that both facts are observable.  A2 removes two families
from the window outright, so a frozen set that were a function of the flags would
be visibly smaller on the A2 cells; ``k_present`` shows those families really are
gone from the scored samples while ``k_scored`` stays at the frozen size, which
is the zero-contribution the macro average must keep charging.
"""

from __future__ import annotations

import pandas as pd
import pytest

from conftest import metrics_row
from sift.config import ControlFlags

CUT = 2019
MODEL = "logreg"
SEED = 0
DESIGN = "temporal"

#: Samples per family.  Above ``MIN_COMMON_SAMPLES`` so that a cell holding a
#: single family is still feasible and the feasibility floor is exercised only
#: by the test that means to exercise it.
PER_FAMILY = 25

#: Families of the baseline (all controls off) window.
BASE_FAMILIES = (0, 1, 2, 3, 4, 5)

#: Families of the window control B2 selects, overlapping the baseline in three
#: families and introducing two the baseline never contained.
B2_FAMILIES = (3, 4, 5, 6, 7)

#: Families present in the training half, i.e. the ones control A2 keeps.
SEEN_IN_TRAINING = (0, 1, 2, 3)


def _sample_ids(family: int, per_family: int = PER_FAMILY) -> range:
    return range(family * per_family, (family + 1) * per_family)


def _window(config_id: int, base_families, b2_families, seen) -> list[int]:
    """Families of one cell's test window, as the controls would leave them."""
    flags = ControlFlags.from_index(config_id)
    families = list(b2_families if flags.b2_axis else base_families)
    if flags.a2_labels:
        families = [family for family in families if family in seen]
    return families


def build_group(
    cut: int = CUT,
    model: str = MODEL,
    seed: int = SEED,
    design: str = DESIGN,
    base_families=BASE_FAMILIES,
    b2_families=B2_FAMILIES,
    seen=SEEN_IN_TRAINING,
    per_family: int = PER_FAMILY,
    tag: str = "g",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One complete sixteen-cell lattice group, as predictions plus metrics.

    B2 swaps the window for a different family range, A2 filters whichever
    window is in force, and A1 and B1 leave the sample composition alone, which
    is the behaviour of ``sift.controls`` this fixture stands in for.
    """
    prediction_rows: list[dict] = []
    metric_rows: list[dict] = []
    for config_id in range(16):
        flags = ControlFlags.from_index(config_id)
        fit_id = f"{tag}_{config_id}"
        families = _window(config_id, base_families, b2_families, seen)
        for family in families:
            for position, sample_id in enumerate(_sample_ids(family, per_family)):
                if position * 5 < per_family * 3:
                    predicted = family
                else:
                    predicted = (family + 1) % 8
                prediction_rows.append(
                    {
                        "fit_id": fit_id,
                        "sample_id": sample_id,
                        "grp_id": family,
                        "y_true": family,
                        "y_pred": predicted,
                        "test_year": cut + 1,
                    }
                )
        metric_rows.append(
            metrics_row(
                fit_id=fit_id,
                config_id=config_id,
                design=design,
                cut=cut,
                model=model,
                seed=seed,
                a1_prior=flags.a1_prior,
                a2_labels=flags.a2_labels,
                b1_fs=flags.b1_fs,
                b2_axis=flags.b2_axis,
                n_test=len(families) * per_family,
                k_test=len(families),
            )
        )
    return pd.DataFrame(prediction_rows), pd.DataFrame(metric_rows)


@pytest.fixture()
def group() -> tuple[pd.DataFrame, pd.DataFrame]:
    return build_group()


@pytest.fixture()
def scored(group) -> pd.DataFrame:
    from sift.reporting import rescore_predictions

    predictions, metrics = group
    return rescore_predictions(predictions, metrics, basis="baseline_labels")


# ---------------------------------------------------------------------------
# The defining property
# ---------------------------------------------------------------------------


def test_frozen_set_is_identical_across_all_sixteen_coalitions(scored):
    """The frozen support must not be a function of which controls are on."""
    assert len(scored) == 16
    frozen = set(scored["frozen_labels"])
    assert len(frozen) == 1, (
        "the frozen support differs between coalitions of one group, so it is "
        f"a function of the flags after all: {sorted(frozen)}"
    )
    assert frozen.pop() == tuple(BASE_FAMILIES)
    assert set(scored["n_frozen_labels"]) == {len(BASE_FAMILIES)}


def test_frozen_set_does_not_shrink_when_a2_is_switched_on(scored):
    """A2 filters the window; it must not be allowed to filter the yardstick."""
    on = scored[scored["a2_labels"]]
    off = scored[~scored["a2_labels"]]
    assert len(on) == len(off) == 8
    assert set(on["frozen_labels"]) == set(off["frozen_labels"])
    assert on["n_frozen_labels"].min() == off["n_frozen_labels"].max()
    # The fixture is not vacuous: holding B2 fixed, A2 really does remove
    # families from the cells it is compared against.
    for axis, stratum in scored.groupby("b2_axis"):
        filtered = stratum[stratum["a2_labels"]]["k_present"]
        intact = stratum[~stratum["a2_labels"]]["k_present"]
        assert filtered.max() < intact.min(), f"A2 removed nothing at b2_axis={axis}"
    # Yet the macro average keeps charging for them.
    assert set(scored["k_scored"]) == {len(BASE_FAMILIES)}


def test_frozen_set_survives_a_harsher_a2_and_a_different_b2_window(group):
    """Only the baseline cell may move the frozen set.

    Re-running the group with a stricter A2 and a different B2 window changes
    fifteen of the sixteen realised windows and leaves the baseline alone.  A
    frozen set derived from the realised windows, the intersection included,
    moves under this edit; an a priori one cannot.
    """
    from sift.reporting import rescore_predictions

    predictions, metrics = group
    before = rescore_predictions(predictions, metrics, basis="baseline_labels")

    harsher_predictions, harsher_metrics = build_group(
        b2_families=(4, 5, 6, 7), seen=(0, 1)
    )
    after = rescore_predictions(
        harsher_predictions, harsher_metrics, basis="baseline_labels"
    )

    assert set(before["frozen_labels"]) == set(after["frozen_labels"])
    # The edit did change what the cells were scored on, so the invariance above
    # is a property of the rule and not of an inert fixture.
    assert set(before["n_scored"]) != set(after["n_scored"])


def test_frozen_set_is_read_from_the_baseline_window_of_its_own_group():
    """Groups are frozen independently; no support leaks across a cut."""
    from sift.reporting import rescore_predictions

    first_predictions, first_metrics = build_group(tag="a")
    second_predictions, second_metrics = build_group(
        cut=2021, base_families=(0, 1, 2), b2_families=(2, 3), seen=(0, 1), tag="b"
    )
    predictions = pd.concat([first_predictions, second_predictions], ignore_index=True)
    metrics = pd.concat([first_metrics, second_metrics], ignore_index=True)

    scored = rescore_predictions(predictions, metrics, basis="baseline_labels")
    by_cut = scored.groupby("cut")["frozen_labels"].agg(set)
    assert by_cut.loc[CUT] == {tuple(BASE_FAMILIES)}
    assert by_cut.loc[2021] == {(0, 1, 2)}


def test_baseline_cell_is_scored_exactly_as_it_reports_itself(group):
    """The frozen basis must agree with the as-reported basis at v(empty).

    Delta and every Shapley value are differences from the no-controls cell, so
    a basis that moved that cell would rebase the whole game.  The baseline
    window is its own support, so the two bases coincide there.
    """
    from sift.reporting import rescore_predictions

    predictions, metrics = group
    as_reported = rescore_predictions(predictions, metrics, basis="as_reported")
    frozen = rescore_predictions(predictions, metrics, basis="baseline_labels")
    merged = as_reported.merge(frozen, on="fit_id", suffixes=("_ar", "_bl"))
    baseline = merged[merged["config_id_ar"] == 0].iloc[0]
    assert baseline["macro_f1_ar"] == pytest.approx(baseline["macro_f1_bl"])
    assert baseline["k_scored_ar"] == baseline["k_scored_bl"]
    assert baseline["n_scored_ar"] == baseline["n_scored_bl"]


# ---------------------------------------------------------------------------
# Reported coverage
# ---------------------------------------------------------------------------


def test_coverage_is_reported_per_cell(scored):
    """``n_scored`` and ``n_window`` make coverage visible rather than inferred."""
    for column in ("n_scored", "n_window", "k_present", "n_frozen_labels"):
        assert column in scored.columns

    by_config = scored.set_index("config_id")
    # Controls off: the whole baseline window is in support.
    assert by_config.loc[0, "n_window"] == len(BASE_FAMILIES) * PER_FAMILY
    assert by_config.loc[0, "n_scored"] == by_config.loc[0, "n_window"]
    # B2 on: two of its five families lie outside the frozen support and are
    # dropped, which is exactly the coverage cost the column exists to show.
    b2_only = int(ControlFlags(b2_axis=True).to_index())
    assert by_config.loc[b2_only, "n_window"] == len(B2_FAMILIES) * PER_FAMILY
    assert by_config.loc[b2_only, "n_scored"] == 3 * PER_FAMILY
    # A2 on: the window is already inside the support, so nothing further drops.
    a2_only = int(ControlFlags(a2_labels=True).to_index())
    assert by_config.loc[a2_only, "n_scored"] == by_config.loc[a2_only, "n_window"]
    assert by_config.loc[a2_only, "n_scored"] == 4 * PER_FAMILY


def test_scored_samples_are_exactly_the_in_support_samples_of_each_own_window(
    group, scored
):
    predictions, metrics = group
    frozen = set(scored["frozen_labels"].iloc[0])
    for row in scored.itertuples():
        own = predictions[predictions["fit_id"] == row.fit_id]
        assert row.n_window == len(own)
        assert row.n_scored == int(own["y_true"].isin(frozen).sum())


def test_a_cell_scored_below_the_floor_is_flagged_and_left_unscored():
    """The feasibility floor applies to the samples the cell actually keeps.

    Here B2 selects a window that overlaps the frozen support in one family of
    ten samples.  The cell is not scored, but its coverage is still reported, so
    the reader sees why it dropped out instead of finding a hole.
    """
    from sift.reporting import MIN_COMMON_SAMPLES, rescore_predictions

    predictions, metrics = build_group(
        base_families=(0, 1, 2),
        b2_families=(2, 3),
        seen=(0, 1, 2, 3),
        per_family=MIN_COMMON_SAMPLES // 2,
    )
    scored = rescore_predictions(predictions, metrics, basis="baseline_labels")
    assert len(scored) == 16

    starved = scored[scored["b2_axis"]]
    fed = scored[~scored["b2_axis"]]
    assert not starved["baseline_feasible"].any()
    assert fed["baseline_feasible"].all()
    assert set(starved["n_scored"]) == {MIN_COMMON_SAMPLES // 2}
    assert set(starved["k_scored"]) == {0}
    assert starved["macro_f1"].isna().all()
    assert fed["macro_f1"].notna().all()


# ---------------------------------------------------------------------------
# Refusals and the untouched older bases
# ---------------------------------------------------------------------------


def test_a_group_without_its_baseline_cell_is_refused(group):
    from sift.reporting import rescore_predictions

    predictions, metrics = group
    metrics = metrics[metrics["config_id"] != 0]
    predictions = predictions[predictions["fit_id"] != "g_0"]
    with pytest.raises(ValueError, match="no configuration 0"):
        rescore_predictions(predictions, metrics, basis="baseline_labels")


def test_unknown_basis_names_the_three_that_exist(group):
    from sift.reporting import rescore_predictions

    predictions, metrics = group
    with pytest.raises(ValueError, match="baseline_labels"):
        rescore_predictions(predictions, metrics, basis="frozen")


def test_the_three_bases_are_advertised():
    from sift.reporting import BASELINE_LABELS_BASIS, BASES

    assert BASES == ("as_reported", "common", "baseline_labels")
    assert BASELINE_LABELS_BASIS in BASES


def test_common_basis_still_intersects_and_still_sits_inside_the_a2_window(group):
    """Pin the old behaviour, both that it works and why it cannot check A2.

    The common basis is kept so the published numbers stay reproducible.  This
    test records what it does: the frozen set it uses is contained in the A2-on
    window, so A2 has already been applied to it before phi_A2 is measured.
    """
    from sift.reporting import rescore_predictions

    predictions, metrics = group
    common = rescore_predictions(predictions, metrics, basis="common")
    assert set(common["n_common"]) == {PER_FAMILY}
    assert "frozen_labels" not in common.columns

    a2_only = int(ControlFlags(a2_labels=True, b2_axis=True).to_index())
    shared_families = set(
        predictions[predictions["fit_id"] == f"g_{a2_only}"]["y_true"]
    )
    # The whole intersection is what survives A2 on the B2 window: one family.
    assert shared_families == {3}
    assert set(common["k_scored"]) == {1}


def test_the_frozen_basis_keeps_a2_measurable_where_the_common_basis_cannot(scored):
    """A2 must still be able to move the value on the new basis.

    Under the common basis every cell is scored on samples A2 keeps, so A2 costs
    nothing there by construction.  Under the frozen support the families A2
    drops are still in the denominator and score zero, so the value moves.
    """
    by_config = scored.set_index("config_id")
    a2_only = int(ControlFlags(a2_labels=True).to_index())
    assert by_config.loc[a2_only, "macro_f1"] != pytest.approx(
        by_config.loc[0, "macro_f1"]
    )
