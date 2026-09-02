"""Tests for the recovery-and-selectivity check.

The claim ``sift.recovery`` exists to support is that a panel built by
:func:`~sift.recovery.build_recovery_panel` contains exactly the confound it
says it contains. Everything downstream -- the fits, the Shapley values, the
bootstrap -- is worthless if that construction claim is false, so most of what
follows asserts the construction rather than the result.

The suite deliberately never runs the 96-fit lattice. It exercises the panel
builders on a synthetic panel, the measurement functions against hand-computable
answers, and the value-function and bootstrap plumbing against games whose
Shapley values are known in closed form.
"""

from __future__ import annotations

import numpy
import pandas
import pytest

from sift.config import CONTROL_NAMES, ControlFlags, PanelSpec
from sift.controls import match_class_prior
from sift.experiment import LATTICE_DESIGNS, N_COALITIONS, label_categories
from sift.features import feature_columns
from sift.recovery import (
    A1_ALIGNMENTS,
    DECOY_PREFIX,
    MAGNITUDE_KIND,
    PANEL_INJECTED,
    PANEL_NAMES,
    InjectionSpec,
    _values_from_scores,
    _skewed_distribution,
    arm_prior_gap,
    build_recovery_panel,
    confound_diagnostics,
    null_panel,
    per_class_dispersion,
    recovery_cache,
    selected_decoy_share,
    rebuild_test_weights,
    write_recovery,
)
from sift.shapley import exact_shapley
from sift.splits import temporal_split

TARGET = "ransomware_family"

#: Below this, a measured confound counts as absent. The construction rounds
#: per-family row counts to integers, so an uninjected confound lands near but
#: not exactly on zero.
ABSENT = 0.02


@pytest.fixture(scope="module")
def family_panel():
    """A ransomware-only panel with enough rows per family to be dealt three ways."""
    from sift.mock import synthetic_panel

    panel = synthetic_panel(n_samples=900, n_features=60, n_families=10, seed=7)
    panel = panel[panel["sample_type"] == 1].reset_index(drop=True)
    return panel


@pytest.fixture(scope="module")
def panel_spec():
    return PanelSpec(task="family")


@pytest.fixture(scope="module")
def spec(family_panel):
    """An injection spec scaled to the synthetic panel, not to MLRan."""
    total = len(family_panel)
    return InjectionSpec(
        cut=2018,
        test_window=3,
        n_train=int(round(total * 0.36)),
        n_test=int(round(total * 0.35)),
        a2_families=2,
        b1_decoys=8,
        n_features=20,
        seeds=(0,),
        n_boot=16,
    )


@pytest.fixture(scope="module")
def panels(family_panel, panel_spec, spec):
    return {
        name: build_recovery_panel(family_panel, panel_spec, name, spec, seed=0)
        for name in PANEL_NAMES
    }


# ---------------------------------------------------------------------------
# The null panel really is null
# ---------------------------------------------------------------------------


def test_null_panel_leaves_every_column_but_the_two_axes_alone(
    family_panel, panel_spec, spec
):
    built = null_panel(family_panel, panel_spec, spec, seed=0)
    axes = {panel_spec.primary_axis, panel_spec.secondary_axis}
    assert list(built.columns) == list(family_panel.columns)
    for column in built.columns:
        if column in axes:
            continue
        pandas.testing.assert_series_equal(built[column], family_panel[column])


def test_null_panel_does_not_mutate_its_input(family_panel, panel_spec, spec):
    before = family_panel.copy()
    null_panel(family_panel, panel_spec, spec, seed=0)
    pandas.testing.assert_frame_equal(family_panel, before)


def test_null_panel_axes_agree_everywhere(family_panel, panel_spec, spec):
    built = null_panel(family_panel, panel_spec, spec, seed=0)
    assert (built[panel_spec.primary_axis] == built[panel_spec.secondary_axis]).all()


def test_null_panel_year_carries_no_information_about_the_label(
    family_panel, panel_spec, spec
):
    """Every family is dealt into the three zones in the same proportions.

    This is the whole point of the null panel: if year still predicted family,
    an injected confound could not be told apart from that leftover signal.
    """
    built = null_panel(family_panel, panel_spec, spec, seed=0)
    train_idx, test_idx = temporal_split(
        built, spec.split_spec("temporal"), panel_spec.primary_axis
    )
    corpus = built[TARGET].value_counts(normalize=True)
    for index in (train_idx, test_idx):
        observed = built.loc[index, TARGET].value_counts(normalize=True)
        assert 0.5 * (observed - corpus).abs().sum() < ABSENT


def test_null_panel_is_deterministic(family_panel, panel_spec, spec):
    first = null_panel(family_panel, panel_spec, spec, seed=3)
    second = null_panel(family_panel, panel_spec, spec, seed=3)
    pandas.testing.assert_frame_equal(first, second)
    other = null_panel(family_panel, panel_spec, spec, seed=4)
    assert not other[panel_spec.primary_axis].equals(first[panel_spec.primary_axis])


def test_null_panel_zone_sizes_hit_the_requested_targets(
    family_panel, panel_spec, spec
):
    built = null_panel(family_panel, panel_spec, spec, seed=0)
    train_idx, test_idx = temporal_split(
        built, spec.split_spec("temporal"), panel_spec.primary_axis
    )
    # Per-family rounding is the only source of slack, so one row per family.
    slack = family_panel[TARGET].nunique()
    assert abs(len(train_idx) - spec.n_train) <= slack
    assert abs(len(test_idx) - spec.n_test) <= slack
    assert len(train_idx) + len(test_idx) < len(built)


# ---------------------------------------------------------------------------
# Each injection injects exactly one thing
# ---------------------------------------------------------------------------


def test_every_panel_name_declares_its_ground_truth():
    assert set(PANEL_INJECTED) == set(PANEL_NAMES)
    for injected in PANEL_INJECTED.values():
        assert set(injected) <= set(CONTROL_NAMES)


def test_unknown_panel_name_is_refused(family_panel, panel_spec, spec):
    with pytest.raises(ValueError, match="unknown recovery panel"):
        build_recovery_panel(family_panel, panel_spec, "a3_only", spec, seed=0)


def test_null_panel_measures_all_four_confounds_at_zero(panels):
    magnitude = panels["null"].magnitude
    assert set(magnitude) == set(CONTROL_NAMES)
    for control, value in magnitude.items():
        assert value < ABSENT, f"{control} is not absent from the null panel: {value}"


@pytest.mark.parametrize(
    "name, control",
    [
        ("a1_only", "a1_prior"),
        ("a2_only", "a2_labels"),
        ("b1_only", "b1_fs"),
        ("b2_only", "b2_axis"),
    ],
)
def test_single_confound_panel_injects_its_own_control(panels, name, control):
    assert panels[name].injected == (control,)
    assert panels[name].magnitude[control] > 10 * ABSENT


@pytest.mark.parametrize(
    "name, absent",
    [
        ("a1_only", ("a2_labels", "b1_fs", "b2_axis")),
        ("a2_only", ("b1_fs", "b2_axis")),
        ("b1_only", ("a1_prior", "a2_labels", "b2_axis")),
        ("b2_only", ("a1_prior", "a2_labels", "b1_fs")),
    ],
)
def test_single_confound_panel_leaves_the_others_absent(panels, name, absent):
    for control in absent:
        assert panels[name].magnitude[control] < ABSENT, (
            f"{control} leaked into the {name} panel at "
            f"{panels[name].magnitude[control]}"
        )


def test_a2_injection_necessarily_moves_the_window_prior(panels):
    """A2 cannot be injected without moving A1, and the table must show it.

    Dealing whole families into the test window raises their share of that
    window above their corpus share by construction, so the A1 magnitude of an
    A2 panel is not zero and no honest construction can make it zero. The
    coupling is a property of what novelty *is*, not a defect in the deal, so it
    is measured and reported rather than suppressed, and any reading of
    ``phi_a1`` on that panel has to be made against this number.
    """
    assert panels["a2_only"].magnitude["a1_prior"] > ABSENT
    assert panels["a2_b2"].magnitude["a1_prior"] > ABSENT
    assert panels["null"].magnitude["a1_prior"] < ABSENT


def test_b2_injection_skews_the_secondary_training_half_only(panels):
    built = panels["b2_only"]
    assert built.diagnostics["train_prior_tv_secondary"] > 10 * ABSENT
    assert built.diagnostics["train_prior_tv_primary"] < ABSENT
    assert built.diagnostics["window_prior_tv_secondary"] < ABSENT
    assert built.diagnostics["unseen_share_secondary"] < ABSENT


def test_b2_injection_keeps_every_family_in_the_training_half(panels, panel_spec, spec):
    """Otherwise the B2 panel would also carry novelty and selectivity would fail."""
    built = panels["b2_only"].panel
    train_idx, _ = temporal_split(
        built, spec.split_spec("temporal"), panel_spec.secondary_axis
    )
    assert set(built.loc[train_idx, TARGET].unique()) == set(built[TARGET].unique())


def test_a2_b2_panel_carries_both_and_hides_novelty_on_the_wrong_axis(panels):
    """The interaction panel's whole point: the axis decides whether novelty exists."""
    built = panels["a2_b2"]
    assert built.injected == ("a2_labels", "b2_axis")
    assert built.magnitude["a2_labels"] > 10 * ABSENT
    assert built.magnitude["b2_axis"] > 10 * ABSENT
    assert built.diagnostics["unseen_share_secondary"] < ABSENT


def test_a2_b2_panel_matches_the_single_confound_magnitudes(panels):
    """The interaction panel must inject at the same strength, or it is a third panel."""
    assert panels["a2_b2"].magnitude["a2_labels"] == pytest.approx(
        panels["a2_only"].magnitude["a2_labels"], abs=0.05
    )
    assert panels["a2_b2"].magnitude["b2_axis"] == pytest.approx(
        panels["b2_only"].magnitude["b2_axis"], abs=0.05
    )


def test_a2_injection_holds_out_exactly_k_families(panels, panel_spec, spec):
    built = panels["a2_only"].panel
    train_idx, test_idx = temporal_split(
        built, spec.split_spec("temporal"), panel_spec.primary_axis
    )
    seen = set(built.loc[train_idx, TARGET].unique())
    unseen = set(built.loc[test_idx, TARGET].unique()) - seen
    assert len(unseen) == spec.a2_families
    assert panels["a2_only"].diagnostics["n_held_out_families"] == spec.a2_families


def test_build_is_deterministic_for_every_panel(family_panel, panel_spec, spec, panels):
    for name in PANEL_NAMES:
        again = build_recovery_panel(family_panel, panel_spec, name, spec, seed=0)
        pandas.testing.assert_frame_equal(again.panel, panels[name].panel)
        assert again.magnitude == panels[name].magnitude


def test_build_does_not_mutate_its_input(family_panel, panel_spec, spec):
    before = family_panel.copy()
    for name in PANEL_NAMES:
        build_recovery_panel(family_panel, panel_spec, name, spec, seed=0)
    pandas.testing.assert_frame_equal(family_panel, before)


# ---------------------------------------------------------------------------
# The B1 decoys
# ---------------------------------------------------------------------------


def test_only_the_b1_panel_carries_decoy_columns(panels):
    for name, built in panels.items():
        decoys = [c for c in built.panel.columns if str(c).startswith(DECOY_PREFIX)]
        assert bool(decoys) is (name == "b1_only")


def test_decoys_are_counted_as_features(panels, spec):
    built = panels["b1_only"].panel
    columns = feature_columns(built)
    assert sum(str(c).startswith(DECOY_PREFIX) for c in columns) == spec.b1_decoys


def test_decoys_are_bought_by_leaky_selection_and_rejected_by_honest_selection(
    panels, panel_spec, spec
):
    """The mechanism the B1 injection rests on, asserted rather than assumed."""
    share = selected_decoy_share(panels["b1_only"].panel, panel_spec, spec, seed=0)
    assert share["leaky"] > 0.1
    assert share["honest"] < 0.5 * share["leaky"]


def test_decoys_are_noise_outside_the_test_window(panels, panel_spec, spec):
    """A decoy must carry no signal on the temporal training half, or B1-on fails too."""
    built = panels["b1_only"].panel
    train_idx, _ = temporal_split(
        built, spec.split_spec("temporal"), panel_spec.primary_axis
    )
    train = built.loc[train_idx]
    decoys = [c for c in built.columns if str(c).startswith(DECOY_PREFIX)]
    # Within the training half a decoy is non-zero at roughly the noise rate and
    # its value is independent of the family, so no family may own a level.
    for column in decoys[:3]:
        table = pandas.crosstab(train[TARGET], train[column], normalize="index")
        zero_level = table.get(0)
        assert zero_level is not None
        assert zero_level.min() > 0.5


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def test_every_control_declares_what_its_magnitude_means():
    assert set(MAGNITUDE_KIND) == set(CONTROL_NAMES)


def test_diagnostics_report_the_realised_half_sizes(panels, panel_spec, spec):
    for built in panels.values():
        train_idx, test_idx = temporal_split(
            built.panel, spec.split_spec("temporal"), panel_spec.primary_axis
        )
        assert built.diagnostics["n_train"] == len(train_idx)
        assert built.diagnostics["n_test"] == len(test_idx)


def test_confound_diagnostics_reads_a_hand_built_novelty(family_panel, panel_spec, spec):
    """A panel with one family placed only after the cut must measure exactly that."""
    built = null_panel(family_panel, panel_spec, spec, seed=0)
    victim = sorted(built[TARGET].unique())[0]
    mask = built[TARGET] == victim
    built.loc[mask, panel_spec.primary_axis] = spec.cut
    built.loc[mask, panel_spec.secondary_axis] = spec.cut

    magnitude, _ = confound_diagnostics(built, panel_spec, spec)
    _, test_idx = temporal_split(
        built, spec.split_spec("temporal"), panel_spec.primary_axis
    )
    expected = int(mask.sum()) / len(test_idx)
    assert magnitude["a2_labels"] == pytest.approx(expected)
    assert magnitude["b2_axis"] == 0.0


# ---------------------------------------------------------------------------
# Weight reconstruction
# ---------------------------------------------------------------------------


def test_reconstructed_weights_equal_the_controls_own_weights(family_panel, panel_spec):
    """The bootstrap must resample the estimand the point estimate reported."""
    panel = family_panel
    categories = label_categories(panel, TARGET)
    window = panel.iloc[:150]
    encoded = pandas.Categorical(window[TARGET], categories=categories).codes

    reference = panel[TARGET].value_counts(normalize=True)
    _, expected = match_class_prior(window, window, TARGET, True, panel_spec, reference)
    rebuilt = rebuild_test_weights(encoded, panel, panel_spec, True)
    numpy.testing.assert_allclose(rebuilt, expected)


def test_reconstructed_weights_are_none_when_a1_is_off(family_panel, panel_spec):
    encoded = numpy.zeros(10, dtype=numpy.int16)
    assert rebuild_test_weights(encoded, family_panel, panel_spec, False) is None


def test_reconstructed_weights_average_to_one(family_panel, panel_spec):
    categories = label_categories(family_panel, TARGET)
    encoded = pandas.Categorical(
        family_panel[TARGET].iloc[:200], categories=categories
    ).codes
    weights = rebuild_test_weights(encoded, family_panel, panel_spec, True)
    assert weights.mean() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Value function and attribution
# ---------------------------------------------------------------------------


def _scores_where_only(control: str, size: float) -> dict[tuple[str, int], float]:
    """Build a 32-cell score map in which exactly one control closes the gap."""
    temporal, reference = LATTICE_DESIGNS
    scores: dict[tuple[str, int], float] = {}
    for config_id in range(N_COALITIONS):
        active = ControlFlags.from_index(config_id).active()
        scores[(reference, config_id)] = 0.8
        scores[(temporal, config_id)] = 0.8 - (0.0 if control in active else size)
    return scores


def test_value_function_is_zero_on_the_empty_coalition():
    values, delta = _values_from_scores(_scores_where_only("a2_labels", 0.3))
    assert values[frozenset()] == 0.0
    assert delta == pytest.approx(0.3)


def test_value_function_has_all_sixteen_coalitions():
    values, _ = _values_from_scores(_scores_where_only("b2_axis", 0.1))
    assert len(values) == N_COALITIONS


@pytest.mark.parametrize("control", CONTROL_NAMES)
def test_attribution_lands_entirely_on_the_control_that_moves_the_gap(control):
    """The end-to-end claim, on a game where the answer is known exactly.

    If a single control accounts for the whole gap, its Shapley value must be the
    whole gap and the other three must be exactly zero. A decomposition that
    failed this could not recover anything from a real panel either.
    """
    values, _ = _values_from_scores(_scores_where_only(control, 0.25))
    phi = exact_shapley(values)
    assert phi[control] == pytest.approx(0.25)
    for other in CONTROL_NAMES:
        if other != control:
            assert phi[other] == pytest.approx(0.0, abs=1e-12)


def test_value_function_matches_gap_values_on_the_same_numbers():
    """``_values_from_scores`` is a fast path and must agree with the public one."""
    from sift.experiment import LATTICE_ROLE
    from sift.shapley import gap_values

    rng = numpy.random.default_rng(0)
    temporal = {i: float(v) for i, v in enumerate(rng.uniform(0.3, 0.7, N_COALITIONS))}
    reference = {i: float(v) for i, v in enumerate(rng.uniform(0.6, 0.9, N_COALITIONS))}
    rows = []
    for config_id in range(N_COALITIONS):
        flags = ControlFlags.from_index(config_id)
        for design, table in (
            (LATTICE_DESIGNS[0], temporal),
            (LATTICE_DESIGNS[1], reference),
        ):
            rows.append(
                {
                    "design": design,
                    "role": LATTICE_ROLE,
                    "config_id": config_id,
                    "model": "logreg",
                    "cut": 2019,
                    "seed": 0,
                    "macro_f1": table[config_id],
                    **{name: bool(getattr(flags, name)) for name in CONTROL_NAMES},
                }
            )
    metrics = pandas.DataFrame(rows)

    expected = gap_values(metrics, "logreg", 2019)
    scores = {
        (design, config_id): (reference if design == LATTICE_DESIGNS[1] else temporal)[
            config_id
        ]
        for design in LATTICE_DESIGNS
        for config_id in range(N_COALITIONS)
    }
    observed, _ = _values_from_scores(scores)
    assert set(observed) == set(expected)
    for coalition, value in expected.items():
        assert observed[coalition] == pytest.approx(value)


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------


def test_recovery_cache_restores_the_previous_directory(tmp_path):
    import sift.cache as cache_module

    before = cache_module.CACHE_DIR
    with recovery_cache(tmp_path / "cache") as directory:
        assert cache_module.CACHE_DIR == directory
        assert directory.is_dir()
    assert cache_module.CACHE_DIR == before


def test_recovery_cache_restores_on_failure(tmp_path):
    import sift.cache as cache_module

    before = cache_module.CACHE_DIR
    with pytest.raises(RuntimeError):
        with recovery_cache(tmp_path / "cache"):
            raise RuntimeError("boom")
    assert cache_module.CACHE_DIR == before


def test_write_recovery_round_trips(tmp_path):
    frame = pandas.DataFrame(
        {
            "panel": ["null", "a1_only"],
            "control": ["a1_prior", "a1_prior"],
            "injected": [False, True],
            "injected_magnitude": [0.004, 0.244],
            "phi": [-0.001, 0.081],
            "phi_lo": [-0.02, 0.03],
            "phi_hi": [0.02, 0.13],
        }
    )
    written = write_recovery({"recovery": frame}, tmp_path)
    assert set(written) == {"recovery"}
    back = pandas.read_parquet(written["recovery"])
    pandas.testing.assert_frame_equal(back, frame)


def test_injection_spec_refuses_a_skew_that_injects_nothing():
    with pytest.raises(ValueError, match="skew"):
        InjectionSpec(a1_skew=1.0)


def test_injection_spec_refuses_an_unknown_a1_alignment():
    with pytest.raises(ValueError, match="a1_align"):
        InjectionSpec(a1_align="difficulty")


# ---------------------------------------------------------------------------
# Where the A1 skew is placed
# ---------------------------------------------------------------------------


@pytest.fixture()
def toy_prior():
    """Four families whose sizes are strictly ordered against their names.

    Alphabetical order boosts ``a`` and ``c``; size order boosts ``c`` and ``d``.
    The two alignments therefore disagree on half the families, which is what
    makes the pair a test rather than a coincidence.
    """
    return pandas.Series({"a": 0.4, "b": 0.3, "c": 0.2, "d": 0.1})


@pytest.mark.parametrize("align", A1_ALIGNMENTS)
def test_skewed_distribution_is_a_distribution(toy_prior, align):
    skewed = _skewed_distribution(toy_prior, 3.0, align)
    assert skewed.sum() == pytest.approx(1.0)
    assert list(skewed.index) == sorted(toy_prior.index)
    assert (skewed > 0).all()


def test_name_alignment_boosts_every_second_family(toy_prior):
    skewed = _skewed_distribution(toy_prior, 3.0, "name")
    boosted = {name for name in toy_prior.index if skewed[name] > toy_prior[name]}
    assert boosted == {"a", "c"}


def test_size_alignment_boosts_the_smaller_half(toy_prior):
    """The half that carries the least training support, hence the hardest half."""
    skewed = _skewed_distribution(toy_prior, 3.0, "size")
    boosted = {name for name in toy_prior.index if skewed[name] > toy_prior[name]}
    assert boosted == {"c", "d"}


def test_both_alignments_can_reach_the_same_distance_from_the_prior(toy_prior):
    """Total variation does not distinguish them; only where the mass lands does.

    This is the whole reason ``a1_align`` exists. The first A1 recovery attempt
    injected a window-versus-corpus distance of 0.244 and recovered nothing,
    because control A1 leaves per-class recall exactly invariant and can only
    move a class-averaged metric through the covariance between the skew and
    per-class difficulty. Distance alone does not bound that covariance.
    """
    by_name = _skewed_distribution(toy_prior, 3.0, "name")
    by_size = _skewed_distribution(toy_prior, 3.0, "size")
    distance = lambda other: float(0.5 * (other - toy_prior).abs().sum())
    assert distance(by_name) == pytest.approx(distance(by_size), abs=0.1)
    assert not by_name.equals(by_size)


def test_a1_alignment_reaches_the_realised_panel(family_panel, panel_spec, spec):
    """The parameter must change the panel, not merely the target it aims at."""
    import dataclasses

    by_name = build_recovery_panel(
        family_panel, panel_spec, "a1_only", dataclasses.replace(spec, a1_align="name"), 0
    )
    by_size = build_recovery_panel(
        family_panel, panel_spec, "a1_only", dataclasses.replace(spec, a1_align="size"), 0
    )
    train_name, test_name = temporal_split(
        by_name.panel, spec.split_spec("temporal"), panel_spec.primary_axis
    )
    train_size, test_size = temporal_split(
        by_size.panel, spec.split_spec("temporal"), panel_spec.primary_axis
    )
    del train_name, train_size
    assert by_name.magnitude["a1_prior"] > ABSENT
    assert by_size.magnitude["a1_prior"] > ABSENT
    name_window = by_name.panel.loc[test_name, TARGET].value_counts(normalize=True)
    size_window = by_size.panel.loc[test_size, TARGET].value_counts(normalize=True)
    assert not numpy.allclose(
        name_window.sort_index().to_numpy(), size_window.reindex(name_window.index).fillna(0.0).to_numpy()
    )


# ---------------------------------------------------------------------------
# The two diagnostics
# ---------------------------------------------------------------------------


def test_arm_prior_gap_reports_the_distance_the_game_actually_sees(
    panels, panel_spec, spec
):
    """A1's magnitude is a window-versus-corpus distance; the game sees an
    arm-versus-arm one, and the two need not agree."""
    gap = arm_prior_gap(panels["a1_only"].panel, panel_spec, spec)
    assert set(gap) == {
        "tv_window_vs_corpus",
        "tv_reference_vs_corpus",
        "tv_arm_to_arm",
        "tv_arm_to_arm_sd",
    }
    assert gap["tv_window_vs_corpus"] == pytest.approx(
        panels["a1_only"].magnitude["a1_prior"]
    )
    for value in gap.values():
        assert 0.0 <= value <= 1.0


def test_arm_prior_gap_is_small_on_the_null_panel(panels, panel_spec, spec):
    """Neither arm is skewed there, so the contrast A1 could act on is absent."""
    gap = arm_prior_gap(panels["null"].panel, panel_spec, spec)
    assert gap["tv_arm_to_arm"] < 0.15


def test_per_class_dispersion_is_zero_when_every_class_is_perfect():
    y = numpy.array([0, 0, 1, 1, 2, 2])
    out = per_class_dispersion(y, y)
    assert out["k_present"] == 3.0
    assert out["f1_sd"] == pytest.approx(0.0)
    assert out["f1_min"] == pytest.approx(1.0)
    assert out["recall_mean"] == pytest.approx(1.0)


def test_per_class_dispersion_reports_the_spread_it_is_given():
    """Class 0 perfect, class 1 never recalled: recall runs the whole range."""
    y_true = numpy.array([0, 0, 1, 1])
    y_pred = numpy.array([0, 0, 0, 0])
    out = per_class_dispersion(y_true, y_pred)
    assert out["recall_min"] == pytest.approx(0.0)
    assert out["recall_max"] == pytest.approx(1.0)
    assert out["recall_sd"] == pytest.approx(0.5)
    assert out["k_present"] == 2.0


def test_injection_spec_refuses_a_binary_decoy_level_below_two():
    with pytest.raises(ValueError, match="b1_decoy_levels"):
        InjectionSpec(b1_decoy_levels=1)


def test_injection_spec_refuses_an_empty_seed_list():
    with pytest.raises(ValueError, match="seed"):
        InjectionSpec(seeds=())


def test_a2_injection_refuses_a_holdout_that_cannot_fit(family_panel, panel_spec):
    """A silently shrunk holdout would report a magnitude it did not inject."""
    greedy = InjectionSpec(a2_families=family_panel[TARGET].nunique(), n_test=50)
    with pytest.raises(ValueError):
        build_recovery_panel(family_panel, panel_spec, "a2_only", greedy, seed=0)


# ---------------------------------------------------------------------------
# End to end on one real fit
# ---------------------------------------------------------------------------


def test_rebuilt_weights_reproduce_the_recorded_score(family_panel, panel_spec, spec, tmp_path):
    """Recompute an A1-enabled cell's score from its stored predictions alone.

    The bootstrap never sees the weight vector the fit was scored with; it
    rebuilds one. If that reconstruction were wrong the interval would describe a
    different quantity from the point estimate it is printed beside, and nothing
    downstream would notice. So the reconstruction is checked the only way that
    settles it: run a cell with A1 on, then reproduce its recorded ``macro_f1``
    from the stored predictions and the rebuilt weights.
    """
    from sift.config import ExperimentConfig
    from sift.metrics import compute_metrics
    from sift.recovery import _config, recovery_cache

    built = null_panel(family_panel, panel_spec, spec, seed=0)
    cfg = ExperimentConfig(
        panel=panel_spec,
        split=spec.split_spec("temporal"),
        flags=ControlFlags(a1_prior=True),
        model_name="logreg",
        seed=0,
        target=TARGET,
        n_features=spec.n_features,
    )
    assert cfg == _config(panel_spec, spec, "temporal", cfg.flags.to_index(), 0)

    from sift.experiment import run_cell

    with recovery_cache(tmp_path / "cache"):
        record = run_cell(built, cfg)

    stored = record["predictions"]
    weights = rebuild_test_weights(stored["y_true"], built, panel_spec, True)
    rescored = compute_metrics(stored["y_true"], stored["y_pred"], sample_weight=weights)
    assert rescored["macro_f1"] == pytest.approx(record["macro_f1"])

    unweighted = compute_metrics(stored["y_true"], stored["y_pred"])
    assert unweighted["macro_f1"] != pytest.approx(record["macro_f1"])
