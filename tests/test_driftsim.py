"""Tests for the synthetic concept-drift generator.

The module under test exists to give the residual ``R`` a ground truth, so the
suite is organised around the two properties that make the ground truth
trustworthy:

1. the injected drift is what the specification says it is -- a rotation of the
   weight vector by ``level * pi / 2`` at constant norm, with no covariate shift
   riding along; and
2. the four evaluation controls are null on the generated panel, so that any
   Shapley value the decomposition reports is a false positive rather than a
   confound the generator quietly planted.

Everything except the two integration tests runs on explicitly supplied Bernoulli
parameters, so the file does not need the MLRan release.
"""

from __future__ import annotations

import math

import numpy
import pytest

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

#: Bernoulli parameters standing in for the real MLRan marginals: right-skewed
#: and sparse, like the release, but fixed so the tests do not need the data.
FAKE_MARGINALS = numpy.concatenate(
    [
        numpy.linspace(0.01, 0.15, 40),
        numpy.linspace(0.15, 0.45, 20),
        numpy.linspace(0.45, 0.85, 8),
    ]
)


@pytest.fixture()
def driftsim():
    """The module under test, imported lazily as the suite requires."""
    import sift.driftsim as module

    return module


@pytest.fixture()
def small_spec(driftsim):
    """A specification small enough to fit sixteen coalitions in a second."""
    return driftsim.DriftSpec(
        drift_level=0.0,
        n_features=12,
        samples_per_year=40,
        year_min=2012,
        year_max=2019,
        cut_year=2016,
        test_window=2,
        seed=7,
    )


@pytest.fixture()
def small_probabilities(driftsim, small_spec):
    """Bernoulli parameters for ``small_spec``, drawn without touching MLRan."""
    return driftsim.feature_probabilities(small_spec, FAKE_MARGINALS)


# --------------------------------------------------------------------------
# The rotation is the drift definition
# --------------------------------------------------------------------------
def test_drift_level_is_the_angle_over_a_right_angle(driftsim):
    for level in (0.0, 0.25, 0.5, 0.75, 1.0):
        spec = driftsim.DriftSpec(drift_level=level)
        assert spec.theta == pytest.approx(level * math.pi / 2.0)
        assert driftsim.RIGHT_ANGLE == pytest.approx(math.pi / 2.0)


@pytest.mark.parametrize("level", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_rotation_holds_the_norm_of_w_constant(driftsim, level):
    """Sharpness must not move with the drift level, only direction."""
    spec = driftsim.DriftSpec(drift_level=level, n_features=64, signal_sd=2.5)
    w_early, w_late = driftsim.drift_weights(spec)
    assert numpy.linalg.norm(w_early) == pytest.approx(2.5, abs=1e-12)
    assert numpy.linalg.norm(w_late) == pytest.approx(2.5, abs=1e-12)


@pytest.mark.parametrize("level", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_rotation_turns_w_by_exactly_theta(driftsim, level):
    spec = driftsim.DriftSpec(drift_level=level, n_features=64)
    w_early, w_late = driftsim.drift_weights(spec)
    cosine = float(w_early @ w_late) / float(
        numpy.linalg.norm(w_early) * numpy.linalg.norm(w_late)
    )
    assert cosine == pytest.approx(math.cos(spec.theta), abs=1e-12)


def test_level_zero_leaves_the_rule_untouched(driftsim):
    spec = driftsim.DriftSpec(drift_level=0.0, n_features=64)
    w_early, w_late = driftsim.drift_weights(spec)
    numpy.testing.assert_allclose(w_early, w_late, atol=1e-12)
    assert driftsim.rule_disagreement(spec, FAKE_MARGINALS[:64], n_draw=2000) == 0.0


def test_level_one_replaces_the_rule(driftsim):
    """A full right angle means the two rules share no direction at all."""
    spec = driftsim.DriftSpec(drift_level=1.0, n_features=64)
    w_early, w_late = driftsim.drift_weights(spec)
    assert float(w_early @ w_late) == pytest.approx(0.0, abs=1e-12)


def test_rule_disagreement_increases_with_the_level(driftsim):
    probabilities = FAKE_MARGINALS[:64]
    flips = [
        driftsim.rule_disagreement(
            driftsim.DriftSpec(drift_level=level, n_features=64),
            probabilities,
            n_draw=20_000,
        )
        for level in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]
    assert flips[0] == 0.0
    assert all(later > earlier for earlier, later in zip(flips, flips[1:]))
    # A right angle in standardised coordinates makes the two rules independent.
    assert flips[-1] == pytest.approx(0.5, abs=0.02)


def test_rotation_basis_is_orthonormal(driftsim):
    u, v = driftsim.rotation_basis(50, seed=3)
    assert numpy.linalg.norm(u) == pytest.approx(1.0, abs=1e-12)
    assert numpy.linalg.norm(v) == pytest.approx(1.0, abs=1e-12)
    assert float(u @ v) == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------
# The Bernoulli parameters come from the real marginals
# --------------------------------------------------------------------------
def test_feature_probabilities_are_drawn_from_the_supplied_marginals(driftsim):
    spec = driftsim.DriftSpec(n_features=30)
    p = driftsim.feature_probabilities(spec, FAKE_MARGINALS)
    assert p.shape == (30,)
    source = set(numpy.round(numpy.clip(FAKE_MARGINALS, 0.01, 0.99), 12))
    assert set(numpy.round(p, 12)).issubset(source)
    assert p.min() >= 0.01 and p.max() <= 0.99


def test_feature_probabilities_are_deterministic(driftsim):
    spec = driftsim.DriftSpec(n_features=30)
    first = driftsim.feature_probabilities(spec, FAKE_MARGINALS)
    second = driftsim.feature_probabilities(spec, FAKE_MARGINALS)
    numpy.testing.assert_array_equal(first, second)


def test_feature_probabilities_rejects_an_empty_source(driftsim):
    with pytest.raises(ValueError, match="non-empty"):
        driftsim.feature_probabilities(driftsim.DriftSpec(), numpy.array([]))


# --------------------------------------------------------------------------
# The panel
# --------------------------------------------------------------------------
def test_panel_carries_the_columns_the_framework_requires(
    driftsim, small_spec, small_probabilities
):
    from sift.features import feature_columns

    panel = driftsim.generate_panel(small_spec, small_probabilities)
    for column in (
        "sample_id",
        "grp_id",
        "sample_type",
        "ransomware_family",
        "first_submission_date_year",
        "Year",
        "n_active",
    ):
        assert column in panel.columns
    assert len(feature_columns(panel)) == small_spec.n_features
    assert len(panel) == small_spec.n_samples
    assert panel["sample_id"].is_unique
    assert panel["grp_id"].is_unique


def test_base_rate_is_exactly_constant_in_every_year(
    driftsim, small_spec, small_probabilities
):
    """This is what makes control A1 null rather than merely small."""
    for level in (0.0, 0.5, 1.0):
        spec = driftsim.DriftSpec(**{**vars(small_spec), "drift_level": level})
        panel = driftsim.generate_panel(spec, small_probabilities)
        rates = panel.groupby("first_submission_date_year")["sample_type"].mean()
        numpy.testing.assert_array_equal(rates.to_numpy(), numpy.full(rates.size, 0.5))


def test_the_two_temporal_axes_are_identical(driftsim, small_spec, small_probabilities):
    """This is what makes control B2 null."""
    panel = driftsim.generate_panel(small_spec, small_probabilities)
    numpy.testing.assert_array_equal(
        panel["Year"].to_numpy(), panel["first_submission_date_year"].to_numpy()
    )


def test_every_family_appears_in_every_year(driftsim, small_spec, small_probabilities):
    """This is what makes control A2 null."""
    panel = driftsim.generate_panel(small_spec, small_probabilities)
    per_year = panel.groupby("first_submission_date_year")["ransomware_family"].nunique()
    assert set(panel["ransomware_family"]) == {"goodware", "ransomware"}
    assert (per_year == 2).all()


def test_the_feature_budget_covers_every_column(driftsim, small_spec, small_probabilities):
    """This is what makes control B1 null: there is nothing to select."""
    from sift.features import feature_columns, select_features

    panel = driftsim.generate_panel(small_spec, small_probabilities)
    columns = feature_columns(panel)
    cfg = driftsim.base_config(small_spec)
    assert cfg.n_features >= len(columns)
    whole = select_features(panel, columns, "sample_type", cfg.n_features, seed=0)
    half = select_features(panel.iloc[:100], columns, "sample_type", cfg.n_features, seed=0)
    assert whole == half == columns


def test_no_covariate_shift_between_the_two_windows(driftsim):
    """Only P(y | x) moves; P(x) is one distribution for the whole panel."""
    module = driftsim

    spec = module.DriftSpec(drift_level=1.0, n_features=60, samples_per_year=400)
    probabilities = module.feature_probabilities(spec, FAKE_MARGINALS)
    panel = module.generate_panel(spec, probabilities)
    columns = [str(i) for i in range(spec.n_features)]
    early = panel[panel["first_submission_date_year"] < spec.cut_year][columns].mean()
    late = panel[panel["first_submission_date_year"] >= spec.cut_year][columns].mean()
    # Both halves hold 2400 rows, so a Bernoulli marginal has a standard error
    # near 0.01 at worst; 0.05 is five of those and still far below any shift a
    # domain classifier could exploit.
    assert float((early - late).abs().max()) < 0.05
    numpy.testing.assert_allclose(
        panel[columns].mean().to_numpy(), probabilities, atol=0.05
    )


def test_the_label_rule_is_the_only_thing_the_level_changes(driftsim, small_probabilities):
    """Two levels sharing a seed must share their feature matrix exactly."""
    import sift.driftsim as module

    quiet = module.DriftSpec(drift_level=0.0, n_features=12, samples_per_year=40, seed=7)
    loud = module.DriftSpec(drift_level=1.0, n_features=12, samples_per_year=40, seed=7)
    columns = [str(i) for i in range(12)]
    a = module.generate_panel(quiet, small_probabilities)
    b = module.generate_panel(loud, small_probabilities)
    numpy.testing.assert_array_equal(a[columns].to_numpy(), b[columns].to_numpy())
    numpy.testing.assert_array_equal(
        a["first_submission_date_year"].to_numpy(), b["first_submission_date_year"].to_numpy()
    )
    # The early window keeps its labels; the late window is relabelled.
    early = a["first_submission_date_year"] < quiet.cut_year
    numpy.testing.assert_array_equal(
        a.loc[early, "sample_type"].to_numpy(), b.loc[early, "sample_type"].to_numpy()
    )
    assert not numpy.array_equal(
        a.loc[~early, "sample_type"].to_numpy(), b.loc[~early, "sample_type"].to_numpy()
    )


def test_generation_is_reproducible(driftsim, small_spec, small_probabilities):
    first = driftsim.generate_panel(small_spec, small_probabilities)
    second = driftsim.generate_panel(small_spec, small_probabilities)
    import pandas

    pandas.testing.assert_frame_equal(first, second)


def test_generate_panel_rejects_a_mismatched_parameter_vector(driftsim, small_spec):
    with pytest.raises(ValueError, match="spec.n_features"):
        driftsim.generate_panel(small_spec, numpy.full(3, 0.2))


# --------------------------------------------------------------------------
# Specification validation
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"drift_level": 1.5}, "drift_level"),
        ({"drift_level": -0.1}, "drift_level"),
        ({"n_features": 1}, "n_features"),
        ({"samples_per_year": 41}, "samples_per_year"),
        ({"year_min": 2020, "year_max": 2015}, "year_min"),
        ({"cut_year": 2011}, "cut_year"),
        ({"cut_year": 2023, "test_window": 3}, "test window"),
        ({"base_rate": 0.0}, "base_rate"),
        ({"signal_sd": 0.0}, "signal_sd"),
        ({"base_rate": 0.333}, "reference prior exactly"),
    ],
)
def test_spec_refuses_a_configuration_that_would_plant_a_confound(
    driftsim, overrides, message
):
    with pytest.raises(ValueError, match=message):
        driftsim.DriftSpec(**overrides)


def test_panel_spec_targets_the_prior_the_data_carries(driftsim, small_spec):
    """A1 can only be null if it aims at the rate the generator injected."""
    spec = driftsim.panel_spec_for(small_spec)
    assert spec.task == "binary"
    assert spec.prior_reference_rate == small_spec.base_rate
    assert (spec.year_min, spec.year_max) == (small_spec.year_min, small_spec.year_max)


# --------------------------------------------------------------------------
# The null-by-construction guarantee, on the realised panel
# --------------------------------------------------------------------------
def test_no_coalition_changes_anything_on_the_generated_panel(
    driftsim, small_spec, small_probabilities
):
    """All sixteen coalitions must produce the identical split and feature set."""
    panel = driftsim.generate_panel(small_spec, small_probabilities)
    report = driftsim.assert_controls_are_null(panel, small_spec, seeds=(0, 1))
    assert report["max_weight_deviation"] == 0.0


def test_the_nullity_check_catches_a_planted_confound(
    driftsim, small_spec, small_probabilities
):
    """A guard that never fires proves nothing, so plant a B2 axis discrepancy."""
    panel = driftsim.generate_panel(small_spec, small_probabilities)
    tampered = panel.copy()
    tampered.loc[tampered.index[::3], "Year"] = small_spec.year_min
    with pytest.raises(AssertionError, match="not confound-free"):
        driftsim.assert_controls_are_null(tampered, small_spec, seeds=(0,))


def test_the_nullity_check_catches_a_planted_prior_shift(
    driftsim, small_spec, small_probabilities
):
    """Move the class prior of one window and control A1 stops being a no-op."""
    panel = driftsim.generate_panel(small_spec, small_probabilities)
    tampered = panel.copy()
    window = tampered["first_submission_date_year"].between(
        small_spec.cut_year, small_spec.cut_year + small_spec.test_window - 1
    )
    flip = tampered.index[window & (tampered["sample_type"] == 1)][:10]
    tampered.loc[flip, "sample_type"] = 0
    tampered.loc[flip, "ransomware_family"] = "goodware"
    with pytest.raises(AssertionError, match="weights deviating from one"):
        driftsim.assert_controls_are_null(tampered, small_spec, seeds=(0,))


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------
def test_one_level_attributes_nothing_to_the_four_controls(
    driftsim, small_spec, small_probabilities, tmp_path
):
    """The headline expectation: on a confound-free panel every phi is zero."""
    spec = driftsim.DriftSpec(**{**vars(small_spec), "drift_level": 1.0})
    with driftsim._cache_dir(tmp_path / "fit_cache"):
        outcome = driftsim.run_level(
            spec, seeds=(0,), n_boot=50, probabilities=small_probabilities
        )
    assert outcome["checks_passed"]
    for control, value in outcome["shapley"].items():
        assert value == pytest.approx(0.0, abs=1e-12), control
    # With every phi zero, efficiency forces the whole gap into the residual.
    assert outcome["residual"] == pytest.approx(outcome["delta"], abs=1e-12)
    low, high = outcome["residual_ci"]
    assert low <= outcome["residual"] <= high


def test_run_ladder_writes_one_row_per_level_and_control(
    driftsim, small_spec, small_probabilities, tmp_path
):
    levels = (0.0, 1.0)
    tables = driftsim.run_ladder(
        levels=levels,
        template=small_spec,
        seeds=(0,),
        n_boot=50,
        results_dir=tmp_path,
        probabilities=small_probabilities,
    )
    from sift.config import CONTROL_NAMES

    assert len(tables["shapley"]) == len(levels) * len(CONTROL_NAMES)
    assert len(tables["residual"]) == len(levels)
    # Two arms, sixteen coalitions, one seed, per level.
    assert len(tables["fits"]) == len(levels) * 2 * 16
    assert set(tables["shapley"]["control"]) == set(CONTROL_NAMES)

    import pandas

    for name in ("shapley", "residual", "fits"):
        path = tmp_path / f"{name}.parquet"
        assert path.is_file()
        pandas.testing.assert_frame_equal(pandas.read_parquet(path), tables[name])

    # The ground truth is recorded next to the recovered quantity.
    residual = tables["residual"].set_index("drift_level")
    assert residual.loc[0.0, "rule_disagreement"] == 0.0
    assert residual.loc[1.0, "rule_disagreement"] > 0.4
    assert (tables["residual"]["max_abs_phi"] < 1e-9).all()


def test_the_ladder_writes_nothing_outside_its_own_directory(
    driftsim, small_spec, small_probabilities, tmp_path
):
    from sift.paths import RESULTS_DIR

    before = sorted(p.name for p in RESULTS_DIR.iterdir()) if RESULTS_DIR.exists() else []
    driftsim.run_ladder(
        levels=(0.0,),
        template=small_spec,
        seeds=(0,),
        n_boot=20,
        results_dir=tmp_path / "out",
        probabilities=small_probabilities,
    )
    after = sorted(p.name for p in RESULTS_DIR.iterdir()) if RESULTS_DIR.exists() else []
    assert before == after
