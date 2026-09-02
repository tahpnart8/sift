"""Importance-weighted risk, the module's only valid test of ``p(y|x)``.

Two families of test live here.

The first pins the *estimator* against cases with an analytic answer. The
density ratio of two unit Gaussians whose means differ by one is
``exp(x - 0.5)`` exactly, so the fitted log-ratio must have slope one and
intercept minus one half. That single test is what fixes the prior correction:
choosing the wrong option rescales every weight by a constant, which moves the
intercept and nothing else, so a test that only checked the *shape* of the
weights would pass under a wrong correction.

The second family is the pair of smoke tests the deliverable asks for, on
synthetic windows where the truth is known by construction.

``test_covariate_shift_alone_is_not_attributed_to_the_conditional``
    Both windows share one curved decision boundary, so ``p(y|x)`` is identical
    by construction. Only ``p(x)`` moves. The model is *linear*, hence
    misspecified, so the optimal linear boundary rotates as the input mass moves
    and the raw degradation is large. The method must report that large
    degradation and still decline to implicate the conditional.

``test_a_changed_decision_rule_is_attributed_to_the_conditional``
    ``p(x)`` is identical in both windows, byte for byte in distribution, and the
    late window labels the same inputs by the opposite rule. Nothing but the
    conditional has moved, so the method must implicate it.

The first of these is the load-bearing one, and it is only informative because
``raw_gap`` is materially positive: a scenario where nothing degrades would pass
vacuously. The assertion that degradation exists is therefore part of the test,
not decoration.

A third case runs both perturbations at once, because distinguishing them when
both are present is the situation the real windows are in.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sift.iwrisk import (
    DEFAULT_MAX_DECISIVE_WIDTH,
    IWRISK_METRICS,
    MIN_INTERPRETABLE_ESS,
    DensityRatioResult,
    IWRiskEstimate,
    IWRiskResult,
    default_risk_model,
    effective_sample_size,
    estimate_density_ratio,
    importance_weighted_risk,
)

# Bootstrap size for the scenario tests. Smaller than the module default of
# 1000 so the file runs in seconds; large enough that a percentile interval is
# stable to the third decimal, which is finer than any threshold asserted here.
N_BOOT = 400

# Analysis seed. Fixed rather than parametrised: the scenarios below assert
# thresholds observed on this seed, and a sweep would turn a fast smoke test
# into an experiment.
SEED = 11


# ---------------------------------------------------------------------------
# Scenario construction
# ---------------------------------------------------------------------------
def make_windows(
    seed: int = 3,
    n_early: int = 800,
    n_late: int = 600,
    shift: float = 1.2,
    curve: float = 2.5,
    sharpness: float = 4.0,
    late_rule: str = "same",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build two windows with a controllable shift and a controllable conditional.

    The true decision boundary is the parabola ``x1 = curve * (x0**2 - 1)``,
    shared by both windows whenever ``late_rule == "same"``. The *optimal linear*
    boundary has slope ``2 * curve * E[x0]``, which rotates as the mean of ``x0``
    moves. That is what lets a pure covariate shift produce a large drop in a
    linear model without any change to ``p(y|x)``.

    Parameters
    ----------
    shift : float
        Separation between the two windows in the first coordinate. Zero gives
        identical input distributions.
    late_rule : {"same", "flipped"}
        ``"same"`` keeps ``p(y|x)`` identical, the covariate-shift case.
        ``"flipped"`` negates the late logit, so the late window labels the same
        inputs by the opposite rule. The inputs are untouched either way: this
        parameter moves ``p(y|x)`` and nothing else.
    """
    rng = np.random.default_rng(seed)
    early = np.column_stack(
        [
            rng.normal(-shift / 2.0, 1.0, n_early),
            rng.normal(0.0, 1.0, n_early),
            rng.normal(0.0, 1.0, n_early),
        ]
    )
    late = np.column_stack(
        [
            rng.normal(+shift / 2.0, 1.0, n_late),
            rng.normal(0.0, 1.0, n_late),
            rng.normal(0.0, 1.0, n_late),
        ]
    )

    def draw(x: np.ndarray, rule: str) -> np.ndarray:
        logit = sharpness * (x[:, 1] - curve * x[:, 0] ** 2 + curve)
        if rule == "flipped":
            logit = -logit
        elif rule != "same":  # pragma: no cover - guarded by the callers here
            raise ValueError(f"unknown rule {rule!r}")
        p = 1.0 / (1.0 + np.exp(-logit))
        return (rng.random(x.shape[0]) < p).astype(int)

    return early, draw(early, "same"), late, draw(late, late_rule)


@pytest.fixture(scope="module")
def covariate_shift_only() -> IWRiskResult:
    """Genuine covariate shift, ``p(y|x)`` identical by construction."""
    x_e, y_e, x_l, y_l = make_windows(late_rule="same")
    return importance_weighted_risk(x_e, y_e, x_l, y_l, seed=SEED, n_boot=N_BOOT)


@pytest.fixture(scope="module")
def changed_rule_only() -> IWRiskResult:
    """Identical input distributions, a genuinely different decision rule."""
    x_e, y_e, x_l, y_l = make_windows(
        n_early=500, n_late=400, shift=0.0, late_rule="flipped"
    )
    return importance_weighted_risk(x_e, y_e, x_l, y_l, seed=SEED, n_boot=N_BOOT)


@pytest.fixture(scope="module")
def both_perturbations() -> IWRiskResult:
    """Covariate shift and a changed conditional at the same time."""
    x_e, y_e, x_l, y_l = make_windows(late_rule="flipped")
    return importance_weighted_risk(x_e, y_e, x_l, y_l, seed=SEED, n_boot=N_BOOT)


# ---------------------------------------------------------------------------
# Effective sample size
# ---------------------------------------------------------------------------
def test_ess_of_uniform_weights_is_the_row_count() -> None:
    """Uniform weights waste nothing, so the effective count is the raw count."""
    assert effective_sample_size(np.ones(50)) == pytest.approx(50.0)
    assert effective_sample_size(np.full(50, 7.3)) == pytest.approx(50.0)


def test_ess_collapses_to_one_when_a_single_row_carries_the_mass() -> None:
    """The degenerate case the diagnostic exists to catch."""
    weights = np.zeros(500)
    weights[0] = 1.0
    assert effective_sample_size(weights) == pytest.approx(1.0)


def test_ess_is_scale_invariant() -> None:
    """Normalising the weights must not move the diagnostic."""
    rng = np.random.default_rng(0)
    w = rng.lognormal(0.0, 1.5, 400)
    assert effective_sample_size(w) == pytest.approx(effective_sample_size(w * 1000.0))


def test_ess_matches_the_hand_computed_formula() -> None:
    """(1+2+3)^2 / (1+4+9) = 36/14."""
    assert effective_sample_size(np.array([1.0, 2.0, 3.0])) == pytest.approx(36.0 / 14.0)


@pytest.mark.parametrize(
    "bad",
    [np.array([]), np.array([1.0, -1.0]), np.zeros(5)],
    ids=["empty", "negative", "all_zero"],
)
def test_ess_rejects_undefined_input(bad: np.ndarray) -> None:
    """No silent nan: an undefined effective size is an error."""
    with pytest.raises(ValueError):
        effective_sample_size(bad)


# ---------------------------------------------------------------------------
# Density ratio, against an analytic answer
# ---------------------------------------------------------------------------
def test_density_ratio_recovers_a_known_gaussian_ratio() -> None:
    """For N(0,1) against N(1,1) the true ratio is exp(x - 0.5), exactly.

    The window sizes are deliberately unequal, 2000 against 800. With equal
    sizes the balanced and sampling prior corrections coincide and the test
    could not tell a wrong correction from a right one. The intercept is the
    quantity that discriminates them: an uncorrected odds ratio from an
    unbalanced fit would sit near ``-0.5 + log(2000/800) = 0.42`` instead.
    """
    rng = np.random.default_rng(7)
    early = rng.normal(0.0, 1.0, size=(2000, 1))
    late = rng.normal(1.0, 1.0, size=(800, 1))

    ratio = estimate_density_ratio(early, late, seed=1, normalise=False)

    design = np.column_stack([np.ones(early.shape[0]), early[:, 0]])
    intercept, slope = np.linalg.lstsq(design, np.log(ratio.weights), rcond=None)[0]

    assert slope == pytest.approx(1.0, abs=0.15)
    assert intercept == pytest.approx(-0.5, abs=0.15)


def test_sampling_correction_rescales_the_ratio_by_the_window_proportions() -> None:
    """``"sampling"`` differs from ``"balanced"`` by exactly n_early / n_late."""
    rng = np.random.default_rng(4)
    early = rng.normal(0.0, 1.0, size=(300, 2))
    late = rng.normal(0.6, 1.0, size=(150, 2))

    balanced = estimate_density_ratio(
        early, late, seed=2, prior_correction="balanced", normalise=False
    )
    sampling = estimate_density_ratio(
        early, late, seed=2, prior_correction="sampling", normalise=False
    )

    assert sampling.prior_correction == pytest.approx(300.0 / 150.0)
    np.testing.assert_allclose(sampling.weights, balanced.weights * 2.0, rtol=1e-10)
    # A constant rescaling cannot change how concentrated the weights are.
    assert sampling.ess == pytest.approx(balanced.ess)


def test_identical_windows_give_near_uniform_weights() -> None:
    """With no shift there is nothing to correct, and the diagnostic says so."""
    rng = np.random.default_rng(5)
    pooled = rng.normal(0.0, 1.0, size=(900, 3))
    ratio = estimate_density_ratio(pooled[:500], pooled[500:], seed=3)

    assert ratio.oof_auc == pytest.approx(0.5, abs=0.08)
    assert ratio.ess_fraction > 0.85
    assert ratio.weights.mean() == pytest.approx(1.0)


def test_normalisation_fixes_the_mean_without_moving_the_ess() -> None:
    """Normalising is a scale choice, so it must be invisible to the diagnostic."""
    rng = np.random.default_rng(6)
    early = rng.normal(0.0, 1.0, size=(400, 2))
    late = rng.normal(0.9, 1.0, size=(300, 2))

    raw = estimate_density_ratio(early, late, seed=4, normalise=False)
    normed = estimate_density_ratio(early, late, seed=4, normalise=True)

    assert normed.weights.mean() == pytest.approx(1.0)
    assert normed.ess == pytest.approx(raw.ess)
    np.testing.assert_allclose(normed.raw_weights, raw.raw_weights, rtol=1e-12)


def test_clipping_truncates_the_tail_and_lifts_the_effective_sample_size() -> None:
    """Clipping is the variance control, so it must demonstrably reduce variance."""
    rng = np.random.default_rng(8)
    early = rng.normal(0.0, 1.0, size=(600, 2))
    late = rng.normal(1.5, 1.0, size=(400, 2))

    loose = estimate_density_ratio(early, late, seed=5, clip_quantile=None)
    tight = estimate_density_ratio(early, late, seed=5, clip_quantile=0.90)

    assert tight.n_clipped == pytest.approx(0.10 * 600, abs=3)
    assert tight.clip_threshold is not None
    assert tight.max_weight < loose.max_weight
    assert tight.ess > loose.ess
    assert loose.n_clipped == 0
    assert loose.clip_threshold is None


def test_density_ratio_reports_the_discrimination_it_was_built_from() -> None:
    """The AUC is the number that explains a collapsed ESS, so it must be present."""
    rng = np.random.default_rng(9)
    early = rng.normal(0.0, 1.0, size=(400, 2))
    late = rng.normal(2.0, 1.0, size=(300, 2))
    ratio = estimate_density_ratio(early, late, seed=6)

    assert ratio.oof_auc > 0.85
    assert 0.0 <= ratio.oof_accuracy <= 1.0
    assert ratio.ess_fraction < 0.5  # strong shift, weights necessarily concentrated
    assert ratio.n_early == 400
    assert ratio.n_late == 300


def test_density_ratio_is_deterministic_in_its_seed() -> None:
    """Two calls with one seed must agree bit for bit."""
    rng = np.random.default_rng(10)
    early = rng.normal(0.0, 1.0, size=(200, 2))
    late = rng.normal(0.7, 1.0, size=(150, 2))
    first = estimate_density_ratio(early, late, seed=12)
    second = estimate_density_ratio(early, late, seed=12)
    np.testing.assert_array_equal(first.weights, second.weights)


@pytest.mark.parametrize("bad_quantile", [0.0, -0.5, 1.5])
def test_density_ratio_rejects_an_out_of_range_clip_quantile(bad_quantile: float) -> None:
    rng = np.random.default_rng(11)
    early = rng.normal(size=(50, 2))
    late = rng.normal(size=(50, 2))
    with pytest.raises(ValueError, match="clip_quantile"):
        estimate_density_ratio(early, late, seed=1, clip_quantile=bad_quantile)


def test_density_ratio_rejects_mismatched_feature_dimensions() -> None:
    with pytest.raises(ValueError, match="feature dimensions disagree"):
        estimate_density_ratio(np.zeros((10, 3)), np.zeros((10, 4)), seed=1)


def test_density_ratio_rejects_an_unknown_prior_correction() -> None:
    rng = np.random.default_rng(12)
    with pytest.raises(ValueError, match="prior_correction"):
        estimate_density_ratio(
            rng.normal(size=(40, 2)),
            rng.normal(size=(40, 2)),
            seed=1,
            prior_correction="whatever",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Smoke test one: covariate shift alone
# ---------------------------------------------------------------------------
def test_covariate_shift_case_actually_degrades(covariate_shift_only: IWRiskResult) -> None:
    """Guard against a vacuous pass.

    If the unweighted fit did not degrade there would be nothing for the
    correction to explain, and the next test would pass no matter what the
    estimator did. The whole point of this scenario is a large drop that is
    nonetheless entirely attributable to ``p(x)``.
    """
    assert covariate_shift_only.raw_gap > 0.15


def test_covariate_shift_alone_is_not_attributed_to_the_conditional(
    covariate_shift_only: IWRiskResult,
) -> None:
    """``p(y|x)`` is identical by construction, so it must not be implicated."""
    result = covariate_shift_only

    assert result.interpretable, result.reasons
    assert not result.implicates_conditional()
    assert result.verdict() in ("covariate_shift_sufficient", "inconclusive")
    # The interval must straddle zero, and both variants must agree that it does.
    assert result.clipped.gap_ci_low < 0.0 < result.clipped.gap_ci_high
    assert result.unclipped.gap_ci_low < 0.0 < result.unclipped.gap_ci_high
    assert result.clipped.direction == "none"


def test_the_correction_removes_most_of_the_raw_degradation(
    covariate_shift_only: IWRiskResult,
) -> None:
    """The corrected gap must be far smaller than the uncorrected one.

    This is the substantive claim, stronger than "the interval covers zero": a
    useless estimator with a very wide interval would also cover zero.
    """
    result = covariate_shift_only
    assert abs(result.clipped.conditional_gap) < 0.25 * result.raw_gap


def test_the_covariate_shift_case_sits_in_the_realistic_discrimination_band(
    covariate_shift_only: IWRiskResult,
) -> None:
    """The scenario is calibrated to the discrimination the real windows show.

    A synthetic case that separated the windows at 0.99 would exercise a regime
    the paper never reaches; one at 0.55 would make the weights a no-op. The
    real SIFT windows sit at 0.80 to 0.84 domain-classifier accuracy.
    """
    assert 0.70 < covariate_shift_only.ratio_unclipped.oof_auc < 0.90


# ---------------------------------------------------------------------------
# Smoke test two: a changed decision rule
# ---------------------------------------------------------------------------
def test_a_changed_decision_rule_is_attributed_to_the_conditional(
    changed_rule_only: IWRiskResult,
) -> None:
    """``p(x)`` is identical, so only the conditional can explain the gap."""
    result = changed_rule_only

    assert result.interpretable, result.reasons
    assert result.implicates_conditional()
    assert result.verdict() == "conditional_change_implicated"
    assert result.clipped.gap_ci_low > 0.0
    assert result.clipped.direction == "late_worse"


def test_the_changed_rule_case_has_nothing_for_the_weights_to_correct(
    changed_rule_only: IWRiskResult,
) -> None:
    """Identical inputs, so the weights are near-uniform and the note fires."""
    result = changed_rule_only

    assert result.ratio_unclipped.oof_auc < 0.60
    assert result.clipped.ess_fraction_early > 0.85
    assert any("no-op" in note for note in result.notes)
    # The correction changes nothing, so corrected and raw gaps must agree.
    assert result.clipped.conditional_gap == pytest.approx(result.raw_gap, abs=0.06)


def test_a_changed_rule_is_still_detected_underneath_a_real_covariate_shift(
    both_perturbations: IWRiskResult,
) -> None:
    """The case the real windows are in: both perturbations present at once.

    The shift here is the same one that, on its own, the method correctly
    declines to attribute to the conditional. Adding a genuine rule change on
    top must flip the verdict, which is what separates this instrument from the
    refuted step-three test.
    """
    result = both_perturbations

    assert result.ratio_unclipped.oof_auc > 0.70  # the shift really is present
    assert result.interpretable, result.reasons
    assert result.implicates_conditional()
    assert result.clipped.gap_ci_low > 0.0


# ---------------------------------------------------------------------------
# Interpretability: the diagnostics must override the interval
# ---------------------------------------------------------------------------
def test_a_tiny_late_window_is_undetermined_whatever_the_interval_says() -> None:
    """Below the domain-classifier threshold no verdict is available."""
    x_e, y_e, x_l, y_l = make_windows(n_early=400, n_late=40, shift=0.0)
    result = importance_weighted_risk(x_e, y_e, x_l, y_l, seed=SEED, n_boot=100)

    assert not result.interpretable
    assert result.verdict() == "undetermined"
    assert not result.implicates_conditional()
    assert any("late window" in reason for reason in result.reasons)
    assert result.warning is not None


def test_a_collapsed_effective_sample_size_blocks_the_verdict() -> None:
    """A severe shift destroys the overlap, and the code must refuse to conclude.

    This is the outcome the deliverable warns is likely on the real windows: the
    estimate is still computed and returned, but it is not reported as usable.

    The scenario is built directly rather than through ``make_windows`` because
    the label must stay balanced under a shift this severe. The shift is put on
    ``x0`` while the label depends only on ``x1``, so both windows keep two
    classes and the only thing the shift destroys is the overlap.
    """
    rng = np.random.default_rng(2)
    x_e = rng.normal(0.0, 1.0, size=(300, 3))
    x_l = rng.normal(0.0, 1.0, size=(300, 3))
    x_e[:, 0] -= 2.5
    x_l[:, 0] += 2.5
    y_e = (x_e[:, 1] > 0.0).astype(int)
    y_l = (x_l[:, 1] > 0.0).astype(int)

    result = importance_weighted_risk(x_e, y_e, x_l, y_l, seed=SEED, n_boot=100)

    assert result.clipped.ess_early < MIN_INTERPRETABLE_ESS
    assert not result.interpretable
    assert result.verdict() == "undetermined"
    assert any("effective" in reason for reason in result.reasons)
    # The number is still there; it is simply flagged rather than suppressed.
    assert np.isfinite(result.clipped.conditional_gap)


def test_unseen_late_classes_are_reported_as_a_label_space_change() -> None:
    """Importance weighting cannot fix a class the early window never held."""
    x_e, y_e, x_l, y_l = make_windows(n_early=400, n_late=300, shift=0.0)
    y_l = y_l.copy()
    y_l[:25] = 7  # a family that does not exist in the early window

    result = importance_weighted_risk(x_e, y_e, x_l, y_l, seed=SEED, n_boot=100)

    assert result.n_late_rows_unseen_class == 25
    assert any("label space" in note for note in result.notes)
    np.testing.assert_array_equal(result.labels, np.array([0, 1]))


# ---------------------------------------------------------------------------
# The verdict rule itself, on hand-built results
# ---------------------------------------------------------------------------
def _estimate(low: float, high: float, label: str = "clipped(q=0.99)") -> IWRiskEstimate:
    """One estimate with a chosen interval; the diagnostics are healthy."""
    direction = "late_worse" if low > 0 else ("late_better" if high < 0 else "none")
    return IWRiskEstimate(
        label=label,
        risk_source_weighted=0.8,
        risk_target=0.8 - (low + high) / 2.0,
        conditional_gap=(low + high) / 2.0,
        gap_ci_low=low,
        gap_ci_high=high,
        gap_ci_width=high - low,
        ess_early=400.0,
        ess_fit=200.0,
        ess_heldout=200.0,
        ess_fraction_early=0.8,
        ess_fraction_heldout=0.8,
        max_weight=3.0,
        n_clipped=5,
        direction=direction,  # type: ignore[arg-type]
    )


def _ratio() -> DensityRatioResult:
    """A healthy density-ratio result, for building a hand-made IWRiskResult."""
    weights = np.ones(500)
    return DensityRatioResult(
        weights=weights,
        raw_weights=weights,
        ess=500.0,
        ess_fraction=1.0,
        n_early=500,
        n_late=400,
        oof_auc=0.80,
        oof_accuracy=0.74,
        prior_correction=1.0,
        n_folds=5,
        clip_quantile=None,
        clip_threshold=None,
        n_clipped=0,
        max_weight=1.0,
        mean_raw_weight=1.0,
    )


def _result(
    clipped: IWRiskEstimate,
    unclipped: IWRiskEstimate | None = None,
    interpretable: bool = True,
    sensitive: bool = False,
) -> IWRiskResult:
    """Assemble an IWRiskResult around chosen estimates."""
    return IWRiskResult(
        metric="macro_f1",
        n_early=500,
        n_late=400,
        n_fit=250,
        n_heldout=250,
        labels=np.array([0, 1]),
        n_late_rows_unseen_class=0,
        risk_source_unweighted=0.8,
        risk_target_unweighted_fit=0.6,
        raw_gap=0.2,
        ratio_unclipped=_ratio(),
        ratio_clipped=_ratio(),
        unclipped=unclipped if unclipped is not None else clipped,
        clipped=clipped,
        sensitive_to_clipping=sensitive,
        interpretable=interpretable,
        max_decisive_width=DEFAULT_MAX_DECISIVE_WIDTH,
        reasons=() if interpretable else ("blocked for the sake of the test",),
    )


def test_an_interval_excluding_zero_implicates_the_conditional() -> None:
    assert _result(_estimate(0.05, 0.12)).verdict() == "conditional_change_implicated"
    assert _result(_estimate(-0.12, -0.05)).verdict() == "conditional_change_implicated"


def test_a_narrow_interval_covering_zero_reads_as_covariate_shift() -> None:
    assert _result(_estimate(-0.04, 0.05)).verdict() == "covariate_shift_sufficient"


def test_a_wide_interval_covering_zero_reads_as_inconclusive() -> None:
    """The expected outcome at this discrimination, and a clean result.

    A wide interval must never be reported as evidence that covariate shift
    suffices; it is evidence of nothing, and the verdict names that state.
    """
    verdict = _result(_estimate(-0.30, 0.30)).verdict()
    assert verdict == "inconclusive"
    assert verdict != "covariate_shift_sufficient"


def test_uninterpretable_overrides_even_a_decisive_interval() -> None:
    """A degenerate weight vector can make any interval arbitrarily tight.

    The sample-size guard must therefore win over the p-value, exactly as it
    does in DomainClassifierResult.verdict.
    """
    result = _result(_estimate(0.20, 0.30), interpretable=False)
    assert result.verdict() == "undetermined"
    assert not result.implicates_conditional()


def test_disagreement_between_clipped_and_unclipped_downgrades_to_inconclusive() -> None:
    """A conclusion that depends on the truncation quantile is not a conclusion."""
    result = _result(
        clipped=_estimate(0.02, 0.10),
        unclipped=_estimate(-0.05, 0.09, label="unclipped"),
        sensitive=True,
    )
    assert result.verdict() == "inconclusive"
    assert not result.implicates_conditional()


def test_reading_ignores_interpretability_by_design() -> None:
    """Interpretability belongs to the result, not to a single estimate."""
    assert _estimate(0.05, 0.12).reading() == "conditional_change_implicated"
    assert _estimate(-0.5, 0.5).reading() == "inconclusive"
    assert _estimate(-0.01, 0.01).reading() == "covariate_shift_sufficient"


def test_excludes_zero_is_exact_at_the_boundary() -> None:
    """An interval touching zero still covers it."""
    assert not _estimate(0.0, 0.1).excludes_zero
    assert not _estimate(-0.1, 0.0).excludes_zero
    assert _estimate(1e-9, 0.1).excludes_zero


# ---------------------------------------------------------------------------
# Reporting surface and error paths
# ---------------------------------------------------------------------------
def test_summary_carries_every_field_a_results_table_needs(
    covariate_shift_only: IWRiskResult,
) -> None:
    """Both variants and both effective sample sizes must reach the table.

    Reporting only the clipped number would hide the sensitivity the deliverable
    requires to be visible.
    """
    summary = covariate_shift_only.summary()
    for key in (
        "verdict",
        "interpretable",
        "raw_gap",
        "conditional_gap_clipped",
        "conditional_gap_unclipped",
        "gap_ci_clipped",
        "gap_ci_unclipped",
        "ess_clipped",
        "ess_unclipped",
        "ess_fraction_clipped",
        "ess_fraction_unclipped",
        "domain_auc",
        "sensitive_to_clipping",
    ):
        assert key in summary, key
    assert summary["verdict"] == covariate_shift_only.verdict()


def test_the_primary_estimate_is_the_clipped_one(
    covariate_shift_only: IWRiskResult,
) -> None:
    """Pre-registered: clipping is the variance-controlled variant."""
    assert covariate_shift_only.primary is covariate_shift_only.clipped
    assert covariate_shift_only.clipped.label.startswith("clipped")
    assert covariate_shift_only.unclipped.label == "unclipped"
    assert covariate_shift_only.unclipped.n_clipped == 0


def test_the_whole_analysis_is_deterministic_in_its_seed() -> None:
    """A reported number must be reproducible from the seed alone."""
    x_e, y_e, x_l, y_l = make_windows(n_early=300, n_late=250, shift=0.8)
    first = importance_weighted_risk(x_e, y_e, x_l, y_l, seed=21, n_boot=120)
    second = importance_weighted_risk(x_e, y_e, x_l, y_l, seed=21, n_boot=120)

    assert first.clipped.conditional_gap == second.clipped.conditional_gap
    assert first.clipped.gap_ci_low == second.clipped.gap_ci_low
    assert first.clipped.ess_early == second.clipped.ess_early
    assert first.verdict() == second.verdict()


@pytest.mark.parametrize("metric", IWRISK_METRICS)
def test_every_supported_metric_produces_a_finite_gap(metric: str) -> None:
    """All four metrics are higher-is-better, so the gap sign is readable."""
    x_e, y_e, x_l, y_l = make_windows(n_early=250, n_late=200, shift=0.6)
    result = importance_weighted_risk(
        x_e, y_e, x_l, y_l, seed=SEED, metric=metric, n_boot=60
    )
    assert np.isfinite(result.clipped.conditional_gap)
    assert result.clipped.gap_ci_low <= result.clipped.gap_ci_high
    assert result.metric == metric


def test_a_model_without_sample_weight_support_is_rejected_not_downgraded() -> None:
    """Silently fitting unweighted would answer a different question.

    KNeighborsClassifier.fit takes no sample_weight, the same reason control A1
    keeps some models out of sift.models.SUPPORTS_SAMPLE_WEIGHT. The failure must
    be loud, because an unweighted fit reported under this module's name would
    look like a corrected estimate.
    """
    x_e, y_e, x_l, y_l = make_windows(n_early=120, n_late=100, shift=0.5)
    with pytest.raises(ValueError, match="sample_weight"):
        importance_weighted_risk(
            x_e,
            y_e,
            x_l,
            y_l,
            seed=SEED,
            estimator=KNeighborsClassifier(n_neighbors=3),
            n_boot=20,
        )


def test_a_bare_estimator_without_a_pipeline_is_accepted() -> None:
    """Weight routing must work for a plain estimator as well as a Pipeline."""
    x_e, y_e, x_l, y_l = make_windows(n_early=250, n_late=200, shift=0.6)
    result = importance_weighted_risk(
        x_e,
        y_e,
        x_l,
        y_l,
        seed=SEED,
        estimator=LogisticRegression(max_iter=500, random_state=0),
        n_boot=60,
    )
    assert np.isfinite(result.clipped.conditional_gap)


def test_default_risk_model_accepts_weights_through_the_pipeline() -> None:
    """The default is a Pipeline, whose weight kwarg needs the step prefix."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(80, 3))
    y = (x[:, 0] > 0).astype(int)
    model = default_risk_model(seed=0)
    model.fit(x, y, clf__sample_weight=np.abs(rng.normal(1.0, 0.2, 80)))
    assert model.predict(x).shape == (80,)


def test_an_unknown_metric_is_rejected() -> None:
    x_e, y_e, x_l, y_l = make_windows(n_early=60, n_late=50, shift=0.5)
    with pytest.raises(ValueError, match="unknown metric"):
        importance_weighted_risk(x_e, y_e, x_l, y_l, seed=1, metric="auc", n_boot=10)


def test_a_single_class_early_window_is_rejected() -> None:
    """No risk contrast is defined when the early window holds one class."""
    x_e, _, x_l, y_l = make_windows(n_early=60, n_late=50, shift=0.5)
    with pytest.raises(ValueError, match="single class"):
        importance_weighted_risk(
            x_e, np.zeros(60, dtype=int), x_l, y_l, seed=1, n_boot=10
        )


def test_mismatched_labels_are_rejected() -> None:
    x_e, y_e, x_l, y_l = make_windows(n_early=60, n_late=50, shift=0.5)
    with pytest.raises(ValueError, match="labels"):
        importance_weighted_risk(x_e, y_e[:-5], x_l, y_l, seed=1, n_boot=10)


@pytest.mark.parametrize("fraction", [0.0, 1.0, 1.5])
def test_an_out_of_range_heldout_fraction_is_rejected(fraction: float) -> None:
    x_e, y_e, x_l, y_l = make_windows(n_early=60, n_late=50, shift=0.5)
    with pytest.raises(ValueError, match="heldout_fraction"):
        importance_weighted_risk(
            x_e, y_e, x_l, y_l, seed=1, heldout_fraction=fraction, n_boot=10
        )
