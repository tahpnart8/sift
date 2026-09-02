"""Importance-weighted risk: a direct probe of whether ``p(y|x)`` changed.

Why this module exists
----------------------
The SIFT residual ``R = Delta - v(N)`` is what the four controls leave
unexplained, never a measurement of genuine concept drift. The framework's step-three test of the
conditional was refuted by the paper's own counterexample: the statistic it
computes summarises ``p(x|y)``, not ``p(y|x)``, so a pure covariate shift can
move it. That left the framework with no valid instrument for the one quantity
concept drift is defined by.

This module supplies the standard instrument. Under the covariate-shift
assumption ``p_early(y|x) = p_late(y|x)`` with ``p_early(x) != p_late(x)``, a
model fitted on the early window *reweighted by the density ratio*

.. math:: w(x) = p_{late}(x) / p_{early}(x)

is fitted, in expectation, to the late window's input distribution. Its risk
under ``p_late(x)`` is then estimable from the early sample alone. Comparing
that estimate against the risk actually observed on the late window isolates
the conditional: both quantities live under the same ``p_late(x)``, so a
difference between them cannot be produced by covariate shift.

The contrast, precisely
-----------------------
Let the early window be split into a fit half and a held-out half, and let
``f`` be the model fitted on the fit half with weights ``w``.

``A = risk_source_weighted``
    Performance of ``f`` on the *early* held-out half, scored with the same
    weights. Estimand: performance under ``p_late(x) p_early(y|x)``.
``B = risk_target``
    Performance of ``f`` on the *late* window, unweighted, because the late
    window already is its own distribution. Estimand: performance under
    ``p_late(x) p_late(y|x)``.
``conditional_gap = A - B``
    The two estimands differ only in the conditional. Every reported metric is
    oriented so that higher is better, so a positive gap means the late window
    performs *worse* than covariate shift alone can explain, which implicates
    ``p(y|x)``. A gap whose interval covers zero means covariate shift suffices
    to account for the observed degradation.

Scoring the same fitted model under both distributions is what makes the
contrast valid under a misspecified model. The weighted fit does not need to
recover the truth; it only needs to be the same function on both sides, so that
the only remaining difference between A and B is the conditional.

For context the module also reports the naive contrast, ``raw_gap``, from an
unweighted fit on the same fit half. ``raw_gap`` is the degradation the paper
already reports; ``conditional_gap`` is what survives the correction for
``p(x)``.

Assumptions this rests on, all of them falsifiable
--------------------------------------------------
1. **Common support.** ``p_early(x) > 0`` wherever ``p_late(x) > 0``. If the
   late window occupies a region the early window never visits, the true
   density ratio is unbounded there and no finite reweighting reaches it. The
   effective-sample-size diagnostic is the observable symptom of a support
   violation, and it is reported with every result.
2. **A usable density-ratio estimate.** The ratio is obtained from a domain
   classifier, so it inherits that classifier's calibration error. At the 0.80
   to 0.84 discrimination the SIFT windows exhibit, the ratio is genuinely
   badly behaved and the estimator's variance is large. This is a property of
   the data, not a defect of the code, and the module is built to say so rather
   than to hide it.
3. **The label space is shared.** A class present in the late window but absent
   from the early window is a change in the label space, not in ``p(y|x)``, and
   importance weighting cannot address it. Such rows are counted and reported
   as a note.

What the interval does and does not cover
-----------------------------------------
The bootstrap resamples the two *evaluation* sets. It therefore covers sampling
variation in the two risk estimates. It does **not** cover variation from
re-estimating the density ratio, nor from refitting the model, both of which are
held fixed across replicates. The reported interval is consequently a *lower*
bound on total uncertainty, and is labelled as such. A wide interval that covers
zero is a legitimate and expected outcome at this level of discrimination; the
pre-registered rule reads it as ``"inconclusive"``, which is a clean result and
not a failure.

References
----------
.. [1] H. Shimodaira, "Improving predictive inference under covariate shift by
       weighting the log-likelihood function", *Journal of Statistical Planning
       and Inference*, 90(2):227-244, 2000.
.. [2] M. Sugiyama, T. Suzuki and T. Kanamori, *Density Ratio Estimation in
       Machine Learning*, Cambridge University Press, 2012.
.. [3] S. Rabanser, S. Guennemann and Z. C. Lipton, "Failing Loudly: An
       Empirical Study of Methods for Detecting Dataset Shift", *NeurIPS*, 2019.
.. [4] A. Kong, "A note on importance sampling using standardized weights",
       University of Chicago, Technical Report 348, 1992. Source of the
       effective-sample-size expression used here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sift.drift import MIN_INTERPRETABLE_HELDOUT, default_domain_classifier

__all__ = [
    "DEFAULT_CLIP_QUANTILE",
    "DEFAULT_MAX_DECISIVE_WIDTH",
    "DEFAULT_MIN_ESS_FRACTION",
    "IWRISK_METRICS",
    "MIN_INTERPRETABLE_ESS",
    "DensityRatioResult",
    "IWRiskEstimate",
    "IWRiskResult",
    "PriorCorrection",
    "Verdict",
    "default_risk_model",
    "effective_sample_size",
    "estimate_density_ratio",
    "importance_weighted_risk",
]

#: Metrics this module can evaluate. All are oriented so that higher is better,
#: which is what makes ``conditional_gap = A - B`` readable in one direction.
IWRISK_METRICS: tuple[str, ...] = ("macro_f1", "balanced_accuracy", "mcc", "accuracy")

MIN_INTERPRETABLE_ESS: int = 100
"""Effective sample size below which a weighted estimate is not interpretable.

Chosen to match :data:`sift.drift.MIN_INTERPRETABLE_HELDOUT` rather than derived
independently. The reasoning transfers: Rabanser et al. report that a two-sample
probe fails at or below one hundred samples, and an importance-weighted mean
built on an *effective* sample of that size is subject to the same objection.
The raw row count is not the relevant quantity once the rows carry wildly
unequal weights, so the threshold is applied to the effective count.
"""

DEFAULT_MIN_ESS_FRACTION: float = 0.10
"""Fraction of the raw early count below which the weights are treated as degenerate.

An effective sample below a tenth of the raw count means the reweighted estimate
is being carried by a small minority of rows. The point estimate is still
computed and returned, but it is not reported as usable.
"""

DEFAULT_CLIP_QUANTILE: float = 0.99
"""Quantile at which weights are clipped for the variance-controlled estimate."""

DEFAULT_MAX_DECISIVE_WIDTH: float = 0.20
"""Interval width above which a zero-covering interval is called inconclusive.

A pre-registered threshold, not a tuned one. An interval of width 0.20 on a
macro-F1 gap spans a fifth of the whole metric range; an interval that wide is
consistent with almost any hypothesis about the conditional and must not be
reported as evidence that covariate shift suffices.
"""

PriorCorrection = Literal["balanced", "sampling", "none"]
Verdict = Literal[
    "conditional_change_implicated",
    "covariate_shift_sufficient",
    "inconclusive",
    "undetermined",
]

# Probabilities are clipped before the odds transform. Without this a domain
# classifier that returns a hard 0 or 1 on some row, which regularised logistic
# regression will do once the windows separate cleanly in some direction,
# produces an infinite or zero weight and destroys the whole vector.
_PROBA_EPS: float = 1e-6


@dataclass(frozen=True)
class DensityRatioResult:
    """Estimated density ratio for the rows of the early window.

    Attributes
    ----------
    weights : numpy.ndarray
        One weight per early row, aligned to the input order. Normalised to mean
        one when ``normalise`` was requested.
    raw_weights : numpy.ndarray
        The same weights before clipping and before normalisation, kept so that
        the effect of both operations stays auditable.
    ess : float
        Effective sample size of ``weights`` over the whole early window.
    ess_fraction : float
        ``ess`` divided by the raw early row count.
    n_early, n_late : int
        Raw row counts of the two windows.
    oof_auc : float
        Out-of-fold ROC AUC of the domain classifier over the pooled windows.
        This is the discrimination level that drives the weight variance; the
        SIFT windows sit near 0.80 to 0.84.
    oof_accuracy : float
        Out-of-fold accuracy of the same classifier, comparable to the figure
        :func:`sift.drift.domain_classifier_test` reports.
    prior_correction : float
        Multiplicative constant applied to the odds. See
        :func:`estimate_density_ratio`.
    n_folds : int
        Number of cross-fitting folds actually used.
    clip_quantile : float or None
        Quantile at which the raw weights were clipped, or None if unclipped.
    clip_threshold : float or None
        The weight value that quantile corresponds to.
    n_clipped : int
        Rows whose weight was reduced by clipping.
    max_weight, mean_raw_weight : float
        Extremes of the returned and raw vectors, reported because a single
        dominating weight is the usual failure mode.
    """

    weights: np.ndarray
    raw_weights: np.ndarray
    ess: float
    ess_fraction: float
    n_early: int
    n_late: int
    oof_auc: float
    oof_accuracy: float
    prior_correction: float
    n_folds: int
    clip_quantile: float | None
    clip_threshold: float | None
    n_clipped: int
    max_weight: float
    mean_raw_weight: float


@dataclass(frozen=True)
class IWRiskEstimate:
    """One importance-weighted risk estimate, clipped or unclipped.

    Attributes
    ----------
    label : str
        ``"unclipped"`` or ``"clipped(q=...)"``.
    risk_source_weighted : float
        Score A: the weighted model scored on the weighted early held-out half.
        Estimates performance under ``p_late(x) p_early(y|x)``.
    risk_target : float
        Score B: the same model scored on the late window. Estimates
        performance under ``p_late(x) p_late(y|x)``.
    conditional_gap : float
        ``risk_source_weighted - risk_target``. Positive means the late window
        underperforms what covariate shift alone predicts.
    gap_ci_low, gap_ci_high, gap_ci_width : float
        Percentile bootstrap interval for the gap, covering evaluation variance
        only.
    ess_early, ess_fit, ess_heldout : float
        Effective sample sizes of the weights over the whole early window, over
        the fit half, and over the held-out half on which score A is computed.
    ess_fraction_early, ess_fraction_heldout : float
        The same quantities as fractions of their raw counts.
    max_weight : float
        Largest weight in the vector used.
    n_clipped : int
        Rows reduced by clipping. Zero for the unclipped estimate.
    direction : {"late_worse", "late_better", "none"}
        Sign of the gap when the interval excludes zero.
    """

    label: str
    risk_source_weighted: float
    risk_target: float
    conditional_gap: float
    gap_ci_low: float
    gap_ci_high: float
    gap_ci_width: float
    ess_early: float
    ess_fit: float
    ess_heldout: float
    ess_fraction_early: float
    ess_fraction_heldout: float
    max_weight: float
    n_clipped: int
    direction: Literal["late_worse", "late_better", "none"]

    @property
    def excludes_zero(self) -> bool:
        """True when the bootstrap interval lies entirely on one side of zero."""
        return self.gap_ci_low > 0.0 or self.gap_ci_high < 0.0

    def reading(self, max_decisive_width: float = DEFAULT_MAX_DECISIVE_WIDTH) -> Verdict:
        """Return the pre-registered reading of this estimate alone.

        Interpretability is a property of the whole result rather than of one
        estimate, so it is applied in :meth:`IWRiskResult.verdict` and not here.
        """
        if self.excludes_zero:
            return "conditional_change_implicated"
        if self.gap_ci_width > max_decisive_width:
            return "inconclusive"
        return "covariate_shift_sufficient"


@dataclass(frozen=True)
class IWRiskResult:
    """Full outcome of one importance-weighted risk analysis.

    Attributes
    ----------
    metric : str
        Metric the risks were computed with.
    n_early, n_late, n_fit, n_heldout : int
        Row counts: the whole early window, the late window, and the two halves
        the early window was split into.
    labels : numpy.ndarray
        Label set both risks were scored over, pinned to the classes present in
        the early window. A model fitted on the early window cannot predict a
        class it never saw, so scoring the two sides over different label sets
        would make the gap incomparable.
    n_late_rows_unseen_class : int
        Late rows whose class is absent from the early window. These are a
        label-space change, which importance weighting does not address.
    risk_source_unweighted : float
        Unweighted model on the unweighted early held-out half. The
        in-distribution reference.
    risk_target_unweighted_fit : float
        Unweighted model on the late window. The degradation as usually
        reported.
    raw_gap : float
        ``risk_source_unweighted - risk_target_unweighted_fit``. The uncorrected
        degradation this module is trying to explain.
    ratio_unclipped, ratio_clipped : DensityRatioResult
        Density-ratio estimates and their diagnostics.
    unclipped, clipped : IWRiskEstimate
        The two risk estimates. Both are always reported so that sensitivity to
        clipping is visible rather than a hidden analyst choice.
    sensitive_to_clipping : bool
        True when the two estimates read differently. When they do, the combined
        verdict is downgraded to ``"inconclusive"``: an answer that depends on
        where the weights were truncated is not an answer.
    interpretable : bool
        False when any blocking reason applies.
    reasons : tuple of str
        Blocking reasons. Empty when ``interpretable`` is true.
    notes : tuple of str
        Non-blocking observations worth printing beside the number.
    warning : str or None
        Joined ``reasons``, for callers that log a single string.
    """

    metric: str
    n_early: int
    n_late: int
    n_fit: int
    n_heldout: int
    labels: np.ndarray
    n_late_rows_unseen_class: int
    risk_source_unweighted: float
    risk_target_unweighted_fit: float
    raw_gap: float
    ratio_unclipped: DensityRatioResult
    ratio_clipped: DensityRatioResult
    unclipped: IWRiskEstimate
    clipped: IWRiskEstimate
    sensitive_to_clipping: bool
    interpretable: bool
    max_decisive_width: float
    reasons: tuple[str, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)
    warning: str | None = None

    @property
    def primary(self) -> IWRiskEstimate:
        """The clipped estimate, which the verdict is read from.

        Clipping is the variance-controlled variant, so it is primary; the
        unclipped estimate acts as the sensitivity check that can veto it.
        """
        return self.clipped

    def verdict(self) -> Verdict:
        """Return the pre-registered reading of the whole analysis.

        Returns
        -------
        {"conditional_change_implicated", "covariate_shift_sufficient",
         "inconclusive", "undetermined"}
            ``"undetermined"`` whenever a blocking diagnostic fires, regardless
            of where the interval sits. The ordering is deliberate and mirrors
            :meth:`sift.drift.DomainClassifierResult.verdict`: a degenerate
            weight vector can produce an arbitrarily tight interval around an
            arbitrary point, so the diagnostic must override the interval and not
            the other way round.

            ``"inconclusive"`` when the interval covers zero but is too wide to
            distinguish the hypotheses, or when clipping changes the reading.
            Under the pre-registered interpretation rule this is a clean result:
            it reports that the data cannot separate concept drift from covariate
            shift at this sample size, which is itself a finding.
        """
        if not self.interpretable:
            return "undetermined"
        if self.sensitive_to_clipping:
            return "inconclusive"
        return self.clipped.reading(self.max_decisive_width)

    def implicates_conditional(self) -> bool:
        """Return True only when the verdict positively implicates ``p(y|x)``."""
        return self.verdict() == "conditional_change_implicated"

    def summary(self) -> dict[str, Any]:
        """Return a flat dictionary of the reportable fields, for a results table."""
        return {
            "metric": self.metric,
            "verdict": self.verdict(),
            "interpretable": self.interpretable,
            "n_early": self.n_early,
            "n_late": self.n_late,
            "raw_gap": self.raw_gap,
            "conditional_gap_clipped": self.clipped.conditional_gap,
            "gap_ci_clipped": (self.clipped.gap_ci_low, self.clipped.gap_ci_high),
            "conditional_gap_unclipped": self.unclipped.conditional_gap,
            "gap_ci_unclipped": (self.unclipped.gap_ci_low, self.unclipped.gap_ci_high),
            "ess_clipped": self.clipped.ess_early,
            "ess_fraction_clipped": self.clipped.ess_fraction_early,
            "ess_unclipped": self.unclipped.ess_early,
            "ess_fraction_unclipped": self.unclipped.ess_fraction_early,
            "max_weight_unclipped": self.unclipped.max_weight,
            "domain_auc": self.ratio_unclipped.oof_auc,
            "domain_accuracy": self.ratio_unclipped.oof_accuracy,
            "sensitive_to_clipping": self.sensitive_to_clipping,
            "reasons": "; ".join(self.reasons),
            "notes": "; ".join(self.notes),
        }


def effective_sample_size(weights: np.ndarray) -> float:
    """Kish effective sample size of a weight vector, ``(sum w)^2 / sum(w^2)``.

    Parameters
    ----------
    weights : numpy.ndarray
        Non-negative weights.

    Returns
    -------
    float
        The effective sample size. Equals ``len(weights)`` when the weights are
        uniform and falls toward one as the mass concentrates on a single row.
        Scale-invariant, so normalising the weights does not change it.

    Raises
    ------
    ValueError
        If the vector is empty, holds a negative entry, or sums to zero.
    """
    w = np.asarray(weights, dtype=float)
    if w.size == 0:
        raise ValueError("cannot compute an effective sample size of an empty vector")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    total = float(np.sum(w))
    if total <= 0.0:
        raise ValueError("weights sum to zero, so no effective sample size is defined")
    return float(total * total / float(np.sum(w * w)))


def default_risk_model(seed: int) -> Pipeline:
    """Build the default model refitted under the importance weights.

    Parameters
    ----------
    seed : int
        Seed for the estimator, normally from :func:`sift.seeding.derive_seed`.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Standardisation followed by logistic regression.

    Notes
    -----
    A linear model is the default for the same reason it is the default domain
    classifier: the estimand here is a *difference* of two risks, and a
    high-capacity model inflates the variance of both terms without sharpening
    the contrast. Callers who want the lattice's own model families pass an
    estimator explicitly; anything whose ``fit`` accepts ``sample_weight`` works,
    including a :class:`~sklearn.pipeline.Pipeline`.
    """
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=seed)),
        ]
    )


def _score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str,
    labels: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> float:
    """Evaluate one metric against a label set fixed by the caller.

    The label set is pinned by the caller rather than inferred from ``y_true``
    so that the two sides of the gap are scored on the same scale. Inferring it
    would let a resample that happens to miss a class change the denominator of
    a macro average and so change the estimand mid-bootstrap.
    """
    if metric == "macro_f1":
        return float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0.0,
                sample_weight=sample_weight,
            )
        )
    if metric == "accuracy":
        return float(accuracy_score(y_true, y_pred, sample_weight=sample_weight))
    if metric == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, y_pred, sample_weight=sample_weight))
    if metric == "mcc":
        value = float(matthews_corrcoef(y_true, y_pred, sample_weight=sample_weight))
        # A resample that lands on a single predicted class leaves the
        # correlation undefined. Zero is the right substitute: no association.
        return 0.0 if not np.isfinite(value) else value
    raise ValueError(f"unknown metric {metric!r}; expected one of {IWRISK_METRICS}")


def _fit_with_weights(
    estimator: BaseEstimator,
    x: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None,
) -> BaseEstimator:
    """Fit a clone of ``estimator``, routing ``sample_weight`` through a Pipeline.

    Raises
    ------
    ValueError
        If the estimator's ``fit`` does not accept ``sample_weight``. The failure
        is raised rather than silently downgraded to an unweighted fit, because
        an unweighted fit answers a different question and would be reported
        under this module's name as though it had been corrected.
    """
    model = clone(estimator)
    if sample_weight is None:
        model.fit(x, y)
        return model
    try:
        if isinstance(model, Pipeline):
            final_step = model.steps[-1][0]
            model.fit(x, y, **{f"{final_step}__sample_weight": sample_weight})
        else:
            model.fit(x, y, sample_weight=sample_weight)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{type(model).__name__} does not accept sample_weight, so it cannot "
            "express the importance-weighted fit this estimator requires; pass a "
            "model whose fit takes sample_weight"
        ) from exc
    return model


def estimate_density_ratio(
    x_early: np.ndarray,
    x_late: np.ndarray,
    seed: int,
    *,
    estimator: BaseEstimator | None = None,
    n_folds: int = 5,
    prior_correction: PriorCorrection | float = "balanced",
    clip_quantile: float | None = None,
    normalise: bool = True,
) -> DensityRatioResult:
    """Estimate ``w(x) = p_late(x) / p_early(x)`` for every row of the early window.

    The estimate is the classifier trick: a domain classifier is trained to
    separate the two windows, and its calibrated probability is converted to a
    ratio through the odds transform

    .. math:: w(x) = \\frac{p(\\text{late}|x)}{1 - p(\\text{late}|x)} \\cdot c

    where ``c`` corrects for the proportions the two windows contributed to the
    classifier's training set. Bayes' rule gives
    ``p(late|x) / p(early|x) = w(x) p(late) / p(early)``, so ``c`` is the inverse
    of the training prior odds.

    Parameters
    ----------
    x_early, x_late : numpy.ndarray of shape (n, n_features)
        Feature matrices of the two windows. Labels are not used: the ratio is a
        statement about ``p(x)`` alone.
    seed : int
        Seed for the fold assignment and the classifier. Derive it with
        :func:`sift.seeding.derive_seed`.
    estimator : sklearn.base.BaseEstimator, optional
        Domain classifier, cloned before each fold fit. Defaults to
        :func:`sift.drift.default_domain_classifier`, which is the same
        discriminator :func:`sift.drift.domain_classifier_test` reports on, so
        the AUC returned here and the accuracy reported there describe one
        object rather than two.
    n_folds : int, optional
        Cross-fitting folds. Reduced automatically when a window is too small.
    prior_correction : {"balanced", "sampling", "none"} or float, optional
        ``"balanced"``, the default, assumes the classifier equalised the two
        windows through ``class_weight="balanced"``, which is what the SIFT
        default domain classifier does. Its training prior is then one half on
        each side, the prior odds are one, and ``c = 1``. ``"sampling"`` assumes
        an unweighted fit, whose training prior odds are ``n_late / n_early``,
        giving ``c = n_early / n_late``. ``"none"`` forces ``c = 1``, and a float
        sets it directly. Passing the wrong option rescales every weight by a
        constant, which leaves the effective sample size and, under
        ``normalise``, the returned weights unchanged, but it would corrupt an
        absolute reading of the ratio.
    clip_quantile : float, optional
        Quantile of the raw weights at which to truncate, for example 0.99. None
        leaves the weights untouched.
    normalise : bool, optional
        Rescale the returned weights to mean one over the early window. On by
        default, because a regularised fit is not scale-invariant in its weights:
        multiplying every weight by ten weakens the penalty tenfold relative to
        the likelihood and so changes the fitted model for reasons that have
        nothing to do with the shift. Normalising pins the total weight to the
        raw row count, which is the scale an unweighted fit would have used.

    Returns
    -------
    DensityRatioResult
        Weights for the early rows and the diagnostics that say whether they can
        be trusted.

    Raises
    ------
    ValueError
        If the feature dimensions disagree, a window is too small to cross-fit,
        or ``clip_quantile`` is outside ``(0, 1]``.

    Notes
    -----
    Probabilities are obtained **out of fold**. An in-sample probability from a
    classifier that has already seen the row is optimistic about its own window,
    which pushes the weights of early rows toward zero and manufactures an
    effective sample size smaller than the data warrants. Cross-fitting also
    means the weight of an early row never depends on that row's own
    contribution to the discriminator, so splitting the early window afterwards
    into a fit half and a held-out half does not leak.
    """
    early = np.asarray(x_early, dtype=float)
    late = np.asarray(x_late, dtype=float)
    if early.ndim != 2 or late.ndim != 2:
        raise ValueError(
            f"both windows must be 2-D, got shapes {early.shape} and {late.shape}"
        )
    if early.shape[1] != late.shape[1]:
        raise ValueError(
            f"feature dimensions disagree: {early.shape[1]} against {late.shape[1]}"
        )
    n_early = int(early.shape[0])
    n_late = int(late.shape[0])
    if n_early < 2 or n_late < 2:
        raise ValueError(
            f"each window needs at least two rows, got {n_early} and {n_late}"
        )
    if clip_quantile is not None and not 0.0 < clip_quantile <= 1.0:
        raise ValueError(f"clip_quantile must lie in (0, 1], got {clip_quantile}")

    folds = int(min(n_folds, n_early, n_late))
    if folds < 2:
        raise ValueError(
            f"cross-fitting needs at least two folds, but the smaller window holds "
            f"{min(n_early, n_late)} rows"
        )

    pooled = np.vstack([early, late])
    domain = np.concatenate(
        [np.zeros(n_early, dtype=np.int8), np.ones(n_late, dtype=np.int8)]
    )
    base = default_domain_classifier(seed) if estimator is None else estimator

    oof_proba = np.full(pooled.shape[0], np.nan, dtype=float)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train_idx, test_idx in splitter.split(pooled, domain):
        model = clone(base)
        model.fit(pooled[train_idx], domain[train_idx])
        # Column order follows model.classes_, which is sorted, so the late
        # window (class 1) is the last column. Locating it explicitly rather
        # than assuming index 1 keeps this correct for a classifier that saw
        # only one domain in a degenerate fold.
        classes = np.asarray(model.classes_)
        proba = model.predict_proba(pooled[test_idx])
        late_column = int(np.flatnonzero(classes == 1)[0]) if 1 in classes else None
        oof_proba[test_idx] = 0.0 if late_column is None else proba[:, late_column]

    if np.any(np.isnan(oof_proba)):  # pragma: no cover - StratifiedKFold covers all rows
        raise RuntimeError("cross-fitting left rows without an out-of-fold probability")

    oof_auc = float(roc_auc_score(domain, oof_proba))
    oof_accuracy = float(accuracy_score(domain, (oof_proba >= 0.5).astype(np.int8)))

    if isinstance(prior_correction, (int, float)) and not isinstance(
        prior_correction, bool
    ):
        correction = float(prior_correction)
    elif prior_correction == "sampling":
        correction = n_early / n_late
    elif prior_correction in ("balanced", "none"):
        correction = 1.0
    else:
        raise ValueError(
            f"unknown prior_correction {prior_correction!r}; expected 'balanced', "
            "'sampling', 'none' or a float"
        )

    p_late = np.clip(oof_proba[:n_early], _PROBA_EPS, 1.0 - _PROBA_EPS)
    raw = (p_late / (1.0 - p_late)) * correction

    clip_threshold: float | None = None
    n_clipped = 0
    weights = raw.copy()
    if clip_quantile is not None:
        clip_threshold = float(np.quantile(raw, clip_quantile))
        n_clipped = int(np.sum(raw > clip_threshold))
        weights = np.minimum(weights, clip_threshold)

    if normalise:
        mean_weight = float(np.mean(weights))
        if mean_weight <= 0.0:
            raise ValueError(
                "the estimated weights are all zero, so the early window carries no "
                "mass under the late input distribution; the windows do not overlap"
            )
        weights = weights / mean_weight

    ess = effective_sample_size(weights)
    return DensityRatioResult(
        weights=weights,
        raw_weights=raw,
        ess=ess,
        ess_fraction=ess / n_early,
        n_early=n_early,
        n_late=n_late,
        oof_auc=oof_auc,
        oof_accuracy=oof_accuracy,
        prior_correction=correction,
        n_folds=folds,
        clip_quantile=clip_quantile,
        clip_threshold=clip_threshold,
        n_clipped=n_clipped,
        max_weight=float(np.max(weights)),
        mean_raw_weight=float(np.mean(raw)),
    )


def _bootstrap_gap(
    y_source: np.ndarray,
    pred_source: np.ndarray,
    weight_source: np.ndarray,
    y_target: np.ndarray,
    pred_target: np.ndarray,
    metric: str,
    labels: np.ndarray,
    n_boot: int,
    seed: int,
    alpha: float,
) -> tuple[float, float]:
    """Percentile interval for ``score(source, weighted) - score(target)``.

    The two evaluation sets are resampled independently, which is the correct
    scheme because they are independent samples; the difference of the two
    resampled statistics is then a draw from the sampling distribution of the
    gap. Weights travel with the rows their index draw selected, so a replicate
    that happens to draw a heavy row carries that row's influence, which is
    exactly the extra variance importance weighting introduces and the reason
    the interval must not be computed from the unweighted rows.
    """
    rng = np.random.default_rng(seed)
    n_source = int(y_source.shape[0])
    n_target = int(y_target.shape[0])
    gaps = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx_s = rng.integers(0, n_source, n_source)
        idx_t = rng.integers(0, n_target, n_target)
        score_s = _score(
            y_source[idx_s], pred_source[idx_s], metric, labels, weight_source[idx_s]
        )
        score_t = _score(y_target[idx_t], pred_target[idx_t], metric, labels)
        gaps[i] = score_s - score_t
    low = float(np.quantile(gaps, alpha / 2.0))
    high = float(np.quantile(gaps, 1.0 - alpha / 2.0))
    return low, high


def _build_estimate(
    label: str,
    ratio: DensityRatioResult,
    fit_idx: np.ndarray,
    heldout_idx: np.ndarray,
    x_early: np.ndarray,
    y_early: np.ndarray,
    x_late: np.ndarray,
    y_late: np.ndarray,
    estimator: BaseEstimator,
    metric: str,
    labels: np.ndarray,
    n_boot: int,
    seed: int,
    alpha: float,
) -> IWRiskEstimate:
    """Fit under one weight vector and score both sides of the contrast."""
    w_fit = ratio.weights[fit_idx]
    w_heldout = ratio.weights[heldout_idx]

    model = _fit_with_weights(estimator, x_early[fit_idx], y_early[fit_idx], w_fit)
    pred_heldout = model.predict(x_early[heldout_idx])
    pred_late = model.predict(x_late)

    risk_source = _score(
        y_early[heldout_idx], pred_heldout, metric, labels, w_heldout
    )
    risk_target = _score(y_late, pred_late, metric, labels)
    gap = risk_source - risk_target
    low, high = _bootstrap_gap(
        y_early[heldout_idx],
        pred_heldout,
        w_heldout,
        y_late,
        pred_late,
        metric,
        labels,
        n_boot,
        seed,
        alpha,
    )

    direction: Literal["late_worse", "late_better", "none"] = "none"
    if low > 0.0:
        direction = "late_worse"
    elif high < 0.0:
        direction = "late_better"

    ess_heldout = effective_sample_size(w_heldout)
    return IWRiskEstimate(
        label=label,
        risk_source_weighted=risk_source,
        risk_target=risk_target,
        conditional_gap=gap,
        gap_ci_low=low,
        gap_ci_high=high,
        gap_ci_width=high - low,
        ess_early=ratio.ess,
        ess_fit=effective_sample_size(w_fit),
        ess_heldout=ess_heldout,
        ess_fraction_early=ratio.ess_fraction,
        ess_fraction_heldout=ess_heldout / float(heldout_idx.size),
        max_weight=ratio.max_weight,
        n_clipped=ratio.n_clipped,
        direction=direction,
    )


def importance_weighted_risk(
    x_early: np.ndarray,
    y_early: np.ndarray,
    x_late: np.ndarray,
    y_late: np.ndarray,
    seed: int,
    *,
    estimator: BaseEstimator | None = None,
    domain_estimator: BaseEstimator | None = None,
    metric: str = "macro_f1",
    n_folds: int = 5,
    clip_quantile: float = DEFAULT_CLIP_QUANTILE,
    heldout_fraction: float = 0.5,
    n_boot: int = 1000,
    alpha: float = 0.05,
    prior_correction: PriorCorrection | float = "balanced",
    min_ess: int = MIN_INTERPRETABLE_ESS,
    min_ess_fraction: float = DEFAULT_MIN_ESS_FRACTION,
    min_target: int = MIN_INTERPRETABLE_HELDOUT,
    max_decisive_width: float = DEFAULT_MAX_DECISIVE_WIDTH,
) -> IWRiskResult:
    """Test whether ``p(y|x)`` changed, after correcting the early window for ``p(x)``.

    The procedure is the one described in the module docstring: estimate the
    density ratio out of fold, split the early window, refit on the fit half
    under the weights, then compare the weighted early held-out risk against the
    late-window risk. Both estimates are produced twice, once with the raw
    weights and once with the weights clipped at ``clip_quantile``, and both are
    returned so that sensitivity to the truncation is visible.

    Parameters
    ----------
    x_early, y_early : numpy.ndarray
        Features and labels of the early window.
    x_late, y_late : numpy.ndarray
        Features and labels of the late window.
    seed : int
        Base seed. All stochastic components derive their own seed from it, so
        the whole analysis is reproducible from this integer.
    estimator : sklearn.base.BaseEstimator, optional
        Model refitted under the weights. Its ``fit`` must accept
        ``sample_weight``. Defaults to :func:`default_risk_model`.
    domain_estimator : sklearn.base.BaseEstimator, optional
        Classifier used for the density ratio. Defaults to
        :func:`sift.drift.default_domain_classifier`.
    metric : {'macro_f1', 'balanced_accuracy', 'mcc', 'accuracy'}, optional
        Risk measure, oriented so that higher is better.
    n_folds : int, optional
        Cross-fitting folds for the density ratio.
    clip_quantile : float, optional
        Quantile at which the clipped variant truncates the weights.
    heldout_fraction : float, optional
        Fraction of the early window reserved for scoring rather than fitting.
    n_boot : int, optional
        Bootstrap replicates for the gap interval.
    alpha : float, optional
        Two-sided interval level; 0.05 gives a 95 per cent interval.
    prior_correction : {"balanced", "sampling", "none"} or float, optional
        Passed to :func:`estimate_density_ratio`.
    min_ess : int, optional
        Effective sample size below which the result is flagged uninterpretable.
    min_ess_fraction : float, optional
        Effective fraction of the early window below which the same flag fires.
    min_target : int, optional
        Late-window size below which the result is flagged uninterpretable.
    max_decisive_width : float, optional
        Interval width above which a zero-covering interval is called
        inconclusive rather than read as evidence for covariate shift.

    Returns
    -------
    IWRiskResult
        Point estimates, intervals, weight diagnostics and the interpretability
        verdict.

    Raises
    ------
    ValueError
        If shapes disagree, either window is too small to split, the metric is
        unknown, or the early window holds a single class.

    Notes
    -----
    A result whose interval covers zero is **not** proof that ``p(y|x)`` is
    unchanged. It says only that this instrument, at this sample size and this
    level of window discrimination, cannot separate a changed conditional from
    covariate shift. The verdict names that state explicitly rather than
    collapsing it into the covariate-shift branch.
    """
    x_early = np.asarray(x_early, dtype=float)
    x_late = np.asarray(x_late, dtype=float)
    y_early = np.asarray(y_early)
    y_late = np.asarray(y_late)
    if metric not in IWRISK_METRICS:
        raise ValueError(f"unknown metric {metric!r}; expected one of {IWRISK_METRICS}")
    if x_early.shape[0] != y_early.shape[0]:
        raise ValueError(
            f"early window holds {x_early.shape[0]} rows but {y_early.shape[0]} labels"
        )
    if x_late.shape[0] != y_late.shape[0]:
        raise ValueError(
            f"late window holds {x_late.shape[0]} rows but {y_late.shape[0]} labels"
        )
    if not 0.0 < heldout_fraction < 1.0:
        raise ValueError(
            f"heldout_fraction must lie strictly in (0, 1), got {heldout_fraction}"
        )
    if n_boot < 1:
        raise ValueError(f"n_boot must be positive, got {n_boot}")

    n_early = int(x_early.shape[0])
    n_late = int(x_late.shape[0])
    labels = np.unique(y_early)
    if labels.size < 2:
        raise ValueError(
            "the early window holds a single class, so no risk contrast is defined"
        )

    # Every stochastic component gets its own derived seed so that changing, say,
    # the bootstrap size cannot silently move the fold assignment.
    ratio_seed = int((seed + 0x9E3779B1) % (2**32))
    split_seed = int((seed + 0x85EBCA6B) % (2**32))
    model_seed = int((seed + 0xC2B2AE35) % (2**32))
    boot_seed = int((seed + 0x27D4EB2F) % (2**32))

    ratio_unclipped = estimate_density_ratio(
        x_early,
        x_late,
        ratio_seed,
        estimator=domain_estimator,
        n_folds=n_folds,
        prior_correction=prior_correction,
        clip_quantile=None,
        normalise=True,
    )
    ratio_clipped = estimate_density_ratio(
        x_early,
        x_late,
        ratio_seed,
        estimator=domain_estimator,
        n_folds=n_folds,
        prior_correction=prior_correction,
        clip_quantile=clip_quantile,
        normalise=True,
    )

    # Stratify the early split when every class can appear on both sides. A
    # class with a single member cannot, and forcing it would raise rather than
    # degrade gracefully.
    _, class_counts = np.unique(y_early, return_counts=True)
    stratify = y_early if int(class_counts.min()) >= 2 else None
    indices = np.arange(n_early)
    fit_idx, heldout_idx = train_test_split(
        indices,
        test_size=heldout_fraction,
        random_state=split_seed,
        stratify=stratify,
    )
    if fit_idx.size < 2 or heldout_idx.size < 2:
        raise ValueError(
            f"the early window of {n_early} rows does not split into two usable "
            f"halves at heldout_fraction={heldout_fraction}"
        )

    model = default_risk_model(model_seed) if estimator is None else estimator

    # The uncorrected reference: same fit half, same held-out half, no weights.
    # This is the degradation the framework already reports, kept beside the
    # corrected one so a reader can see how much of it the correction removed.
    plain = _fit_with_weights(model, x_early[fit_idx], y_early[fit_idx], None)
    risk_source_unweighted = _score(
        y_early[heldout_idx], plain.predict(x_early[heldout_idx]), metric, labels
    )
    risk_target_unweighted_fit = _score(y_late, plain.predict(x_late), metric, labels)

    unclipped = _build_estimate(
        "unclipped",
        ratio_unclipped,
        fit_idx,
        heldout_idx,
        x_early,
        y_early,
        x_late,
        y_late,
        model,
        metric,
        labels,
        n_boot,
        boot_seed,
        alpha,
    )
    clipped = _build_estimate(
        f"clipped(q={clip_quantile:g})",
        ratio_clipped,
        fit_idx,
        heldout_idx,
        x_early,
        y_early,
        x_late,
        y_late,
        model,
        metric,
        labels,
        n_boot,
        boot_seed,
        alpha,
    )

    reasons: list[str] = []
    notes: list[str] = []

    if clipped.ess_early < min_ess:
        reasons.append(
            f"effective sample size of the weighted early window is "
            f"{clipped.ess_early:.1f}, below the threshold of {min_ess}; the "
            "reweighted fit rests on too few effective rows for the number to mean "
            "anything"
        )
    if clipped.ess_fraction_early < min_ess_fraction:
        reasons.append(
            f"the weights retain only {clipped.ess_fraction_early:.1%} of the "
            f"{n_early} raw early rows in effective terms, below the "
            f"{min_ess_fraction:.0%} floor; this is the signature of a support "
            "violation between the windows rather than of a usable correction"
        )
    if clipped.ess_heldout < min_ess:
        reasons.append(
            f"the weighted early held-out half has an effective size of "
            f"{clipped.ess_heldout:.1f}, below {min_ess}; the source-risk term of "
            "the gap is estimated from too few effective rows"
        )
    if n_late < min_target:
        reasons.append(
            f"the late window holds {n_late} rows, below the interpretability "
            f"threshold of {min_target} carried over from the domain-classifier "
            "probe; the target-risk term is not trustworthy at this size"
        )

    if ratio_unclipped.oof_auc < 0.55:
        notes.append(
            f"the domain classifier barely separates the windows "
            f"(out-of-fold AUC {ratio_unclipped.oof_auc:.3f}), so the weights are "
            "close to uniform and the correction is close to a no-op; a gap found "
            "here is not evidence that reweighting was needed"
        )
    if ratio_unclipped.max_weight > 0.25 * ratio_unclipped.weights.sum():
        notes.append(
            "a single early row carries more than a quarter of the total weight; "
            "the unclipped estimate is effectively a one-row estimate"
        )
    unseen = int(np.sum(~np.isin(y_late, labels)))
    if unseen > 0:
        notes.append(
            f"{unseen} of {n_late} late rows belong to a class absent from the early "
            "window; that is a change in the label space, which importance weighting "
            "does not correct and which the gap will attribute to the conditional"
        )
    if unclipped.reading(max_decisive_width) != clipped.reading(max_decisive_width):
        notes.append(
            f"clipping changes the reading, from "
            f"'{unclipped.reading(max_decisive_width)}' unclipped to "
            f"'{clipped.reading(max_decisive_width)}' at q={clip_quantile:g}; the "
            "verdict is downgraded to inconclusive because a conclusion that depends "
            "on where the weights were truncated is not a conclusion"
        )

    sensitive = unclipped.reading(max_decisive_width) != clipped.reading(
        max_decisive_width
    )
    interpretable = not reasons

    return IWRiskResult(
        metric=metric,
        n_early=n_early,
        n_late=n_late,
        n_fit=int(fit_idx.size),
        n_heldout=int(heldout_idx.size),
        labels=labels,
        n_late_rows_unseen_class=unseen,
        risk_source_unweighted=risk_source_unweighted,
        risk_target_unweighted_fit=risk_target_unweighted_fit,
        raw_gap=risk_source_unweighted - risk_target_unweighted_fit,
        ratio_unclipped=ratio_unclipped,
        ratio_clipped=ratio_clipped,
        unclipped=unclipped,
        clipped=clipped,
        sensitive_to_clipping=sensitive,
        interpretable=interpretable,
        max_decisive_width=max_decisive_width,
        reasons=tuple(reasons),
        notes=tuple(notes),
        warning="; ".join(reasons) if reasons else None,
    )
