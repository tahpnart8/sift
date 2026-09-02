"""Independent drift diagnostics for the SIFT residual.

The residual ``R = Delta - v(N)`` is what the four controls do not explain,
never a measurement of concept drift on its own. It is not an upper bound on
the drift contribution either: that reading would require every uncontrolled
component to be non-negative, and ``phi_b1_fs`` is negative in these results.
This module provides the independent probe the residual diagnosis needs: a
domain classifier that tests whether the input distribution of the late window
differs from that of the early window, plus a helper for correlating a drift
statistic with the residual across the twenty model-by-cut-point cells.

References
----------
.. [1] S. Rabanser, S. Guennemann and Z. C. Lipton, "Failing Loudly: An
       Empirical Study of Methods for Detecting Dataset Shift", *NeurIPS*,
       2019. arXiv:1810.11953.
.. [2] Y. Ovadia et al., "Can you trust your model's uncertainty? Evaluating
       predictive uncertainty under dataset shift", *NeurIPS*, 2019.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

__all__ = [
    "MIN_INTERPRETABLE_HELDOUT",
    "DomainClassifierResult",
    "bootstrap_gap_distribution",
    "conditional_structure_agreement",
    "correlate_residual_with_drift",
    "default_domain_classifier",
    "domain_classifier_test",
    "expected_calibration_error",
    "fast_accuracy",
    "fast_macro_f1",
]

MIN_INTERPRETABLE_HELDOUT: int = 100
"""Held-out size below which the domain classifier verdict is not interpretable.

Rabanser et al. report that the method "performs badly in the low-sample regime
(<= 100 samples)". Below this size the test is still computed and reported, but
``DomainClassifierResult.interpretable`` is false and the paper must not draw a
covariate-shift conclusion from it.
"""


def default_domain_classifier(seed: int) -> Pipeline:
    """Build the default domain classifier used when the caller supplies none.

    Parameters
    ----------
    seed : int
        Seed for the estimator. Passed as an integer rather than a
        ``RandomState`` instance, matching the contract requirement that a
        configuration reproduce bit for bit.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Standardisation followed by regularised logistic regression with
        balanced class weights.

    Notes
    -----
    A linear model is the default because the smallest window carries about
    sixty samples, where a high-capacity discriminator would separate the
    windows by memorisation rather than by distributional difference and so
    manufacture the very shift the test is meant to detect. ``class_weight`` is
    balanced because the early window is far larger than the late one, and an
    unweighted fit on such a training set degenerates to predicting the
    majority window.
    """
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )


@dataclass(frozen=True)
class DomainClassifierResult:
    """Outcome of one domain-classifier two-sample test.

    Attributes
    ----------
    accuracy : float
        Held-out accuracy of the domain classifier.
    n_heldout : int
        Size of the balanced held-out set actually scored.
    n_correct : int
        Number of correct held-out predictions, the binomial success count.
    p_value : float
        Two-sided binomial p-value against ``H0: accuracy = 0.5``.
    n_train : int
        Size of the training set built from the first halves.
    n_source_available : int
        Samples available in the source, that is early, window.
    n_target_available : int
        Samples available in the target, that is late, window.
    n_heldout_discarded : int
        Held-out samples dropped when balancing the larger half down to the smaller.
    interpretable : bool
        False when ``n_heldout < MIN_INTERPRETABLE_HELDOUT``.
    warning : str or None
        Explanation attached whenever ``interpretable`` is false.
    """

    accuracy: float
    n_heldout: int
    n_correct: int
    p_value: float
    n_train: int
    n_source_available: int
    n_target_available: int
    n_heldout_discarded: int
    interpretable: bool
    warning: str | None = None

    @property
    def excess_accuracy(self) -> float:
        """Accuracy above the chance level of one half, the effect size."""
        return self.accuracy - 0.5

    def verdict(self, alpha: float = 0.05) -> Literal["shift", "no_shift", "undetermined"]:
        """Return the pre-registered reading of this test.

        Parameters
        ----------
        alpha : float, optional
            Significance level.

        Returns
        -------
        {"shift", "no_shift", "undetermined"}
            ``"undetermined"`` whenever the held-out set is too small for the method to
            be trusted, regardless of the p-value.
        """
        if not self.interpretable:
            return "undetermined"
        return "shift" if self.p_value < alpha else "no_shift"


def _halve(n_samples: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return a random permutation of ``n_samples`` split into two halves."""
    order = rng.permutation(n_samples)
    midpoint = n_samples // 2
    return order[:midpoint], order[midpoint:]


def domain_classifier_test(
    x_source: np.ndarray,
    x_target: np.ndarray,
    seed: int,
    estimator: BaseEstimator | None = None,
    min_heldout: int = MIN_INTERPRETABLE_HELDOUT,
) -> DomainClassifierResult:
    """Test whether two windows are distinguishable from their features alone.

    Parameters
    ----------
    x_source : numpy.ndarray of shape (n_source, n_features)
        Feature matrix of the source, that is early, window. Labels are not
        used, since this probe targets the input distribution only.
    x_target : numpy.ndarray of shape (n_target, n_features)
        Feature matrix of the target, that is late, window. Must have the same
        number of columns as ``x_source``.
    seed : int
        Seed for the window permutation, the balancing subsample and the estimator.
    estimator : sklearn.base.BaseEstimator, optional
        Binary classifier to use. Cloned before fitting. Defaults to
        :func:`default_domain_classifier`.
    min_heldout : int, optional
        Held-out size below which the result is flagged uninterpretable.

    Returns
    -------
    DomainClassifierResult
        Held-out accuracy, the balanced sample count and the two-sided binomial
        p-value against a null accuracy of one half.

    Raises
    ------
    ValueError
        If the feature dimensions disagree, or if either window is too small to
        be split into a training half and a non-empty held-out half.

    Notes
    -----
    The procedure is the one in Rabanser et al.: both windows are partitioned
    into halves, the first halves train a classifier to tell source from
    target, and the second halves are scored. Two departures are ours. The
    held-out set is balanced by subsampling the larger half, without which the
    majority rate would exceed one half and the null of 0.5 would be wrong.
    Results below ``min_heldout`` are flagged, because the same paper reports
    the method fails in that regime and our 2023 window reaches it.
    """
    source = np.asarray(x_source)
    target = np.asarray(x_target)
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError(
            f"both windows must be 2-D, got shapes {source.shape} and {target.shape}"
        )
    if source.shape[1] != target.shape[1]:
        raise ValueError(
            f"feature dimensions disagree: {source.shape[1]} against {target.shape[1]}"
        )
    if source.shape[0] < 2 or target.shape[0] < 2:
        raise ValueError(
            "each window needs at least two samples to be split into halves, got "
            f"{source.shape[0]} and {target.shape[0]}"
        )

    rng = np.random.default_rng(seed)
    source_train_idx, source_test_idx = _halve(source.shape[0], rng)
    target_train_idx, target_test_idx = _halve(target.shape[0], rng)

    # Balancing the held-out set is what makes 0.5 the correct null; without it
    # the majority-window rate alone would beat chance.
    n_balanced = int(min(source_test_idx.size, target_test_idx.size))
    if n_balanced < 1:
        raise ValueError(
            "the held-out half of at least one window is empty; increase the "
            "window size or widen the cut-point"
        )
    discarded = int(
        (source_test_idx.size - n_balanced) + (target_test_idx.size - n_balanced)
    )
    source_test_idx = rng.permutation(source_test_idx)[:n_balanced]
    target_test_idx = rng.permutation(target_test_idx)[:n_balanced]

    x_train = np.vstack([source[source_train_idx], target[target_train_idx]])
    y_train = np.concatenate(
        [
            np.zeros(source_train_idx.size, dtype=np.int8),
            np.ones(target_train_idx.size, dtype=np.int8),
        ]
    )
    x_heldout = np.vstack([source[source_test_idx], target[target_test_idx]])
    y_heldout = np.concatenate(
        [np.zeros(n_balanced, dtype=np.int8), np.ones(n_balanced, dtype=np.int8)]
    )

    model = default_domain_classifier(seed) if estimator is None else clone(estimator)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_heldout)

    n_heldout = int(y_heldout.size)
    n_correct = int(np.sum(y_pred == y_heldout))
    accuracy = n_correct / n_heldout
    p_value = float(
        stats.binomtest(n_correct, n_heldout, p=0.5, alternative="two-sided").pvalue
    )

    interpretable = n_heldout >= min_heldout
    warning = None
    if not interpretable:
        warning = (
            f"held-out set has {n_heldout} samples, below the threshold of "
            f"{min_heldout}; Rabanser et al. report that the domain classifier "
            "performs badly at or below 100 samples, so this p-value must not "
            "be read as evidence for or against covariate shift"
        )

    return DomainClassifierResult(
        accuracy=accuracy,
        n_heldout=n_heldout,
        n_correct=n_correct,
        p_value=p_value,
        n_train=int(y_train.size),
        n_source_available=int(source.shape[0]),
        n_target_available=int(target.shape[0]),
        n_heldout_discarded=discarded,
        interpretable=interpretable,
        warning=warning,
    )


def correlate_residual_with_drift(
    residuals: np.ndarray,
    drift_statistic: np.ndarray,
    method: Literal["spearman", "pearson"] = "spearman",
) -> dict[str, Any]:
    """Correlate the per-cell residual with an independent drift statistic.

    Parameters
    ----------
    residuals : numpy.ndarray of shape (n_cells,)
        Residual ``R`` for each model-by-cut-point cell.
    drift_statistic : numpy.ndarray of shape (n_cells,)
        An independent drift measurement per cell, for instance the excess
        accuracy of the domain classifier or a distributional distance between
        the training and the test window.
    method : {"spearman", "pearson"}, optional
        Correlation to compute. Spearman is the default because neither
        quantity is expected to be linearly related to the other and the cell
        count is small.

    Returns
    -------
    dict
        Keys ``method``, ``statistic``, ``p_value`` and ``n_cells``.

    Raises
    ------
    ValueError
        If the two arrays have different lengths or fewer than three finite
        pairs.
    """
    left = np.asarray(residuals, dtype=float).ravel()
    right = np.asarray(drift_statistic, dtype=float).ravel()
    if left.size != right.size:
        raise ValueError(
            f"length mismatch: {left.size} residuals against {right.size} statistics"
        )
    finite = np.isfinite(left) & np.isfinite(right)
    if int(finite.sum()) < 3:
        raise ValueError(
            f"need at least three finite pairs, got {int(finite.sum())}"
        )

    if method == "spearman":
        result = stats.spearmanr(left[finite], right[finite])
    elif method == "pearson":
        result = stats.pearsonr(left[finite], right[finite])
    else:
        raise ValueError(f"unknown method {method!r}")

    return {
        "method": method,
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "n_cells": int(finite.sum()),
    }


def bootstrap_gap_distribution(
    reference: tuple[np.ndarray, np.ndarray, np.ndarray | None],
    temporal: tuple[np.ndarray, np.ndarray, np.ndarray | None],
    metric: str = "macro_f1",
    n_boot: int = 1000,
    seed: int = 0,
) -> np.ndarray:
    """Resample both arms and return the sampling distribution of their gap.

    Parameters
    ----------
    reference, temporal : tuple of (numpy.ndarray, numpy.ndarray, numpy.ndarray or None)
        ``(y_true, y_pred, sample_weight)`` for one arm. The weight vector is
        the control A1 weighting when that control is active, or ``None``.
    metric : {"macro_f1", "accuracy", "balanced_accuracy"}, optional
        Metric the gap is measured in.
    n_boot : int, optional
        Number of resamples.
    seed : int, optional
        Seed for the resampling generator.

    Returns
    -------
    numpy.ndarray
        ``n_boot`` values of ``metric(reference) - metric(temporal)``.
    """
    from sklearn.metrics import balanced_accuracy_score  # noqa: PLC0415

    def score(y_true, y_pred, weight, labels):
        # The two primary metrics use the bincount implementations, which are
        # exact against scikit-learn and about an order of magnitude cheaper;
        # a per-coalition bootstrap over the whole lattice is not affordable
        # otherwise.
        if metric == "macro_f1":
            return fast_macro_f1(y_true, y_pred, labels, weight)
        if metric == "accuracy":
            return fast_accuracy(y_true, y_pred, weight)
        if metric == "balanced_accuracy":
            return float(balanced_accuracy_score(y_true, y_pred, sample_weight=weight))
        raise ValueError(f"unsupported metric {metric!r}")

    if n_boot < 1:
        raise ValueError(f"n_boot must be positive, got {n_boot}")

    rng = np.random.default_rng(seed)
    prepared = []
    for y_true, y_pred, weight in (reference, temporal):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError("y_true and y_pred differ in length")
        if y_true.shape[0] == 0:
            raise ValueError("cannot bootstrap an empty test window")
        weight = None if weight is None else np.asarray(weight, dtype=np.float64)
        prepared.append((y_true, y_pred, weight, np.unique(y_true)))

    draws = np.empty(n_boot, dtype=np.float64)
    for index in range(n_boot):
        values = []
        for y_true, y_pred, weight, labels in prepared:
            position = rng.integers(0, y_true.shape[0], size=y_true.shape[0])
            values.append(
                score(
                    y_true[position],
                    y_pred[position],
                    None if weight is None else weight[position],
                    labels,
                )
            )
        draws[index] = values[0] - values[1]
    return draws


def _label_correlation_vector(
    features: np.ndarray, labels: np.ndarray, classes: np.ndarray
) -> np.ndarray:
    """Return the flattened point-biserial correlation of each feature with each class."""
    columns = []
    for target_class in classes:
        indicator = (labels == target_class).astype(np.float64)
        spread = indicator.std()
        if spread == 0.0:
            columns.append(np.zeros(features.shape[1]))
            continue
        centred = features - features.mean(axis=0, keepdims=True)
        scale = features.std(axis=0)
        scale[scale == 0.0] = np.inf
        columns.append(
            (centred * (indicator - indicator.mean())[:, None]).mean(axis=0)
            / (scale * spread)
        )
    return np.concatenate(columns)


def conditional_structure_agreement(
    x_early: np.ndarray,
    y_early: np.ndarray,
    x_late: np.ndarray,
    y_late: np.ndarray,
    seed: int,
    n_repeats: int = 500,
) -> dict[str, object]:
    """Compare the feature-label correlation structure of two windows.

    Parameters
    ----------
    x_early, x_late : numpy.ndarray of shape (n_samples, n_features)
        Feature matrices of the earlier and later window.
    y_early, y_late : numpy.ndarray of shape (n_samples,)
        Labels, used only to form the class indicators.
    seed : int
        Seed for the within-window control.
    n_repeats : int, optional
        Number of random splits used to estimate the within-window ceiling, and equally
        the number of class-matched resamples.

    Returns
    -------
    dict
        ``across`` is the Spearman agreement between the two windows' correlation
        vectors.

        Four further fields record the null distribution itself, so that a
        statement about how the observed value sits inside it can be checked
        against counts rather than taken on trust:

        ``n_within_draws`` : int
            Number of within-window null draws actually collected. This is at
            most ``n_repeats`` and can be smaller, because a split that leaves
            fewer than two classes on a side is skipped rather than redrawn.
            It is the ``N`` any "k of N draws" claim must quote.
        ``n_within_at_or_below_across`` : int
            Number of those draws whose agreement is less than or equal to the
            observed ``across``. This is the ``k``. Zero means no split of the
            early window against itself ever fell as low as the across-window
            value.
        ``p_within_empirical`` : float
            One-sided empirical p-value for ``across`` against the within-window
            null, ``(n_within_at_or_below_across + 1) / (n_within_draws + 1)``.
            The added one is the standard Monte-Carlo correction of Davison and
            Hinkley: a null estimated from a finite sample cannot support a
            p-value of exactly zero, so the smallest reportable value is
            ``1 / (n_within_draws + 1)`` and must be reported as an upper bound
            rather than as an exact figure. ``nan`` when no draw was collected
            or ``across`` is not finite.
        ``n_matched_draws`` : int
            Number of class-matched resamples that ``across_matched`` averages
            over, reported for the same reason: the mean alone does not say how
            many draws entered it.

    Notes
    -----
    A domain classifier sees only ``p(x)``: it cannot separate covariate shift
    from label shift and it cannot observe ``p(y|x)`` at all, so it can never on
    its own license a concept-drift conclusion. This statistic is the
    conditional counterpart. It is computed on the shared label space with the
    class prior already fixed, so a change in it is a change in the relationship
    between features and labels rather than in how often each label occurs.
    """
    from scipy import stats  # noqa: PLC0415

    rng = np.random.default_rng(seed)
    x_early = np.asarray(x_early, dtype=np.float64)
    x_late = np.asarray(x_late, dtype=np.float64)
    y_early = np.asarray(y_early)
    y_late = np.asarray(y_late)

    shared = np.intersect1d(np.unique(y_early), np.unique(y_late))
    if shared.size < 2:
        raise ValueError(
            f"only {shared.size} class(es) shared between the windows; the "
            "conditional structure is not defined"
        )

    keep = (x_early.std(axis=0) > 0) & (x_late.std(axis=0) > 0)
    if int(keep.sum()) < 2:
        raise ValueError("fewer than two features vary in both windows")
    x_early, x_late = x_early[:, keep], x_late[:, keep]

    early_mask = np.isin(y_early, shared)
    late_mask = np.isin(y_late, shared)
    x_early, y_early = x_early[early_mask], y_early[early_mask]
    x_late, y_late = x_late[late_mask], y_late[late_mask]

    late_vector = _label_correlation_vector(x_late, y_late, shared)
    across = float(
        stats.spearmanr(
            _label_correlation_vector(x_early, y_early, shared), late_vector
        ).statistic
    )

    # The raw statistic conflates a change in p(y|x) with a change in the class
    # composition of the window, because a correlation with a class indicator
    # depends on how often that class occurs. Resampling the early window to the
    # later window's class proportions removes the composition component, so
    # what remains is attributable to the conditional itself.
    shares = np.array([float((y_late == c).mean()) for c in shared])
    target = min(y_early.shape[0], y_late.shape[0])
    matched: list[float] = []
    pools = {c: np.flatnonzero(y_early == c) for c in shared}
    for _ in range(n_repeats):
        picks = []
        for target_class, share in zip(shared, shares, strict=True):
            pool = pools[target_class]
            take = int(round(share * target))
            if pool.size == 0 or take == 0:
                continue
            picks.append(rng.choice(pool, size=take, replace=take > pool.size))
        if not picks:
            continue
        index = np.concatenate(picks)
        if np.unique(y_early[index]).size < 2:
            continue
        value = float(
            stats.spearmanr(
                _label_correlation_vector(x_early[index], y_early[index], shared),
                late_vector,
            ).statistic
        )
        if value == value:
            matched.append(value)

    half = min(y_late.shape[0], y_early.shape[0] // 2)
    within: list[float] = []
    for _ in range(n_repeats):
        order = rng.permutation(y_early.shape[0])
        left, right = order[:half], order[half : 2 * half]
        try:
            value = float(
                stats.spearmanr(
                    _label_correlation_vector(x_early[left], y_early[left], shared),
                    _label_correlation_vector(x_early[right], y_early[right], shared),
                ).statistic
            )
        except ValueError:
            continue
        if value == value:
            within.append(value)

    within_mean = float(np.mean(within)) if within else float("nan")

    # The summary statistics above cannot substantiate a claim of the form "no
    # null draw reached the observed value": a percentile does not say how many
    # draws produced it. The count and the draw total are returned so that such
    # a claim is checkable from the output alone.
    within_draws = np.asarray(within, dtype=np.float64)
    n_within_draws = int(within_draws.size)
    if n_within_draws > 0 and across == across:
        n_at_or_below = int(np.count_nonzero(within_draws <= across))
        # Davison and Hinkley's add-one estimator. A null of finite size cannot
        # license a p-value of exactly zero, so the floor is 1 / (N + 1) and is
        # an upper bound on the true tail probability, not a measurement of it.
        p_within_empirical = float((n_at_or_below + 1) / (n_within_draws + 1))
    else:
        n_at_or_below = 0
        p_within_empirical = float("nan")

    return {
        "across": across,
        "across_matched": float(np.mean(matched)) if matched else float("nan"),
        "below_within_null": (
            bool(across < np.percentile(within, 2.5)) if within else False
        ),
        "within_mean": within_mean,
        "within_lo": float(np.percentile(within, 2.5)) if within else float("nan"),
        "within_hi": float(np.percentile(within, 97.5)) if within else float("nan"),
        "ratio": across / within_mean if within_mean == within_mean else float("nan"),
        "n_shared_classes": int(shared.size),
        "n_features": int(keep.sum()),
        "n_late": int(y_late.shape[0]),
        "n_within_draws": n_within_draws,
        "n_within_at_or_below_across": n_at_or_below,
        "p_within_empirical": p_within_empirical,
        "n_matched_draws": int(len(matched)),
    }


def expected_calibration_error(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, "object"]:
    """Compute the expected calibration error with its per-bin counts.

    Parameters
    ----------
    probabilities : numpy.ndarray of shape (n_samples, n_classes)
        Predicted class probabilities.
    y_true : numpy.ndarray of shape (n_samples,)
        True labels, as positions into the columns of ``probabilities``.
    n_bins : int, optional
        Number of equal-width confidence bins.

    Returns
    -------
    tuple of (float, pandas.DataFrame)
        The ECE, and a frame with ``bin``, ``n``, ``confidence`` and
        ``accuracy`` for each occupied bin.
    """
    import pandas as pd  # noqa: PLC0415

    probabilities = np.asarray(probabilities, dtype=np.float64)
    y_true = np.asarray(y_true)
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = (predicted == y_true).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    assignment = np.clip(np.digitize(confidence, edges[1:-1]), 0, n_bins - 1)

    rows, total = [], confidence.shape[0]
    error = 0.0
    for index in range(n_bins):
        mask = assignment == index
        count = int(mask.sum())
        if count == 0:
            continue
        bin_confidence = float(confidence[mask].mean())
        bin_accuracy = float(correct[mask].mean())
        error += (count / total) * abs(bin_accuracy - bin_confidence)
        rows.append(
            {
                "bin": f"[{edges[index]:.1f},{edges[index + 1]:.1f})",
                "n": count,
                "confidence": bin_confidence,
                "accuracy": bin_accuracy,
            }
        )
    return float(error), pd.DataFrame.from_records(rows)


def fast_macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: np.ndarray,
    weight: np.ndarray | None = None,
) -> float:
    """Weighted macro-F1 over a pinned label set, computed with bincount.

    Parameters
    ----------
    y_true, y_pred : numpy.ndarray
        Integer label codes for one window.
    labels : numpy.ndarray
        Sorted label set to average over, pinned by the caller.
    weight : numpy.ndarray, optional
        Per-sample weights, or ``None`` for unweighted.

    Returns
    -------
    float
        Macro-averaged F1, with zero contributed by any class whose support and
        predictions are both empty.

    Notes
    -----
    Identical in value to ``sklearn.metrics.f1_score`` with ``average="macro"``,
    ``zero_division=0.0`` and the same pinned ``labels``, but roughly an order
    of magnitude cheaper, which is what makes a per-coalition bootstrap over the
    whole lattice affordable. Predicted classes outside ``labels`` count as
    errors against the true class and contribute to no class's false positives,
    which is the behaviour ``labels`` selects in scikit-learn.
    """
    size = labels.shape[0]
    true_code = np.searchsorted(labels, y_true)
    pred_code = np.searchsorted(labels, y_pred, side="left")
    in_range = pred_code < size
    pred_valid = np.zeros_like(in_range)
    pred_valid[in_range] = labels[pred_code[in_range]] == y_pred[in_range]

    if weight is None:
        weight = np.ones(y_true.shape[0], dtype=np.float64)
    correct = y_true == y_pred

    true_positive = np.bincount(true_code[correct], weight[correct], minlength=size)
    predicted = np.bincount(pred_code[pred_valid], weight[pred_valid], minlength=size)
    actual = np.bincount(true_code, weight, minlength=size)

    denominator = predicted + actual
    scores = np.zeros(size, dtype=np.float64)
    nonzero = denominator > 0
    scores[nonzero] = 2.0 * true_positive[nonzero] / denominator[nonzero]
    return float(scores.mean())


def fast_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weight: np.ndarray | None = None,
) -> float:
    """Weighted accuracy, matching ``sklearn.metrics.accuracy_score``."""
    correct = y_true == y_pred
    if weight is None:
        return float(correct.mean())
    return float(weight[correct].sum() / weight.sum())
