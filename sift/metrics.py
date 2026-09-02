"""Classification metrics with an explicitly pinned label set.

Pinning ``labels`` is not a stylistic preference, it is required for the
comparison to mean anything. Left at its default, scikit-learn averages over the
union of the labels present in ``y_true`` and ``y_pred``. A temporal test window
contains markedly fewer families than a random test split, so the two designs
would average over different numbers of classes and their scores would not be
comparable. Pinning the label set to the classes actually present in ``y_true``
makes the denominator a property of the test window rather than of the model's
guesses, and ``k_test`` records it on every result row.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
)

__all__ = [
    "METRIC_NAMES",
    "compute_metrics",
    "bootstrap_distribution",
    "bootstrap_ci",
    "AUT_CUT_YEARS",
    "AUT_TEST_WINDOW",
    "assert_equal_disjoint_slots",
    "aut",
    "aut_label",
]

#: Metric columns written to every result row, in reporting order.
METRIC_NAMES: tuple[str, ...] = ("macro_f1", "balanced_accuracy", "mcc", "accuracy")


def compute_metrics(
    y_true: numpy.ndarray,
    y_pred: numpy.ndarray,
    sample_weight: numpy.ndarray | None = None,
) -> dict[str, float]:
    """Compute the reported metrics for one fit.

    Parameters
    ----------
    y_true : numpy.ndarray
        True labels of the test window.
    y_pred : numpy.ndarray
        Predicted labels, aligned to ``y_true``.
    sample_weight : numpy.ndarray, optional
        Per-sample weights from control A1. When supplied, every metric is
        computed under the reweighted class prior.

    Returns
    -------
    dict
        ``macro_f1``, ``balanced_accuracy``, ``mcc`` and ``accuracy`` as floats,
        plus ``k_test``, the number of classes the macro average was taken over.

    Raises
    ------
    ValueError
        If the inputs differ in length or are empty.
    """
    y_true = numpy.asarray(y_true)
    y_pred = numpy.asarray(y_pred)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            f"y_true holds {y_true.shape[0]} labels but y_pred holds {y_pred.shape[0]}"
        )
    if y_true.shape[0] == 0:
        raise ValueError("cannot compute metrics on an empty test window")

    labels = numpy.unique(y_true)

    return {
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0.0,
                sample_weight=sample_weight,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred, sample_weight=sample_weight)
        ),
        "mcc": float(matthews_corrcoef(y_true, y_pred, sample_weight=sample_weight)),
        "accuracy": float(accuracy_score(y_true, y_pred, sample_weight=sample_weight)),
        "k_test": int(labels.size),
    }


def _metric_from_labels(
    y_true: numpy.ndarray,
    y_pred: numpy.ndarray,
    metric: str,
    labels: numpy.ndarray,
    sample_weight: numpy.ndarray | None = None,
) -> float:
    """Evaluate one metric against a label set fixed by the caller.

    Every branch forwards ``sample_weight``. Control A1 is expressed entirely as
    a weight at metric time, so a branch that dropped it would silently report an
    unweighted quantity for an A1-enabled configuration.
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
        return float(
            balanced_accuracy_score(y_true, y_pred, sample_weight=sample_weight)
        )
    if metric == "mcc":
        return float(matthews_corrcoef(y_true, y_pred, sample_weight=sample_weight))
    raise ValueError(f"unknown metric {metric!r}; expected one of {METRIC_NAMES}")


def bootstrap_distribution(
    y_true: numpy.ndarray,
    y_pred: numpy.ndarray,
    metric: str = "macro_f1",
    n_boot: int = 1000,
    seed: int = 0,
    sample_weight: numpy.ndarray | None = None,
) -> numpy.ndarray:
    """Resample the test window and recompute a metric on each draw.

    The label set is pinned once, to the classes present in the original
    ``y_true``, and reused for every resample. This is the decision the design
    doc requires to be stated: a resample of a small window will routinely miss
    a class entirely, and holding the denominator fixed keeps every draw on the
    same scale as the point estimate. A missing class contributes a genuine zero
    rather than shrinking the average. Resamples are never discarded and
    redrawn, which would bias the distribution upward and silently change the
    estimand.

    Parameters
    ----------
    y_true, y_pred : numpy.ndarray
        Stored predictions for one fit.
    metric : str
        One of :data:`METRIC_NAMES`.
    n_boot : int
        Number of resamples. Exactly this many statistics are returned.
    seed : int
        Seed for the resampling generator.
    sample_weight : numpy.ndarray, optional
        Per-sample weights from control A1, aligned to ``y_true``. Weights are resampled
        by the same index draw as the labels, so each resample carries the weights of
        the rows it actually drew.

    Returns
    -------
    numpy.ndarray
        ``n_boot`` metric values.
    """
    y_true = numpy.asarray(y_true)
    y_pred = numpy.asarray(y_pred)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred differ in length")
    if y_true.shape[0] == 0:
        raise ValueError("cannot bootstrap an empty test window")
    if n_boot < 1:
        raise ValueError(f"n_boot must be positive, got {n_boot}")
    weights = None if sample_weight is None else numpy.asarray(sample_weight)
    if weights is not None and weights.shape[0] != y_true.shape[0]:
        raise ValueError(
            f"sample_weight holds {weights.shape[0]} entries but y_true holds "
            f"{y_true.shape[0]}"
        )

    labels = numpy.unique(y_true)
    rng = numpy.random.default_rng(seed)
    n = y_true.shape[0]
    draws = numpy.empty(n_boot, dtype=numpy.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        draws[i] = _metric_from_labels(
            y_true[idx],
            y_pred[idx],
            metric,
            labels,
            None if weights is None else weights[idx],
        )
    return draws


def bootstrap_ci(
    y_true: numpy.ndarray,
    y_pred: numpy.ndarray,
    metric: str = "macro_f1",
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
    sample_weight: numpy.ndarray | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap interval for one metric on one fit.

    Parameters
    ----------
    y_true, y_pred : numpy.ndarray
        Stored predictions for one fit.
    metric : str
        One of :data:`METRIC_NAMES`.
    n_boot : int
        Number of resamples.
    seed : int
        Seed for the resampling generator.
    alpha : float
        Two-sided miscoverage. ``0.05`` gives a 95 percent interval.
    sample_weight : numpy.ndarray, optional
        Per-sample weights from control A1, forwarded to :func:`bootstrap_distribution`.

    Returns
    -------
    tuple of float
        Lower and upper percentile bounds.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    draws = bootstrap_distribution(
        y_true, y_pred, metric, n_boot, seed, sample_weight=sample_weight
    )
    low, high = numpy.percentile(draws, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return float(low), float(high)


#: Cut years of the four AUT evaluation slots.
#:
#: Each slot trains on everything before its cut and tests on the two years from
#: it, giving ``2016-2017``, ``2018-2019``, ``2020-2021`` and ``2022-2023``.
#: Disjoint, equal in width and contiguous, which is what makes the trapezoidal
#: average well defined.
AUT_CUT_YEARS: tuple[int, ...] = (2016, 2018, 2020, 2022)

#: Width in years of every AUT slot. Equal widths are a precondition, not a
#: preference: see :func:`assert_equal_disjoint_slots`.
AUT_TEST_WINDOW: int = 2


def assert_equal_disjoint_slots(
    cut_years: Sequence[int],
    test_window: int = AUT_TEST_WINDOW,
) -> None:
    """Raise unless the evaluation slots are equal in width, disjoint and evenly spaced.

    The TESSERACT definition places evaluation points at ``W + k * Delta`` and
    then drops ``Delta`` from the sum, which is only valid when every slot has
    the same width and the points are evenly spaced. Overlapping slots break the
    trapezoid outright, because the same sample is then counted in two adjacent
    terms.

    Parameters
    ----------
    cut_years : Sequence of int
        First year of each slot, in any order.
    test_window : int, default :data:`AUT_TEST_WINDOW`
        Width of every slot in years.

    Raises
    ------
    ValueError
        If fewer than two slots are given, if any cut repeats, or if the spacing
        between consecutive cuts is not constant and equal to ``test_window``.

    Examples
    --------
    The section 4.3 series is accepted:

    >>> assert_equal_disjoint_slots((2016, 2018, 2020, 2022), 2)

    The section 4.2 series is not, because a three-year window advancing every
    two years overlaps:

    >>> assert_equal_disjoint_slots(  # doctest: +ELLIPSIS
    ...     (2015, 2017, 2019, 2021, 2023), 3
    ... )
    Traceback (most recent call last):
        ...
    ValueError: AUT is undefined for these slots: ...overlap
    """
    if test_window < 1:
        raise ValueError(f"test_window must be at least 1, got {test_window}")
    years = list(cut_years)
    if len(years) < 2:
        raise ValueError(
            f"AUT needs at least two slots to form a trapezoid, got {len(years)}"
        )
    if len(set(years)) != len(years):
        raise ValueError(f"AUT slots must be distinct, got {years}")

    ordered = sorted(years)
    gaps = {b - a for a, b in zip(ordered, ordered[1:])}
    if len(gaps) > 1:
        raise ValueError(
            f"AUT is undefined for these slots: spacing is uneven, cuts {ordered} "
            f"leave gaps {sorted(gaps)}"
        )
    gap = gaps.pop()
    if gap < test_window:
        raise ValueError(
            f"AUT is undefined for these slots: a window of {test_window} years "
            f"advancing every {gap} years makes consecutive slots overlap"
        )
    if gap > test_window:
        raise ValueError(
            f"AUT is undefined for these slots: a window of {test_window} years "
            f"advancing every {gap} years leaves {gap - test_window} year(s) "
            "uncovered between slots"
        )


def aut(scores: Sequence[float]) -> float:
    """Area under time, the trapezoidal average of a metric over evaluation slots.

    Implements the TESSERACT definition

    Parameters
    ----------
    scores : Sequence of float
        Metric value for each slot, in chronological order. Validate the slot
        series with :func:`assert_equal_disjoint_slots` before calling.

    Returns
    -------
    float
        The trapezoidal average, on the same scale as the input scores.

    Raises
    ------
    ValueError
        If fewer than two scores are given, or any score is not finite.

    Notes
    -----
    Every slot carries equal weight regardless of how many samples it holds. On
    this panel the four slots hold 79, 134, 567 and 157 samples, so the first
    slot, the smallest and the noisiest, counts as much as the third, which is
    seven times larger. That is a property of the definition rather than of this
    implementation, and it must be said aloud whenever AUT is reported.

    Examples
    --------
    A flat series returns its own level:

    >>> aut([0.5, 0.5, 0.5])
    0.5
    """
    values = [float(score) for score in scores]
    if len(values) < 2:
        raise ValueError(f"AUT needs at least two slots, got {len(values)}")
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"AUT requires finite scores, got {values}")
    trapezoids = [
        (later + earlier) / 2.0 for earlier, later in zip(values, values[1:])
    ]
    return math.fsum(trapezoids) / (len(values) - 1)


def aut_label(
    metric: str = "macro-F1",
    cut_years: Sequence[int] = AUT_CUT_YEARS,
    test_window: int = AUT_TEST_WINDOW,
) -> str:
    """Return the TESSERACT reporting label for an AUT figure.

    Parameters
    ----------
    metric : str, default 'macro-F1'
        Name of the underlying metric, written as it should appear in the paper.
    cut_years : Sequence of int, default :data:`AUT_CUT_YEARS`
        Slot cut years, used to derive the total span.
    test_window : int, default :data:`AUT_TEST_WINDOW`
        Width of every slot in years.

    Returns
    -------
    str
        For example ``'AUT(macro-F1, 8y)'`` for four two-year slots.

    Examples
    --------
    >>> aut_label()
    'AUT(macro-F1, 8y)'
    """
    span = len(list(cut_years)) * test_window
    return f"AUT({metric}, {span}y)"
