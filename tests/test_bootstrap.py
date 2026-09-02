"""Bootstrap confidence intervals for macro-F1.

Design doc section 7 puts every confidence interval on the stored prediction
table.  The statistical hazard is specific: macro-F1 averages over classes, and
a resample of a 60 sample test window will routinely contain zero instances of
some class.  Whatever the project does there must be a stated, deterministic
rule.  Discarding and redrawing such resamples is not acceptable because it
biases the estimate upward and changes the estimand silently, so the suite
asserts that the resample count is exactly what was asked for.
"""

from __future__ import annotations

import numpy as np
import pytest

from sift.metrics import bootstrap_ci, compute_metrics


def _toy_predictions(n: int = 200, n_classes: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, n_classes, size=n)
    noise = rng.random(n) < 0.3
    y_pred = np.where(noise, rng.integers(0, n_classes, size=n), y_true)
    return y_true, y_pred


def test_bootstrap_interval_brackets_the_point_estimate() -> None:
    y_true, y_pred = _toy_predictions()
    point = compute_metrics(y_true, y_pred)["macro_f1"]
    low, high = bootstrap_ci(y_true, y_pred, metric="macro_f1", n_boot=500, seed=0)
    assert low <= point <= high
    assert 0.0 <= low <= high <= 1.0


def test_bootstrap_is_deterministic_for_a_fixed_seed() -> None:
    y_true, y_pred = _toy_predictions()
    first = bootstrap_ci(y_true, y_pred, metric="macro_f1", n_boot=300, seed=7)
    second = bootstrap_ci(y_true, y_pred, metric="macro_f1", n_boot=300, seed=7)
    assert first == second


def test_bootstrap_responds_to_the_seed() -> None:
    """Two different seeds must not give a bit identical interval."""
    y_true, y_pred = _toy_predictions()
    first = bootstrap_ci(y_true, y_pred, metric="macro_f1", n_boot=300, seed=1)
    second = bootstrap_ci(y_true, y_pred, metric="macro_f1", n_boot=300, seed=2)
    assert first != second


def test_bootstrap_survives_a_window_where_classes_are_nearly_empty() -> None:
    """Sixty samples over thirty two classes is cut 2023, the worst real cell.

    Most resamples miss several classes entirely.  The call must return a finite
    interval rather than raise or produce nan.
    """
    rng = np.random.default_rng(0)
    y_true = np.concatenate([np.arange(32), rng.integers(0, 32, size=28)])
    y_pred = rng.integers(0, 32, size=60)
    low, high = bootstrap_ci(y_true, y_pred, metric="macro_f1", n_boot=400, seed=0)
    assert np.isfinite(low) and np.isfinite(high)
    assert low <= high


def test_bootstrap_keeps_every_requested_resample() -> None:
    """No discard and redraw.  n_boot resamples in, n_boot statistics out."""
    from sift.metrics import bootstrap_distribution

    rng = np.random.default_rng(0)
    y_true = np.concatenate([np.arange(32), rng.integers(0, 32, size=28)])
    y_pred = rng.integers(0, 32, size=60)
    draws = bootstrap_distribution(y_true, y_pred, metric="macro_f1", n_boot=400, seed=0)
    assert len(draws) == 400
    assert np.all(np.isfinite(draws))


def test_interval_is_wider_for_the_smaller_test_window() -> None:
    """Uncertainty must visibly grow at the low confidence cut points."""
    y_true_large, y_pred_large = _toy_predictions(n=600, n_classes=8, seed=1)
    y_true_small, y_pred_small = _toy_predictions(n=60, n_classes=8, seed=1)
    wide_low, wide_high = bootstrap_ci(
        y_true_small, y_pred_small, metric="macro_f1", n_boot=500, seed=0
    )
    tight_low, tight_high = bootstrap_ci(
        y_true_large, y_pred_large, metric="macro_f1", n_boot=500, seed=0
    )
    assert (wide_high - wide_low) > (tight_high - tight_low)


def test_a_perfect_prediction_gives_a_zero_width_interval() -> None:
    """Perfect prediction leaves macro-F1 with no sampling variability."""
    y_true = np.tile(np.arange(4), 40)
    low, high = bootstrap_ci(y_true, y_true.copy(), metric="macro_f1", n_boot=200, seed=0)
    assert low == pytest.approx(1.0, abs=1e-12)
    assert high == pytest.approx(1.0, abs=1e-12)
