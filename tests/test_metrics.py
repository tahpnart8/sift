"""Metric configuration, the single highest-risk item in the design.

CONTRACT.md section 5 makes two things mandatory: ``labels`` pinned to
``numpy.unique(y_true)`` and ``zero_division`` set explicitly to 0.0.  Left at
the sklearn default the macro average is taken over the union of the labels in
``y_true`` and ``y_pred``, so the random design and the temporal design would
average over different numbers of classes and Delta would be partly a metric
artefact rather than a measurement.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from sklearn.exceptions import UndefinedMetricWarning

from sift.metrics import compute_metrics


def test_macro_f1_matches_a_hand_computed_value() -> None:
    """y_true = [0,0,1,1,2], y_pred = [0,1,1,1,0].

    class 0: precision 1/2, recall 1/2, F1 0.5
    class 1: precision 2/3, recall 1,   F1 0.8
    class 2: no true positive and no prediction, F1 0.0 under zero_division=0
    macro F1 = (0.5 + 0.8 + 0.0) / 3 = 0.4333...
    """
    y_true = np.array([0, 0, 1, 1, 2])
    y_pred = np.array([0, 1, 1, 1, 0])
    out = compute_metrics(y_true, y_pred)
    assert out["macro_f1"] == pytest.approx(1.3 / 3.0, abs=1e-12)


def test_macro_f1_ignores_predicted_labels_absent_from_y_true() -> None:
    """The pinned label set is what separates a correct macro from the default.

    y_true = [0,0,1,1], y_pred = [0,0,1,3].  With labels pinned to {0, 1} the
    macro is (1.0 + 2/3) / 2 = 0.8333.  With sklearn's default label inference
    the phantom class 3 enters the denominator and the answer becomes 0.5555.
    """
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 3])
    out = compute_metrics(y_true, y_pred)
    assert out["macro_f1"] == pytest.approx((1.0 + 2.0 / 3.0) / 2.0, abs=1e-12)
    assert out["macro_f1"] != pytest.approx((1.0 + 2.0 / 3.0) / 3.0, abs=1e-6)


def test_k_test_counts_the_classes_actually_averaged_over() -> None:
    """k_test must be recorded on every row, per CONTRACT.md section 5."""
    y_true = np.array([0, 0, 1, 1, 2])
    y_pred = np.array([0, 1, 1, 1, 0])
    assert compute_metrics(y_true, y_pred)["k_test"] == 3

    y_true_narrow = np.array([0, 0, 1, 1])
    y_pred_wide = np.array([0, 0, 1, 3])
    assert compute_metrics(y_true_narrow, y_pred_wide)["k_test"] == 2


def test_a_class_with_no_prediction_scores_zero_without_warning() -> None:
    """zero_division must be the literal 0.0, never the sklearn "warn" default.

    Leaving it at "warn" numerically behaves like zero but emits
    UndefinedMetricWarning on every fit.  Across 1920 fits that noise hides real
    problems, so the suite promotes the warning to an error.

    class 0: precision 2/3, recall 1, F1 0.8
    class 1: precision 2/3, recall 1, F1 0.8
    class 2: never predicted and never correct, F1 0.0
    macro F1 = 1.6 / 3
    """
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 0, 1])
    with warnings.catch_warnings():
        warnings.simplefilter("error", UndefinedMetricWarning)
        out = compute_metrics(y_true, y_pred)
    assert out["macro_f1"] == pytest.approx(1.6 / 3.0, abs=1e-12)


def test_all_four_contract_metrics_are_returned() -> None:
    """CONTRACT.md section 6 names four metric columns; section 7 names their roles."""
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    out = compute_metrics(y_true, y_pred)
    for key in ("macro_f1", "accuracy", "balanced_accuracy", "mcc"):
        assert key in out, f"metric {key} missing from compute_metrics output"


def test_metrics_stay_inside_their_valid_ranges() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        y_true = rng.integers(0, 6, size=120)
        y_pred = rng.integers(0, 6, size=120)
        out = compute_metrics(y_true, y_pred)
        assert 0.0 <= out["macro_f1"] <= 1.0
        assert 0.0 <= out["accuracy"] <= 1.0
        assert 0.0 <= out["balanced_accuracy"] <= 1.0
        assert -1.0 <= out["mcc"] <= 1.0


def test_a_perfect_prediction_scores_one_on_every_metric() -> None:
    y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    out = compute_metrics(y_true, y_true.copy())
    assert out["macro_f1"] == pytest.approx(1.0, abs=1e-12)
    assert out["accuracy"] == pytest.approx(1.0, abs=1e-12)
    assert out["balanced_accuracy"] == pytest.approx(1.0, abs=1e-12)
    assert out["mcc"] == pytest.approx(1.0, abs=1e-12)


def test_macro_f1_and_balanced_accuracy_diverge_on_a_skewed_panel() -> None:
    """A regression guard for the finding of macro-F1 0.244 against bal-acc 0.471.

    Six classes, one dominant.  Every rare class is recalled perfectly but with
    precision 1/20, and the dominant class is recalled at 0.05 with precision 1.
    Every per-class F1 is therefore 2*(1/20)/(1 + 1/20) = 0.0952, giving macro-F1
    0.0952, while balanced accuracy, being macro recall, is (0.05 + 5)/6 = 0.8417.

    If these two ever agree for this input, one of them is on the wrong code
    path and the paper's "conclusion does not depend on one metric" claim is
    unsupported.
    """
    y_true = np.array([0] * 100 + [1, 2, 3, 4, 5])
    y_pred = np.array([0] * 5 + [1] * 19 + [2] * 19 + [3] * 19 + [4] * 19 + [5] * 19 + [1, 2, 3, 4, 5])
    assert y_true.shape == y_pred.shape
    out = compute_metrics(y_true, y_pred)
    assert out["macro_f1"] == pytest.approx(2 * 0.05 / 1.05, abs=1e-12)
    assert out["balanced_accuracy"] == pytest.approx(5.05 / 6.0, abs=1e-12)
    assert out["balanced_accuracy"] - out["macro_f1"] > 0.5
