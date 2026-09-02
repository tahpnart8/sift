"""Can A1's weights be recovered after the fact, without being stored?

A1's sample weights are not written to ``predictions.parquet``, and adding a
column would mean restarting the production run. The weights are therefore recomputed
them in the notebook from the config and the panel. This module checks that
claim independently of that recomputation, because one checked only
by the person who wrote it is not checked.

The weights are ``w_c = p_reference(c) / p_window(c)``, normalised to average
one. Two things fix what those two distributions are, and both are easy to get
wrong:

* ``p_reference`` is the class distribution of the **whole analysis panel**, not
  of the training window and not of the post-A2 test set. ``apply_controls``
  computes it as ``panel[target].value_counts(normalize=True)`` before any
  control has filtered anything.
* ``p_window`` is the class distribution of the test half **after** A2 has run,
  since the contract order is ``b2_axis, a2_labels, a1_prior, b1_fs``. The
  stored ``y_true`` already reflects that, so it is the right thing to count.

The practical consequence, and the reason this is recoverable at all: the
recomputation needs only the stored ``y_true`` and the panel. It does not need
the split to be redone.

Every negative control below was measured to give a different answer, so these
tests can actually detect a wrong choice rather than passing regardless.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import requires_mlran
from sift.config import ControlFlags, ExperimentConfig, PanelSpec, SplitSpec
from sift.experiment import label_categories, run_cell
from sift.metrics import compute_metrics

TARGET = "ransomware_family"
CUT = 2019
#: The eight coalitions with a1_prior active. Four of them also have A2 active,
#: which is what exercises p_window being counted after the label restriction.
A1_ON = (1, 3, 5, 7, 9, 11, 13, 15)
A1_OFF = (0, 2, 4, 6, 8, 10, 12, 14)


def decode_labels(y_true_codes, panel: pd.DataFrame, target: str = TARGET) -> pd.Series:
    """Turn stored int16 label codes back into family names."""
    categories = list(label_categories(panel, target))
    return pd.Series(
        pd.Categorical.from_codes(np.asarray(y_true_codes), categories=categories).astype(str)
    )


def recompute_a1_weights(
    y_true_codes, panel: pd.DataFrame, target: str = TARGET
) -> np.ndarray:
    """Independent implementation of the A1 weight formula.

    Deliberately does not call :func:`sift.controls.match_class_prior`; the
    point is to re-derive the number from the definition rather than to confirm
    the pipeline agrees with itself.
    """
    reference = panel[target].value_counts(normalize=True)
    names = decode_labels(y_true_codes, panel, target)
    window = names.value_counts(normalize=True)
    ratio = names.map(lambda value: reference.get(value, 0.0) / window[value])
    weights = ratio.to_numpy(dtype=float)
    return weights * (len(weights) / weights.sum())


def _config(config_id: int, cut: int = CUT) -> ExperimentConfig:
    return ExperimentConfig(
        panel=PanelSpec(),
        split=SplitSpec(design="temporal", cut_year=cut),
        flags=ControlFlags.from_index(config_id),
        model_name="logreg",
        seed=0,
        n_features=20,
    )


# --------------------------------------------------------------------------
# The claim itself
# --------------------------------------------------------------------------
@pytest.mark.parametrize("config_id", A1_ON)
def test_recomputed_weights_reproduce_the_stored_macro_f1(panel, config_id: int) -> None:
    """Measured exact, difference 0.0, on all eight A1-on coalitions."""
    record = run_cell(panel, _config(config_id))
    assert bool(record["a1_prior"]) is True

    y_true = record["predictions"]["y_true"]
    y_pred = record["predictions"]["y_pred"]
    weights = recompute_a1_weights(y_true, panel)

    recomputed = compute_metrics(y_true, y_pred, sample_weight=weights)["macro_f1"]
    assert recomputed == pytest.approx(record["macro_f1"], abs=1e-12)


@pytest.mark.parametrize("config_id", A1_OFF)
def test_unweighted_recomputation_reproduces_the_a1_off_cells(panel, config_id: int) -> None:
    """The other half of the claim: A1-off cells must need no weights at all."""
    record = run_cell(panel, _config(config_id))
    y_true = record["predictions"]["y_true"]
    y_pred = record["predictions"]["y_pred"]
    recomputed = compute_metrics(y_true, y_pred)["macro_f1"]
    assert recomputed == pytest.approx(record["macro_f1"], abs=1e-12)


def test_recomputed_weights_equal_the_vector_the_pipeline_used(panel) -> None:
    """Element-wise, against the weights apply_controls actually produced."""
    from sift.controls import apply_controls

    cfg = _config(1)
    controlled = apply_controls(panel, cfg)
    record = run_cell(panel, cfg)

    mine = recompute_a1_weights(record["predictions"]["y_true"], panel)
    theirs = np.asarray(controlled.test_weights, dtype=float)

    assert mine.shape == theirs.shape
    np.testing.assert_allclose(mine, theirs, rtol=0, atol=1e-12)


# --------------------------------------------------------------------------
# Negative controls: each measured to give a different answer
# --------------------------------------------------------------------------
def test_the_reference_must_be_the_whole_panel_not_the_training_window(panel) -> None:
    """Measured 0.971500721501 against a stored 0.972518360578."""
    from sift.controls import apply_controls

    cfg = _config(3)
    record = run_cell(panel, cfg)
    controlled = apply_controls(panel, cfg)

    y_true = record["predictions"]["y_true"]
    y_pred = record["predictions"]["y_pred"]
    names = decode_labels(y_true, panel)
    window = names.value_counts(normalize=True)

    wrong_reference = controlled.train[TARGET].value_counts(normalize=True)
    ratio = names.map(lambda v: wrong_reference.get(v, 0.0) / window[v]).to_numpy(float)
    wrong = ratio * (len(ratio) / ratio.sum())

    wrong_score = compute_metrics(y_true, y_pred, sample_weight=wrong)["macro_f1"]
    assert wrong_score != pytest.approx(record["macro_f1"], abs=1e-9), (
        "using the training window as the reference gave the same answer, so "
        "this suite cannot tell the two apart"
    )


def test_a_reference_taken_from_the_test_window_degenerates_to_unweighted(panel) -> None:
    """If p_reference equals p_window every weight is one and A1 does nothing.

    Worth pinning because it is the most natural wrong guess: it produces a
    perfectly plausible number that silently equals the A1-off result.
    """
    cfg = _config(3)
    record = run_cell(panel, cfg)
    y_true = record["predictions"]["y_true"]
    y_pred = record["predictions"]["y_pred"]

    names = decode_labels(y_true, panel)
    window = names.value_counts(normalize=True)
    ratio = names.map(lambda v: window.get(v, 0.0) / window[v]).to_numpy(float)

    np.testing.assert_allclose(ratio, np.ones_like(ratio), atol=1e-12)
    degenerate = compute_metrics(y_true, y_pred, sample_weight=ratio)["macro_f1"]
    unweighted = compute_metrics(y_true, y_pred)["macro_f1"]
    assert degenerate == pytest.approx(unweighted, abs=1e-12)
    assert degenerate != pytest.approx(record["macro_f1"], abs=1e-9)


def test_the_label_codes_must_be_decoded_with_label_categories(panel) -> None:
    """A different category order silently rewrites every class.

    Measured 0.969985569986 against a stored 0.972518360578. The codes in
    predictions.parquet are meaningless without the exact ordering that
    label_categories produces from the panel.
    """
    cfg = _config(3)
    record = run_cell(panel, cfg)
    y_true = record["predictions"]["y_true"]
    y_pred = record["predictions"]["y_pred"]

    reversed_categories = list(reversed(list(label_categories(panel, TARGET))))
    names = pd.Series(
        pd.Categorical.from_codes(
            np.asarray(y_true), categories=reversed_categories
        ).astype(str)
    )
    window = names.value_counts(normalize=True)
    reference = panel[TARGET].value_counts(normalize=True)
    ratio = names.map(lambda v: reference.get(v, 0.0) / window[v]).to_numpy(float)
    wrong = ratio * (len(ratio) / ratio.sum())

    wrong_score = compute_metrics(y_true, y_pred, sample_weight=wrong)["macro_f1"]
    assert wrong_score != pytest.approx(record["macro_f1"], abs=1e-9)


def test_the_metric_is_invariant_to_weight_normalisation(panel) -> None:
    """One thing the recomputation cannot get wrong, pinned so nobody re-derives it.

    macro-F1 is scale invariant in sample_weight, so omitting the mean-one
    normalisation changes the weight vector but not the score.
    """
    cfg = _config(1)
    record = run_cell(panel, cfg)
    y_true = record["predictions"]["y_true"]
    y_pred = record["predictions"]["y_pred"]

    normalised = recompute_a1_weights(y_true, panel)
    unnormalised = normalised * 7.5

    assert compute_metrics(y_true, y_pred, sample_weight=unnormalised)[
        "macro_f1"
    ] == pytest.approx(record["macro_f1"], abs=1e-12)


# --------------------------------------------------------------------------
# The same claim on the real panel, where the prior is far more skewed
# --------------------------------------------------------------------------
@requires_mlran
@pytest.mark.parametrize("config_id", (1, 3))
def test_recomputation_holds_on_the_real_panel(config_id: int) -> None:
    """The mock panel has 8 near-balanced families; MLRan has 32 skewed ones."""
    from sift.data import build_panel, load_mlran

    real_panel, _ = build_panel(load_mlran(), PanelSpec())
    record = run_cell(real_panel, _config(config_id))

    y_true = record["predictions"]["y_true"]
    y_pred = record["predictions"]["y_pred"]
    weights = recompute_a1_weights(y_true, real_panel)

    assert weights.min() < 1.0 < weights.max(), "weights are degenerate on real data"
    recomputed = compute_metrics(y_true, y_pred, sample_weight=weights)["macro_f1"]
    assert recomputed == pytest.approx(record["macro_f1"], abs=1e-12)
