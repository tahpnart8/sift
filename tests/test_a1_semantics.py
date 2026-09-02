"""Control A1 is an intervention on the measurement, not on the learner.

The R1 suite stayed green while A1 was moved from the training side to the test
side, which means the two opposite semantics were indistinguishable to it.  This
module removes that freedom: it pins A1 as a test-side reweighting and would
fail if the semantics were reversed, or if A1 were made to resample the training
window, or if it silently stopped affecting the metric at all.

Four properties, from CONTRACT.md section 1 and design doc section 3:

1. no ``sample_weight`` may reach ``estimator.fit`` under any coalition;
2. A1 active changes the reported score, otherwise the control is inert;
3. A1 active must not remove test rows, which is what separates it from A2;
4. all four models, the MLP included, must survive all sixteen coalitions.
"""

from __future__ import annotations

import numpy as np
import pytest

from sift.config import ControlFlags, ExperimentConfig, PanelSpec, SplitSpec
from sift.controls import apply_controls
from sift.experiment import run_cell
from sift.metrics import compute_metrics

CUT = 2019
FIT_CALLS: list[dict[str, object]] = []


def _config(flags: ControlFlags, model: str = "logreg", cut: int = CUT) -> ExperimentConfig:
    return ExperimentConfig(
        panel=PanelSpec(),
        split=SplitSpec(design="temporal", cut_year=cut),
        flags=flags,
        model_name=model,
        seed=0,
        n_features=20,
    )


class _FitSpy:
    """Delegating wrapper that records exactly how ``fit`` was invoked."""

    def __init__(self, inner: object) -> None:
        self.inner = inner

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.inner.get_params(deep)

    def fit(self, X, y, **kwargs):
        FIT_CALLS.append(
            {
                "kwargs": sorted(kwargs),
                "X": np.array(X, copy=True),
                "y": np.array(y, copy=True),
            }
        )
        return self.inner.fit(X, y, **kwargs)

    def predict(self, X):
        return self.inner.predict(X)


@pytest.fixture()
def fit_spy(monkeypatch: pytest.MonkeyPatch):
    """Wrap the model factory and disable the fit cache.

    Disabling the cache is essential rather than tidy.  ``run_cell`` returns a
    cached record before it ever builds an estimator, so a cache hit would let
    every assertion below pass without a single fit taking place.
    """
    import sift.cache as cache_module
    import sift.models as models_module

    real_factory = models_module.build_model

    def spying_factory(name: str, seed: int, **kwargs):
        return _FitSpy(real_factory(name, seed, **kwargs))

    monkeypatch.setattr(models_module, "build_model", spying_factory)
    monkeypatch.setattr(cache_module, "get", lambda identifier: None)
    monkeypatch.setattr(cache_module, "put", lambda identifier, record: None)

    FIT_CALLS.clear()
    yield FIT_CALLS
    FIT_CALLS.clear()


def test_no_sample_weight_reaches_fit_under_any_coalition(panel, fit_spy) -> None:
    """A1 must never become a training-side intervention.

    Sixteen coalitions, eight of which have A1 active.  If any of them started
    passing weights into the learner, A1 would be measuring a mitigation rather
    than a confound, and the MLP arm would break as well because the
    scikit-learn MLP takes no sample_weight at fit time.
    """
    for index in range(16):
        run_cell(panel, _config(ControlFlags.from_index(index)))

    assert len(fit_spy) == 16, "a cached record slipped through and skipped a fit"
    for index, call in enumerate(fit_spy):
        assert call["kwargs"] == [], (
            "coalition {0} passed {1} into fit; A1 is a test-side control and "
            "the learner must see no weights".format(index, call["kwargs"])
        )


def test_a1_does_not_change_what_the_estimator_is_fitted_on(panel, fit_spy) -> None:
    """Same training matrix and same training labels with A1 on and A1 off."""
    for a1 in (False, True):
        run_cell(panel, _config(ControlFlags(a1_prior=a1)))

    without, with_a1 = fit_spy
    np.testing.assert_array_equal(without["X"], with_a1["X"])
    np.testing.assert_array_equal(without["y"], with_a1["y"])


def test_a1_changes_the_reported_score(panel) -> None:
    """An inert control would take a Shapley share of zero for the wrong reason.

    Accuracy is the sensitive metric here.  Design doc section 3 says macro-F1
    is deliberately close to prior-invariant, which is why accuracy is carried
    as the counter-metric, so accuracy is what this asserts on.
    """
    off = run_cell(panel, _config(ControlFlags(a1_prior=False)))
    on = run_cell(panel, _config(ControlFlags(a1_prior=True)))
    assert abs(on["accuracy"] - off["accuracy"]) > 1e-6, (
        "A1 left accuracy untouched; either the weights are all one or they are "
        "not reaching the metric"
    )


def test_a1_weights_are_non_degenerate_and_average_to_one(panel) -> None:
    """A vector of ones would satisfy the test above only by accident."""
    controlled = apply_controls(panel, _config(ControlFlags(a1_prior=True)))
    weights = np.asarray(controlled.test_weights, dtype=float)
    assert weights.shape == (len(controlled.test_idx),)
    assert np.all(weights > 0)
    assert weights.mean() == pytest.approx(1.0, abs=1e-9)
    assert weights.max() - weights.min() > 1e-6, "A1 produced a flat weight vector"


def test_the_metric_function_actually_consumes_sample_weight() -> None:
    """Guards the other half of the path: metrics must honour the weights."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 0, 1, 1, 0, 0])
    flat = compute_metrics(y_true, y_pred)
    skewed = compute_metrics(
        y_true, y_pred, sample_weight=np.array([0.25] * 4 + [1.75] * 4)
    )
    assert abs(skewed["accuracy"] - flat["accuracy"]) > 1e-6


def test_a1_does_not_remove_test_rows(panel) -> None:
    """This is what separates A1 from A2.

    A2 restricts the label space and legitimately drops rows.  A1 reweights and
    must keep every one.  Checked across all eight settings of the other three
    flags so the property cannot hold by luck at one corner of the lattice.
    """
    for a2 in (False, True):
        for b1 in (False, True):
            for b2 in (False, True):
                off = ControlFlags(a1_prior=False, a2_labels=a2, b1_fs=b1, b2_axis=b2)
                on = ControlFlags(a1_prior=True, a2_labels=a2, b1_fs=b1, b2_axis=b2)
                left = run_cell(panel, _config(off))
                right = run_cell(panel, _config(on))
                assert left["n_test"] == right["n_test"], (
                    "A1 changed n_test from {0} to {1} at a2={2} b1={3} b2={4}; "
                    "A1 must not drop rows".format(
                        left["n_test"], right["n_test"], a2, b1, b2
                    )
                )
                assert left["n_train"] == right["n_train"]


def test_a2_does_remove_test_rows_so_the_previous_test_can_discriminate(
    panel_with_novel_family, no_cache
) -> None:
    """Proves the A1 test above can tell A1 and A2 apart.

    Run on the plain synthetic panel this assertion fails, because that fixture
    spreads every family across the whole time range and A2 removes nothing.
    That is a property of the mock, not of the code, and it means A2 cannot be
    exercised without a panel that carries a genuinely unseen family.
    """
    off = run_cell(panel_with_novel_family, _config(ControlFlags(a2_labels=False)))
    on = run_cell(panel_with_novel_family, _config(ControlFlags(a2_labels=True)))
    assert on["n_test"] < off["n_test"], (
        "A2 removed nothing, so test_a1_does_not_remove_test_rows would pass "
        "even if A1 and A2 were swapped"
    )


def test_a1_still_removes_no_rows_when_a_novel_family_is_present(
    panel_with_novel_family, no_cache
) -> None:
    """The A1 invariant must survive the panel on which A2 actually bites."""
    off = run_cell(panel_with_novel_family, _config(ControlFlags(a1_prior=False)))
    on = run_cell(panel_with_novel_family, _config(ControlFlags(a1_prior=True)))
    assert off["n_test"] == on["n_test"]
    assert off["n_train"] == on["n_train"]


@pytest.mark.slow
@pytest.mark.parametrize("model_name", ["logreg", "random_forest", "lightgbm", "mlp"])
def test_every_model_survives_all_sixteen_coalitions(panel, model_name: str) -> None:
    """The MLP arm is the one the old training-side form of A1 used to block."""
    for index in range(16):
        record = run_cell(panel, _config(ControlFlags.from_index(index), model=model_name))
        assert np.isfinite(record["macro_f1"])
        assert record["n_test"] > 0
