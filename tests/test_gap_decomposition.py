"""The three-way decomposition of the reported gap.

From the ``SplitSpec`` docstring in ``sift/config.py``, with
``Delta(d) = M_d - M_temporal`` at the baseline coalition::

    size effect              = Delta(random) - Delta(random_matched)
    class-composition effect = Delta(random_matched) - Delta(random_fully_matched)
    remainder                = Delta(random_fully_matched)

The three telescope onto ``Delta(random)``, the naively reported gap.  The
identity is exact arithmetic, not an empirical finding, so the tolerance is
floating point rather than statistical.

Both the size term and the class-composition term may legitimately be negative,
and a clamp anywhere would destroy a real result silently:

* the class term measured -0.033 at cut 2023 and +0.237 at cut 2015;
* the size term goes negative wherever the temporal design trains on more than
  the naive random reference, which at cut 2023 it does, 1317 against 1032.

``ASSUMED API``: the suite expects
``sift.reporting.build_gap_decomposition(metrics, model=..., cut=...,
metric_column=...)`` returning a mapping with keys ``size``,
``class_composition``, ``remainder`` and ``delta_reported``.  A failure on that
name today is the signal.
"""

from __future__ import annotations

import pandas as pd
import pytest

from conftest import build_metrics, metrics_row
from sift.experiment import BASELINE_CONFIG_ID, REFERENCE_ROLE

TOL = 1e-12


def reference_three_way(
    m_temporal: float,
    m_random: float,
    m_random_matched: float,
    m_fully_matched: float,
) -> dict[str, float]:
    """The definition written out, used to check the production path against."""
    delta_random = m_random - m_temporal
    delta_matched = m_random_matched - m_temporal
    delta_fully = m_fully_matched - m_temporal
    return {
        "size": delta_random - delta_matched,
        "class_composition": delta_matched - delta_fully,
        "remainder": delta_fully,
        "delta_reported": delta_random,
    }


def _metrics_with_references(
    m_temporal: float,
    m_random: float,
    m_random_matched: float,
    m_fully_matched: float,
    cut: int = 2019,
) -> pd.DataFrame:
    """A lattice table plus the two baseline-only reference rows."""
    temporal_scores = {c: m_temporal for c in range(16)}
    fully_scores = {c: m_fully_matched for c in range(16)}
    frame = build_metrics(temporal_scores, fully_scores, cut=cut)

    extra = pd.DataFrame(
        [
            metrics_row(
                design="random",
                role=REFERENCE_ROLE,
                config_id=BASELINE_CONFIG_ID,
                cut=cut,
                macro_f1=m_random,
                fit_id="f_random",
            ),
            metrics_row(
                design="random_matched",
                role=REFERENCE_ROLE,
                config_id=BASELINE_CONFIG_ID,
                cut=cut,
                macro_f1=m_random_matched,
                fit_id="f_random_matched",
            ),
        ],
        columns=frame.columns,
    )
    return pd.concat([frame, extra], ignore_index=True)


# --------------------------------------------------------------------------
# The identity, checked on the definition itself
# --------------------------------------------------------------------------
CASES = [
    # temporal, random, random_matched, random_fully_matched
    (0.20, 0.76, 0.60, 0.55),  # both terms positive, the ordinary early cut
    (0.62, 0.55, 0.61, 0.64),  # size term negative: temporal trains on more
    (0.40, 0.70, 0.66, 0.69),  # class term negative
    (0.50, 0.50, 0.50, 0.50),  # degenerate: every term zero
    (0.31, 0.29, 0.72, 0.13),  # both terms negative and large
]


@pytest.mark.parametrize("scores", CASES)
def test_the_three_terms_sum_to_the_reported_gap(scores) -> None:
    terms = reference_three_way(*scores)
    total = terms["size"] + terms["class_composition"] + terms["remainder"]
    assert total == pytest.approx(terms["delta_reported"], abs=TOL)


def test_the_class_composition_term_is_not_clipped_at_zero() -> None:
    """Measured -0.033 at cut 2023. A clamp would report 0.000 instead."""
    terms = reference_three_way(0.40, 0.70, 0.66, 0.69)
    assert terms["class_composition"] < 0
    assert terms["class_composition"] == pytest.approx(-0.03, abs=1e-9)


def test_the_size_term_is_not_clipped_at_zero() -> None:
    """Negative wherever the temporal arm trains on more than the naive random one."""
    terms = reference_three_way(0.62, 0.55, 0.61, 0.64)
    assert terms["size"] < 0


# --------------------------------------------------------------------------
# The same properties demanded of the production path
# --------------------------------------------------------------------------
@pytest.mark.parametrize("scores", CASES)
def test_production_decomposition_sums_to_the_reported_gap(scores) -> None:
    from sift.reporting import build_gap_decomposition

    metrics = _metrics_with_references(*scores)
    terms = build_gap_decomposition(metrics, model="logreg", cut=2019)
    total = terms["size"] + terms["class_composition"] + terms["remainder"]
    assert total == pytest.approx(terms["delta_reported"], abs=TOL)


def test_production_decomposition_matches_the_written_definition() -> None:
    from sift.reporting import build_gap_decomposition

    scores = (0.20, 0.76, 0.60, 0.55)
    metrics = _metrics_with_references(*scores)
    expected = reference_three_way(*scores)
    produced = build_gap_decomposition(metrics, model="logreg", cut=2019)
    for key, value in expected.items():
        assert produced[key] == pytest.approx(value, abs=TOL)


def test_production_decomposition_preserves_a_negative_class_term() -> None:
    """The clamp guard, on the production path rather than on the definition."""
    from sift.reporting import build_gap_decomposition

    metrics = _metrics_with_references(0.40, 0.70, 0.66, 0.69)
    terms = build_gap_decomposition(metrics, model="logreg", cut=2019)
    assert terms["class_composition"] < 0, (
        "the class-composition term was clipped; it measured -0.033 at cut 2023 "
        "and a clamp would silently report that real effect as absent"
    )


def test_production_decomposition_preserves_a_negative_size_term() -> None:
    from sift.reporting import build_gap_decomposition

    metrics = _metrics_with_references(0.62, 0.55, 0.61, 0.64)
    terms = build_gap_decomposition(metrics, model="logreg", cut=2019)
    assert terms["size"] < 0


def test_production_decomposition_needs_both_reference_designs() -> None:
    """Dropping random_matched must raise, not silently fold size into class."""
    from sift.reporting import build_gap_decomposition

    metrics = _metrics_with_references(0.20, 0.76, 0.60, 0.55)
    without = metrics.loc[metrics["design"] != "random_matched"]
    with pytest.raises(Exception):
        build_gap_decomposition(without, model="logreg", cut=2019)
