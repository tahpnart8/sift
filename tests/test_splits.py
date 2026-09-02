"""Split disjointness and temporal ordering.

These are the invariants a reviewer checks first.  They are asserted on the
synthetic panel so the suite runs without MLRan, and repeated on the real panel
where it is present.

The constructor keywords for ``SplitSpec`` are assumed rather than introspected;
if the implementation names them differently the fix is to rename here, never to
relax the assertion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import requires_mlran
from sift.config import SplitSpec
from sift.splits import assert_temporal_consistency, make_split

TIME_COLUMN = "first_submission_date_year"


def _random_spec() -> SplitSpec:
    return SplitSpec(design="random", test_size=0.25)


def _temporal_spec(cut: int = 2019) -> SplitSpec:
    return SplitSpec(design="temporal", cut_year=cut)


def test_train_and_test_indices_are_disjoint(panel: pd.DataFrame) -> None:
    for spec in (_random_spec(), _temporal_spec()):
        train_idx, test_idx = make_split(panel, spec, TIME_COLUMN, seed=0)
        assert len(set(train_idx) & set(test_idx)) == 0, f"overlap under {spec}"


def test_split_indices_contain_no_repeats(panel: pd.DataFrame) -> None:
    for spec in (_random_spec(), _temporal_spec()):
        train_idx, test_idx = make_split(panel, spec, TIME_COLUMN, seed=0)
        assert len(set(train_idx)) == len(train_idx)
        assert len(set(test_idx)) == len(test_idx)


def test_split_indices_all_belong_to_the_panel(panel: pd.DataFrame) -> None:
    for spec in (_random_spec(), _temporal_spec()):
        train_idx, test_idx = make_split(panel, spec, TIME_COLUMN, seed=0)
        assert set(train_idx) <= set(panel.index)
        assert set(test_idx) <= set(panel.index)
        assert len(train_idx) > 0 and len(test_idx) > 0


def test_temporal_training_rows_strictly_precede_test_rows(panel: pd.DataFrame) -> None:
    """TESSERACT constraint C1, stated as a strict inequality."""
    train_idx, test_idx = make_split(panel, _temporal_spec(), TIME_COLUMN, seed=0)
    latest_train = panel.loc[train_idx, TIME_COLUMN].max()
    earliest_test = panel.loc[test_idx, TIME_COLUMN].min()
    assert latest_train < earliest_test


def test_assert_temporal_consistency_rejects_a_leaking_split(panel: pd.DataFrame) -> None:
    """Moving one late row into training must make the guard raise."""
    train_idx, test_idx = make_split(panel, _temporal_spec(), TIME_COLUMN, seed=0)
    assert_temporal_consistency(panel, train_idx, test_idx, TIME_COLUMN)

    leaked_train = list(train_idx) + [list(test_idx)[0]]
    leaked_test = list(test_idx)[1:]
    with pytest.raises(AssertionError):
        assert_temporal_consistency(panel, leaked_train, leaked_test, TIME_COLUMN)


def test_random_split_is_reproducible_for_a_fixed_seed(panel: pd.DataFrame) -> None:
    first = make_split(panel, _random_spec(), TIME_COLUMN, seed=3)
    second = make_split(panel, _random_spec(), TIME_COLUMN, seed=3)
    assert list(first[0]) == list(second[0])
    assert list(first[1]) == list(second[1])


def test_random_split_actually_responds_to_the_seed(panel: pd.DataFrame) -> None:
    """Guards against the pilot bug where random_state was hard coded to 42."""
    a_train, _ = make_split(panel, _random_spec(), TIME_COLUMN, seed=0)
    b_train, _ = make_split(panel, _random_spec(), TIME_COLUMN, seed=1)
    assert list(a_train) != list(b_train)


def test_temporal_split_is_independent_of_the_seed(panel: pd.DataFrame) -> None:
    """The temporal partition is deterministic; only the model carries a seed."""
    a = make_split(panel, _temporal_spec(), TIME_COLUMN, seed=0)
    b = make_split(panel, _temporal_spec(), TIME_COLUMN, seed=4)
    assert list(a[0]) == list(b[0])
    assert list(a[1]) == list(b[1])


def test_random_split_is_stratified_over_the_family_label(panel: pd.DataFrame) -> None:
    """Design doc section 4.1 requires stratification by family."""
    train_idx, test_idx = make_split(
        panel,
        _random_spec(),
        TIME_COLUMN,
        seed=0,
        stratify_labels=panel["ransomware_family"],
    )
    train_classes = set(panel.loc[train_idx, "ransomware_family"])
    test_classes = set(panel.loc[test_idx, "ransomware_family"])
    assert test_classes <= train_classes, "stratified split left an unseen class in test"


@requires_mlran
def test_real_panel_split_holds_the_same_invariants() -> None:
    from sift.config import PanelSpec
    from sift.data import build_panel, load_mlran

    raw = load_mlran()
    real_panel, provenance = build_panel(raw, PanelSpec())
    assert isinstance(provenance, dict) and len(provenance) > 0
    train_idx, test_idx = make_split(real_panel, _temporal_spec(), TIME_COLUMN, seed=0)
    assert len(set(train_idx) & set(test_idx)) == 0
    assert_temporal_consistency(real_panel, train_idx, test_idx, TIME_COLUMN)
