"""Lattice totality and the Shapley-Shorrocks arithmetic.

CONTRACT.md section 1 fixes the four players and the order in which controls are
applied.  The paper fixes the three arithmetic checks: efficiency, the weight
sum, and the Harsanyi cross-check.  This module
adds what those three cannot catch, namely absolute values, because a symmetric
implementation error satisfies all three.
"""

from __future__ import annotations

from itertools import combinations

import pytest

from conftest import (
    A1,
    A2,
    B1,
    B2,
    GOLDEN_DIVIDENDS,
    GOLDEN_GRAND_VALUE,
    GOLDEN_OWEN,
    GOLDEN_SHAPLEY,
    GOLDEN_VALUES,
    OWEN_GROUPS,
    PLAYERS,
    TOL,
)
from sift.config import CONTROL_NAMES, ControlFlags
from sift.controls import CONTROL_ORDER
from sift.shapley import exact_shapley, harsanyi_dividends, owen_value


def test_control_names_and_order_match_the_contract() -> None:
    assert CONTROL_NAMES == ("a1_prior", "a2_labels", "b1_fs", "b2_axis")
    assert CONTROL_ORDER == ("b2_axis", "a2_labels", "a1_prior", "b1_fs")
    assert set(CONTROL_ORDER) == set(CONTROL_NAMES)


def test_control_flags_roundtrip_all_sixteen_indices() -> None:
    seen = set()
    for index in range(16):
        flags = ControlFlags.from_index(index)
        assert flags.to_index() == index
        active = flags.active()
        assert isinstance(active, frozenset)
        assert active <= set(CONTROL_NAMES)
        seen.add(active)
    # Sixteen distinct coalitions, so no index collapses onto another.
    assert len(seen) == 16


def test_the_lattice_covers_every_one_of_the_sixteen_coalitions() -> None:
    expected = {
        frozenset(c)
        for size in range(len(CONTROL_NAMES) + 1)
        for c in combinations(CONTROL_NAMES, size)
    }
    produced = {ControlFlags.from_index(i).active() for i in range(16)}
    assert produced == expected


def test_exact_shapley_matches_hand_computed_golden_values() -> None:
    phi = exact_shapley(GOLDEN_VALUES)
    assert set(phi) == set(PLAYERS)
    for player, expected in GOLDEN_SHAPLEY.items():
        assert phi[player] == pytest.approx(expected, abs=TOL)


def test_exact_shapley_satisfies_efficiency() -> None:
    phi = exact_shapley(GOLDEN_VALUES)
    assert sum(phi.values()) == pytest.approx(GOLDEN_GRAND_VALUE, abs=TOL)


def test_harsanyi_dividends_match_hand_computed_values() -> None:
    produced = harsanyi_dividends(GOLDEN_VALUES)
    for coalition, expected in GOLDEN_DIVIDENDS.items():
        assert produced[coalition] == pytest.approx(expected, abs=TOL)
    # Every coalition absent from the golden table must come back as zero, not
    # be missing from the mapping.
    for coalition, value in produced.items():
        if coalition not in GOLDEN_DIVIDENDS:
            assert value == pytest.approx(0.0, abs=TOL)


def test_shapley_from_dividends_agrees_with_the_direct_computation() -> None:
    """The Moebius route is an independent implementation path."""
    dividends = harsanyi_dividends(GOLDEN_VALUES)
    via_moebius = {
        player: sum(m / len(t) for t, m in dividends.items() if player in t)
        for player in PLAYERS
    }
    direct = exact_shapley(GOLDEN_VALUES)
    for player in PLAYERS:
        assert via_moebius[player] == pytest.approx(direct[player], abs=TOL)


def test_exact_shapley_assigns_zero_to_a_null_player() -> None:
    """b1_fs contributes nothing anywhere, so its share must be exactly zero."""
    values = {
        s: (0.5 if A1 in s else 0.0) + (0.25 if A2 in s else 0.0) + (0.125 if B2 in s else 0.0)
        for s in GOLDEN_VALUES
    }
    phi = exact_shapley(values)
    assert phi[B1] == pytest.approx(0.0, abs=TOL)


def test_exact_shapley_is_symmetric_for_interchangeable_players() -> None:
    """A game depending only on coalition size must split v(N) four ways."""
    values = {s: float(len(s)) ** 2 for s in GOLDEN_VALUES}
    phi = exact_shapley(values)
    for player in PLAYERS:
        assert phi[player] == pytest.approx(16.0 / 4.0, abs=TOL)


def test_exact_shapley_reproduces_the_coefficients_of_an_additive_game() -> None:
    coefficients = {A1: 0.10, A2: 0.25, B1: 0.05, B2: 0.40}
    values = {s: sum(coefficients[p] for p in s) for s in GOLDEN_VALUES}
    phi = exact_shapley(values)
    for player, expected in coefficients.items():
        assert phi[player] == pytest.approx(expected, abs=TOL)


def test_exact_shapley_raises_when_a_coalition_is_missing() -> None:
    """A silently dropped cell must be loud, per CONTRACT.md section 1."""
    incomplete = dict(GOLDEN_VALUES)
    del incomplete[frozenset({A1, B2})]
    with pytest.raises((KeyError, ValueError)):
        exact_shapley(incomplete)


def test_exact_shapley_raises_when_the_empty_coalition_is_missing() -> None:
    incomplete = dict(GOLDEN_VALUES)
    del incomplete[frozenset()]
    with pytest.raises((KeyError, ValueError)):
        exact_shapley(incomplete)


def test_owen_value_matches_hand_computed_golden_values() -> None:
    owen = owen_value(GOLDEN_VALUES, OWEN_GROUPS)
    for player, expected in GOLDEN_OWEN.items():
        assert owen[player] == pytest.approx(expected, abs=TOL)


def test_owen_group_sums_equal_the_quotient_game_shapley() -> None:
    """Owen restricted to groups is the Shapley value of the game over groups.

    Quotient game: w({A}) = v({a1_prior, a2_labels}) = 0.24,
    w({B}) = v({b1_fs, b2_axis}) = 0.46, w({A, B}) = 0.784.
    Two-player Shapley gives 0.282 and 0.502.
    """
    owen = owen_value(GOLDEN_VALUES, OWEN_GROUPS)
    data_side = owen[A1] + owen[A2]
    protocol_side = owen[B1] + owen[B2]
    assert data_side == pytest.approx(0.282, abs=TOL)
    assert protocol_side == pytest.approx(0.502, abs=TOL)
    assert data_side + protocol_side == pytest.approx(GOLDEN_GRAND_VALUE, abs=TOL)


def test_owen_differs_from_shapley_on_an_interacting_game() -> None:
    """Guards against owen_value being an alias for exact_shapley."""
    owen = owen_value(GOLDEN_VALUES, OWEN_GROUPS)
    phi = exact_shapley(GOLDEN_VALUES)
    differing = [p for p in PLAYERS if abs(owen[p] - phi[p]) > 1e-6]
    assert len(differing) == 3
