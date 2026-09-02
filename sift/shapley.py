"""Exact Shapley-Shorrocks decomposition over the four-control ablation lattice.

This module implements the aggregation mechanism the paper specifies in its
attribution section. The players of the cooperative game are
the four *pipeline controls* of the SIFT experiment, never input features, and
the characteristic function ``v`` is measured rather than modelled: all
``2 ** 4 = 16`` coalitions are actually run, so the Shapley value is obtained in
closed form and no Monte Carlo sampling is involved.

References
----------
.. [1] L. S. Shapley, "A value for n-person games", in *Contributions to the
       Theory of Games II*, Princeton University Press, 1953, pp. 307-317.
.. [2] A. F. Shorrocks, "Decomposition procedures for distributional analysis:
       a unified framework based on the Shapley value", *Journal of Economic
       Inequality*, vol. 11, no. 1, pp. 99-126, 2013.
.. [3] G. Owen, "Values of games with a priori unions", in *Mathematical
       Economics and Game Theory*, Springer, 1977, pp. 76-88.
.. [4] J. C. Harsanyi, "A simplified bargaining model for the n-person
       cooperative game", *International Economic Review*, vol. 4, no. 2,
       pp. 194-220, 1963.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import combinations
from math import factorial, fsum

__all__ = [
    "DEFAULT_CONTROL_COLUMNS",
    "DEFAULT_TOLERANCE",
    "SHAPLEY_WEIGHTS_K4",
    "Decomposition",
    "VerificationResult",
    "check_efficiency",
    "check_mobius_consistency",
    "check_owen_group_consistency",
    "check_weights_sum_to_one",
    "decompose",
    "exact_shapley",
    "gap_values",
    "harsanyi_dividends",
    "owen_value",
    "quotient_game_shapley",
    "run_all_checks",
    "shapley_from_dividends",
    "shapley_weights",
]

DEFAULT_TOLERANCE: float = 1e-9
"""Numerical tolerance for the three arithmetic checks."""

DEFAULT_CONTROL_COLUMNS: tuple[str, ...] = ("a1_prior", "a2_labels", "b1_fs", "b2_axis")
"""Flag columns :func:`gap_values` reads when no player set is named.

The four-control lattice. Everything else in this module is already written for
``n`` players -- ``_infer_players`` derives the player set from the coalition
keys, ``_weight_table`` computes the weights for any ``n``, and
``exact_shapley``, ``harsanyi_dividends``, ``owen_value`` and
``quotient_game_shapley`` loop over it -- so ``gap_values`` was the only place
where four was written down. Passing ``control_names`` overrides it; see the
five-control extension in :mod:`sift.config`.
"""

SHAPLEY_WEIGHTS_K4: dict[int, float] = {
    0: 1.0 / 4.0,
    1: 1.0 / 12.0,
    2: 1.0 / 12.0,
    3: 1.0 / 4.0,
}
"""Closed-form Shapley weights for ``n = 4``, indexed by the size of ``S``.

At four players the weight ``|S|! (n - |S| - 1)! / n!`` collapses to four
constants. They are tabulated so that no factorial is evaluated inside the
accumulation loop.
"""


# --- Weights ----------------------------------------------------------------


@lru_cache(maxsize=None)
def _weight_table(n_players: int) -> tuple[float, ...]:
    """Return Shapley weights by coalition size as an immutable tuple."""
    if n_players < 1:
        raise ValueError(f"n_players must be >= 1, got {n_players}")
    if n_players == 4:
        return tuple(SHAPLEY_WEIGHTS_K4[size] for size in range(4))
    denominator = factorial(n_players)
    return tuple(
        factorial(size) * factorial(n_players - size - 1) / denominator
        for size in range(n_players)
    )


def shapley_weights(n_players: int) -> dict[int, float]:
    """Return the Shapley weight for each coalition size.

    Parameters
    ----------
    n_players : int
        Number of players ``n`` in the game. Must be at least one.

    Returns
    -------
    dict of {int: float}
        Mapping from ``|S|``, ranging over ``0 .. n_players - 1``, to the
        weight ``|S|! (n - |S| - 1)! / n!`` applied to the marginal
        contribution ``v(S + {i}) - v(S)``.
    """
    table = _weight_table(n_players)
    return {size: table[size] for size in range(n_players)}


# --- Lattice validation -----------------------------------------------------


def _infer_players(values: Mapping[frozenset[str], float]) -> tuple[str, ...]:
    """Return the sorted player set implied by the keys of ``values``."""
    if not values:
        raise ValueError("the characteristic function is empty")
    grand: set[str] = set()
    for coalition in values:
        if not isinstance(coalition, frozenset):
            raise TypeError(
                f"coalition keys must be frozenset, got {type(coalition).__name__}"
            )
        grand.update(coalition)
    return tuple(sorted(grand))


def _validate_lattice(
    values: Mapping[frozenset[str], float], players: Sequence[str]
) -> None:
    """Assert that ``values`` covers the whole lattice and vanishes on the empty set.

    Raises
    ------
    ValueError
        If any of the ``2 ** n`` coalitions is missing, if extra keys are
        present, or if ``v(empty set)`` deviates from zero by more than
        :data:`DEFAULT_TOLERANCE`.
    """
    n_players = len(players)
    expected = 1 << n_players
    if len(values) != expected:
        raise ValueError(
            f"incomplete lattice: expected {expected} coalitions for "
            f"{n_players} players, got {len(values)}"
        )
    for size in range(n_players + 1):
        for subset in combinations(players, size):
            key = frozenset(subset)
            if key not in values:
                raise ValueError(f"missing coalition {sorted(key)!r} in the lattice")
    empty_value = float(values[frozenset()])
    if abs(empty_value) > DEFAULT_TOLERANCE:
        raise ValueError(
            f"v(empty set) must be zero, got {empty_value!r}; v(S) is defined "
            "as M(S) - M(baseline), so the empty coalition is the baseline and "
            "cancels by construction"
        )


# --- Shapley value ----------------------------------------------------------


def exact_shapley(values: Mapping[frozenset[str], float]) -> dict[str, float]:
    """Compute the exact Shapley value of every player from a complete lattice.

    Parameters
    ----------
    values : Mapping of {frozenset of str: float}
        The characteristic function ``v``. Must contain all ``2 ** n``
        coalitions over the inferred player set, with ``v(empty set) == 0``.

    Returns
    -------
    dict of {str: float}
        Shapley value ``phi_i`` for each player, keyed by player name.

    Notes
    -----
    The closed form is

    .. math::

        \phi_i = \sum_{S \subseteq N \setminus \{i\}}
                  \frac{|S|!\,(n-|S|-1)!}{n!}
                  \bigl(v(S \cup \{i\}) - v(S)\bigr).
    """
    players = _infer_players(values)
    _validate_lattice(values, players)
    n_players = len(players)
    weights = _weight_table(n_players)

    phi: dict[str, float] = {}
    for player in players:
        singleton = frozenset({player})
        others = [candidate for candidate in players if candidate != player]
        terms: list[float] = []
        for size in range(n_players):
            weight = weights[size]
            for subset in combinations(others, size):
                without = frozenset(subset)
                terms.append(weight * (values[without | singleton] - values[without]))
        phi[player] = fsum(terms)
    return phi


# --- Harsanyi dividends, that is the Moebius transform ----------------------


def harsanyi_dividends(
    values: Mapping[frozenset[str], float],
) -> dict[frozenset[str], float]:
    """Compute the Harsanyi dividend of every coalition.

    Parameters
    ----------
    values : Mapping of {frozenset of str: float}
        The characteristic function ``v`` over a complete lattice.

    Returns
    -------
    dict of {frozenset of str: float}
        The Moebius transform ``m(S)`` for all ``2 ** n`` coalitions.

    Notes
    -----
    The Moebius transform is

    .. math::

        m(S) = \sum_{T \subseteq S} (-1)^{|S| - |T|} v(T).
    """
    players = _infer_players(values)
    _validate_lattice(values, players)

    dividends: dict[frozenset[str], float] = {}
    for size in range(len(players) + 1):
        for subset in combinations(players, size):
            terms: list[float] = []
            for inner_size in range(size + 1):
                sign = 1.0 if (size - inner_size) % 2 == 0 else -1.0
                for inner in combinations(subset, inner_size):
                    terms.append(sign * values[frozenset(inner)])
            dividends[frozenset(subset)] = fsum(terms)
    return dividends


def shapley_from_dividends(
    dividends: Mapping[frozenset[str], float],
) -> dict[str, float]:
    """Recover the Shapley value by splitting each Harsanyi dividend equally.

    Parameters
    ----------
    dividends : Mapping of {frozenset of str: float}
        Output of :func:`harsanyi_dividends`.

    Returns
    -------
    dict of {str: float}
        Shapley value per player.

    Notes
    -----
    Uses the identity ``phi_i = sum_{S containing i} m(S) / |S|``. This is an
    arithmetically independent route to the same quantity, which is why it
    catches implementation errors that the efficiency check alone would miss;
    see :func:`check_mobius_consistency`.
    """
    players: set[str] = set()
    for coalition in dividends:
        players.update(coalition)

    phi: dict[str, float] = {}
    for player in sorted(players):
        terms = [
            dividend / len(coalition)
            for coalition, dividend in dividends.items()
            if player in coalition
        ]
        phi[player] = fsum(terms)
    return phi


# --- Owen value for the two-tier taxonomy -----------------------------------


def _validate_partition(
    groups: Sequence[Sequence[str]], players: Sequence[str]
) -> tuple[frozenset[str], ...]:
    """Assert that ``groups`` partitions ``players`` and return the frozen unions."""
    unions = tuple(frozenset(group) for group in groups)
    if not unions:
        raise ValueError("at least one a priori union is required")
    if any(len(union) == 0 for union in unions):
        raise ValueError("a priori unions must be non-empty")
    covered: set[str] = set()
    for union in unions:
        overlap = covered & union
        if overlap:
            raise ValueError(f"a priori unions overlap on {sorted(overlap)!r}")
        covered |= union
    if covered != set(players):
        missing = sorted(set(players) - covered)
        unknown = sorted(covered - set(players))
        raise ValueError(
            "a priori unions must partition the player set; "
            f"missing={missing!r}, unknown={unknown!r}"
        )
    return unions


def owen_value(
    values: Mapping[frozenset[str], float],
    groups: Sequence[Sequence[str]],
) -> dict[str, float]:
    """Compute the Owen value under an a priori partition of the players.

    Parameters
    ----------
    values : Mapping of {frozenset of str: float}
        The characteristic function ``v`` over a complete lattice.
    groups : sequence of sequence of str
        The a priori unions. Must partition the player set exactly. For SIFT
        the two tiers are ``[("a1_prior", "a2_labels"), ("b1_fs", "b2_axis")]``,
        that is the data-side controls and the protocol-side controls.

    Returns
    -------
    dict of {str: float}
        Owen value per player.

    Notes
    -----
    Following Owen (1977), for a player ``i`` in union ``B_k`` among ``m``
    unions,

    .. math::

        \phi_i^{Ow} = \sum_{R \subseteq M \setminus \{k\}}
                       \sum_{T \subseteq B_k \setminus \{i\}}
                       w(|R|, m)\, w(|T|, b_k)
                       \bigl(v(Q \cup T \cup \{i\}) - v(Q \cup T)\bigr),
    """
    players = _infer_players(values)
    _validate_lattice(values, players)
    unions = _validate_partition(groups, players)

    n_unions = len(unions)
    outer_weights = _weight_table(n_unions)

    phi: dict[str, float] = {}
    for index, union in enumerate(unions):
        union_size = len(union)
        inner_weights = _weight_table(union_size)
        other_unions = [unions[other] for other in range(n_unions) if other != index]
        for player in sorted(union):
            singleton = frozenset({player})
            rest_of_union = sorted(union - singleton)
            terms: list[float] = []
            for outer_size in range(n_unions):
                outer_weight = outer_weights[outer_size]
                for selected in combinations(other_unions, outer_size):
                    quotient: frozenset[str] = frozenset()
                    for member in selected:
                        quotient |= member
                    for inner_size in range(union_size):
                        weight = outer_weight * inner_weights[inner_size]
                        for inner in combinations(rest_of_union, inner_size):
                            base = quotient | frozenset(inner)
                            terms.append(
                                weight * (values[base | singleton] - values[base])
                            )
            phi[player] = fsum(terms)
    return phi


def quotient_game_shapley(
    values: Mapping[frozenset[str], float],
    groups: Sequence[Sequence[str]],
    group_names: Sequence[str] | None = None,
) -> dict[str, float]:
    """Compute the Shapley value of each a priori union in the quotient game.

    Parameters
    ----------
    values : Mapping of {frozenset of str: float}
        The characteristic function ``v`` over a complete lattice.
    groups : sequence of sequence of str
        The a priori unions, partitioning the player set.
    group_names : sequence of str, optional
        Labels for the unions. Defaults to ``"tier_0"``, ``"tier_1"`` and so on.

    Returns
    -------
    dict of {str: float}
        Shapley value per union, computed on the quotient game in which each
        union acts as a single player.
    """
    players = _infer_players(values)
    _validate_lattice(values, players)
    unions = _validate_partition(groups, players)

    if group_names is None:
        names = tuple(f"tier_{index}" for index in range(len(unions)))
    else:
        names = tuple(group_names)
        if len(names) != len(unions):
            raise ValueError(
                f"group_names has {len(names)} entries but there are "
                f"{len(unions)} unions"
            )
        if len(set(names)) != len(names):
            raise ValueError(f"group_names must be unique, got {names!r}")

    name_to_union = dict(zip(names, unions, strict=True))
    quotient: dict[frozenset[str], float] = {}
    for size in range(len(names) + 1):
        for subset in combinations(names, size):
            members: frozenset[str] = frozenset()
            for name in subset:
                members |= name_to_union[name]
            quotient[frozenset(subset)] = float(values[members])
    return exact_shapley(quotient)


# --- Verification -----------------------------------------------------------


@dataclass(frozen=True)
class VerificationResult:
    """Outcome of one arithmetic check on a decomposition.

    Attributes
    ----------
    name : str
        Identifier of the check.
    passed : bool
        Whether the deviation lies within ``tolerance``.
    deviation : float
        Absolute deviation from the identity being checked.
    tolerance : float
        Tolerance the deviation was compared against.
    detail : str
        Human-readable statement of what was compared.
    """

    name: str
    passed: bool
    deviation: float
    tolerance: float
    detail: str

    def raise_if_failed(self) -> None:
        """Raise :class:`AssertionError` when the check did not pass."""
        if not self.passed:
            raise AssertionError(
                f"{self.name} failed: deviation {self.deviation:.3e} exceeds "
                f"tolerance {self.tolerance:.3e}; {self.detail}"
            )


def check_weights_sum_to_one(
    n_players: int, tolerance: float = DEFAULT_TOLERANCE
) -> VerificationResult:
    """Check that the Shapley weights, counted with multiplicity, sum to one.

    Parameters
    ----------
    n_players : int
        Number of players.
    tolerance : float, optional
        Absolute tolerance.

    Returns
    -------
    VerificationResult
        ``passed`` is true when ``sum_s C(n - 1, s) * w(s) == 1``.
    """
    weights = _weight_table(n_players)
    total = fsum(
        factorial(n_players - 1)
        / (factorial(size) * factorial(n_players - 1 - size))
        * weights[size]
        for size in range(n_players)
    )
    deviation = abs(total - 1.0)
    return VerificationResult(
        name="weights_sum_to_one",
        passed=deviation <= tolerance,
        deviation=deviation,
        tolerance=tolerance,
        detail=f"weighted subset counts sum to {total!r} for n_players={n_players}",
    )


def check_efficiency(
    values: Mapping[frozenset[str], float],
    phi: Mapping[str, float],
    tolerance: float = DEFAULT_TOLERANCE,
    name: str = "efficiency",
) -> VerificationResult:
    """Check the efficiency axiom, ``sum_i phi_i == v(N) - v(empty set)``.

    Parameters
    ----------
    values : Mapping of {frozenset of str: float}
        The characteristic function.
    phi : Mapping of {str: float}
        Attribution to check, from :func:`exact_shapley` or :func:`owen_value`.
    tolerance : float, optional
        Absolute tolerance.
    name : str, optional
        Identifier recorded in the result, so that the Shapley and the Owen
        efficiency checks can be distinguished in a report.

    Returns
    -------
    VerificationResult
        ``passed`` is true when the attributions exhaust the grand coalition.
    """
    players = _infer_players(values)
    grand = float(values[frozenset(players)]) - float(values[frozenset()])
    total = fsum(float(value) for value in phi.values())
    deviation = abs(total - grand)
    return VerificationResult(
        name=name,
        passed=deviation <= tolerance,
        deviation=deviation,
        tolerance=tolerance,
        detail=f"sum(phi) = {total!r} against v(N) = {grand!r}",
    )


def check_mobius_consistency(
    values: Mapping[frozenset[str], float],
    phi: Mapping[str, float],
    tolerance: float = DEFAULT_TOLERANCE,
) -> VerificationResult:
    """Cross-check the Shapley value against the Harsanyi dividend route.

    Parameters
    ----------
    values : Mapping of {frozenset of str: float}
        The characteristic function.
    phi : Mapping of {str: float}
        Attribution produced by :func:`exact_shapley`.
    tolerance : float, optional
        Absolute tolerance on the largest per-player discrepancy.

    Returns
    -------
    VerificationResult
        ``passed`` is true when both routes agree for every player.
    """
    recovered = shapley_from_dividends(harsanyi_dividends(values))
    if set(recovered) != set(phi):
        return VerificationResult(
            name="mobius_consistency",
            passed=False,
            deviation=float("inf"),
            tolerance=tolerance,
            detail=f"player sets differ: {sorted(recovered)!r} against {sorted(phi)!r}",
        )
    worst = max(recovered, key=lambda player: abs(recovered[player] - float(phi[player])))
    deviation = abs(recovered[worst] - float(phi[worst]))
    return VerificationResult(
        name="mobius_consistency",
        passed=deviation <= tolerance,
        deviation=deviation,
        tolerance=tolerance,
        detail=(
            f"largest discrepancy at {worst!r}: direct {float(phi[worst])!r} "
            f"against Moebius {recovered[worst]!r}"
        ),
    )


def check_owen_group_consistency(
    values: Mapping[frozenset[str], float],
    groups: Sequence[Sequence[str]],
    owen: Mapping[str, float],
    tolerance: float = DEFAULT_TOLERANCE,
) -> VerificationResult:
    """Check that Owen values within a union sum to its quotient-game Shapley value.

    Parameters
    ----------
    values : Mapping of {frozenset of str: float}
        The characteristic function.
    groups : sequence of sequence of str
        The a priori unions.
    owen : Mapping of {str: float}
        Attribution produced by :func:`owen_value`.
    tolerance : float, optional
        Absolute tolerance on the largest per-union discrepancy.

    Returns
    -------
    VerificationResult
        ``passed`` is true when every union balances.
    """
    names = tuple(f"tier_{index}" for index in range(len(groups)))
    tier_shapley = quotient_game_shapley(values, groups, names)
    worst_name = names[0]
    worst_deviation = -1.0
    for name, group in zip(names, groups, strict=True):
        subtotal = fsum(float(owen[player]) for player in group)
        deviation = abs(subtotal - tier_shapley[name])
        if deviation > worst_deviation:
            worst_deviation, worst_name = deviation, name
    return VerificationResult(
        name="owen_group_consistency",
        passed=worst_deviation <= tolerance,
        deviation=worst_deviation,
        tolerance=tolerance,
        detail=f"largest union discrepancy at {worst_name!r}: {worst_deviation:.3e}",
    )


def run_all_checks(
    values: Mapping[frozenset[str], float],
    phi: Mapping[str, float] | None = None,
    groups: Sequence[Sequence[str]] | None = None,
    owen: Mapping[str, float] | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
) -> list[VerificationResult]:
    """Run every applicable arithmetic check and return the results as data.

    Parameters
    ----------
    values : Mapping of {frozenset of str: float}
        The characteristic function.
    phi : Mapping of {str: float}, optional
        Shapley attribution. Recomputed from ``values`` when omitted.
    groups : sequence of sequence of str, optional
        A priori unions. When given, the Owen checks are included.
    owen : Mapping of {str: float}, optional
        Owen attribution. Recomputed when ``groups`` is given and this is
        omitted.
    tolerance : float, optional
        Absolute tolerance shared by all checks.

    Returns
    -------
    list of VerificationResult
        One entry per check, in a stable order. Nothing is raised; the caller
        decides whether a failure is fatal via
        :meth:`VerificationResult.raise_if_failed`.
    """
    players = _infer_players(values)
    attribution = dict(phi) if phi is not None else exact_shapley(values)
    results = [
        check_weights_sum_to_one(len(players), tolerance),
        check_efficiency(values, attribution, tolerance),
        check_mobius_consistency(values, attribution, tolerance),
    ]
    if groups is not None:
        owen_attribution = dict(owen) if owen is not None else owen_value(values, groups)
        results.append(
            check_efficiency(values, owen_attribution, tolerance, name="efficiency_owen")
        )
        results.append(
            check_owen_group_consistency(values, groups, owen_attribution, tolerance)
        )
    return results


# --- Convenience container --------------------------------------------------


@dataclass(frozen=True)
class Decomposition:
    """One complete Shapley-Shorrocks decomposition of an evaluation gap.

    Attributes
    ----------
    delta : float
        The measured evaluation gap, macro-F1 under the random design minus
        macro-F1 under the temporal design.
    shapley : dict of {str: float}
        Per-control Shapley value.
    residual : float
        ``delta - v(N)``. What the four controls leave unexplained, never to
        be reported as concept drift on its own.
    dividends : dict of {frozenset of str: float}
        Harsanyi dividends, carrying the interaction table.
    checks : list of VerificationResult
        Every arithmetic check that was run.
    owen : dict of {str: float} or None
        Per-control Owen value under the two-tier partition, when requested.
    tier_shapley : dict of {str: float} or None
        Quotient-game Shapley value per tier, when a partition was given.
    """

    delta: float
    shapley: dict[str, float]
    residual: float
    dividends: dict[frozenset[str], float]
    checks: list[VerificationResult] = field(default_factory=list)
    owen: dict[str, float] | None = None
    tier_shapley: dict[str, float] | None = None

    @property
    def all_checks_passed(self) -> bool:
        """Whether every recorded check passed."""
        return all(result.passed for result in self.checks)

    def raise_if_any_check_failed(self) -> None:
        """Raise :class:`AssertionError` on the first failing check."""
        for result in self.checks:
            result.raise_if_failed()


def decompose(
    delta: float,
    values: Mapping[frozenset[str], float],
    groups: Sequence[Sequence[str]] | None = None,
    group_names: Sequence[str] | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
) -> Decomposition:
    """Decompose an evaluation gap into per-control contributions and a residual.

    Parameters
    ----------
    delta : float
        The evaluation gap to be decomposed.
    values : Mapping of {frozenset of str: float}
        The characteristic function over the complete lattice, with
        ``v(S) = M(S) - M(baseline)`` so that ``v(empty set) == 0``.
    groups : sequence of sequence of str, optional
        A priori unions for the Owen value, for SIFT the data-side and the
        protocol-side tier.
    group_names : sequence of str, optional
        Labels for the tiers.
    tolerance : float, optional
        Absolute tolerance for the arithmetic checks.

    Returns
    -------
    Decomposition
        The attribution, the residual, the interaction table and the checks.
    """
    players = _infer_players(values)
    shapley = exact_shapley(values)
    dividends = harsanyi_dividends(values)
    residual = float(delta) - float(values[frozenset(players)])

    owen: dict[str, float] | None = None
    tier_shapley: dict[str, float] | None = None
    if groups is not None:
        owen = owen_value(values, groups)
        tier_shapley = quotient_game_shapley(values, groups, group_names)

    checks = run_all_checks(values, shapley, groups, owen, tolerance)
    return Decomposition(
        delta=float(delta),
        shapley=shapley,
        residual=residual,
        dividends=dividends,
        checks=checks,
        owen=owen,
        tier_shapley=tier_shapley,
    )


def gap_values(
    metrics: "object",
    model: str,
    cut: int,
    seed: int | None = None,
    metric_column: str = "macro_f1",
    temporal_design: str = "temporal",
    reference_design: str = "random_fully_matched",
    control_names: Sequence[str] | None = None,
) -> dict[frozenset[str], float]:
    """Build the gap-based characteristic function for one game instance.

    Parameters
    ----------
    metrics : pandas.DataFrame
        Contents of ``results/metrics.parquet``. Only rows whose ``role`` is the
        lattice role take part; reference-role rows are refused rather than
        averaged in.
    model : str
        Model to select.
    cut : int
        Cut-point to select.
    seed : int, optional
        Seed to select. When omitted, seeds are averaged within each coalition.
    metric_column : str, optional
        Metric forming the gap.
    temporal_design, reference_design : str, optional
        The two arms of the game.
    control_names : sequence of str, optional
        The players. Defaults to :data:`DEFAULT_CONTROL_COLUMNS`, the four controls, and
        then ``2 ** 4`` coalitions are required.

    Returns
    -------
    dict of {frozenset of str: float}
        ``v(S) = Delta(empty) - Delta(S)`` for all ``2 ** n`` coalitions, ready for
        :func:`exact_shapley`.

    Raises
    ------
    ValueError
        If reference-role rows are present in the selection, if either arm is
        missing, if a named control column is absent, or if the selection does
        not carry all ``2 ** n`` coalitions.
    """
    frame = metrics
    required = {"design", "config_id", "model", "cut", metric_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"metrics is missing column(s) {sorted(missing)!r}")

    selected = frame[(frame["model"] == model) & (frame["cut"] == cut)]
    if seed is not None and "seed" in selected.columns:
        selected = selected[selected["seed"] == seed]
    if selected.empty:
        raise ValueError(f"no rows for model={model!r} cut={cut!r} seed={seed!r}")

    if "role" in selected.columns:
        offending = sorted(
            set(selected.loc[selected["role"] != "lattice", "design"].unique())
        )
        if offending:
            raise ValueError(
                f"reference-role designs {offending!r} reached the value function; "
                "they carry the baseline coalition only and are never players. "
                "Filter with sift.experiment.lattice_cells first."
            )

    present = set(selected["design"].unique())
    for design in (temporal_design, reference_design):
        if design not in present:
            raise ValueError(
                f"arm {design!r} is absent; present designs are {sorted(present)!r}"
            )
    unexpected = present - {temporal_design, reference_design}
    if unexpected:
        raise ValueError(
            f"unexpected design(s) {sorted(unexpected)!r} in the lattice selection"
        )

    if control_names is None:
        control_columns = [
            name for name in DEFAULT_CONTROL_COLUMNS if name in selected.columns
        ]
        # Held at 16 rather than derived from the columns actually found, so a
        # table that has lost a flag column still fails with the message the
        # four-control path has always raised.
        expected_coalitions = 2 ** len(DEFAULT_CONTROL_COLUMNS)
    else:
        control_columns = [str(name) for name in control_names]
        absent = [name for name in control_columns if name not in selected.columns]
        if absent:
            raise ValueError(f"metrics is missing control column(s) {absent!r}")
        expected_coalitions = 2 ** len(control_columns)
    if not control_columns:
        raise ValueError("no control flag columns found in metrics")

    gaps: dict[frozenset[str], float] = {}
    for config_id, block in selected.groupby("config_id", sort=True):
        temporal = block[block["design"] == temporal_design][metric_column]
        reference = block[block["design"] == reference_design][metric_column]
        if temporal.empty or reference.empty:
            raise ValueError(
                f"config_id {int(config_id)} lacks one of the two arms; the lattice "
                "must be complete for the exact Shapley value to be defined"
            )
        head = block.iloc[0]
        coalition = frozenset(
            name for name in control_columns if bool(head[name])
        )
        gaps[coalition] = float(reference.mean()) - float(temporal.mean())

    if len(gaps) != expected_coalitions:
        raise ValueError(
            f"expected {expected_coalitions} coalitions for model={model!r} "
            f"cut={cut!r}, got {len(gaps)}"
        )

    baseline = gaps[frozenset()]
    return {coalition: baseline - gap for coalition, gap in gaps.items()}
