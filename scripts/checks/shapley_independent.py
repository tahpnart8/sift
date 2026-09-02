"""An independent recomputation of the Shapley decomposition.

Nothing in :mod:`sift.shapley` is imported here, and nothing in it is consulted
at run time. The characteristic function is rebuilt from
``results/metrics.parquet`` and the value is formed by the permutation
definition: for each of the ``4! = 24`` orderings of the four controls the
marginal contribution of every player is accumulated, and the mean over
orderings is the Shapley value. The reference implementation instead uses the
closed-form coalition weights, so agreement between the two is evidence that the
weighting is right, not merely that one formula was typed twice.

The value function is the one the paper defines::

    Delta(S) = M_random_fully_matched(S) - M_temporal(S)
    v(S)     = Delta(empty) - Delta(S),        v(empty) = 0

Reference values are read from
``results/shapley_2024/decomposition_as_reported_macro_f1.parquet`` at the
(cut, model, seed) level and from ``results/decomp_2024.parquet`` at the
(cut, model) level.

Usage::

    python check_shapley_independent.py
"""

from __future__ import annotations

from itertools import permutations

import pandas as pd
from sift.paths import RESULTS_DIR

METRICS = RESULTS_DIR / "metrics.parquet"
REFERENCE_SEED = RESULTS_DIR / "shapley_2024" / "decomposition_as_reported_macro_f1.parquet"
REFERENCE_CELL = RESULTS_DIR / "decomp_2024.parquet"
OUT_DIR = RESULTS_DIR / "evidence"
OUT = OUT_DIR / "shapley_independent_check.parquet"

CONTROLS = ("a1_prior", "a2_labels", "b1_fs", "b2_axis")
METRIC = "macro_f1"
TEMPORAL = "temporal"
REFERENCE_DESIGN = "random_fully_matched"
TOLERANCE = 1e-9


def coalition_of(row) -> frozenset[str]:
    """Return the set of controls switched on in one metrics row."""
    return frozenset(name for name in CONTROLS if bool(row[name]))


def value_function(block: pd.DataFrame) -> dict[frozenset[str], float]:
    """Return ``v(S)`` for all sixteen coalitions of one game instance."""
    deltas: dict[frozenset[str], float] = {}
    for _, row in block.iterrows():
        deltas.setdefault(coalition_of(row), {})[row["design"]] = float(row[METRIC])
    if len(deltas) != 2 ** len(CONTROLS):
        raise ValueError(f"expected 16 coalitions, got {len(deltas)}")
    gaps = {}
    for coalition, arms in deltas.items():
        if set(arms) != {TEMPORAL, REFERENCE_DESIGN}:
            raise ValueError(f"coalition {sorted(coalition)} lacks an arm: {sorted(arms)}")
        gaps[coalition] = arms[REFERENCE_DESIGN] - arms[TEMPORAL]
    base = gaps[frozenset()]
    return {coalition: base - gap for coalition, gap in gaps.items()}


def shapley_by_permutation(value: dict[frozenset[str], float]) -> dict[str, float]:
    """Return the Shapley value from an exhaustive walk of all 24 orderings.

    No weights are used. Each ordering contributes ``v(prefix + {i}) - v(prefix)``
    to player ``i``; the mean over orderings is the value.
    """
    totals = {name: 0.0 for name in CONTROLS}
    orders = list(permutations(CONTROLS))
    for order in orders:
        prefix: set[str] = set()
        for player in order:
            before = value[frozenset(prefix)]
            prefix.add(player)
            totals[player] += value[frozenset(prefix)] - before
    return {name: total / len(orders) for name, total in totals.items()}


def main() -> None:
    metrics = pd.read_parquet(METRICS)
    lattice = metrics[metrics["role"] == "lattice"]
    reference_seed = pd.read_parquet(REFERENCE_SEED)
    reference_cell = pd.read_parquet(REFERENCE_CELL)

    rows = []
    per_cell: dict[tuple, dict[str, float]] = {}
    for (cut, model, seed), block in lattice.groupby(["cut", "model", "seed"], sort=True):
        value = value_function(block)
        phi = shapley_by_permutation(value)
        # Efficiency is a property of the recomputation itself, not of the
        # reference: sum_i phi_i + R must equal Delta(empty) = v(N) + R.
        efficiency_error = abs(sum(phi.values()) - value[frozenset(CONTROLS)])
        match = reference_seed[
            (reference_seed["cut"] == cut)
            & (reference_seed["model"] == model)
            & (reference_seed["seed"] == seed)
        ]
        if len(match) != 1:
            raise ValueError(f"no unique reference row for {cut} {model} seed {seed}")
        ref = match.iloc[0]
        per_cell.setdefault((int(cut), model), {name: 0.0 for name in CONTROLS})
        for name in CONTROLS:
            per_cell[(int(cut), model)][name] += phi[name] / 5.0
            rows.append(
                {
                    "level": "seed",
                    "cell_id": f"{int(cut)}|{model}|seed{int(seed)}",
                    "cut": int(cut),
                    "model": model,
                    "seed": int(seed),
                    "control": name,
                    "phi_reference": float(ref[f"phi_{name}"]),
                    "phi_independent": phi[name],
                    "abs_diff": abs(float(ref[f"phi_{name}"]) - phi[name]),
                    "efficiency_error": efficiency_error,
                }
            )

    for (cut, model), phi in sorted(per_cell.items()):
        match = reference_cell[
            (reference_cell["cut"] == cut) & (reference_cell["model"] == model)
        ]
        if len(match) != 1:
            raise ValueError(f"no unique cell reference for {cut} {model}")
        ref = match.iloc[0]
        for name in CONTROLS:
            rows.append(
                {
                    "level": "cell",
                    "cell_id": f"{cut}|{model}",
                    "cut": cut,
                    "model": model,
                    "seed": -1,
                    "control": name,
                    "phi_reference": float(ref[f"phi_{name}"]),
                    "phi_independent": phi[name],
                    "abs_diff": abs(float(ref[f"phi_{name}"]) - phi[name]),
                    "efficiency_error": float("nan"),
                }
            )

    out = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)

    for level in ("seed", "cell"):
        block = out[out["level"] == level]
        worst = block.loc[block["abs_diff"].idxmax()]
        print(
            f"{level}-level: {len(block)} comparisons, "
            f"max |diff| = {block['abs_diff'].max():.3e} at "
            f"{worst['cell_id']} / {worst['control']}, "
            f"n over {TOLERANCE:g} = {int((block['abs_diff'] > TOLERANCE).sum())}"
        )
    seeds = out[out["level"] == "seed"]
    print(f"max efficiency error of the independent value: {seeds['efficiency_error'].max():.3e}")
    verdict = "AGREE" if float(out["abs_diff"].max()) <= TOLERANCE else "DISAGREE"
    print(f"verdict: {verdict} at tolerance {TOLERANCE:g}")
    print(f"wrote {OUT}  {out.shape}")


if __name__ == "__main__":
    main()
