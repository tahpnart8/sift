"""Sequential ablation against the full-lattice Shapley value.

Turning the controls off one at a time is an ablation *in some order*. The
marginal contribution it credits to a control is
``v(prefix + {i}) - v(prefix)``, which depends on the prefix, hence on the order.
This script walks all ``4! = 24`` orders on the same characteristic function the
decomposition uses and reports, for every control in every cell, the smallest and
the largest credit any order can produce, the spread between them, and the
Shapley value that averages them.

The characteristic function is rebuilt from ``results/metrics.parquet`` exactly
as in ``shapley_independent.py``; nothing is imported from
:mod:`sift.shapley`.

Usage::

    python check_sequential_vs_lattice.py
"""

from __future__ import annotations

from itertools import permutations

import pandas as pd

from shapley_independent import CONTROLS, value_function
from sift.paths import RESULTS_DIR

METRICS = RESULTS_DIR / "metrics.parquet"
OUT_DIR = RESULTS_DIR / "evidence"
OUT = OUT_DIR / "sequential_vs_lattice.parquet"

ORDERS = list(permutations(CONTROLS))


def marginals(value: dict[frozenset[str], float], order: tuple[str, ...]) -> dict[str, float]:
    """Return the credit one sequential ablation order gives each control."""
    out: dict[str, float] = {}
    prefix: set[str] = set()
    for player in order:
        before = value[frozenset(prefix)]
        prefix.add(player)
        out[player] = value[frozenset(prefix)] - before
    return out


def main() -> None:
    metrics = pd.read_parquet(METRICS)
    lattice = metrics[metrics["role"] == "lattice"]

    # Seed-averaged characteristic function per (cut, model): the twenty cells
    # the decomposition table reports.
    values: dict[tuple[int, str], dict[frozenset[str], float]] = {}
    for (cut, model), block in lattice.groupby(["cut", "model"], sort=True):
        per_seed = [value_function(sub) for _, sub in block.groupby("seed")]
        keys = per_seed[0].keys()
        values[(int(cut), model)] = {
            key: sum(v[key] for v in per_seed) / len(per_seed) for key in keys
        }

    # The pooled game: the grand mean of v over all twenty cells. It is the
    # single game a reader who never disaggregates would be looking at.
    keys = next(iter(values.values())).keys()
    values[(-1, "pooled")] = {
        key: sum(v[key] for v in values.values()) / len(values) for key in keys
    }

    rows, order_rows = [], []
    for (cut, model), value in sorted(values.items()):
        table = {order: marginals(value, order) for order in ORDERS}
        shapley = {
            name: sum(table[order][name] for order in ORDERS) / len(ORDERS)
            for name in CONTROLS
        }
        for name in CONTROLS:
            credits = [table[order][name] for order in ORDERS]
            lo, hi = min(credits), max(credits)
            rows.append(
                {
                    "cell_id": f"{cut}|{model}",
                    "cut": cut,
                    "model": model,
                    "control": name,
                    "shapley": shapley[name],
                    "seq_min": lo,
                    "seq_max": hi,
                    "seq_width": hi - lo,
                    # The two extreme orders are the ones a reader would have to
                    # be told about if only one ablation order were run.
                    "order_at_min": ">".join(
                        min(ORDERS, key=lambda o: table[o][name])
                    ),
                    "order_at_max": ">".join(
                        max(ORDERS, key=lambda o: table[o][name])
                    ),
                    "width_over_abs_shapley": (
                        (hi - lo) / abs(shapley[name]) if shapley[name] else float("nan")
                    ),
                }
            )
        # How far a whole ablation run can sit from the lattice answer, and
        # whether it reorders the controls.
        shapley_rank = sorted(CONTROLS, key=lambda n: -shapley[n])
        for order in ORDERS:
            credit = table[order]
            order_rank = sorted(CONTROLS, key=lambda n: -credit[n])
            order_rows.append(
                {
                    "cell_id": f"{cut}|{model}",
                    "cut": cut,
                    "model": model,
                    "order": ">".join(order),
                    "l1_from_shapley": sum(abs(credit[n] - shapley[n]) for n in CONTROLS),
                    "max_abs_from_shapley": max(abs(credit[n] - shapley[n]) for n in CONTROLS),
                    "top_control": order_rank[0],
                    "top_control_shapley": shapley_rank[0],
                    "ranking_matches_shapley": order_rank == shapley_rank,
                    "top_matches_shapley": order_rank[0] == shapley_rank[0],
                    **{f"credit_{n}": credit[n] for n in CONTROLS},
                }
            )

    per_control = pd.DataFrame(rows)
    per_order = pd.DataFrame(order_rows)

    frames = []
    block = per_control.copy()
    block.insert(0, "block", "per_control")
    frames.append(block)
    block = per_order.copy()
    block.insert(0, "block", "per_order")
    frames.append(block)
    out = pd.concat(frames, ignore_index=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)

    pd.set_option("display.width", 250)
    print("== widest spread per control, over the twenty cells ==")
    print(
        per_control[per_control["model"] != "pooled"].groupby("control")
        .agg(
            shapley_mean=("shapley", "mean"),
            width_min=("seq_width", "min"),
            width_median=("seq_width", "median"),
            width_max=("seq_width", "max"),
        )
        .to_string()
    )
    print("\n== the ten widest (cell, control) pairs ==")
    print(
        per_control[per_control["model"] != "pooled"].sort_values("seq_width", ascending=False)
        .head(10)[
            ["cell_id", "control", "shapley", "seq_min", "seq_max", "seq_width",
             "order_at_min", "order_at_max"]
        ]
        .to_string(index=False)
    )
    print("\n== the order that departs most from the lattice, per cell ==")
    cells_only = per_order[per_order["model"] != "pooled"]
    worst = cells_only.loc[cells_only.groupby("cell_id")["l1_from_shapley"].idxmax()]
    print(
        worst[["cell_id", "order", "l1_from_shapley", "max_abs_from_shapley",
               "top_control", "top_control_shapley"]].to_string(index=False)
    )
    print(
        f"\norders reproducing the full Shapley ranking: "
        f"{int(cells_only['ranking_matches_shapley'].sum())} of {len(cells_only)} "
        f"({cells_only['ranking_matches_shapley'].mean():.3f})"
    )
    print(
        f"orders agreeing on the single largest control: "
        f"{int(cells_only['top_matches_shapley'].sum())} of {len(cells_only)} "
        f"({cells_only['top_matches_shapley'].mean():.3f})"
    )
    print(f"\nwrote {OUT}  {out.shape}")


if __name__ == "__main__":
    main()
