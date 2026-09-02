"""Counterexample to the conditional-structure test.

Construction:

* ``n_triggers`` activation features, exactly one ON per sample, and ``y`` is the
  index of the ON trigger. ``p(y|x)`` is therefore a deterministic read of ``x``
  and is IDENTICAL in both windows by construction.
* ``n_satellites`` further features, each owned by one class; a satellite fires
  with probability ``p_on`` on rows of its owner and ``p_bg`` elsewhere.
* In the late window a fraction of the satellites are handed to a different
  owner. That changes ``p(x)`` and ``p(x|y)`` and cannot touch ``p(y|x)``.

The exact sample sizes, firing rates and seed of the original run were never
recorded, so they are stated here and carried in the output as columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from sift.drift import conditional_structure_agreement
from sift.paths import RESULTS_DIR

OUT = RESULTS_DIR / "evidence" / "item1_conditional_counterexample.parquet"

N_TRIGGERS = 6
N_SATELLITES = 294
N_EARLY = 1000
N_LATE = 300
P_ON = 0.5
P_BG = 0.05
SEED = 20260827
N_REPEATS = 500
FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)


def make_window(n, owners, rng):
    y = rng.integers(0, N_TRIGGERS, size=n)
    triggers = np.zeros((n, N_TRIGGERS), dtype=np.float64)
    triggers[np.arange(n), y] = 1.0
    owned = owners[None, :] == y[:, None]          # (n, n_satellites)
    prob = np.where(owned, P_ON, P_BG)
    satellites = (rng.random((n, N_SATELLITES)) < prob).astype(np.float64)
    return np.hstack([triggers, satellites]), y


def reassign(owners, fraction, rng):
    owners = owners.copy()
    k = int(round(fraction * N_SATELLITES))
    if k == 0:
        return owners
    picked = rng.choice(N_SATELLITES, size=k, replace=False)
    shift = rng.integers(1, N_TRIGGERS, size=k)     # never 0, so the owner changes
    owners[picked] = (owners[picked] + shift) % N_TRIGGERS
    return owners


def perfect_recovery(x_early, y_early, x_late, y_late, seed):
    """Does a classifier reading x recover y in BOTH windows?"""
    out = {}
    for name, (xtr, ytr, xte, yte) in {
        "early_on_early": (x_early, y_early, x_early, y_early),
        "late_on_late": (x_late, y_late, x_late, y_late),
        "early_to_late": (x_early, y_early, x_late, y_late),
        "late_to_early": (x_late, y_late, x_early, y_early),
    }.items():
        model = LogisticRegression(max_iter=2000, random_state=seed).fit(xtr, ytr)
        out[f"acc_{name}"] = float((model.predict(xte) == yte).mean())
    return out


def main() -> None:
    rows = []
    for fraction in FRACTIONS:
        rng = np.random.default_rng(SEED)
        owners = rng.integers(0, N_TRIGGERS, size=N_SATELLITES)
        x_early, y_early = make_window(N_EARLY, owners, rng)
        late_owners = reassign(owners, fraction, rng)
        x_late, y_late = make_window(N_LATE, late_owners, rng)

        result = conditional_structure_agreement(
            x_early, y_early, x_late, y_late, seed=SEED, n_repeats=N_REPEATS
        )
        acc = perfect_recovery(x_early, y_early, x_late, y_late, SEED)
        flagged = bool(result["below_within_null"])
        rows.append(
            {
                "item": "item1_conditional_counterexample",
                "scenario": (
                    "identical p(x)" if fraction == 0 else f"{fraction:.0%} satellites re-owned"
                ),
                "fraction_reowned": fraction,
                "n_triggers": N_TRIGGERS,
                "n_satellites": N_SATELLITES,
                "n_features": N_TRIGGERS + N_SATELLITES,
                "n_early": N_EARLY,
                "n_late": N_LATE,
                "p_on": P_ON,
                "p_background": P_BG,
                "seed": SEED,
                "n_repeats": N_REPEATS,
                "conditional_value_across": float(result["across"]),
                "across_matched": float(result["across_matched"]),
                "within_mean": float(result["within_mean"]),
                "within_lo": float(result["within_lo"]),
                "within_hi": float(result["within_hi"]),
                "ratio_to_null": float(result["ratio"]),
                "below_within_null": flagged,
                "rule_says": "concept drift" if flagged else "no evidence",
                "p_within_empirical": float(result["p_within_empirical"]),
                "n_within_draws": int(result["n_within_draws"]),
                "n_shared_classes": int(result["n_shared_classes"]),
                "metric": "spearman agreement of the feature-label correlation vector",
                **acc,
                "recovers_y_perfectly_both_windows": bool(
                    acc["acc_early_on_early"] == 1.0
                    and acc["acc_late_on_late"] == 1.0
                ),
            }
        )
        print(rows[-1]["scenario"], "across=%.4f ratio=%.4f flagged=%s" % (
            rows[-1]["conditional_value_across"], rows[-1]["ratio_to_null"], flagged), flush=True)

    frame = pd.DataFrame(rows)
    frame.to_parquet(OUT, index=False)
    cols = [
        "scenario", "conditional_value_across", "within_mean", "ratio_to_null",
        "rule_says", "acc_early_on_early", "acc_late_on_late",
        "acc_early_to_late", "acc_late_to_early",
    ]
    print(frame[cols].to_string(index=False))
    print("wrote", OUT, frame.shape)


if __name__ == "__main__":
    main()
