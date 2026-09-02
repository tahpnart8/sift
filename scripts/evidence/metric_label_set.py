"""Item 3: the scikit-learn ``labels`` default in macro-F1.

Recomputes macro-F1 twice on the SAME stored predictions, once with ``labels``
pinned to ``numpy.unique(y_true)`` (what :func:`sift.metrics.compute_metrics`
does) and once with the library default (the union of true and predicted
labels). No model is refitted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sift.paths import RESULTS_DIR

ROOT = "results"
OUT = RESULTS_DIR / "evidence" / "item3_labels_default.parquet"


def main() -> None:
    preds = pd.read_parquet(f"{ROOT}/predictions.parquet")
    meta = pd.read_parquet(f"{ROOT}/metrics.parquet")
    meta = meta[["fit_id", "config_id", "design", "role", "cut", "seed", "model"]]
    joined = preds.merge(meta, on="fit_id", how="inner")

    rows = []
    for (design, config_id, cut, model, seed), g in joined.groupby(
        ["design", "config_id", "cut", "model", "seed"], observed=True
    ):
        y_true = g["y_true"].to_numpy()
        y_pred = g["y_pred"].to_numpy()
        present = np.unique(y_true)
        union = np.union1d(present, np.unique(y_pred))
        rows.append(
            {
                "design": design,
                "config_id": int(config_id),
                "cut": int(cut),
                "model": model,
                "seed": int(seed),
                "n": int(len(g)),
                "k_true": int(present.size),
                "k_union": int(union.size),
                "macro_f1_labels_pinned": f1_score(
                    y_true, y_pred, labels=present, average="macro", zero_division=0.0
                ),
                "macro_f1_labels_default": f1_score(
                    y_true, y_pred, average="macro", zero_division=0.0
                ),
            }
        )
    per_fit = pd.DataFrame(rows)

    agg = (
        per_fit.groupby(["design", "config_id", "cut"], as_index=False)
        .agg(
            n_fits=("n", "size"),
            n=("n", "mean"),
            k_true=("k_true", "mean"),
            k_union_min=("k_union", "min"),
            k_union_max=("k_union", "max"),
            macro_f1_labels_pinned=("macro_f1_labels_pinned", "mean"),
            macro_f1_labels_default=("macro_f1_labels_default", "mean"),
        )
    )
    agg["drop"] = agg["macro_f1_labels_pinned"] - agg["macro_f1_labels_default"]
    agg["item"] = "item3_labels_default"
    agg["metric"] = "macro_f1"
    agg["aggregation"] = "mean over 4 models x 10 seeds"
    agg.to_parquet(OUT, index=False)

    show = agg[(agg.design == "temporal") & (agg.config_id == 15)]
    print(show.to_string(index=False))
    print()
    print("--- baseline coalition (config 0), temporal ---")
    print(
        agg[(agg.design == "temporal") & (agg.config_id == 0)].to_string(index=False)
    )
    print("wrote", OUT, agg.shape)


if __name__ == "__main__":
    main()
