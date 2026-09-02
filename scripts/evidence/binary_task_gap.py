"""Item 2: the four numbers of the binary ransomware-vs-goodware task.

(a) The gap in PR-AUC between a random split and a temporal split, four models,
    five cuts.
(b) ROC-AUC of predicting the collection source from the same 483 features,
    WITHIN the ransomware class only, pairwise between sources, 5-fold CV.
(c) PR-AUC of a model trained on ransomware 2012-2014 and tested on 2021-2023.
(d) Recall of that model on the test-window families never seen in training.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from sift import (
    MODEL_NAMES,
    PanelSpec,
    SplitSpec,
    build_model,
    build_panel,
    load_mlran,
    make_split,
    temporal_split,
)
from sift.features import feature_columns
from sift.seeding import derive_seed
from sift.paths import RESULTS_DIR

CUTS = (2015, 2017, 2019, 2021, 2023)
SEEDS = (0, 1, 2, 3, 4)
OUT_GAP = RESULTS_DIR / "evidence" / "item2_binary_gap.parquet"
OUT_SRC = RESULTS_DIR / "evidence" / "item2_source_shortcut.parquet"
OUT_STRESS = RESULTS_DIR / "evidence" / "item2_binary_stress_test.parquet"


def _proba(model, x):
    p = model.predict_proba(x)
    return p[:, list(model.classes_).index(1)]


def part_a(panel, cols, spec):
    x = panel[cols].to_numpy(float)
    y = panel["sample_type"].to_numpy(int)
    rows = []
    for model_name in MODEL_NAMES:
        for seed in SEEDS:
            # Random arm: stratified 75/25, independent of the cut.
            tr, te = make_split(
                panel,
                SplitSpec(design="random", test_size=0.25),
                spec.primary_axis,
                derive_seed(seed, "split", "random", 0),
                stratify_labels=panel["sample_type"],
                label_column="sample_type",
                restrict_labels=False,
            )
            m = build_model(model_name, derive_seed(seed, "model", model_name, 0))
            itr = panel.index.get_indexer(tr)
            ite = panel.index.get_indexer(te)
            m.fit(x[itr], y[itr])
            random_pr = float(average_precision_score(y[ite], _proba(m, x[ite])))
            for cut in CUTS:
                tr2, te2 = temporal_split(
                    panel,
                    SplitSpec(design="temporal", cut_year=cut, test_window=3),
                    spec.primary_axis,
                )
                i2 = panel.index.get_indexer(tr2)
                j2 = panel.index.get_indexer(te2)
                if len(np.unique(y[i2])) < 2 or len(np.unique(y[j2])) < 2:
                    continue
                m2 = build_model(model_name, derive_seed(seed, "model", model_name, cut))
                m2.fit(x[i2], y[i2])
                temporal_pr = float(average_precision_score(y[j2], _proba(m2, x[j2])))
                rows.append(
                    {
                        "item": "item2_binary_gap",
                        "task": "binary",
                        "metric": "pr_auc",
                        "model": model_name,
                        "seed": seed,
                        "cut": cut,
                        "n_train_temporal": int(len(tr2)),
                        "n_test_temporal": int(len(te2)),
                        "n_train_random": int(len(tr)),
                        "n_test_random": int(len(te)),
                        "pr_auc_random": random_pr,
                        "pr_auc_temporal": temporal_pr,
                        "gap": random_pr - temporal_pr,
                    }
                )
        print("part a:", model_name, "done", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(OUT_GAP, index=False)
    print(
        frame.groupby("model")[["pr_auc_random", "pr_auc_temporal", "gap"]]
        .mean()
        .to_string()
    )
    print("overall mean gap %.4f" % frame["gap"].mean())
    print(frame.groupby("cut")["gap"].mean().to_string())
    print("wrote", OUT_GAP, frame.shape)
    return frame


def part_b(panel, cols, spec):
    ranso = panel[panel["sample_type"] == 1]
    rows = []
    sources = sorted(ranso["source"].unique())
    for a, b in itertools.combinations(sources, 2):
        sub = ranso[ranso["source"].isin([a, b])]
        x = sub[cols].to_numpy(float)
        y = (sub["source"] == b).to_numpy(int)
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        for model_name in ("logreg", "random_forest"):
            scores = []
            for tr, te in folds.split(x, y):
                m = build_model(model_name, 0)
                m.fit(x[tr], y[tr])
                scores.append(float(roc_auc_score(y[te], m.predict_proba(x[te])[:, 1])))
            rows.append(
                {
                    "item": "item2_source_shortcut",
                    "scope": "ransomware class only",
                    "pair": a + " vs " + b,
                    "source_a": a,
                    "source_b": b,
                    "model": model_name,
                    "metric": "roc_auc",
                    "n": int(len(sub)),
                    "n_a": int((sub["source"] == a).sum()),
                    "n_b": int((sub["source"] == b).sum()),
                    "n_folds": 5,
                    "roc_auc": float(np.mean(scores)),
                    "roc_auc_sd": float(np.std(scores)),
                }
            )
        print("part b:", a, b, "done", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(OUT_SRC, index=False)
    print(frame[["pair", "model", "n", "roc_auc"]].to_string(index=False))
    print("wrote", OUT_SRC, frame.shape)
    return frame


def part_c(panel, cols, spec):
    axis = spec.primary_axis
    train = panel[panel[axis].between(2012, 2014)]
    test = panel[panel[axis].between(2021, 2023)]
    seen = set(train.loc[train["sample_type"] == 1, "ransomware_family"])
    test_r = test[test["sample_type"] == 1]
    unseen_mask = ~test_r["ransomware_family"].isin(seen)
    n_unseen_fam = int(test_r.loc[unseen_mask, "ransomware_family"].nunique())
    n_unseen_rows = int(unseen_mask.sum())

    xtr = train[cols].to_numpy(float)
    ytr = train["sample_type"].to_numpy(int)
    xte = test[cols].to_numpy(float)
    yte = test["sample_type"].to_numpy(int)
    unseen_idx = test.index.get_indexer(test_r.index[unseen_mask])
    rows = []
    for model_name in MODEL_NAMES:
        m = build_model(model_name, 0)
        m.fit(xtr, ytr)
        proba = _proba(m, xte)
        pred = m.predict(xte)
        rows.append(
            {
                "item": "item2_binary_stress_test",
                "train_years": "2012-2014",
                "test_years": "2021-2023",
                "model": model_name,
                "seed": 0,
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "n_unseen_family_rows": n_unseen_rows,
                "n_unseen_families": n_unseen_fam,
                "pr_auc": float(average_precision_score(yte, proba)),
                "accuracy": float((pred == yte).mean()),
                "recall_all_ransomware": float(
                    recall_score(yte, pred, pos_label=1, zero_division=0)
                ),
                "recall_unseen_families": float((pred[unseen_idx] == 1).mean()),
            }
        )
        print("part c:", model_name, "done", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(OUT_STRESS, index=False)
    print(frame.to_string(index=False))
    print("wrote", OUT_STRESS, frame.shape)
    return frame


def main() -> None:
    raw = load_mlran()
    spec = PanelSpec(task="binary")
    panel, prov = build_panel(raw, spec)
    cols = feature_columns(panel)
    print("binary panel:", prov["after_task_filter"], "rows,", len(cols), "features")
    part_c(panel, cols, spec)
    print()
    part_b(panel, cols, spec)
    print()
    part_a(panel, cols, spec)


if __name__ == "__main__":
    main()
