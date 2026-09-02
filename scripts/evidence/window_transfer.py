"""Item 6: within-window learnability against cross-window transfer.

Logistic regression, shared classes only, 5-fold stratified CV inside each
window, and a straight fit-on-one-window / test-on-the-other in both directions.
Run at every cut; the paper quotes cut 2015.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from sift import PanelSpec, SplitSpec, build_panel, build_model, load_mlran, temporal_split
from sift.features import feature_columns
from sift.paths import RESULTS_DIR

CUTS = (2015, 2017, 2019, 2021, 2023)
OUT = RESULTS_DIR / "evidence" / "item6_transfer_cut2015.parquet"
SEED = 0


def _scores(y_true, y_pred):
    labels = np.unique(y_true)
    return (
        float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0.0)),
        float(accuracy_score(y_true, y_pred)),
    )


def cv_score(x, y, seed):
    counts = pd.Series(y).value_counts()
    if counts.size < 2 or len(y) < 5:
        return np.nan, np.nan, 0
    # A class with fewer than five members makes StratifiedKFold warn but it
    # still produces five folds; the warning is left visible rather than
    # silenced, and the guard above only refuses what is actually undefined.
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    f1s, accs = [], []
    for tr, te in folds.split(x, y):
        model = build_model("logreg", seed)
        model.fit(x[tr], y[tr])
        f1, acc = _scores(y[te], model.predict(x[te]))
        f1s.append(f1)
        accs.append(acc)
    return float(np.mean(f1s)), float(np.mean(accs)), 5


def main() -> None:
    raw = load_mlran()
    spec = PanelSpec(task="family")
    panel, _ = build_panel(raw, spec)
    cols = feature_columns(panel)
    target = spec.target_column

    rows = []
    for cut in CUTS:
        split = SplitSpec(design="temporal", cut_year=cut, test_window=3)
        early_idx, late_idx = temporal_split(panel, split, spec.primary_axis)
        early = panel.loc[early_idx]
        late = panel.loc[late_idx]
        shared = sorted(set(early[target]) & set(late[target]))
        e = early[early[target].isin(shared)]
        l = late[late[target].isin(shared)]
        if len(shared) < 2 or e.empty or l.empty:
            continue
        xe, ye = e[cols].to_numpy(float), e[target].to_numpy()
        xl, yl = l[cols].to_numpy(float), l[target].to_numpy()

        common = {
            "item": "item6_transfer",
            "cut": cut,
            "model": "logreg",
            "n_shared_classes": len(shared),
            "n_early": int(len(e)),
            "n_late": int(len(l)),
            "seed": SEED,
        }
        f1, acc, k = cv_score(xe, ye, SEED)
        rows.append({**common, "comparison": "cv_early", "n_folds": k,
                     "n": int(len(e)), "macro_f1": f1, "accuracy": acc})
        f1, acc, k = cv_score(xl, yl, SEED)
        rows.append({**common, "comparison": "cv_late", "n_folds": k,
                     "n": int(len(l)), "macro_f1": f1, "accuracy": acc})

        model = build_model("logreg", SEED)
        model.fit(xe, ye)
        f1, acc = _scores(yl, model.predict(xl))
        rows.append({**common, "comparison": "early_to_late", "n_folds": 0,
                     "n": int(len(l)), "macro_f1": f1, "accuracy": acc})

        model = build_model("logreg", SEED)
        model.fit(xl, yl)
        f1, acc = _scores(ye, model.predict(xe))
        rows.append({**common, "comparison": "late_to_early", "n_folds": 0,
                     "n": int(len(e)), "macro_f1": f1, "accuracy": acc})
        print("cut", cut, "done", flush=True)

    frame = pd.DataFrame(rows)
    frame["metric_note"] = "5-fold CV within window; direct transfer across windows"
    frame.to_parquet(OUT, index=False)
    print(frame.to_string(index=False))
    print("wrote", OUT, frame.shape)


if __name__ == "__main__":
    main()
