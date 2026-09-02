"""F4: is the section 4.4 input shift a collection-source artefact?

Four measurements on the FAMILY panel (PanelSpec() default):

1. source composition by first_submission_date_year (Cramer's V, per-year mix);
2. predictability of source from the 483 features, held out;
3. source composition of train vs test window at each of the five temporal cuts
   (total variation distance);
4. the decisive test: the early-vs-late domain classifier rerun WITHIN the single
   most numerous source, at the three in-scope cuts, with a size-matched
   all-source control so that a drop in accuracy is not read off window size.

Writes results/evidence/f4_*.parquet. Reads sift/ only; modifies nothing there.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sift.config import ControlFlags, ExperimentConfig, PanelSpec, SplitSpec
from sift.controls import apply_controls
from sift.data import build_panel, load_mlran
from sift.drift import MIN_INTERPRETABLE_HELDOUT, domain_classifier_test
from sift.features import feature_columns
from sift.paths import RESULTS_DIR
from sift.seeding import derive_seed

OUT = RESULTS_DIR / "evidence"
TARGET = "ransomware_family"
YEAR = "first_submission_date_year"
COALITION = ControlFlags(a1_prior=True, a2_labels=True, b1_fs=False, b2_axis=True)
CUTS = (2015, 2017, 2019, 2021, 2023)
IN_SCOPE = (2015, 2019, 2021)
SEED = 0
N_MATCH_REPS = 25


def cramers_v(table):
    chi2, p, dof, _ = stats.chi2_contingency(table)
    n = table.sum()
    k = min(table.shape) - 1
    return float(np.sqrt(chi2 / (n * k))), float(p), int(dof)


def tv_distance(a, b):
    keys = sorted(set(a.index) | set(b.index))
    pa = a.reindex(keys).fillna(0.0).to_numpy()
    pb = b.reindex(keys).fillna(0.0).to_numpy()
    return float(0.5 * np.abs(pa - pb).sum())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = load_mlran()
    panel, prov = build_panel(raw, PanelSpec())
    cols = feature_columns(panel)
    print("panel %d samples, %d classes, %d features"
          % (len(panel), panel[TARGET].nunique(), len(cols)))

    # ---- M1 source composition by year ---------------------------------
    tab = pd.crosstab(panel["source"], panel[YEAR])
    v, p, dof = cramers_v(tab.to_numpy())
    print("\n[M1] source x year counts")
    print(tab.to_string())
    mix = tab / tab.sum(axis=0)
    print("\n[M1] per-year source mix (column proportions)")
    print(mix.round(3).to_string())
    print("\n[M1] Cramer's V = %.4f  chi2 p = %.3e  dof = %d  n = %d"
          % (v, p, dof, int(tab.to_numpy().sum())))
    m1 = tab.stack().rename("n").reset_index()
    m1.columns = ["source", "year", "n"]
    totals = tab.sum(axis=0)
    m1["prop_within_year"] = [r.n / totals[r.year] for r in m1.itertuples()]
    m1["cramers_v"] = v
    m1["chi2_p"] = p
    m1["dof"] = dof
    m1["n_panel"] = len(panel)
    m1.to_parquet(OUT / "f4_m1_source_by_year.parquet", index=False)

    x = panel[cols].to_numpy(dtype=np.float64)
    src = panel["source"].to_numpy().astype(str)
    classes = np.unique(src)

    # ---- M2 source predictability from the 483 features ------------------
    models = {
        "logreg": Pipeline([("scale", StandardScaler()),
                            ("clf", LogisticRegression(max_iter=2000,
                                                       class_weight="balanced",
                                                       random_state=SEED))]),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=SEED,
                                                class_weight="balanced", n_jobs=-1),
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    m2_rows = []
    for name, est in models.items():
        oof_pred = np.empty(len(src), dtype=object)
        oof_proba = np.zeros((len(src), len(classes)))
        for tr, te in skf.split(x, src):
            fit = clone(est).fit(x[tr], src[tr])
            oof_pred[te] = fit.predict(x[te])
            proba = fit.predict_proba(x[te])
            order = [list(fit.classes_).index(c) for c in classes]
            oof_proba[te] = proba[:, order]
        acc = accuracy_score(src, oof_pred.astype(str))
        auc = roc_auc_score(src, oof_proba, multi_class="ovr", average="macro",
                            labels=list(classes))
        base = pd.Series(src).value_counts(normalize=True).max()
        print("[M2] %s: accuracy %.4f (majority %.4f)  macro OVR ROC-AUC %.4f"
              % (name, acc, base, auc))
        m2_rows.append({"item": "f4_m2", "model": name, "n": len(src),
                        "n_classes": len(classes), "accuracy": acc,
                        "majority_baseline": float(base), "macro_ovr_roc_auc": auc,
                        "n_folds": 5, "n_features": len(cols)})
    pd.DataFrame(m2_rows).to_parquet(OUT / "f4_m2_source_predictability.parquet",
                                     index=False)

    # ---- windows, exactly as notebook 04 builds them ---------------------
    windows = {}
    for cut in CUTS:
        cfg = ExperimentConfig(panel=PanelSpec(),
                               split=SplitSpec(design="temporal", cut_year=int(cut)),
                               flags=COALITION, model_name="logreg", seed=SEED)
        windows[cut] = apply_controls(panel, cfg)

    # ---- M3 source composition across the split, per cut -----------------
    m3_rows = []
    for cut in CUTS:
        cd = windows[cut]
        ptr = cd.train["source"].value_counts(normalize=True)
        pte = cd.test["source"].value_counts(normalize=True)
        tv = tv_distance(ptr, pte)
        print("[M3] cut %d: axis %s  n_train %4d  n_test %4d  TV = %.4f"
              % (cut, cd.time_column, len(cd.train), len(cd.test), tv))
        print("      train ", {k: round(float(x), 3) for k, x in ptr.items()})
        print("      test  ", {k: round(float(x), 3) for k, x in pte.items()})
        for s in sorted(set(ptr.index) | set(pte.index)):
            m3_rows.append({"item": "f4_m3", "cut": cut, "source": s,
                            "n_train": int((cd.train["source"] == s).sum()),
                            "n_test": int((cd.test["source"] == s).sum()),
                            "p_train": float(ptr.get(s, 0.0)),
                            "p_test": float(pte.get(s, 0.0)),
                            "tv_distance": tv,
                            "n_train_total": len(cd.train),
                            "n_test_total": len(cd.test),
                            "time_column": cd.time_column})
    pd.DataFrame(m3_rows).to_parquet(OUT / "f4_m3_split_source_tv.parquet", index=False)

    # ---- M4 within-source domain classification --------------------------
    top = panel["source"].value_counts().idxmax()
    print("\n[M4] most numerous source on the family panel: %s (n = %d of %d)"
          % (top, int((panel["source"] == top).sum()), len(panel)))

    m4_rows = []
    for cut in IN_SCOPE:
        cd = windows[cut]
        seed = derive_seed(SEED, "domain", cut)
        xe_all = cd.train[cols].to_numpy(dtype=np.float64)
        xl_all = cd.test[cols].to_numpy(dtype=np.float64)
        full = domain_classifier_test(xe_all, xl_all, seed=seed)
        m4_rows.append({"item": "f4_m4", "cut": cut, "source": "ALL",
                        "arm": "all_source_full",
                        "n_early": len(xe_all), "n_late": len(xl_all),
                        "n_early_all": len(xe_all), "n_late_all": len(xl_all),
                        "accuracy": full.accuracy, "excess": full.excess_accuracy,
                        "n_heldout": full.n_heldout, "p_value": full.p_value,
                        "interpretable": full.interpretable,
                        "verdict": full.verdict(0.05)})

        # every source, so that a cut the most numerous source cannot straddle is
        # still answerable by whichever source does straddle it.
        for s in sorted(set(panel["source"].unique())):
            me = (cd.train["source"] == s).to_numpy()
            ml = (cd.test["source"] == s).to_numpy()
            xe, xl = xe_all[me], xl_all[ml]
            arm = "within_source_top" if s == top else "within_source_other"
            row = {"item": "f4_m4", "cut": cut, "source": s, "arm": arm,
                   "n_early": int(me.sum()), "n_late": int(ml.sum()),
                   "n_early_all": len(xe_all), "n_late_all": len(xl_all)}
            if xe.shape[0] >= 2 and xl.shape[0] >= 2:
                w = domain_classifier_test(xe, xl, seed=seed)
                row.update({"accuracy": w.accuracy, "excess": w.excess_accuracy,
                            "n_heldout": w.n_heldout, "p_value": w.p_value,
                            "interpretable": w.interpretable,
                            "verdict": w.verdict(0.05)})
            else:
                row.update({"accuracy": np.nan, "excess": np.nan, "n_heldout": 0,
                            "p_value": np.nan, "interpretable": False,
                            "verdict": "undetermined"})
            m4_rows.append(row)

            # size-matched all-source control: identical window sizes, sources
            # unrestricted, so a fall in accuracy cannot be read off window size.
            rng = np.random.default_rng(seed)
            ne, nl = int(me.sum()), int(ml.sum())
            accs, n_sig = [], 0
            if ne >= 2 and nl >= 2:
                for r in range(N_MATCH_REPS):
                    ie = rng.choice(len(xe_all), size=min(ne, len(xe_all)), replace=False)
                    il = rng.choice(len(xl_all), size=min(nl, len(xl_all)), replace=False)
                    res = domain_classifier_test(xe_all[ie], xl_all[il],
                                                 seed=derive_seed(SEED, "match", cut, s, r))
                    accs.append(res.accuracy)
                    n_sig += int(res.p_value < 0.05)
            m4_rows.append({"item": "f4_m4", "cut": cut, "source": "ALL",
                            "arm": "size_matched_to_" + s,
                            "n_early": ne, "n_late": nl,
                            "n_early_all": len(xe_all), "n_late_all": len(xl_all),
                            "accuracy": float(np.mean(accs)) if accs else np.nan,
                            "accuracy_sd": float(np.std(accs, ddof=1)) if len(accs) > 1 else np.nan,
                            "excess": float(np.mean(accs) - 0.5) if accs else np.nan,
                            "n_heldout": 2 * (min(ne, nl) // 2) if accs else 0,
                            "p_value": np.nan,
                            "frac_significant": (n_sig / len(accs)) if accs else np.nan,
                            "n_reps": len(accs),
                            "interpretable": bool(accs) and 2 * (min(ne, nl) // 2) >= MIN_INTERPRETABLE_HELDOUT,
                            "verdict": "size_matched_control"})

    m4 = pd.DataFrame(m4_rows)
    print("\n[M4]")
    print(m4[["cut", "arm", "n_early", "n_late", "accuracy", "n_heldout",
              "p_value", "interpretable", "verdict"]].to_string(index=False))
    m4.to_parquet(OUT / "f4_m4_within_source_domain.parquet", index=False)
    print("\nwrote:", sorted(f.name for f in OUT.glob("f4_*.parquet")))


if __name__ == "__main__":
    main()
