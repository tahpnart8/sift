"""Item 7: correlation between per-class prior deviation and per-class recall.

Measured on the ORIGINAL ``a1_only`` recovery panel, the one whose target
distribution boosts every second family in alphabetical order
(``InjectionSpec(a1_align='name')``). That is the panel on which control A1
recovered nothing despite a total variation of 0.244.

The fits are pulled from the recovery fit cache; nothing is refitted when the
cache is warm.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import recall_score

from sift import PanelSpec, build_panel, load_mlran
from sift.experiment import panel_fingerprint, run_cell
from sift.paths import RESULTS_DIR

from sift.recovery import (
    InjectionSpec,
    _config,
    _halves,
    build_recovery_panel,
    recovery_cache,
)

OUT = RESULTS_DIR / "evidence" / "item7_a1_prior_recall_correlation.parquet"


def main() -> None:
    raw = load_mlran()
    panel_spec = PanelSpec(task="family")
    panel, _ = build_panel(raw, panel_spec)
    spec = InjectionSpec()  # a1_align='name' is the default, i.e. the original
    assert spec.a1_align == "name"

    built = build_recovery_panel(panel, panel_spec, "a1_only", spec, seed=0)
    rp = built.panel
    target = panel_spec.target_column

    _, test_idx = _halves(rp, spec, panel_spec.primary_axis)
    corpus = rp[target].value_counts(normalize=True)
    window = rp.loc[test_idx, target].value_counts(normalize=True)
    classes = sorted(set(corpus.index) | set(window.index))
    corpus_share = corpus.reindex(classes, fill_value=0.0)
    window_share = window.reindex(classes, fill_value=0.0)

    digest = panel_fingerprint(rp)
    rows = []
    with recovery_cache():
        for seed in spec.seeds:
            cfg = _config(panel_spec, spec, "temporal", 0, seed)
            record = run_cell(rp, cfg, panel_digest=digest)
            pred = record["predictions"]
            y_true = np.asarray(pred["y_true"])
            y_pred = np.asarray(pred["y_pred"])
            # run_cell label-encodes; recover the class names via the panel rows.
            sample_ids = np.asarray(pred["sample_id"])
            name_of = rp.set_index("sample_id")[target]
            true_names = name_of.loc[sample_ids].to_numpy()
            code_to_name = {}
            for code, nm in zip(y_true, true_names):
                code_to_name[int(code)] = nm
            present_codes = np.unique(y_true)
            rec = recall_score(
                y_true, y_pred, labels=present_codes, average=None, zero_division=0
            )
            present_names = [code_to_name[int(c)] for c in present_codes]

            dev = np.array(
                [float(window_share[n] - corpus_share[n]) for n in present_names]
            )
            ratio = np.array(
                [
                    float(window_share[n] / corpus_share[n])
                    if corpus_share[n] > 0
                    else np.nan
                    for n in present_names
                ]
            )
            for label, x in (
                ("window_share_minus_corpus_share", dev),
                ("window_share_over_corpus_share", ratio),
            ):
                ok = np.isfinite(x)
                pear = stats.pearsonr(x[ok], rec[ok])
                spear = stats.spearmanr(x[ok], rec[ok])
                rows.append(
                    {
                        "item": "item7_a1_prior_recall_correlation",
                        "panel": "a1_only",
                        "a1_align": spec.a1_align,
                        "a1_skew": spec.a1_skew,
                        "cut": spec.cut,
                        "model": spec.model_name,
                        "seed": int(seed),
                        "design": "temporal",
                        "config_id": 0,
                        "deviation_definition": label,
                        "metric": "per_class_recall",
                        "n": int(ok.sum()),
                        "n_test": int(len(y_true)),
                        "pearson_r": float(pear.statistic),
                        "pearson_p": float(pear.pvalue),
                        "spearman_rho": float(spear.statistic),
                        "spearman_p": float(spear.pvalue),
                        "tv_window_vs_corpus": float(
                            0.5 * np.abs(window_share - corpus_share).sum()
                        ),
                        "mean_recall": float(rec.mean()),
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_parquet(OUT, index=False)
    print(
        frame[
            [
                "seed",
                "deviation_definition",
                "n",
                "pearson_r",
                "pearson_p",
                "spearman_rho",
                "tv_window_vs_corpus",
            ]
        ].to_string(index=False)
    )
    print()
    print(
        frame.groupby("deviation_definition")[["pearson_r", "spearman_rho"]]
        .mean()
        .to_string()
    )
    print("wrote", OUT, frame.shape)


if __name__ == "__main__":
    main()
