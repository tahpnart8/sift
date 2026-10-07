# -*- coding: utf-8 -*-
"""R1.2 phan ho tro: tinh lai toan bo phan tach Shapley tren 4 metric da luu san
trong moi fit, de tra loi cau hoi 48.7% co phu thuoc vao viec chon macro-F1 khong.

KHONG huan luyen lai. Chi doc results/metrics.parquet.
Da kiem ngay 06/10/2026 trong phien chinh, ket qua khop voi Bang 4 cua bai o cot macro_f1.
"""
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sift.shapley import exact_shapley, gap_values  # noqa: E402

P = ["a1_prior", "a2_labels", "b1_fs", "b2_axis"]
TEN = dict(a1_prior="A1", a2_labels="A2", b1_fs="B1", b2_axis="B2")


def main():
    m = pd.read_parquet("results/metrics.parquet")
    m = m[m.role == "lattice"]
    cuts, models = sorted(m.cut.unique()), sorted(m.model.unique())

    rows = []
    for met in ["macro_f1", "balanced_accuracy", "mcc", "accuracy"]:
        phis = {p: [] for p in P}
        vN, D0 = [], []
        for c in cuts:
            for mo in models:
                g = gap_values(m, mo, int(c), metric_column=met)
                ph = exact_shapley(g)
                for p in P:
                    phis[p].append(ph[p])
                vN.append(g[frozenset(P)])
                sel = m[(m.model == mo) & (m.cut == c) & (m.config_id == 0)]
                D0.append(
                    sel[sel.design == "random_fully_matched"][met].mean()
                    - sel[sel.design == "temporal"][met].mean()
                )
        mp = {p: float(np.mean(v)) for p, v in phis.items()}
        d0, vn = float(np.mean(D0)), float(np.mean(vN))
        xep = [TEN[k] for k, _ in sorted(mp.items(), key=lambda kv: -kv[1])]
        rows.append(
            dict(
                metric=met,
                D0=round(d0, 4),
                phi_A1=round(mp["a1_prior"], 4),
                phi_A2=round(mp["a2_labels"], 4),
                phi_B1=round(mp["b1_fs"], 4),
                phi_B2=round(mp["b2_axis"], 4),
                giai_thich_pct=round(100 * vn / d0, 1),
                R=round(d0 - vn, 4),
                thu_tu=" > ".join(xep),
            )
        )
    out = pd.DataFrame(rows)
    out.to_csv("results/revision/r12_metric_robustness.csv", index=False)
    print(out.to_string(index=False))

    # kiem tra: thu tu co giong het macro_f1 khong qua ca 4 metric
    thu_tu_macro = out.loc[out.metric == "macro_f1", "thu_tu"].iloc[0]
    tat_giong_nhau = (out.thu_tu == thu_tu_macro).all()
    with open("results/revision/r12_metric_robustness.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "thu_tu_giong_nhau_qua_4_metric": bool(tat_giong_nhau),
                "macro_f1_la_bao_thu_nhat": bool(
                    out.loc[out.metric == "macro_f1", "giai_thich_pct"].iloc[0]
                    == out.giai_thich_pct.min()
                ),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print("\nThu tu giong nhau qua 4 metric:", tat_giong_nhau)


if __name__ == "__main__":
    main()
