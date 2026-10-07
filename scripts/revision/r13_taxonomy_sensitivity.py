# -*- coding: utf-8 -*-
"""R1.3: leave-one-control-out tren khung 4 control chinh, cong khung 5 control
co B3 lam taxonomy thay the. 11 taxonomy tong cong.

KHONG huan luyen lai. Chi doc results/metrics.parquet va results/c1/metrics.parquet.
Da kiem ngay 06/10/2026, khop voi Bang 4 cua bai o dong "A1 A2 B1 B2".
"""
import itertools
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sift import config  # noqa: E402
from sift.shapley import exact_shapley, gap_values  # noqa: E402

TEN = dict(a1_prior="A1", a2_labels="A2", b1_fs="B1", b2_axis="B2", c1_dedup="B3")


def nap(path, players):
    m = pd.read_parquet(path)
    m = m[m.role == "lattice"]
    games = []
    for c in sorted(m.cut.unique()):
        for mo in sorted(m.model.unique()):
            g = gap_values(m, mo, int(c), control_names=players)
            sel = m[(m.model == mo) & (m.cut == c)]
            base = sel[sel.config_id == 0]
            d0 = (
                base[base.design == "random_fully_matched"].macro_f1.mean()
                - base[base.design == "temporal"].macro_f1.mean()
            )
            games.append((g, d0))
    return games


def danh_gia(games, players, nhan_taxonomy):
    T = frozenset(players)
    phis, vN, D0 = {}, [], []
    for g, d0 in games:
        ph = exact_shapley({S: g[S] for S in g if S <= T})
        for p, x in ph.items():
            phis.setdefault(p, []).append(x)
        vN.append(g[T])
        D0.append(d0)
    mp = {TEN[p]: float(np.mean(x)) for p, x in phis.items()}
    d0, vn = float(np.mean(D0)), float(np.mean(vN))
    xep = [k for k, _ in sorted(mp.items(), key=lambda kv: -kv[1])]
    return dict(
        taxonomy=nhan_taxonomy,
        so_control=len(players),
        D0=round(d0, 4),
        giai_thich_pct=round(100 * vn / d0, 1),
        R=round(d0 - vn, 4),
        vN=round(vn, 4),
        phi_A1=round(mp["A1"], 4) if "A1" in mp else None,
        phi_A2=round(mp["A2"], 4) if "A2" in mp else None,
        phi_B1=round(mp["B1"], 4) if "B1" in mp else None,
        phi_B2=round(mp["B2"], 4) if "B2" in mp else None,
        phi_B3=round(mp["B3"], 4) if "B3" in mp else None,
        control_lon_nhat=xep[0],
        thu_tu=" > ".join(xep),
    )


def main():
    P4 = ["a1_prior", "a2_labels", "b1_fs", "b2_axis"]
    P5 = list(config.CONTROL_NAMES_C1)
    g4 = nap("results/metrics.parquet", P4)
    g5 = nap("results/c1/metrics.parquet", P5)

    rows = [danh_gia(g4, P4, "A1 A2 B1 B2 (khung chinh)")]
    for p in P4:
        rows.append(danh_gia(g4, [x for x in P4 if x != p], "bo " + TEN[p]))
    rows.append(danh_gia(g5, P5, "A1 A2 B1 B2 B3 (5 control)"))
    for p in P5:
        rows.append(danh_gia(g5, [x for x in P5 if x != p], "5-control, bo " + TEN[p]))

    out = pd.DataFrame(rows)
    out.to_csv("results/revision/r13_taxonomy_sensitivity.csv", index=False)
    print(out.to_string(index=False))

    co_a2 = out[out.phi_A2.notna()]
    so_a2_lon_nhat = int((co_a2.control_lon_nhat == "A2").sum())
    print(
        "\nA2 lon nhat o %d / %d taxonomy co chua A2"
        % (so_a2_lon_nhat, len(co_a2))
    )
    print("Khoang giai thich: %.1f%% toi %.1f%%" % (out.giai_thich_pct.min(), out.giai_thich_pct.max()))
    print("")
    print("=== Bo mot control: phan cua no co chui tron vao R khong? ===")
    goc = out.iloc[0]
    for _, r in out.iloc[1:5].iterrows():
        bi_bo = r.taxonomy.replace("bo ", "")
        phi_bi_bo = goc["phi_" + bi_bo]
        R_neu_chui_tron = goc.R + phi_bi_bo
        con_lai = [c for c in ("A1", "A2", "B1", "B2") if c != bi_bo]
        tong_cu = sum(goc["phi_" + c] for c in con_lai)
        tong_moi = sum(r["phi_" + c] for c in con_lai)
        print(
            "  bo %s: phi cu %+.4f | R %.4f -> %.4f (neu chui tron: %.4f, lech %+.4f)"
            % (bi_bo, phi_bi_bo, goc.R, r.R, R_neu_chui_tron, r.R - R_neu_chui_tron)
        )
        print(
            "    tong phi cua %s: %+.4f -> %+.4f (lech %+.4f); tung cai: %s"
            % (
                "+".join(con_lai), tong_cu, tong_moi, tong_moi - tong_cu,
                ", ".join("%s %+.4f->%+.4f" % (c, goc["phi_" + c], r["phi_" + c]) for c in con_lai),
            )
        )

    ngoai_le = co_a2[co_a2.control_lon_nhat != "A2"]
    print("Ngoai le (A2 co mat nhung khong lon nhat):")
    print(ngoai_le[["taxonomy", "control_lon_nhat"]].to_string(index=False) if len(ngoai_le) else "  khong co")


if __name__ == "__main__":
    main()
