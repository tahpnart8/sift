# -*- coding: utf-8 -*-
"""R1.4: tai tao DUNG phuong phap bootstrap cua bai (ghep noi draw qua 5 seed,
khong lay trung binh), roi them hieu chinh Holm cho 20 phep kiem va bootstrap
ghep cap giua 4 mo hinh trong cung mot cut.

CANH BAO DA TUNG SAI: lay trung binh qua seed thay vi ghep noi se cho khoang
tin cay hep hon that, va se bao sai so o loai tru 0 (20/20 thay vi dung 17/20).
Luon doi chieu voi results/shapley_2024/residual_ci.parquet truoc, lech phai
bang 0.0 thi moi tin cac buoc sau.

Da kiem ngay 06/10/2026 trong phien chinh: lech = 0.0, 17/20 giu nguyen sau Holm,
5/5 cut loai tru 0 khi ghep cap qua mo hinh.
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sift.config import PanelSpec  # noqa: E402
from sift.data import build_panel, load_mlran  # noqa: E402
from sift.drift import bootstrap_gap_distribution  # noqa: E402
from sift.experiment import lattice_cells  # noqa: E402
from sift.reporting import class_prior_reference, recompute_a1_weights  # noqa: E402
from sift.seeding import derive_seed  # noqa: E402

ALPHA, N_BOOT, FULL = 0.05, 1000, 15


def arms(by_fit, prior_reference, block, seed):
    out = {}
    for design in ("random_fully_matched", "temporal"):
        row = block[(block.seed == seed) & (block.design == design)].iloc[0]
        fr = by_fit[row["fit_id"]]
        yt = fr["y_true"].to_numpy()
        w = recompute_a1_weights(yt, prior_reference) if bool(row["a1_prior"]) else None
        out[design] = (yt, fr["y_pred"].to_numpy(), w)
    return out


def holm(ps):
    n = len(ps)
    order = np.argsort(ps)
    adj = np.empty(n)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (n - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def pval(d):
    return min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0)))


def main():
    metrics = pd.read_parquet("results/metrics.parquet")
    predictions = pd.read_parquet("results/predictions.parquet")
    panel, _ = build_panel(load_mlran(), PanelSpec())
    prior_reference = class_prior_reference(panel)
    by_fit = {n: b for n, b in predictions.groupby("fit_id", sort=False)}
    lat = lattice_cells(metrics)
    full = lat[lat.config_id == FULL]

    # ---- buoc 0: doi chieu voi ket qua goc cua bai, BAT BUOC lech = 0.0 ----
    rows, draws_cell = [], {}
    for (cut, model), block in full.groupby(["cut", "model"], sort=True):
        ds = []
        for seed in sorted(block.seed.unique()):
            a = arms(by_fit, prior_reference, block, seed)
            ds.append(
                bootstrap_gap_distribution(
                    a["random_fully_matched"],
                    a["temporal"],
                    n_boot=N_BOOT,
                    seed=derive_seed(0, "boot_R", int(cut), model, int(seed)),
                )
            )
        pooled = np.concatenate(ds)
        draws_cell[(int(cut), model)] = pooled
        lo, hi = np.percentile(pooled, [2.5, 97.5])
        rows.append(
            dict(
                cut=int(cut),
                model=model,
                R=pooled.mean(),
                ci_low=lo,
                ci_high=hi,
                excludes_zero=bool(lo > 0 or hi < 0),
                p=pval(pooled),
            )
        )
    t = pd.DataFrame(rows)

    goc = pd.read_parquet("results/shapley_2024/residual_ci.parquet")
    ss = t.merge(
        goc[["cut", "model", "R", "excludes_zero"]],
        on=["cut", "model"],
        suffixes=("_moi", "_bai"),
    )
    lech = (ss.R_moi - ss.R_bai).abs().max()
    assert lech < 1e-6, (
        "LECH SO VOI BAI: %.6f. DUNG LAI, DUNG DUA SO NAY VAO BAI. "
        "Kiem tra lai phuong phap ghep noi qua seed truoc khi tiep tuc." % lech
    )
    print("Doi chieu voi bai: lech = %.2e (dat, phai bang 0)" % lech)
    print("So o loai tru 0, bai ghi: %d | tai tao: %d" % (int(goc.excludes_zero.sum()), int(t.excludes_zero.sum())))

    # ---- buoc 1: hieu chinh Holm ----
    t["p_holm"] = holm(t.p.to_numpy())
    t["excludes_zero_holm"] = t.p_holm < 0.05
    t.to_csv("results/revision/r14_per_cell_holm.csv", index=False)
    print()
    print(t.round(4).to_string(index=False))
    print()
    print("Loai tru 0 truoc Holm: %d / 20" % int(t.excludes_zero.sum()))
    print("Loai tru 0 sau Holm  : %d / 20" % int(t.excludes_zero_holm.sum()))

    # ---- buoc 2: bootstrap ghep cap qua 4 mo hinh, theo tung cut ----
    paired_rows = []
    for cut in sorted(full.cut.unique()):
        block_all = full[full.cut == cut]
        per_model = []
        for model in sorted(block_all.model.unique()):
            block = block_all[block_all.model == model]
            ds = []
            for seed in sorted(block.seed.unique()):
                a = arms(by_fit, prior_reference, block, seed)
                ds.append(
                    bootstrap_gap_distribution(
                        a["random_fully_matched"],
                        a["temporal"],
                        n_boot=N_BOOT,
                        seed=derive_seed(0, "paired", int(cut), int(seed)),
                    )
                )
            per_model.append(np.concatenate(ds))
        gop = np.mean(per_model, axis=0)
        lo, hi = np.percentile(gop, [2.5, 97.5])
        paired_rows.append(
            dict(
                cut=int(cut),
                R_gop_4_mohinh=gop.mean(),
                ci_low=lo,
                ci_high=hi,
                excludes_zero=bool(lo > 0 or hi < 0),
                p=pval(gop),
            )
        )
    tp = pd.DataFrame(paired_rows)
    tp.to_csv("results/revision/r14_paired_bootstrap.csv", index=False)
    print()
    print(tp.round(4).to_string(index=False))
    print()
    print("Cut loai tru 0 khi ghep cap qua mo hinh: %d / 5" % int(tp.excludes_zero.sum()))


if __name__ == "__main__":
    main()
