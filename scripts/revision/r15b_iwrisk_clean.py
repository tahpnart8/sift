# -*- coding: utf-8 -*-
"""R1.5, tang 2: phien ban gon hon cua kich ban tong hop, de co it nhat mot ca
sach chung minh dung/sai thay vi bi guard ESS chan toan bo.

Phat hien tu lan chay truoc (r15_iwrisk_validation.py, luu o
r15_iwrisk_validation.csv): voi 294 dac trung ve tinh gan nhieu va 1000 mau,
guard ESS chan hau het moi kich ban, ke ca khi khong co khac biet thuc su nao
theo cau tao. Day la mot phat hien that, khong phai loi: no cho thay guard rat
than trong khi dac trung nhieu chieu, dung kieu du lieu MLRan (483 dac trung).
Giu nguyen ket qua do lam bang chung phu, xem r15_iwrisk_validation.csv.

Ban nay giam con 20 dac trung ve tinh va tang mau len 2000, de density ratio
uoc luong duoc on dinh hon, nham co it nhat mot kich ban interpretable=True
cho ca hai phia covariate-shift-thuan va concept-drift-that.
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sift.iwrisk import importance_weighted_risk  # noqa: E402

N_TRIGGERS = 6
N_SATELLITES = 20          # giam manh so voi 294 cua ban goc
N_EARLY = 2000              # tang so voi 1000 cua ban goc
N_LATE = 500
P_ON = 0.6
P_BG = 0.1
SEED = 20261007


def make_window(n, owners, rng, trigger_pool=None, label_shift=0):
    pool = np.arange(N_TRIGGERS) if trigger_pool is None else np.asarray(trigger_pool)
    trig_idx = rng.integers(0, len(pool), size=n)
    x_trig_class = pool[trig_idx]
    y = (x_trig_class + label_shift) % N_TRIGGERS

    triggers = np.zeros((n, N_TRIGGERS), dtype=np.float64)
    triggers[np.arange(n), x_trig_class] = 1.0
    owned = owners[None, :] == x_trig_class[:, None]
    prob = np.where(owned, P_ON, P_BG)
    satellites = (rng.random((n, N_SATELLITES)) < prob).astype(np.float64)
    return np.hstack([triggers, satellites]), y


def reassign(owners, fraction, rng):
    owners = owners.copy()
    k = int(round(fraction * N_SATELLITES))
    if k == 0:
        return owners
    picked = rng.choice(N_SATELLITES, size=k, replace=False)
    shift = rng.integers(1, N_TRIGGERS, size=k)
    owners[picked] = (owners[picked] + shift) % N_TRIGGERS
    return owners


def chay(ten, x_e, y_e, x_l, y_l, seed):
    r = importance_weighted_risk(x_e, y_e, x_l, y_l, seed=seed, metric="macro_f1")
    s = r.summary()
    s["kich_ban"] = ten
    print(
        "%-34s verdict=%-30s gap=%.4f KTC[%.4f,%.4f] ESS=%.1f(%.1f%%) interpretable=%s"
        % (
            ten, s["verdict"], s["conditional_gap_clipped"],
            s["gap_ci_clipped"][0], s["gap_ci_clipped"][1],
            s["ess_clipped"], 100 * s["ess_fraction_clipped"], s["interpretable"],
        )
    )
    return s


def main():
    ket_qua = []

    print("=== Tang 2a: covariate shift thuan, muc do tang dan ===")
    for frac in (0.0, 0.3, 0.6, 1.0):
        rng = np.random.default_rng(SEED)
        owners = rng.integers(0, N_TRIGGERS, size=N_SATELLITES)
        x_e, y_e = make_window(N_EARLY, owners, rng)
        late_owners = reassign(owners, frac, rng)
        x_l, y_l = make_window(N_LATE, late_owners, rng)
        ket_qua.append(chay("2a_covariate_frac%.1f" % frac, x_e, y_e, x_l, y_l, SEED))

    print()
    print("=== Tang 2b: bom concept drift that, p(x) gan nhu khong doi ===")
    for shift in (0, 1, 3, 5):
        rng = np.random.default_rng(SEED)
        owners = rng.integers(0, N_TRIGGERS, size=N_SATELLITES)
        x_e, y_e = make_window(N_EARLY, owners, rng, label_shift=0)
        x_l, y_l = make_window(N_LATE, owners, rng, label_shift=shift)
        ket_qua.append(chay("2b_conceptdrift_shift%d" % shift, x_e, y_e, x_l, y_l, SEED))

    out = pd.DataFrame(ket_qua)
    out.to_csv("results/revision/r15b_iwrisk_clean.csv", index=False)

    print()
    print("=== Kiem tieu chi ===")
    a = out[out.kich_ban.str.startswith("2a_")]
    b = out[out.kich_ban.str.startswith("2b_")]
    a_interp = a[a.interpretable]
    b_interp = b[b.interpretable]
    print("2a (covariate shift) co ket qua interpretable: %d/%d" % (len(a_interp), len(a)))
    if len(a_interp):
        khong_bao_drift = (a_interp.verdict != "conditional_change_implicated").all()
        print("  trong do, khong co ca nao bao conditional_change_implicated: %s" % khong_bao_drift)
    print("2b (concept drift that) co ket qua interpretable: %d/%d" % (len(b_interp), len(b)))
    if len(b_interp):
        bao_drift_dung = (b_interp[b_interp.kich_ban != "2b_conceptdrift_shift0"].verdict == "conditional_change_implicated").all()
        print("  trong do, cac muc co bom drift (shift>0) deu bao dung: %s" % bao_drift_dung)

    print()
    print(out[["kich_ban", "verdict", "conditional_gap_clipped", "ess_fraction_clipped", "interpretable"]].to_string(index=False))


if __name__ == "__main__":
    main()
