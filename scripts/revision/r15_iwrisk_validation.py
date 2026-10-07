# -*- coding: utf-8 -*-
"""R1.5: kiem chung phep thay the importance_weighted_risk tren du lieu tong hop
co kiem soat, cung khuon voi phan vi du da co o
results/evidence/item1_conditional_counterexample.parquet (trigger + satellite).

Ba kich ban:
  1. Covariate shift thuan, p(y|x) khong doi theo cau tao (giong phan vi du cu,
     chi doi tu phep kiem conditional_structure_agreement sang importance_weighted_risk).
     Ky vong: verdict KHONG duoc la conditional_change_implicated.
  2. Co bom concept drift that: doi quy tac trigger -> nhan o cua so muon, giu
     p(x) gan nhu khong doi (vi trigger van sinh deu nhu cu).
     Ky vong: verdict PHAI la conditional_change_implicated.
  3. Giam dan do chong lap giua hai cua so bang cach cho tung cua so chi active
     mot tap trigger rieng, tien toi khong giao nhau.
     Ky vong: guard ESS bat dau chan, verdict chuyen thanh undetermined dung luc
     ESS giam duoi nguong, khong phai mot gia tri ngau nhien.
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sift.iwrisk import importance_weighted_risk  # noqa: E402

N_TRIGGERS = 6
N_SATELLITES = 294
N_EARLY = 1000
N_LATE = 300
P_ON = 0.5
P_BG = 0.05
SEED = 20261007


def make_window(n, owners, rng, trigger_pool=None, label_shift=0):
    """trigger_pool: neu khac None, chi sinh trigger trong tap nay (dung cho kich ban 3).
    label_shift: cong them vao nhan sau khi sinh trigger (dung cho kich ban 2, bom drift that)."""
    pool = np.arange(N_TRIGGERS) if trigger_pool is None else np.asarray(trigger_pool)
    trig_idx = rng.integers(0, len(pool), size=n)
    x_trig_class = pool[trig_idx]                      # lop ma trigger the hien, quyet dinh p(x)
    y = (x_trig_class + label_shift) % N_TRIGGERS       # nhan thuc te, co the lech khoi trigger

    triggers = np.zeros((n, N_TRIGGERS), dtype=np.float64)
    triggers[np.arange(n), x_trig_class] = 1.0          # dac trung quan sat duoc la trigger GOC
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


def chay_mot_kich_ban(ten, x_early, y_early, x_late, y_late, seed):
    r = importance_weighted_risk(x_early, y_early, x_late, y_late, seed=seed, metric="macro_f1")
    s = r.summary()
    s["kich_ban"] = ten
    print(
        "%-42s verdict=%-30s conditional_gap=%.4f KTC [%.4f, %.4f]  ESS=%.1f (%.1f%%)  interpretable=%s  reasons=%s"
        % (
            ten,
            s["verdict"],
            s["conditional_gap_clipped"],
            s["gap_ci_clipped"][0],
            s["gap_ci_clipped"][1],
            s["ess_clipped"],
            100 * s["ess_fraction_clipped"],
            s["interpretable"],
            s["reasons"],
        )
    )
    return s


def main():
    ket_qua = []

    # ---------------- Kich ban 1: covariate shift thuan ----------------
    print("=== Kich ban 1: covariate shift thuan (p(y|x) khong doi theo cau tao) ===")
    for frac in (0.5, 1.0):
        rng = np.random.default_rng(SEED)
        owners = rng.integers(0, N_TRIGGERS, size=N_SATELLITES)
        x_e, y_e = make_window(N_EARLY, owners, rng)
        late_owners = reassign(owners, frac, rng)
        x_l, y_l = make_window(N_LATE, late_owners, rng)
        ket_qua.append(chay_mot_kich_ban("1_covariate_shift_frac%.2f" % frac, x_e, y_e, x_l, y_l, SEED))

    # ---------------- Kich ban 2: bom concept drift that ----------------
    print()
    print("=== Kich ban 2: bom concept drift that (doi quy tac trigger -> nhan) ===")
    for shift in (1, 3):
        rng = np.random.default_rng(SEED)
        owners = rng.integers(0, N_TRIGGERS, size=N_SATELLITES)
        x_e, y_e = make_window(N_EARLY, owners, rng, label_shift=0)
        # p(x) giu gan nhu khong doi: cung owners, khong bom covariate shift
        x_l, y_l = make_window(N_LATE, owners, rng, label_shift=shift)
        ket_qua.append(chay_mot_kich_ban("2_concept_drift_shift%d" % shift, x_e, y_e, x_l, y_l, SEED))

    # ---------------- Kich ban 3: giam dan do chong lap ----------------
    print()
    print("=== Kich ban 3: giam do chong lap giua hai cua so (kiem guard ESS) ===")
    for so_trigger_rieng in (6, 4, 2, 1):
        rng = np.random.default_rng(SEED)
        owners = rng.integers(0, N_TRIGGERS, size=N_SATELLITES)
        pool_early = np.arange(0, so_trigger_rieng) if so_trigger_rieng < N_TRIGGERS else None
        pool_late = (
            np.arange(N_TRIGGERS - so_trigger_rieng, N_TRIGGERS) if so_trigger_rieng < N_TRIGGERS else None
        )
        x_e, y_e = make_window(N_EARLY, owners, rng, trigger_pool=pool_early)
        x_l, y_l = make_window(N_LATE, owners, rng, trigger_pool=pool_late)
        ten = "3_overlap_%dtrigger_rieng" % so_trigger_rieng
        try:
            ket_qua.append(chay_mot_kich_ban(ten, x_e, y_e, x_l, y_l, SEED))
        except ValueError as e:
            # chong lap qua it: cua so som chi con mot lop, phep so rui ro khong xac dinh
            print("%-42s verdict=not_defined  reasons=%s" % (ten, e))
            ket_qua.append(dict(kich_ban=ten, verdict="not_defined", interpretable=False, reasons=str(e)))

    out = pd.DataFrame(ket_qua)
    out.to_csv("results/revision/r15_iwrisk_validation.csv", index=False)

    print()
    print("=== Kiem tra tieu chi pass/fail dat truoc ===")
    kb1 = out[out.kich_ban.str.startswith("1_")]
    kb2 = out[out.kich_ban.str.startswith("2_")]
    kb3 = out[out.kich_ban.str.startswith("3_")]

    p1 = bool((kb1.verdict != "conditional_change_implicated").all())
    print("Kich ban 1 (covariate shift), khong duoc bao drift: %s" % ("DAT" if p1 else "KHONG DAT"))

    p2 = bool((kb2.verdict == "conditional_change_implicated").all())
    print("Kich ban 2 (concept drift that), phai bao drift   : %s" % ("DAT" if p2 else "KHONG DAT"))

    # guard: ESS fraction phai giam dan khi so trigger rieng giam (chong lap giam)
    ess_giam_dan = bool((kb3.ess_fraction_clipped.diff().dropna() <= 1e-9).all())
    co_undetermined = bool((kb3.verdict == "undetermined").any())
    print("Kich ban 3, ESS giam dan khi giam chong lap        : %s" % ("DAT" if ess_giam_dan else "KHONG DAT"))
    print("Kich ban 3, co it nhat mot muc bi guard chan (undetermined): %s" % ("DAT" if co_undetermined else "KHONG DAT"))

    print()
    print(out[["kich_ban", "verdict", "conditional_gap_clipped", "ess_fraction_clipped", "interpretable"]].to_string(index=False))


if __name__ == "__main__":
    main()
