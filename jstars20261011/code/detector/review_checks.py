"""Checks requested by the pre-submission review (2026-10-09); stored scores only, no retraining.

R1 common valid-frame mask for fixed vs tail rule (first frame of every sequence excluded for both)
R2 sensitivity of the tail rule to the history length W and to trimming the top-1 / top-2 cells per frame
   (scale c re-calibrated on the validation record for every setting; IPIX1993 LORO only)
R3 background-cell activation rate (cells at range distance > 1 from the target) besides the all-H0 rate
R4 warm-up frames (history < 64) versus steady-state frames
R5 tail calibration: realized test H0 rate versus nominal alpha in {1e-1, 3e-2, 1e-2, 3e-3, 1e-3} (c set at 1e-2)
R6 IPIX1998 micro (pooled windows) versus macro (record-averaged) Pd and Pfa
R7 parameter count and per-cell input width of every arm
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "phaseA_20261008"))
from common import RECORDS, calibrate, loro_split  # noqa: E402

C = np.arange(14)
RUNS = HERE / "runs_loro"


def tail(S, W=64, trim=1):
    """Per-frame (u, beta) from the previous W frames after removing the `trim` highest cells of each frame."""
    order = np.argsort(-S, axis=-1)
    ref = S.copy()
    for k in range(trim):
        np.put_along_axis(ref, order[..., k:k + 1], np.nan, axis=-1)
    u = np.full(S.shape[:2], np.nan); b = np.full(S.shape[:2], np.nan)
    for f in range(1, S.shape[1]):
        R = ref[:, max(0, f - W):f].reshape(S.shape[0], -1)
        uu = np.nanquantile(R, 0.9, axis=1); u[:, f] = uu
        b[:, f] = np.nanmean(np.where(R > uu[:, None], R - uu[:, None], np.nan), axis=1)
    return u, b


def tau_of(u, b, a, c):
    return u + b * np.log(0.1 / (c * a))


def h0rate(S, tau, t, mask=None):
    m = (C[None, None] != t) if mask is None else mask
    ok = np.isfinite(tau)[..., None] & m
    return ((S >= tau[..., None]) & ok).sum() / ok.sum()


def calib_c(Sv, tv, u, b, a=1e-2):
    lo, hi = 1e-3, 1e3
    for _ in range(40):
        mid = np.sqrt(lo * hi)
        lo, hi = (mid, hi) if h0rate(Sv, tau_of(u, b, a, mid), tv) <= a else (lo, mid)
    return lo


def main():
    out = {}
    rows2, rows1, rows3, rows5 = [], [], [], []
    for arm in ("G7", "G3"):
        for k in RECORDS:
            for s in (0, 1, 2):
                z = np.load(RUNS / f"{arm}_f{k:02d}_s{s}_scores.npz")
                Sv, St, tv, tt = z["val"], z["test"], int(z["val_t"]), int(z["test_t"])
                adj = np.abs(C - tt) == 1; bg = np.abs(C - tt) > 1; h0 = C != tt
                for W, trim in ((64, 1), (32, 1), (16, 1), (64, 2), (32, 2)):
                    if arm == "G3" and (W, trim) != (64, 1):
                        continue
                    uv, bv = tail(Sv, W, trim); ut, bt = tail(St, W, trim); c = calib_c(Sv, tv, uv, bv)
                    for a in (1e-2, 1e-3):
                        tau = tau_of(ut, bt, a, c); ok = np.isfinite(tau)
                        rows2.append(dict(arm=arm, fold=k, seed=s, W=W, trim=trim, alpha=a,
                                          pfa=h0rate(St, tau, tt), pd=float((St[:, :, tt] >= tau)[ok].mean())))
                    if (W, trim) == (64, 1):
                        tau = tau_of(ut, bt, 1e-2, c); ok = np.isfinite(tau)
                        fixed = np.full(St.shape[:2], calibrate(Sv[:, :, C != tv], 1e-2)); fixed[~ok] = np.nan
                        frames = np.arange(St.shape[1])[None].repeat(St.shape[0], 0)
                        for rule, T in (("tail", tau), ("fixed", fixed)):
                            okr = np.isfinite(T)
                            rows1.append(dict(arm=arm, fold=k, seed=s, rule=rule, n_frames=int(okr.sum()),
                                              n_h0=int(okr.sum() * 13), pfa=h0rate(St, T, tt),
                                              pfa_bg=h0rate(St, T, tt, mask=bg[None, None] & okr[..., None]),
                                              pfa_adj=h0rate(St, T, tt, mask=adj[None, None] & okr[..., None]),
                                              pd=float((St[:, :, tt] >= T)[okr].mean())))
                        warm = np.isfinite(tau) & (frames < 64); steady = np.isfinite(tau) & (frames >= 64)
                        rows3.append(dict(arm=arm, fold=k, seed=s,
                                          pfa_warm=h0rate(St, np.where(warm, tau, np.nan), tt),
                                          pfa_steady=h0rate(St, np.where(steady, tau, np.nan), tt)))
                        for a in (1e-1, 3e-2, 1e-2, 3e-3, 1e-3):
                            rows5.append(dict(arm=arm, fold=k, seed=s, alpha=a, pfa=h0rate(St, tau_of(ut, bt, a, c), tt)))
    R1, R2, R3, R5 = map(pd.DataFrame, (rows1, rows2, rows3, rows5))
    for df, n in ((R1, "R1_common_mask"), (R2, "R2_W_trim"), (R3, "R3_warmup"), (R5, "R5_tail_calibration")):
        df.to_csv(HERE / f"C_review_{n}.csv", index=False)
    pd.set_option("display.width", 220); pd.set_option("display.precision", 4)
    le = lambda x, a: float(np.mean(np.abs(np.log10(np.clip(x, 1e-5, 1) / a))))
    f = R1.groupby(["arm", "rule", "fold"]).agg(pfa=("pfa", "mean"), pfa_bg=("pfa_bg", "mean"), pd=("pd", "mean")).reset_index()
    print("R1 common mask (fold means):")
    for (arm, rule), g in f.groupby(["arm", "rule"]):
        print(f"  {arm} {rule}: Pfa/a {g.pfa.mean() / 1e-2:.2f}  logerr {le(g.pfa, 1e-2):.3f}  bg-Pfa/a {g.pfa_bg.mean() / 1e-2:.2f}  "
              f"Pd {g.pd.mean():.3f}  runs>2a {int((R1[(R1.arm == arm) & (R1.rule == rule)].pfa > 0.02).sum())}/30  "
              f"frames {R1[(R1.arm == arm) & (R1.rule == rule)].n_frames.iloc[0]}")
    from scipy.stats import wilcoxon
    a_ = f[(f.arm == "G7") & (f.rule == "tail")].set_index("fold").pfa; b_ = f[(f.arm == "G7") & (f.rule == "fixed")].set_index("fold").pfa
    ea = np.abs(np.log10(np.clip(a_, 1e-5, 1) / 1e-2)); eb = np.abs(np.log10(np.clip(b_, 1e-5, 1) / 1e-2))
    print("  G7 tail vs fixed logerr lower in", int((ea < eb).sum()), "folds; p =", wilcoxon(ea, eb).pvalue)
    print("R2 W/trim sensitivity (G7, 30 runs):")
    print(R2[R2.arm == "G7"].groupby(["W", "trim", "alpha"]).agg(pd=("pd", "mean"), pfa_a=("pfa", lambda x: x.mean()),
                                                                  over2=("pfa", lambda x: None)).drop(columns="over2")
          .assign(pfa_a=lambda d: d.pfa_a / d.index.get_level_values("alpha")).to_string())
    for (W, trim, a), g in R2[R2.arm == "G7"].groupby(["W", "trim", "alpha"]):
        print(f"   W={W} trim={trim} a={a:g}: runs>2a {int((g.pfa > 2 * a).sum())}/30")
    print("R4 warm-up vs steady Pfa/a (G7):", round(R3[R3.arm == "G7"].pfa_warm.mean() / 1e-2, 2),
          round(R3[R3.arm == "G7"].pfa_steady.mean() / 1e-2, 2))
    print("R5 tail calibration (realized/nominal, G7):")
    print(R5[R5.arm == "G7"].groupby("alpha").pfa.mean().pipe(lambda s: s / s.index).round(2).to_string())
    # R6 IPIX1998 micro vs macro
    R = json.loads((HERE / "runs_1998" / "summary.json").read_text())
    for arm in ("G7", "G3"):
        rr = [r for r in R if r["arm"] == arm]
        micro = np.mean([r["0.01"]["pd"] for r in rr]); macro = np.mean([np.mean(list(r["0.01"]["pd_by_record"].values())) for r in rr])
        pmi = np.mean([r["0.01"]["pfa"] for r in rr]); pma = np.mean([np.mean(list(r["0.01"]["pfa_by_record"].values())) for r in rr])
        print(f"R6 1998 {arm}: Pd micro {micro:.3f} macro {macro:.3f}; Pfa/a micro {pmi / 1e-2:.2f} macro {pma / 1e-2:.2f}")
        out[f"1998_{arm}"] = dict(pd_micro=micro, pd_macro=macro, pfa_micro=pmi, pfa_macro=pma)
    # R7 parameters and input width
    import torch  # noqa: F401
    from explore_context import build
    from run_loro import build_model
    for arm in ("G7", "G3", "D7", "P7", "G7-noSCN", "G7-absC"):
        m = build_model(arm)
        print(f"R7 {arm}: params {sum(p.numel() for p in m.parameters())}, head input {m.head[0].in_channels}")
    m = build("margin1", "G7")
    print(f"R7 G7-peak-margin: params {sum(p.numel() for p in m.parameters())}, head input {m.head[0].in_channels}")
    (HERE / "C_review_checks.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
