"""Recompute first-design IPIX1993 LORO rows (Table VI) from stored scores.

Run from the unzipped supplementary folder:  python verify_scores.py
Needs only numpy (and pandas for printing). Paths are relative to this file.

For every arm, fold and seed it reads results/phaseC_loro/<arm>_f<kk>_s<seed>_scores.npz (validation and test
logits, shape sequences x frames x 14 cells, plus the target cell index) and the per-run JSON (calibrated tail
scale c), applies the clutter-map exponential-tail CFAR and the fixed validation threshold, and checks the
result against the per-run JSON written during the original run.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RUNS = HERE / "results" / "phaseC_loro"
ARMS = ("G7", "G3", "D7", "P7", "G7-noSCN", "G7-absC")
ALPHA = 1e-2
C = np.arange(14)


def tail_params(S, W=64):
    """Copy of rgcs.tail_params: per-frame (u, beta) from the previous W frames, highest cell of each frame omitted."""
    ref = np.where(C[None, None] == S.argmax(-1)[..., None], np.nan, S)
    u = np.full(S.shape[:2], np.nan); b = np.full(S.shape[:2], np.nan)
    for f in range(1, S.shape[1]):
        R = ref[:, max(0, f - W):f].reshape(S.shape[0], -1)
        uu = np.nanquantile(R, 0.9, axis=1)
        u[:, f] = uu
        b[:, f] = np.nanmean(np.where(R > uu[:, None], R - uu[:, None], np.nan), axis=1)
    return u, b


def calibrate(S_h0, a):
    """Copy of common.calibrate: smallest threshold whose H0 activation rate does not exceed a."""
    v = np.sort(S_h0.ravel())[::-1]
    k = int(np.floor(a * v.size))
    return np.nextafter(v[k], np.inf) if k < v.size else -np.inf


def rates(S, tau, t):
    ok = np.isfinite(tau)
    act = S >= tau[..., None]
    h0 = C != t
    adj = np.abs(C - t) == 1
    return dict(pd=act[..., t][ok].mean(), pfa=act[..., h0][ok].mean(), pfa_adj=act[..., adj][ok].mean())


rows, worst = [], 0.0
for arm in ARMS:
    for k in range(1, 11):
        for s in (0, 1, 2):
            z = np.load(RUNS / f"{arm}_f{k:02d}_s{s}_scores.npz")
            js = json.loads((RUNS / f"{arm}_f{k:02d}_s{s}.json").read_text())
            Sv, St, tv, tt = z["val"], z["test"], int(z["val_t"]), int(z["test_t"])
            u, b = tail_params(St)
            r_tail = rates(St, u + b * np.log(0.1 / (js["texp_c"] * ALPHA)), tt)
            r_fix = rates(St, np.full(St.shape[:2], calibrate(Sv[:, :, C != tv], ALPHA)), tt)
            for key, r in (("texp64", r_tail), ("fixed", r_fix)):
                ref = js[f"{key}_{ALPHA}"]
                worst = max(worst, abs(r["pd"] - ref["pd"]), abs(r["pfa"] - ref["pfa"]))
                rows.append(dict(arm=arm if key == "texp64" else arm + " fixed", fold=k, seed=s, **r))

df = pd.DataFrame(rows)
df["over"] = df.pfa > 2 * ALPHA
df["terr"] = np.abs(np.log10(np.maximum(df.pfa, 1e-5) / ALPHA))
tab = df.groupby("arm", sort=False).agg(Pd=("pd", "mean"), Pfa_over_alpha=("pfa", lambda x: x.mean() / ALPHA),
                                        runs_over_2a=("over", "sum"), transfer_err=("terr", "mean"), adj_pfa=("pfa_adj", "mean"))
print(tab.round(3).to_string())
print(f"\nmax |difference| to the per-run JSON written during training: {worst:.2e}")
print("Note: the fixed rule is evaluated on all 1024 frames here, as in the per-run JSON; the manuscript compares rules on the common 1020 frames.")


# ---------------------------------------------------------------- final detector (manuscript Table III)
def calibrate_c(S, t, u, b, a=1e-2):
    """Copy of rgcs.calibrate_c: largest c (log bisection) with validation H0 rate <= a."""
    h0 = C[None, None] != np.asarray(t)[:, None, None]
    lo, hi = 1e-3, 1e3
    for _ in range(40):
        mid = np.sqrt(lo * hi)
        tau = u + b * np.log(0.1 / (mid * a)); ok = np.isfinite(tau)[..., None] & h0
        lo, hi = (mid, hi) if ((S >= tau[..., None]) & h0)[ok].mean() <= a else (lo, mid)
    return lo


def tail_eval(Sv, tv, St, tt, a):
    tv, tt = np.broadcast_to(tv, Sv.shape[:1]), np.broadcast_to(tt, St.shape[:1])
    uv, bv = tail_params(Sv); ut, bt = tail_params(St); c = calibrate_c(Sv, tv, uv, bv)
    tau = ut + bt * np.log(0.1 / (c * a)); ok = np.isfinite(tau)
    h0 = C[None, None] != tt[:, None, None]
    h1 = np.take_along_axis(St, tt[:, None, None].repeat(St.shape[1], 1), 2)[..., 0]
    return float((h1 >= tau)[ok].mean()), float(((St >= tau[..., None]) & h0)[ok[..., None] & h0].mean())


FIN = HERE / "results" / "final_detector" / "scores"
rows = []
for arm in ("G7", "G3"):
    for K in (1, 4):
        for k in range(1, 11):
            for s in (0, 1, 2):
                z = np.load(FIN / f"{arm}_K{K}_f{k:02d}_s{s}_scores.npz")
                for a in (1e-2, 1e-3):
                    p_d, p_fa = tail_eval(z["val"], int(z["val_t"]), z["test"], int(z["test_t"]), a)
                    rows.append(dict(arm=arm, K=K, protocol="IPIX1993", fold=k, seed=s, alpha=a, pd=p_d, pfa=p_fa))
        for s in (0, 1, 2):
            z = np.load(FIN / f"{arm}_K{K}_1998_s{s}_scores.npz")
            p_d, p_fa = tail_eval(z["val"], z["val_t"], z["test"], z["test_t"], 1e-2)
            rows.append(dict(arm=arm, K=K, protocol="IPIX1998", fold=0, seed=s, alpha=1e-2, pd=p_d, pfa=p_fa))
f = pd.DataFrame(rows)
f["over"] = f.pfa > 2 * f.alpha
t5 = f.groupby(["arm", "K", "protocol", "alpha"]).agg(Pd=("pd", "mean"), Pfa=("pfa", "mean"), runs_over_2a=("over", "sum"))
t5["Pfa_over_alpha"] = t5.Pfa / t5.index.get_level_values("alpha")
print("\nFinal detector, tail CFAR (manuscript Table III):")
print(t5.drop(columns="Pfa").round(3).to_string())
log = pd.read_csv(HERE / "results" / "final_detector" / "N3_final.csv")
log = log[log.rule == "texp"].assign(protocol=lambda d: d.protocol.map({"loro": "IPIX1993", "1998": "IPIX1998"}))
m = f.merge(log, on=["arm", "K", "protocol", "fold", "seed", "alpha"], suffixes=("", "_log"), validate="one_to_one")
dev = float(max((m.pd - m.pd_log).abs().max(), (m.pfa - m.pfa_log).abs().max()))
print(f"\nfinal detector: {len(m)} runs compared with the logged N3_final.csv, max |difference| {dev:.2e}")
if (not np.isfinite(m[["pd", "pfa", "pd_log", "pfa_log"]].to_numpy()).all()
        or not np.isfinite(worst) or not np.isfinite(dev)
        or worst > 1e-9 or dev > 1e-9 or len(m) != len(f) or len(f) != 252):
    raise SystemExit("verification FAILED")
print("verification passed")
