"""Reference detectors under the same protocols (added after the fixed protocol; not part of its decision rules).

CA-CFAR   cell-averaging CFAR on amplitude: RAA is the cell's mean amplitude over the mean of the other 13 cells,
          i.e., the CA-CFAR statistic with a 13-cell reference; score = log RAA.
HGB-G7/G3 histogram gradient boosting on the same per-cell inputs as RadarGrid-CS (per-frame normalization +
          relative context, Phase B configuration C2).
Each with a fixed validation threshold and with the tail CFAR (scale c calibrated on validation at 1e-2).
Protocols: IPIX1993 LORO (10 folds) and IPIX1993 -> IPIX1998 windows (chronological 1993 train/validation).
Also: Pd of the RadarGrid-CS G7/G3 runs on IPIX1998 at an equal realized test Pfa (oracle), from stored scores.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "phaseA_20261008"))
from a2_subset_np import GROUPS  # noqa: E402
from b_dev_round2 import build  # noqa: E402
from common import FEATURES, POLS, RECORDS, calibrate, load, loro_split, pd_at_pfa  # noqa: E402
from rgcs import calibrate_c, tail_params, texp_tau  # noqa: E402

C = np.arange(14)


def evaluate(Sv, tv, St, tt):
    """Sv (nv,F,14) validation scores with targets tv (nv,), St (nt,F,14) test with tt (nt,)."""
    uv, bv = tail_params(Sv); ut, bt = tail_params(St); c = calibrate_c(Sv, tv, uv, bv, 1e-2)
    h0v = Sv[np.broadcast_to(C[None, None] != tv[:, None, None], Sv.shape)]
    h0m = C[None, None] != tt[:, None, None]; adjm = np.abs(C[None, None] - tt[:, None, None]) == 1
    h1 = np.take_along_axis(St, np.broadcast_to(tt[:, None, None], St.shape[:2] + (1,)), 2)[..., 0]
    out = {}
    for a in (1e-2, 1e-3):
        for rule, tau in (("fixed", np.full(St.shape[:2], calibrate(h0v, a))), ("texp", texp_tau(ut, bt, a, c))):
            ok = np.isfinite(tau); T = tau[..., None]; okc = ok[..., None]
            act = St >= T
            out[(rule, a)] = dict(pd=float((h1 >= tau)[ok].mean()),
                                  pfa=float((act & h0m & okc).sum() / (np.broadcast_to(h0m, act.shape) & okc).sum()),
                                  pfa_adj=float((act & adjm & okc).sum() / (np.broadcast_to(adjm, act.shape) & okc).sum()))
        out[("oracle", a)] = dict(pd=pd_at_pfa(h1.ravel(), St[np.broadcast_to(h0m, St.shape)], a))
    return out


def hgb_scores(Z, data, train_slices, others):
    Xtr = np.concatenate([z.reshape(-1, z.shape[-1]) for z, _ in train_slices])
    ytr = np.concatenate([np.broadcast_to(C == t, z.shape[:3]).ravel() for z, t in train_slices])
    m = HistGradientBoostingClassifier(max_iter=200, class_weight="balanced", random_state=0).fit(Xtr, ytr)
    res = []
    for z in others:
        p = np.clip(m.predict_proba(z.reshape(-1, z.shape[-1]))[:, 1], 1e-12, 1 - 1e-12)
        res.append(np.log(p / (1 - p)).reshape(z.shape[:3]))
    return res


def windows_1998():
    d = np.load(HERE.parent / "phaseA_20261008" / "out" / "ipix1998_7feat.npz")
    meta = json.load(open(HERE.parent / "phaseA_20261008" / "out" / "ipix1998_7feat.json", encoding="utf8"))
    X, T = [], []
    for r in meta["records"]:
        for p in POLS:
            k = f"r{r['rec']:02d}_{p}"
            for j, s in enumerate(d[k + "_starts"]):
                X.append(d[k + "_win"][:, j].astype(np.float64)); T.append(r["target_cell"] - 1 - int(s))
    return np.stack(X), np.array(T)  # (n_seq,117,14,7), (n_seq,)


def main():
    data = load(); rows = []
    X98, T98 = windows_1998()
    TR, VA = 181, 45
    # ---------------- CA-CFAR
    raa = FEATURES.index("RAA")
    for k in RECORDS:
        tr, val, test = loro_split(k)
        Sv = np.log(data[val]["X"][..., raa]); St = np.log(data[test]["X"][..., raa])
        for (rule, a), v in evaluate(Sv, np.full(4, data[val]["t"]), St, np.full(4, data[test]["t"])).items():
            rows.append(dict(method="CA-CFAR", protocol="loro", fold=k, rule=rule, alpha=a, **v))
    Sv = np.concatenate([np.log(data[g]["X"][:, TR:TR + VA, :, raa]) for g in RECORDS])
    Tv = np.repeat([data[g]["t"] for g in RECORDS], 4)
    for (rule, a), v in evaluate(Sv, Tv, np.log(X98[..., raa]), T98).items():
        rows.append(dict(method="CA-CFAR", protocol="1998", fold=0, rule=rule, alpha=a, **v))
    print("CA-CFAR done", flush=True)
    # ---------------- HGB on the RadarGrid-CS inputs
    for grp in ("G7", "G3"):
        cols = [FEATURES.index(f) for f in GROUPS[grp]]
        Z = {g: build("C2", data[g]["X"][..., cols]) for g in RECORDS}
        for k in RECORDS:
            tr, val, test = loro_split(k)
            Sv, St = hgb_scores(Z, data, [(Z[g], data[g]["t"]) for g in tr], (Z[val], Z[test]))
            for (rule, a), v in evaluate(Sv, np.full(4, data[val]["t"]), St, np.full(4, data[test]["t"])).items():
                rows.append(dict(method=f"HGB-{grp}", protocol="loro", fold=k, rule=rule, alpha=a, **v))
        Z98 = build("C2", X98[..., cols])
        Sv, St = hgb_scores(Z, data, [(Z[g][:, :TR], data[g]["t"]) for g in RECORDS],
                            (np.concatenate([Z[g][:, TR:TR + VA] for g in RECORDS]), Z98))
        for (rule, a), v in evaluate(Sv, Tv, St, T98).items():
            rows.append(dict(method=f"HGB-{grp}", protocol="1998", fold=0, rule=rule, alpha=a, **v))
        print(grp, "done", flush=True)
    # ---------------- RadarGrid-CS on IPIX1998 at equal realized Pfa
    for arm in ("G7", "G3", "D7"):
        for s in (0, 1, 2):
            z = np.load(HERE / "runs_1998" / f"{arm}_s{s}_scores.npz")
            St, Tt = z["test"], z["t"]
            h1 = St[np.arange(len(Tt)), :, Tt].ravel(); h0 = St[np.broadcast_to(C[None, None] != Tt[:, None, None], St.shape)]
            for a in (1e-2, 1e-3):
                rows.append(dict(method=f"RadarGrid-CS {arm}", protocol="1998", fold=s, rule="oracle", alpha=a,
                                 pd=pd_at_pfa(h1, h0, a)))
    df = pd.DataFrame(rows); df.to_csv(HERE / "C_reference_baselines.csv", index=False)
    pd.set_option("display.width", 200); pd.set_option("display.precision", 4)
    g = df.groupby(["protocol", "method", "rule", "alpha"]).agg(pd=("pd", "mean"), pfa=("pfa", "mean"),
                                                                 adj=("pfa_adj", "mean")).reset_index()
    over = df[df.protocol == "loro"].groupby(["method", "rule", "alpha"]).apply(
        lambda d: int((d.pfa > 2 * d.alpha.iloc[0]).sum()) if d.pfa.notna().any() else -1)
    print(g.to_string(index=False)); print(over.to_string())


if __name__ == "__main__":
    main()
