"""A6: cross-site 1993 -> 1998 transfer of shared per-cell scorers on genuine sliding windows.

Train on the chronological IPIX1993 training frames (first 181 of 256 per sequence, all 10 records), calibrate
tau on the 1993 validation frames (next 45), test on every IPIX1998 14-cell window containing the target with
window-local reference features (no shift, no padding). Also report the full-record-reference variant and the
per-feature 1998-vs-1993 clutter drift.
"""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from a2_subset_np import GROUPS, with_context
from common import FEATURES, OUT, POLS, RECORDS, auc, calibrate, frame_normalize, load, masks, pd_at_pfa

ALPHAS = (1e-2, 1e-3)
TR, VA = 181, 45


def load_1998():
    d = np.load(OUT / "ipix1998_7feat.npz")
    meta = json.load(open(OUT / "ipix1998_7feat.json", encoding="utf8"))
    wins = []  # list of (rec, pol, Z (frames,14,7), t_in_window)
    full = {}
    for r in meta["records"]:
        t = r["target_cell"] - 1
        for p in POLS:
            k = f"r{r['rec']:02d}_{p}"
            W, starts = d[k + "_win"].astype(np.float64), d[k + "_starts"]
            for j, s in enumerate(starts):
                wins.append((r["rec"], p, W[:, j], t - s, s))
            full[(r["rec"], p)] = (d[k + "_full"].astype(np.float64), t)
    return wins, full


def main():
    data = load()
    wins, full = load_1998()
    rows, drift = [], []
    for rep in ("raw", "framenorm"):
        f = (lambda Z: Z) if rep == "raw" else frame_normalize
        X93 = {g: f(data[g]["X"]) for g in RECORDS}
        W98 = [(r, p, f(Z), tp, s) for r, p, Z, tp, s in wins]
        # clutter drift 1998 vs 1993 (pure clutter cells)
        for j, name in enumerate(FEATURES):
            c93 = np.concatenate([X93[g][:, :, masks(data[g]["t"])[1], j].ravel() for g in RECORDS])
            c98 = np.concatenate([Z[:, masks(tp)[1], j].ravel() for _, _, Z, tp, _ in W98])
            h93 = np.concatenate([X93[g][:, :, data[g]["t"], j].ravel() for g in RECORDS])
            h98 = np.concatenate([Z[:, tp, j] for _, _, Z, tp, _ in W98])
            iqr = np.subtract(*np.percentile(c93, [75, 25]))
            drift.append(dict(rep=rep, feature=name, shift_in_iqr93=(np.median(c98) - np.median(c93)) / iqr,
                              iqr_ratio=np.subtract(*np.percentile(c98, [75, 25])) / iqr,
                              auc93=auc(h93, c93), auc98=auc(h98, c98)))
        for ctx in (False, True):
            for gname, feats in GROUPS.items():
                cols = [FEATURES.index(x) for x in feats]
                prep = (lambda Z: with_context(Z[..., cols])) if ctx else (lambda Z: Z[..., cols])
                Ztr = [prep(X93[g][:, :TR]) for g in RECORDS]
                Xtr = np.concatenate([z.reshape(-1, z.shape[-1]) for z in Ztr])
                ytr = np.concatenate([(np.arange(14) == data[g]["t"])[None, None].repeat(4, 0).repeat(TR, 1).ravel()
                                      for g in RECORDS])
                sc = StandardScaler().fit(Xtr)
                for mname, mk in (("LR", lambda: LogisticRegression(class_weight="balanced", max_iter=2000)),
                                  ("HGB", lambda: HistGradientBoostingClassifier(max_iter=200, class_weight="balanced",
                                                                                 random_state=0))):
                    m = mk().fit(sc.transform(Xtr), ytr)
                    sfun = lambda Z: m.predict_proba(sc.transform(Z.reshape(-1, Z.shape[-1])))[:, 1].reshape(Z.shape[:-1])
                    # calibrate on 1993 validation frames
                    v0 = np.concatenate([sfun(prep(X93[g][:, TR:TR + VA]))[..., masks(data[g]["t"])[0]].ravel()
                                         for g in RECORDS])
                    taus = {a: calibrate(v0, a) for a in ALPHAS}
                    S = [(r, tp, sfun(prep(Z[None]))[0]) for r, p, Z, tp, s in W98]  # (frames,14)
                    h1 = np.concatenate([s[:, tp] for _, tp, s in S])
                    h0 = np.concatenate([s[:, masks(tp)[0]].ravel() for _, tp, s in S])
                    pos = np.concatenate([np.full(s.shape[0], tp) for _, tp, s in S])
                    exact = np.concatenate([s.argmax(1) == tp for _, tp, s in S])
                    row = dict(rep=rep, ctx=ctx, model=mname, group=gname, exact_cell=exact.mean(), auc=auc(h1, h0))
                    for a, tau in taus.items():
                        row[f"pfa_{a:g}"] = float(np.mean(h0 >= tau)); row[f"pd_{a:g}"] = float(np.mean(h1 >= tau))
                        row[f"pdor_{a:g}"] = pd_at_pfa(h1, h0, a)
                    # oracle Pd at 1e-2 by window position of the target
                    tau_o = calibrate(h0, 1e-2)
                    for q in range(14):
                        if (pos == q).any():
                            row[f"pdor_pos{q}"] = float(np.mean(h1[pos == q] >= tau_o))
                    rows.append(row)
                    print(rep, ctx, gname, mname, {k: round(v, 3) for k, v in row.items() if k.startswith(("pfa", "pdor_0"))},
                          flush=True)
    pd.DataFrame(drift).to_csv(OUT / "A6_drift_1998_vs_1993.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "A6_cross_site.csv", index=False)


if __name__ == "__main__":
    main()
