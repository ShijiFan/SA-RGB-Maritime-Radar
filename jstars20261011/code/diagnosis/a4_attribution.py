"""A4: attribute LORO failures to (i) unseen target position, (ii) val->test clutter drift, (iii) low SCR.

Fold-level factors per test record:
  novel_pos      target cell absent from the fold's training records
  drift_vt       mean over features of |median_test - median_val| / IQR_train on pure-clutter cells (raw features)
  drift_tt       same, test vs training records
  scr_db         median over frames of 20*log10(target RAA / median pure-clutter RAA) -- a SCR proxy
Frame-level: hit (target cell >= tau at alpha=1e-2) versus per-frame SCR proxy, for detectors (A3) and for
position-free per-cell HGB-G7 scorers (frame-normalized, with and without +-2-cell context).
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from a2_subset_np import with_context
from common import FEATURES, OUT, RECORDS, calibrate, frame_normalize, load, loro_split, masks

ALPHA = 1e-2


def fold_factors(data):
    rows = []
    for k in RECORDS:
        train, val, test = loro_split(k)
        pure = lambda g: data[g]["X"][:, :, masks(data[g]["t"])[1], :].reshape(-1, 7)
        Ctr = np.concatenate([pure(g) for g in train]); Cv, Ct = pure(val), pure(test)
        iqr = np.subtract(*np.percentile(Ctr, [75, 25], axis=0))
        Xt = data[test]["X"]; t = data[test]["t"]
        scr = 20 * np.log10(Xt[:, :, t, 1] / np.median(Xt[:, :, masks(t)[1], 1], axis=-1))
        rows.append(dict(fold=k, target_cell=t + 1,
                         novel_pos=int(all(data[g]["t"] != t for g in train)),
                         drift_vt=float(np.mean(np.abs(np.median(Ct, 0) - np.median(Cv, 0)) / iqr)),
                         drift_tt=float(np.mean(np.abs(np.median(Ct, 0) - np.median(Ctr, 0)) / iqr)),
                         **{f"drift_vt_{f}": float(abs(np.median(Ct[:, j]) - np.median(Cv[:, j])) / iqr[j])
                            for j, f in enumerate(FEATURES)},
                         scr_db=float(np.median(scr))))
    return pd.DataFrame(rows)


def percell_scores(data, ctx):
    """HGB-G7 on frame-normalized features; returns {fold: (P,F,14) test scores, tau}."""
    out = {}
    Z = {g: frame_normalize(data[g]["X"]) for g in RECORDS}
    prep = (lambda A: with_context(A)) if ctx else (lambda A: A)
    for k in RECORDS:
        train, val, test = loro_split(k)
        Xtr = np.concatenate([prep(Z[g]).reshape(-1, 7 * (5 if ctx else 1)) for g in train])
        ytr = np.concatenate([np.broadcast_to(np.arange(14) == data[g]["t"], Z[g].shape[:3]).ravel() for g in train])
        sc = StandardScaler().fit(Xtr)
        m = HistGradientBoostingClassifier(max_iter=200, class_weight="balanced", random_state=0).fit(sc.transform(Xtr), ytr)
        s = lambda g: m.predict_proba(sc.transform(prep(Z[g]).reshape(-1, Xtr.shape[1])))[:, 1].reshape(Z[g].shape[:3])
        sv = s(val)
        out[k] = (s(test), calibrate(sv[:, :, masks(data[val]["t"])[0]], ALPHA))
    return out


def main():
    data = load()
    F = fold_factors(data)
    POLS = ["HH", "HV", "VH", "VV"]
    frames, fold_out = [], []
    # position-free per-cell scorers
    for ctx in (False, True):
        name = f"HGB-G7{'+ctx' if ctx else ''}"
        for k, (S, tau) in percell_scores(data, ctx).items():
            t = data[k]["t"]; X = data[k]["X"]
            scr = 20 * np.log10(X[:, :, t, 1] / np.median(X[:, :, masks(t)[1], 1], axis=-1))
            hit = S[:, :, t] >= tau
            h0 = S[:, :, masks(t)[0]]
            fold_out.append(dict(method=name, fold=k, pd=float(hit.mean()), pfa=float(np.mean(h0 >= tau))))
            for pi in range(4):
                frames.append(pd.DataFrame(dict(method=name, fold=k, pol=POLS[pi], frame=np.arange(S.shape[1]),
                                                scr_db=scr[pi], hit=hit[pi])))
    # historical detectors (A3), aligned to cache frames by (pol, frame number)
    try:
        A3 = pd.read_csv(OUT / "A3_detector_pdpfa.csv")
        Z = np.load(OUT / "A3_detector_cell_scores.npz")
        for _, r in A3.iterrows():
            g = r.method.split()[-1]; k = int(r.fold); key = f"{g}_f{k:02d}"
            St, tt, pol, fr = Z[key + "_test"], Z[key + "_testt"], Z[key + "_pol"], Z[key + "_frame"]
            t = data[k]["t"]; X = data[k]["X"]
            pidx = np.array([POLS.index(p) for p in pol]); fidx = fr - fr.min()
            scr = 20 * np.log10(X[pidx, fidx, t, 1] / np.median(X[pidx, fidx][:, masks(t)[1], 1], axis=-1))
            hit = St[np.arange(len(tt)), tt] >= r["tau_0.01"]
            fold_out.append(dict(method=f"YOLO11n SA-RGB {g}", fold=k, pd=r["pd_0.01"], pfa=r["pfa_0.01"]))
            frames.append(pd.DataFrame(dict(method=f"YOLO11n SA-RGB {g}", fold=k, pol=pol, frame=fidx, scr_db=scr, hit=hit)))
    except FileNotFoundError:
        print("A3 outputs missing; detectors skipped")
    FO = pd.DataFrame(fold_out).merge(F, on="fold")
    FR = pd.concat(frames)
    F.to_csv(OUT / "A4_fold_factors.csv", index=False)
    FO.to_csv(OUT / "A4_fold_outcomes.csv", index=False)
    FR.to_csv(OUT / "A4_frame_hits.csv", index=False)

    pd.set_option("display.width", 250); pd.set_option("display.precision", 3)
    print(F[["fold", "target_cell", "novel_pos", "drift_vt", "drift_tt", "scr_db"]].to_string(index=False))
    print(FO.pivot_table(index="fold", columns="method", values=["pd", "pfa"]).to_string())
    print("\nSpearman over 10 folds (novel folds 1,6 excluded for pd):")
    for m, d in FO.groupby("method"):
        dn = d[d.novel_pos == 0]
        r1 = spearmanr(dn.scr_db, dn.pd); r2 = spearmanr(d.drift_vt, np.log10(np.clip(d.pfa, 1e-4, 1) / ALPHA))
        print(f"{m:22s} pd~scr rho={r1.statistic:+.2f} p={r1.pvalue:.3f} | log(pfa/a)~drift_vt rho={r2.statistic:+.2f} p={r2.pvalue:.3f}")
    FR["scr_bin"] = pd.cut(FR.scr_db, [-60, 0, 3, 6, 10, 15, 60])
    print("\nPd by per-frame SCR proxy (dB), novel folds excluded:")
    print(FR[~FR.fold.isin([1, 6])].pivot_table(index="scr_bin", columns="method", values="hit", aggfunc="mean",
                                                 observed=True).to_string())
    print(FR[~FR.fold.isin([1, 6])].groupby("scr_bin", observed=True).size().to_string())


if __name__ == "__main__":
    main()
