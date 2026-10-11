"""Nested design selection: would the Phase B choices have been made without the outer test record?

For every outer test record k, the three Phase B decisions are re-run by inner LORO over the other nine records
(inner test r, inner validation = next of the nine in cyclic order, training = remaining seven), with the HGB-G7
scorer used in Phase B. Selection rules, fixed before running:
  D1 normalization  {raw, frame (B1), guarded CUT (B2)}          -> highest mean inner oracle Pd at 1e-2
  D2 context on B1  {none, absolute (C1), relative (C2)}          -> highest mean inner oracle Pd at 1e-2
  D3 threshold on C2{fixed, empirical clutter map, exponential tail} -> lowest mean inner |log10(Pfa/alpha)| at 1e-2,
                                                                       ties broken by the same at 1e-3
The selected HGB pipeline is then evaluated once on the outer test record (nested estimate).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from b_dev_modules import cn_cut, cn_frame, neighbours
from b_tail import bisect, cmap_tau, rate, tail_params, texp_tau
from common import OUT, RECORDS, calibrate, load, masks, pd_at_pfa

C = np.arange(14)


def feats(cfg, X):
    if cfg == "raw":
        return X
    if cfg == "cut":
        return cn_cut(X)
    Z = cn_frame(X)
    if cfg in ("frame", "none"):
        return Z
    Nb = neighbours(Z)
    if cfg == "abs":
        return np.concatenate([Z, Nb.reshape(*Z.shape[:3], -1)], -1)
    return np.concatenate([Z, (Z[..., None] - Nb).reshape(*Z.shape[:3], -1), Z - np.nanmax(Nb, -1)], -1)  # rel


def fit_score(Z, data, train, others):
    Xtr = np.concatenate([Z[g].reshape(-1, Z[g].shape[-1]) for g in train])
    ytr = np.concatenate([np.broadcast_to(C == data[g]["t"], Z[g].shape[:3]).ravel() for g in train])
    m = HistGradientBoostingClassifier(max_iter=200, class_weight="balanced", random_state=0).fit(Xtr, ytr)
    out = {}
    for g in others:
        p = np.clip(m.predict_proba(Z[g].reshape(-1, Z[g].shape[-1]))[:, 1], 1e-12, 1 - 1e-12)
        out[g] = np.log(p / (1 - p)).reshape(Z[g].shape[:3])
    return out


def rules(sv, tv, st, tt):
    """Realized test Pfa and Pd at 1e-2 and 1e-3 for the three threshold rules."""
    res = {}
    uv, bv = tail_params(sv, 64); ut, bt = tail_params(st, 64)
    c = bisect(lambda x: rate(sv, texp_tau(uv, bv, 1e-2, x), tv), 1e-3, 1e3, 1e-2, log=True)
    for a in (1e-2, 1e-3):
        q = 1 - bisect(lambda x: rate(sv, cmap_tau(sv, 64, 1 - x), tv), 1e-6, 0.5, a, n=25, log=True)
        taus = {"fixed": np.full(st.shape[:2], calibrate(sv[:, :, masks(tv)[0]], a)), "cmap": cmap_tau(st, 64, q),
                "texp": texp_tau(ut, bt, a, c)}
        for r, tau in taus.items():
            ok = np.isfinite(tau)
            res[(r, a)] = (float((st[:, :, masks(tt)[0]] >= tau[..., None])[ok].mean()), float((st[:, :, tt] >= tau)[ok].mean()))
    return res


def logerr(p, a):
    return abs(np.log10(max(p, 1e-5) / a))


def main():
    data = load()
    Zc = {cfg: {g: feats(cfg, data[g]["X"]) for g in RECORDS} for cfg in ("raw", "frame", "cut", "abs", "rel")}
    rows, sel = [], []
    for k in RECORDS:
        inner = [r for r in RECORDS if r != k]
        stats = {}
        for i, r in enumerate(inner):
            v = inner[(i + 1) % 9]; tr = [g for g in inner if g not in (r, v)]
            for cfg in ("raw", "frame", "cut", "abs", "rel"):
                S = fit_score(Zc[cfg], data, tr, (v, r))
                st, tt = S[r], data[r]["t"]
                stats.setdefault(cfg, []).append(pd_at_pfa(st[:, :, tt].ravel(), st[:, :, masks(tt)[0]].ravel(), 1e-2))
                if cfg == "rel":
                    res = rules(S[v], data[v]["t"], st, tt)
                    for (rule, a), (pfa, _) in res.items():
                        stats.setdefault((rule, a), []).append(logerr(pfa, a))
        m = {kk: float(np.mean(vv)) for kk, vv in stats.items()}
        d1 = max(("raw", "frame", "cut"), key=lambda c: m[c])
        d2 = max(("frame", "abs", "rel"), key=lambda c: m[c])
        d3 = min(("fixed", "cmap", "texp"), key=lambda r: (round(m[(r, 1e-2)], 6), m[(r, 1e-3)]))
        sel.append(dict(outer=k, D1=d1, D2={"frame": "none"}.get(d2, d2), D3=d3,
                        **{f"inner_pdor_{c}": m[c] for c in ("raw", "frame", "cut", "abs", "rel")},
                        **{f"inner_logerr_{r}_{a:g}": m[(r, a)] for r in ("fixed", "cmap", "texp") for a in (1e-2, 1e-3)}))
        # nested outer estimate with the selected pipeline (normalization fixed by D1/D2 as one feature set)
        cfg = d2 if d1 == "frame" else d1  # context variants were defined on the frame-normalized base
        val = 9 if k == 10 else k + 1
        tr = [g for g in RECORDS if g not in (k, val)]
        S = fit_score(Zc[cfg], data, tr, (val, k))
        res = rules(S[val], data[val]["t"], S[k], data[k]["t"])
        for a in (1e-2, 1e-3):
            pfa, pdv = res[(d3, a)]
            rows.append(dict(outer=k, cfg=cfg, rule=d3, alpha=a, pfa=pfa, pd=pdv))
        print(sel[-1]["outer"], d1, d2, d3, cfg, flush=True)
    pd.DataFrame(sel).to_csv(OUT / "B_nested_selection.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "B_nested_outer.csv", index=False)
    s = pd.DataFrame(sel)
    print(s[["outer", "D1", "D2", "D3"]].to_string(index=False))
    o = pd.DataFrame(rows)
    for a in (1e-2, 1e-3):
        d = o[o.alpha == a]
        print(a, "nested HGB: Pd", round(d.pd.mean(), 3), "Pfa/a", round(d.pfa.mean() / a, 2), ">2a", int((d.pfa > 2 * a).sum()))


if __name__ == "__main__":
    main()
