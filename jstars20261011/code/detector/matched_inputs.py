"""Input-matched scorer comparison (review item G).

Both scorers receive the identical per-cell tensor of the range-ordered peak-margin design: 8-bit quantized SA-RGB G7,
slot readout, slot clutter normalization, and the range-ordered +-1 peak margin (72 values per cell). The network is the
peak-margin cell head (8,897 parameters, seeds 0-2); the tree scorer is histogram gradient boosting on the same tensor.
Same LORO folds, same IPIX1993 -> IPIX1998 protocol, same tail CFAR, plus Pd at equal realized Pfa (oracle).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "phaseA_20261008"))
import run_loro  # noqa: E402
from baselines_ref import evaluate, windows_1998  # noqa: E402
from common import RECORDS, load, loro_split  # noqa: E402
from explore_context import build  # noqa: E402
from rgcs import RadarGridCS, fit_percentiles, quantize  # noqa: E402
from shift_margin1 import train  # noqa: E402

TR, VA = 181, 45


def tensor(model, Q):
    """Q (N,14,7) uint8 -> (N,14,72) features exactly as the network's head receives them (range order)."""
    x = run_loro.make_inputs("G7", Q)
    with torch.no_grad():
        f = model.ctx(model.scn(model.readout(x)))  # (N,72,2,7)
    return f.permute(0, 3, 2, 1).reshape(f.shape[0], 14, -1).numpy()


def net_scores(m, Q):
    return RadarGridCS.to_range(m(run_loro.make_inputs("G7", Q))).detach().numpy()


def hgb(Ftr, Ytr, others):
    clf = HistGradientBoostingClassifier(max_iter=200, class_weight="balanced", random_state=0)
    clf.fit(Ftr.reshape(-1, Ftr.shape[-1]), Ytr.ravel())
    res = []
    for F in others:
        p = np.clip(clf.predict_proba(F.reshape(-1, F.shape[-1]))[:, 1], 1e-12, 1 - 1e-12)
        res.append(np.log(p / (1 - p)).reshape(F.shape[:2]))
    return res


def main():
    data = load(); rows = []; dev = "cuda"
    feat_model = build("margin1", "G7").eval()  # parameter-free part (readout, SCN, context) of the design
    X98, T98 = windows_1998()
    for k in RECORDS:
        tr, val, test = loro_split(k)
        lo, hi = fit_percentiles(np.concatenate([data[g]["X"].reshape(-1, 7) for g in tr]))
        Q = {g: quantize(data[g]["X"].reshape(-1, 14, 7), lo, hi) for g in (*tr, val, test)}
        Y = {g: np.broadcast_to(np.arange(14) == data[g]["t"], (len(Q[g]), 14)).astype(np.float32) for g in Q}
        F = {g: tensor(feat_model, Q[g]) for g in Q}
        sv, st = hgb(np.concatenate([F[g] for g in tr]), np.concatenate([Y[g] for g in tr]), (F[val], F[test]))
        for (rule, a), v in evaluate(sv.reshape(4, 256, 14), np.full(4, data[val]["t"]), st.reshape(4, 256, 14),
                                     np.full(4, data[test]["t"])).items():
            rows.append(dict(scorer="HGB", protocol="loro", fold=k, seed=0, rule=rule, alpha=a, **v))
        Xtr = torch.cat([run_loro.make_inputs("G7", Q[g]) for g in tr]); Ytr = torch.from_numpy(np.concatenate([Y[g] for g in tr]))
        Xv, Yv = run_loro.make_inputs("G7", Q[val]), torch.from_numpy(Y[val])
        for s in (0, 1, 2):
            m = train("G7", s, Xtr, Ytr, Xv, Yv, dev).cpu().eval()
            with torch.no_grad():
                sv, st = net_scores(m, Q[val]), net_scores(m, Q[test])
            for (rule, a), v in evaluate(sv.reshape(4, 256, 14), np.full(4, data[val]["t"]), st.reshape(4, 256, 14),
                                         np.full(4, data[test]["t"])).items():
                rows.append(dict(scorer="network", protocol="loro", fold=k, seed=s, rule=rule, alpha=a, **v))
        print("fold", k, flush=True)
    # cross-site
    lo, hi = fit_percentiles(np.concatenate([data[g]["X"][:, :TR].reshape(-1, 7) for g in RECORDS]))
    Qtr = np.concatenate([quantize(data[g]["X"][:, :TR].reshape(-1, 14, 7), lo, hi) for g in RECORDS])
    Qv = np.concatenate([quantize(data[g]["X"][:, TR:TR + VA].reshape(-1, 14, 7), lo, hi) for g in RECORDS])
    Ytr = np.concatenate([np.broadcast_to(np.arange(14) == data[g]["t"], (4 * TR, 14)) for g in RECORDS]).astype(np.float32)
    Yv = np.concatenate([np.broadcast_to(np.arange(14) == data[g]["t"], (4 * VA, 14)) for g in RECORDS]).astype(np.float32)
    Tv = np.repeat([data[g]["t"] for g in RECORDS], 4)
    Q98 = quantize(X98.reshape(-1, 14, 7), lo, hi)
    Ftr, Fv, F98 = tensor(feat_model, Qtr), tensor(feat_model, Qv), tensor(feat_model, Q98)
    sv, st = hgb(Ftr, Ytr, (Fv, F98))
    for (rule, a), v in evaluate(sv.reshape(40, VA, 14), Tv, st.reshape(len(X98), 117, 14), T98).items():
        rows.append(dict(scorer="HGB", protocol="1998", fold=0, seed=0, rule=rule, alpha=a, **v))
    Xt, Yt = run_loro.make_inputs("G7", Qtr), torch.from_numpy(Ytr)
    Xvv, Yvv = run_loro.make_inputs("G7", Qv), torch.from_numpy(Yv)
    for s in (0, 1, 2):
        m = train("G7", s, Xt, Yt, Xvv, Yvv, dev).cpu().eval()
        with torch.no_grad():
            sv = net_scores(m, Qv)
            st = np.concatenate([net_scores(m, Q98[i:i + 4096]) for i in range(0, len(Q98), 4096)])
        for (rule, a), v in evaluate(sv.reshape(40, VA, 14), Tv, st.reshape(len(X98), 117, 14), T98).items():
            rows.append(dict(scorer="network", protocol="1998", fold=0, seed=s, rule=rule, alpha=a, **v))
    df = pd.DataFrame(rows); df.to_csv(HERE / "C_matched_inputs.csv", index=False)
    pd.set_option("display.width", 200); pd.set_option("display.precision", 4)
    print(df.groupby(["protocol", "scorer", "rule", "alpha"])[["pd", "pfa", "pfa_adj"]].mean().to_string())
    for sc in ("HGB", "network"):
        d = df[(df.protocol == "loro") & (df.scorer == sc) & (df.rule == "texp") & (df.alpha == 0.01)]
        print(sc, "LORO runs>2a", int((d.pfa > 0.02).sum()), "of", len(d))


if __name__ == "__main__":
    main()
