"""Checks M (tail mechanism) and Q (sample-count confound); criteria in D_PLAN.md."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from n1_temporal_net import (DEV, TR, VA, evaluate, feat_front, fit_net, lab, load, loro_split,  # noqa: E402
                             net_scores, windows_1998, RECORDS)
from rgcs import calibrate_c, fit_percentiles, quantize, tail_params, texp_tau  # noqa: E402
from matched_inputs import tensor  # noqa: E402

C = np.arange(14)
ALPHAS = (1e-1, 3e-2, 1e-2, 3e-3, 1e-3)


def tail_stats(Sv, tv, St, tt):
    """Calibration ratios over ALPHAS, tail-shape index h and location drift of the test record."""
    uv, bv = tail_params(Sv); ut, bt = tail_params(St); c = calibrate_c(Sv, tv, uv, bv, 1e-2)
    h0t = St[np.broadcast_to(C[None, None] != tt[:, None, None], St.shape)]
    h0v = Sv[np.broadcast_to(C[None, None] != tv[:, None, None], Sv.shape)]
    out = {}
    for a in ALPHAS:
        tau = texp_tau(ut, bt, a, c); ok = np.isfinite(tau)
        act = (St >= tau[..., None]) & (C[None, None] != tt[:, None, None])
        out[f"ratio_{a:g}"] = float(act[ok].sum() / ((C[None, None] != tt[:, None, None]) & ok[..., None]).sum() / a)
    q90, q99, q999 = np.quantile(h0t, [0.9, 0.99, 0.999])
    out["h"] = (q999 - q99) / (q99 - q90)
    beta = (q99 - q90) / np.log(10)
    out["drift"] = abs(q99 - np.quantile(h0v, 0.99)) / beta
    return out


def check_m(front, feat):
    data = load(); rows = []
    n_tr, n_ev = np.arange(7, 256), np.arange(15, 256)
    for k in RECORDS:
        tr, va, te = loro_split(k)
        lo, hi = fit_percentiles(np.concatenate([data[g]["X"].reshape(-1, 7) for g in tr]))
        F = {g: feat(data[g]["X"], lo, hi) for g in (*tr, va, te)}
        Ftr = torch.cat([F[g] for g in tr]); ytr = lab(np.repeat([data[g]["t"] for g in tr], 4), len(tr) * 4)
        yv = lab(np.full(4, data[va]["t"]), 4)
        for K in (1, 4):
            for s in (0, 1, 2):
                m = fit_net(s, K, Ftr, n_tr, ytr, F[va], n_ev, yv)
                Sv, St = net_scores(m, F[va], n_ev, K), net_scores(m, F[te], n_ev, K)
                np.savez_compressed(HERE / "scores" / f"loro_K{K}_f{k:02d}_s{s}.npz", val=Sv, test=St, val_t=data[va]["t"], test_t=data[te]["t"])
                rows.append(dict(protocol="loro", K=K, fold=k, seed=s, **tail_stats(Sv, np.full(4, data[va]["t"]), St, np.full(4, data[te]["t"]))))
        print("M fold", k, flush=True)
    lo, hi = fit_percentiles(np.concatenate([data[g]["X"][:, :TR].reshape(-1, 7) for g in RECORDS]))
    F = {g: feat(data[g]["X"], lo, hi) for g in RECORDS}
    Ftr = torch.cat([F[g] for g in RECORDS]); ytr = lab(np.repeat([data[g]["t"] for g in RECORDS], 4), 40)
    X98, T98 = windows_1998()
    F98 = torch.cat([feat(X98[i:i + 32], lo, hi) for i in range(0, len(X98), 32)])
    Tv = np.repeat([data[g]["t"] for g in RECORDS], 4)
    for K in (1, 4):
        for s in (0, 1, 2):
            m = fit_net(s, K, Ftr, np.arange(7, TR), ytr, Ftr, np.arange(TR, TR + VA), ytr)
            Sv, St = net_scores(m, Ftr, np.arange(TR, TR + VA), K), net_scores(m, F98, np.arange(7, 117), K)
            rows.append(dict(protocol="1998", K=K, fold=0, seed=s, **tail_stats(Sv, Tv, St, T98)))
    pd.DataFrame(rows).to_csv(HERE / "CHK_M_mechanism.csv", index=False)


def check_q(front, feat):
    rows = []
    cfg = {"0p256": (0.256, 4, (4, 8)), "0p512": (0.512, 2, (2, 4)), "1p024": (1.024, 1, (1, 2))}
    for tag, (d, stride, Ks) in cfg.items():
        data = load(tag); Fr = data[1]["X"].shape[1]
        n_tr = np.arange(7 * stride, Fr, stride)[: 121]          # 121 training frames per sequence, as at 1.024 s
        n_tr = n_tr[n_tr >= 7]
        n_ev = np.arange(int(np.ceil(8.192 / d)) - 1, Fr)
        for k in RECORDS:
            tr, va, te = loro_split(k)
            lo, hi = fit_percentiles(np.concatenate([data[g]["X"].reshape(-1, 7) for g in tr]))
            F = {g: feat(data[g]["X"], lo, hi) for g in (*tr, va, te)}
            Ftr = torch.cat([F[g] for g in tr]); ytr = lab(np.repeat([data[g]["t"] for g in tr], 4), len(tr) * 4)
            yv = lab(np.full(4, data[va]["t"]), 4)
            for K in Ks:
                for s in (0, 1, 2):
                    m = fit_net(s, K, Ftr, n_tr, ytr, F[va], n_ev, yv)
                    v = evaluate(net_scores(m, F[va], n_ev, K), np.full(4, data[va]["t"]), net_scores(m, F[te], n_ev, K),
                                 np.full(4, data[te]["t"]))[("texp", 1e-2)]
                    rows.append(dict(dur=d, K=K, dwell=round(d * K, 3), n_train_frames=len(n_tr), fold=k, seed=s, **v))
            print(f"Q d={d} fold {k}", flush=True)
            pd.DataFrame(rows).to_csv(HERE / "CHK_Q_confound.csv", index=False)


if __name__ == "__main__":
    (HERE / "scores").mkdir(exist_ok=True)
    front = feat_front()
    feat = lambda X, lo, hi: torch.as_tensor(tensor(front, quantize(X.reshape(-1, 14, 7), lo, hi)).reshape(*X.shape[:3], 72), device=DEV)
    check_q(front, feat)
    check_m(front, feat)
