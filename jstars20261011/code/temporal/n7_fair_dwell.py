"""F: fair equal-dwell control (rules in D_PLAN.md, 'Round-2 review controls').
Same pulses per training window, common decision times, same physical background history (32.768 s)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from n1_temporal_net import fit_net, lab, load, loro_split, net_scores, RECORDS  # noqa: E402
from n3_final import make_feat  # noqa: E402
from rgcs import calibrate_c, fit_percentiles, tail_params, texp_tau  # noqa: E402
from common import calibrate, pd_at_pfa  # noqa: E402

OUT = HERE / "fair"; OUT.mkdir(exist_ok=True)
C = np.arange(14)
J_TR, J_EV = np.arange(7, 128), np.arange(40, 128)
BG_SECONDS = 32.768
CFG = {"0p256": (0.256, (4, 8)), "0p512": (0.512, (2, 4)), "1p024": (1.024, (1, 2))}


def metrics(Sv, St, tv, tt, d, K):
    """Native-rate scores (4, F_native - K + 1, 14) starting at native index K-1; decisions at common times."""
    s = int(round(1.024 / d)); W = int(round(BG_SECONDS / d))
    dec = (J_EV + 1) * s - 1 - (K - 1)                      # positions in the score arrays
    uv, bv = tail_params(Sv, W); ut, bt = tail_params(St, W)
    Svd, uvd, bvd = Sv[:, dec], uv[:, dec], bv[:, dec]
    Std, utd, btd = St[:, dec], ut[:, dec], bt[:, dec]
    c = calibrate_c(Svd, np.full(4, tv), uvd, bvd, 1e-2)
    h0, adj = C != tt, np.abs(C - tt) == 1
    out = {}
    for a in (1e-2, 1e-3):
        tau = texp_tau(utd, btd, a, c); act = Std >= tau[..., None]
        out[f"pd_{a:g}"] = float(act[..., tt].mean()); out[f"pfa_{a:g}"] = float(act[..., h0].mean())
        out[f"adj_{a:g}"] = float(act[..., adj].mean())
    out["oracle_0.01"] = pd_at_pfa(Std[..., tt].ravel(), Std[..., h0].ravel(), 1e-2)
    out["c"] = c
    return out


def main():
    feat = make_feat("G7"); rows = []
    for tag, (d, Ks) in CFG.items():
        data = load(tag); s = int(round(1.024 / d))
        for k in RECORDS:
            tr, va, te = loro_split(k)
            lo, hi = fit_percentiles(np.concatenate([data[g]["X"].reshape(-1, 7) for g in tr]))
            F = {g: feat(data[g]["X"], lo, hi) for g in (*tr, va, te)}
            Ftr = torch.cat([F[g] for g in tr]); ytr = lab(np.repeat([data[g]["t"] for g in tr], 4), len(tr) * 4)
            yv = lab(np.full(4, data[va]["t"]), 4)
            Fn = F[te].shape[1]
            for K in Ks:
                n_tr = (J_TR + 1) * s - 1                    # window end indices: same pulses for every d
                n_tr = n_tr[n_tr >= K - 1]
                n_val = (J_EV + 1) * s - 1
                n_all = np.arange(K - 1, Fn)
                for seed in (0, 1, 2):
                    m = fit_net(seed, K, Ftr, n_tr, ytr, F[va], n_val, yv)
                    Sv, St = net_scores(m, F[va], n_all, K), net_scores(m, F[te], n_all, K)
                    np.savez_compressed(OUT / f"d{tag}_K{K}_f{k:02d}_s{seed}_scores.npz", val=Sv, test=St,
                                        val_t=data[va]["t"], test_t=data[te]["t"], first_native=K - 1)
                    rows.append(dict(dur=d, K=K, dwell=round(d * K, 3), fold=k, seed=seed, n_train=len(n_tr),
                                     **metrics(Sv, St, data[va]["t"], data[te]["t"], d, K)))
            print(f"F d={d} fold {k}", flush=True)
            pd.DataFrame(rows).to_csv(HERE / "F_fair_dwell.csv", index=False)


if __name__ == "__main__":
    main()
