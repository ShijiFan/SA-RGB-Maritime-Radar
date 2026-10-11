"""N4: cyclic range-shift test of the final detector (G7, K = 4) with the saved checkpoints, as in phaseC/shift_sensitivity."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from n1_temporal_net import DEV, TemporalCellNet, load, loro_split, net_scores, RECORDS  # noqa: E402
from n3_final import make_feat  # noqa: E402
from rgcs import calibrate_c, fit_percentiles, tail_params, texp_tau  # noqa: E402

C = np.arange(14)
n_ev = np.arange(15, 256)


def main():
    data = load(); feat = make_feat("G7"); rows = []
    for k in RECORDS:
        tr, va, te = loro_split(k)
        lo, hi = fit_percentiles(np.concatenate([data[g]["X"].reshape(-1, 7) for g in tr]))
        Fv = feat(data[va]["X"], lo, hi)
        for s in (0, 1, 2):
            m = TemporalCellNet().to(DEV); m.load_state_dict(torch.load(HERE / "final" / f"G7_K4_f{k:02d}_s{s}.pt")); m.eval()
            Sv = net_scores(m, Fv, n_ev, 4); uv, bv = tail_params(Sv); c = calibrate_c(Sv, np.full(4, data[va]["t"]), uv, bv, 1e-2)
            for sh in (-2, -1, 0, 1, 2):
                X = np.roll(data[te]["X"], sh, axis=2); t = (data[te]["t"] + sh) % 14
                St = net_scores(m, feat(X, lo, hi), n_ev, 4); ut, bt = tail_params(St); tau = texp_tau(ut, bt, 1e-2, c)
                ok = np.isfinite(tau)
                rows.append(dict(fold=k, seed=s, shift=sh, pd=float((St[:, :, t] >= tau)[ok].mean()),
                                 pfa=float((St[:, :, C != t] >= tau[..., None])[ok].mean())))
        print("fold", k, flush=True)
    df = pd.DataFrame(rows); df.to_csv(HERE / "N4_shift_final.csv", index=False)
    df["a"] = df["shift"].abs()
    print(df.groupby("a")[["pd", "pfa"]].mean().round(4))


if __name__ == "__main__":
    main()
