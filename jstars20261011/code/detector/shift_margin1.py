"""EXPLORATORY: shift sensitivity of the range-ordered peak-margin context (margin1), retrained with the same seeds.
Same shifts and evaluation as shift_sensitivity.py."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "phaseA_20261008"))
import run_loro  # noqa: E402
from common import RECORDS, load, loro_split  # noqa: E402
from explore_context import build  # noqa: E402
from rgcs import RadarGridCS, calibrate_c, fit_percentiles, quantize, tail_params, texp_tau  # noqa: E402

SHIFTS = (-2, -1, 0, 1, 2)


def train(arm, seed, Xtr, Ytr, Xv, Yv, dev):
    torch.manual_seed(seed); np.random.seed(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    m = build("margin1", arm).to(dev); HP = run_loro.HP
    opt = torch.optim.AdamW(m.parameters(), lr=HP["lr"], weight_decay=HP["wd"])
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(HP["pos_weight"], device=dev))
    g = torch.Generator().manual_seed(seed); best = np.inf
    for _ in range(HP["epochs"]):
        m.train(); perm = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(Xtr), HP["batch"]):
            idx = perm[i:i + HP["batch"]]
            loss = lossf(RadarGridCS.to_range(m(Xtr[idx].to(dev))), Ytr[idx].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
        lv = float(lossf(torch.from_numpy(run_loro.predict(m, Xv, False, dev)).to(dev), Yv.to(dev)))
        if lv < best:
            best, state = lv, {k: v.detach().clone() for k, v in m.state_dict().items()}
    m.load_state_dict(state)
    return m


def main():
    data = load(); dev = "cuda"; rows = []
    for arm in ("G7", "G3"):
        for fold in RECORDS:
            train_r, val, test = loro_split(fold)
            lo, hi = fit_percentiles(np.concatenate([data[g]["X"].reshape(-1, 7) for g in train_r]))
            tr = [run_loro.record_arrays(data, g, lo, hi) for g in train_r]
            Qv, yv, tv = run_loro.record_arrays(data, val, lo, hi)
            Xtr = torch.cat([run_loro.make_inputs(arm, q) for q, _, _ in tr]); Ytr = torch.from_numpy(np.concatenate([y for _, y, _ in tr]))
            Xv, Yv = run_loro.make_inputs(arm, Qv), torch.from_numpy(yv)
            for seed in (0, 1, 2):
                m = train(arm, seed, Xtr, Ytr, Xv, Yv, dev)
                Sv = run_loro.predict(m, Xv, False, dev).reshape(4, 256, 14)
                uv, bv = tail_params(Sv); c = calibrate_c(Sv, np.full(4, tv), uv, bv, 1e-2)
                for s in SHIFTS:
                    X = np.roll(data[test]["X"], s, axis=2); t = (data[test]["t"] + s) % 14
                    St = run_loro.predict(m, run_loro.make_inputs(arm, quantize(X.reshape(-1, 14, 7), lo, hi)), False, dev).reshape(4, 256, 14)
                    ut, bt = tail_params(St); tau = texp_tau(ut, bt, 1e-2, c); ok = np.isfinite(tau); C = np.arange(14)
                    rows.append(dict(arm=arm, fold=fold, seed=seed, shift=s, pd=float((St[:, :, t] >= tau)[ok].mean()),
                                     pfa=float((St[:, :, C != t] >= tau[..., None])[ok].mean())))
            print(arm, fold, flush=True)
    df = pd.DataFrame(rows); df.to_csv(HERE / "C_shift_margin1.csv", index=False)
    f = df.groupby(["arm", "shift", "fold"])[["pd", "pfa"]].mean().reset_index()
    print(f.groupby(["arm", "shift"])[["pd", "pfa"]].mean().round(4).to_string())
    print(f[f.arm == "G7"].pivot(index="fold", columns="shift", values="pd").round(3).to_string())


if __name__ == "__main__":
    main()
