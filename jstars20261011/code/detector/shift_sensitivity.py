"""EXPLORATORY: residual position cues of the frozen G7/G3 checkpoints (no retraining).

The test record's 14-cell range profile is cyclically shifted by s cells before rendering, so the target moves to
another grid row (odd s) or column (even s) while keeping its neighbours (except across the wrap). The frozen
checkpoint and its validation-calibrated tail scale c are unchanged. If decisions relied on grid row/edge cues
exposed by the validity masks, Pd and Pfa would change with s.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "phaseA_20261008"))
from common import RECORDS, load, loro_split  # noqa: E402
from rgcs import calibrate_c, fit_percentiles, quantize, tail_params, texp_tau  # noqa: E402
from run_loro import build_model, make_inputs, predict, record_arrays  # noqa: E402

SHIFTS = (-2, -1, 0, 1, 2)


def main():
    data = load(); dev = "cuda"; rows = []
    for arm in ("G7", "G3"):
        for fold in RECORDS:
            train, val, test = loro_split(fold)
            lo, hi = fit_percentiles(np.concatenate([data[g]["X"].reshape(-1, 7) for g in train]))
            Qv, _, tv = record_arrays(data, val, lo, hi)
            for seed in (0, 1, 2):
                m = build_model(arm).to(dev)
                m.load_state_dict(torch.load(HERE / "runs_loro" / f"{arm}_f{fold:02d}_s{seed}.pt", map_location=dev))
                Sv = predict(m, make_inputs(arm, Qv), False, dev).reshape(4, 256, 14)
                uv, bv = tail_params(Sv); c = calibrate_c(Sv, np.full(4, tv), uv, bv, 1e-2)
                for s in SHIFTS:
                    X = np.roll(data[test]["X"], s, axis=2); t = (data[test]["t"] + s) % 14
                    Q = quantize(X.reshape(-1, 14, 7), lo, hi)
                    St = predict(m, make_inputs(arm, Q), False, dev).reshape(4, 256, 14)
                    ut, bt = tail_params(St); tau = texp_tau(ut, bt, 1e-2, c); ok = np.isfinite(tau)
                    C = np.arange(14)
                    rows.append(dict(arm=arm, fold=fold, seed=seed, shift=s, target_row=t % 2,
                                     pd=float((St[:, :, t] >= tau)[ok].mean()),
                                     pfa=float((St[:, :, C != t] >= tau[..., None])[ok].mean())))
            print(arm, fold, flush=True)
    df = pd.DataFrame(rows); df.to_csv(HERE / "C_shift_sensitivity.csv", index=False)
    f = df.groupby(["arm", "shift", "fold"])[["pd", "pfa"]].mean().reset_index()
    print(f.groupby(["arm", "shift"])[["pd", "pfa"]].mean().round(4).to_string())
    for arm in ("G7", "G3"):
        base = f[(f.arm == arm) & (f["shift"] == 0)].set_index("fold")
        for s in SHIFTS:
            if s == 0:
                continue
            d = f[(f.arm == arm) & (f["shift"] == s)].set_index("fold")
            dd = (d.pd - base.pd)
            print(arm, s, "dPd mean", round(dd.mean(), 4), "max|dPd|", round(dd.abs().max(), 4),
                  "dPfa/a mean", round(((d.pfa - base.pfa) / 0.01).mean(), 3))


if __name__ == "__main__":
    main()
