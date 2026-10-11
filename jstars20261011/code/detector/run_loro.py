"""Phase C IPIX1993 LORO runner (see phaseA_20261008/PHASE_C_PREREGISTRATION.md).

Usage:  python run_loro.py --arms G7 G3 D7 P7 G7-noSCN G7-absC --folds 1-10 --seeds 0 1 2
        python run_loro.py --smoke   (fold 1, seed 0, G7: prints training/validation loss only, no test metrics)
Every arm reports both threshold rules: texp64 (primary) and fixed (G7-fixed ablation = G7 scores, fixed rule).
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "phaseA_20261008"))
from common import RECORDS, calibrate, load, loro_split, pd_at_pfa  # noqa: E402
from rgcs import (RadarGridCS, calibrate_c, cells_to_grid, fit_percentiles, quantize, render, tail_params,  # noqa: E402
                  texp_tau)

ARMS = {
    "G7": dict(input="image", group="G7", scn=True, absolute_ctx=False),
    "G3": dict(input="image", group="G3", scn=True, absolute_ctx=False),
    "D7": dict(input="vector", proj=False, scn=True, absolute_ctx=False),
    "P7": dict(input="vector", proj=True, scn=True, absolute_ctx=False),
    "G7-noSCN": dict(input="image", group="G7", scn=False, absolute_ctx=False),
    "G7-absC": dict(input="image", group="G7", scn=True, absolute_ctx=True),
}
HP = dict(epochs=40, batch=256, lr=1e-3, wd=1e-4, pos_weight=13.0, W=64)
OUT = HERE / "runs_loro"


def make_inputs(arm, Q):
    """Q (N,14,7) uint8 -> float tensor input for the arm."""
    cfg = ARMS[arm]
    if cfg["input"] == "image":
        return torch.from_numpy(render(Q, cfg["group"])).float() / 255
    return torch.from_numpy(cells_to_grid(Q.astype(np.float32) / 255).copy())


def build_model(arm):
    cfg = ARMS[arm]
    if cfg["input"] == "image":
        return RadarGridCS(in_addr=36, scn=cfg["scn"], absolute_ctx=cfg["absolute_ctx"])
    return RadarGridCS(in_addr=9 if cfg["proj"] else 21, scn=cfg["scn"], absolute_ctx=cfg["absolute_ctx"],
                       proj7to3=cfg["proj"])


def record_arrays(data, g, lo, hi):
    X, t = data[g]["X"], data[g]["t"]  # (4,256,14,7)
    Q = quantize(X.reshape(-1, 14, 7), lo, hi)
    y = np.zeros((Q.shape[0], 14), np.float32); y[:, t] = 1
    return Q, y, t


def predict(model, x, vector, dev):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(x), 1024):
            out.append(RadarGridCS.to_range(model(x[i:i + 1024].to(dev), vector=vector)).cpu())
    return torch.cat(out).numpy()


def run(arm, fold, seed, data, dev, smoke=False):
    torch.manual_seed(seed); np.random.seed(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    train, val, test = loro_split(fold)
    lo, hi = fit_percentiles(np.concatenate([data[g]["X"].reshape(-1, 7) for g in train]))
    tr = [record_arrays(data, g, lo, hi) for g in train]
    Qv, yv, tv = record_arrays(data, val, lo, hi)
    vector = ARMS[arm]["input"] == "vector"
    xtr = torch.cat([make_inputs(arm, q) for q, _, _ in tr]); ytr = torch.from_numpy(np.concatenate([y for _, y, _ in tr]))
    xv, yvt = make_inputs(arm, Qv), torch.from_numpy(yv)
    model = build_model(arm).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=HP["lr"], weight_decay=HP["wd"])
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(HP["pos_weight"], device=dev))
    g = torch.Generator().manual_seed(seed)
    best, best_state, hist = np.inf, None, []
    for ep in range(HP["epochs"]):
        model.train(); perm = torch.randperm(len(xtr), generator=g); tl = 0.0
        for i in range(0, len(xtr), HP["batch"]):
            idx = perm[i:i + HP["batch"]]
            logit = RadarGridCS.to_range(model(xtr[idx].to(dev), vector=vector))
            loss = lossf(logit, ytr[idx].to(dev))
            opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item() * len(idx)
        lv = float(lossf(torch.from_numpy(predict(model, xv, vector, dev)).to(dev), yvt.to(dev)))
        hist.append((tl / len(xtr), lv))
        if lv < best:
            best, best_state, best_ep = lv, {k: v.detach().clone() for k, v in model.state_dict().items()}, ep
    if smoke:
        for ep, (a, b) in enumerate(hist):
            print(f"epoch {ep:2d} train {a:.4f} val {b:.4f}")
        print("params", sum(p.numel() for p in model.parameters()), "best epoch", best_ep)
        return None
    model.load_state_dict(best_state)
    Qt, yt, tt = record_arrays(data, test, lo, hi)
    Sv = predict(model, xv, vector, dev).reshape(4, 256, 14)
    St = predict(model, make_inputs(arm, Qt), vector, dev).reshape(4, 256, 14)
    res = dict(arm=arm, fold=fold, seed=seed, val_record=val, best_epoch=best_ep, best_val_bce=best,
               params=sum(p.numel() for p in model.parameters()), hp=HP)
    uv, bv = tail_params(Sv, HP["W"]); ut, bt = tail_params(St, HP["W"])
    c = calibrate_c(Sv, np.full(4, tv), uv, bv, 1e-2)
    res["texp_c"] = c
    C = np.arange(14); h0m = C != tt; adjm = np.abs(C - tt) == 1
    for a in (1e-2, 1e-3):
        rules = {"texp64": texp_tau(ut, bt, a, c),
                 "fixed": np.full(St.shape[:2], calibrate(Sv[:, :, C != tv], a))}
        for rule, tau in rules.items():
            ok = np.isfinite(tau)
            T = tau[..., None]
            res[f"{rule}_{a:g}"] = dict(pd=float((St[:, :, tt] >= tau)[ok].mean()),
                                        pfa=float((St[:, :, h0m] >= T)[ok].mean()),
                                        pfa_adj=float((St[:, :, adjm] >= T)[ok].mean()))
        res[f"pdor_{a:g}"] = pd_at_pfa(St[:, :, tt].ravel(), St[:, :, h0m].ravel(), a)
    res["exact_cell"] = float((St.argmax(-1) == tt).mean())
    tag = f"{arm}_f{fold:02d}_s{seed}"
    np.savez_compressed(OUT / f"{tag}_scores.npz", val=Sv, test=St, val_t=tv, test_t=tt)
    torch.save(best_state, OUT / f"{tag}.pt")
    (OUT / f"{tag}.json").write_text(json.dumps(res, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["G7", "G3"])
    ap.add_argument("--folds", default="1-10")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = load()
    if a.smoke:
        run("G7", 1, 0, data, dev, smoke=True); return
    OUT.mkdir(exist_ok=True)
    (OUT / "config_hash.txt").write_text(hashlib.sha256(Path(__file__).read_bytes() + (HERE / "rgcs.py").read_bytes())
                                         .hexdigest())
    f0, f1 = map(int, a.folds.split("-"))
    for arm in a.arms:
        for fold in range(f0, f1 + 1):
            for seed in a.seeds:
                if (OUT / f"{arm}_f{fold:02d}_s{seed}.json").exists():
                    continue
                t0 = time.time(); r = run(arm, fold, seed, data, dev)
                print(arm, fold, seed, f"{time.time() - t0:.0f}s", "best_ep", r["best_epoch"], flush=True)


if __name__ == "__main__":
    main()
