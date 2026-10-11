"""EXPLORATORY (post hoc; IPIX1998 already seen in E3): which context form keeps adjacent-cell false alarms under
control across sites? Frozen Phase C code (rgcs.py, run_loro.py) is imported unchanged; variants subclass it.

Context variants (all on SA-RGB G7 unless noted; SCN on; texp64 CFAR):
  grid3x3   frozen Phase C design (8 grid neighbours = range offsets +-1, +-2, +-3)
  none      cell under test only
  r1        range neighbours +-1 only (differences + masks + peak margin)
  r2        range neighbours +-2 only (skips the leakage cells)
  r12       range neighbours +-1 and +-2
  margin1   only the peak margin to +-1 neighbours (one scalar per address)
Both protocols are run: IPIX1993 LORO (10 folds x 3 seeds) and IPIX1993 -> IPIX1998 (3 seeds). G3 is run for the
variant(s) of interest to keep the G7 - G3 contrast.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "phaseA_20261008"))
import rgcs  # noqa: E402
import run_loro  # noqa: E402
from common import POLS, RECORDS, load, loro_split  # noqa: E402
from rgcs import RadarGridCS, calibrate_c, fit_percentiles, quantize, tail_params, texp_tau  # noqa: E402

OUT = HERE / "explore_context"
VARIANTS = {"grid3x3": None, "none": (), "r1": (-1, 1), "r2": (-2, 2), "r12": (-2, -1, 1, 2), "margin1": "margin1"}


class RangeContext(nn.Module):
    """Context in range order: z (B,C,2,7) -> features on the grid, built from range offsets."""

    def __init__(self, offsets):
        super().__init__()
        self.offsets = offsets

    def forward(self, z):
        B, C, H, W = z.shape
        r = z.permute(0, 1, 3, 2).reshape(B, C, 14)  # range order (cell k = 2*col + row)
        feats = [r]
        if self.offsets == "margin1":
            nbs = []
            for o in (-1, 1):
                nb = torch.full_like(r, -1e6)
                if o < 0:
                    nb[..., -o:] = r[..., :o]
                else:
                    nb[..., :-o] = r[..., o:]
                nbs.append(nb)
            feats.append(r - torch.stack(nbs).max(0).values)
        elif self.offsets:
            nbs, masks = [], []
            for o in self.offsets:
                nb = torch.zeros_like(r); m = torch.zeros_like(r[:, :1])
                if o < 0:
                    nb[..., -o:] = r[..., :o]; m[..., -o:] = 1
                else:
                    nb[..., :-o] = r[..., o:]; m[..., :-o] = 1
                feats.append((r - nb) * m); masks.append(m)
                nbs.append(torch.where(m.bool(), nb, torch.full_like(nb, -1e6)))
            feats.append(r - torch.stack(nbs).max(0).values)
            feats += masks
        out = torch.cat(feats, 1)  # (B, C', 14)
        return out.view(B, -1, 7, 2).permute(0, 1, 3, 2)  # back to (B, C', 2, 7)


def build(variant, arm):
    off = VARIANTS[variant]
    model = run_loro.build_model(arm)
    if off is None:
        return model
    model.ctx = RangeContext(off)
    c = 36 if run_loro.ARMS[arm]["input"] == "image" else 21
    n = c * (2 if off == "margin1" else 1 + (len(off) + 1 if off else 0)) + (len(off) if off and off != "margin1" else 0)
    hidden = model.head[0].out_channels
    model.head[0] = nn.Conv2d(n, hidden, 1)
    return model


def train_eval(variant, arm, seed, Xtr, Ytr, Xv, Yv, Tv, Xt, Tt, seq_shape_v, seq_shape_t, dev, adj_t):
    torch.manual_seed(seed); np.random.seed(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    vector = run_loro.ARMS[arm]["input"] == "vector"
    model = build(variant, arm).to(dev)
    HP = run_loro.HP
    opt = torch.optim.AdamW(model.parameters(), lr=HP["lr"], weight_decay=HP["wd"])
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(HP["pos_weight"], device=dev))
    g = torch.Generator().manual_seed(seed); best = np.inf
    for ep in range(HP["epochs"]):
        model.train(); perm = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(Xtr), HP["batch"]):
            idx = perm[i:i + HP["batch"]]
            loss = lossf(RadarGridCS.to_range(model(Xtr[idx].to(dev), vector=vector)), Ytr[idx].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
        lv = float(lossf(torch.from_numpy(run_loro.predict(model, Xv, vector, dev)).to(dev), Yv.to(dev)))
        if lv < best:
            best, state = lv, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    Sv = run_loro.predict(model, Xv, vector, dev).reshape(seq_shape_v)
    St = run_loro.predict(model, Xt, vector, dev).reshape(seq_shape_t)
    uv, bv = tail_params(Sv); c = calibrate_c(Sv, Tv, uv, bv, 1e-2)
    ut, bt = tail_params(St)
    C = np.arange(14)
    h1 = St[np.arange(len(Tt)), :, Tt]
    h0m = (C[None] != Tt[:, None])[:, None]; adjm = (np.abs(C[None] - Tt[:, None]) == 1)[:, None]
    res = {}
    for a in (1e-2, 1e-3):
        tau = texp_tau(ut, bt, a, c); ok = np.isfinite(tau)
        act = St >= tau[..., None]; okc = ok[..., None]
        res[f"{a:g}"] = dict(pd=float(((h1 >= tau) & ok).sum() / ok.sum()),
                             pfa=float((act & h0m & okc).sum() / (h0m & okc).sum()),
                             pfa_adj=float((act & adjm & okc).sum() / (adjm & okc).sum()))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--arms", nargs="+", default=["G7"])
    ap.add_argument("--protocols", nargs="+", default=["loro", "1998"])
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True); dev = "cuda"
    data = load()
    d98 = np.load(HERE.parent / "phaseA_20261008" / "out" / "ipix1998_7feat.npz")
    meta = json.load(open(HERE.parent / "phaseA_20261008" / "out" / "ipix1998_7feat.json", encoding="utf8"))
    rows = []
    for arm in a.arms:
        mk = lambda Q: run_loro.make_inputs(arm, Q)
        if "loro" in a.protocols:
            for fold in RECORDS:
                train, val, test = loro_split(fold)
                lo, hi = fit_percentiles(np.concatenate([data[g]["X"].reshape(-1, 7) for g in train]))
                ra = lambda g: run_loro.record_arrays(data, g, lo, hi)
                tr = [ra(g) for g in train]; Qv, yv, tv = ra(val); Qt, _, tt = ra(test)
                Xtr = torch.cat([mk(q) for q, _, _ in tr]); Ytr = torch.from_numpy(np.concatenate([y for _, y, _ in tr]))
                Xv, Yv, Xt = mk(Qv), torch.from_numpy(yv), mk(Qt)
                for v in a.variants:
                    for seed in (0, 1, 2):
                        r = train_eval(v, arm, seed, Xtr, Ytr, Xv, Yv, np.full(4, tv), Xt, np.full(4, tt),
                                       (4, 256, 14), (4, 256, 14), dev, None)
                        rows.append(dict(protocol="loro", arm=arm, variant=v, fold=fold, seed=seed, **{
                            f"{m}_{al}": r[al][m] for al in r for m in r[al]}))
                print(arm, "loro fold", fold, flush=True)
        if "1998" in a.protocols:
            TR, VA = 181, 45
            lo, hi = fit_percentiles(np.concatenate([data[g]["X"][:, :TR].reshape(-1, 7) for g in RECORDS]))

            def split(sl):
                Q, Y, T = [], [], []
                for g in RECORDS:
                    Q.append(quantize(data[g]["X"][:, sl].reshape(-1, 14, 7), lo, hi))
                    y = np.zeros((Q[-1].shape[0], 14), np.float32); y[:, data[g]["t"]] = 1; Y.append(y)
                    T += [data[g]["t"]] * 4
                return Q, Y, np.array(T)

            Qtr, Ytr_, _ = split(slice(0, TR)); Qv, Yv_, Tv = split(slice(TR, TR + VA))
            seqs = []
            for rr in meta["records"]:
                t = rr["target_cell"] - 1
                for p in POLS:
                    k = f"r{rr['rec']:02d}_{p}"
                    for j, s in enumerate(d98[k + "_starts"]):
                        seqs.append((quantize(d98[k + "_win"][:, j].astype(np.float64), lo, hi), t - int(s)))
            Xtr = torch.cat([mk(q) for q in Qtr]); Ytr = torch.from_numpy(np.concatenate(Ytr_))
            Xv = torch.cat([mk(q) for q in Qv]); Yv = torch.from_numpy(np.concatenate(Yv_))
            Xt = torch.cat([mk(s[0]) for s in seqs]); Tt = np.array([s[1] for s in seqs])
            for v in a.variants:
                for seed in (0, 1, 2):
                    r = train_eval(v, arm, seed, Xtr, Ytr, Xv, Yv, Tv, Xt, Tt, (40, VA, 14), (len(seqs), 117, 14),
                                   dev, None)
                    rows.append(dict(protocol="1998", arm=arm, variant=v, fold=0, seed=seed, **{
                        f"{m}_{al}": r[al][m] for al in r for m in r[al]}))
                print(arm, "1998", v, rows[-1], flush=True)
        import pandas as pd
        pd.DataFrame(rows).to_csv(OUT / f"rows_{'_'.join(a.arms)}.csv", index=False)


if __name__ == "__main__":
    main()
