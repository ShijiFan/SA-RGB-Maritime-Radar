"""Phase C IPIX1998 cross-site confirmation (pre-registered E3). Run ONCE, after C_loro_endpoints.json exists.

Train on the chronological IPIX1993 training frames (first 181 of 256 per sequence, all 10 records), select the
checkpoint and calibrate the texp64 scale on the IPIX1993 validation frames (next 45), and test on every IPIX1998
14-cell window (window-local reference) with the 1993 training percentiles. Arms: G7, G3 and D7; seeds 0, 1, 2.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "phaseA_20261008"))
from common import POLS, RECORDS, load  # noqa: E402
from rgcs import RadarGridCS, calibrate_c, fit_percentiles, quantize, tail_params, texp_tau  # noqa: E402
from run_loro import HP, build_model, make_inputs, predict  # noqa: E402

OUT = HERE / "runs_1998"
TR, VA = 181, 45


def main():
    if not (HERE / "C_loro_endpoints.json").exists():
        raise SystemExit("Pre-registration order: summarize IPIX1993 LORO first.")
    OUT.mkdir(exist_ok=True)
    data = load()
    d98 = np.load(HERE.parent / "phaseA_20261008" / "out" / "ipix1998_7feat.npz")
    meta = json.load(open(HERE.parent / "phaseA_20261008" / "out" / "ipix1998_7feat.json", encoding="utf8"))
    lo, hi = fit_percentiles(np.concatenate([data[g]["X"][:, :TR].reshape(-1, 7) for g in RECORDS]))

    def split(sl):
        Q, Y, T = [], [], []
        for g in RECORDS:
            X, t = data[g]["X"][:, sl], data[g]["t"]
            Q.append(quantize(X.reshape(-1, 14, 7), lo, hi))
            y = np.zeros((Q[-1].shape[0], 14), np.float32); y[:, t] = 1; Y.append(y); T += [t] * 4
        return Q, Y, np.array(T)

    Qtr, Ytr, _ = split(slice(0, TR)); Qv, Yv, Tv = split(slice(TR, TR + VA))
    seqs = []  # (rec, pol, start, Q (117,14,7), t_in_window)
    for r in meta["records"]:
        t = r["target_cell"] - 1
        for p in POLS:
            k = f"r{r['rec']:02d}_{p}"
            for j, s in enumerate(d98[k + "_starts"]):
                seqs.append((r["rec"], p, int(s), quantize(d98[k + "_win"][:, j].astype(np.float64), lo, hi), t - int(s)))
    dev = "cuda"
    results = []
    for arm in ("G7", "G3", "D7"):
        vector = arm == "D7"
        xtr = torch.cat([make_inputs(arm, q) for q in Qtr]); ytr = torch.from_numpy(np.concatenate(Ytr))
        xv = torch.cat([make_inputs(arm, q) for q in Qv]); yv = torch.from_numpy(np.concatenate(Yv))
        xt = torch.cat([make_inputs(arm, s[3]) for s in seqs])
        for seed in (0, 1, 2):
            torch.manual_seed(seed); np.random.seed(seed)
            torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
            model = build_model(arm).to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=HP["lr"], weight_decay=HP["wd"])
            lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(HP["pos_weight"], device=dev))
            g = torch.Generator().manual_seed(seed); best = np.inf
            for ep in range(HP["epochs"]):
                model.train(); perm = torch.randperm(len(xtr), generator=g)
                for i in range(0, len(xtr), HP["batch"]):
                    idx = perm[i:i + HP["batch"]]
                    loss = lossf(RadarGridCS.to_range(model(xtr[idx].to(dev), vector=vector)), ytr[idx].to(dev))
                    opt.zero_grad(); loss.backward(); opt.step()
                lv = float(lossf(torch.from_numpy(predict(model, xv, vector, dev)).to(dev), yv.to(dev)))
                if lv < best:
                    best, state, best_ep = lv, {k: v.detach().clone() for k, v in model.state_dict().items()}, ep
            model.load_state_dict(state)
            Sv = predict(model, xv, vector, dev).reshape(40, VA, 14)
            uv, bv = tail_params(Sv, HP["W"]); c = calibrate_c(Sv, Tv, uv, bv, 1e-2)
            St = predict(model, xt, vector, dev).reshape(len(seqs), 117, 14)
            Tt = np.array([s[4] for s in seqs]); recs = np.array([s[0] for s in seqs])
            ut, bt = tail_params(St, HP["W"])
            C = np.arange(14)
            h1 = St[np.arange(len(seqs)), :, Tt]  # (n_seq,117)
            h0m = C[None] != Tt[:, None]; adjm = np.abs(C[None] - Tt[:, None]) == 1
            res = dict(arm=arm, seed=seed, best_epoch=best_ep, texp_c=c)
            for a in (1e-2, 1e-3):
                tau = texp_tau(ut, bt, a, c); ok = np.isfinite(tau)
                act = St >= tau[..., None]
                res[f"{a:g}"] = dict(pd=float(((h1 >= tau) & ok).sum() / ok.sum()),
                                     pfa=float((act & h0m[:, None] & ok[..., None]).sum() / (h0m[:, None] & ok[..., None]).sum()),
                                     pfa_adj=float((act & adjm[:, None] & ok[..., None]).sum() / (adjm[:, None] & ok[..., None]).sum()),
                                     pd_by_record={int(r): float(((h1 >= tau) & ok)[recs == r].sum() / ok[recs == r].sum())
                                                   for r in np.unique(recs)},
                                     pfa_by_record={int(r): float((act & h0m[:, None] & ok[..., None])[recs == r].sum() /
                                                                  (h0m[:, None] & ok[..., None])[recs == r].sum())
                                                    for r in np.unique(recs)})
            res["exact_cell"] = float((St.argmax(-1) == Tt[:, None]).mean())
            np.savez_compressed(OUT / f"{arm}_s{seed}_scores.npz", test=St, t=Tt, rec=recs)
            (OUT / f"{arm}_s{seed}.json").write_text(json.dumps(res, indent=1))
            results.append(res)
            print(arm, seed, {k: (v["pd"], v["pfa"]) for k, v in res.items() if k in ("0.01", "0.001")}, flush=True)
    (OUT / "summary.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
