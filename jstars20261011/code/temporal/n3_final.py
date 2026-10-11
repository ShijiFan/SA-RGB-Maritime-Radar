"""N3: final detector (range-ordered front end + temporal pooling), G7 vs G3, K in {1, 4}; scores, checkpoints, per-record
IPIX1998 results and latency. Rule F-a in D_PLAN.md."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from n1_temporal_net import (DEV, TR, VA, TemporalCellNet, evaluate, fit_net, lab, load, loro_split,  # noqa: E402
                             net_scores, windows, RECORDS)
import run_loro  # noqa: E402
from explore_context import build  # noqa: E402
from rgcs import fit_percentiles, quantize  # noqa: E402

OUT = HERE / "final"; OUT.mkdir(exist_ok=True)
C = np.arange(14)


def windows_1998_rec():
    d = np.load(HERE.parent / "phaseA_20261008" / "out" / "ipix1998_7feat.npz")
    meta = json.load(open(HERE.parent / "phaseA_20261008" / "out" / "ipix1998_7feat.json", encoding="utf8"))
    X, T, R = [], [], []
    for r in meta["records"]:
        for p in ("HH", "HV", "VH", "VV"):
            k = f"r{r['rec']:02d}_{p}"
            for j, s in enumerate(d[k + "_starts"]):
                X.append(d[k + "_win"][:, j].astype(np.float64)); T.append(r["target_cell"] - 1 - int(s)); R.append(r["rec"])
    return np.stack(X), np.array(T), np.array(R)


def make_feat(arm):
    front = build("margin1", arm).eval()

    def feat(X, lo, hi):
        Q = quantize(X.reshape(-1, 14, 7), lo, hi)
        with torch.no_grad():
            f = front.ctx(front.scn(front.readout(run_loro.make_inputs(arm, Q))))
        f = f.permute(0, 3, 2, 1).reshape(f.shape[0], 14, -1).numpy()
        return torch.as_tensor(f.reshape(*X.shape[:3], 72), device=DEV)
    return feat


def main():
    data = load(); rows = []
    n_tr, n_ev = np.arange(7, 256), np.arange(15, 256)
    X98, T98, R98 = windows_1998_rec()
    for arm in ("G7", "G3"):
        feat = make_feat(arm)
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
                    np.savez_compressed(OUT / f"{arm}_K{K}_f{k:02d}_s{s}_scores.npz", val=Sv, test=St, val_t=data[va]["t"], test_t=data[te]["t"])
                    torch.save(m.state_dict(), OUT / f"{arm}_K{K}_f{k:02d}_s{s}.pt")
                    for (rule, a), v in evaluate(Sv, np.full(4, data[va]["t"]), St, np.full(4, data[te]["t"])).items():
                        rows.append(dict(arm=arm, K=K, protocol="loro", fold=k, seed=s, rule=rule, alpha=a, **v))
            print(arm, "fold", k, flush=True)
        lo, hi = fit_percentiles(np.concatenate([data[g]["X"][:, :TR].reshape(-1, 7) for g in RECORDS]))
        F = {g: feat(data[g]["X"], lo, hi) for g in RECORDS}
        Ftr = torch.cat([F[g] for g in RECORDS]); ytr = lab(np.repeat([data[g]["t"] for g in RECORDS], 4), 40)
        F98 = torch.cat([feat(X98[i:i + 32], lo, hi) for i in range(0, len(X98), 32)])
        Tv = np.repeat([data[g]["t"] for g in RECORDS], 4)
        n98 = np.arange(7, 117)
        for K in (1, 4):
            for s in (0, 1, 2):
                m = fit_net(s, K, Ftr, np.arange(7, TR), ytr, Ftr, np.arange(TR, TR + VA), ytr)
                Sv, St = net_scores(m, Ftr, np.arange(TR, TR + VA), K), net_scores(m, F98, n98, K)
                np.savez_compressed(OUT / f"{arm}_K{K}_1998_s{s}_scores.npz", val=Sv, test=St, val_t=Tv, test_t=T98, rec=R98)
                torch.save(m.state_dict(), OUT / f"{arm}_K{K}_1998_s{s}.pt")
                for (rule, a), v in evaluate(Sv, Tv, St, T98).items():
                    rows.append(dict(arm=arm, K=K, protocol="1998", fold=0, seed=s, rule=rule, alpha=a, **v))
                for r in np.unique(R98):  # per record, tail rule (thresholds from each window's own history)
                    sel = R98 == r
                    v = evaluate(Sv, Tv, St[sel], T98[sel])[("texp", 1e-2)]
                    rows.append(dict(arm=arm, K=K, protocol="1998rec", fold=int(r), seed=s, rule="texp", alpha=1e-2, **v))
        print(arm, "1998 done", flush=True)
        pd.DataFrame(rows).to_csv(HERE / "N3_final.csv", index=False)
    # latency: one frame update of the K = 4 detector (batch 1 = one frame of 14 cells with its 4-frame window)
    m = TemporalCellNet().eval(); x = torch.randn(1, 4, 14, 72); lat = {}
    for dev in ("cpu", "cuda"):
        mm, xx = m.to(dev), x.to(dev); ts = []
        with torch.no_grad():
            for i in range(1100):
                if dev == "cuda": torch.cuda.synchronize()
                t0 = time.perf_counter(); mm(xx)
                if dev == "cuda": torch.cuda.synchronize()
                if i >= 100: ts.append((time.perf_counter() - t0) * 1e3)
        lat[dev] = dict(median=float(np.median(ts)), p95=float(np.percentile(ts, 95)))
    lat["params"] = sum(p.numel() for p in m.parameters())
    (HERE / "N3_latency.json").write_text(json.dumps(lat, indent=1)); print(lat)


if __name__ == "__main__":
    main()
