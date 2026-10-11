"""Round-2 review: unified-mask rescoring of the final table, HGB K = 4 with saved scores, 1998 record x seed table,
and the simple temporal-mean control S (rules in D_PLAN.md). No network training."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from n1_temporal_net import DEV, TR, VA, hgb_scores, lab, load, loro_split, RECORDS  # noqa: E402
from n3_final import make_feat, windows_1998_rec  # noqa: E402
from rgcs import calibrate_c, tail_params, texp_tau  # noqa: E402
from common import calibrate, pd_at_pfa  # noqa: E402

FIN = HERE / "final"
C = np.arange(14)


def evaluate_all(Sv, tv, St, tt, rec=None, start=1):
    """Every rule on the same decision frames: positions >= start of the given arrays (tail history from earlier ones)."""
    tv, tt = np.broadcast_to(tv, Sv.shape[:1]), np.broadcast_to(tt, St.shape[:1])
    uv, bv = tail_params(Sv); ut, bt = tail_params(St)
    sv, st = Sv[:, start:], St[:, start:]
    c = calibrate_c(sv, tv, uv[:, start:], bv[:, start:], 1e-2)
    h0 = C[None, None] != tt[:, None, None]; adj = np.abs(C[None, None] - tt[:, None, None]) == 1
    bg = np.abs(C[None, None] - tt[:, None, None]) > 1
    h1 = np.take_along_axis(st, np.broadcast_to(tt[:, None, None], st.shape[:2] + (1,)), 2)[..., 0]
    h0v = sv[np.broadcast_to(C[None, None] != tv[:, None, None], sv.shape)]
    res, per_rec = {}, []
    for a in (1e-2, 1e-3):
        for rule, tau in (("texp", texp_tau(ut[:, start:], bt[:, start:], a, c)), ("fixed", np.full(st.shape[:2], calibrate(h0v, a)))):
            act = st >= tau[..., None]; B = lambda m: np.broadcast_to(m, act.shape)
            res[f"{rule}_{a:g}"] = dict(pd=float((h1 >= tau).mean()), pfa=float(act[B(h0)].mean()),
                                         adj=float(act[B(adj)].mean()), bg=float(act[B(bg)].mean()))
            if rec is not None and rule == "texp" and a == 1e-2:
                for r in np.unique(rec):
                    q = rec == r
                    per_rec.append(dict(record=int(r), pd=float((h1[q] >= tau[q]).mean()),
                                        pfa=float(act[q][np.broadcast_to(h0[q], act[q].shape)].mean())))
        res[f"oracle_{a:g}"] = dict(pd=pd_at_pfa(h1.ravel(), st[np.broadcast_to(h0, st.shape)], a))
    res["n_frames"] = int(st.shape[0] * st.shape[1]); res["n_h0"] = int(np.broadcast_to(h0, st.shape).sum())
    return res, per_rec


def flat(res, **kw):
    rows = []
    for k, v in res.items():
        if isinstance(v, dict):
            rule, a = k.rsplit("_", 1)
            rows.append(dict(kw, rule=rule, alpha=float(a), **v, n_frames=res["n_frames"], n_h0=res["n_h0"]))
    return rows


def mean4(S):
    return np.stack([S[:, i:S.shape[1] - 3 + i] for i in range(4)]).mean(0)  # aligned to positions 3..


def main():
    rows, recs = [], []
    # ---- saved network scores, unified mask (decision frames 16-255 / 8-116)
    for arm in ("G7", "G3"):
        for K in (1, 4):
            for k in RECORDS:
                for s in (0, 1, 2):
                    z = np.load(FIN / f"{arm}_K{K}_f{k:02d}_s{s}_scores.npz")
                    r, _ = evaluate_all(z["val"], int(z["val_t"]), z["test"], int(z["test_t"]))
                    rows += flat(r, scorer=f"net-{arm}", K=K, protocol="loro", fold=k, seed=s)
            for s in (0, 1, 2):
                z = np.load(FIN / f"{arm}_K{K}_1998_s{s}_scores.npz")
                r, pr = evaluate_all(z["val"], z["val_t"], z["test"], z["test_t"], rec=z["rec"])
                rows += flat(r, scorer=f"net-{arm}", K=K, protocol="1998", fold=0, seed=s)
                recs += [dict(p, scorer=f"net-{arm}", K=K, seed=s) for p in pr]
    print("network rescored", flush=True)
    # ---- simple temporal mean control S (G7): truncate every series to start at frame 18 (1993) / 10 (1998)
    for k in RECORDS:
        for s in (0, 1, 2):
            z1 = np.load(FIN / f"G7_K1_f{k:02d}_s{s}_scores.npz"); z4 = np.load(FIN / f"G7_K4_f{k:02d}_s{s}_scores.npz")
            for name, Sv, St in (("single", z1["val"][:, 3:], z1["test"][:, 3:]), ("mean4", mean4(z1["val"]), mean4(z1["test"])),
                                 ("net-K4", z4["val"][:, 3:], z4["test"][:, 3:])):
                r, _ = evaluate_all(Sv, int(z1["val_t"]), St, int(z1["test_t"]))
                rows += flat(r, scorer=f"S-{name}", K=4, protocol="loro", fold=k, seed=s)
    for s in (0, 1, 2):
        z1 = np.load(FIN / f"G7_K1_1998_s{s}_scores.npz"); z4 = np.load(FIN / f"G7_K4_1998_s{s}_scores.npz")
        for name, Sv, St in (("single", z1["val"][:, 3:], z1["test"][:, 3:]), ("mean4", mean4(z1["val"]), mean4(z1["test"])),
                             ("net-K4", z4["val"][:, 3:], z4["test"][:, 3:])):
            r, _ = evaluate_all(Sv, z1["val_t"], St, z1["test_t"])
            rows += flat(r, scorer=f"S-{name}", K=4, protocol="1998", fold=0, seed=s)
    print("control S done", flush=True)
    pd.DataFrame(rows).to_csv(HERE / "R2_rescored.csv", index=False)
    pd.DataFrame(recs).to_csv(HERE / "R2_1998_record_seed.csv", index=False)
    # ---- HGB K = 4 on the same G7 tensor, scores saved
    data = load(); feat = make_feat("G7"); n_tr, n_ev = np.arange(7, 256), np.arange(15, 256)
    from rgcs import fit_percentiles
    for k in RECORDS:
        tr, va, te = loro_split(k)
        lo, hi = fit_percentiles(np.concatenate([data[g]["X"].reshape(-1, 7) for g in tr]))
        F = {g: feat(data[g]["X"], lo, hi) for g in (*tr, va, te)}
        Ftr = torch.cat([F[g] for g in tr]); ytr = lab(np.repeat([data[g]["t"] for g in tr], 4), len(tr) * 4)
        sv, st = hgb_scores(Ftr, n_tr, ytr, ((F[va], n_ev), (F[te], n_ev)), 4)
        np.savez_compressed(FIN / f"HGB_K4_f{k:02d}_scores.npz", val=sv, test=st, val_t=data[va]["t"], test_t=data[te]["t"])
        r, _ = evaluate_all(sv, data[va]["t"], st, data[te]["t"])
        rows += flat(r, scorer="hgb-G7", K=4, protocol="loro", fold=k, seed=0)
        print("HGB fold", k, flush=True)
    lo, hi = fit_percentiles(np.concatenate([data[g]["X"][:, :TR].reshape(-1, 7) for g in RECORDS]))
    F = {g: feat(data[g]["X"], lo, hi) for g in RECORDS}
    Ftr = torch.cat([F[g] for g in RECORDS]); ytr = lab(np.repeat([data[g]["t"] for g in RECORDS], 4), 40)
    X98, T98, R98 = windows_1998_rec()
    F98 = torch.cat([feat(X98[i:i + 32], lo, hi) for i in range(0, len(X98), 32)])
    Tv = np.repeat([data[g]["t"] for g in RECORDS], 4)
    sv, st = hgb_scores(Ftr, np.arange(7, TR), ytr, ((Ftr, np.arange(TR, TR + VA)), (F98, np.arange(7, 117))), 4)
    np.savez_compressed(FIN / "HGB_K4_1998_scores.npz", val=sv, test=st, val_t=Tv, test_t=T98, rec=R98)
    r, pr = evaluate_all(sv, Tv, st, T98, rec=R98)
    rows += flat(r, scorer="hgb-G7", K=4, protocol="1998", fold=0, seed=0)
    recs += [dict(p, scorer="hgb-G7", K=4, seed=0) for p in pr]
    pd.DataFrame(rows).to_csv(HERE / "R2_rescored.csv", index=False)
    pd.DataFrame(recs).to_csv(HERE / "R2_1998_record_seed.csv", index=False)
    print("all done")


if __name__ == "__main__":
    main()
