"""N1: temporal RadarGrid-CS (SA-RGB frame sequence -> range-equivariant cell scores); design and rules in D_PLAN.md."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "phaseA_20261008"))
sys.path.insert(0, str(HERE.parent / "phaseC"))
from baselines_ref import evaluate, windows_1998  # noqa: E402
from common import RECORDS, load, loro_split  # noqa: E402
from explore_context import build  # noqa: E402
from matched_inputs import tensor  # noqa: E402
from rgcs import fit_percentiles, quantize  # noqa: E402

HP = dict(epochs=40, batch=256, lr=1e-3, wd=1e-4, pos_weight=13.0)
KS = (1, 4, 8)
FMIN, N0, N0_98, TR, VA = 7, 15, 7, 181, 45
DEV = "cuda"


class TemporalCellNet(nn.Module):
    def __init__(self, d_in=72, h=64):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(d_in, h), nn.SiLU(), nn.Dropout(0.1), nn.Linear(h, h), nn.SiLU())
        self.psi = nn.Sequential(nn.Dropout(0.1), nn.Linear(3 * h, h), nn.SiLU(), nn.Linear(h, 1))

    def forward(self, x):  # x (B, K, 14, 72) -> (B, 14)
        e = self.phi(x)
        z = torch.cat([e.mean(1), e.std(1, unbiased=False), e.amin(1)], -1)
        return self.psi(z).squeeze(-1)


def windows(F, n_idx, K):
    """F (S, Fr, 14, D) tensor; returns (S*len(n_idx), K, 14, D) causal windows ending at n_idx."""
    idx = torch.as_tensor(n_idx)[:, None] + torch.arange(-K + 1, 1)[None]  # (N, K)
    return F[:, idx].reshape(-1, K, *F.shape[2:])


def fit_net(seed, K, Ftr, ntr, ytr, Fv, nv, yv):
    torch.manual_seed(seed); np.random.seed(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    m = TemporalCellNet().to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=HP["lr"], weight_decay=HP["wd"])
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(HP["pos_weight"], device=DEV))
    # training samples: (sequence, frame) pairs
    S, N = Ftr.shape[0], len(ntr)
    seq = torch.arange(S).repeat_interleave(N); fr = torch.as_tensor(ntr).repeat(S)
    lab = ytr[seq]  # (S*N, 14)
    off = torch.arange(-K + 1, 1)
    Xv = windows(Fv, nv, K); Yv = yv.repeat_interleave(len(nv), 0)
    g = torch.Generator().manual_seed(seed); best = np.inf
    for _ in range(HP["epochs"]):
        m.train(); perm = torch.randperm(len(seq), generator=g)
        for i in range(0, len(seq), HP["batch"]):
            b = perm[i:i + HP["batch"]]
            xb = Ftr[seq[b][:, None], fr[b][:, None] + off[None]]  # (B, K, 14, D)
            loss = lossf(m(xb), lab[b])
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            lv = float(sum(lossf(m(Xv[i:i + 4096]), Yv[i:i + 4096]) * len(Xv[i:i + 4096]) for i in range(0, len(Xv), 4096)) / len(Xv))
        if lv < best:
            best, state = lv, {k: v.detach().clone() for k, v in m.state_dict().items()}
    m.load_state_dict(state); m.eval()
    return m


def net_scores(m, F, n_idx, K):
    with torch.no_grad():
        X = windows(F, n_idx, K)
        s = torch.cat([m(X[i:i + 8192]) for i in range(0, len(X), 8192)])
    return s.reshape(F.shape[0], len(n_idx), 14).cpu().numpy()


def hgb_feats(F, n_idx, K):
    W = windows(F, n_idx, K)
    z = torch.cat([W.mean(1), W.std(1, unbiased=False), W.amin(1)], -1)
    return z.reshape(F.shape[0], len(n_idx), 14, -1).cpu().numpy()


def hgb_scores(Ftr, ntr, ytr, others, K):
    Z = hgb_feats(Ftr, ntr, K)
    y = np.broadcast_to(ytr.cpu().numpy()[:, None], Z.shape[:3])
    clf = HistGradientBoostingClassifier(max_iter=200, class_weight="balanced", random_state=0)
    clf.fit(Z.reshape(-1, Z.shape[-1]), y.ravel())
    out = []
    for F, n_idx in others:
        Zo = hgb_feats(F, n_idx, K)
        p = np.clip(clf.predict_proba(Zo.reshape(-1, Zo.shape[-1]))[:, 1], 1e-12, 1 - 1e-12)
        out.append(np.log(p / (1 - p)).reshape(Zo.shape[:3]))
    return out


def lab(t, n):
    return torch.as_tensor(np.broadcast_to(np.arange(14) == np.asarray(t)[:, None], (n, 14)).astype(np.float32), device=DEV)


def feat_front():
    return build("margin1", "G7").eval()  # parameter-free readout, SCN and +-1 range peak margin


def main():
    data = load(); front = feat_front(); rows = []
    feat = lambda X, lo, hi: torch.as_tensor(tensor(front, quantize(X.reshape(-1, 14, 7), lo, hi)).reshape(*X.shape[:3], 72), device=DEV)
    n_tr, n_ev = np.arange(FMIN, 256), np.arange(N0, 256)
    for k in RECORDS:
        tr, va, te = loro_split(k)
        lo, hi = fit_percentiles(np.concatenate([data[g]["X"].reshape(-1, 7) for g in tr]))
        F = {g: feat(data[g]["X"], lo, hi) for g in (*tr, va, te)}
        Ftr = torch.cat([F[g] for g in tr]); ytr = lab(np.repeat([data[g]["t"] for g in tr], 4), len(tr) * 4)
        yv = lab(np.full(4, data[va]["t"]), 4)
        tv, tt = np.full(4, data[va]["t"]), np.full(4, data[te]["t"])
        for K in KS:
            t0 = time.time()
            sv, st = hgb_scores(Ftr, n_tr, ytr, ((F[va], n_ev), (F[te], n_ev)), K)
            for (rule, a), v in evaluate(sv, tv, st, tt).items():
                rows.append(dict(scorer="HGB", K=K, protocol="loro", fold=k, seed=0, rule=rule, alpha=a, **v))
            for s in (0, 1, 2):
                m = fit_net(s, K, Ftr, n_tr, ytr, F[va], n_ev, yv)
                for (rule, a), v in evaluate(net_scores(m, F[va], n_ev, K), tv, net_scores(m, F[te], n_ev, K), tt).items():
                    rows.append(dict(scorer="network", K=K, protocol="loro", fold=k, seed=s, rule=rule, alpha=a, **v))
                if s == 0 and K == 4:
                    torch.save(m.state_dict(), HERE / "ckpt" / f"tnet_K4_f{k:02d}_s0.pt")
            print(f"fold {k} K={K} ({time.time() - t0:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(HERE / "N1_temporal_net.csv", index=False)
    # ---- IPIX1993 -> IPIX1998
    lo, hi = fit_percentiles(np.concatenate([data[g]["X"][:, :TR].reshape(-1, 7) for g in RECORDS]))
    F = {g: feat(data[g]["X"], lo, hi) for g in RECORDS}
    Ftr = torch.cat([F[g] for g in RECORDS]); ytr = lab(np.repeat([data[g]["t"] for g in RECORDS], 4), 40)
    n_tr98, n_v98 = np.arange(FMIN, TR), np.arange(TR, TR + VA)
    X98, T98 = windows_1998()
    F98 = torch.cat([feat(X98[i:i + 32], lo, hi) for i in range(0, len(X98), 32)])
    n_98 = np.arange(N0_98, 117)
    Tv = np.repeat([data[g]["t"] for g in RECORDS], 4)
    for K in KS:
        sv, st = hgb_scores(Ftr, n_tr98, ytr, ((Ftr, n_v98), (F98, n_98)), K)
        for (rule, a), v in evaluate(sv, Tv, st, T98).items():
            rows.append(dict(scorer="HGB", K=K, protocol="1998", fold=0, seed=0, rule=rule, alpha=a, **v))
        for s in (0, 1, 2):
            m = fit_net(s, K, Ftr, n_tr98, ytr, Ftr, n_v98, ytr)
            for (rule, a), v in evaluate(net_scores(m, Ftr, n_v98, K), Tv, net_scores(m, F98, n_98, K), T98).items():
                rows.append(dict(scorer="network", K=K, protocol="1998", fold=0, seed=s, rule=rule, alpha=a, **v))
        print(f"1998 K={K} done", flush=True)
        pd.DataFrame(rows).to_csv(HERE / "N1_temporal_net.csv", index=False)


if __name__ == "__main__":
    (HERE / "ckpt").mkdir(exist_ok=True)
    main()
