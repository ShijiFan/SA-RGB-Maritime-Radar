"""RadarGrid-CS: SA-RGB rendering, cell-scoring network, and clutter-map tail CFAR (Phase C pre-registration).

Rendering follows the paper3 address map at 16-px macro-cells (8-px slots); blocks are constant, so this is lossless
relative to the 100-px rendering.
"""
import numpy as np
import torch
from torch import nn

FEATURES = ["Hurst", "RAA", "RDPH", "RVE", "RI", "NR", "MCS"]
ANCHOR = 128
# slot -> (feature or None) per RGB channel; slots ordered s1 (top-left), s2 (top-right), s3 (bottom-left), s4
SLOT_MAP = {"G7": [("RAA", "RDPH", "RVE"), ("Hurst", "RI", "NR"), ("MCS", None, None), (None, None, None)],
            "G3": [("RAA", "RDPH", "RVE"), (None, None, None), (None, None, None), (None, None, None)]}
CELL_PX, SLOT_PX = 16, 8


def fit_percentiles(feats):
    """feats: (n, 7) raw training values -> (lo, hi) 1st/99th percentiles per feature."""
    return np.percentile(feats, 1, axis=0), np.percentile(feats, 99, axis=0)


def quantize(X, lo, hi):
    """X (..., 7) raw -> uint8 (..., 7)."""
    return np.round(255 * np.clip((X - lo) / (hi - lo + 1e-12), 0, 1)).astype(np.uint8)


def render(Q, group):
    """Q (N, 14, 7) uint8 -> SA-RGB images (N, 3, 32, 112) uint8, odd-even 2x7 layout (cell k: row k%2, col k//2)."""
    N = Q.shape[0]
    img = np.full((N, 3, 2 * CELL_PX, 7 * CELL_PX), ANCHOR, np.uint8)
    for k in range(14):
        r, c = k % 2, k // 2
        for s, chans in enumerate(SLOT_MAP[group]):
            sr, sc = s // 2, s % 2
            y0, x0 = r * CELL_PX + sr * SLOT_PX, c * CELL_PX + sc * SLOT_PX
            for ch, f in enumerate(chans):
                if f is not None:
                    img[:, ch, y0:y0 + SLOT_PX, x0:x0 + SLOT_PX] = Q[:, k, FEATURES.index(f)][:, None, None]
    return img


# 2x7 grid index of range cell k and the 8 neighbour offsets of the 3x3 cell window
GRID_RC = [(k % 2, k // 2) for k in range(14)]
OFFS = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]


class SlotReadout(nn.Module):
    """SAFR lift + slot averaging -> (B, 12*3, 2, 7): 12 addresses x {I, I-a, |I-a|} per macro-cell."""

    def forward(self, x):  # x (B,3,32,112) in [0,1]
        a = ANCHOR / 255
        lift = torch.cat([x, x - a, (x - a).abs()], 1)  # (B,9,32,112)
        s = nn.functional.avg_pool2d(lift, SLOT_PX)  # (B,9,4,14): slot grid
        B = s.shape[0]
        s = s.view(B, 9, 2, 2, 7, 2).permute(0, 1, 3, 5, 2, 4)  # (B,9,sr,sc,row,col)
        return s.reshape(B, 9 * 4, 2, 7)  # address-major channels


class SlotClutterNorm(nn.Module):
    """Per-frame, per-address normalization across the 14 cells: (x - median) / IQR; zero-IQR addresses -> 0."""

    def forward(self, z):
        B, C, H, W = z.shape
        f = z.reshape(B, C, H * W)
        med = f.median(-1, keepdim=True).values
        q = torch.quantile(f, torch.tensor([0.25, 0.75], device=z.device), dim=-1, keepdim=True)
        iqr = (q[1] - q[0])
        out = torch.where(iqr > 1e-6, (f - med) / (iqr + 1e-6), torch.zeros_like(f))
        return out.view(B, C, H, W)


class RelativeContext(nn.Module):
    """Differences to the 8 neighbours in the 3x3 cell window, validity masks, and peak margin."""

    def __init__(self, absolute=False):
        super().__init__()
        self.absolute = absolute

    def forward(self, z):
        B, C, H, W = z.shape
        pad = nn.functional.pad(z, (1, 1, 1, 1), value=float("nan"))
        feats, masks, nbs = [z], [], []
        for dr, dc in OFFS:
            nb = pad[:, :, 1 + dr:1 + dr + H, 1 + dc:1 + dc + W]
            valid = ~torch.isnan(nb[:, :1])
            nbv = torch.nan_to_num(nb, nan=0.0)
            feats.append(nbv if self.absolute else torch.where(valid, z - nbv, torch.zeros_like(z)))
            masks.append(valid.float())
            nbs.append(torch.where(valid, nbv, torch.full_like(nbv, -1e6)))
        if not self.absolute:
            feats.append(z - torch.stack(nbs, 0).max(0).values)
        return torch.cat(feats + masks, 1)


class RadarGridCS(nn.Module):
    def __init__(self, in_addr=36, scn=True, absolute_ctx=False, proj7to3=False, hidden=64):
        super().__init__()
        self.proj = nn.Conv2d(7, 3, 1, bias=False) if proj7to3 else None
        self.scn = SlotClutterNorm() if scn else nn.Identity()
        self.ctx = RelativeContext(absolute_ctx)
        n = in_addr * (9 if absolute_ctx else 10) + 8
        self.head = nn.Sequential(nn.Conv2d(n, hidden, 1), nn.SiLU(), nn.Dropout(0.1),
                                  nn.Conv2d(hidden, hidden, 1), nn.SiLU(), nn.Dropout(0.1), nn.Conv2d(hidden, 1, 1))
        self.readout = SlotReadout()

    def forward(self, x, vector=False):
        """x: SA-RGB image (B,3,32,112) in [0,1], or with vector=True a cell grid (B,K,2,7)."""
        if vector:  # D7 / P7: quantized values in [0,1] on the cell grid, same SAFR lift after optional projection
            z = self.proj(x) if self.proj is not None else x
            a = ANCHOR / 255
            z = torch.cat([z, z - a, (z - a).abs()], 1)
        else:
            z = self.readout(x)
        z = self.scn(z)
        return self.head(self.ctx(z)).squeeze(1)  # (B,2,7) logits

    @staticmethod
    def to_range(logits):
        """(B,2,7) grid -> (B,14) in range-cell order (cell k at row k%2, col k//2)."""
        return logits.permute(0, 2, 1).reshape(logits.shape[0], 14)


def cells_to_grid(V):
    """(N,14,K) range order -> (N,K,2,7) grid."""
    N, _, K = V.shape
    return V.reshape(N, 7, 2, K).transpose(0, 3, 2, 1)


# ---------------- clutter-map exponential-tail CFAR (texp64) ----------------
def tail_params(S, W=64):
    """S (n_seq, F, 14) logits -> per-frame (u, beta) from the previous W frames' non-argmax cells."""
    C = np.arange(S.shape[-1])
    ref = np.where(C[None, None] == S.argmax(-1)[..., None], np.nan, S)
    u = np.full(S.shape[:2], np.nan); b = np.full(S.shape[:2], np.nan)
    for f in range(1, S.shape[1]):
        R = ref[:, max(0, f - W):f].reshape(S.shape[0], -1)
        uu = np.nanquantile(R, 0.9, axis=1)
        u[:, f] = uu
        b[:, f] = np.nanmean(np.where(R > uu[:, None], R - uu[:, None], np.nan), axis=1)
    return u, b


def texp_tau(u, b, a, c):
    return u + b * np.log(0.1 / (c * a))


def h0_rate(S, tau, t):
    """S (n_seq,F,14), tau (n_seq,F), t (n_seq,) target index -> H0 activation rate over finite-threshold frames."""
    C = np.arange(14)
    h0 = C[None, None] != t[:, None, None]
    act = (S >= tau[..., None]) & h0
    ok = np.isfinite(tau)[..., None] & h0
    return act[ok].mean()


def calibrate_c(S, t, u, b, a=1e-2):
    """Largest c (log bisection) with validation H0 rate <= a."""
    lo, hi = 1e-3, 1e3
    for _ in range(40):
        mid = np.sqrt(lo * hi)
        lo, hi = (mid, hi) if h0_rate(S, texp_tau(u, b, a, mid), t) <= a else (lo, mid)
    return lo
