"""Figures for the JSTARS rewrite (style: fig_style -> tools/taes_r5_style.py, enlarged Times lettering).

fig_chain      Fig. 1   detection chain: slot-addressable grid and RadarGrid-CS
fig_data       Fig. 2   real grids, descriptor range profiles, record drift before/after per-frame normalization
fig_protocol   Fig. 3   record-disjoint folds and the clutter-map tail CFAR (real scores)
fig_diagnosis  Fig. 4   information vs controllability by descriptor count
fig_anatomy    Fig. 5   descriptor AUC by site, adjacent-cell false alarms, Pd vs SCR (position prior)
fig_loro       Fig. 6   per-fold LORO Pfa and Pd
fig_internals  Fig. 7   inside RadarGrid-CS on held-out records (input, SCN, peak margin, G7/G3 scores, decisions)
fig_ablation   Fig. 8   arms of the fixed protocol
fig_crosssite  Fig. 9   IPIX1998 per record and context variants
fig_shift      Fig. 10  range-shift test
Every value is read from phaseA_20261008/out or phaseC.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Patch, Rectangle
from matplotlib.ticker import FixedLocator, NullFormatter

from fig_style import (BLACK, CORAL, CORAL_E, FS, FS_S, SAGE, SAND, SAND_E, SLATE, TEAL, TEAL_E, TINT, W2,
                       panel_labels, plt, save)

ROOT = Path(__file__).resolve().parents[1]
A, C = ROOT / "phaseA_20261008" / "out", ROOT / "phaseC"
sys.path.insert(0, str(C)); sys.path.insert(0, str(ROOT / "phaseA_20261008"))
FEATS = ["Hurst", "RAA", "RDPH", "RVE", "RI", "NR", "MCS"]
ALPHA, FLOOR = 1e-2, 1e-4
UNSEEN = "#9A3A22"
PEACH, LAV, SKY, MINT, CREAM, CORAL_T = "#FCEBE2", "#ECEAF4", "#E6EFF7", "#E5EFE0", "#F5F4EA", "#FBE3DA"
DIV = LinearSegmentedColormap.from_list("div", [TEAL_E, "#9FD3CB", "#F7F7F5", "#F2B6A1", CORAL_E])
SEQ = LinearSegmentedColormap.from_list("seq", ["#F7F7F5", "#E9C46A", CORAL, "#7A2614"])
ARM = {  # key -> (label, colour, marker, edge)
    "G7": ("G7, tail CFAR", CORAL, "o", CORAL_E),
    "G3": ("G3, tail CFAR", SLATE, "s", SLATE),
    "G7fix": ("G7, fixed threshold", TEAL, "^", TEAL_E),
    "hull": ("Convex hull [5]", SAND, "D", SAND_E),
    "hgb": ("HGB-G7, same descriptors", SAND, "D", SAND_E),
}


# ------------------------------------------------------------------ plot helpers
def log_pfa(ax, hi=0.3, zero=True, lo=FLOOR):
    ax.set_yscale("log"); ax.set_ylim(lo * 0.7, hi)
    ticks, labels = [1e-3, 1e-2, 1e-1], [r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$"]
    if zero:
        ticks, labels = [lo] + ticks, ["0"] + labels
    ax.yaxis.set_major_locator(FixedLocator(ticks)); ax.set_yticklabels(labels)
    ax.yaxis.set_minor_locator(FixedLocator([m * 10.0 ** e for e in (-3, -2, -1) for m in range(2, 10) if m * 10.0 ** e < hi]))
    ax.yaxis.set_minor_formatter(NullFormatter()); ax.grid(which="minor", visible=False)
    if zero:
        yb = np.sqrt(lo * 1e-3); tr = ax.get_yaxis_transform()
        for x in (0, 1):
            ax.plot([x, x], [yb / 1.5, yb * 1.5], transform=tr, color="white", linewidth=2.2, clip_on=False, zorder=6,
                    solid_capstyle="butt")
            for f in (1.5, 1 / 1.5):
                ax.plot([x - 0.012, x + 0.012], [yb / f / 1.3, yb / f * 1.3], transform=tr, color=BLACK,
                        linewidth=0.9, clip_on=False, zorder=7)


def budget(ax, text_x=None, ha="left"):
    ax.axhline(ALPHA, color=BLACK, linestyle="--", linewidth=1.0, zorder=2)
    ax.axhline(2 * ALPHA, color=BLACK, linestyle="-.", linewidth=1.0, zorder=2)
    if text_x is not None:
        log = ax.get_yscale() == "log"
        ax.text(text_x, ALPHA * 0.8 if log else ALPHA - 0.0012, r"$\alpha$", ha=ha, va="top", fontsize=FS)
        ax.text(text_x, 2 * ALPHA * 1.12 if log else 2 * ALPHA + 0.0012, r"$2\alpha$", ha=ha, va="bottom", fontsize=FS)


def below(fig, axes, gap=0.012):
    fig.canvas.draw(); r = fig.canvas.get_renderer(); inv = fig.transFigure.inverted()
    return min(a.get_tightbbox(r).transformed(inv).y0 for a in axes) - gap


def handle(key, line=True, label=None):
    lab, col, mk, ec = ARM[key]
    return Line2D([], [], color=col, marker=mk, markersize=6, markeredgecolor=ec, linestyle="-" if line else "none",
                  linewidth=1.5, label=label or lab)


def shade(ax, xs, text, width=0.5):
    for f in xs:
        ax.axvspan(f - width, f + width, color=TINT, zorder=0, linewidth=0)
        if text:
            ax.text(f, 1.015, text, ha="center", va="bottom", fontsize=FS_S - 0.5, color=UNSEEN, linespacing=0.9,
                    transform=ax.get_xaxis_transform())


# ------------------------------------------------------------------ schematic primitives (tools/taes_r5_schematic.py)
def box(ax, x, y, w, h, text="", fs=FS_S, fc="white", ec=BLACK, lw=0.8, ls="-", r=0.05, **kw):
    r = min(r, w / 2.5, h / 2.5) if r else 0
    patch = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", facecolor=fc, edgecolor=ec,
                           linewidth=lw, linestyle=ls, zorder=2) if r else \
        Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=2)
    ax.add_patch(patch)
    if text:
        kw.setdefault("linespacing", 1.15)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=3, **kw)


def frame(ax, x, y, w, h, title, ec="#AEBB8A", color=SLATE, fc=CREAM, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.1", facecolor=fc, edgecolor=ec,
                                linewidth=1.0, linestyle=ls, zorder=0.5))
    ax.text(x + 0.1, y + h - 0.07, title, ha="left", va="top", fontsize=FS, color=color, fontweight="bold")


def arrow(ax, p, q, lw=0.9, ls="-", color=BLACK):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, color=color, linewidth=lw, linestyle=ls,
                                 shrinkA=0, shrinkB=0, zorder=4))


def line(ax, xs, ys, lw=0.8, ls="-", color=BLACK):
    ax.plot(xs, ys, color=color, linewidth=lw, linestyle=ls, zorder=4, solid_capstyle="butt")


def slot_grid(ax, img, x0, y0, cw, target=None, lw_t=1.8):
    """Draw a rendered SA-RGB image (3,32,112) as 2x7 cells of 2x2 slots at (x0, y0), cell width cw."""
    for k in range(14):
        r, c = k % 2, k // 2
        for s in range(4):
            sr, sc = s // 2, s % 2
            col = img[:, r * 16 + sr * 8 + 4, c * 16 + sc * 8 + 4] / 255
            ax.add_patch(Rectangle((x0 + c * cw + sc * cw / 2, y0 + (1 - r) * cw + (1 - sr) * cw / 2), cw / 2, cw / 2,
                                   facecolor=col, edgecolor="none", zorder=2))
        ax.add_patch(Rectangle((x0 + c * cw, y0 + (1 - r) * cw), cw, cw, fill=False, edgecolor="white", linewidth=0.8,
                               zorder=3))
    if target is not None:
        ax.add_patch(Rectangle((x0 + (target // 2) * cw, y0 + (1 - target % 2) * cw), cw, cw, fill=False,
                               edgecolor=CORAL, linewidth=lw_t, zorder=5))


def rendered(data, rec, pol=0, frame=100, train=range(2, 10), group="G7"):
    from rgcs import fit_percentiles, quantize, render
    lo, hi = fit_percentiles(np.concatenate([data[g]["X"].reshape(-1, 7) for g in train]))
    return render(quantize(data[rec]["X"][pol, frame][None], lo, hi), group)[0]


# ------------------------------------------------------------------ Fig. 1: detection chain
def fig_chain():
    from common import load
    data = load()
    img = rendered(data, 5); t5 = data[5]["t"]
    H = 3.35
    fig = plt.figure(figsize=(W2, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W2); ax.set_ylim(0, H); ax.axis("off")
    frame(ax, 0.05, 1.62, 7.06, 1.68, "Slot-addressable descriptor grid")
    yc = 2.4
    ex, ey, ew, eh = 0.35, yc - 0.33, 0.66, 0.62
    ax.add_patch(Rectangle((ex, ey), ew, eh, facecolor=SKY, edgecolor="none", zorder=1))
    for i in range(8):
        line(ax, [ex + i * ew / 7] * 2, [ey, ey + eh], lw=0.45)
    for j in range(7):
        line(ax, [ex, ex + ew], [ey + j * eh / 6] * 2, lw=0.45)
    ax.text(ex + ew / 2, ey - 0.06, "Range cell", ha="center", va="top", fontsize=FS_S)
    ax.text(ex - 0.06, ey + eh / 2, "Pulse", ha="right", va="center", fontsize=FS_S, rotation=90)
    ax.text(ex + ew / 2, ey + eh + 0.05, "Echoes", ha="center", va="bottom", fontsize=FS)
    arrow(ax, (ex + ew + 0.06, yc), (1.25, yc))
    box(ax, 1.25, yc - 0.42, 1.18, 0.84, "Seven descriptors\nper range cell\n(Hurst, RAA, RDPH,\nRVE, RI, NR, MCS)",
        fs=FS_S, fc=PEACH, ec="#C98B6B")
    arrow(ax, (2.43, yc), (2.6, yc))
    gx, cw = 2.6, 0.25
    slot_grid(ax, img, gx, yc - cw, cw, target=t5)
    ax.text(gx + 3.5 * cw, yc + cw + 0.06, r"$2\times7$ grid (record 5, HH)", ha="center", va="bottom", fontsize=FS_S)
    ax.text(gx + 3.5 * cw, yc - cw - 0.06, "coral: target cell", ha="center", va="top", fontsize=FS_S - 0.5,
            color=CORAL_E)
    zx, zy, zw = 4.42, 1.9, 0.95
    line(ax, [gx + 4 * cw, zx], [yc + cw, zy + zw], lw=0.5, ls=":"); line(ax, [gx + 4 * cw, zx], [yc, zy], lw=0.5, ls=":")
    for s, (lab, fc) in enumerate((("RAA\nRDPH\nRVE", "#F4C9B8"), ("Hurst\nRI\nNR", "#CFE6E1"),
                                   ("MCS\n$a$, $a$", "#F6E3B0"), ("$a$, $a$, $a$", "#E9E9E9"))):
        sr, sc = s // 2, s % 2
        box(ax, zx + sc * zw / 2, zy + (1 - sr) * zw / 2, zw / 2, zw / 2, lab, fs=FS_S - 0.8, fc=fc, ec="white",
            lw=0.8, r=0, linespacing=1.0)
        ax.text(zx + sc * zw / 2 + 0.025, zy + (1 - sr) * zw / 2 + zw / 2 - 0.02, f"$s_{s + 1}$", ha="left", va="top",
                fontsize=FS_S - 1.5, zorder=5, color="#5A5A5A")
    ax.add_patch(Rectangle((zx, zy), zw, zw, fill=False, edgecolor=BLACK, linewidth=0.8, zorder=5))
    ax.text(zx + zw + 0.12, zy + zw / 2, "One cell, four slots:\nslot $\\times$ channel = descriptor,\n"
            "intensity = its value,\n$a$ = empty address", ha="left", va="center", fontsize=FS_S, linespacing=1.2)
    lat = json.loads((ROOT / "phaseD_temporal" / "N3_latency.json").read_text())
    frame(ax, 0.05, 0.05, 7.06, 1.48, f"RadarGrid-CS: {lat['params']:,} parameters, "
          f"encoder and head {lat['cpu']['median']:.2f} ms per update on a CPU", ec=CORAL, color=CORAL_E, fc="#FEF5F1", ls=(0, (4, 2.5)))
    yb = 0.93
    bx, bw, gap = 0.16, 0.86, 0.15
    blocks = [("Slot readout", SKY, "#8FB3D3", "anchor lift,\nmean per slot:\n36 per cell"),
              ("SCN", CORAL_T, CORAL, "each address:\n(value $-$ median)\n/IQR over cells"),
              ("Range\npeak margin", MINT, SAGE, "cell minus max of\n$\\pm1$ range\nneighbors"),
              ("Frame\nencoder $\\phi$", LAV, "#8E86B5", r"$72\to64\to64$" + "\nshared by all\ncells and frames"),
              ("Temporal\npooling", "#FAF0D2", SAND_E, "mean, std, min\nover the last $K$\nframes (causal)"),
              ("Cell head $\\psi$", LAV, "#8E86B5", r"$192\to64\to1$" + "\nno cell index")]
    for i, (t, fc, ec, sub) in enumerate(blocks):
        x = bx + i * (bw + gap)
        hot = t.startswith(("SCN", "Temporal"))
        box(ax, x, yb - 0.24, bw, 0.48, t, fs=FS_S + 0.3, fc=fc, ec=ec, lw=1.2 if hot else 0.8,
            **({"color": "#B2442A"} if hot else {}))
        ax.text(x + bw / 2, yb - 0.32, sub, ha="center", va="top", fontsize=FS_S - 1.0, linespacing=1.1)
        if i:
            arrow(ax, (x - gap, yb), (x, yb))
    sx = bx + 6 * (bw + gap) - gap
    cx = 6.5
    arrow(ax, (sx, yb), (cx - 0.21, yb))
    ax.text((sx + cx - 0.21) / 2, yb + 0.06, r"$s_k$", ha="center", va="bottom", fontsize=FS)
    ax.add_patch(Circle((cx, yb), 0.21, facecolor="#FAF0D2", edgecolor=SAND_E, linewidth=0.9, zorder=2))
    ax.text(cx, yb, r"$\gtrless\tau_n$", ha="center", va="center", fontsize=FS_S, zorder=3)
    ax.text(cx, yb - 0.3, "tail CFAR\n(Fig. 3)", ha="center", va="top", fontsize=FS_S - 1.0, linespacing=1.1)
    arrow(ax, (cx + 0.21, yb), (6.84, yb))
    ax.text(6.86, yb + 0.13, r"$H_1$", ha="left", va="center", fontsize=FS_S)
    ax.text(6.86, yb - 0.13, r"$H_0$", ha="left", va="center", fontsize=FS_S)
    save(fig, "fig_chain")


# ------------------------------------------------------------------ Fig. 2: data and problem
def fig_data():
    from common import frame_normalize, load, masks
    data = load()
    order = [1, 2, 3, 0, 4, 5, 6]
    fig = plt.figure(figsize=(W2, 4.9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.25], height_ratios=[1, 1.05], wspace=0.3, hspace=0.75,
                          left=0.085, right=0.9, top=0.955, bottom=0.13)
    ax_a = fig.add_subplot(gs[0, 0]); ax_a.set_xlim(0, 2.75); ax_a.set_ylim(0, 1.95); ax_a.axis("off")
    for i, rec in enumerate((1, 6, 5)):
        img = rendered(data, rec, frame=120); t = data[rec]["t"]
        y0 = 1.38 - i * 0.66
        slot_grid(ax_a, img, 0.62, y0, 0.29, target=t)
        ax_a.text(0.56, y0 + 0.29, f"Record {rec}\ncell {t + 1}", ha="right", va="center", fontsize=FS_S, linespacing=1.0)
    ax_b = fig.add_subplot(gs[0, 1])
    X = data[5]["X"].reshape(-1, 14, 7)
    Z = (X - np.median(X, axis=1, keepdims=True)) / (np.subtract(*np.percentile(X, [75, 25], axis=1, keepdims=True)) + 1e-9)
    prof = np.median(Z, axis=0)
    im = ax_b.imshow(prof[:, order].T, aspect="auto", cmap=DIV, norm=TwoSlopeNorm(0, -2.5, 2.5))
    ax_b.set_yticks(range(7)); ax_b.set_yticklabels([FEATS[j] for j in order])
    ax_b.set_xticks(range(14)); ax_b.set_xticklabels([str(k + 1) for k in range(14)], fontsize=FS_S)
    ax_b.set_xlabel("Range cell (record 5, target in cell 7)"); ax_b.grid(False)
    ax_b.add_patch(Rectangle((data[5]["t"] - 0.5, -0.5), 1, 7, fill=False, edgecolor=BLACK, linewidth=1.5))
    pb = ax_b.get_position()
    cax = fig.add_axes([0.915, pb.y0, 0.013, pb.height]); cb = fig.colorbar(im, cax=cax)
    cb.set_label("(x$-$med)/IQR", fontsize=FS_S); cb.ax.tick_params(labelsize=FS_S - 0.5)
    axes_cd = []
    for j, (norm, title) in enumerate(((False, "Raw descriptors"), (True, "After per-frame normalization"))):
        ax = fig.add_subplot(gs[1, j]); axes_cd.append(ax)
        M = np.zeros((7, 10)); allc = []
        for g in range(1, 11):
            Xg = frame_normalize(data[g]["X"]) if norm else data[g]["X"]
            c = Xg[:, :, masks(data[g]["t"])[1], :].reshape(-1, 7)
            M[:, g - 1] = np.median(c, axis=0); allc.append(c)
        allc = np.concatenate(allc)
        iqr = np.subtract(*np.percentile(allc, [75, 25], axis=0))
        D = (M - np.median(M, axis=1, keepdims=True)) / iqr[:, None]
        im2 = ax.imshow(D[order], aspect="auto", cmap=DIV, norm=TwoSlopeNorm(0, -1.0, 1.0))
        ax.set_yticks(range(7)); ax.set_yticklabels([FEATS[k] for k in order])
        ax.set_xticks(range(10)); ax.set_xticklabels([str(g) for g in range(1, 11)]); ax.grid(False)
        ax.set_xlabel("IPIX1993 record"); ax.set_title(title, fontsize=FS, pad=4)
    pd_ = axes_cd[1].get_position()
    cax2 = fig.add_axes([0.915, pd_.y0, 0.013, pd_.height]); cb2 = fig.colorbar(im2, cax=cax2)
    cb2.set_label("Shift (pooled IQR)", fontsize=FS_S); cb2.ax.tick_params(labelsize=FS_S - 0.5)
    pa = ax_a.get_position()
    y_top = below(fig, [ax_b], gap=0.01)
    fig.text((pa.x0 + pa.x1) / 2, y_top, "(a)", ha="center", va="top", fontsize=FS)
    fig.text((pb.x0 + pb.x1) / 2, y_top, "(b)", ha="center", va="top", fontsize=FS)
    yb = below(fig, axes_cd, gap=0.01)
    for ax, l in zip(axes_cd, "cd"):
        p = ax.get_position(); fig.text((p.x0 + p.x1) / 2, yb, f"({l})", ha="center", va="top", fontsize=FS)
    save(fig, "fig_data")


# ------------------------------------------------------------------ Fig. 3: protocol and tail CFAR
def fig_protocol():
    H = 2.8
    fig = plt.figure(figsize=(W2, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W2); ax.set_ylim(0, H); ax.axis("off")
    shades = {"tr": MINT, "va": SAND, "te": CORAL}
    frame(ax, 0.05, 0.05, 3.25, 2.7, "Leave-one-record-out (fold $k$)")
    rx, ry, rw, rh, g2 = 0.25, 1.15, 0.24, 0.62, 0.06
    for i in range(10):
        x = rx + i * (rw + g2)
        box(ax, x, ry, rw, rh, lw=0.6, fc=shades["va"] if i == 8 else shades["te"] if i == 9 else shades["tr"])
    ax.text(rx + 4 * (rw + g2) - g2 / 2, ry - 0.07, "Eight training records", ha="center", va="top", fontsize=FS_S)
    ax.text(rx + 8 * (rw + g2) + rw / 2, ry - 0.07, r"$k{+}1$", ha="center", va="top", fontsize=FS_S)
    ax.text(rx + 9 * (rw + g2) + rw / 2, ry - 0.07, r"$k$", ha="center", va="top", fontsize=FS_S)
    for j, (k, t) in enumerate((("tr", "Training: network weights"), ("va", r"Validation: tail scale $c$ at $\alpha=10^{-2}$"),
                                ("te", "Test: weights and $c$ unchanged"))):
        y = 0.66 - j * 0.24
        box(ax, 0.3, y, 0.2, 0.14, fc=shades[k], lw=0.6)
        ax.text(0.58, y + 0.07, t, ha="left", va="center", fontsize=FS_S)
    ax.text(rx, 2.2, "Ten folds: every IPIX1993 record is\ntested once by a model that never saw it.", ha="left",
            va="center", fontsize=FS_S, linespacing=1.15)
    frame(ax, 3.4, 0.05, 3.71, 2.7, "Clutter-map tail CFAR on the test record")
    from rgcs import calibrate_c, tail_params
    z = np.load(ROOT / "phaseD_temporal" / "final" / "G7_K4_f10_s0_scores.npz")  # final detector, frames 15-255
    Sv = z["val"]; uv, bv = tail_params(Sv)
    c_fin = calibrate_c(Sv[:, 1:], np.full(4, int(z["val_t"])), uv[:, 1:], bv[:, 1:], 1e-2)
    S = z["test"][0]; n = 200 - 15
    Rn = np.where(np.arange(14)[None] == S[n - 64:n].argmax(-1)[:, None], np.nan, S[n - 64:n]).ravel()
    Rn = np.sort(Rn[~np.isnan(Rn)])
    u = np.quantile(Rn, 0.9); b = (Rn[Rn > u] - u).mean(); c = c_fin
    t2, t3 = u + b * np.log(0.1 / (c * 1e-2)), u + b * np.log(0.1 / (c * 1e-3))
    ins = fig.add_axes([0.575, 0.18, 0.4, 0.52])
    xs = np.linspace(u - 6, t3 + 1.5, 400)
    surv = 1 - np.searchsorted(Rn, xs, side="left") / len(Rn)
    ins.step(xs, np.where(surv > 0, surv, np.nan), where="post", color=SLATE, linewidth=1.6,
             label=r"Reference $\mathcal{R}_n$")
    xt = xs[xs >= u]
    ins.plot(xt, 0.1 * np.exp(-(xt - u) / b), color=CORAL, linewidth=1.7, linestyle="--", label="Exponential tail")
    for tau, ls, lab in ((t2, "--", r"$\tau_n(10^{-2})$"), (t3, "-.", r"$\tau_n(10^{-3})$")):
        ins.axvline(tau, color=BLACK, linestyle=ls, linewidth=1.0)
        ins.text(tau - 0.2, 0.62, lab, ha="right", va="bottom", fontsize=FS_S, transform=ins.get_xaxis_transform())
    ins.axvline(u, color="#4E7A43", linewidth=1.0, linestyle=":")
    ins.text(u - 0.2, 0.9, "$u_n$", ha="right", va="top", fontsize=FS_S, color="#4E7A43",
             transform=ins.get_xaxis_transform())
    ins.set_yscale("log"); ins.set_ylim(3e-4, 1.3); ins.set_xlim(u - 6, t3 + 1.5); ins.grid(which="minor", visible=False)
    ins.set_xlabel(r"Cell logit $s_k$"); ins.set_ylabel("Exceedance")
    ins.legend(loc="lower left", fontsize=FS_S - 0.5, handlelength=1.6)
    ax.text(5.25, 2.33, r"$\tau_n(\alpha)=u_n+\beta_n\ln\dfrac{0.1}{c\,\alpha}$", ha="center", va="center", fontsize=FS + 0.5)
    save(fig, "fig_protocol")


# ------------------------------------------------------------------ Fig. 4
def fig_diagnosis():
    s = pd.read_csv(A / "A2_per_subset.csv")
    fig, axes = plt.subplots(1, 2, figsize=(W2, 3.2), gridspec_kw=dict(wspace=0.28))
    fig.subplots_adjust(left=0.08, right=0.985, top=0.76, bottom=0.24)
    a0, a1 = axes
    styles = {("LR", "framenorm"): (SLATE, "s", "-", True), ("LR", "raw"): (SLATE, "s", "--", False),
              ("HGB", "framenorm"): (CORAL, "o", "-", True), ("HGB", "raw"): (CORAL, "o", "--", False)}
    named = {"RAA+RDPH+RVE": 3, "RAA+RDPH+RVE+MCS": 4, "Hurst+RAA+RDPH+RVE+RI": 5, "Hurst+RAA+RDPH+RVE+RI+NR+MCS": 7}
    for rep, ls in (("framenorm", "-"), ("raw", "--")):
        g = s[(s.model == "LR") & (s.rep == rep) & (~s.ctx)].groupby("k")[["pdor_001", "over2_001"]].mean()
        a0.plot(g.index, g.pdor_001, color="#AEB5B8", linestyle=ls, linewidth=1.0, zorder=1)
        a1.plot(g.index, g.over2_001, color="#AEB5B8", linestyle=ls, linewidth=1.0, zorder=1)
    for (mdl, rep), (col, mk, ls, filled) in styles.items():
        d = s[(s.model == mdl) & (s.rep == rep) & (~s.ctx)]
        if mdl == "LR":
            d = d[d.subset.isin(named)].assign(k=lambda x: x.subset.map(named))
        g = d.groupby("k")[["pdor_001", "over2_001"]].mean()
        kw = dict(color=col, marker=mk, linestyle=ls, markersize=6, linewidth=1.5,
                  markerfacecolor=col if filled else "white", markeredgecolor=CORAL_E if col == CORAL else SLATE)
        a0.plot(g.index, g.pdor_001, **kw); a1.plot(g.index, g.over2_001, **kw)
    hg = s[(s.model == "HGB") & (s.rep == "framenorm") & (~s.ctx)].sort_values("k")
    for _, r in hg.iterrows():
        off = {"G3": (-17, 0), "G4": (12, -12), "G5": (0, 8), "G7": (0, 8)}[r.subset]
        a0.annotate(r.subset, (r.k, r.pdor_001), textcoords="offset points", xytext=off, ha="center", fontsize=FS_S)
    a0.set_ylabel(r"$\hat{P}_d$ at test $\hat{P}_{fa}=10^{-2}$"); a0.set_ylim(0, 0.62)
    a1.set_ylabel(r"Folds with $\hat{P}_{fa}>2\alpha$"); a1.set_ylim(-0.15, 3.6)
    for ax in axes:
        ax.set_xticks(range(1, 8)); ax.set_xlim(0.6, 7.4); ax.set_xlabel("Number of descriptors")
    h = [Line2D([], [], color=CORAL, marker="o", markeredgecolor=CORAL_E, label="Nonlinear scorer"),
         Line2D([], [], color=SLATE, marker="s", label="Linear scorer"),
         Line2D([], [], color="#AEB5B8", label="Linear, all subsets (mean)"),
         Line2D([], [], color=BLACK, linestyle="-", label="Frame-normalized"),
         Line2D([], [], color=BLACK, linestyle="--", marker="o", markerfacecolor="white", label="Raw")]
    fig.legend(handles=h, ncol=3, loc="upper center", bbox_to_anchor=(0.53, 1.0), handlelength=2.0, columnspacing=1.2)
    panel_labels(fig, axes, y=below(fig, axes))
    save(fig, "fig_diagnosis")


# ------------------------------------------------------------------ Fig. 5
def fig_anatomy():
    fig = plt.figure(figsize=(W2, 3.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.2, 1.05], wspace=0.5, left=0.085, right=0.985, top=0.76, bottom=0.27)
    d = pd.read_csv(A / "A6_drift_1998_vs_1993.csv"); d = d[d.rep == "raw"].set_index("feature").loc[FEATS]
    o93 = np.where(d.auc93 >= 0.5, d.auc93, 1 - d.auc93); o98 = np.where(d.auc93 >= 0.5, d.auc98, 1 - d.auc98)
    ax0 = fig.add_subplot(gs[0]); y = np.arange(7)
    ax0.hlines(y, o93, o98, color="#BFC5C7", linewidth=2.4, zorder=2)
    ax0.plot(o93, y, "s", color=SLATE, markersize=6.5, zorder=3, label="IPIX1993")
    ax0.plot(o98, y, "o", color=CORAL, markeredgecolor=CORAL_E, markersize=7, zorder=3, label="IPIX1998")
    ax0.axvline(0.5, color=BLACK, linestyle=":", linewidth=0.9)
    ax0.set_yticks(y); ax0.set_yticklabels(FEATS); ax0.invert_yaxis(); ax0.set_xlim(0.1, 1.0); ax0.set_xticks([0.25, 0.5, 0.75, 1.0])
    ax0.grid(axis="y", visible=False); ax0.grid(axis="x", visible=True)
    ax0.set_xlabel("Target vs. clutter AUC")
    ax0.legend(loc="lower center", bbox_to_anchor=(0.45, 1.01), ncol=1, handlelength=1.0)
    a3 = pd.read_csv(A / "A3_detector_pdpfa.csv"); g = a3[a3.method == "SA-RGB G7"].sort_values("fold")
    adj = g["pfa_adj_0.01"].to_numpy() * 2 / 13; oth = g["pfa_0.01"].to_numpy() - adj
    ax1 = fig.add_subplot(gs[1]); shade(ax1, (1, 6), "Unseen\ncell")
    ax1.bar(g.fold, adj, width=0.68, color=CORAL, zorder=3, label="Adjacent cells")
    ax1.bar(g.fold, oth, bottom=adj, width=0.68, color=SLATE, zorder=3, label="Other $H_0$ cells")
    ax1.axhline(2 * ALPHA, color=BLACK, linestyle="-.", linewidth=1.0)
    ax1.text(10.6, 2 * ALPHA, r"$2\alpha$", ha="left", va="center", fontsize=FS)
    ax1.set_xticks(range(1, 11)); ax1.set_xlim(0.5, 10.5); ax1.set_ylim(0, 0.23)
    ax1.set_xlabel("LORO fold"); ax1.set_ylabel(r"Test $\hat{P}_{fa}$ (YOLO11n, G7)")
    ax1.legend(loc="upper right", bbox_to_anchor=(1.0, 0.84), handlelength=1.2, fontsize=FS_S)
    h = pd.read_csv(A / "A4_frame_hits.csv"); h = h[~h.fold.isin([1, 6])]
    h["b"] = pd.cut(h.scr_db, [-60, 0, 3, 6, 10, 15, 60])
    ax2 = fig.add_subplot(gs[2])
    for m, col, mk, ec, lab in (("YOLO11n SA-RGB G7", SLATE, "s", SLATE, "YOLO11n (grid detector)"),
                                ("HGB-G7", CORAL, "o", CORAL_E, "Position-free scorer")):
        v = h[h.method == m].groupby("b", observed=True).hit.mean().to_numpy()
        ax2.plot(range(6), v, color=col, marker=mk, markeredgecolor=ec, markersize=6, label=lab)
    ax2.set_xticks(range(6)); ax2.set_xticklabels(["≤0", "0–3", "3–6", "6–10", "10–15", ">15"], fontsize=FS_S, rotation=30)
    ax2.set_xlabel("Per-frame SCR proxy (dB)"); ax2.set_ylabel(r"$\hat{P}_d$ at calibrated $\tau$"); ax2.set_ylim(0, 1.05)
    ax2.legend(loc="lower center", bbox_to_anchor=(0.45, 1.01), ncol=1, handlelength=1.8)
    axes = [ax0, ax1, ax2]
    panel_labels(fig, axes, y=below(fig, axes))
    save(fig, "fig_anatomy")


# ------------------------------------------------------------------ Fig. 6
def _loro_series():
    L = pd.read_csv(C / "C_loro_long.csv"); L = L[L.alpha == ALPHA]
    out = {}
    for key, arm, rule in (("G7", "G7", "texp64"), ("G3", "G3", "texp64"), ("G7fix", "G7", "fixed")):
        g = L[(L.arm == arm) & (L.rule == rule)].groupby("fold")
        out[key] = (g.pfa.mean(), g.pd.mean())
    Hh = pd.read_csv(C / "C_classical_hull.csv"); Hh = Hh[(Hh.protocol == "loro") & (Hh.alpha == ALPHA)].set_index("fold")
    out["hull"] = (Hh.pfa, Hh.pd)
    RB = pd.read_csv(C / "C_reference_baselines.csv")
    RB = RB[(RB.protocol == "loro") & (RB.method == "HGB-G7") & (RB.rule == "texp") & (RB.alpha == ALPHA)].set_index("fold")
    out["hgb"] = (RB.pfa, RB.pd)
    return out


def fig_loro():
    s = _loro_series()
    fig, axes = plt.subplots(1, 2, figsize=(W2, 3.2), gridspec_kw=dict(wspace=0.28))
    fig.subplots_adjust(left=0.08, right=0.965, top=0.73, bottom=0.24)
    folds = np.arange(1, 11)
    for ax in axes:
        shade(ax, (1, 6), "Unseen\ncell")
    for key in ("hgb", "G7fix", "G3", "G7"):
        lab, col, mk, ec = ARM[key]
        pfa, pd_ = s[key]
        for ax, v in ((axes[0], np.maximum(pfa.reindex(folds).to_numpy(), FLOOR)), (axes[1], pd_.reindex(folds).to_numpy())):
            ax.plot(folds, v, color=col, marker=mk, markersize=5.6, markeredgecolor=ec, markeredgewidth=0.7,
                    linewidth=1.5, zorder=4 if key == "G7" else 3)
    log_pfa(axes[0]); budget(axes[0], text_x=10.62)
    axes[0].set_ylabel(r"Test $\hat{P}_{fa}$"); axes[1].set_ylabel(r"Test $\hat{P}_{d}$"); axes[1].set_ylim(-0.03, 1.06)
    for ax in axes:
        ax.set_xticks(folds); ax.set_xlim(0.5, 10.5); ax.set_xlabel("Held-out IPIX1993 record")
    h = [handle(k) for k in ("G7", "G3", "G7fix", "hgb")] + \
        [Line2D([], [], color=BLACK, linestyle="--", linewidth=1.0, label=r"$\alpha$"),
         Line2D([], [], color=BLACK, linestyle="-.", linewidth=1.0, label=r"$2\alpha$")]
    fig.legend(handles=h, ncol=3, loc="upper center", bbox_to_anchor=(0.52, 1.0), handlelength=2.2, columnspacing=1.6)
    panel_labels(fig, axes, y=below(fig, axes))
    save(fig, "fig_loro")


# ------------------------------------------------------------------ Fig. 7: inside RadarGrid-CS
def fig_internals():
    import torch
    from common import load, loro_split
    from explore_context import build
    from rgcs import calibrate_c, fit_percentiles, quantize, render, tail_params, texp_tau
    from run_loro import make_inputs
    data = load(); recs = (3, 5, 9, 6); POL, FR, F0 = 0, 128, 15
    FIN = ROOT / "phaseD_temporal" / "final"
    rows = ["Input grid", "SCN: RAA", "SCN: Hurst", "Range peak\nmargin (RAA)", "G7 firing\nrate", "G3 firing\nrate"]
    fig = plt.figure(figsize=(W2, 5.4))
    gs = fig.add_gridspec(len(rows), len(recs), wspace=0.07, hspace=0.3, left=0.11, right=0.9, top=0.95, bottom=0.03)
    cax_im = {}
    front = build("margin1", "G7").eval()
    for j, rec in enumerate(recs):
        train, val, test = loro_split(rec)
        lo, hi = fit_percentiles(np.concatenate([data[g]["X"].reshape(-1, 7) for g in train]))
        Q = quantize(data[rec]["X"][POL, FR][None], lo, hi)
        t = data[rec]["t"]
        with torch.no_grad():
            f = front.ctx(front.scn(front.readout(make_inputs("G7", Q))))[0].numpy()  # (72, 2, 7)
        out = {}
        for arm in ("G7", "G3"):
            z = np.load(FIN / f"{arm}_K4_f{rec:02d}_s0_scores.npz")
            Sv, St = z["val"], z["test"]
            uv, bv = tail_params(Sv); c = calibrate_c(Sv[:, 1:], np.full(4, int(z["val_t"])), uv[:, 1:], bv[:, 1:], 1e-2)
            u, b = tail_params(St); tau = texp_tau(u, b, 1e-2, c)[:, 1:]
            out[arm] = dict(rate=(St[:, 1:] >= tau[..., None]).mean((0, 1)))
        grid = lambda v: v.reshape(7, 2).T
        mats = [None, f[0], f[1], f[36], grid(out["G7"]["rate"]), grid(out["G3"]["rate"])]
        for i, name in enumerate(rows):
            ax = fig.add_subplot(gs[i, j]); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for sp in ax.spines.values():
                sp.set_linewidth(0.6)
            if i == 0:
                ax.set_xlim(0, 7); ax.set_ylim(0, 2); ax.set_aspect("equal")
                slot_grid(ax, render(Q, "G7")[0], 0, 0, 1.0, target=t, lw_t=2.0)
                ttl = f"Record {rec}" + (" (unseen cell)" if rec == 6 else "")
                ax.set_title(ttl, fontsize=FS_S, color=UNSEEN if rec == 6 else BLACK, pad=3)
            else:
                M = mats[i]
                if i <= 3:
                    im = ax.imshow(M, cmap=DIV, norm=TwoSlopeNorm(0, -6, 6), aspect="equal")
                else:
                    im = ax.imshow(M, cmap=SEQ, vmin=0, vmax=1, aspect="equal")
                    rt = out["G7" if i == 4 else "G3"]["rate"][t]
                    ax.text(t // 2, t % 2, f"{rt:.2f}", ha="center", va="center", fontsize=FS_S - 1.5,
                            color="white" if rt > 0.55 else BLACK, zorder=6)
                ax.add_patch(Rectangle((t // 2 - 0.5, t % 2 - 0.5), 1, 1, fill=False, edgecolor=BLACK, linewidth=1.3,
                                       linestyle=(0, (2, 1.5)), zorder=5))
                if j == len(recs) - 1:
                    cax_im[i] = (ax, im)
            if j == 0:
                ax.set_ylabel(name, fontsize=FS_S, rotation=90, labelpad=6, linespacing=1.0)
        print(rec, {a: np.round(out[a]["rate"], 3).tolist() for a in out}, "t", t,
              "margin RAA", np.round(f[36].T.ravel(), 2).tolist(), "scnRAA", np.round(f[0].T.ravel(), 2).tolist())
    for i, (ax, im) in cax_im.items():
        pos = ax.get_position()
        cax = fig.add_axes([0.915, pos.y0 + 0.004, 0.012, pos.height - 0.008])
        cb = fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=FS_S - 1.5)
    save(fig, "fig_internals")

# ------------------------------------------------------------------ Fig. 8
def fig_ablation():
    L = pd.read_csv(C / "C_loro_long.csv"); L = L[L.alpha == ALPHA]
    rows = [("G7", "texp64", "G7 (full)", CORAL), ("G3", "texp64", "G3", SLATE), ("D7", "texp64", "D7: vector input", TEAL),
            ("P7", "texp64", r"P7: learned 7$\to$3", TEAL), ("G7-noSCN", "texp64", "G7 without SCN", SAND),
            ("G7-absC", "texp64", "G7 absolute context", SAND), ("G7", "fixed", "G7 fixed threshold", SAGE)]
    fig, axes = plt.subplots(1, 3, figsize=(W2, 3.1), sharey=True, gridspec_kw=dict(wspace=0.12))
    fig.subplots_adjust(left=0.2, right=0.985, top=0.86, bottom=0.27)
    y = np.arange(len(rows)); vals = []
    for arm, rule, _, _ in rows:
        d = L[(L.arm == arm) & (L.rule == rule)]; f = d.groupby("fold").pd.mean()
        vals.append((f.mean(), f.std(ddof=1) / np.sqrt(10), int((d.pfa > 2 * ALPHA).sum()), d.pfa_adj.mean()))
    cols = [r[3] for r in rows]
    a0, a1, a2 = axes
    a0.barh(y, [v[0] for v in vals], height=0.64, color=cols, zorder=3, xerr=[v[1] for v in vals],
            error_kw=dict(ecolor=BLACK, elinewidth=0.9, capsize=2.5))
    for yi, v in zip(y, vals):
        a0.text(v[0] + v[1] + 0.02, yi, f"{v[0]:.3f}", va="center", fontsize=FS_S - 0.5)
    a0.set_xlim(0, 0.95); a0.set_xlabel(r"Test $\hat{P}_d$ ($\pm$ s.e.)")
    a1.barh(y, [v[2] for v in vals], height=0.64, color=cols, zorder=3)
    for yi, v in zip(y, vals):
        a1.text(v[2] + 0.15, yi, f"{v[2]}/30", va="center", fontsize=FS_S - 0.5)
    a1.set_xlim(0, 9.5); a1.set_xticks([0, 2, 4, 6, 8]); a1.set_xlabel(r"Runs with $\hat{P}_{fa}>2\alpha$")
    a2.barh(y, [v[3] for v in vals], height=0.64, color=cols, zorder=3)
    for yi, v in zip(y, vals):
        a2.text(v[3] + 0.0015, yi, f"{v[3]:.3f}", va="center", fontsize=FS_S - 0.5)
    a2.set_xlim(0, 0.078); a2.set_xlabel(r"Adjacent-cell $\hat{P}_{fa}$")
    a0.set_yticks(y); a0.set_yticklabels([r[2] for r in rows]); a0.invert_yaxis()
    for ax in axes:
        ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True)
    h = [Patch(color=CORAL, label="G7"), Patch(color=SLATE, label="G3"), Patch(color=TEAL, label="Input form"),
         Patch(color=SAND, label="Module ablation"), Patch(color=SAGE, label="Threshold ablation")]
    fig.legend(handles=h, ncol=5, loc="upper center", bbox_to_anchor=(0.58, 1.0), handlelength=1.2, columnspacing=1.2)
    panel_labels(fig, axes, y=below(fig, axes))
    save(fig, "fig_ablation")

# ------------------------------------------------------------------ Fig. 9
def fig_crosssite():
    R = json.loads((C / "runs_1998" / "summary.json").read_text())
    recs = sorted(R[0]["0.01"]["pd_by_record"], key=int)
    res = {"8": "15 m", "9": "9 m"}
    fig = plt.figure(figsize=(W2, 3.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.12, 1.12, 1], wspace=0.5, left=0.075, right=0.985, top=0.76, bottom=0.27)
    x = np.arange(1, 10)
    ax0, ax1 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    for ax in (ax0, ax1):
        shade(ax, (8, 9), None)
        ax.text(8.5, 1.015, "15 m   9 m", ha="center", va="bottom", fontsize=FS_S - 0.5, color=UNSEEN,
                transform=ax.get_xaxis_transform())
    for arm in ("G3", "G7"):
        lab, col, mk, ec = ARM[arm]
        pdv = [np.mean([r["0.01"]["pd_by_record"][k] for r in R if r["arm"] == arm]) for k in recs]
        pfv = [np.mean([r["0.01"]["pfa_by_record"][k] for r in R if r["arm"] == arm]) for k in recs]
        ax0.plot(x, pdv, color=col, marker=mk, markeredgecolor=ec, markersize=6, linewidth=1.5)
        ax1.plot(x, pfv, color=col, marker=mk, markeredgecolor=ec, markersize=6, linewidth=1.5)
    budget(ax1, text_x=9.65); ax1.set_ylim(0, 0.042)
    labels = [f"R{k}" for k in recs]
    for ax in (ax0, ax1):
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=FS_S - 0.5, linespacing=0.95); ax.set_xlim(0.5, 9.5)
        ax.set_xlabel("IPIX1998 record")
    ax0.set_ylabel(r"Test $\hat{P}_d$"); ax0.set_ylim(0, 0.8); ax1.set_ylabel(r"Test $\hat{P}_{fa}$")
    e = pd.read_csv(C / "explore_context" / "summary.csv"); e = e[e.arm == "G7"]
    e = e.pivot_table(index="variant", columns="protocol", values=["pd1e2", "pfa1e2"])
    names = {"grid3x3": "Grid 3×3\n(fixed design)", "none": "None", "r1": "±1", "r2": "±2", "r12": "±1, ±2",
             "margin1": "Peak margin ±1"}
    pos = {"grid3x3": (0.522, 2.2), "none": (0.545, 1.33), "r1": (0.668, 2.13), "r2": (0.555, 2.53), "r12": (0.672, 2.43),
           "margin1": (0.575, 1.82)}
    ax2 = fig.add_subplot(gs[2])
    ax2.axhspan(0, 2.0, color="#EEF4EA", zorder=0, linewidth=0)
    for v, r in e.iterrows():
        col, ec, mk = (CORAL, CORAL_E, "o") if v == "margin1" else (SLATE, SLATE, "s") if v == "grid3x3" else ("#9AA5A9", "#7E898D", "D")
        p = (r[("pd1e2", "loro")], r[("pfa1e2", "1998")] / ALPHA)
        ax2.plot(*p, marker=mk, color=col, markeredgecolor=ec, markersize=7, linestyle="none", zorder=3)
        ax2.annotate(names[v], p, xytext=pos[v], fontsize=FS_S - 0.5, linespacing=0.95, va="center",
                     arrowprops=dict(arrowstyle="-", color="#7E898D", linewidth=0.6, shrinkA=1, shrinkB=4))
    ax2.axhline(2.0, color=BLACK, linestyle="-.", linewidth=1.0)
    ax2.set_xlim(0.5, 0.72); ax2.set_ylim(1.2, 2.65); ax2.set_xticks([0.5, 0.6, 0.7])
    ax2.set_xlabel(r"IPIX1993 LORO $\hat{P}_d$"); ax2.set_ylabel(r"IPIX1998 $\hat{P}_{fa}/\alpha$")
    ax2.set_title("Context variants (second stage)", fontsize=FS_S, pad=4)
    h = [handle("G7", label="G7"), handle("G3", label="G3"),
         Line2D([], [], color=BLACK, linestyle="--", linewidth=1.0, label=r"$\alpha$"),
         Line2D([], [], color=BLACK, linestyle="-.", linewidth=1.0, label=r"$2\alpha$")]
    fig.legend(handles=h, ncol=4, loc="upper center", bbox_to_anchor=(0.36, 1.0), handlelength=2.0)
    axes = [ax0, ax1, ax2]
    panel_labels(fig, axes, y=below(fig, axes))
    save(fig, "fig_crosssite")


# ------------------------------------------------------------------ Fig. 10
def fig_shift():
    gsens = pd.read_csv(C / "C_shift_sensitivity.csv"); gm = pd.read_csv(C / "C_shift_margin1.csv")
    fig = plt.figure(figsize=(W2, 3.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.2, 0.9], wspace=0.42, left=0.075, right=0.985, top=0.75, bottom=0.25)
    shifts = [-2, -1, 0, 1, 2]
    axes = []
    for j, (df, title) in enumerate(((gsens, "Grid-direction context"), (gm, "Range-ordered peak margin"))):
        ax = fig.add_subplot(gs[j]); axes.append(ax)
        shade(ax, (-1, 1), "Other\nrow", width=0.4)
        for arm, col, mk, ec in (("G7", CORAL, "o", CORAL_E), ("G3", SLATE, "s", SLATE)):
            m = df[df.arm == arm].groupby(["shift", "fold"]).pd.mean().unstack().mean(axis=1).reindex(shifts)
            ax.plot(shifts, m, color=col, marker=mk, markeredgecolor=ec, markersize=6, linewidth=1.5, zorder=3)
        ax.set_xticks(shifts); ax.set_xticklabels(["$-2$", "$-1$", "0", "$+1$", "$+2$"])
        ax.set_ylim(0.38, 0.75); ax.set_xlim(-2.5, 2.5)
        ax.set_xlabel("Cyclic range shift (cells)"); ax.set_ylabel(r"Test $\hat{P}_d$")
        ax.text(0.5, 0.04, title, transform=ax.transAxes, ha="center", va="bottom", fontsize=FS_S,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    ax = fig.add_subplot(gs[2]); axes.append(ax)
    vals = []
    for df in (gsens, gm):
        d = df[(df.arm == "G7") & (df.fold == 6)]
        vals.append((d[d["shift"] == 0].pd.mean(), d[d["shift"].abs() == 1].pd.mean()))
    xb = np.array([0, 1])
    ax.bar(xb - 0.19, [v[0] for v in vals], width=0.38, color=CORAL, zorder=3, label="Original")
    ax.bar(xb + 0.19, [v[1] for v in vals], width=0.38, color=SLATE, zorder=3, label=r"Shifted $\pm1$")
    for xi, v in zip(xb, vals):
        ax.text(xi - 0.19, v[0] + 0.02, f"{v[0]:.2f}", ha="center", va="bottom", fontsize=FS_S - 0.5)
        ax.text(xi + 0.19, v[1] + 0.02, f"{v[1]:.2f}", ha="center", va="bottom", fontsize=FS_S - 0.5)
    ax.set_xticks(xb); ax.set_xticklabels(["Grid\ndirection", "Peak\nmargin"], fontsize=FS_S, linespacing=0.95)
    ax.set_ylim(0, 0.8); ax.set_xlim(-0.6, 1.6); ax.set_ylabel(r"Fold 6 $\hat{P}_d$ (G7)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=1, handlelength=1.2)
    fig.legend(handles=[handle("G7", label="G7"), handle("G3", label="G3")], ncol=2, loc="upper center",
               bbox_to_anchor=(0.36, 1.0), handlelength=2.0)
    panel_labels(fig, axes, y=below(fig, axes))
    save(fig, "fig_shift")


# ------------------------------------------------------------------ Fig. 11: short-observation accumulation
def fig_temporal():
    D_ = ROOT / "phaseD_temporal"
    N2 = pd.read_csv(D_ / "N2_duration.csv"); M = pd.read_csv(D_ / "CHK_M_mechanism.csv"); N3 = pd.read_csv(D_ / "N3_final.csv")
    fig = plt.figure(figsize=(W2, 3.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 1.15], wspace=0.48, left=0.075, right=0.985, top=0.76, bottom=0.27)
    axes = [fig.add_subplot(gs[i]) for i in range(3)]
    # (a) Pd versus dwell for three frame lengths
    ax = axes[0]
    t = N2[(N2.rule == "texp") & (N2.alpha == 1e-2)]
    sty = {0.256: (TEAL, "^", TEAL_E), 0.512: (SAND, "D", SAND_E), 1.024: ("#9AA5A9", "v", "#6E797D")}
    for d, (col, mk, ec) in sty.items():
        g = t[t.dur == d].groupby("dwell").pd.mean()
        ax.plot(g.index, g.values, color=col, marker=mk, markeredgecolor=ec, markersize=6, linewidth=1.5,
                label=f"{d:.3f} s frames".replace("0.", "0."), zorder=3)
    for T in (1.024, 2.048):
        ax.axvline(T, color="#B8B8B8", linewidth=0.7, linestyle=":", zorder=1)
    ax.set_xscale("log", base=2); ax.set_xticks([0.256, 0.512, 1.024, 2.048, 4.096, 8.192])
    ax.set_xticklabels(["0.26", "0.51", "1.02", "2.05", "4.10", "8.19"], fontsize=FS_S - 0.5)
    ax.set_xlabel("Total dwell $Kd$ (s)"); ax.set_ylabel(r"Test $\hat{P}_d$"); ax.set_ylim(0.5, 0.92)
    ax.legend(loc="lower right", handlelength=1.6, fontsize=FS_S - 0.5)
    # (b) calibration ratio across alpha
    ax = axes[1]
    al = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3]
    for prot, ls in (("loro", "-"), ("1998", "--")):
        for K, (col, mk, ec) in ((1, (SLATE, "s", SLATE)), (4, (CORAL, "o", CORAL_E))):
            r = [M[(M.protocol == prot) & (M.K == K)][f"ratio_{a:g}"].mean() for a in al]
            ax.plot(al, r, color=col, marker=mk, markeredgecolor=ec, markersize=5.5, linewidth=1.4, linestyle=ls, zorder=3)
    ax.axhline(1, color=BLACK, linewidth=0.9, linestyle="--"); ax.axhline(2, color=BLACK, linewidth=0.9, linestyle="-.")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis()
    ax.set_xticks(al[::2]); ax.set_xticklabels([r"$10^{-1}$", r"$10^{-2}$", r"$10^{-3}$"])
    ax.set_yticks([0.5, 1, 2, 4]); ax.set_yticklabels(["0.5", "1", "2", "4"]); ax.set_ylim(0.5, 6)
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"Budget $\alpha$"); ax.set_ylabel(r"Realized $\hat{P}_{fa}/\alpha$")
    ax.legend(handles=[Line2D([], [], color=BLACK, linestyle="-", label="IPIX1993"),
                       Line2D([], [], color=BLACK, linestyle="--", label="IPIX1998")],
              loc="upper left", handlelength=1.8, fontsize=FS_S - 0.5)
    # (c) IPIX1998 per record: G7 K = 1, G3 K = 4, G7 K = 4
    ax = axes[2]
    r = N3[N3.protocol == "1998rec"].groupby(["arm", "K", "fold"]).pd.mean()
    x = np.arange(1, 10)
    shade(ax, (8, 9), None)
    ax.text(8.5, 1.015, "15 m   9 m", ha="center", va="bottom", fontsize=FS_S - 0.5, color=UNSEEN,
            transform=ax.get_xaxis_transform())
    bars = ((("G7", 1), SLATE, -0.27), (("G3", 4), SAND, 0.0), (("G7", 4), CORAL, 0.27))
    for key, col, dx in bars:
        ax.bar(x + dx, r[key].reindex(x).values, width=0.26, color=col, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([f"R{k}" for k in x], fontsize=FS_S - 0.5); ax.set_xlim(0.4, 9.6)
    ax.set_ylim(0, 1); ax.set_xlabel("IPIX1998 record"); ax.set_ylabel(r"Test $\hat{P}_d$")
    fig.legend(handles=[Line2D([], [], color=SLATE, marker="s", markersize=6, linewidth=1.5, label="$K=1$, G7"),
                        Line2D([], [], color=CORAL, marker="o", markeredgecolor=CORAL_E, markersize=6, linewidth=1.5,
                               label="$K=4$, G7"),
                        Patch(facecolor=SAND, label="G3, $K=4$")],
               ncol=3, loc="upper center", bbox_to_anchor=(0.66, 1.0), handlelength=1.8)
    panel_labels(fig, axes, y=below(fig, axes))
    save(fig, "fig_temporal")


if __name__ == "__main__":
    for w in sys.argv[1:] or ["chain", "data", "protocol", "diagnosis", "anatomy", "loro", "internals", "ablation",
                              "crosssite", "shift", "temporal"]:
        globals()[f"fig_{w}"]()







