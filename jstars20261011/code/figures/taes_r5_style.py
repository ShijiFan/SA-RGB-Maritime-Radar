"""Figure style for the TAES resubmission (v5), matched to the accepted TRS paper (TriRadar, Figs. 9-16).

Reference: paper1 figures (fig_4_2_*, fig_4_5_*, fig_4_6_*): Times New Roman text, fully boxed axes with
outward ticks on the bottom and left only, light horizontal grid, solid colour fills without edge lines,
thick lines, legend without frame placed above the axes. Palette of the TRS paper:
dark slate #264653, coral #E76F51, teal #2A9D8F, sand #E9C46A, sage #8AB17D.
Figures are drawn at printed size (IEEE: 3.5 in / 7.16 in) and exported as vector PDF and EPS plus
600-dpi PNG and TIFF.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("E:/multipletrifeatures")
T = ROOT / "generated" / "taes_r2"
FIG = ROOT / "投稿2" / "TAES重投_v3" / "source" / "figures"

# TRS palette
SLATE, CORAL, TEAL, SAND, SAGE = "#264653", "#E76F51", "#2A9D8F", "#E9C46A", "#8AB17D"
BLACK, GRID = "#000000", "#e6e6e6"
RED = CORAL
TINT = "#FBE6DF"  # light coral tint for highlighted bands
MODEL = {  # key -> (label, face colour, marker, edge colour)
    "cn": ("CN-TFER (proposed)", CORAL, "o", "#C4502F"),
    "baseline": ("Baseline [33]", SLATE, "s", SLATE),
    "sctfer": ("SC-TFER", TEAL, "^", "#1F7A6F"),
    "yolo11n": ("YOLO11n", SAND, "D", "#B8901F"),
}
ORDER = ["cn", "baseline", "sctfer", "yolo11n"]
W1, W2 = 3.5, 7.16

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral"], "mathtext.fontset": "stix",
    "font.size": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.9, "axes.edgecolor": BLACK, "axes.labelcolor": BLACK, "axes.labelpad": 3,
    "xtick.direction": "out", "ytick.direction": "out", "xtick.top": False, "ytick.right": False,
    "xtick.major.size": 3.2, "ytick.major.size": 3.2, "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
    "xtick.major.width": 0.9, "ytick.major.width": 0.9, "xtick.minor.width": 0.6, "ytick.minor.width": 0.6,
    "xtick.major.pad": 2.5, "ytick.major.pad": 2.5,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.axisbelow": True,
    "legend.frameon": False, "legend.handlelength": 1.8, "legend.handletextpad": 0.5,
    "legend.labelspacing": 0.3, "legend.columnspacing": 1.6, "legend.borderaxespad": 0.2,
    "lines.linewidth": 1.4, "lines.markersize": 4.5, "lines.markeredgewidth": 0.8,
    "patch.linewidth": 0.0,
    "figure.facecolor": "white", "savefig.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42,
    "svg.fonttype": "none",
})


def mark(ax, x, y, key, size=4.5, filled=True, line="none", lw=1.4, **kw):
    """Plot with the detector's colour and marker (filled by default, as in the TRS figures)."""
    _, col, mk, ec = MODEL[key]
    return ax.plot(x, y, linestyle=line, color=col, marker=mk, markersize=size, linewidth=lw,
                   markerfacecolor=col if filled else "white", markeredgecolor=ec if filled else col, **kw)


def panel_labels(fig, axes, letters="abcdefgh", gap=0.012, y=None):
    """Place '(a)', '(b)', ... centred below each axes, under its tick and axis labels."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for ax, l in zip(axes, letters):
        bb = ax.get_tightbbox(r).transformed(inv)
        pos = ax.get_position()
        fig.text((pos.x0 + pos.x1) / 2, (y if y is not None else bb.y0 - gap), f"({l})", ha="center", va="top",
                 fontsize=8.5)


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", dpi=600)
    fig.savefig(FIG / f"{name}.eps", dpi=600)
    fig.savefig(FIG / f"{name}.png", dpi=600)
    fig.savefig(FIG / f"{name}.tif", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print("saved", name)
