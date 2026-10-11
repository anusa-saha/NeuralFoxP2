"""Figure fig:tradeoff (E9 sweeps, E8 single-point methods).

usage: python fig_tradeoff.py <data_dir> <out_dir>
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import pandas as pd

data_dir, out_dir = sys.argv[1], sys.argv[2]
for f in ("LiberationSerif-Regular.ttf", "LiberationSerif-Bold.ttf", "LiberationSerif-Italic.ttf"):
    font_manager.fontManager.addfont(f"/usr/share/fonts/truetype/liberation/{f}")
plt.rcParams.update({
    "font.family": "Liberation Serif", "mathtext.fontset": "stix", "font.size": 7,
    "axes.linewidth": 0.5, "axes.edgecolor": "#52514e", "xtick.color": "#52514e",
    "ytick.color": "#52514e", "xtick.major.width": 0.4, "ytick.major.width": 0.4,
    "xtick.major.size": 2, "ytick.major.size": 2, "axes.spines.top": False,
    "axes.spines.right": False, "pdf.fonttype": 42, "legend.frameon": False,
})
GRID, INK2 = "#e6e5e1", "#52514e"
OP = 6  # index of the operating point in each sweep (matches Table 2)
METHODS = [  # (key, label, colour, marker) - M9 takes slot 1, baselines follow in fixed order
    ("M9", "Neural FOXP2", "#2a78d6", "o"),
    ("M4", "Difference-in-means", "#eb6834", "s"),
    ("M5", "DiM + projection", "#1baf7a", "^"),
    ("M6", "LAPE neurons", "#eda100", "D"),
    ("M7", "SAE features", "#e87ba4", "v"),
]
POINTS = [("M2", "best prompt", "P"), ("M3", "system prompt", "X"), ("M8", "LoRA", "*")]
# label placement per panel: M8 by offset; M2/M3 by leader line to empty space (data coords)
M8_OFF = [(6, 0, "left"), None, None]
LEADER = [{"M2": (0.11, 19), "M3": (0.11, 11)},
          {"M2": (0.9, 19), "M3": (0.9, 11), "M8": (2.5, 64)},
          {"M2": (0.16, -0.30), "M3": (0.16, -0.38), "M8": (0.27, 0.07)}]

sw = pd.read_csv(f"{data_dir}/e9_sweeps.csv")
main = pd.read_csv(f"{data_dir}/e8_main.csv")
main["dM"] = main[["L", "G9", "G2", "Q"]].mean(axis=1)
sw["loss"] = -sw["dS"]
main["loss"] = -main["dS"]

fig, axes = plt.subplots(1, 3, figsize=(6.3, 1.85), gridspec_kw={"wspace": 0.32})
panels = [("kl", "full", r"KL drift (nats)", "full-response target rate (%)", "(a)"),
          ("loss", "full", r"utility loss $-\Delta S$ (points)", "full-response target rate (%)", "(b)"),
          ("kl", "dM", r"KL drift (nats)", r"mean $\Delta M$", "(c)")]
for pi, (ax, (xk, yk, xl, yl, tag)) in enumerate(zip(axes, panels)):
    for key, lab, col, mk in reversed(METHODS):  # draw M9 last, on top
        d = sw[sw.method == key].sort_values("step")
        lw = 1.6 if key == "M9" else 1.0
        ax.plot(d[xk], d[yk], color=col, lw=lw, marker=mk, ms=2.2, zorder=3 if key == "M9" else 2)
        op = d[d.step == OP].iloc[0]
        ax.plot(op[xk], op[yk], marker=mk, ms=6, color=col, mec="white", mew=0.8,
                zorder=5 if key == "M9" else 4)
    for key, lab, mk in POINTS:
        r = main[main.method == key].iloc[0]
        ax.plot(r[xk], r[yk], marker=mk, ms=6 if mk != "*" else 8, color="#6f6e69",
                mec="white", mew=0.6, zorder=4)
        if key == "M8" and M8_OFF[pi] is not None:
            dx, dy, ha = M8_OFF[pi]
            ax.annotate(lab, (r[xk], r[yk]), xytext=(dx, dy), textcoords="offset points",
                        fontsize=6, color=INK2, va="center", ha=ha)
        else:
            ax.annotate(lab, (r[xk], r[yk]), xytext=LEADER[pi][key], textcoords="data",
                        fontsize=6, color=INK2, va="center", ha="left",
                        arrowprops=dict(arrowstyle="-", color="#8a8984", lw=0.4,
                                        shrinkA=1, shrinkB=3))
    if yk == "dM":
        ax.axhline(0, color="#8a8984", lw=0.5, zorder=1)
    ax.grid(axis="y", color=GRID, linewidth=0.4)
    ax.set_axisbelow(True)
    ax.set_xlabel(xl, labelpad=1)
    ax.set_ylabel(yl, labelpad=1)
    ax.text(-0.2, 1.04, tag, transform=ax.transAxes, fontweight="bold", fontsize=7)
axes[0].set_xlim(-0.01, 0.40)
axes[2].set_xlim(-0.01, 0.40)
axes[1].set_xlim(-0.1, 4.0)
axes[0].set_ylim(0, 70)
axes[1].set_ylim(0, 70)

handles = [Line2D([], [], color=c, marker=m, ms=3.5, lw=1.6 if k == "M9" else 1.0, label=l)
           for k, l, c, m in METHODS]
handles.append(Line2D([], [], color="#6f6e69", marker="o", ms=5, lw=0, mec="white",
                      label="large marker: operating point"))
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.97), ncol=6,
           fontsize=6.3, handlelength=1.8, columnspacing=1.1)
fig.savefig(f"{out_dir}/fig_tradeoff.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(f"{out_dir}/fig_tradeoff.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
