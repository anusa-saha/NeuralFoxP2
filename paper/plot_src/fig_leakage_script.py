"""Figures fig:leakage (E11) and fig:app-script (E11/E12).

usage: python fig_leakage_script.py <data_dir> <out_dir>
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_rgb
import pandas as pd
import numpy as np

data_dir, out_dir = sys.argv[1], sys.argv[2]
for f in ("LiberationSerif-Regular.ttf", "LiberationSerif-Bold.ttf", "LiberationSerif-Italic.ttf"):
    font_manager.fontManager.addfont(f"/usr/share/fonts/truetype/liberation/{f}")
plt.rcParams.update({
    "font.family": "Liberation Serif", "mathtext.fontset": "stix", "font.size": 6.5,
    "axes.linewidth": 0.5, "axes.edgecolor": "#52514e", "xtick.color": "#52514e",
    "ytick.color": "#52514e", "xtick.major.width": 0.4, "ytick.major.width": 0.4,
    "xtick.major.size": 2, "ytick.major.size": 2, "pdf.fonttype": 42, "legend.frameon": False,
})
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
LANGS = ["es", "fr", "zh", "ru", "hi", "ar"]

# ------------------------------------------------------------ Figure 6: leakage
m = pd.read_csv(f"{data_dir}/e11_leakage.csv", index_col=0)
cols = LANGS + ["English"]
M = m.loc[LANGS, cols].values
# Same diverging scale as Figure 3: red = toward English / decrease, blue = increase.
cmap = LinearSegmentedColormap.from_list("div", ["#8f2524", "#e34948", "#f0efec", "#3987e5", "#104281"])
norm = TwoSlopeNorm(vmin=-45, vcenter=0, vmax=45)
fig, ax = plt.subplots(figsize=(3.03, 1.95))
im = ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
ax.set_xticks([x - 0.5 for x in range(1, 7)], minor=True)
ax.set_yticks([y - 0.5 for y in range(1, 6)], minor=True)
ax.grid(which="minor", color="white", linewidth=1.2)
ax.tick_params(which="both", length=0)
ax.axvline(5.5, color="white", lw=3.0)  # separate English column
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        r, g, b = to_rgb(cmap(norm(M[i, j])))
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        ax.text(j, i, f"{M[i, j]:+.1f}".replace("-", "−"), ha="center", va="center",
                fontsize=6, color="white" if lum < 0.45 else INK,
                fontweight="bold" if i == j else "normal")
ax.set_xticks(range(7), ["es", "fr", "zh", "ru", "hi", "ar", "en"])
ax.set_yticks(range(6), [f"{l} edit" for l in LANGS])
ax.xaxis.tick_top()
ax.set_xlabel("measured language", labelpad=3)
ax.xaxis.set_label_position("top")
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03, aspect=18)
cb.set_label("change (pp)", fontsize=6.5)
cb.ax.tick_params(labelsize=6, width=0.4, length=2)
cb.outline.set_linewidth(0.4)
cb.set_ticks([-40, -20, 0, 20, 40])
fig.savefig(f"{out_dir}/fig_leakage.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(f"{out_dir}/fig_leakage.png", dpi=300, bbox_inches="tight", pad_inches=0.02)

# ------------------------------------------------------------ Figure: script control
rom = pd.read_csv(f"{data_dir}/e12_romanized.csv")
SHORT = ["Llama-3.1-8B", "Gemma-2-9B", "Gemma-2-2B", "Qwen3-8B"]
C1, C2 = "#2a78d6", "#eb6834"  # categorical slots 1, 2
fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.3, 1.65), gridspec_kw={"width_ratios": [1, 1.6], "wspace": 0.25})

# (a) same-script control: bars grouped by edit, colour = measured language
w = 0.36
for k, (meas, col) in enumerate([("es", C1), ("fr", C2)]):
    vals = [m.loc["es", meas], m.loc["fr", meas]]
    xs = np.arange(2) + (k - 0.5) * w
    bars = axa.bar(xs, vals, width=w - 0.04, color=col, label=f"measured {meas}", zorder=2)
    for x, v in zip(xs, vals):
        axa.text(x, v + 1.0, f"+{v:.1f}", ha="center", va="bottom", fontsize=6, color=INK2)
axa.set_xticks(range(2), ["Spanish edit", "French edit"])
axa.set_ylabel("change (pp)", labelpad=1)
axa.set_ylim(0, 50)
axa.legend(loc="upper right", fontsize=6, handlelength=1.0, ncol=2, bbox_to_anchor=(1.0, 1.12))
axa.text(-0.17, 1.04, "(a)", transform=axa.transAxes, fontweight="bold", fontsize=7)

# (b) romanized mass under native-script edits, per model
x = np.arange(4)
for k, (lang, col, lab) in enumerate([("hi", C1, "Hindi edit"), ("ru", C2, "Russian edit")]):
    xs = x + (k - 0.5) * w
    axb.bar(xs, rom[lang], width=w - 0.04, color=col, label=lab, zorder=2)
    for xx, v in zip(xs, rom[lang]):
        axb.text(xx, v + 0.1, f"+{v:.1f}", ha="center", va="bottom", fontsize=6, color=INK2)
axb.set_xticks(x, SHORT)
axb.set_ylabel("romanized mass change (pp)", labelpad=1)
axb.set_ylim(0, 5.6)
axb.legend(loc="upper right", fontsize=6, handlelength=1.0, ncol=2, bbox_to_anchor=(1.0, 1.12))
axb.text(-0.10, 1.04, "(b)", transform=axb.transAxes, fontweight="bold", fontsize=7)
for ax in (axa, axb):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)
fig.savefig(f"{out_dir}/fig_app_script_control.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(f"{out_dir}/fig_app_script_control.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
