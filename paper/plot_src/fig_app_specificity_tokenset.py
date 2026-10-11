"""Figures fig:app-specificity (E24) and fig:app-tokenset (E25).

usage: python fig_app_specificity_tokenset.py <data_dir> <out_dir>
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_rgb
import numpy as np
import pandas as pd

data_dir, out_dir = sys.argv[1], sys.argv[2]
for f in ("LiberationSerif-Regular.ttf", "LiberationSerif-Bold.ttf", "LiberationSerif-Italic.ttf"):
    font_manager.fontManager.addfont(f"/usr/share/fonts/truetype/liberation/{f}")
plt.rcParams.update({
    "font.family": "Liberation Serif", "mathtext.fontset": "stix", "font.size": 7,
    "axes.linewidth": 0.5, "axes.edgecolor": "#52514e", "xtick.color": "#52514e",
    "ytick.color": "#52514e", "xtick.major.width": 0.4, "ytick.major.width": 0.4,
    "xtick.major.size": 2, "ytick.major.size": 2, "pdf.fonttype": 42, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False,
})
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # models, fixed order as in other figures
MARK = ["o", "s", "^", "D"]
MODELS = ["Llama-3.1-8B-Instruct", "Gemma-2-9B-it", "Gemma-2-2B-it", "Qwen3-8B"]
SHORT = ["Llama-3.1-8B", "Gemma-2-9B", "Gemma-2-2B", "Qwen3-8B"]
LANGS = ["es", "fr", "zh", "ru", "hi", "ar"]

# ---------------------------------------------------- Figure 10: specificity
s = pd.read_csv(f"{data_dir}/e24_specificity.csv", index_col=0).loc[LANGS, LANGS]
M = s.values
cmap = LinearSegmentedColormap.from_list("div", ["#8f2524", "#e34948", "#f0efec", "#3987e5", "#104281"])
norm = TwoSlopeNorm(vmin=-0.8, vcenter=0, vmax=0.8)  # same scale as Figure 3
fig, ax = plt.subplots(figsize=(3.03, 1.95))
im = ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
ax.set_xticks([x - 0.5 for x in range(1, 6)], minor=True)
ax.set_yticks([y - 0.5 for y in range(1, 6)], minor=True)
ax.grid(which="minor", color="white", linewidth=1.2)
ax.tick_params(which="both", length=0)
for i in range(6):
    for j in range(6):
        r, g, b = to_rgb(cmap(norm(M[i, j])))
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        ax.text(j, i, f"{M[i, j]:+.2f}".replace("-", "−"), ha="center", va="center", fontsize=6.3,
                color="white" if lum < 0.45 else INK, fontweight="bold" if i == j else "normal")
ax.set_xticks(range(6), LANGS)
ax.set_yticks(range(6), [rf"$\Delta M_{{\mathrm{{{l}}}}}$" for l in LANGS])
ax.xaxis.tick_top()
ax.set_xlabel("language of the response", labelpad=3)
ax.xaxis.set_label_position("top")
for sp in ax.spines.values():
    sp.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03, aspect=18)
cb.set_label(r"mean $\Delta M_a$", fontsize=6.5)
cb.ax.tick_params(labelsize=6, width=0.4, length=2)
cb.outline.set_linewidth(0.4)
cb.set_ticks([-0.8, -0.4, 0, 0.4, 0.8])
fig.savefig(f"{out_dir}/fig_app_specificity.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(f"{out_dir}/fig_app_specificity.png", dpi=300, bbox_inches="tight", pad_inches=0.02)

# ---------------------------------------------------- Figure 11: token sets
c = pd.read_csv(f"{data_dir}/e25_constructions.csv").set_index("model").loc[MODELS]
d = pd.read_csv(f"{data_dir}/e25_dropout.csv").set_index("model").loc[MODELS]
n = pd.read_csv(f"{data_dir}/e25_null.csv").set_index("model").loc[MODELS]
fig, (a, b, cc) = plt.subplots(1, 3, figsize=(6.3, 1.75), gridspec_kw={"wspace": 0.38})

cons = [("script", "Script"), ("diagnostic", "Diag."), ("translit", "Translit."), ("freq", "Freq.")]
x = np.arange(len(cons))
for k, m in enumerate(MODELS):
    off = (k - 1.5) * 0.12
    a.plot(x + off, [c.loc[m, key] for key, _ in cons], ls="none", marker=MARK[k], ms=3.6,
           color=SLOT[k], label=SHORT[k])
a.plot(x, [c[key].mean() for key, _ in cons], ls="none", marker="_", ms=14, mew=1.2, color=INK, label="mean")
a.set_xticks(x, [lab for _, lab in cons], fontsize=6.5)
a.set_xlabel("token-set construction", labelpad=1)
a.set_ylim(0.40, 0.65)
a.set_ylabel("DEFAULT under Neural FOXP2", labelpad=1)

cols = ["0", "10", "20", "30", "40", "50"]
xs = [int(v) for v in cols]
for k, m in enumerate(MODELS):
    b.plot(xs, d.loc[m, cols].values, color=SLOT[k], marker=MARK[k], ms=2.8, lw=1.0, label=SHORT[k])
b.set_xticks(xs)
b.set_xlabel("diagnostic tokens removed (%)", labelpad=1)
b.set_ylabel("DEFAULT retained (%)", labelpad=1)
b.set_ylim(50, 102)

y = np.arange(len(MODELS))[::-1]
for k, m in enumerate(MODELS):
    yy = y[k]
    cc.errorbar(n.loc[m, "null_mean"], yy, xerr=n.loc[m, "null_sd"], fmt="o", ms=3, color="#8a8984",
                ecolor="#8a8984", elinewidth=0.8, capsize=1.5, label="random-set null (mean ± SD)" if k == 0 else None)
    cc.plot(n.loc[m, "null_p95"], yy, marker="|", ms=7, mew=1.2, color=INK2, ls="none",
            label="null 95th percentile" if k == 0 else None)
    cc.plot(n.loc[m, "real"], yy, marker=MARK[k], ms=4.5, color=SLOT[k], ls="none")
cc.set_yticks(y, ["L-8B", "G-9B", "G-2B", "Q-8B"])
cc.set_xlim(-0.05, 0.7)
cc.set_xlabel("DEFAULT", labelpad=1)
cc.legend(loc="center", fontsize=5.6, handlelength=1.0, bbox_to_anchor=(0.5, 0.5))

for ax, tag in zip((a, b, cc), ("(a)", "(b)", "(c)")):
    ax.grid(axis="y" if ax is not cc else "x", color=GRID, linewidth=0.4)
    ax.set_axisbelow(True)
    ax.text(-0.22 if ax is not cc else -0.25, 1.04, tag, transform=ax.transAxes, fontweight="bold", fontsize=7)
h, l = b.get_legend_handles_labels()
fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=4, fontsize=6.3, handlelength=1.6)
fig.savefig(f"{out_dir}/fig_app_tokenset.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(f"{out_dir}/fig_app_tokenset.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
