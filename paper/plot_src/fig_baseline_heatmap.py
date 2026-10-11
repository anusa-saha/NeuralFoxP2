"""Figure fig:baseline - unedited models, mean Delta M (colour) and full-response target rate (text)."""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_rgb
import pandas as pd

data_dir, out = sys.argv[1], sys.argv[2]
for f in ("LiberationSerif-Regular.ttf", "LiberationSerif-Bold.ttf", "LiberationSerif-Italic.ttf"):
    font_manager.fontManager.addfont(f"/usr/share/fonts/truetype/liberation/{f}")
plt.rcParams.update({
    "font.family": "Liberation Serif", "mathtext.fontset": "stix",
    "font.size": 6.5, "axes.linewidth": 0.5, "pdf.fonttype": 42,
})

LANGS = ["es", "fr", "zh", "ru", "hi", "ar"]
MODELS = ["Llama-3.1-8B-Instruct", "Gemma-2-9B-it", "Gemma-2-2B-it", "Qwen3-8B"]
SHORT = ["Llama-3.1-8B", "Gemma-2-9B", "Gemma-2-2B", "Qwen3-8B"]
shares = pd.read_csv(f"{data_dir}/e1a_shares.csv")
dm = pd.read_csv(f"{data_dir}/e1b_deltaM.csv")

# Diverging: red = English-leaning (negative), grey midpoint at 0, blue = target-leaning (positive).
cmap = LinearSegmentedColormap.from_list(
    "div", ["#8f2524", "#e34948", "#f0efec", "#3987e5", "#104281"])
norm = TwoSlopeNorm(vmin=-0.8, vcenter=0.0, vmax=0.8)

fig, axes = plt.subplots(2, 1, figsize=(3.03, 1.78), sharex=True)
for ax, cond in zip(axes, ["C1", "C2"]):
    M = dm[dm.cond == cond].set_index("model").loc[MODELS, LANGS].values
    R = shares[shares.cond == cond].set_index("model").loc[MODELS, LANGS].values
    im = ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
    # 2px-style surface gap between cells
    ax.set_xticks([x - 0.5 for x in range(1, 6)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, 4)], minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="both", length=0)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            r, g, b = to_rgb(cmap(norm(M[i, j])))
            lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
            ax.text(j, i, f"{R[i, j]:.1f}", ha="center", va="center", fontsize=6,
                    color="white" if lum < 0.45 else "#0b0b0b")
    ax.set_yticks(range(4), SHORT)
    ax.set_xticks(range(6), LANGS)
    for s in ax.spines.values():
        s.set_visible(False)
    label = "No-Cue: English prompt" if cond == "C1" else "Neutral: language-neutral prompt"
    ax.set_title(label, fontsize=6.5, loc="left", pad=2)

cb = fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.05, pad=0.03, aspect=20)
cb.set_label(r"mean $\Delta M$", fontsize=6.5)
cb.ax.tick_params(labelsize=6, width=0.4, length=2)
cb.outline.set_linewidth(0.4)
cb.set_ticks([-0.8, -0.4, 0, 0.4, 0.8])
fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
fig.savefig(out.replace(".pdf", ".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
