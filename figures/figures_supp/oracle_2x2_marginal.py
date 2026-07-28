"""
oracle_2x2_marginal.py
FIG H — Marginal effects of the 2x2 oracle factorial (v19).

Shows Delta(20M -> 250M) Correct% for each of the 4 cells (model x doc-source)
as a 2x2 heatmap, plus a marginal-effect bar panel comparing the doc-source
axis (gold vs retrieved, averaged over model) against the model axis
(mini vs full, averaged over doc-source).

Conveys: the doc-source axis dominates (~-11.25% avg swing), the model axis
is comparatively minor (~-2.4% avg swing) -- doc-source is the true driver,
not model capacity.

Colorblind-safe: Okabe-Ito palette, consistent with figures_v2 / v17-v18 style.
Data source: v19_oracle_vis_directive.md (oracle 2x2x3 factorial, exact numbers).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# ── Palette (Okabe-Ito) ──────────────────────────────────────────────────────
BLUE   = "#0072B2"
ORANGE = "#E69F00"
GRAY   = "#555555"
GREEN  = "#009E73"

# ── Raw data (Correct%, 20M / 60M / 250M) ───────────────────────────────────
mini_gold      = [82.50, 84.00, 82.25]
mini_retrieved = [84.00, 78.25, 73.50]
full_gold      = [85.75, 85.00, 83.75]
full_retrieved = [70.25, 66.00, 56.25]

# Delta(20M -> 250M) for each cell
d_mini_gold      = mini_gold[2]      - mini_gold[0]        # -0.25
d_mini_retrieved = mini_retrieved[2] - mini_retrieved[0]    # -10.50
d_full_gold      = full_gold[2]      - full_gold[0]         # -2.00
d_full_retrieved = full_retrieved[2] - full_retrieved[0]    # -14.00

# Marginal effects
doc_source_gold_avg      = (d_mini_gold + d_full_gold) / 2            # -1.125
doc_source_retrieved_avg = (d_mini_retrieved + d_full_retrieved) / 2  # -12.25
model_mini_avg = (d_mini_gold + d_mini_retrieved) / 2                 # -5.375
model_full_avg = (d_full_gold + d_full_retrieved) / 2                 # -8.0

# ── Figure: 1x2 panel (heatmap | marginal bars) ─────────────────────────────
fig, (ax_hm, ax_bar) = plt.subplots(1, 2, figsize=(12.5, 5.6),
                                      gridspec_kw={"width_ratios": [1.0, 1.15]})

# ─────────────────────────── Panel 1: 2x2 heatmap ──────────────────────────
delta_matrix = np.array([
    [d_mini_gold,  d_mini_retrieved],
    [d_full_gold,  d_full_retrieved],
])  # rows: model (mini, full); cols: doc-source (gold, retrieved)

vmin, vmax = -16, 0
cmap = mcolors.LinearSegmentedColormap.from_list(
    "okabe_diverge", ["#B30000", "#F2D9C4", "#FFFFFF"], N=256
)
im = ax_hm.imshow(delta_matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

ax_hm.set_xticks([0, 1])
ax_hm.set_xticklabels(["Gold", "Retrieved"], fontsize=12.5, fontweight="bold")
ax_hm.set_yticks([0, 1])
ax_hm.set_yticklabels(["compact model", "large model"], fontsize=12.5, fontweight="bold")
ax_hm.set_xlabel("Doc-source", fontsize=12.5, fontweight="bold", labelpad=8)
ax_hm.set_ylabel("Model", fontsize=12.5, fontweight="bold", labelpad=8)

for i in range(2):
    for j in range(2):
        val = delta_matrix[i, j]
        txt_color = "white" if val < -8 else "#333333"
        ax_hm.text(j, i, f"{val:+.2f}%", ha="center", va="center",
                   fontsize=16, fontweight="bold", color=txt_color)

ax_hm.set_title("\u0394 Correct% (20M \u2192 250M)\nper cell", fontsize=13, fontweight="bold", pad=10)

# Gridlines between cells
ax_hm.set_xticks(np.arange(-0.5, 2, 1), minor=True)
ax_hm.set_yticks(np.arange(-0.5, 2, 1), minor=True)
ax_hm.grid(which="minor", color="white", linestyle="-", linewidth=3)
ax_hm.tick_params(which="minor", bottom=False, left=False)

cbar = fig.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.06)
cbar.set_label("\u0394 Correct% (20M\u2192250M)", fontsize=10)
cbar.ax.tick_params(labelsize=9)

# ─────────────────────── Panel 2: marginal effect bars ─────────────────────
labels = ["Gold\n(doc-source)", "Retrieved\n(doc-source)", "Mini\n(model)", "Full\n(model)"]
values = [doc_source_gold_avg, doc_source_retrieved_avg, model_mini_avg, model_full_avg]
colors = [GRAY, GRAY, ORANGE, BLUE]
alphas = [0.55, 0.95, 0.85, 0.85]

xpos = np.arange(4)
bars = ax_bar.bar(xpos, values, width=0.62, color=colors, zorder=3,
                   edgecolor="#333333", linewidth=0.9)
for b, a in zip(bars, alphas):
    b.set_alpha(a)

for xi, v in zip(xpos, values):
    ax_bar.text(xi, v - 0.4, f"{v:+.2f}%", ha="center",
                va="top" if v < 0 else "bottom",
                fontsize=12, fontweight="bold",
                color="white" if abs(v) > 4 else "#222222")

ax_bar.axhline(0, color="#333333", linewidth=1.0, zorder=2)
ax_bar.set_xticks(xpos)
ax_bar.set_xticklabels(labels, fontsize=11.5, fontweight="bold")
ax_bar.set_ylabel("Avg. \u0394 Correct% (20M\u2192250M)", fontsize=12.5, fontweight="bold")
ax_bar.set_ylim(-16, 2)
ax_bar.yaxis.grid(True, alpha=0.25, linestyle=":", linewidth=0.8)
ax_bar.set_axisbelow(True)
ax_bar.spines[["top", "right"]].set_visible(False)
ax_bar.set_title("Marginal effect: doc-source axis dominates,\nmodel axis is minor",
                  fontsize=13, fontweight="bold", pad=10)

# Divider + group labels to visually separate doc-source vs model marginals
ax_bar.axvline(1.5, color="#cccccc", linewidth=1.2, linestyle="--", zorder=1)
ax_bar.text(0.5, 1.2, "DOC-SOURCE AXIS", ha="center", fontsize=9.5,
            color=GRAY, fontweight="bold")
ax_bar.text(2.5, 1.2, "MODEL AXIS", ha="center", fontsize=9.5,
            color="#555555", fontweight="bold")

# Callout: magnitude comparison
ax_bar.annotate(
    "~11\u00d7 larger swing",
    xy=(1, doc_source_retrieved_avg + 0.3), xytext=(2.0, -13.5),
    fontsize=10.5, fontweight="bold", color="#00543D", ha="center",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#E6F7F2",
              edgecolor=GREEN, linewidth=1.2),
    arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.6,
                     connectionstyle="arc3,rad=-0.25"),
)

fig.suptitle(
    "Oracle 2\u00d72 Factorial: Marginal Effects Isolate Doc-Source as the True Driver",
    fontsize=15, fontweight="bold", y=1.03,
)

fig.text(0.5, -0.06,
         "Deltas computed from Correct% at 20M vs. 250M corpus size, oracle 2\u00d72\u00d73 factorial "
         "(large vs. compact model \u00d7 gold vs. retrieved documents).",
         ha="center", fontsize=8.5, color="#777777", style="italic")

fig.tight_layout(pad=1.0)

# ── Save ─────────────────────────────────────────────────────────────────────
out_dir = "./paper/v19/figures_supp"
fig.savefig(f"{out_dir}/oracle_2x2_marginal.pdf", dpi=300, bbox_inches="tight")
fig.savefig(f"{out_dir}/oracle_2x2_marginal.png", dpi=200, bbox_inches="tight")
print("\u2713 oracle_2x2_marginal saved")
