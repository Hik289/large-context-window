#!/usr/bin/env python3
"""
prelim_evolution.pdf/png — Preliminary study evolution.
Plots Overall (Combined) % across the exploration trajectory of methods,
highlighting DDI_v4 as a failed branch and DDI_v3xf as the final converged method.

Data source: hardcoded (narrative figure; values from the paper) (this is a narrative
trajectory across distinct preliminary runs, not a single results.json table).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import FancyArrowPatch

# ---- Okabe-Ito colorblind-safe palette ----
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
RED = "#D55E00"
GRAY = "#555555"

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.labelweight": "bold",
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "lines.linewidth": 2.0,
    "lines.markersize": 7,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# Ordered trajectory (x-axis left -> right)
labels = ["HDM", "CDM", "Combined", "DDI_Orig", "DDI_Improved",
          "DDI_v3", "DDI_v4", "DDI_v3xe", "DDI_v3xf"]
values = [60.48, 59.16, 66.03, 67.42, 72.44, 74.46, 66.40, 78.29, 78.29]

x = list(range(len(labels)))
fail_idx = labels.index("DDI_v4")
final_idx = labels.index("DDI_v3xf")

fig, ax = plt.subplots(figsize=(8.2, 4.6))

# Main trajectory line (draw as separate segments so the failed branch pops out
# without implying a smooth continuation through it)
main_x = [i for i in x if i != fail_idx]
main_y = [values[i] for i in x if i != fail_idx]

ax.plot(x, values, color=BLUE, linewidth=1.4, linestyle="-", zorder=1, alpha=0.55)

# Overlay markers: normal points in blue, failed branch in red/hatched
for i, (xi, yi) in enumerate(zip(x, values)):
    if i == fail_idx:
        continue
    elif i == final_idx:
        ax.scatter(xi, yi, s=110, color=GREEN, edgecolor="black", linewidth=1.0,
                   zorder=3, marker="*", label="Final converged method" if i == final_idx else None)
    else:
        ax.scatter(xi, yi, s=60, color=BLUE, edgecolor="black", linewidth=0.6, zorder=3)

# Failed branch marker: red with hatch pattern (use scatter + overlay marker with hatch via Patch)
ax.scatter(fail_idx, values[fail_idx], s=140, facecolor=RED, edgecolor="black",
           linewidth=1.2, marker="X", zorder=4, label="Failed branch")

# Dashed connector from DDI_v3 to DDI_v4 (the failed branch) to show it's a divergent attempt
ddi_v3_idx = labels.index("DDI_v3")
ax.plot([ddi_v3_idx, fail_idx], [values[ddi_v3_idx], values[fail_idx]],
        color=RED, linestyle="--", linewidth=1.6, zorder=2, alpha=0.8)

# Value labels above each point
for i, (xi, yi) in enumerate(zip(x, values)):
    offset = 8
    va = "bottom"
    if i == fail_idx:
        offset = -14
        va = "top"
    ax.annotate(f"{yi:.2f}%", (xi, yi), textcoords="offset points",
                xytext=(0, offset), ha="center", va=va, fontsize=8.5,
                color=(RED if i == fail_idx else GRAY))

# Annotate "failed branch" near DDI_v4
ax.annotate("failed branch", xy=(fail_idx, values[fail_idx]),
            xytext=(fail_idx + 0.15, values[fail_idx] - 9),
            fontsize=9, color=RED, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.2),
            ha="left", va="top")

# Annotate final converged method near DDI_v3xf
ax.annotate("final converged\nmethod", xy=(final_idx, values[final_idx]),
            xytext=(final_idx - 1.7, values[final_idx] + 4.5),
            fontsize=9, color=GREEN, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.2),
            ha="center", va="bottom")

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=20, ha="right")
ax.set_ylabel("Overall (Combined) %")
ax.set_title("Preliminary exploration converging on DDI-v3xf")
ax.set_ylim(50, 85)
ax.set_xlim(-0.5, len(labels) - 0.5)

legend_elems = [
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=BLUE,
               markeredgecolor="black", markersize=8, label="Exploration step"),
    plt.Line2D([0], [0], marker="X", color="w", markerfacecolor=RED,
               markeredgecolor="black", markersize=10, label="Failed branch (DDI_v4)"),
    plt.Line2D([0], [0], marker="*", color="w", markerfacecolor=GREEN,
               markeredgecolor="black", markersize=13, label="Final converged (DDI-v3xf)"),
]
ax.legend(handles=legend_elems, loc="lower right", frameon=True, framealpha=0.9)

fig.tight_layout()
fig.savefig("prelim_evolution.pdf")
fig.savefig("prelim_evolution.png", dpi=200)
print("Saved prelim_evolution.pdf/.png")
