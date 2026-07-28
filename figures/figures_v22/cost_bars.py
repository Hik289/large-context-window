#!/usr/bin/env python3
"""
cost_bars.pdf/png — Token cost summary: total cost over 400 queries, and per-query cost.
Data hardcoded from the paper's reported values (total=$3.75 for 400 queries, per-query=$0.0094).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

BLUE = "#0072B2"
ORANGE = "#E69F00"

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.labelweight": "bold",
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 4.0))

# Panel 1: total cost
total_cost = 3.75
n_queries = 400
bar1 = ax1.bar(["Total cost"], [total_cost], color=BLUE,
                edgecolor="black", linewidth=0.8, width=0.5)
ax1.set_ylabel("Cost (USD)")
ax1.set_ylim(0, total_cost * 1.35)
ax1.bar_label(bar1, labels=[f"${total_cost:.2f}"], padding=6, fontsize=11, fontweight="bold")
ax1.text(0.5, -0.16, f"({n_queries} queries)", ha="center", va="top",
         fontsize=8.5, color="#555555", transform=ax1.transAxes, style="italic")

# Panel 2: per-query cost
per_query = 0.0094
bar2 = ax2.bar(["Per-query cost"], [per_query], color=ORANGE,
                edgecolor="black", linewidth=0.8, width=0.5)
ax2.set_ylabel("Cost (USD)")
ax2.set_ylim(0, per_query * 1.35)
ax2.bar_label(bar2, labels=[f"${per_query:.4f}"], padding=6, fontsize=11, fontweight="bold")

fig.suptitle("Token cost of DDI-v3xf memory queries", fontsize=12, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig("cost_bars.pdf")
fig.savefig("cost_bars.png", dpi=200)
print("Saved cost_bars.pdf/.png")
