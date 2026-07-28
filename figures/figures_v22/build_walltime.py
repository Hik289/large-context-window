#!/usr/bin/env python3
"""
build_walltime.pdf/png — Offline build wall time vs. online query latency tradeoff.
Two-panel figure: (left) one-time offline distillation build time by corpus scale
(60M ~9.3h, 250M multi-day, shown as a lower-bound/open-ended bar);
(right) interactive per-query latency breakdown (60M retrieval 0.30s vs ~2.5s total/query).

Data hardcoded from the paper's reported values.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
RED = "#D55E00"
GRAY = "#555555"

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.labelweight": "bold",
    "axes.titlesize": 11,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 8.5,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 4.2))

# ---------------- Panel 1: offline build wall time (one-time) ----------------
build_labels = ["60M tokens", "250M tokens"]
build_hours = [9.3, 60.0]  # 250M shown as an open-ended bar (multi-day floor ~2.5 days = 60h)
bars1 = ax1.bar(build_labels, build_hours, color=[BLUE, ORANGE],
                 edgecolor="black", linewidth=0.8, width=0.55)

# Label bars: exact value for 60M, ">X days" / open-ended for 250M
ax1.text(0, build_hours[0] + 3, "9.3 h", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax1.set_ylabel("Offline build wall time (hours)")
ax1.set_title("One-time offline distillation build", fontsize=11)
ax1.set_ylim(0, max(build_hours) * 1.55)

# Visually indicate the 250M bar is a lower bound / continues beyond frame
ax1.plot([1 - 0.22, 1 + 0.22], [build_hours[1], build_hours[1]], color="black", lw=0.8)
for dx in (-0.14, 0.0, 0.14):
    ax1.annotate("", xy=(1 + dx, build_hours[1] + 10), xytext=(1 + dx, build_hours[1] + 2),
                 arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.2))

# Text label placed clearly above the open-ended arrows (no overlap)
ax1.text(1, build_hours[1] + 15, "multi-day\n(open-ended)", ha="center", va="bottom",
         fontsize=9.5, fontweight="bold", color=RED)
ax1.text(0.5, -0.28, "(heavy, one-time cost per corpus)", ha="center", va="top",
         transform=ax1.transAxes, fontsize=8.5, color=GRAY, style="italic")

# ---------------- Panel 2: online query latency (interactive) ----------------
query_labels = ["Retrieval\n(60M)", "Total\nper query"]
query_seconds = [0.30, 2.5]
bars2 = ax2.bar(query_labels, query_seconds, color=[GREEN, BLUE],
                 edgecolor="black", linewidth=0.8, width=0.55)
ax2.bar_label(bars2, labels=[f"{v:.2f} s" for v in query_seconds], padding=5,
              fontsize=10, fontweight="bold")
ax2.set_ylabel("Query-time latency (seconds)")
ax2.set_title("Interactive online serving", fontsize=11)
ax2.set_ylim(0, max(query_seconds) * 1.35)
ax2.text(0.5, -0.28, "(fast, interactive per-query cost)", ha="center", va="top",
         transform=ax2.transAxes, fontsize=8.5, color=GRAY, style="italic")

fig.suptitle("Build vs. query tradeoff: heavy offline build, interactive online serving",
              fontsize=12, fontweight="bold", y=1.03)
fig.tight_layout()
fig.savefig("build_walltime.pdf")
fig.savefig("build_walltime.png", dpi=200)
print("Saved build_walltime.pdf/.png")
