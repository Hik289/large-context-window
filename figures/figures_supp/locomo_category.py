"""
locomo_category.py
FIG F — LoCoMo per-category Correct% contrast.
Illustrates the structural capacity limit WITHIN a single benchmark:
Category 4 (open-domain) 70.7% vs Category 2 (multi-hop) 7.0%.

Gap is annotated in "%" units (the maintainer's standing rule: never use "pp").
Colorblind-safe: Okabe-Ito palette, consistent with figures_v2 style.
Data source: paper v16 (exact numbers, no reload needed).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Palette (Okabe-Ito, matches figures_v2) ─────────────────────────────────
BLUE  = "#0072B2"   # cat-4 open-domain (strong)
GRAY  = "#555555"   # cat-2 multi-hop (fails)
DELTA_CLR = "#D55E00"  # gap annotation — vermillion, WCAG-pass on white

# ── Data (paper v16, exact) ──────────────────────────────────────────────────
categories = ["Category 4\n(Open-Domain)", "Category 2\n(Multi-Hop)"]
values     = [70.7, 7.0]
colors     = [BLUE, GRAY]
gap        = values[0] - values[1]   # 63.7

x = np.arange(len(categories))
width = 0.5

# ── Figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.5, 6))

bars = ax.bar(x, values, width, color=colors, zorder=3,
              edgecolor=["#004C77", "#2B2B2B"], linewidth=0.9)

# ── Value labels above bars ──────────────────────────────────────────────────
for bar, val in zip(bars, values):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2,
             h + 1.8,
             f"{val:.1f}%",
             ha="center", va="bottom",
             fontsize=15, fontweight="bold", color="#222222")

# ── Gap bracket + annotation ─────────────────────────────────────────────────
bracket_x = x[1] + width / 2 + 0.15
y_top, y_bot = values[0], values[1]

ax.annotate(
    "", xy=(bracket_x, y_top), xytext=(bracket_x, y_bot),
    arrowprops=dict(arrowstyle="-", color=DELTA_CLR, linewidth=1.4,
                    shrinkA=0, shrinkB=0),
)
# bracket caps
ax.plot([bracket_x - 0.05, bracket_x], [y_top, y_top], color=DELTA_CLR, linewidth=1.4)
ax.plot([bracket_x - 0.05, bracket_x], [y_bot, y_bot], color=DELTA_CLR, linewidth=1.4)

ax.text(bracket_x + 0.12, (y_top + y_bot) / 2,
        f"Gap:\n{gap:.1f}%",
        ha="left", va="center", fontsize=13, fontweight="bold", color=DELTA_CLR)

# ── Axes ─────────────────────────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=12.5, fontweight="bold")
ax.set_ylabel("Correct (%)", fontsize=13, fontweight="bold")
ax.set_ylim(0, 92)
ax.set_xlim(-0.6, bracket_x + 0.9)
ax.tick_params(axis="y", labelsize=11)
ax.yaxis.grid(True, alpha=0.25, linestyle=":", linewidth=0.8)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)

# ── Title ─────────────────────────────────────────────────────────────────────
ax.set_title(
    "Structural Capacity Limit Within LoCoMo",
    fontsize=15, fontweight="bold", pad=12,
)
ax.set_xlabel("LoCoMo Question Category", fontsize=12, fontweight="bold", labelpad=8)

# ── Footnote ─────────────────────────────────────────────────────────────────
fig.text(0.5, -0.01,
         "Open-domain questions (cat-4) remain well within retrieval capacity; multi-hop "
         "questions (cat-2) exceed it, reproducing the capacity-vs-noise boundary within a "
         "single benchmark. Data from paper v16.",
         ha="center", fontsize=8.5, color="#777777", style="italic")

fig.tight_layout(pad=0.5)

# ── Save ─────────────────────────────────────────────────────────────────────
out_dir = "./paper/v17/figures_supp"
fig.savefig(f"{out_dir}/locomo_category.pdf", dpi=300, bbox_inches="tight")
fig.savefig(f"{out_dir}/locomo_category.png", dpi=200, bbox_inches="tight")
print("✓ locomo_category saved")
