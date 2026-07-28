"""
token_breakdown.py
FIG — Per-query token breakdown: question input vs. answer output (§5a).
Shows mean token counts with median/p95 context annotated, illustrating the
right-skewed distribution of answer length (long tail of detailed answers).

Colorblind-safe: Okabe-Ito palette, consistent with figures_v2 / figures_supp.
Data source: v22_figures_data.md, §5a.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Palette (Okabe-Ito) ──────────────────────────────────────────────────────
BLUE   = "#0072B2"   # question
ORANGE = "#E69F00"   # answer

# ── Data (v22_figures_data.md §5a) ──────────────────────────────────────────
categories = ["Question\n(input)", "Answer\n(output)"]
means   = [37, 439]
medians = [34, 221]
p95s    = [63, 1622]
colors  = [BLUE, ORANGE]

x = np.arange(len(categories))
width = 0.5

# ── Figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8.2, 5.5))

bars = ax.bar(x, means, width, color=colors, zorder=3,
              edgecolor="#2B2B2B", linewidth=0.9)

# ── p95 range markers (whisker showing median -> p95 spread) ────────────────
for xi, med, p95, c in zip(x, medians, p95s, ["#004C77", "#A66E00"]):
    ax.vlines(xi + width / 2 + 0.06, med, p95, color=c, linewidth=1.8, zorder=4)
    ax.hlines(med, xi + width / 2 - 0.02, xi + width / 2 + 0.14, color=c,
              linewidth=1.8, zorder=4)
    ax.hlines(p95, xi + width / 2 - 0.02, xi + width / 2 + 0.14, color=c,
              linewidth=1.8, zorder=4)

# ── Value labels ──────────────────────────────────────────────────────────────
for bar, m, med, p95 in zip(bars, means, medians, p95s):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + max(means) * 0.02,
             f"mean {m}", ha="center", va="bottom", fontsize=11.5,
             fontweight="bold", color="#1a1a1a")
    # nudge median label down slightly (log scale) to avoid crowding the bar top
    med_label_y = med * 0.82
    ax.text(bar.get_x() + bar.get_width() / 2 + 0.40, med_label_y,
             f"median: {med}", ha="left", va="center", fontsize=9,
             color="#444444")
    ax.text(bar.get_x() + bar.get_width() / 2 + 0.40, p95,
             f"p95: {p95}", ha="left", va="center", fontsize=9,
             color="#444444")

# ── Axes ─────────────────────────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=12.5, fontweight="bold")
ax.set_ylabel("Tokens per Query", fontsize=12.5, fontweight="bold", labelpad=8)
ax.set_yscale("log")
ax.set_ylim(10, 3000)
ax.set_xlim(-0.6, 1.9)  # extra right margin for offset whisker labels
ax.yaxis.grid(True, alpha=0.25, linestyle=":", linewidth=0.8, which="both")
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)

# ── Title ─────────────────────────────────────────────────────────────────────
ax.set_title("Per-Query Token Breakdown: Question vs. Answer",
              fontsize=14.5, fontweight="bold", pad=12)

# ── Reserve bottom margin, then place footnote within that margin ───────────
fig.subplots_adjust(bottom=0.30)

# ── Footnote ─────────────────────────────────────────────────────────────────
fig.text(0.5, 0.03,
         "Bars show mean tokens (log scale); whiskers span median\u2192p95, illustrating\n"
         "the right-skewed tail of longer answers. Retrieved chunks/q: mean 34.3 (not shown).\n"
         "Data: v22_figures_data.md \u00a75a.",
         ha="center", va="bottom", fontsize=8.5, color="#777777", style="italic")

# ── Save ─────────────────────────────────────────────────────────────────────
out_dir = "./paper/v21/figures_v22"
fig.savefig(f"{out_dir}/token_breakdown.pdf", dpi=300)
fig.savefig(f"{out_dir}/token_breakdown.png", dpi=200)
print("\u2713 token_breakdown saved")
