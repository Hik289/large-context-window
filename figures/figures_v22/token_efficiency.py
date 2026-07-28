"""
token_efficiency.py
FIG — Token vs. accuracy tradeoff across 3 context modes (§5b).
Shows that "mixed" (ours) sits at the efficiency sweet spot: near-detailed_full
accuracy at 3.2x fewer input tokens than detailed_full, while distilled_only
is far cheaper but loses substantial accuracy.

Colorblind-safe: Okabe-Ito palette, consistent with figures_v2 / figures_supp.
Data source: v22_figures_data.md, §5b.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Palette (Okabe-Ito) ──────────────────────────────────────────────────────
BLUE   = "#0072B2"   # distilled_only
ORANGE = "#E69F00"   # mixed (ours) — highlighted
GRAY   = "#555555"   # detailed_full

# ── Data (v22_figures_data.md §5b) ──────────────────────────────────────────
modes   = ["distilled_only", "mixed (ours)", "detailed_full"]
tokens  = [1010, 1958, 6219]
correct = [20.0, 32.0, 39.0]
colors  = [BLUE, ORANGE, GRAY]
sizes   = [180, 320, 180]

# ── Figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5.5))

# connecting line to show the tradeoff trajectory
ax.plot(tokens, correct, color="#999999", linewidth=1.6, linestyle="--",
         zorder=2)

for x, yv, c, s, m in zip(tokens, correct, colors, sizes, modes):
    edge = "#A66E00" if m == "mixed (ours)" else "#2B2B2B"
    marker = "*" if m == "mixed (ours)" else "o"
    ax.scatter(x, yv, s=s * (2.2 if marker == "*" else 1.0), color=c,
               edgecolor=edge, linewidth=1.4, zorder=4, marker=marker)

# ── Annotations ──────────────────────────────────────────────────────────────
ax.annotate("distilled_only\n1010 tok/q, 20.0%",
            xy=(1010, 20.0), xytext=(1010, 8.5),
            ha="center", fontsize=10, color=BLUE, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=BLUE, lw=1.0))

ax.annotate("mixed (ours)\n1958 tok/q, 32.0%\nefficiency sweet spot",
            xy=(1958, 32.0), xytext=(2450, 24.0),
            ha="left", fontsize=10.5, color="#A66E00", fontweight="bold",
            arrowprops=dict(arrowstyle="-", color="#A66E00", lw=1.2))

ax.annotate("detailed_full\n6219 tok/q, 39.0%",
            xy=(6219, 39.0), xytext=(5000, 44.0),
            ha="center", fontsize=10, color=GRAY, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=GRAY, lw=1.0))

# ── Highlight box around "mixed (ours)" ─────────────────────────────────────
ax.scatter([1958], [32.0], s=1400, facecolor="none",
           edgecolor="#E69F00", linewidth=1.4, linestyle=":", zorder=3)

# ── Axes ─────────────────────────────────────────────────────────────────────
ax.set_xlabel("Input Tokens per Query", fontsize=12.5, fontweight="bold",
              labelpad=8)
ax.set_ylabel("Correct (%)", fontsize=12.5, fontweight="bold", labelpad=8)
ax.set_xlim(0, 7200)
ax.set_ylim(0, 50)
ax.grid(True, alpha=0.25, linestyle=":", linewidth=0.8)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)

# ── Title ─────────────────────────────────────────────────────────────────────
ax.set_title("Token vs. Accuracy Tradeoff Across Context Modes",
              fontsize=14.5, fontweight="bold", pad=12)

# ── Footnote ─────────────────────────────────────────────────────────────────
fig.text(0.5, -0.02,
         "\"Mixed (ours)\" achieves 82% of detailed_full's accuracy gain at 3.2x fewer input tokens, "
         "the best accuracy-per-token operating point. Data: v22_figures_data.md \u00a75b.",
         ha="center", fontsize=8.5, color="#777777", style="italic")

fig.tight_layout(pad=0.5)

# ── Save ─────────────────────────────────────────────────────────────────────
out_dir = "./paper/v21/figures_v22"
fig.savefig(f"{out_dir}/token_efficiency.pdf", dpi=300, bbox_inches="tight")
fig.savefig(f"{out_dir}/token_efficiency.png", dpi=200, bbox_inches="tight")
print("\u2713 token_efficiency saved")
