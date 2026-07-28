"""
oracle_2x2_lines.py
FIG G — HERO figure for the 2x2 oracle factorial (v19).

4-line scaling comparison across corpus tiers (20M / 60M / 250M) showing
the capacity-vs-noise falsification: gold-context lines stay flat for BOTH
models, while retrieved-context lines collapse for BOTH models -- and the
full (stronger) model's retrieved line collapses MORE than the mini model's,
which is inconsistent with a "not enough capacity" (mini) explanation.

Encoding:
  - Line STYLE: solid = gold docs, dashed = retrieved docs
  - Color:      dark blue = large model
                light orange = compact model

Colorblind-safe: Okabe-Ito palette, consistent with figures_v2 / v17-v18 style.
Data source: v19_oracle_vis_directive.md (oracle 2x2x3 factorial, exact numbers).

v2: fixed vision-QA issues from first pass — 20M/250M label clustering,
teal callout box overlapping the mini-retrieved 250M point, legend
colliding with the "retrieved: collapses" annotation, diagonal arrow
crossing a data line.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Palette (Okabe-Ito) ──────────────────────────────────────────────────────
BLUE   = "#0072B2"   # large model
ORANGE = "#E69F00"   # compact model
GRAY   = "#555555"   # annotation / neutral text
GREEN  = "#009E73"   # callout accent

# ── Data (Correct%, oracle 2x2x3 factorial) ─────────────────────────────────
tiers = ["20M", "60M", "250M"]
x = np.arange(len(tiers))

mini_gold      = [82.50, 84.00, 82.25]
mini_retrieved = [84.00, 78.25, 73.50]
full_gold      = [85.75, 85.00, 83.75]
full_retrieved = [70.25, 66.00, 56.25]

# ── Figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10.5, 7.0))

lw_gold = 2.6
lw_ret  = 2.6
ms = 9

l_full_gold, = ax.plot(x, full_gold, color=BLUE, linestyle="-", marker="o",
                        markersize=ms, linewidth=lw_gold, zorder=5,
                        markeredgecolor="white", markeredgewidth=1.0,
                        label="large model - gold")
l_mini_gold, = ax.plot(x, mini_gold, color=ORANGE, linestyle="-", marker="o",
                        markersize=ms, linewidth=lw_gold, zorder=5,
                        markeredgecolor="white", markeredgewidth=1.0,
                        label="compact model - gold")
l_full_ret,  = ax.plot(x, full_retrieved, color=BLUE, linestyle="--", marker="s",
                        markersize=ms, linewidth=lw_ret, zorder=5,
                        markeredgecolor="white", markeredgewidth=1.0,
                        label="large model - retrieved")
l_mini_ret,  = ax.plot(x, mini_retrieved, color=ORANGE, linestyle="--", marker="s",
                        markersize=ms, linewidth=lw_ret, zorder=5,
                        markeredgecolor="white", markeredgewidth=1.0,
                        label="compact model - retrieved")

# ── Axes (set ranges early so annotation coords are meaningful) ────────────
ax.set_xticks(x)
ax.set_xticklabels(tiers, fontsize=13, fontweight="bold")
ax.set_xlabel("Corpus size", fontsize=13, fontweight="bold", labelpad=8)
ax.set_ylabel("Correct (%)", fontsize=13, fontweight="bold")
ax.set_xlim(-0.75, 3.05)
ax.set_ylim(48, 96)
ax.tick_params(axis="y", labelsize=11)
ax.yaxis.grid(True, alpha=0.25, linestyle=":", linewidth=0.8)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)

# ── Endpoint value labels: vertical "stacked list" to the left of x=0 and
#    right of x=2, with thin leader lines to the true data points. This
#    guarantees no overlap regardless of how close the raw values are. ──────
def leader_labels(x_anchor, side, entries):
    """entries: list of (true_y, label_y, text, color) sorted for spacing."""
    dx = -0.30 if side == "left" else 0.30
    ha = "right" if side == "left" else "left"
    lx = x_anchor + dx
    for true_y, label_y, text, color in entries:
        ax.annotate("", xy=(x_anchor, true_y), xytext=(lx, label_y),
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.9,
                                     alpha=0.55, shrinkA=0, shrinkB=6),
                    zorder=3)
        ax.text(lx + (-0.03 if side == "left" else 0.03), label_y, text,
                ha=ha, va="center", fontsize=9.5, fontweight="bold",
                color=color, zorder=6)

# Left side (20M): sorted descending true value, spaced list y-positions
leader_labels(0, "left", [
    (85.75, 93.5, "85.8", "#004C77"),   # full gold
    (84.00, 89.5, "84.0", "#A66E00"),   # mini retrieved
    (82.50, 85.5, "82.5", "#A66E00"),   # mini gold
    (70.25, 70.25, "70.3", "#004C77"),  # full retrieved (already well separated)
])

# Right side (250M): sorted descending true value, spaced list y-positions
leader_labels(2, "right", [
    (83.75, 92.5, "83.8", "#004C77"),   # full gold
    (82.25, 88.5, "82.3", "#A66E00"),   # mini gold
    (73.50, 78.0, "73.5", "#A66E00"),   # mini retrieved
    (56.25, 56.25, "56.2", "#004C77"),  # full retrieved
])

ax.set_title(
    "Oracle 2\u00d72\u00d73 Factorial: Gold Context Stays Flat, Retrieved Context Collapses\n"
    "(regardless of model — falsifies the \u201ccapacity\u201d explanation)",
    fontsize=14.5, fontweight="bold", pad=14,
)

# ── Region annotations: "gold: flat" / "retrieved: collapses" ───────────────
ax.annotate("gold: flat (both models)", xy=(1.0, 84.6),
            xytext=(1.0, 95.5), ha="center", fontsize=10.5, color=GRAY,
            fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=GRAY, lw=1.0, alpha=0.7))

ax.annotate("retrieved: collapses\n(both models)", xy=(0.75, 72.5),
            xytext=(0.75, 61.5), ha="center", fontsize=10.5, color=GRAY,
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.3, alpha=0.85))

# ── KEY callout: full retrieved falls steeper than mini retrieved ──────────
# Placed low and clear of the mini-retrieved dashed line (which stays above
# ~73 for x>=1), and clear of the right-side endpoint label list (x>=2.15).
callout_text = (
    "$\\bf{-14\\%}$ (full model, retrieved) falls STEEPER\n"
    "than $\\bf{-10.5\\%}$ (mini, retrieved) \u2192\n"
    "stronger model degrades MORE \u2014 not a capacity limit"
)
ax.annotate(callout_text, xy=(1.95, 58.0), xytext=(1.30, 64.5),
            ha="left", va="center", fontsize=9.8,
            color="#00543D", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.42", facecolor="#E6F7F2",
                      edgecolor=GREEN, linewidth=1.4),
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.6,
                             connectionstyle="arc3,rad=0.12"))

# ── Legend: placed fully OUTSIDE the plotting area (right margin) so it can
#    never collide with data, leader labels, or in-plot annotations. ────────
legend = ax.legend(handles=[l_full_gold, l_mini_gold, l_full_ret, l_mini_ret],
                    fontsize=10, loc="center left", framealpha=0.95,
                    edgecolor="#cccccc", ncol=1,
                    bbox_to_anchor=(1.16, 0.5),
                    title="Model \u00d7 Doc-source", title_fontsize=9.5)
legend.get_title().set_fontweight("bold")
legend.set_zorder(10)

# ── Footnote ─────────────────────────────────────────────────────────────────
fig.text(0.5, -0.045,
         "Oracle 2\u00d72\u00d73 factorial (large vs. compact model \u00d7 gold vs. retrieved documents, "
         "20M/60M/250M corpus). Correct% is the primary metric; markers show exact means.",
         ha="center", fontsize=9, color="#666666", style="italic")

fig.tight_layout(pad=0.5)

# ── Save ─────────────────────────────────────────────────────────────────────
out_dir = "./paper/v19/figures_supp"
fig.savefig(f"{out_dir}/oracle_2x2_lines.pdf", dpi=300, bbox_inches="tight")
fig.savefig(f"{out_dir}/oracle_2x2_lines.png", dpi=200, bbox_inches="tight")
print("\u2713 oracle_2x2_lines saved")
