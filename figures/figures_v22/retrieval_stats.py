"""
retrieval_stats.py
FIG — Retrieval/citation gating statistics (§5d, optional/appendix).
Two-panel figure: (a) Query-Expansion target recovery rate,
(b) Citation gate funnel (docs retrieved -> docs cited).

Colorblind-safe: Okabe-Ito palette, consistent with figures_v2 / figures_supp.
Data source: v22_figures_data.md, §5d.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Palette (Okabe-Ito) ──────────────────────────────────────────────────────
BLUE   = "#0072B2"
ORANGE = "#E69F00"
GREEN  = "#009E73"
GRAY   = "#555555"

# ── Data (v22_figures_data.md §5d) ──────────────────────────────────────────
qe_targets    = 33
qe_recovered  = 13
qe_pct        = 39.4

gate_in  = 5.0
gate_out = 2.31
gate_pct = 46.0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# ── Panel (a): QE recovery ───────────────────────────────────────────────────
cats_a = ["Targets\n(missed by\nbase retrieval)", "Recovered\nby QE"]
vals_a = [qe_targets, qe_recovered]
colors_a = [GRAY, BLUE]

bars_a = ax1.bar(cats_a, vals_a, width=0.55, color=colors_a, zorder=3,
                  edgecolor="#2B2B2B", linewidth=0.9)
for bar, v in zip(bars_a, vals_a):
    ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.8, f"{v}",
              ha="center", va="bottom", fontsize=12, fontweight="bold")

ax1.annotate(f"{qe_pct:.1f}% recovery",
             xy=(1, qe_recovered), xytext=(0.5, 27),
             ha="center", fontsize=11, fontweight="bold", color=BLUE,
             arrowprops=dict(arrowstyle="-", color=BLUE, lw=1.0))

ax1.set_ylabel("Number of Targets", fontsize=12, fontweight="bold", labelpad=8)
ax1.set_ylim(0, 40)
ax1.set_title("(a) Query-Expansion Recovery", fontsize=13, fontweight="bold",
               pad=10)
ax1.yaxis.grid(True, alpha=0.25, linestyle=":", linewidth=0.8)
ax1.set_axisbelow(True)
ax1.spines[["top", "right"]].set_visible(False)

# ── Panel (b): Citation gate funnel ──────────────────────────────────────────
cats_b = ["Docs Retrieved\n(per query)", "Docs Cited\n(after gate)"]
vals_b = [gate_in, gate_out]
colors_b = [GRAY, GREEN]

bars_b = ax2.bar(cats_b, vals_b, width=0.55, color=colors_b, zorder=3,
                  edgecolor="#2B2B2B", linewidth=0.9)
for bar, v in zip(bars_b, vals_b):
    ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.08, f"{v:.2f}",
              ha="center", va="bottom", fontsize=12, fontweight="bold")

ax2.annotate(f"{gate_pct:.0f}% pass rate",
             xy=(1, gate_out + 0.15), xytext=(0.5, 4.5),
             ha="center", fontsize=11, fontweight="bold", color=GREEN,
             arrowprops=dict(arrowstyle="-", color=GREEN, lw=1.0))

ax2.set_ylabel("Documents per Query", fontsize=12, fontweight="bold",
               labelpad=8)
ax2.set_ylim(0, 6)
ax2.set_title("(b) Citation Gate Funnel", fontsize=13, fontweight="bold",
               pad=10)
ax2.yaxis.grid(True, alpha=0.25, linestyle=":", linewidth=0.8)
ax2.set_axisbelow(True)
ax2.spines[["top", "right"]].set_visible(False)

# ── Suptitle ─────────────────────────────────────────────────────────────────
fig.suptitle("Retrieval and Citation Gating Statistics", fontsize=15,
             fontweight="bold", y=1.02)

# ── Footnote ─────────────────────────────────────────────────────────────────
fig.text(0.5, -0.04,
         "(a) Query-Expansion recovers 13/33 (39.4%) of targets missed by base retrieval. "
         "(b) Citation gate filters ~5 retrieved docs down to 2.31 cited (46%). "
         "Data: v22_figures_data.md \u00a75d.",
         ha="center", fontsize=8.5, color="#777777", style="italic")

fig.tight_layout(pad=0.8)

# ── Save ─────────────────────────────────────────────────────────────────────
out_dir = "./paper/v21/figures_v22"
fig.savefig(f"{out_dir}/retrieval_stats.pdf", dpi=300, bbox_inches="tight")
fig.savefig(f"{out_dir}/retrieval_stats.png", dpi=200, bbox_inches="tight")
print("\u2713 retrieval_stats saved")
