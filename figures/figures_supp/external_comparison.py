"""
external_comparison.py
FIG E — 3-benchmark grouped bar comparison.
Shows the capacity-vs-noise boundary reproduces across benchmarks:
EnterpriseRAG (20M, low-noise / high-capacity regime) vs LoCoMo (moderate
distractor density) vs UltraDomain (high distractor density) across
Overall / Correct / Complete / DocRcl.

Colorblind-safe: Okabe-Ito palette, consistent with figures_v2 style.
Data source: paper v16 (exact numbers, no reload needed).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Palette (Okabe-Ito, matches figures_v2) ─────────────────────────────────
BLUE   = "#0072B2"   # EnterpriseRAG (20M) — high-capacity regime
ORANGE = "#E69F00"   # LoCoMo — moderate distractor density
GRAY   = "#555555"   # UltraDomain — high distractor density

# ── Data (paper v16, exact) ──────────────────────────────────────────────────
metrics = ["Overall", "Correct%", "Complete%", "DocRcl%"]

enterprise_rag = [71.67, 84.00, 73.78, 84.00]   # 20M, pure-mini best
locomo         = [48.66, 44.25, 47.15, 61.97]   # conversational memory
ultradomain    = [21.08, 13.25, 18.86, 44.00]   # cross-domain QA

x = np.arange(len(metrics))
width = 0.26

# ── Figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

bars_erag = ax.bar(x - width, enterprise_rag, width, color=BLUE,
                    label="EnterpriseRAG (20M)", zorder=3,
                    edgecolor="#004C77", linewidth=0.8)
bars_loc  = ax.bar(x, locomo, width, color=ORANGE,
                    label="LoCoMo", zorder=3,
                    edgecolor="#A66E00", linewidth=0.8)
bars_ultr = ax.bar(x + width, ultradomain, width, color=GRAY,
                    label="UltraDomain", zorder=3,
                    edgecolor="#2B2B2B", linewidth=0.8)

# ── Value labels above each bar ──────────────────────────────────────────────
def label_bars(bars, values, color):
    for bar, val in zip(bars, values):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2,
                 h + 1.3,
                 f"{val:.1f}",
                 ha="center", va="bottom",
                 fontsize=9.5, color=color, fontweight="bold")

label_bars(bars_erag, enterprise_rag, "#004C77")
label_bars(bars_loc, locomo, "#A66E00")
label_bars(bars_ultr, ultradomain, "#2B2B2B")

# ── Axes ─────────────────────────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=13, fontweight="bold")
ax.set_ylabel("Score (%)", fontsize=13, fontweight="bold")
ax.set_ylim(0, 96)
ax.tick_params(axis="y", labelsize=11)
ax.yaxis.grid(True, alpha=0.25, linestyle=":", linewidth=0.8)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)

# ── Title ─────────────────────────────────────────────────────────────────────
ax.set_title(
    "Capacity-vs-Noise Boundary Reproduces Across Benchmarks",
    fontsize=15, fontweight="bold", pad=12,
)

ax.set_xlabel("Evaluation Metric", fontsize=12, fontweight="bold", labelpad=8)

# ── Legend ────────────────────────────────────────────────────────────────────
ax.legend(fontsize=11, loc="upper right", framealpha=0.95, edgecolor="#cccccc",
          title="Benchmark (distractor density: low → high)", title_fontsize=9.5)

# ── Footnote ─────────────────────────────────────────────────────────────────
fig.text(0.5, -0.01,
         "EnterpriseRAG (20M pure-mini best) shows highest scores under low distractor density; "
         "scores decline as cross-document noise increases (LoCoMo → UltraDomain). Data from paper v16.",
         ha="center", fontsize=8.5, color="#777777", style="italic")

fig.tight_layout(pad=0.5)

# ── Save ─────────────────────────────────────────────────────────────────────
out_dir = "./paper/v17/figures_supp"
fig.savefig(f"{out_dir}/external_comparison.pdf", dpi=300, bbox_inches="tight")
fig.savefig(f"{out_dir}/external_comparison.png", dpi=200, bbox_inches="tight")
print("✓ external_comparison saved")
