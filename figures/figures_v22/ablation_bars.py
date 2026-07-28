"""
ablation_bars.py
FIG — 6-component leave-one-out ablation (§10).
Shows ΔCombined vs full v3xf baseline when each component is removed.
Dual-Index + Distillation are the core components (largest drop when
removed); Rerank / Query-Expansion are mid-tier; Citation is provenance-only
(no accuracy impact when removed).

Colorblind-safe: Okabe-Ito palette, consistent with figures_v2 / figures_supp.
Data source: v22_figures_data.md, section 10 (main index, general API, n=400).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Palette (Okabe-Ito) ──────────────────────────────────────────────────────
DARK_CORE = "#0072B2"   # core components (Dual-Index, Distillation)
MID       = "#E69F00"   # mid-tier components (Rerank, Query-Expansion)
NEUTRAL   = "#555555"   # provenance-only (Citation)
FULL      = "#009E73"   # full model reference bar

# ── Data (v22_figures_data.md §10) ──────────────────────────────────────────
labels_raw = [
    ("full v3xf", 0.00, FULL),
    ("\u2212Query-Expansion", -2.92, MID),
    ("\u2212Rerank", -4.09, MID),
    ("\u2212Distillation", -11.64, DARK_CORE),
    ("\u2212Dual-Index", -14.73, DARK_CORE),
    ("\u2212Citation", 0.00, NEUTRAL),
]

# Sort by magnitude (ascending |delta|), keep "full v3xf" first as reference
non_full = [x for x in labels_raw if x[0] != "full v3xf"]
non_full_sorted = sorted(non_full, key=lambda t: abs(t[1]))
ordered = [labels_raw[0]] + non_full_sorted  # full first, then by magnitude

names  = [x[0] for x in ordered]
deltas = [x[1] for x in ordered]
colors = [x[2] for x in ordered]

y = np.arange(len(names))

# ── Figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8.5, 5.2))

bars = ax.barh(y, deltas, color=colors, zorder=3, height=0.6,
                edgecolor="#2B2B2B", linewidth=0.8)

# ── Value labels ──────────────────────────────────────────────────────────────
for bar, val, name in zip(bars, deltas, names):
    w = bar.get_width()
    if name == "full v3xf":
        label = "0.00% (ref.)"
    else:
        label = f"{val:+.2f}%"
    xpos = w - 0.35 if w < 0 else w + 0.35
    ha = "right" if w < 0 else "left"
    ax.text(xpos, bar.get_y() + bar.get_height() / 2, label,
             va="center", ha=ha, fontsize=10, fontweight="bold",
             color="#1a1a1a")

# ── Axes ─────────────────────────────────────────────────────────────────────
ax.set_yticks(y)
ax.set_yticklabels(names, fontsize=11.5)
ax.invert_yaxis()  # full v3xf at top, then ascending magnitude of drop
ax.axvline(0, color="#333333", linewidth=1.0, zorder=2)
ax.set_xlabel(r"$\Delta$ Combined Score vs. full v3xf (%)", fontsize=12,
              fontweight="bold", labelpad=8)
ax.set_xlim(-18, 3)
ax.xaxis.grid(True, alpha=0.25, linestyle=":", linewidth=0.8)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)

# ── Title ─────────────────────────────────────────────────────────────────────
ax.set_title("Component Necessity: Leave-One-Out Ablation",
              fontsize=14.5, fontweight="bold", pad=12)

# ── Legend (placed above the axes to avoid overlapping bars) ────────────────
from matplotlib.patches import Patch
legend_handles = [
    Patch(facecolor=FULL, edgecolor="#2B2B2B", label="Full model (reference)"),
    Patch(facecolor=DARK_CORE, edgecolor="#2B2B2B", label="Core (large drop)"),
    Patch(facecolor=MID, edgecolor="#2B2B2B", label="Mid-tier"),
    Patch(facecolor=NEUTRAL, edgecolor="#2B2B2B", label="Provenance-only (no drop)"),
]
ax.legend(handles=legend_handles, fontsize=8.8, loc="upper center",
          bbox_to_anchor=(0.5, -0.16), ncol=2, framealpha=0.95,
          edgecolor="#cccccc")

# ── Footnote ─────────────────────────────────────────────────────────────────
fig.text(0.5, -0.13,
         "Leave-one-out ablation on the main index (general API, n=400).\n"
         "Dual-Index and Distillation are load-bearing; Citation affects provenance only, not accuracy.",
         ha="center", fontsize=8.5, color="#777777", style="italic")

fig.tight_layout(pad=0.5)

# ── Save ─────────────────────────────────────────────────────────────────────
out_dir = "./paper/v21/figures_v22"
fig.savefig(f"{out_dir}/ablation_bars.pdf", dpi=300, bbox_inches="tight")
fig.savefig(f"{out_dir}/ablation_bars.png", dpi=200, bbox_inches="tight")
print("\u2713 ablation_bars saved")
