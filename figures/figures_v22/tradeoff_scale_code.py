#!/usr/bin/env python3
"""
tradeoff_scale_code.py — Fig: Context-Packing Token/Quality Trade-off, Scale-Invariance (10M vs 20M)

Data sources (Option B, unified k=20 — supersedes earlier "aligned"/v25-published data):
  10M: experiments/token_analysis/token_5bc_10m_optionB_k20.json
  20M: experiments/token_analysis/token_5bc_20m_optionB_k20.json

Key story: 3 context-packing modes (distilled_only / mixed / detailed_full) at 10M and 20M scale.
"mixed" (v3xf default) sits at the token-efficiency sweet spot — ~1/3 the tokens of detailed_full
at close-to-detailed_full Correct%. The token savings_ratio (mixed vs detailed_full) is
scale-invariant: 3.19x @ 10M ~= 3.21x @ 20M.

Run from project root:
  python3 paper/v26/figures_v22/tradeoff_scale_code.py

Output:
  paper/v26/figures_v22/tradeoff_scale.pdf
  paper/v26/figures_v22/tradeoff_scale.png
"""

import json
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, "../../.."))

def load(path):
    with open(os.path.join(PROJ_ROOT, path)) as f:
        return json.load(f)

# ── Load data (Option B, unified k=20 — supersedes earlier "aligned"/v25 data) ─
data_10m = load("experiments/token_analysis/token_5bc_10m_optionB_k20.json")
data_20m = load("experiments/token_analysis/token_5bc_20m_optionB_k20.json")

MODES = ["distilled_only", "mixed", "detailed_full"]
MODE_LABELS = {
    "distilled_only": "distilled-only",
    "mixed":          "mixed (v3xf default)",
    "detailed_full":  "detailed-full",
}

tok_10m = {m: data_10m["modes"][m]["mean_input_tokens"] for m in MODES}
cor_10m = {m: data_10m["modes"][m]["Correct"]           for m in MODES}
tok_20m = {m: data_20m["modes"][m]["mean_input_tokens"] for m in MODES}
cor_20m = {m: data_20m["modes"][m]["Correct"]           for m in MODES}

savings_10m = data_10m["token_5c_budget"]["savings_ratio"]
savings_20m = data_20m["token_5c_budget"]["savings_ratio"]

# ── Print verification ──────────────────────────────────────────────────────
print("=== Data verification ===")
print(f"{'Mode':16s}  {'10M tok':>9}  {'10M Corr%':>10}  {'20M tok':>9}  {'20M Corr%':>10}")
for m in MODES:
    print(f"{m:16s}  {tok_10m[m]:>9.1f}  {cor_10m[m]:>10.1f}  {tok_20m[m]:>9.1f}  {cor_20m[m]:>10.1f}")
print(f"\nsavings_ratio (mixed vs detailed_full): 10M={savings_10m:.2f}x  20M={savings_20m:.2f}x")
print(f"Recomputed check: 10M={tok_10m['detailed_full']/tok_10m['mixed']:.3f}x  "
      f"20M={tok_20m['detailed_full']/tok_20m['mixed']:.3f}x")

# ── Style ──────────────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-paper")
mpl.rcParams.update({
    "font.family":        "serif",
    "font.size":          10,
    "axes.labelsize":     11,
    "axes.labelweight":   "bold",
    "axes.titlesize":     10.5,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "legend.fontsize":    9,
    "lines.linewidth":    2.2,
    "lines.markersize":   9,
    "axes.grid":          True,
    "grid.alpha":         0.35,
    "grid.linestyle":     "--",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":         150,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.08,
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
})

# IBM Colorblind-Safe palette — one color per mode
COLORS = {
    "distilled_only": "#648FFF",   # blue
    "mixed":          "#785EF0",   # violet (the star of the show)
    "detailed_full":  "#DC267F",   # magenta
}
MARKERS = {
    "distilled_only": "o",
    "mixed":          "*",   # star marker for the highlighted default
    "detailed_full":  "^",
}
MARKER_SIZES = {
    "distilled_only": 9,
    "mixed":          16,
    "detailed_full":  9,
}

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.4, 4.6))

# 10M = solid line, filled markers
# 20M = dashed line, open markers
for m in MODES:
    x10, y10 = tok_10m[m], cor_10m[m]
    x20, y20 = tok_20m[m], cor_20m[m]
    c = COLORS[m]
    ax.plot(
        [x10, x20], [y10, y20],
        color=c, linewidth=1.3, linestyle=":", alpha=0.55, zorder=2,
    )
    # 10M point (solid, filled)
    ax.plot(
        x10, y10, marker=MARKERS[m], color=c, markersize=MARKER_SIZES[m],
        markerfacecolor=c, markeredgecolor="white", markeredgewidth=1.2,
        linestyle="", zorder=5,
    )
    # 20M point (open/hollow marker on same color, distinguishing scale)
    ax.plot(
        x20, y20, marker=MARKERS[m], color=c, markersize=MARKER_SIZES[m],
        markerfacecolor="white", markeredgecolor=c, markeredgewidth=2.0,
        linestyle="", zorder=5,
    )

# Connect same-mode points (10M -> 20M) already drawn above as dotted; also
# connect across modes within each scale (the "trade-off curve") with solid/dashed
order = ["distilled_only", "mixed", "detailed_full"]
x10_line = [tok_10m[m] for m in order]
y10_line = [cor_10m[m] for m in order]
x20_line = [tok_20m[m] for m in order]
y20_line = [cor_20m[m] for m in order]

ax.plot(x10_line, y10_line, color="#555555", linewidth=1.8, linestyle="-",
        alpha=0.55, zorder=3, label="10M (solid, filled \u25CF)")
ax.plot(x20_line, y20_line, color="#555555", linewidth=1.8, linestyle="--",
        alpha=0.55, zorder=3, label="20M (dashed, hollow \u25CB)")

# Log-scale x-axis since tokens span 1K-6K, roughly 6x range
ax.set_xscale("log")

# Mode labels — anchored below/above the 10M point with a leader line to avoid
# overlapping the 20M hollow marker (10M and 20M points sit close together on x)
label_offsets = {
    "distilled_only": (-28, -26),
    "mixed":           (5,   34),
    "detailed_full":   (18, -16),
}
label_ha = {
    "distilled_only": "center",
    "mixed":           "center",
    "detailed_full":   "left",
}
for m in MODES:
    dx, dy = label_offsets[m]
    va = "bottom" if dy > 0 else "top"
    ax.annotate(
        MODE_LABELS[m],
        xy=(tok_10m[m], cor_10m[m]),
        xytext=(dx, dy), textcoords="offset points",
        fontsize=8.5, color=COLORS[m], fontweight="bold",
        ha=label_ha[m], va=va,
        arrowprops=dict(arrowstyle="-", color=COLORS[m], lw=0.7, alpha=0.7,
                        shrinkA=2, shrinkB=8),
    )

# savings_ratio annotation — bracket between mixed and detailed_full at 10M
def draw_savings_bracket(ax, x_mixed, x_full, y, ratio, scale_label, color):
    y_arrow = y
    ax.annotate(
        "", xy=(x_full, y_arrow), xytext=(x_mixed, y_arrow),
        arrowprops=dict(arrowstyle="<->", color=color, lw=1.4,
                        shrinkA=2, shrinkB=2),
    )
    x_mid = np.sqrt(x_mixed * x_full)  # geometric mean for log-scale midpoint
    ax.text(
        x_mid, y_arrow + 1.8,
        f"{ratio:.2f}\u00d7 ({scale_label})",
        ha="center", va="bottom", fontsize=8, color=color, fontweight="bold",
    )

draw_savings_bracket(ax, tok_10m["mixed"], tok_10m["detailed_full"],
                      6, savings_10m, "10M", "#333333")
draw_savings_bracket(ax, tok_20m["mixed"], tok_20m["detailed_full"],
                      12, savings_20m, "20M", "#777777")

# Scale-invariance callout
ax.text(
    0.98, 0.06,
    f"savings\u2248scale-invariant: {savings_10m:.2f}\u00d7 (10M) \u2248 {savings_20m:.2f}\u00d7 (20M)",
    transform=ax.transAxes, ha="right", va="bottom",
    fontsize=8.5, color="#222222", style="italic",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF3CD",
              edgecolor="#D4AC0D", alpha=0.9),
)

ax.set_xlabel("Mean Input Tokens per Query (log scale)", labelpad=6)
ax.set_ylabel("Correct (%)", labelpad=6)
ax.set_title(
    "Context-Packing Trade-off: Token Cost vs. Answer Quality\n"
    "(3 packing modes \u00d7 2 scales, n=100)",
    pad=10,
)

ax.set_xlim(700, 9500)
ax.set_ylim(0, 65)

ax.legend(loc="upper left", framealpha=0.92, edgecolor="lightgrey",
          fancybox=False, fontsize=8.5)

plt.tight_layout()

# ── Save ───────────────────────────────────────────────────────────────────────
out_dir = os.path.dirname(os.path.abspath(__file__))
for ext in ["pdf", "png"]:
    out = os.path.join(out_dir, f"tradeoff_scale.{ext}")
    plt.savefig(out, dpi=300 if ext == "png" else None)
    print(f"Saved: {out}")
plt.close()
print("Done.")
