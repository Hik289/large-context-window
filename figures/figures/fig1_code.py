#!/usr/bin/env python3
"""
fig1_scaling_trajectory.py — Fig 1: Scaling Trajectory (0M → 60M → 100M)
NeurIPS-style academic figure.

Data sources:
  0M:   experiments/stage1_mvb_0m/0m_eval_summary.json          (single-run, dev n=100)
  60M:  experiments/stage2_phase21_60m/final_eval_test/{m}/summary.json  (test n=400, paper values)
  100M: experiments/stage3_phase_100m_test_n400/{m}/summary.json          (test n=400)
  σ:    analysis/stage2_sigma_60m_raw.json   (σ_std only; NOT used for point estimates)

Run from project root:
  python3 paper/v3_polish/figures/fig1_code.py
"""

import json, os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, "../../.."))

def load(path):
    with open(os.path.join(PROJ_ROOT, path)) as f:
        return json.load(f)

METHODS = ["ddi", "combined", "cdm", "hdm"]
LABELS  = {"ddi": "DDI", "combined": "Combined", "cdm": "CDM", "hdm": "HDM (routing off)"}

# ── Load data ─────────────────────────────────────────────────────────────────
data_0m    = load("experiments/stage1_mvb_0m/0m_eval_summary.json")
data_sigma = load("analysis/stage2_sigma_60m_raw.json")

pt_0m = {m: data_0m["methods"][m]["llm_score"] for m in METHODS}

# 60M: original paper test values (match Tables); σ for error bars only
pt_60m, std_60m = {}, {}
for m in METHODS:
    s = load(f"experiments/stage2_phase21_60m/final_eval_test/{m}/summary.json")
    agg = s.get("aggregate", s)
    pt_60m[m]  = float(agg["llm_score"])
    std_60m[m] = float(np.std(data_sigma[m]["llm_score"], ddof=1))

pt_100m = {}
for m in METHODS:
    s = load(f"experiments/stage3_phase_100m_test_n400/{m}/summary.json")
    agg = s.get("aggregate", s)
    pt_100m[m] = float(agg["llm_score"])

print("=== Data verification ===")
print(f"{'Method':12s}  {'0M(dev)':>8}  {'60M(test)':>9}  {'60M σ':>6}  {'100M':>8}")
for m in METHODS:
    print(f"{m:12s}  {pt_0m[m]:>8.4f}  {pt_60m[m]:>9.4f}  {std_60m[m]:>6.4f}  {pt_100m[m]:>8.4f}")
    print(f"  Δ(0→60)={( pt_60m[m] - pt_0m[m])*100:+.1f}pp  "
          f"Δ(60→100)={(pt_100m[m] - pt_60m[m])*100:+.1f}pp  "
          f"Δ(0→100)={(pt_100m[m] - pt_0m[m])*100:+.1f}pp")

# ── Style ──────────────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-paper")
mpl.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.labelsize": 11, "axes.labelweight": "bold",
    "axes.titlesize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 9, "lines.linewidth": 2.2, "lines.markersize": 7,
    "errorbar.capsize": 5, "axes.grid": True, "grid.alpha": 0.35,
    "grid.linestyle": "--", "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
COLORS  = {"ddi": "#648FFF", "combined": "#785EF0", "cdm": "#DC267F", "hdm": "#FE6100"}
MARKERS = {"ddi": "o", "combined": "s", "cdm": "^", "hdm": "D"}

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.2, 4.0))
x = np.array([0, 1, 2])

for m in METHODS:
    y    = np.array([pt_0m[m], pt_60m[m], pt_100m[m]])
    yerr = np.array([0.0, std_60m[m], 0.0])
    ax.errorbar(x, y, yerr=yerr,
        fmt=f"-{MARKERS[m]}", color=COLORS[m], label=LABELS[m],
        linewidth=2.2, markersize=7,
        markerfacecolor=COLORS[m], markeredgecolor="white", markeredgewidth=1.2,
        capsize=5, capthick=2.0, elinewidth=2.0, ecolor=COLORS[m],
        alpha=0.92, zorder=4)

# Sub-linear annotation
mid_y = (pt_60m["combined"] + pt_100m["combined"]) / 2
ax.annotate("sub-linear\nscale-up",
    xy=(1.5, mid_y),
    xytext=(1.56, 0.510),
    fontsize=7.5, color="#444444", ha="left", va="center",
    arrowprops=dict(arrowstyle="->", color="#888888", lw=1.0,
                    connectionstyle="arc3,rad=-0.25"))

# Staggered delta labels (0→100 total) at right margin — 1 decimal place
MIN_GAP = 0.052
sorted_m = sorted(METHODS, key=lambda m: pt_100m[m], reverse=True)
label_y, last_y = {}, None
for m in sorted_m:
    y_new = pt_100m[m]
    if last_y is not None and abs(y_new - last_y) < MIN_GAP:
        y_new = last_y - MIN_GAP
    label_y[m] = y_new
    last_y = y_new

for m in sorted_m:
    d = (pt_100m[m] - pt_0m[m]) * 100
    y_data, y_lbl = pt_100m[m], label_y[m]
    if abs(y_lbl - y_data) > 0.004:
        ax.annotate("", xy=(2.03, y_data), xytext=(2.06, y_lbl),
            arrowprops=dict(arrowstyle="-", color=COLORS[m], lw=0.8,
                            connectionstyle="arc3,rad=0"), clip_on=False)
    ax.text(2.08, y_lbl, f"{d:+.1f}pp",
            ha="left", va="center", fontsize=8,
            color=COLORS[m], fontweight="bold", clip_on=False)

ax.set_xticks([0, 1, 2])
ax.set_xticklabels(["0M tier\n(722 docs, dev)", "60M tier\n(49.5K docs, test)",
                     "100M tier\n(82.8K docs, test)"])
ax.set_xlim(-0.35, 2.95)
ax.set_ylim(0.22, 0.68)
ax.set_xlabel("Index Scale", labelpad=6)
ax.set_ylabel("LLM Judge Score", labelpad=6)
ax.legend(loc="upper right", framealpha=0.9, edgecolor="lightgrey",
          fancybox=False, fontsize=9)
ax.text(0.02, 0.03,
        "Error bars: ±1σ (5 reruns) at 60M only; σ_max=0.0104.",
        transform=ax.transAxes, ha="left", va="bottom",
        fontsize=7.5, color="gray", style="italic")

plt.tight_layout()
out_dir = os.path.dirname(os.path.abspath(__file__))
for ext in ["pdf", "png"]:
    out = os.path.join(out_dir, f"fig1_scaling.{ext}")
    plt.savefig(out, dpi=300 if ext == "png" else None)
    print(f"Saved: {out}")
plt.close()
print("Done.")
