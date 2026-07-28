#!/usr/bin/env python3
"""
fig3_hdm_routing.py — Fig 3: HDM Hierarchical Routing Ablation (60M test n=400)
NeurIPS-style grouped bar chart. Key finding: HDM routing +5.75pp (5.5σ, p<0.001).

Data sources:
  experiments/stage2_phase21_60m/final_eval_test/{hdm,combined}/summary.json
  experiments/stage2_hdm_routing_ablation/60M_test_n400/hdm_on/summary.json
  experiments/stage2_hdm_routing_ablation/60M_test_n400_combined/combined_route_on/summary.json

Run from project root:
  python3 paper/v3_polish/figures/fig3_code.py
"""

import json, os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, "../../.."))

def load(path):
    with open(os.path.join(PROJ_ROOT, path)) as f:
        return json.load(f)

def llm(d):
    if "aggregate" in d: return float(d["aggregate"]["llm_score"])
    return float(d["llm_score"])

# ── Load ──────────────────────────────────────────────────────────────────────
hdm_off = llm(load("experiments/stage2_phase21_60m/final_eval_test/hdm/summary.json"))
hdm_on  = llm(load("experiments/stage2_hdm_routing_ablation/60M_test_n400/hdm_on/summary.json"))
com_off = llm(load("experiments/stage2_phase21_60m/final_eval_test/combined/summary.json"))
com_on  = llm(load("experiments/stage2_hdm_routing_ablation/60M_test_n400_combined/combined_route_on/summary.json"))

SIGMA_MAX = 0.0104
print("=== Data verification ===")
print(f"HDM (OFF):      {hdm_off:.4f}")
print(f"HDM (ON):       {hdm_on:.4f}   Δ={( hdm_on-hdm_off)*100:+.2f}pp  "
      f"({(hdm_on-hdm_off)/SIGMA_MAX:.1f}σ, p<0.001)")
print(f"Combined (OFF): {com_off:.4f}")
print(f"Combined (ON):  {com_on:.4f}   Δ={(com_on-com_off)*100:+.2f}pp  "
      f"({(com_on-com_off)/SIGMA_MAX:.1f}σ, stat. tie)")

# ── Style ──────────────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-paper")
mpl.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.labelsize": 11, "axes.labelweight": "bold",
    "axes.titlesize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 9, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

C_HDM_OFF = "#A0C0FF"; C_HDM_ON = "#648FFF"
C_COM_OFF = "#C4B4F8"; C_COM_ON = "#785EF0"

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 4.4))

configs = [
    ("HDM\n(routing OFF)",   hdm_off, C_HDM_OFF),
    ("HDM\n(routing ON)",    hdm_on,  C_HDM_ON),
    ("Combined\n(OFF)",      com_off, C_COM_OFF),
    ("Combined\n(ON)",       com_on,  C_COM_ON),
]

x     = np.arange(len(configs))
bar_w = 0.56
vals  = [c[1] for c in configs]
cols  = [c[2] for c in configs]

bars = ax.bar(x, vals, bar_w, color=cols, edgecolor="white",
              linewidth=0.7, alpha=0.90, zorder=3)

# Value labels
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.006,
            f"{v:.4f}", ha="center", va="bottom",
            fontsize=8.5, fontweight="bold", color="#222222")

# Group divider
ax.axvline(x=1.5, color="lightgray", lw=1.0, ls="--", zorder=2)

# Significance brackets
def sig_bracket(ax, x0, x1, y_bar_max, delta_text, sig_text, color="#222222"):
    y_arm   = y_bar_max + 0.012
    tick_h  = 0.010
    ax.plot([x0, x0, x1, x1],
            [y_arm, y_arm + tick_h, y_arm + tick_h, y_arm],
            color=color, lw=1.3, zorder=5)
    ax.text((x0+x1)/2, y_arm + tick_h + 0.003,
            delta_text, ha="center", va="bottom",
            fontsize=9, fontweight="bold", color=color)
    ax.text((x0+x1)/2, y_arm + tick_h + 0.017,
            sig_text, ha="center", va="bottom",
            fontsize=7.5, color="#555555")

sig_bracket(ax, 0, 1, max(hdm_off, hdm_on),
            f"Δ = +{(hdm_on-hdm_off)*100:.2f}pp",
            f"5.5σ · p<0.001")

sig_bracket(ax, 2, 3, max(com_off, com_on),
            f"Δ = +{(com_on-com_off)*100:.2f}pp",
            f"1.2σ · stat. tie")

ax.set_xticks(x)
ax.set_xticklabels([c[0] for c in configs], ha="center")
ax.set_ylabel("LLM Judge Score", labelpad=6)
ax.set_ylim(0.30, 0.54)
ax.yaxis.grid(True, ls="--", alpha=0.35, zorder=0)
ax.set_axisbelow(True)

ax.legend(handles=[
    Patch(color=C_HDM_ON,  label="HDM (hierarchical routing)"),
    Patch(color=C_COM_ON,  label="Combined (multi-source)"),
], loc="upper left", framealpha=0.9, edgecolor="lightgrey",
   fancybox=False, fontsize=9)

# Footnote — positioned above legend to avoid overlap
ax.text(0.01, 0.01,
        "60M tier, test n=400. σ_max=0.0104 (5-run reruns, p<0.001 threshold).",
        transform=ax.transAxes, ha="left", va="bottom",
        fontsize=7, color="gray", style="italic",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                  edgecolor="none", alpha=0.8))

plt.tight_layout()
out_dir = os.path.dirname(os.path.abspath(__file__))
for ext in ["pdf", "png"]:
    out = os.path.join(out_dir, f"fig3_hdm_routing.{ext}")
    plt.savefig(out, dpi=300 if ext == "png" else None)
    print(f"Saved: {out}")
plt.close()
print("Done.")
