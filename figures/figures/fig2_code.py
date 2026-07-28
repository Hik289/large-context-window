#!/usr/bin/env python3
"""
fig2_baselines.py — Fig 2: Comparison with External Baselines (60M tier)
NeurIPS-style grouped bar chart. Conv.-LRM shown as horizontal ref line.

Data sources:
  Our methods @ 60M test n=400:
    experiments/stage2_phase21_60m/final_eval_test/{ddi,combined,hdm}/summary.json
    experiments/stage2_hdm_routing_ablation/60M_test_n400/hdm_on/summary.json
  Baselines @ 60M test n=400:
    experiments/stage3_r2_baselines_60m/{bm25,vanilla_dense}/results.json
  Conversational-LRM @ 0M dev n=100 (reference only — different tier):
    experiments/stage3_solution2_baseline_0m/results.json

Run from project root:
  python3 paper/v3_polish/figures/fig2_code.py
"""

import json, os
import numpy as np
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, "../../.."))

def load(path):
    with open(os.path.join(PROJ_ROOT, path)) as f:
        return json.load(f)

def get_metrics(d):
    if "aggregate" in d:
        a = d["aggregate"]
        return float(a["llm_score"]), float(a["doc_recall"])
    if "results" in d and "five_metrics" in d["results"]:
        fm = d["results"]["five_metrics"]
        return float(fm["llm_score"]), float(fm["doc_recall"])
    return float(d["llm_score"]), float(d["doc_recall"])

# ── Load ──────────────────────────────────────────────────────────────────────
ddi_l, ddi_d     = get_metrics(load("experiments/stage2_phase21_60m/final_eval_test/ddi/summary.json"))
com_l, com_d     = get_metrics(load("experiments/stage2_phase21_60m/final_eval_test/combined/summary.json"))
hdm_on_l, hdm_on_d = get_metrics(load("experiments/stage2_hdm_routing_ablation/60M_test_n400/hdm_on/summary.json"))
bm25_l, bm25_d   = get_metrics(load("experiments/stage3_r2_baselines_60m/bm25/results.json"))
vd_l, vd_d       = get_metrics(load("experiments/stage3_r2_baselines_60m/vanilla_dense/results.json"))
s2_l, s2_d       = get_metrics(load("experiments/stage3_solution2_baseline_0m/results.json"))

print("=== Data verification ===")
rows = [
    ("DDI",              ddi_l,    ddi_d),
    ("Combined",         com_l,    com_d),
    ("HDM+routing",      hdm_on_l, hdm_on_d),
    ("BM25",             bm25_l,   bm25_d),
    ("VanillaDense",     vd_l,     vd_d),
]
for name, l, d in rows:
    print(f"{name:16s}  llm={l:.4f}  doc_recall={d:.4f}")
print(f"Conv.-LRM(0M ref): llm={s2_l:.4f}  doc_recall={s2_d:.4f}")
print(f"BM25 doc_recall vs DDI: {(bm25_d-ddi_d)*100:+.1f}pp")
print(f"BM25 llm vs DDI: {(bm25_l-ddi_l)*100:+.1f}pp  ({abs(bm25_l-ddi_l)/0.0104:.1f}σ)")

# ── Style ──────────────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-paper")
mpl.rcParams.update({
    "font.family": "serif", "font.size": 9.5,
    "axes.labelsize": 11, "axes.labelweight": "bold",
    "axes.titlesize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9.5,
    "legend.fontsize": 8.5, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.1,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

C_OUR_LLM  = "#648FFF"; C_OUR_DR   = "#A0C0FF"
C_BASE_LLM = "#FE6100"; C_BASE_DR  = "#FFAB7A"

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8.0, 4.4))

labels     = ["DDI", "Combined", "HDM\n+routing", "BM25", "Vanilla\nDense"]
types      = ["ours", "ours", "ours", "baseline", "baseline"]
llm_vals   = [ddi_l, com_l, hdm_on_l, bm25_l, vd_l]
dr_vals    = [ddi_d, com_d, hdm_on_d, bm25_d, vd_d]
n          = len(labels)
x          = np.arange(n)
bar_w, gap = 0.34, 0.06

for i, (typ, llm, dr) in enumerate(zip(types, llm_vals, dr_vals)):
    c_llm = C_OUR_LLM if typ == "ours" else C_BASE_LLM
    c_dr  = C_OUR_DR  if typ == "ours" else C_BASE_DR
    ax.bar(x[i] - bar_w/2 - gap/2, llm, bar_w,
           color=c_llm, edgecolor="white", linewidth=0.5, alpha=0.90, zorder=3)
    ax.bar(x[i] + bar_w/2 + gap/2, dr,  bar_w,
           color=c_dr,  edgecolor="white", linewidth=0.5, alpha=0.90, zorder=3)

# Value labels
for i, (llm, dr) in enumerate(zip(llm_vals, dr_vals)):
    ax.text(x[i]-bar_w/2-gap/2, llm+0.008, f"{llm:.3f}",
            ha="center", va="bottom", fontsize=7, color="#222222")
    ax.text(x[i]+bar_w/2+gap/2, dr +0.008, f"{dr:.3f}",
            ha="center", va="bottom", fontsize=7, color="#222222")

# Divider our | baselines
ax.axvline(x=2.5, color="lightgray", lw=1.2, ls="--", zorder=2)

# Conv.-LRM reference line
ax.axhline(y=s2_l, color="#999999", lw=1.3, ls=":", zorder=4,
           label=f"Conv.-LRM baseline (0M dev): llm={s2_l:.3f}")
ax.text(0.05, s2_l + 0.010,
        f"Conv.-LRM (0M dev): llm={s2_l:.3f}",
        ha="left", va="bottom", fontsize=7.5, color="#666666",
        style="italic")

# BM25 callout
ax.annotate(
    f"BM25: doc_recall +{(bm25_d - ddi_d)*100:.0f}pp vs DDI\nbut llm −4.6σ (p<0.001)",
    xy=(x[3] + bar_w/2 + gap/2, bm25_d),
    xytext=(x[3] + 0.85, bm25_d - 0.10),
    fontsize=7.5, ha="left", va="top", color=C_BASE_LLM,
    arrowprops=dict(arrowstyle="->", color=C_BASE_LLM, lw=1.0,
                    connectionstyle="arc3,rad=-0.2"))

ax.set_xticks(x)
ax.set_xticklabels(labels, ha="center")
ax.set_ylabel("Score", labelpad=6)
ax.set_ylim(0, 0.82)
ax.yaxis.grid(True, ls="--", alpha=0.35, zorder=0)
ax.set_axisbelow(True)

# Region labels
ax.text(1.0, 0.78, "Our methods (60M)", ha="center", fontsize=8,
        color=C_OUR_LLM, style="italic")
ax.text(3.5, 0.78, "Baselines (60M)", ha="center", fontsize=8,
        color=C_BASE_LLM, style="italic")

# Legend
legend_patches = [
    mpatches.Patch(color=C_OUR_LLM,  label="LLM Judge (ours)"),
    mpatches.Patch(color=C_OUR_DR,   label="Doc Recall (ours)"),
    mpatches.Patch(color=C_BASE_LLM, label="LLM Judge (baseline)"),
    mpatches.Patch(color=C_BASE_DR,  label="Doc Recall (baseline)"),
    mpl.lines.Line2D([0],[0], color="#999999", lw=1.3, ls=":",
                     label="Conv.-LRM (0M, ref)"),
]
ax.legend(handles=legend_patches, loc="lower right", framealpha=0.95,
          edgecolor="lightgrey", fancybox=False, fontsize=7.5, ncol=1, bbox_to_anchor=(0.99, 0.01))

ax.text(0.01, 0.02,
        "All bars: 60M tier, test n=400. Conv.-LRM dotted line: 0M dev n=100 (ref).",
        transform=ax.transAxes, ha="left", va="bottom",
        fontsize=6.8, color="gray", style="italic",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.9))

plt.tight_layout()
out_dir = os.path.dirname(os.path.abspath(__file__))
for ext in ["pdf", "png"]:
    out = os.path.join(out_dir, f"fig2_baselines.{ext}")
    plt.savefig(out, dpi=300 if ext == "png" else None)
    print(f"Saved: {out}")
plt.close()
print("Done.")
