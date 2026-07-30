#!/usr/bin/env python3
"""Generate the final aggregate figures reported by the research artifact.

The script is deliberately self-contained: it does not depend on private
experiment paths, and every output is written under ``figures/generated/``.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "generated"
OUT.mkdir(exist_ok=True)

BLUE = "#0072B2"
SKY = "#56B4E9"
ORANGE = "#E69F00"
GREEN = "#009E73"
RED = "#D55E00"
PURPLE = "#CC79A7"
GRAY = "#626262"
LIGHT = "#F4F5F7"

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linestyle": ":",
        "legend.fontsize": 8.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf", dpi=300)
    fig.savefig(OUT / f"{name}.png", dpi=220)
    plt.close(fig)


def pipeline() -> None:
    fig, ax = plt.subplots(figsize=(11.6, 3.25))
    ax.set_xlim(0, 11.6)
    ax.set_ylim(0, 3.25)
    ax.axis("off")

    stages = [
        (0.15, 1.0, 1.45, 1.55, "Documents", "PDF · mail · chat\nwiki · code · tables", LIGHT),
        (1.95, 1.0, 1.65, 1.55, "Memory build", "section-aware chunks\n+ typed distillation", "#EAF3F8"),
        (3.95, 0.30, 1.65, 1.05, "Raw index", "verbatim evidence\n+ provenance", "#DCEEF7"),
        (3.95, 1.75, 1.65, 1.05, "Memory index", "retrieval keys\n+ atomic facts", "#FCEBCB"),
        (6.00, 1.0, 1.55, 1.55, "Recall", "original + rewritten\nqueries; rank fusion", "#E8F3EF"),
        (7.95, 1.0, 1.45, 1.55, "Evidence control", "cross-encoder rerank\n+ raw-source fetch", "#F3E8F1"),
        (9.80, 1.0, 1.55, 1.55, "Reader", "bounded context\n+ citation extraction", "#FBE7E1"),
    ]
    for x, y, w, h, title, body, color in stages:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.04,rounding_size=0.06",
            facecolor=color,
            edgecolor="#3F4650",
            linewidth=1.1,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h * 0.70, title, ha="center", va="center",
                fontsize=10, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.34, body, ha="center", va="center",
                fontsize=8.4, color="#343A40", linespacing=1.25)

    arrows = [
        ((1.62, 1.78), (1.91, 1.78)),
        ((3.62, 1.78), (3.91, 2.22)),
        ((3.62, 1.72), (3.91, 0.85)),
        ((5.62, 2.23), (5.96, 1.90)),
        ((5.62, 0.85), (5.96, 1.58)),
        ((7.57, 1.78), (7.91, 1.78)),
        ((9.42, 1.78), (9.76, 1.78)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12,
                                     color="#3F4650", linewidth=1.2))

    ax.text(4.78, 3.02, "persistent, corpus-scale external memory",
            ha="center", va="center", fontsize=9.2, color=BLUE, fontweight="bold")
    ax.annotate("", xy=(5.55, 2.90), xytext=(3.98, 2.90),
                arrowprops=dict(arrowstyle="<->", color=BLUE, lw=1.1))
    ax.text(10.58, 0.62, "small active context",
            ha="center", va="center", fontsize=9.2, color=RED, fontweight="bold")
    ax.annotate("", xy=(11.30, 0.78), xytext=(9.86, 0.78),
                arrowprops=dict(arrowstyle="<->", color=RED, lw=1.1))
    save(fig, "pipeline")


def headline() -> None:
    metrics = ["Overall", "Correctness", "Completeness", "Document recall"]
    leader = np.array([68.22, 81.60, 72.86, 79.02])
    ours = np.array([78.29, 88.00, 80.88, 87.29])
    ci_lo = np.array([74.84, 84.75, 77.72, 84.24])
    ci_hi = np.array([81.70, 91.25, 83.92, 90.24])
    y = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(7.6, 3.65))
    ax.scatter(leader, y + 0.14, marker="s", s=55, color=GRAY, label="External reference", zorder=4)
    ax.errorbar(
        ours, y - 0.14, xerr=np.vstack([ours - ci_lo, ci_hi - ours]),
        fmt="o", markersize=6.5, capsize=3.5, color=BLUE, ecolor=BLUE,
        linewidth=1.4, label="Our method", zorder=5,
    )
    for i, (b, o) in enumerate(zip(leader, ours)):
        ax.plot([b, o], [i + 0.14, i - 0.14], color="#B8BDC4", lw=1, zorder=1)
        ax.text(92.2, i, f"+{o-b:.2f} pp", va="center", ha="right",
                fontsize=8.8, color=GREEN, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(metrics)
    ax.invert_yaxis()
    ax.set_xlim(64, 93)
    ax.set_xlabel("Score (%)")
    ax.set_title("External evaluation comparison")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    save(fig, "headline")


def development() -> None:
    labels = [
        "Hierarchy", "Relations", "Combined", "Initial\nDDI", "Improved\nDDI",
        "Reranked\nDDI", "Failed\nbranch", "Query-\nexpanded", "Final\nDDI",
    ]
    vals = np.array([60.48, 59.16, 65.14, 67.42, 72.44, 74.46, 66.40, 78.29, 78.29])
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    ax.plot(x[:6], vals[:6], color=BLUE, marker="o", lw=2, zorder=3)
    ax.plot(x[5:7], vals[5:7], color=RED, marker="X", lw=1.5, ls="--", zorder=3)
    ax.plot([5, 7, 8], vals[[5, 7, 8]], color=GREEN, marker="o", lw=2, zorder=3)
    ax.scatter([8], [vals[8]], marker="*", s=150, color=GREEN, edgecolor="black", zorder=5)
    for xi, yi in zip(x, vals):
        dy = -2.0 if xi == 6 else 1.0
        ax.text(xi, yi + dy, f"{yi:.2f}", ha="center",
                va="top" if xi == 6 else "bottom", fontsize=7.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(55, 82)
    ax.set_ylabel("Overall score (%)")
    ax.set_title("From memory abstractions to a provenance-preserving dual index")
    ax.text(6, 61.6, "added machinery\nwithout better retrieval", color=RED,
            ha="center", fontsize=8.2)
    fig.tight_layout()
    save(fig, "development")


def ablation() -> None:
    names = ["No citation gate", "No query expansion", "No reranker",
             "No distillation", "No dual index"]
    delta = np.array([0.0, -2.92, -4.09, -11.64, -14.73])
    colors = [GRAY, ORANGE, ORANGE, BLUE, BLUE]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    bars = ax.barh(y, delta, color=colors, edgecolor="#333333", linewidth=0.6)
    ax.axvline(0, color="#333333", lw=0.9)
    for b, v in zip(bars, delta):
        ax.text(v - 0.3 if v < 0 else 0.3, b.get_y() + b.get_height() / 2,
                f"{v:+.2f} pp", va="center", ha="right" if v < 0 else "left",
                fontsize=8.8, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlim(-17, 2.5)
    ax.set_xlabel("Change in Combined score")
    ax.set_title("Controlled leave-one-component-out ablation")
    fig.tight_layout()
    save(fig, "ablation")


def scaling() -> None:
    tiers = np.array([10, 20, 60, 100, 150, 250])
    overall = np.array([70.64, 71.67, 64.19, 63.17, 60.49, 58.02])
    recall = np.array([84.00, 84.00, 78.36, 75.23, 72.16, 66.79])
    correct = np.array([82.50, 84.00, 78.25, 77.00, 75.75, 73.50])
    complete = np.array([74.63, 73.78, 67.71, 66.87, 64.31, 62.01])
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    for vals, label, color, marker in [
        (recall, "Document recall", BLUE, "o"),
        (correct, "Correctness", GREEN, "s"),
        (complete, "Completeness", ORANGE, "^"),
        (overall, "Overall", PURPLE, "D"),
    ]:
        ax.plot(tiers, vals, marker=marker, label=label, color=color, lw=2, ms=5.5)
    ax.axvspan(20, 60, color=RED, alpha=0.08)
    ax.text(40, 57.2, "onset of scale degradation", color=RED, ha="center", fontsize=8.5)
    ax.set_xticks(tiers)
    ax.set_xlabel("Indexed corpus (millions of tokens)")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(55, 88)
    ax.set_title("Six-tier scaling with the low-cost reader")
    ax.legend(ncol=2, frameon=False, loc="upper right")
    fig.tight_layout()
    save(fig, "scaling")


def oracle() -> None:
    labels = ["20M", "60M", "250M"]
    x = np.arange(3)
    series = [
        ([82.50, 84.00, 82.25], "Mini · gold", ORANGE, "-"),
        ([84.00, 78.25, 73.50], "Mini · retrieved", ORANGE, "--"),
        ([85.75, 85.00, 83.75], "Full · gold", BLUE, "-"),
        ([70.25, 66.00, 56.25], "Full · retrieved", BLUE, "--"),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 4.15))
    for vals, label, color, ls in series:
        marker = "o" if ls == "-" else "s"
        ax.plot(x, vals, label=label, color=color, ls=ls, marker=marker, lw=2.2, ms=6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Indexed corpus size")
    ax.set_ylabel("Correctness (%)")
    ax.set_ylim(52, 90)
    ax.set_title("Oracle intervention isolates retrieval as the scaling bottleneck")
    ax.text(1.98, 84.7, "gold evidence: nearly flat", ha="right", color=GREEN,
            fontsize=8.8, fontweight="bold")
    ax.text(1.98, 59.0, "retrieved evidence: declines", ha="right", color=RED,
            fontsize=8.8, fontweight="bold")
    ax.legend(ncol=2, frameon=False, loc="lower left")
    fig.tight_layout()
    save(fig, "oracle")


def external() -> None:
    tasks = ["FinanceBench", "HotpotQA", "LoCoMo", "UltraDomain"]
    combined = [81.05, 86.37, 42.34, 17.73]
    correct = [82.0, 87.0, 38.0, 9.5]
    complete = [87.29, 87.09, 42.49, 13.28]
    x = np.arange(len(tasks))
    width = 0.23
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 3.8),
                                   gridspec_kw={"width_ratios": [1.55, 0.85]})
    ax1.bar(x - width, combined, width, label="Combined", color=BLUE)
    ax1.bar(x, correct, width, label="Correct", color=GREEN)
    ax1.bar(x + width, complete, width, label="Complete", color=ORANGE)
    ax1.set_xticks(x)
    ax1.set_xticklabels(tasks, rotation=12, ha="right")
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("Score (%)")
    ax1.set_title("Transfer across evidence regimes")
    ax1.legend(frameon=False, ncol=3, loc="upper right")

    embed_tasks = ["HotpotQA", "FinanceBench"]
    bge = [86.37, 81.05]
    small = [59.13, 81.03]
    x2 = np.arange(2)
    ax2.bar(x2 - 0.17, bge, 0.34, label="BGE-large", color=BLUE)
    ax2.bar(x2 + 0.17, small, 0.34, label="Smaller hosted embedder", color=GRAY)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(embed_tasks, rotation=12, ha="right")
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("Combined score")
    ax2.set_title("Embedder sensitivity")
    ax2.legend(frameon=False, fontsize=7.8, loc="lower right")
    fig.tight_layout()
    save(fig, "external")


def token_efficiency() -> None:
    modes = ["Distilled only", "Mixed", "Full detail"]
    tok10 = np.array([1012.5, 2049.0, 6532.4])
    tok20 = np.array([1020.4, 2025.2, 6497.4])
    cor10 = np.array([21.0, 35.0, 47.0])
    cor20 = np.array([17.0, 36.0, 56.0])
    colors = [ORANGE, BLUE, GRAY]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.8))
    for i, mode in enumerate(modes):
        ax1.plot([tok10[i], tok20[i]], [cor10[i], cor20[i]], color=colors[i],
                 lw=1.2, alpha=0.7)
        ax1.scatter(tok10[i], cor10[i], color=colors[i], marker="o", s=52,
                    label=f"{mode} · 10M")
        ax1.scatter(tok20[i], cor20[i], facecolor="white", edgecolor=colors[i],
                    marker="o", s=52, linewidth=1.6, label=f"{mode} · 20M")
    ax1.set_xscale("log")
    ax1.set_xlim(750, 8000)
    ax1.set_ylim(10, 62)
    ax1.set_xlabel("Input tokens per query (log scale)")
    ax1.set_ylabel("Correctness (%)")
    ax1.set_title("Quality–context trade-off")
    ax1.legend(frameon=False, fontsize=7.1, ncol=2, loc="upper left")

    labels = ["10M", "20M"]
    ratios = [3.19, 3.21]
    bars = ax2.bar(labels, ratios, color=[BLUE, SKY], width=0.55)
    ax2.bar_label(bars, labels=[f"{v:.2f}×" for v in ratios],
                  padding=4, fontsize=10, fontweight="bold")
    ax2.set_ylim(0, 3.7)
    ax2.set_ylabel("Full-detail / mixed token ratio")
    ax2.set_title("Savings persist across scale")
    ax2.axhline(3.2, color=GRAY, ls=":", lw=1)
    fig.tight_layout()
    save(fig, "token_efficiency")


def retrieval_citation() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.8, 3.6))
    bars = ax1.bar(["Missed targets", "Recovered"], [33, 13],
                   color=[GRAY, BLUE], width=0.58)
    ax1.bar_label(bars, padding=3, fontsize=10, fontweight="bold")
    ax1.set_ylim(0, 38)
    ax1.set_ylabel("Targets")
    ax1.set_title("Query expansion recovers 39.4%")
    bars = ax2.bar(["Reader pool", "Reported sources"], [5.0, 2.31],
                   color=[GRAY, GREEN], width=0.58)
    ax2.bar_label(bars, labels=["5.00", "2.31"], padding=3,
                  fontsize=10, fontweight="bold")
    ax2.set_ylim(0, 5.8)
    ax2.set_ylabel("Documents per query")
    ax2.set_title("Citation gate retains 46%")
    fig.tight_layout()
    save(fig, "retrieval_citation")


def token_breakdown() -> None:
    categories = ["Question", "Answer"]
    means = np.array([37, 439])
    medians = np.array([34, 221])
    p95 = np.array([63, 1622])
    x = np.arange(2)
    fig, ax = plt.subplots(figsize=(5.7, 3.6))
    bars = ax.bar(x, means, color=[BLUE, ORANGE], width=0.52)
    ax.vlines(x, medians, p95, color="#333333", lw=1.5)
    ax.scatter(x, medians, marker="_", s=180, color="#333333", zorder=4)
    ax.scatter(x, p95, marker="_", s=180, color="#333333", zorder=4)
    ax.bar_label(bars, labels=[f"mean {v}" for v in means], padding=3,
                 fontsize=9, fontweight="bold")
    for xi, med, high in zip(x, medians, p95):
        ax.text(xi + 0.11, med, f"median {med}", va="center", fontsize=8)
        ax.text(xi + 0.11, high, f"p95 {high}", va="center", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylim(15, 2500)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Tokens per query (log scale)")
    ax.set_title("Answer length has a long tail")
    fig.tight_layout()
    save(fig, "token_breakdown")


def operations() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.45))
    ax = axes[0]
    vals = [9.3, 5.6 * 24]
    bars = ax.bar(["60M", "250M"], vals, color=[BLUE, ORANGE], width=0.55)
    ax.bar_label(bars, labels=["9.3 h", "5.6 d"], padding=3, fontsize=9, fontweight="bold")
    ax.set_ylabel("Build wall time (hours)")
    ax.set_title("Offline memory build")
    ax.text(0.5, -0.19, "250M duration = 134.4 hours", transform=ax.transAxes,
            ha="center", fontsize=7.8, color=GRAY)

    ax = axes[1]
    x = np.arange(2)
    retrieval = [0.30, 1.10]
    total = [2.5, 3.3]
    ax.bar(x - 0.17, retrieval, 0.34, label="Retrieval", color=GREEN)
    ax.bar(x + 0.17, total, 0.34, label="End-to-end", color=BLUE)
    ax.set_xticks(x)
    ax.set_xticklabels(["60M", "100M"])
    ax.set_ylabel("Seconds per query")
    ax.set_title("Interactive serving")
    ax.legend(frameon=False, fontsize=7.6)

    ax = axes[2]
    costs = [3.75, 0.0094]
    bars = ax.bar(["400 queries", "Per query"], costs, color=[BLUE, ORANGE], width=0.55)
    ax.set_yscale("log")
    ax.set_ylim(0.004, 8)
    ax.bar_label(bars, labels=["$3.75", "$0.0094"], padding=3,
                 fontsize=9, fontweight="bold")
    ax.set_ylabel("Recorded API cost (USD, log scale)")
    ax.set_title("Evaluation cost")
    fig.tight_layout()
    save(fig, "operations")


def main() -> None:
    for fn in [
        pipeline,
        headline,
        development,
        ablation,
        scaling,
        oracle,
        external,
        token_efficiency,
        retrieval_citation,
        token_breakdown,
        operations,
    ]:
        fn()
    print(f"Wrote figures to {OUT}")


if __name__ == "__main__":
    main()
