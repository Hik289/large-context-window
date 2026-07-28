"""
external_benchmarks.py
FIG — §4 External generalization probes: 4 benchmarks x 3 metrics.
Grouped bar chart comparing Correct% / Complete% / Combined across
FinanceBench, HotpotQA, LoCoMo, UltraDomain.

FinanceBench / HotpotQA are strong self-contained-evidence probes (per-question
evidence corpus, no distractor pool) and score high (~80-87%). LoCoMo /
UltraDomain are harder large-KB retrieval probes (must find evidence in a
large multi-document pool) and score much lower (~10-42%), reflecting the
added retrieval difficulty rather than a reasoning failure.

Colorblind-safe: Okabe-Ito palette, consistent with figures_v2 / figures_v22.
Data source: analysis/v22_external_final.md (general API + local BGE-large-en-v1.5
embedder, unified across all 4 benchmarks).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Palette (Okabe-Ito) ──────────────────────────────────────────────────────
C_CORRECT  = "#0072B2"   # blue
C_COMPLETE = "#E69F00"   # orange
C_COMBINED = "#009E73"   # green

# ── Data (analysis/v22_external_final.md) ────────────────────────────────────
benchmarks = ["FinanceBench", "HotpotQA", "LoCoMo", "UltraDomain"]
n_values   = [150, 200, 200, 200]
correct    = [82.0, 87.0, 38.0, 9.5]
complete   = [87.29, 87.09, 42.49, 13.28]
combined   = [81.05, 86.37, 42.34, 17.73]

# Group split: strong self-contained-evidence probes vs harder large-KB probes
group_note = ["self-contained evidence", "self-contained evidence",
              "large-KB retrieval", "large-KB retrieval"]

x = np.arange(len(benchmarks))
width = 0.25

fig, ax = plt.subplots(figsize=(9.5, 5.6))

bars1 = ax.bar(x - width, correct, width, label="Correct %",
               color=C_CORRECT, edgecolor="#2B2B2B", linewidth=0.8, zorder=3)
bars2 = ax.bar(x, complete, width, label="Complete %",
               color=C_COMPLETE, edgecolor="#2B2B2B", linewidth=0.8, zorder=3)
bars3 = ax.bar(x + width, combined, width, label="Combined %",
               color=C_COMBINED, edgecolor="#2B2B2B", linewidth=0.8, zorder=3)

# ── Value labels ──────────────────────────────────────────────────────────────
for bars in (bars1, bars2, bars3):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1.2, f"{h:.1f}",
                 ha="center", va="bottom", fontsize=9, fontweight="bold",
                 color="#1a1a1a")

# ── Shaded background bands to visually separate the two probe regimes ──────
ax.axvspan(-0.5, 1.5, color="#0072B2", alpha=0.05, zorder=0)
ax.axvspan(1.5, 3.5, color="#DC267F", alpha=0.05, zorder=0)

# ── Axes ─────────────────────────────────────────────────────────────────────
xticklabels = [f"{b}\n(n={n})" for b, n in zip(benchmarks, n_values)]
ax.set_xticks(x)
ax.set_xticklabels(xticklabels, fontsize=11)
ax.set_ylabel("Score (%)", fontsize=12, fontweight="bold", labelpad=8)
ax.set_ylim(0, 100)
ax.yaxis.grid(True, alpha=0.25, linestyle=":", linewidth=0.8)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)

# ── Regime annotations above the plot ────────────────────────────────────────
ax.text(0.5, 97, "Self-contained-evidence probes\n(strong)",
        ha="center", va="top", fontsize=9.5, fontweight="bold",
        color="#0072B2", style="italic")
ax.text(2.5, 97, "Large-KB retrieval probes\n(harder)",
        ha="center", va="top", fontsize=9.5, fontweight="bold",
        color="#DC267F", style="italic")

# ── Title ─────────────────────────────────────────────────────────────────────
ax.set_title("External Generalization Probes: Wide Performance Range\nAcross Evidence-Retrieval Difficulty",
              fontsize=14, fontweight="bold", pad=14)

# ── Legend ─────────────────────────────────────────────────────────────────────
ax.legend(fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.14),
          ncol=3, framealpha=0.95, edgecolor="#cccccc")

# ── Footnote ─────────────────────────────────────────────────────────────────
fig.text(0.5, -0.02,
         "Unified evaluation: general API (answer+judge) + local BGE-large-en-v1.5 (1024d) embedder across all 4 benchmarks.\n"
         "FinanceBench/HotpotQA use per-question self-contained evidence corpora (no distractor pool); LoCoMo/UltraDomain require\n"
         "retrieval from a large multi-document knowledge base, explaining the lower scores.",
         ha="center", fontsize=8, color="#777777", style="italic")

fig.tight_layout(pad=0.5)

# ── Save ─────────────────────────────────────────────────────────────────────
out_dir = "./paper/v22/figures_v22"
fig.savefig(f"{out_dir}/external_benchmarks.pdf", dpi=300, bbox_inches="tight")
fig.savefig(f"{out_dir}/external_benchmarks.png", dpi=200, bbox_inches="tight")
print("\u2713 external_benchmarks saved")
