from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHODS = ["A-mem", "Mem0", "agent_memory"]
CONSTRUCTION_TIMES = [5140.5, 1350.9, 739.86]
AVG_TOKENS_PER_CONVERSATION = 16641


def draw_construction_time_plot(output_path: str | None = None) -> Path:
	if output_path is None:
		output_path = str(Path(__file__).resolve().parent / "construction_time_comparison.png")

	tokens_per_second = [AVG_TOKENS_PER_CONVERSATION / t for t in CONSTRUCTION_TIMES]

	metric_labels = ["Construction Time", "Tokens per Second"]
	metric_positions = np.array([0.0, 0.72], dtype=float)
	bar_width = 0.12
	method_colors = ["#94A3B8", "#64748B", "#9F1239"]
	method_offsets = np.array([-bar_width, 0.0, bar_width])

	figure, left_axis = plt.subplots(figsize=(6.5, 5.2))
	right_axis = left_axis.twinx()

	legend_handles = []

	for pos, method_name in enumerate(METHODS):
		offset = method_offsets[pos]

		left_bar = left_axis.bar(
			metric_positions[0] + offset,
			CONSTRUCTION_TIMES[pos],
			width=bar_width,
			label=method_name,
			color=method_colors[pos],
		)
		right_bar = right_axis.bar(
			metric_positions[1] + offset,
			tokens_per_second[pos],
			width=bar_width,
			color=method_colors[pos],
		)
		legend_handles.append(left_bar[0])

		left_axis.text(
			left_bar[0].get_x() + left_bar[0].get_width() / 2,
			CONSTRUCTION_TIMES[pos] + max(CONSTRUCTION_TIMES) * 0.03,
			f"{CONSTRUCTION_TIMES[pos]:.2f}",
			ha="center",
			va="bottom",
			fontsize=9,
		)
		right_axis.text(
			right_bar[0].get_x() + right_bar[0].get_width() / 2,
			tokens_per_second[pos] + max(tokens_per_second) * 0.015,
			f"{tokens_per_second[pos]:.2f}",
			ha="center",
			va="bottom",
			fontsize=9,
		)

	left_axis.set_xticks(metric_positions)
	left_axis.set_xticklabels(metric_labels)
	left_axis.set_xlim(-0.42, 1.14)
	left_axis.set_title("Construction Metrics")
	left_axis.set_ylabel("Construction Time")
	right_axis.set_ylabel("Tokens / Second")

	left_axis.set_ylim(0, max(CONSTRUCTION_TIMES) * 1.18)
	right_axis.set_ylim(0, max(tokens_per_second) * 1.45)
	left_axis.legend(handles=legend_handles, labels=METHODS, title="Method", loc="upper right")

	figure.tight_layout()
	output_file = Path(output_path)
	output_file.parent.mkdir(parents=True, exist_ok=True)
	figure.savefig(output_file, dpi=300)
	plt.close(figure)
	return output_file


if __name__ == "__main__":
	saved_path = draw_construction_time_plot()
	print(f"Saved plot to: {saved_path}")
