"""
Latency post-processing for memory-system evaluation runs.

Two ways to use this module:
1. Run as a standalone script: ``python analyze_latency.py <results_file>``.
2. Import and call ``analyze_and_save_latency`` from another module.

The analyser reports:
- A breakdown of the search stage by sub-component
- Comparison across different retrieval strategies
- The effect of cue-index, hybrid search, and episodic-memory toggles
- Token-count statistics
- Percentile summaries (P50/P90/P95)
"""

import json
import argparse
import logging
import os
import statistics
from typing import Dict, List

logger = logging.getLogger(__name__)


def percentile(data: List[float], p: int) -> float:
    """
    Return the p-th percentile of *data* via linear interpolation.

    Args:
        data: A list of numeric samples.
        p: Percentile in the range [0, 100].

    Returns:
        Interpolated percentile value (0.0 when *data* is empty).
    """
    if not data:
        return 0.0

    ordered = sorted(data)
    total = len(ordered)
    rank = (total - 1) * p / 100
    lo = int(rank)
    frac = rank - lo

    if lo + 1 < total:
        return ordered[lo] + frac * (ordered[lo + 1] - ordered[lo])
    return ordered[lo]


def load_results(result_file: str) -> List[Dict]:
    """Load the results JSON and flatten the per-conversation lists."""
    with open(result_file) as fh:
        raw = json.load(fh)

    flattened: List[Dict] = []
    for per_conv in raw.values():
        flattened.extend(per_conv)

    return flattened


def analyze_and_save_latency(output_file: str, output_dir: str):
    """
    Build a latency summary from a results file and persist it to disk.

    This is invoked from ``run_agent_memory.py`` once the eval finishes.

    Args:
        output_file: Path of the results JSON to analyse.
        output_dir: Destination directory for the latency summary file.
    """
    try:
        questions = load_results(output_file)

        if not questions:
            logger.warning("No questions found in results")
            return

        total_search_times: List[float] = []
        primary_times: List[float] = []
        cue_times: List[float] = []
        hybrid_times: List[float] = []
        rrf_times: List[float] = []
        format_times: List[float] = []
        llm_times: List[float] = []
        total_times: List[float] = []

        num_memories: List[int] = []
        total_tokens: List[int] = []
        avg_tokens_per_memory: List[float] = []

        step_counts: List[int] = []
        action_counts: Dict[str, int] = {}
        policy_llm_times: List[float] = []

        for q in questions:
            latency = q.get("latency_breakdown", {})

            if "total_search_time" in latency:
                total_search_times.append(latency["total_search_time"])

            breakdown = latency.get("search_breakdown", {})
            if "search_primary" in breakdown:
                primary_times.append(breakdown["search_primary"])
            if "search_cue" in breakdown:
                cue_times.append(breakdown["search_cue"])
            if "search_hybrid" in breakdown:
                hybrid_times.append(breakdown["search_hybrid"])
            if "search_rrf_merge" in breakdown:
                rrf_times.append(breakdown["search_rrf_merge"])

            if "format_time" in latency:
                format_times.append(latency["format_time"])
            if "llm_time" in latency:
                llm_times.append(latency["llm_time"])
            if "total_time" in latency:
                total_times.append(latency["total_time"])

            prompt_stats = latency.get("prompt_stats", {})
            if "num_memories" in prompt_stats:
                num_memories.append(prompt_stats["num_memories"])
            if "total_tokens" in prompt_stats:
                total_tokens.append(prompt_stats["total_tokens"])
            if "avg_tokens" in prompt_stats:
                avg_tokens_per_memory.append(prompt_stats["avg_tokens"])

            retrieval_steps = latency.get("retrieval_steps", {})
            if "num_steps" in retrieval_steps:
                step_counts.append(retrieval_steps["num_steps"])

            for step in retrieval_steps.get("steps", []):
                act = step.get("action", "UNKNOWN")
                action_counts[act] = action_counts.get(act, 0) + 1

                step_llm = step.get("llm_duration", 0)
                if step_llm and step_llm > 0:
                    policy_llm_times.append(step_llm)

        summary: Dict = {
            "total_questions": len(questions),
            "end_to_end_latency": {},
            "search_latency": {},
            "memory_statistics": {},
            "retrieval_steps": {},
        }

        if total_times:
            summary["end_to_end_latency"] = {
                "total_time_mean": round(statistics.mean(total_times), 4),
                "total_time_median": round(statistics.median(total_times), 4),
                "total_time_p50": round(percentile(total_times, 50), 4),
                "total_time_p90": round(percentile(total_times, 90), 4),
                "total_time_p95": round(percentile(total_times, 95), 4),
                "total_time_min": round(min(total_times), 4),
                "total_time_max": round(max(total_times), 4),
                "total_time_stdev": round(statistics.stdev(total_times), 4) if len(total_times) > 1 else 0,
            }

            mean_total = statistics.mean(total_times)
            stage_breakdown: Dict = {}
            if total_search_times:
                mean_search = statistics.mean(total_search_times)
                stage_breakdown["search_time"] = round(mean_search, 4)
                stage_breakdown["search_pct"] = round(mean_search / mean_total * 100, 2)
            if format_times:
                mean_format = statistics.mean(format_times)
                stage_breakdown["format_time"] = round(mean_format, 4)
                stage_breakdown["format_pct"] = round(mean_format / mean_total * 100, 2)
            if llm_times:
                mean_llm = statistics.mean(llm_times)
                stage_breakdown["llm_time"] = round(mean_llm, 4)
                stage_breakdown["llm_pct"] = round(mean_llm / mean_total * 100, 2)

            summary["end_to_end_latency"]["breakdown"] = stage_breakdown

        if total_search_times:
            mean_total_search = statistics.mean(total_search_times)
            summary["search_latency"] = {
                "total_search_mean": round(mean_total_search, 4),
                "total_search_median": round(statistics.median(total_search_times), 4),
                "total_search_p50": round(percentile(total_search_times, 50), 4),
                "total_search_p90": round(percentile(total_search_times, 90), 4),
                "total_search_p95": round(percentile(total_search_times, 95), 4),
                "components": {},
            }

            component_specs = (
                ("primary_index", primary_times),
                ("cue_index", cue_times),
                ("hybrid_search", hybrid_times),
                ("rrf_merge", rrf_times),
            )
            for comp_key, samples in component_specs:
                if not samples:
                    continue
                comp_mean = statistics.mean(samples)
                summary["search_latency"]["components"][comp_key] = {
                    "mean": round(comp_mean, 4),
                    "median": round(statistics.median(samples), 4),
                    "p50": round(percentile(samples, 50), 4),
                    "p90": round(percentile(samples, 90), 4),
                    "p95": round(percentile(samples, 95), 4),
                    "pct_of_search": round(comp_mean / mean_total_search * 100, 2),
                }

        if llm_times:
            summary["llm_latency"] = {
                "mean": round(statistics.mean(llm_times), 4),
                "median": round(statistics.median(llm_times), 4),
                "p50": round(percentile(llm_times, 50), 4),
                "p90": round(percentile(llm_times, 90), 4),
                "p95": round(percentile(llm_times, 95), 4),
                "min": round(min(llm_times), 4),
                "max": round(max(llm_times), 4),
            }

        if num_memories:
            summary["memory_statistics"] = {
                "num_memories_mean": round(statistics.mean(num_memories), 2),
                "num_memories_median": int(statistics.median(num_memories)),
                "num_memories_p50": int(percentile(num_memories, 50)),
                "num_memories_p90": int(percentile(num_memories, 90)),
                "num_memories_p95": int(percentile(num_memories, 95)),
                "num_memories_min": min(num_memories),
                "num_memories_max": max(num_memories),
            }

        if total_tokens:
            ms = summary["memory_statistics"]
            ms["total_tokens_mean"] = int(statistics.mean(total_tokens))
            ms["total_tokens_median"] = int(statistics.median(total_tokens))
            ms["total_tokens_p50"] = int(percentile(total_tokens, 50))
            ms["total_tokens_p90"] = int(percentile(total_tokens, 90))
            ms["total_tokens_p95"] = int(percentile(total_tokens, 95))
            ms["total_tokens_min"] = min(total_tokens)
            ms["total_tokens_max"] = max(total_tokens)

        if avg_tokens_per_memory:
            summary["memory_statistics"]["avg_tokens_per_memory"] = round(
                statistics.mean(avg_tokens_per_memory), 2
            )

        if step_counts:
            summary["retrieval_steps"] = {
                "num_steps_mean": round(statistics.mean(step_counts), 2),
                "num_steps_median": int(statistics.median(step_counts)),
                "num_steps_p50": int(percentile(step_counts, 50)),
                "num_steps_p90": int(percentile(step_counts, 90)),
                "num_steps_p95": int(percentile(step_counts, 95)),
                "num_steps_min": min(step_counts),
                "num_steps_max": max(step_counts),
                "action_distribution": {},
            }

            grand_actions = sum(action_counts.values())
            for act, cnt in sorted(action_counts.items(), key=lambda kv: -kv[1]):
                summary["retrieval_steps"]["action_distribution"][act] = {
                    "count": cnt,
                    "percentage": round(cnt / grand_actions * 100, 2),
                }

            if policy_llm_times:
                summary["retrieval_steps"]["policy_llm_latency"] = {
                    "mean": round(statistics.mean(policy_llm_times), 4),
                    "median": round(statistics.median(policy_llm_times), 4),
                    "p50": round(percentile(policy_llm_times, 50), 4),
                    "p90": round(percentile(policy_llm_times, 90), 4),
                    "p95": round(percentile(policy_llm_times, 95), 4),
                    "min": round(min(policy_llm_times), 4),
                    "max": round(max(policy_llm_times), 4),
                    "total_calls": len(policy_llm_times),
                }

        latency_file = os.path.join(output_dir, "latency_analysis.json")
        with open(latency_file, "w") as fh:
            json.dump(summary, fh, indent=4)

        logger.info(f"\n==== Latency analysis saved to: {latency_file} ====")

        print_latency_summary(summary, latency_file)

    except Exception as exc:
        logger.error(f"Error analyzing latency: {str(exc)}")
        import traceback
        traceback.print_exc()


def print_latency_summary(summary: Dict, latency_file: str = None):
    """
    Pretty-print the latency summary to stdout in tabular form.

    Args:
        summary: The latency summary dict.
        latency_file: Optional path to display at the end.
    """
    print("\n" + "=" * 80)
    print(" " * 25 + "LATENCY ANALYSIS SUMMARY")
    print("=" * 80)

    if "end_to_end_latency" in summary and summary["end_to_end_latency"]:
        e2e = summary["end_to_end_latency"]
        print("\n┌─────────────────────────────────────────────────────────────────────────────┐")
        print("│ END-TO-END LATENCY                                                          │")
        print("├──────────────┬──────────┬──────────┬──────────┬──────────┬──────────────────┤")
        print("│   Metric     │   Mean   │  Median  │   P90    │   P95    │   Range          │")
        print("├──────────────┼──────────┼──────────┼──────────┼──────────┼──────────────────┤")
        print(f"│ Total Time   │ {e2e['total_time_mean']:>6.4f}s │ {e2e['total_time_median']:>6.4f}s │ {e2e['total_time_p90']:>6.4f}s │ {e2e['total_time_p95']:>6.4f}s │ {e2e['total_time_min']:>6.4f}s-{e2e['total_time_max']:>5.4f}s │")
        print("└──────────────┴──────────┴──────────┴──────────┴──────────┴──────────────────┘")

        if "breakdown" in e2e:
            bd = e2e["breakdown"]
            print("\n  Stage Breakdown:")
            print("  ┌─────────────────┬──────────┬────────────┐")
            print("  │     Stage       │   Time   │ Percentage │")
            print("  ├─────────────────┼──────────┼────────────┤")
            if "search_time" in bd:
                print(f"  │ Search          │ {bd['search_time']:>6.4f}s │   {bd['search_pct']:>5.1f}%   │")
            if "format_time" in bd:
                print(f"  │ Formatting      │ {bd['format_time']:>6.4f}s │   {bd['format_pct']:>5.1f}%   │")
            if "llm_time" in bd:
                print(f"  │ LLM Generation  │ {bd['llm_time']:>6.4f}s │   {bd['llm_pct']:>5.1f}%   │")
            print("  └─────────────────┴──────────┴────────────┘")

    if "search_latency" in summary and summary["search_latency"]:
        sl = summary["search_latency"]
        print("\n┌─────────────────────────────────────────────────────────────────────────────┐")
        print("│ SEARCH LATENCY                                                              │")
        print("├──────────────┬──────────┬──────────┬──────────┬──────────┬──────────────────┤")
        print("│   Metric     │   Mean   │  Median  │   P90    │   P95    │   Min - Max      │")
        print("├──────────────┼──────────┼──────────┼──────────┼──────────┼──────────────────┤")
        print(f"│ Total Search │ {sl['total_search_mean']:>6.4f}s │ {sl['total_search_median']:>6.4f}s │ {sl['total_search_p90']:>6.4f}s │ {sl['total_search_p95']:>6.4f}s │       N/A        │")
        print("└──────────────┴──────────┴──────────┴──────────┴──────────┴──────────────────┘")

        if "components" in sl and sl["components"]:
            print("\n  Search Components Breakdown:")
            print("  ┌──────────────────┬──────────┬──────────┬──────────┬────────────┐")
            print("  │    Component     │   Mean   │   P90    │   P95    │ % of Search│")
            print("  ├──────────────────┼──────────┼──────────┼──────────┼────────────┤")

            ordered_components = ["primary_index", "cue_index", "hybrid_search", "rrf_merge"]
            for comp_key in ordered_components:
                if comp_key not in sl["components"]:
                    continue
                comp = sl["components"][comp_key]
                pretty = comp_key.replace("_", " ").title()
                print(f"  │ {pretty:16s} │ {comp['mean']:>6.4f}s │ {comp['p90']:>6.4f}s │ {comp['p95']:>6.4f}s │   {comp['pct_of_search']:>5.1f}%   │")

            print("  └──────────────────┴──────────┴──────────┴──────────┴────────────┘")

    if "llm_latency" in summary:
        ll = summary["llm_latency"]
        print("\n┌─────────────────────────────────────────────────────────────────────────────┐")
        print("│ LLM GENERATION LATENCY                                                      │")
        print("├──────────────┬──────────┬──────────┬──────────┬──────────┬──────────────────┤")
        print("│   Metric     │   Mean   │  Median  │   P90    │   P95    │   Min - Max      │")
        print("├──────────────┼──────────┼──────────┼──────────┼──────────┼──────────────────┤")
        print(f"│ LLM Time     │ {ll['mean']:>6.4f}s │ {ll['median']:>6.4f}s │ {ll['p90']:>6.4f}s │ {ll['p95']:>6.4f}s │ {ll['min']:>6.4f}s-{ll['max']:>5.4f}s │")
        print("└──────────────┴──────────┴──────────┴──────────┴──────────┴──────────────────┘")

    if "memory_statistics" in summary and summary["memory_statistics"]:
        ms = summary["memory_statistics"]
        print("\n┌─────────────────────────────────────────────────────────────────────────────┐")
        print("│ MEMORY STATISTICS                                                           │")
        print("├────────────────────┬──────────┬──────────┬──────────┬──────────┬───────────┤")
        print("│      Metric        │   Mean   │  Median  │   P90    │   P95    │    Max    │")
        print("├────────────────────┼──────────┼──────────┼──────────┼──────────┼───────────┤")

        if "num_memories_mean" in ms:
            print(f"│ Memories per Query │ {ms['num_memories_mean']:>8.1f} │ {ms['num_memories_median']:>8d} │ {ms.get('num_memories_p90', 0):>8d} │ {ms.get('num_memories_p95', 0):>8d} │ {ms['num_memories_max']:>9d} │")

        if "total_tokens_mean" in ms:
            print(f"│ Tokens per Query   │ {ms['total_tokens_mean']:>8d} │ {ms['total_tokens_median']:>8d} │ {ms.get('total_tokens_p90', 0):>8d} │ {ms.get('total_tokens_p95', 0):>8d} │ {ms['total_tokens_max']:>9d} │")

        if "avg_tokens_per_memory" in ms:
            print("├────────────────────┴──────────┴──────────┴──────────┴──────────┴───────────┤")
            print(f"│ Avg Tokens per Memory: {ms['avg_tokens_per_memory']:>6.1f}                                              │")

        print("└─────────────────────────────────────────────────────────────────────────────┘")

    if "retrieval_steps" in summary and summary["retrieval_steps"] and "num_steps_mean" in summary["retrieval_steps"]:
        rs = summary["retrieval_steps"]
        print("\n┌─────────────────────────────────────────────────────────────────────────────┐")
        print("│ RETRIEVAL STEPS (Policy Retriever)                                          │")
        print("├────────────────────┬──────────┬──────────┬──────────┬──────────────────────┤")
        print("│      Metric        │   Mean   │  Median  │   P90    │        P95           │")
        print("├────────────────────┼──────────┼──────────┼──────────┼──────────────────────┤")
        print(f"│ Steps per Query    │ {rs['num_steps_mean']:>8.1f} │ {rs['num_steps_median']:>8d} │ {rs.get('num_steps_p90', 0):>8d} │ {rs.get('num_steps_p95', 0):>20d} │")
        print("└────────────────────┴──────────┴──────────┴──────────┴──────────────────────┘")

        if "action_distribution" in rs and rs["action_distribution"]:
            print("\n  Action Distribution:")
            print("  ┌──────────────────┬──────────┬────────────┐")
            print("  │      Action      │  Count   │ Percentage │")
            print("  ├──────────────────┼──────────┼────────────┤")
            for act, info in rs["action_distribution"].items():
                print(f"  │ {act:16s} │ {info['count']:>8d} │   {info['percentage']:>5.1f}%   │")
            print("  └──────────────────┴──────────┴────────────┘")

        if "policy_llm_latency" in rs:
            pll = rs["policy_llm_latency"]
            print("\n  Policy LLM Latency:")
            print("  ┌──────────────┬──────────┬──────────┬──────────┬──────────┬──────────────────┐")
            print("  │   Metric     │   Mean   │  Median  │   P90    │   P95    │   Min - Max      │")
            print("  ├──────────────┼──────────┼──────────┼──────────┼──────────┼──────────────────┤")
            print(f"  │ LLM Time     │ {pll['mean']:>6.4f}s │ {pll['median']:>6.4f}s │ {pll['p90']:>6.4f}s │ {pll['p95']:>6.4f}s │ {pll['min']:>6.4f}s-{pll['max']:>5.4f}s │")
            print(f"  │ Total Calls  │ {pll['total_calls']:>8d} │          │          │          │                  │")
            print("  └──────────────┴──────────┴──────────┴──────────┴──────────┴──────────────────┘")

    print("\n" + "=" * 80)

    if latency_file:
        print(f"\n📊 Latency analysis saved to: {latency_file}")

    print("\n" + "=" * 80 + "\n")


def main():
    """Standalone CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze latency results from memory system evaluation"
    )
    parser.add_argument(
        "result_file",
        type=str,
        help="Path to results JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to save latency analysis (defaults to same dir as result_file)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed per-question analysis",
    )

    args = parser.parse_args()

    out_dir = args.output_dir if args.output_dir else os.path.dirname(args.result_file)

    print(f"Loading results from: {args.result_file}")
    questions = load_results(args.result_file)
    print(f"Loaded {len(questions)} questions")

    analyze_and_save_latency(args.result_file, out_dir)

    if args.verbose:
        print("\n" + "=" * 60)
        print("PER-QUESTION DETAILS")
        print("=" * 60)

        for pos, q in enumerate(questions):
            latency = q.get("latency_breakdown", {})
            print(f"\nQuestion {pos + 1}: {q['question'][:60]}...")
            print(f"  Total time: {latency.get('total_time', 0):.4f}s")
            print(f"  Search:     {latency.get('total_search_time', 0):.4f}s")
            print(f"  Format:     {latency.get('format_time', 0):.4f}s")
            print(f"  LLM:        {latency.get('llm_time', 0):.4f}s")
            print(f"  Memories:   {latency.get('prompt_stats', {}).get('num_memories', 0)}")
            print(f"  Tokens:     {latency.get('prompt_stats', {}).get('total_tokens', 0)}")

            steps = latency.get("retrieval_steps", {})
            if steps:
                print(f"  Steps:      {steps.get('num_steps', 0)}")


if __name__ == "__main__":
    main()
