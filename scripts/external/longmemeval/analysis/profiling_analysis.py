"""
Helpers used by the profiling notebook.

Loads the per-run profiling artefacts and produces token-centric plots and a
sub-operation breakdown.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional


# Standard set of sub-operation columns and their human-readable labels (in
# rough execution order).
_SUB_OP_COLUMNS = [
    ('generate_memory_entries_ms', 'LLM: Extract Memories'),
    ('generate_episodic_memory_ms', 'LLM: Episodic Memory'),
    ('query_candidates_total_ms', 'DB: Query Candidates'),
    ('decide_update_total_ms', 'LLM: Update Decision'),
    ('chromadb_add_total_ms', 'DB: ChromaDB Add'),
    ('check_existing_total_ms', 'DB: Check Existing'),
]


def load_profiling_data(run_id: str, output_dir: str = "./profiling_output"):
    """Read the profiling artefacts produced for ``run_id`` from ``output_dir``."""
    base = Path(output_dir)

    timing_df = pd.read_csv(base / f"timing_records_{run_id}.csv")

    with open(base / f"segment_visualizations_{run_id}.json", 'r') as f:
        segment_viz = json.load(f)

    sub_op_path = base / f"sub_operation_timings_{run_id}.csv"
    sub_op_df = pd.read_csv(sub_op_path) if sub_op_path.exists() else None

    return timing_df, segment_viz, sub_op_df


def plot_sub_operation_breakdown(sub_op_df: pd.DataFrame, question_id: Optional[str] = None):
    """
    Visualise where the time inside ``client.add()`` is spent.
    """
    if sub_op_df is None or len(sub_op_df) == 0:
        print("No sub-operation data available")
        return None

    # Restrict to a single question if requested.
    df = sub_op_df[sub_op_df['question_id'] == question_id] if question_id else sub_op_df

    sub_ops = _SUB_OP_COLUMNS

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Pie chart - share of total time across sub-operations.
    ax1 = axes[0, 0]
    totals = [df[col].sum() for col, _ in sub_ops]
    labels = [name for _, name in sub_ops]
    non_zero = [(t, l) for t, l in zip(totals, labels) if t > 0]

    if non_zero:
        totals_nz, labels_nz = zip(*non_zero)
        colors = plt.cm.Set2(np.linspace(0, 1, len(labels_nz)))
        wedges, texts, autotexts = ax1.pie(
            totals_nz, labels=labels_nz, autopct='%1.1f%%',
            colors=colors, pctdistance=0.75, textprops={'fontsize': 9}
        )
        ax1.set_title('Time Distribution by Sub-Operation', fontweight='bold')

    # 2. Stacked bar - per-segment time breakdown.
    ax2 = axes[0, 1]
    x_axis = range(len(df))
    stack_bottom = np.zeros(len(df))
    palette = plt.cm.Set2(np.linspace(0, 1, len(sub_ops)))

    for (col, name), color in zip(sub_ops, palette):
        if col in df.columns:
            vals = df[col].values
            ax2.bar(x_axis, vals, bottom=stack_bottom, label=name.split(': ')[1], color=color, width=0.8)
            stack_bottom += vals

    ax2.set_xlabel('Segment Index')
    ax2.set_ylabel('Time (ms)')
    ax2.set_title('Time Breakdown per Segment', fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8)

    # 3. Tokens vs LLM extraction time (the key correlation).
    ax3 = axes[1, 0]
    if 'total_tokens' in df.columns:
        ax3.scatter(df['total_tokens'], df['generate_memory_entries_ms'],
                    alpha=0.7, s=60, c='steelblue', label='Memory Extraction')

        # Linear trend line over the points.
        if len(df) > 1:
            coeffs = np.polyfit(df['total_tokens'], df['generate_memory_entries_ms'], 1)
            poly = np.poly1d(coeffs)
            x_line = np.linspace(df['total_tokens'].min(), df['total_tokens'].max(), 100)
            ax3.plot(x_line, poly(x_line), "r--", alpha=0.8,
                     label=f'Trend: {coeffs[0]:.2f}ms/token')

        ax3.set_xlabel('Input Tokens')
        ax3.set_ylabel('LLM Extraction Time (ms)')
        ax3.set_title('Tokens vs LLM Processing Time', fontweight='bold')
        ax3.legend()

    # 4. Number of generated entries vs total upsert time.
    ax4 = axes[1, 1]
    if 'num_entries_generated' in df.columns:
        ax4.scatter(df['num_entries_generated'], df['upsert_entries_total_ms'],
                    alpha=0.7, s=60, c='coral')
        ax4.set_xlabel('Memory Entries Generated')
        ax4.set_ylabel('Total Upsert Time (ms)')
        ax4.set_title('Entries vs Upsert Time', fontweight='bold')

    suptitle = f'Sub-Operation Breakdown (Question {question_id})' if question_id else 'Sub-Operation Breakdown (All)'
    plt.suptitle(suptitle, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_tokens_per_segment(segment_viz: List[Dict], question_id: Optional[str] = None):
    """
    Token-centric per-segment plots (bar / histogram / scatter / cumulative).
    """
    chosen = [s for s in segment_viz if s['question_id'] == question_id] if question_id else segment_viz

    if not chosen:
        print(f"No segments found" + (f" for question {question_id}" if question_id else ""))
        return None

    df = pd.DataFrame(chosen)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Tokens per segment (bar chart).
    ax1 = axes[0, 0]
    ax1.bar(range(len(df)), df['total_tokens'], color='steelblue', edgecolor='black')
    ax1.axhline(df['total_tokens'].mean(), color='red', linestyle='--',
                label=f'Mean: {df["total_tokens"].mean():.0f}')
    ax1.set_xlabel('Segment Index')
    ax1.set_ylabel('Tokens')
    ax1.set_title('Tokens per Segment', fontweight='bold')
    ax1.legend()

    # 2. Distribution of tokens per segment (histogram).
    ax2 = axes[0, 1]
    ax2.hist(df['total_tokens'], bins=min(20, len(df)), edgecolor='black', color='lightcoral')
    ax2.axvline(df['total_tokens'].mean(), color='red', linestyle='--',
                label=f'Mean: {df["total_tokens"].mean():.0f}')
    ax2.axvline(df['total_tokens'].median(), color='green', linestyle='--',
                label=f'Median: {df["total_tokens"].median():.0f}')
    ax2.set_xlabel('Tokens')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Token Distribution', fontweight='bold')
    ax2.legend()

    # 3. Tokens vs total processing time (success/failure colour-coded).
    ax3 = axes[1, 0]
    point_colors = ['green' if s else 'red' for s in df['success']]
    ax3.scatter(df['total_tokens'], df['add_memory_duration_ms'], c=point_colors, alpha=0.7, s=60)

    if len(df) > 1:
        coeffs = np.polyfit(df['total_tokens'], df['add_memory_duration_ms'], 1)
        poly = np.poly1d(coeffs)
        x_line = np.linspace(df['total_tokens'].min(), df['total_tokens'].max(), 100)
        ax3.plot(x_line, poly(x_line), "r--", alpha=0.8, label=f'~{coeffs[0]:.1f}ms/token')
        ax3.legend()

    ax3.set_xlabel('Tokens')
    ax3.set_ylabel('Processing Time (ms)')
    ax3.set_title('Tokens vs Total Processing Time', fontweight='bold')

    # 4. Cumulative tokens across segments.
    ax4 = axes[1, 1]
    cum = df['total_tokens'].cumsum()
    ax4.fill_between(range(len(df)), cum, alpha=0.3, color='steelblue')
    ax4.plot(range(len(df)), cum, marker='o', linewidth=2, markersize=4, color='steelblue')
    ax4.set_xlabel('Segment Index')
    ax4.set_ylabel('Cumulative Tokens')
    ax4.set_title(f'Cumulative Tokens (Total: {cum.iloc[-1]:,})', fontweight='bold')

    suptitle = f'Token Analysis (Question {question_id})' if question_id else 'Token Analysis (All Questions)'
    plt.suptitle(suptitle, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def print_profiling_summary(segment_viz: List[Dict], sub_op_df: pd.DataFrame, question_id: Optional[str] = None):
    """
    Emit a compact text summary of the profiling artefacts.
    """
    chosen_segments = [s for s in segment_viz if s['question_id'] == question_id] if question_id else segment_viz
    df = sub_op_df[sub_op_df['question_id'] == question_id] if question_id and sub_op_df is not None else sub_op_df

    print(f"\n{'='*70}")
    print(f"PROFILING SUMMARY" + (f" - Question {question_id}" if question_id else " - All Questions"))
    print(f"{'='*70}")

    # Aggregated segment-level numbers.
    if chosen_segments:
        token_total = sum(s['total_tokens'] for s in chosen_segments)
        msg_total = sum(s['num_messages'] for s in chosen_segments)
        success_pct = sum(1 for s in chosen_segments if s['success']) / len(chosen_segments) * 100

        print(f"\n SEGMENT STATS:")
        print(f"   Segments: {len(chosen_segments)}")
        print(f"   Total tokens: {token_total:,}")
        print(f"   Total messages: {msg_total:,}")
        print(f"   Avg tokens/segment: {token_total/len(chosen_segments):.0f}")
        print(f"   Success rate: {success_pct:.1f}%")

    # Sub-operation time breakdown.
    if df is not None and len(df) > 0:
        total_time = df['total_ms'].sum()

        sub_ops = [
            ('generate_memory_entries_ms', 'LLM Memory Extraction'),
            ('generate_episodic_memory_ms', 'LLM Episodic Memory'),
            ('query_candidates_total_ms', 'DB Query Candidates'),
            ('decide_update_total_ms', 'LLM Update Decision'),
            ('chromadb_add_total_ms', 'DB ChromaDB Add'),
            ('check_existing_total_ms', 'DB Check Existing'),
        ]

        print(f"\n TIME BREAKDOWN (Total: {total_time/1000:.1f}s):")
        for col, name in sub_ops:
            if col in df.columns:
                val = df[col].sum()
                pct = val / total_time * 100 if total_time > 0 else 0
                bar = '█' * int(pct / 5) + '░' * (20 - int(pct / 5))
                print(f"   {name:25} {bar} {pct:5.1f}% ({val/1000:.1f}s)")

        # Memory-entry counts.
        entries_total = df['num_entries_generated'].sum()
        print(f"\n MEMORY ENTRIES:")
        print(f"   Total generated: {entries_total}")
        print(f"   Avg per segment: {entries_total/len(df):.1f}")

        # High-level performance numbers.
        print(f"\n PERFORMANCE:")
        print(f"   Avg time per segment: {total_time/len(df)/1000:.2f}s")
        print(f"   Tokens processed/sec: {df['total_tokens'].sum()/(total_time/1000):.0f}")

    print(f"{'='*70}\n")


def print_segment_details(segment_viz: List[Dict], question_id: str, max_segments: int = 10):
    """Pretty-print per-segment details for a single question."""
    chosen = sorted(
        [s for s in segment_viz if s['question_id'] == question_id],
        key=lambda x: x['session_idx'],
    )

    print(f"\n{'='*70}")
    print(f"SEGMENT DETAILS - Question {question_id} ({len(chosen)} segments)")
    print(f"{'='*70}")

    for seg in chosen[:max_segments]:
        marker = '✓' if seg['success'] else '✗'
        print(f"\n{marker} Session {seg['session_idx']}: {seg['timestamp']}")
        print(f"   Messages: {seg['num_messages']} (U:{seg['user_messages']}, A:{seg['assistant_messages']})")
        print(f"   Tokens: {seg['total_tokens']} (avg {seg['avg_tokens_per_message']:.0f}/msg)")
        print(f"   Duration: {seg['add_memory_duration_ms']:.0f}ms")

        if seg.get('sub_operations'):
            ops = seg['sub_operations']
            print(f"   Breakdown: LLM={ops['generate_memory_entries_ms']:.0f}ms, "
                  f"Upsert={ops['upsert_entries_total_ms']:.0f}ms "
                  f"({ops['num_entries_generated']} entries)")

    if len(chosen) > max_segments:
        print(f"\n... and {len(chosen) - max_segments} more segments")

    # Compact totals at the bottom.
    token_total = sum(s['total_tokens'] for s in chosen)
    time_total = sum(s['add_memory_duration_ms'] for s in chosen)
    print(f"\n{'─'*70}")
    print(f"TOTAL: {token_total:,} tokens, {time_total/1000:.1f}s processing time")
    print(f"{'='*70}\n")
