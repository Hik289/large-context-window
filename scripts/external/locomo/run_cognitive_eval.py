"""
Cognitive Evaluation Runner — aligned with the LoCoMo-Plus paper (arXiv:2602.10715).

End-to-end pipeline for evaluating the memory system on cognitive items:
  1. Load cognitive items from ``unified_input_samples_v2.json`` (cached after first load).
  2. Build memory from locomo10 conversations augmented with the cognitive
     evidence dialogue (the v2 dataset injects new turns absent from locomo10).
  3. Retrieve memories using each trigger as the query.
  4. Generate a conversational reply to the trigger using the retrieved memories.
  5. Have an LLM judge label whether the reply demonstrates evidence awareness
     (binary: correct/wrong).
  6. Aggregate scores.

Usage (mirrors the override syntax of ``run_experiments.py``)::

  PYTHONPATH=/path/to/ultramem/src python3 run_cognitive_eval.py \\
      retrieval.strategy=semantic \\
      llm.model=YOUR_CHAT_MODEL \\
      general.project_path=/path/to/ultramem/app/locomo
"""

from __future__ import annotations

import os

# Load the .env file before any API client is instantiated.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    def _load_dotenv_fallback(dotenv_path: str) -> None:
        try:
            with open(dotenv_path, "r", encoding="utf-8") as fh:
                for raw_line in fh:
                    line = raw_line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" not in line:
                        continue
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    os.environ.setdefault(key, val)
        except FileNotFoundError:
            return

    _load_dotenv_fallback(os.path.join(os.path.dirname(__file__), ".env"))

import json
import logging
import time
from datetime import datetime
from typing import List

import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from cognitive_data_adapter import (
    CognitiveItem,
    build_augmented_locomo10,
    get_user_ids,
    load_cognitive_data,
)
from cognitive_evals import evaluate_cognitive, generate_cognitive_scores
from metrics.retrieval_recall import compute_recall_for_cognitive
from providers.ultramem.add import UltraMemAdd
from providers.ultramem.search import UltraMemSearch
from ultramem.utils.latency import count_memories_tokens
from ultramem.utils.log import configure_logging

logger = logging.getLogger(__name__)


def _format_memories_simple(memories: list) -> list:
    """Render raw memory dicts as readable strings for evaluation.

    Memories whose post-filter score equals ``3`` (a strong cognitive link)
    are tagged with ``[KEY]`` so the response LLM gives them extra weight.
    """
    rendered: list = []
    for entry in memories:
        body = entry.get("value", "")
        ts = entry.get("timestamp", "")
        if body.startswith("[") and "]\n" in body:
            line = body
        elif ts:
            line = f"{ts}: {body}"
        else:
            line = body

        if entry.get("score") == 3.0:
            line = f"[KEY] {line}"

        rendered.append(line)
    return rendered


# ------------------------------------------------------------------
# Pipeline phases
# ------------------------------------------------------------------

def search_phase(
    cfg: DictConfig,
    items: List[CognitiveItem],
    output_file: str,
    retrieval_strategy: str,
) -> List[dict]:
    """Run retrieval for every cognitive trigger and persist the search results."""
    searcher = UltraMemSearch(
        cfg, output_path=output_file,
        top_k=cfg.memory.top_k,
        retrieval_strategy=retrieval_strategy,
    )
    use_combined = cfg.eval.get("use_combined_user", True)

    results: List[dict] = []
    for entry in tqdm(items, desc="Retrieving memories for cognitive triggers"):
        primary_id, secondary_id = get_user_ids(entry, use_combined)

        if use_combined:
            memories, search_time = searcher.search_memory(primary_id, entry.trigger)
            formatted = _format_memories_simple(memories)
        else:
            half_top_k = cfg.memory.top_k // 2
            mem_a, t_a = searcher.search_memory(
                primary_id, entry.trigger, top_k=half_top_k,
            )
            mem_b, t_b = searcher.search_memory(
                secondary_id, entry.trigger, top_k=half_top_k,
            )
            memories = mem_a + mem_b
            search_time = t_a + t_b
            formatted = (
                _format_memories_simple(mem_a) + _format_memories_simple(mem_b)
            )

        token_info = count_memories_tokens(formatted)
        recall_info = compute_recall_for_cognitive(
            entry.evidence, entry.evidence_after_session, memories, entry.conv_idx,
        )

        record = {
            **entry.to_dict(),
            "retrieved_memories": memories,
            "formatted_memories": formatted,
            "num_retrieved_memories": len(memories),
            "search_time": round(search_time, 4),
            "memory_total_tokens": token_info["total_tokens"],
            "memory_avg_tokens_per_item": round(token_info["avg_tokens"], 2),
            "session_recall": recall_info["session_recall"],
            "text_recall": recall_info["text_recall"],
        }
        results.append(record)

        # Incremental save
        with open(output_file, "w") as fh:
            json.dump(results, fh, indent=4)

    logger.info(f"Search results ({len(results)} items) saved to {output_file}")
    return results


# ------------------------------------------------------------------
# Main experiment driver
# ------------------------------------------------------------------

def _build_memory_if_needed(
    cfg: DictConfig,
    locomo10_path: str,
    items: List[CognitiveItem],
    output_dir: str,
):
    """Build memory from locomo10 conversations augmented with the cognitive evidence.

    The cognitive dataset injects fresh dialogue into the original locomo10
    conversations. Here we merge every evidence line back into the locomo10
    data, write an augmented JSON file, and feed it through the standard
    ``UltraMemAdd`` pipeline so the resulting memory store contains both
    the original and the injected dialogue.

    Skips when ``persist_path`` already exists (unless
    ``memory.force_rebuild`` is set).
    """
    should_build = (
        not os.path.exists(cfg.memory.persist_path) or cfg.memory.force_rebuild
    )
    if not should_build:
        logger.info(f"Memory store exists at {cfg.memory.persist_path} — skipping build")
        return

    augmented_path = os.path.join(output_dir, "augmented_locomo10.json")
    build_augmented_locomo10(locomo10_path, items, augmented_path)

    logger.info(f"Building memory in {cfg.memory.persist_path} from augmented data …")
    build_start = time.time()

    memory_manager = UltraMemAdd(cfg, data_path=augmented_path)
    memory_manager.process_all_conversations()

    build_duration = time.time() - build_start
    duration_str = time.strftime("%H:%M:%S", time.gmtime(build_duration))
    logger.info(f"Memory building completed in {duration_str}")

    timing_data = {
        "start_time": datetime.fromtimestamp(build_start).strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "duration": duration_str,
        "memory_store": cfg.memory.memory_store,
        "data_source": "augmented_locomo10 (locomo10 + cognitive evidence)",
    }
    timing_file = os.path.join(output_dir, "build_timing.json")
    with open(timing_file, "w") as fh:
        json.dump(timing_data, fh, indent=2)


def run_cognitive_experiment(cfg: DictConfig, retrieval_strategy: str = "semantic"):
    """Drive the full cognitive evaluation pipeline."""
    configure_logging(cfg.general.get("log_level", "INFO"))
    logger.info("\n===== Cognitive Evaluation =====")
    logger.info(f"Retrieval strategy: {retrieval_strategy}")
    logger.info(f"Memory store: {cfg.memory.persist_path}")

    data_dir = cfg.general.data_path
    v2_path = os.path.join(data_dir, "unified_input_samples_v2.json")
    locomo10_path = os.path.join(data_dir, "locomo10.json")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        cfg.general.results_path, f"cognitive_{retrieval_strategy}_{timestamp}",
    )
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    params_file = os.path.join(output_dir, "parameters.yaml")
    with open(params_file, "w") as fh:
        fh.write(OmegaConf.to_yaml(cfg))

    # 1. Load cognitive data (needed before building the augmented memory store)
    logger.info("\n==== Loading cognitive data ====")
    items = load_cognitive_data(
        v2_path,
        locomo10_path,
        cache_dir=data_dir,
    )
    logger.info(f"Loaded {len(items)} cognitive items")

    # 0. Build memory from augmented conversations (locomo10 + evidence)
    _build_memory_if_needed(cfg, locomo10_path, items, output_dir)

    # 2. Search phase
    logger.info("\n==== Search phase ====")
    search_output = os.path.join(output_dir, "cognitive_search_output.json")
    search_start = time.time()
    search_results = search_phase(cfg, items, search_output, retrieval_strategy)
    search_duration = time.time() - search_start
    logger.info(f"Search phase completed in {search_duration:.1f}s")

    # 3. Evaluation phase
    logger.info("\n==== Evaluation phase ====")
    eval_output = os.path.join(output_dir, "cognitive_eval.json")
    evaluate_cognitive(cfg, search_output, eval_output)

    # 4. Scoring phase
    logger.info("\n==== Scoring phase ====")
    score_output = os.path.join(output_dir, "cognitive_scores.json")
    scores = generate_cognitive_scores(eval_output, score_output)

    # 5. Persist a timing summary
    timing_file = os.path.join(output_dir, "timing.json")
    perf = scores.get("overall", {}).get("performance", {}) if scores else {}
    avg_search = sum(r["search_time"] for r in search_results) / max(len(search_results), 1)
    with open(timing_file, "w") as fh:
        json.dump({
            "search_duration_s": round(search_duration, 2),
            "total_items": len(items),
            "avg_search_time_s": round(avg_search, 4),
            "avg_retrieval_latency_s": perf.get("avg_retrieval_latency_s"),
            "avg_response_gen_latency_s": perf.get("avg_response_gen_latency_s"),
            "avg_retrieval_plus_response_gen_latency_s": perf.get(
                "avg_retrieval_plus_response_gen_latency_s"
            ),
            "avg_judge_latency_s": perf.get("avg_judge_latency_s"),
            "avg_e2e_latency_s": perf.get("avg_e2e_latency_s"),
        }, fh, indent=2)

    logger.info(f"\n===== Cognitive evaluation complete =====")
    logger.info(f"Results in: {output_dir}")
    return scores


# ------------------------------------------------------------------
# Hydra entry point — accepts the same CLI overrides as run_experiments.py
# ------------------------------------------------------------------

@hydra.main(version_base=None, config_path="./conf", config_name="config")
def run(cfg: DictConfig):
    run_cognitive_experiment(
        cfg,
        retrieval_strategy=cfg.retrieval.strategy,
    )


if __name__ == "__main__":
    run()
