"""
Evaluation logic for the Cognitive category.

Pipeline:
  1. Run the search phase to retrieve memories using the trigger as the query.
  2. Build a conversational *response* to the trigger using the retrieved memories.
  3. Use an LLM judge to label whether the response shows awareness of the
     evidence (binary: correct = 1, wrong = 0).

Judge prompt from the paper:
  "Judge whether the Model Prediction considers or is linked to the Evidence."
"""

import concurrent.futures
import json
import logging
import re
import threading
import time
from collections import defaultdict

from tqdm import tqdm
from openai import BadRequestError

from ultramem.utils.latency import count_memories_tokens
from ultramem.utils.llm import get_general_chat_completion_client
from metrics.utils import extract_json
from prompts import COGNITIVE_RESPONSE_PROMPT, COGNITIVE_JUDGE_PROMPT

logger = logging.getLogger(__name__)

LABEL_TO_SCORE = {"correct": 1.0, "partial": 0.5, "wrong": 0.0}


def _is_content_filter_error(exc: Exception) -> bool:
    """Return ``True`` when the configured API rejects the request by policy."""
    if not isinstance(exc, BadRequestError):
        return False
    msg = str(exc)
    if "content_filter" in msg or "content management policy" in msg:
        return True
    if "ResponsibleAIPolicyViolation" in msg:
        return True
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error") or {}
        if err.get("code") == "content_filter":
            return True
        inner = err.get("innererror") or {}
        if inner.get("code") == "ResponsibleAIPolicyViolation":
            return True
    return False


def _parse_judge_label(raw: str) -> tuple:
    """Pull ``label`` and ``reason`` out of the judge JSON output."""
    raw = (raw or "").strip()
    label, reason = "", ""
    try:
        m = re.search(
            r'\{[^{}]*"label"\s*:\s*["\']([^"\']+)["\'][^{}]*"reason"\s*:\s*["\']([^"\']*)["\']',
            raw, re.DOTALL,
        )
        if m:
            label, reason = m.group(1).strip().lower(), (m.group(2) or "").strip()
        else:
            obj = json.loads(raw)
            label = (obj.get("label") or "").strip().lower()
            reason = (obj.get("reason") or "").strip()
    except Exception:
        lower = raw.lower()
        if "correct" in lower:
            label = "correct"
        elif "wrong" in lower:
            label = "wrong"
        elif "partial" in lower:
            label = "partial"
        reason = raw[:200] if raw else ""
    return label, reason


# ---------------------------------------------------------------------------
# Step 1: build a conversational response to the trigger
# ---------------------------------------------------------------------------

def _safe_content(response) -> str:
    """Return a stripped ``choices[0].message.content`` or ``""`` if missing."""
    content = response.choices[0].message.content
    return content.strip() if content else ""


_MAX_TRANSIENT_RETRIES = 3
_TRANSIENT_BACKOFF = [1, 2, 4]


def generate_response(
    model_client, trigger: str, time_gap: str, memories: list,
    model: str = "YOUR_CHAT_MODEL",
    reasoning_effort: str = None,
) -> str:
    """Use the LLM to produce a conversational response from retrieved memories."""
    if memories:
        memories_text = "\n".join(f"- {m}" for m in memories)
    else:
        memories_text = "(No memories retrieved from past conversations.)"

    prompt = COGNITIVE_RESPONSE_PROMPT.format(
        trigger=trigger, time_gap=time_gap, memories=memories_text,
    )
    call_kwargs: dict = dict(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    if reasoning_effort:
        call_kwargs["reasoning_effort"] = reasoning_effort
    else:
        call_kwargs["temperature"] = 0.3
        call_kwargs["seed"] = 42
    try:
        response = model_client.chat.completions.create(**call_kwargs)
        return _safe_content(response)
    except BadRequestError as exc:
        if _is_content_filter_error(exc):
            logger.warning("Response generation skipped by content filter")
            return "[Skipped: content filter blocked this prompt.]"
        if "temperature" in str(exc) or "unsupported_value" in str(exc):
            logger.info(f"Model {model} rejected temperature/seed; retrying with defaults")
            call_kwargs.pop("temperature", None)
            call_kwargs.pop("seed", None)
            try:
                response = model_client.chat.completions.create(**call_kwargs)
                return _safe_content(response)
            except Exception as exc2:
                logger.error(f"Response generation error (retry): {exc2}")
                return f"ERROR: {exc2}"
        logger.error(f"Response generation error: {exc}")
        return f"ERROR: {exc}"
    except Exception as exc:
        logger.error(f"Response generation error: {exc}")
        return f"ERROR: {exc}"


# ---------------------------------------------------------------------------
# Step 2: ask the judge whether the response demonstrates evidence awareness
# ---------------------------------------------------------------------------

def evaluate_cognitive_judge(
    model_client, evidence: str, prediction: str,
    model: str = "YOUR_JUDGE_MODEL",
) -> dict:
    """Decide whether the response considers/links to the provided evidence.

    Returns a dict with ``label``, ``score`` and ``reason`` keys.
    Scoring matches the paper: correct=1, wrong=0.
    """
    prompt = COGNITIVE_JUDGE_PROMPT.format(
        evidence=evidence, pred=prediction,
    )
    call_kwargs: dict = dict(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.0,
        seed=42,
    )

    last_exc = None
    for attempt in range(_MAX_TRANSIENT_RETRIES):
        try:
            response = model_client.chat.completions.create(**call_kwargs)
            raw = response.choices[0].message.content
            if not raw:
                logger.warning("Judge returned empty content; treating as wrong")
                return {"label": "wrong", "score": 0.0, "reason": "empty response"}
            label, reason = _parse_judge_label(raw)
            score = LABEL_TO_SCORE.get(label, 0.0)
            return {"label": label, "score": score, "reason": reason}
        except BadRequestError as exc:
            if _is_content_filter_error(exc):
                logger.warning("Cognitive judge skipped by content filter")
                return {"label": "skipped", "score": None, "reason": "content filter"}
            if "temperature" in str(exc) or "unsupported_value" in str(exc):
                logger.info(f"Model {model} rejected temperature/seed; retrying with defaults")
                call_kwargs.pop("temperature", None)
                call_kwargs.pop("seed", None)
                continue
            logger.error(f"Cognitive judge error: {exc}")
            raise
        except Exception as exc:
            last_exc = exc
            if attempt < _MAX_TRANSIENT_RETRIES - 1:
                wait = _TRANSIENT_BACKOFF[attempt]
                logger.warning(
                    f"Cognitive judge transient error (attempt {attempt+1}/"
                    f"{_MAX_TRANSIENT_RETRIES}): {exc}. Retrying in {wait}s"
                )
                time.sleep(wait)
            else:
                logger.error(
                    f"Cognitive judge failed after {_MAX_TRANSIENT_RETRIES} attempts: {exc}"
                )

    return {"label": "error", "score": 0.0, "reason": f"Failed after retries: {last_exc}"}


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def _process_single_item(model_client, response_model, judge_model, item,
                         reasoning_effort=None, max_memories=None):
    """Run the two-step evaluation against a single search-output item."""
    trigger = item["trigger"]
    evidence = item["evidence"]
    time_gap = item.get("time_gap", "")
    mem_list = item.get("formatted_memories", [])
    if max_memories and len(mem_list) > max_memories:
        mem_list = mem_list[:max_memories]

    prompt_token_info = count_memories_tokens(mem_list)
    item["memory_prompt_tokens"] = prompt_token_info["total_tokens"]
    item["num_memories_in_prompt"] = len(mem_list)

    # Step 1 — generate response
    t0 = time.perf_counter()
    response_text = generate_response(
        model_client, trigger, time_gap, mem_list,
        model=response_model,
        reasoning_effort=reasoning_effort,
    )
    response_gen_time = time.perf_counter() - t0

    # Step 2 — judge
    t1 = time.perf_counter()
    judge_outcome = evaluate_cognitive_judge(
        model_client, evidence, response_text, model=judge_model,
    )
    judge_time = time.perf_counter() - t1

    eval_total_time = response_gen_time + judge_time
    search_time = item.get("search_time", 0)

    item["response"] = response_text
    item["judge_label"] = judge_outcome["label"]
    item["judge_score"] = judge_outcome["score"]
    item["judge_reason"] = judge_outcome["reason"]
    item["response_gen_time"] = round(response_gen_time, 4)
    item["judge_time"] = round(judge_time, 4)
    item["eval_total_time"] = round(eval_total_time, 4)
    item["e2e_time"] = round(search_time + eval_total_time, 4)
    return item


def evaluate_cognitive(cfg, input_file: str, output_file: str):
    """Evaluate every cognitive search result.

    Args:
        cfg: Hydra config.
        input_file: Path to the search-output JSON (a list of items).
        output_file: Path to write per-item evaluation results.
    """
    max_workers = cfg.eval.max_workers
    response_model = cfg.eval.model
    judge_model = cfg.eval.get("judge_model", None) or cfg.eval.model
    reasoning_effort = cfg.eval.get("reasoning_effort", None)
    max_memories = cfg.eval.get("max_memories", None)

    logger.info(f"Response model: {response_model}")
    logger.info(f"Judge model:    {judge_model}")
    if reasoning_effort:
        logger.info(f"Reasoning effort: {reasoning_effort}")
    if max_memories:
        logger.info(f"Max memories per item: {max_memories}")

    with open(input_file, "r") as fh:
        data = json.load(fh)

    model_client = get_general_chat_completion_client(cfg)

    results: list = []
    results_lock = threading.Lock()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _process_single_item, model_client, response_model, judge_model, item,
                reasoning_effort=reasoning_effort,
                max_memories=max_memories,
            ): pos
            for pos, item in enumerate(data)
        }
        for fut in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Evaluating cognitive items",
        ):
            outcome = fut.result()
            with results_lock:
                results.append(outcome)

    results.sort(key=lambda r: (r.get("conv_idx", 0), r.get("trigger", "")))

    with open(output_file, "w") as fh:
        json.dump(results, fh, indent=4)

    logger.info(f"Cognitive eval results saved to {output_file}")
    return results


# ---------------------------------------------------------------------------
# Scoring / aggregation
# ---------------------------------------------------------------------------

_TIME_GAP_BUCKETS = [
    ("~1 week", ["one week", "a week", "1 week"]),
    ("~2 weeks", ["two week", "2 week", "couple of week", "few week"]),
    ("~3 weeks", ["three week", "3 week"]),
    ("~1 month", ["one month", "a month", "1 month"]),
    ("~6 weeks", ["six week", "6 week", "eight week"]),
    ("~2 months", ["two month", "2 month"]),
    ("~3 months", ["three month", "3 month", "roughly three"]),
    ("~4 months", ["four month", "4 month"]),
    ("~5 months", ["five month", "5 month"]),
    ("~6 months", ["six month", "6 month", "half a year"]),
    ("~7-9 months", ["seven month", "eight month", "nine month",
                      "7 month", "8 month", "9 month"]),
    ("~1 year", ["year", "12 month"]),
    ("several months", ["several month"]),
    ("several weeks", ["several week"]),
]


def _normalize_time_gap(gap: str) -> str:
    lowered = gap.lower().strip()
    for bucket, keywords in _TIME_GAP_BUCKETS:
        if any(kw in lowered for kw in keywords):
            return bucket
    return gap


def _group_accuracy(items, score_key):
    vals = [
        rec[score_key]
        for rec in items
        if score_key in rec and rec[score_key] is not None
    ]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 4)


def _avg(values):
    """Rounded mean (4 dp), or ``None`` for an empty list."""
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _compute_recall_stats(items):
    """Compute ``session_recall`` / ``text_recall`` averages for a batch.

    These fields are populated during the search phase.
    """
    sess_vals = [r["session_recall"] for r in items if "session_recall" in r]
    text_vals = [
        r["text_recall"] for r in items
        if "text_recall" in r and r["text_recall"] is not None
    ]
    return {
        "session_recall": _avg(sess_vals),
        "text_recall": _avg(text_vals),
    }


def _compute_latency_and_token_stats(items):
    """Pull latency and token statistics out of a list of eval items."""
    search_times = [r["search_time"] for r in items if "search_time" in r]
    eval_times = [r["eval_total_time"] for r in items if "eval_total_time" in r]
    e2e_times = [r["e2e_time"] for r in items if "e2e_time" in r]
    resp_gen_times = [r["response_gen_time"] for r in items if "response_gen_time" in r]
    judge_times = [r["judge_time"] for r in items if "judge_time" in r]
    total_tokens = [r["memory_total_tokens"] for r in items if "memory_total_tokens" in r]
    prompt_tokens = [
        r["memory_prompt_tokens"] for r in items if "memory_prompt_tokens" in r
    ]
    prompt_counts = [
        r["num_memories_in_prompt"] for r in items if "num_memories_in_prompt" in r
    ]
    num_memories = [r["num_retrieved_memories"] for r in items if "num_retrieved_memories" in r]

    avg_retrieval = _avg(search_times)
    avg_response_gen = _avg(resp_gen_times)
    if avg_retrieval is not None and avg_response_gen is not None:
        combined_ret_gen = round(avg_retrieval + avg_response_gen, 4)
    else:
        combined_ret_gen = None

    return {
        "avg_retrieval_latency_s": avg_retrieval,
        "avg_response_gen_latency_s": avg_response_gen,
        "avg_retrieval_plus_response_gen_latency_s": combined_ret_gen,
        "avg_judge_latency_s": _avg(judge_times),
        "avg_eval_latency_s": _avg(eval_times),
        "avg_e2e_latency_s": _avg(e2e_times),
        "avg_memory_tokens": _avg(total_tokens),
        "total_memory_tokens": sum(total_tokens) if total_tokens else 0,
        "avg_memory_prompt_tokens": _avg(prompt_tokens),
        "total_memory_prompt_tokens": sum(prompt_tokens) if prompt_tokens else 0,
        "avg_num_memories_in_prompt": _avg(prompt_counts),
        "avg_num_retrieved_memories": _avg(num_memories),
    }


def generate_cognitive_scores(eval_file: str, score_file: str):
    """Roll up per-item evaluation results into summary scores."""
    with open(eval_file, "r") as fh:
        data = json.load(fh)

    if not data:
        logger.warning("No cognitive evaluation results to score")
        return {}

    n_skip = sum(
        1 for r in data if "judge_score" in r and r["judge_score"] is None
    )

    overall_recall = _compute_recall_stats(data)

    overall = {
        "total_items": len(data),
        "cognitive_accuracy": _group_accuracy(data, "judge_score"),
        "session_recall": overall_recall["session_recall"],
        "text_recall": overall_recall["text_recall"],
        "judge_skipped_content_filter": n_skip,
    }

    overall_perf = _compute_latency_and_token_stats(data)
    overall["performance"] = overall_perf

    # Bucket by time gap.
    bucket_groups = defaultdict(list)
    for rec in data:
        bucket_groups[_normalize_time_gap(rec.get("time_gap", "unknown"))].append(rec)

    by_bucket = {}
    for bucket in sorted(bucket_groups):
        bucket_items = bucket_groups[bucket]
        bucket_recall = _compute_recall_stats(bucket_items)
        by_bucket[bucket] = {
            "count": len(bucket_items),
            "cognitive_accuracy": _group_accuracy(bucket_items, "judge_score"),
            "session_recall": bucket_recall["session_recall"],
            "text_recall": bucket_recall["text_recall"],
            "performance": _compute_latency_and_token_stats(bucket_items),
        }

    # Bucket by conversation.
    conv_groups = defaultdict(list)
    for rec in data:
        conv_key = (
            f"{rec.get('speaker_a', '?')}_{rec.get('speaker_b', '?')}_{rec.get('conv_idx', '?')}"
        )
        conv_groups[conv_key].append(rec)

    by_conv = {}
    for conv_key in sorted(conv_groups):
        conv_items = conv_groups[conv_key]
        conv_recall = _compute_recall_stats(conv_items)
        by_conv[conv_key] = {
            "count": len(conv_items),
            "cognitive_accuracy": _group_accuracy(conv_items, "judge_score"),
            "session_recall": conv_recall["session_recall"],
            "text_recall": conv_recall["text_recall"],
            "performance": _compute_latency_and_token_stats(conv_items),
        }

    scores_data = {
        "overall": overall,
        "by_time_gap_bucket": by_bucket,
        "by_conversation": by_conv,
    }

    with open(score_file, "w") as fh:
        json.dump(scores_data, fh, indent=4)

    logger.info(f"\n===== Cognitive Scores (LoCoMo-Plus protocol) =====")
    logger.info(f"  Cognitive Accuracy (memory awareness): {overall['cognitive_accuracy']}")
    logger.info(f"  Retrieval Recall (session):            {overall['session_recall']}")
    logger.info(f"  Retrieval Recall (text overlap):       {overall['text_recall']}")
    if n_skip:
        logger.info(f"  Skipped (content filter):             {n_skip}")
    logger.info(f"\n  Performance (averages across all items):")
    logger.info(f"    Retrieval latency:           {overall_perf['avg_retrieval_latency_s']}s")
    logger.info(f"    Response generation latency: {overall_perf['avg_response_gen_latency_s']}s")
    logger.info(
        f"    Retrieval + response gen:    {overall_perf['avg_retrieval_plus_response_gen_latency_s']}s"
    )
    logger.info(f"    Judge latency:               {overall_perf['avg_judge_latency_s']}s")
    logger.info(f"    End-to-end latency:          {overall_perf['avg_e2e_latency_s']}s")
    logger.info(f"    Avg memory tokens:           {overall_perf['avg_memory_tokens']}")
    logger.info(f"    Total memory tokens:         {overall_perf['total_memory_tokens']}")
    if overall_perf.get("avg_memory_prompt_tokens") is not None:
        logger.info(
            f"    Avg prompt memory tokens:    {overall_perf['avg_memory_prompt_tokens']}"
        )
        logger.info(
            f"    Total prompt memory tokens:  {overall_perf['total_memory_prompt_tokens']}"
        )
    if overall_perf.get("avg_num_memories_in_prompt") is not None:
        logger.info(
            f"    Avg memories in prompt:      {overall_perf['avg_num_memories_in_prompt']}"
        )
    logger.info(f"    Avg retrieved memories:      {overall_perf['avg_num_retrieved_memories']}")
    logger.info(f"\nBy Time Gap Bucket:")
    for bucket, s in by_bucket.items():
        logger.info(
            f"  {bucket:>20s}: accuracy={s['cognitive_accuracy']}  "
            f"sess_recall={s['session_recall']}  text_recall={s['text_recall']}  (n={s['count']})"
        )
    logger.info(f"\nScores saved to {score_file}")
    return scores_data
