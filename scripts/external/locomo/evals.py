import concurrent.futures
import json
import logging
import os
import threading
import pandas as pd
from collections import defaultdict

import hydra
from omegaconf import DictConfig

from ultramem.utils.llm import get_general_chat_completion_client
from ultramem.utils.log import configure_logging
from metrics.llm_judge import evaluate_llm_judge
from metrics.utils import calculate_bleu_scores, calculate_metrics
from tqdm import tqdm

logger = logging.getLogger(__name__)


def process_item(model_client, judge_model, item_data):
    conv_key, qa_items = item_data
    bucket = defaultdict(list)

    for entry in qa_items:
        gt_answer = str(entry["answer"])
        pred_answer = str(entry["response"])
        category = str(entry["category"])
        question = str(entry["question"])

        # Capture the formatted memory snippets when present
        formatted_speaker_1_memories = entry.get("formatted_speaker_1_memories", [])
        formatted_speaker_2_memories = entry.get("formatted_speaker_2_memories", [])

        # Category 5 is excluded from evaluation
        if category == "5":
            continue

        score_metrics = calculate_metrics(pred_answer, gt_answer)
        bleu_scores = calculate_bleu_scores(pred_answer, gt_answer)
        llm_score = evaluate_llm_judge(model_client, question, gt_answer, pred_answer, model=judge_model)

        record = {
            "question": question,
            "answer": gt_answer,
            "response": pred_answer,
            "category": category,
            "bleu_score": bleu_scores["bleu1"],
            "f1_score": score_metrics["f1"],
            "llm_score": llm_score,
            "formatted_speaker_1_memories": formatted_speaker_1_memories,
            "formatted_speaker_2_memories": formatted_speaker_2_memories,
        }
        # Forward retrieval recall fields when they were attached upstream
        if "session_recall" in entry:
            record["session_recall"] = entry["session_recall"]
        if "text_recall" in entry:
            record["text_recall"] = entry["text_recall"]

        bucket[conv_key].append(record)

    return bucket


def evaluate(cfg: DictConfig, input_file: str, output_file: str):
    """Run threaded per-conversation evaluation and persist the results."""

    max_workers = cfg.eval.max_workers

    with open(input_file, "r") as fh:
        data = json.load(fh)

    model_client = get_general_chat_completion_client(cfg)
    judge_model = cfg.eval.get("judge_model", None) or cfg.eval.model

    logger.info(f"Judge model: {judge_model}")

    results = defaultdict(list)
    results_lock = threading.Lock()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(process_item, model_client, judge_model, pair)
            for pair in data.items()
        ]

        for fut in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            local_results = fut.result()
            with results_lock:
                for conv_key, recs in local_results.items():
                    results[conv_key].extend(recs)

    with open(output_file, "w") as fh:
        json.dump(results, fh, indent=4)

    logger.info(f"Results saved to {output_file}")


def generate_scores(metrics_file: str, score_file: str):
    with open(metrics_file, "r") as fh:
        data = json.load(fh)

    # Flatten the per-conversation lists into one big list
    flat_records = []
    for conv_key in data:
        flat_records.extend(data[conv_key])

    df = pd.DataFrame(flat_records)

    df["category"] = pd.to_numeric(df["category"])

    agg_cols = {"bleu_score": "mean", "f1_score": "mean", "llm_score": "mean"}
    has_recall = "session_recall" in df.columns and "text_recall" in df.columns
    if has_recall:
        agg_cols["session_recall"] = "mean"
        agg_cols["text_recall"] = "mean"

    per_category = (
        df.groupby("category")
        .agg(agg_cols)
        .round(4)
    )

    per_category["count"] = df.groupby("category").size()

    logger.info(f"\nMean Scores Per Category:\n{per_category}\n")

    overall_agg_cols = {"bleu_score": "mean", "f1_score": "mean", "llm_score": "mean"}
    if has_recall:
        overall_agg_cols["session_recall"] = "mean"
        overall_agg_cols["text_recall"] = "mean"
    overall_means = df.agg(overall_agg_cols).round(4)

    logger.info(f"\nOverall Mean Scores:\n{overall_means}\n")

    scores_data = {
        "mean_scores_per_category": per_category.to_dict(),
        "overall_mean_scores": overall_means.to_dict(),
        "summary": {
            "total_questions": len(df),
            "categories_evaluated": sorted(df["category"].unique().tolist()),
            "evaluation_timestamp": pd.Timestamp.now().isoformat(),
        },
    }

    with open(score_file, "w") as fh:
        json.dump(scores_data, fh, indent=4)

    logger.info(f"\nScores saved to {score_file}")

    return scores_data


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def run(cfg: DictConfig):

    configure_logging()

    input_file = os.path.join(cfg.general.output_path, cfg.eval.result_file)
    output_file = os.path.join(cfg.general.output_path, cfg.eval.metrics_file)
    score_file = os.path.join(
        cfg.general.output_path, cfg.eval.get("score_file", "scores.json")
    )

    evaluate(cfg, input_file, output_file)

    scores_data = generate_scores(output_file, score_file)

    return scores_data


if __name__ == "__main__":
    run()
