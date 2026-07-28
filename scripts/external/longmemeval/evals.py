import concurrent.futures
import json
import logging
import os
import threading
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Any, Optional

import hydra
from omegaconf import DictConfig

from agent_memory.utils.llm import get_general_chat_completion_client
from agent_memory.utils.log import configure_logging
from metrics.llm_judge import evaluate_llm_judge
from metrics.utils import calculate_bleu_scores, calculate_metrics
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Per-question-type judging prompts (originally adapted from Nemori/Zep).
TEMPORAL_REASONING_PROMPT = """
I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct.

<QUESTION>
B: {question}
</QUESTION>
<CORRECT ANSWER>
{gold_answer}
</CORRECT ANSWER>
<RESPONSE>
A: {response}
</RESPONSE>

Please answer 'yes' or 'no':"""

KNOWLEDGE_UPDATE_PROMPT = """
I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.

<QUESTION>
B: {question}
</QUESTION>
<CORRECT ANSWER>
{gold_answer}
</CORRECT ANSWER>
<RESPONSE>
A: {response}
</RESPONSE>

Please answer 'yes' or 'no':"""

SINGLE_SESSION_PREFERENCE_PROMPT = """
I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.

<QUESTION>
B: {question}
</QUESTION>
<RUBRIC>
{gold_answer}
</RUBRIC>
<RESPONSE>
A: {response}
</RESPONSE>

Please answer 'yes' or 'no':"""

DEFAULT_PROMPT = """
I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no.

<QUESTION>
B: {question}
</QUESTION>
<CORRECT ANSWER>
{gold_answer}
</CORRECT ANSWER>
<RESPONSE>
A: {response}
</RESPONSE>

Please answer 'yes' or 'no':"""


# Mapping from question_type -> prompt template (replaces if/elif chain).
_PROMPT_BY_QTYPE = {
    'temporal-reasoning': TEMPORAL_REASONING_PROMPT,
    'knowledge-update': KNOWLEDGE_UPDATE_PROMPT,
    'single-session-preference': SINGLE_SESSION_PREFERENCE_PROMPT,
}


def evaluate_with_question_type_prompt(model_client, question: str, gold_answer: str,
                                       response: str, question_type: str) -> bool:
    """
    Score a model response using a prompt that is tailored to the question type
    (Nemori/Zep style grading).

    Args:
        model_client: OpenAI-compatible chat client.
        question: The question text.
        gold_answer: Reference (gold) answer.
        response: Candidate model response to grade.
        question_type: One of the recognised question types
            (temporal-reasoning / knowledge-update / single-session-preference / default).

    Returns:
        True iff the judge model considers the response correct.
    """
    sys_prompt = "You are an expert grader that determines if answers to questions match a gold standard answer"

    # Pick the right template; fall back to the default rubric.
    tmpl = _PROMPT_BY_QTYPE.get(question_type, DEFAULT_PROMPT)
    user_prompt = tmpl.format(
        question=question, gold_answer=gold_answer, response=response
    )

    try:
        completion = model_client.chat.completions.create(
            model="YOUR_JUDGE_MODEL",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=10,  # Only "yes"/"no" is expected.
        )

        verdict = completion.choices[0].message.content.strip().lower()
        return 'yes' in verdict

    except Exception as exc:
        logger.error(f"Error in question type evaluation: {exc}")
        return False


def process_item(model_client, item_data, use_question_type_eval: bool = False):
    """
    Score every example associated with a single dictionary key.

    Args:
        model_client: OpenAI-compatible client used by the judge.
        item_data: A ``(key, items)`` pair where ``items`` is a list of records.
        use_question_type_eval: When True, use the LongMemEval question-type-
            specific judge prompts; otherwise fall back to the standard judge.
    """
    bucket_key, records = item_data
    bucket = defaultdict(list)

    for record in records:
        gold = str(record["answer"])
        prediction = str(record["response"])
        q_text = str(record["question"])

        # In LongMemEval each record carries a ``question_type``.
        q_type = record.get("question_type", "")
        q_id = record.get("question_id", "")

        # Lexical / surface metrics.
        metric_dict = calculate_metrics(prediction, gold)
        bleu_dict = calculate_bleu_scores(prediction, gold)

        if use_question_type_eval:
            print(100 * "*")
            print("Evaluating with question specific prompt:")
            print(100 * '*')
            ok = evaluate_with_question_type_prompt(
                model_client, q_text, gold, prediction, q_type
            )
            llm_score = 1.0 if ok else 0.0
        else:
            # Default: generic LLM-judge accuracy scoring.
            llm_score = evaluate_llm_judge(model_client, q_text, gold, prediction)

        bucket[bucket_key].append(
            {
                "question_id": q_id,
                "question": q_text,
                "answer": gold,
                "response": prediction,
                "question_type": q_type,
                "bleu_score": bleu_dict["bleu1"],
                "f1_score": metric_dict["f1"],
                "llm_score": llm_score,
            }
        )

    return bucket


def evaluate(cfg: DictConfig, input_file: str, output_file: str,
             use_question_type_eval: bool = False):
    """
    Run the LLM-judge evaluation over a results file using a thread pool and
    persist the per-record scores.

    Args:
        cfg: Hydra configuration node.
        input_file: Path to the JSON file produced by the search step.
        output_file: Where the scored evaluation records should be written.
        use_question_type_eval: Whether to use LongMemEval question-type
            specific judge prompts.
    """

    worker_count = cfg.eval.max_workers

    with open(input_file, "r") as f:
        payload = json.load(f)

    judge_client = get_general_chat_completion_client(cfg)
    aggregated = defaultdict(list)
    write_lock = threading.Lock()

    # Fan out scoring across worker_count threads.
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as pool:
        pending = [
            pool.submit(process_item, judge_client, pair, use_question_type_eval)
            for pair in payload.items()
        ]

        for fut in tqdm(
            concurrent.futures.as_completed(pending), total=len(pending),
            desc="Evaluating responses"
        ):
            partial = fut.result()
            with write_lock:
                for k, items in partial.items():
                    aggregated[k].extend(items)

    # Persist scored output.
    with open(output_file, "w") as f:
        json.dump(aggregated, f, indent=4)

    logger.info(f"Results saved to {output_file}")


def generate_scores(metrics_file: str, score_file: str):
    """
    Aggregate the per-record metrics into mean scores grouped by question type
    and write a summary to ``score_file``.

    Args:
        metrics_file: Path to the per-record metrics JSON.
        score_file: Path of the summary file to produce.
    """
    # Load the scored records.
    with open(metrics_file, "r") as f:
        records = json.load(f)

    # Flatten all per-key lists into a single list.
    flat = []
    for key in records:
        flat.extend(records[key])

    df = pd.DataFrame(flat)

    # LongMemEval groups by ``question_type``.
    group_col = "question_type"

    # Mean BLEU/F1/LLM scores per question type.
    per_type = (
        df.groupby(group_col)
        .agg({"bleu_score": "mean", "f1_score": "mean", "llm_score": "mean"})
        .round(4)
    )

    # Number of records per question type.
    per_type["count"] = df.groupby(group_col).size()

    # Pretty-print the per-type summary.
    logger.info(f"\n{'='*60}")
    logger.info(f"Mean Scores Per Question Type:")
    logger.info(f"{'='*60}")
    for qtype in per_type.index:
        row = per_type.loc[qtype]
        logger.info(f"{qtype:30s} | Count: {row['count']:4.0f} | "
                    f"BLEU: {row['bleu_score']:.4f} | "
                    f"F1: {row['f1_score']:.4f} | "
                    f"LLM: {row['llm_score']:.4f}")

    # Means computed across the entire dataset.
    overall = df.agg(
        {"bleu_score": "mean", "f1_score": "mean", "llm_score": "mean"}
    ).round(4)

    logger.info(f"\n{'='*60}")
    logger.info(f"Overall Mean Scores:")
    logger.info(f"{'='*60}")
    logger.info(f"BLEU Score:  {overall['bleu_score']:.4f}")
    logger.info(f"F1 Score:    {overall['f1_score']:.4f}")
    logger.info(f"LLM Score:   {overall['llm_score']:.4f}")
    logger.info(f"Total Questions: {len(df)}")
    logger.info(f"{'='*60}\n")

    # Build the JSON payload to persist.
    scores_data = {
        "mean_scores_per_question_type": per_type.to_dict(),
        "overall_mean_scores": overall.to_dict(),
        "summary": {
            "total_questions": len(df),
            "question_types_evaluated": sorted(df[group_col].unique().tolist()),
            "evaluation_timestamp": pd.Timestamp.now().isoformat(),
        },
    }

    with open(score_file, "w") as f:
        json.dump(scores_data, f, indent=4)

    logger.info(f"Scores saved to {score_file}")

    return scores_data


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def run(cfg: DictConfig):
    """
    Hydra entry point: score a results file and emit a summary.

    Args:
        cfg: Hydra configuration object.
    """

    configure_logging()

    in_path = os.path.join(cfg.general.output_path, cfg.eval.result_file)
    out_path = os.path.join(cfg.general.output_path, cfg.eval.metrics_file)
    score_path = os.path.join(
        cfg.general.output_path, cfg.eval.get("score_file", "scores.json")
    )

    use_qt_eval = cfg.eval.get("use_question_type_eval", False)

    if use_qt_eval:
        logger.info("Using Nemori-style question type-specific evaluation prompts")
    else:
        logger.info("Using standard LLM judge evaluation")

    evaluate(cfg, in_path, out_path, use_question_type_eval=use_qt_eval)

    scores_data = generate_scores(out_path, score_path)

    return scores_data


if __name__ == "__main__":
    run()
