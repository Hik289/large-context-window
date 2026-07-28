import argparse
import json
from collections import defaultdict

import numpy as np

from metrics.utils import extract_json

ACCURACY_PROMPT = """
Your task is to label an answer to a question as ’CORRECT’ or ’WRONG’. You will be given the following data:
    (1) a question (posed by one user to another user), 
    (2) a ’gold’ (ground truth) answer, 
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT. 

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. 
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label".
"""


def _parse_judge_label(content):
    """Pull the label out of the LLM judge response, with permissive fallbacks."""
    try:
        return json.loads(extract_json(content))["label"]
    except (json.JSONDecodeError, KeyError):
        upper = content.upper()
        if "CORRECT" in upper and "WRONG" not in upper:
            return "CORRECT"
        if "WRONG" in upper and "CORRECT" not in upper:
            return "WRONG"
        raise


MAX_RETRIES = 3


def evaluate_llm_judge(model_client, question, gold_answer, generated_answer, model="YOUR_JUDGE_MODEL"):
    """Score *generated_answer* against *gold_answer* via an LLM judge call."""

    call_kwargs = dict(
        model=model,
        messages=[
            {
                "role": "user",
                "content": ACCURACY_PROMPT.format(
                    question=question,
                    gold_answer=gold_answer,
                    generated_answer=generated_answer,
                ),
            }
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
        seed=42,
    )

    for attempt in range(MAX_RETRIES):
        try:
            response = model_client.chat.completions.create(**call_kwargs)
            label = _parse_judge_label(response.choices[0].message.content)
            return 1 if label == "CORRECT" else 0
        except Exception as exc:
            if "temperature" in str(exc) or "unsupported_value" in str(exc):
                call_kwargs.pop("temperature", None)
                call_kwargs.pop("seed", None)
                continue
            if isinstance(exc, (json.JSONDecodeError, KeyError)) and attempt < MAX_RETRIES - 1:
                print(f"[LLM judge] JSON parse failed (attempt {attempt + 1}/{MAX_RETRIES}), retrying... Q: {question[:80]}")
                continue
            print(f"\n{'=' * 80}")
            print(f"Unexpected error in LLM judge evaluation:")
            print(f"Error type: {type(exc).__name__}")
            print(f"Error: {exc}")
            print(f"Question: {question}")
            print(f"{'=' * 80}\n")
            raise

    print(f"[LLM judge] All {MAX_RETRIES} retries exhausted, defaulting to WRONG. Q: {question[:80]}")
    return 0


def main():
    """Run the LLM judge against a results file from the command line."""
    parser = argparse.ArgumentParser(description="Evaluate RAG results using LLM judge")
    parser.add_argument(
        "--input_file",
        type=str,
        default="results/default_run_v4_k30_new_graph.json",
        help="Path to the input dataset file",
    )

    args = parser.parse_args()

    dataset_path = args.input_file
    output_path = f"results/llm_judge_{dataset_path.split('/')[-1]}"

    with open(dataset_path, "r") as fh:
        data = json.load(fh)

    LLM_JUDGE = defaultdict(list)
    RESULTS = defaultdict(list)

    pos = 0
    for conv_key, qa_items in data.items():
        for entry in qa_items:
            question = entry["question"]
            gold_answer = entry["answer"]
            generated_answer = entry["response"]
            category = entry["category"]

            # Category 5 is excluded
            if int(category) == 5:
                continue

            label = evaluate_llm_judge(question, gold_answer, generated_answer)
            LLM_JUDGE[category].append(label)

            RESULTS[pos].append(
                {
                    "question": question,
                    "gt_answer": gold_answer,
                    "response": generated_answer,
                    "category": category,
                    "llm_label": label,
                }
            )

            # Persist intermediate results
            with open(output_path, "w") as fh:
                json.dump(RESULTS, fh, indent=4)

            # Per-category accuracy snapshot
            print("All categories accuracy:")
            for cat, results in LLM_JUDGE.items():
                if results:  # only print categories that have data
                    print(f"  Category {cat}: {np.mean(results):.4f} ({sum(results)}/{len(results)})")
            print("------------------------------------------")
        pos += 1

    with open(output_path, "w") as fh:
        json.dump(RESULTS, fh, indent=4)

    print("PATH: ", dataset_path)
    print("------------------------------------------")
    for cat, vals in LLM_JUDGE.items():
        print(cat, np.mean(vals))


if __name__ == "__main__":
    main()
