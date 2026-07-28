"""
Borrowed from https://github.com/WujiangXu/AgenticMemory/blob/main/utils.py

@article{xu2025mem,
    title={A-mem: Agentic memory for llm agents},
    author={Xu, Wujiang and Liang, Zujie and Mei, Kai and Gao, Hang and Tan, Juntao
           and Zhang, Yongfeng},
    journal={arXiv preprint arXiv:2502.12110},
    year={2025}
}
"""

import re
import statistics
from collections import defaultdict
from typing import Dict, List, Union

import os
import nltk
from bert_score import score as bert_score
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer

# from load_dataset import load_locomo_dataset, QA, Turn, Session, Conversation
from sentence_transformers.util import pytorch_cos_sim

# Pull required NLTK assets
try:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("wordnet", quiet=True)
except Exception as exc:
    print(f"Error downloading NLTK data: {exc}")

_sentence_model = None


def _get_sentence_model():
    """
    Lazily load SentenceTransformer (avoid network downloads at import time).

    Environment overrides:
    - ``LOCOMO_DISABLE_SENTENCE_TRANSFORMERS=1``: always disable (returns None).
    - ``LOCOMO_SENTENCE_TRANSFORMERS_LOCAL_ONLY=1``: use only the local cache.
    - ``LOCOMO_SENTENCE_TRANSFORMERS_MODEL=<name-or-path>``: override model.
    """
    global _sentence_model
    if _sentence_model is not None:
        return _sentence_model

    if os.getenv("LOCOMO_DISABLE_SENTENCE_TRANSFORMERS", "").lower() in {"1", "true", "yes"}:
        _sentence_model = None
        return None

    model_name = os.getenv("LOCOMO_SENTENCE_TRANSFORMERS_MODEL", "all-MiniLM-L6-v2")
    local_only = os.getenv("LOCOMO_SENTENCE_TRANSFORMERS_LOCAL_ONLY", "").lower() in {"1", "true", "yes"}

    try:
        # SentenceTransformer forwards most kwargs into HuggingFace under the hood.
        # ``local_files_only`` avoids long retries / network dependency in offline runs.
        _sentence_model = SentenceTransformer(model_name, local_files_only=local_only)
        return _sentence_model
    except Exception as exc:
        print(f"Warning: Could not load SentenceTransformer model ({model_name}): {exc}")
        _sentence_model = None
        return None


def simple_tokenize(text):
    """Quick-and-dirty whitespace tokenizer (lower-case, punctuation stripped)."""
    text = str(text)
    return text.lower().replace(".", " ").replace(",", " ").replace("!", " ").replace("?", " ").split()


def calculate_rouge_scores(prediction: str, reference: str) -> Dict[str, float]:
    """Calculate ROUGE F-scores for *prediction* against *reference*."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return {
        "rouge1_f": scores["rouge1"].fmeasure,
        "rouge2_f": scores["rouge2"].fmeasure,
        "rougeL_f": scores["rougeL"].fmeasure,
    }


def calculate_bleu_scores(prediction: str, reference: str) -> Dict[str, float]:
    """Calculate BLEU 1–4 scores for *prediction* against *reference*."""

    pred_tokens = nltk.word_tokenize(prediction.lower())
    ref_tokens = [nltk.word_tokenize(reference.lower())]

    weights_list = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (0.33, 0.33, 0.33, 0), (0.25, 0.25, 0.25, 0.25)]
    smooth = SmoothingFunction().method1

    scores = {}
    for n, weights in enumerate(weights_list, start=1):
        try:
            score = sentence_bleu(ref_tokens, pred_tokens, weights=weights, smoothing_function=smooth)
        except Exception as exc:
            print(f"Error calculating BLEU score: {exc}")
            score = 0.0
        scores[f"bleu{n}"] = score

    return scores


def calculate_bert_scores(prediction: str, reference: str) -> Dict[str, float]:
    """Calculate BERTScore (semantic similarity)."""
    try:
        P, R, F1 = bert_score([prediction], [reference], lang="en", verbose=False)
        return {"bert_precision": P.item(), "bert_recall": R.item(), "bert_f1": F1.item()}
    except Exception as exc:
        print(f"Error calculating BERTScore: {exc}")
        return {"bert_precision": 0.0, "bert_recall": 0.0, "bert_f1": 0.0}


def calculate_meteor_score(prediction: str, reference: str) -> float:
    """Calculate the METEOR score."""
    try:
        return meteor_score([reference.split()], prediction.split())
    except Exception as exc:
        print(f"Error calculating METEOR score: {exc}")
        return 0.0


def calculate_sentence_similarity(prediction: str, reference: str) -> float:
    """Calculate sentence-embedding cosine similarity via SentenceBERT."""
    sentence_model = _get_sentence_model()
    if sentence_model is None:
        return 0.0
    try:
        embedding1 = sentence_model.encode([prediction], convert_to_tensor=True)
        embedding2 = sentence_model.encode([reference], convert_to_tensor=True)

        similarity = pytorch_cos_sim(embedding1, embedding2).item()
        return float(similarity)
    except Exception as exc:
        print(f"Error calculating sentence similarity: {exc}")
        return 0.0


def calculate_metrics(prediction: str, reference: str) -> Dict[str, float]:
    """Compute the basket of evaluation metrics for one prediction."""
    # Bail out when either side is missing
    if not prediction or not reference:
        return {
            "exact_match": 0,
            "f1": 0.0,
            "rouge1_f": 0.0,
            "rouge2_f": 0.0,
            "rougeL_f": 0.0,
            "bleu1": 0.0,
            "bleu2": 0.0,
            "bleu3": 0.0,
            "bleu4": 0.0,
            "bert_f1": 0.0,
            "meteor": 0.0,
            "sbert_similarity": 0.0,
        }

    prediction = str(prediction).strip()
    reference = str(reference).strip()

    # Exact-match score
    exact_match = int(prediction.lower() == reference.lower())

    # Token-level F1
    pred_tokens = set(simple_tokenize(prediction))
    ref_tokens = set(simple_tokenize(reference))
    common_tokens = pred_tokens & ref_tokens

    if not pred_tokens or not ref_tokens:
        f1 = 0.0
    else:
        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(ref_tokens)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    bleu_scores = calculate_bleu_scores(prediction, reference)

    metrics = {
        "exact_match": exact_match,
        "f1": f1,
        **bleu_scores,
    }

    return metrics


def aggregate_metrics(
    all_metrics: List[Dict[str, float]], all_categories: List[int]
) -> Dict[str, Dict[str, Union[float, Dict[str, float]]]]:
    """Compute aggregate statistics across metrics, broken down by category."""
    if not all_metrics:
        return {}

    # Buckets for overall and per-category
    aggregates = defaultdict(list)
    category_aggregates = defaultdict(lambda: defaultdict(list))

    # Collect all values for each metric (overall + per category)
    for metrics, category in zip(all_metrics, all_categories):
        for metric_name, value in metrics.items():
            aggregates[metric_name].append(value)
            category_aggregates[category][metric_name].append(value)

    # Overall statistics
    results = {"overall": {}}

    for metric_name, values in aggregates.items():
        results["overall"][metric_name] = {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "count": len(values),
        }

    # Per-category statistics
    for category in sorted(category_aggregates.keys()):
        results[f"category_{category}"] = {}
        for metric_name, values in category_aggregates[category].items():
            if values:  # Skip categories that have no values for this metric
                results[f"category_{category}"][metric_name] = {
                    "mean": statistics.mean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "median": statistics.median(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values),
                }

    return results


def extract_json(text):
    """
    Extract JSON content from *text*, peeling off enclosing triple backticks
    and an optional ``json`` tag. When no code block is present, returns
    *text* as-is.
    """
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    return text  # treat as raw JSON
