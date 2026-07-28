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

import nltk
from bert_score import score as bert_score
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer

# from load_dataset import load_locomo_dataset, QA, Turn, Session, Conversation
from sentence_transformers.util import pytorch_cos_sim

# Make sure the NLTK resources we depend on are available locally.
try:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("wordnet", quiet=True)
except Exception as exc:
    print(f"Error downloading NLTK data: {exc}")

# Module-level SentenceTransformer (shared across calls to avoid reloading).
try:
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as exc:
    print(f"Warning: Could not load SentenceTransformer model: {exc}")
    sentence_model = None


def simple_tokenize(text):
    """Lowercase and split on common punctuation/whitespace."""
    body = str(text)
    return body.lower().replace(".", " ").replace(",", " ").replace("!", " ").replace("?", " ").split()


def calculate_rouge_scores(prediction: str, reference: str) -> Dict[str, float]:
    """Compute ROUGE-1/2/L F-measures for ``prediction`` vs ``reference``."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return {
        "rouge1_f": scores["rouge1"].fmeasure,
        "rouge2_f": scores["rouge2"].fmeasure,
        "rougeL_f": scores["rougeL"].fmeasure,
    }


def calculate_bleu_scores(prediction: str, reference: str) -> Dict[str, float]:
    """Compute BLEU-1 through BLEU-4 with method-1 smoothing."""
    pred_tokens = nltk.word_tokenize(prediction.lower())
    ref_tokens = [nltk.word_tokenize(reference.lower())]

    weight_sets = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (0.33, 0.33, 0.33, 0), (0.25, 0.25, 0.25, 0.25)]
    smooth_fn = SmoothingFunction().method1

    out = {}
    for n, weights in enumerate(weight_sets, start=1):
        try:
            val = sentence_bleu(ref_tokens, pred_tokens, weights=weights, smoothing_function=smooth_fn)
        except Exception as exc:
            print(f"Error calculating BLEU score: {exc}")
            val = 0.0
        out[f"bleu{n}"] = val

    return out


def calculate_bert_scores(prediction: str, reference: str) -> Dict[str, float]:
    """Compute BERTScore precision/recall/F1."""
    try:
        P, R, F1 = bert_score([prediction], [reference], lang="en", verbose=False)
        return {"bert_precision": P.item(), "bert_recall": R.item(), "bert_f1": F1.item()}
    except Exception as exc:
        print(f"Error calculating BERTScore: {exc}")
        return {"bert_precision": 0.0, "bert_recall": 0.0, "bert_f1": 0.0}


def calculate_meteor_score(prediction: str, reference: str) -> float:
    """Compute the METEOR score of ``prediction`` against ``reference``."""
    try:
        return meteor_score([reference.split()], prediction.split())
    except Exception as exc:
        print(f"Error calculating METEOR score: {exc}")
        return 0.0


def calculate_sentence_similarity(prediction: str, reference: str) -> float:
    """Cosine similarity between SentenceBERT embeddings of the two strings."""
    if sentence_model is None:
        return 0.0
    try:
        emb_pred = sentence_model.encode([prediction], convert_to_tensor=True)
        emb_ref = sentence_model.encode([reference], convert_to_tensor=True)

        sim = pytorch_cos_sim(emb_pred, emb_ref).item()
        return float(sim)
    except Exception as exc:
        print(f"Error calculating sentence similarity: {exc}")
        return 0.0


def calculate_metrics(prediction: str, reference: str) -> Dict[str, float]:
    """Compute the bundle of evaluation metrics used by the eval pipeline."""
    # Treat empty / falsy inputs as a zero result across the board.
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

    # Exact-match (case-insensitive).
    exact_match = int(prediction.lower() == reference.lower())

    # Token-level F1 between the two token sets.
    pred_tokens = set(simple_tokenize(prediction))
    ref_tokens = set(simple_tokenize(reference))
    overlap = pred_tokens & ref_tokens

    if not pred_tokens or not ref_tokens:
        f1 = 0.0
    else:
        precision = len(overlap) / len(pred_tokens)
        recall = len(overlap) / len(ref_tokens)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    bleu_scores = calculate_bleu_scores(prediction, reference)

    # Combine into a single flat dict.
    metrics = {
        "exact_match": exact_match,
        "f1": f1,
        **bleu_scores,
    }

    return metrics


def aggregate_metrics(
    all_metrics: List[Dict[str, float]], all_categories: List[int]
) -> Dict[str, Dict[str, Union[float, Dict[str, float]]]]:
    """Compute overall + per-category descriptive statistics for a batch of metric dicts."""
    if not all_metrics:
        return {}

    # Collect series of values per metric, both globally and per category.
    overall_buckets = defaultdict(list)
    per_cat_buckets = defaultdict(lambda: defaultdict(list))

    for metric_dict, category in zip(all_metrics, all_categories):
        for metric_name, value in metric_dict.items():
            overall_buckets[metric_name].append(value)
            per_cat_buckets[category][metric_name].append(value)

    results = {"overall": {}}

    # Descriptive stats over the full population.
    for metric_name, values in overall_buckets.items():
        results["overall"][metric_name] = {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "count": len(values),
        }

    # Same descriptive stats, but split by category.
    for category in sorted(per_cat_buckets.keys()):
        results[f"category_{category}"] = {}
        for metric_name, values in per_cat_buckets[category].items():
            if values:
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
    Pull a JSON payload out of ``text``. If the text is wrapped in a ```/```json
    fenced block the inner content is returned; otherwise the input is treated
    as raw JSON.
    """
    body = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", body, re.DOTALL)
    if fenced:
        return fenced.group(1)
    return body
