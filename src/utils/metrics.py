"""
Shared metrics utilities for Tasks 1-3.
Uses nltk for BLEU scores and rouge-score for ROUGE scores.
Levenshtein distance implemented via DP (trivial, no external dep needed).
"""

import math
from typing import Dict, List

import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer


# Ensure nltk punkt tokenizer data is available (downloaded once)
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


def levenshtein_distance(s1: str, s2: str) -> int:
    """DP Levenshtein (edit) distance. O(m*n) time, O(n) space."""
    m, n = len(s1), len(s2)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def char_accuracy(pred: str, ref: str) -> float:
    """Fraction of characters in ref matched at the same position in pred."""
    if len(ref) == 0:
        return 1.0 if len(pred) == 0 else 0.0
    min_len = min(len(pred), len(ref))
    matches = sum(1 for i in range(min_len) if pred[i] == ref[i])
    return matches / len(ref)


def word_accuracy(pred: str, ref: str) -> float:
    """Fraction of words in ref matched at the same position in pred."""
    pred_words = pred.split()
    ref_words = ref.split()
    if len(ref_words) == 0:
        return 1.0 if len(pred_words) == 0 else 0.0
    min_len = min(len(pred_words), len(ref_words))
    matches = sum(1 for i in range(min_len) if pred_words[i] == ref_words[i])
    return matches / len(ref_words)


def compute_decryption_metrics(
    predictions: List[str], references: List[str]
) -> Dict[str, float]:
    """Aggregate char_acc, word_acc, avg_levenshtein over a corpus."""
    assert len(predictions) == len(references)
    n = len(predictions)
    if n == 0:
        return {"char_acc": 0.0, "word_acc": 0.0, "avg_levenshtein": 0.0}

    total_ca = sum(char_accuracy(p, r) for p, r in zip(predictions, references))
    total_wa = sum(word_accuracy(p, r) for p, r in zip(predictions, references))
    total_lev = sum(levenshtein_distance(p, r) for p, r in zip(predictions, references))

    return {
        "char_acc": total_ca / n,
        "word_acc": total_wa / n,
        "avg_levenshtein": total_lev / n,
    }


def perplexity(loss: float) -> float:
    """Perplexity from mean cross-entropy loss."""
    return math.exp(min(loss, 20.0))


def bleu_score(
    predictions: List[str], references: List[str], max_n: int = 4
) -> Dict[str, float]:
    """
    Corpus-level BLEU-1 through BLEU-4 using nltk.
    Returns dict: bleu_1, bleu_2, bleu_3, bleu_4, bleu (cumulative 4-gram).
    """
    refs_tok = [[r.split()] for r in references]  # list of [list of ref tokens]
    preds_tok = [p.split() for p in predictions]
    smooth = SmoothingFunction().method1

    result: Dict[str, float] = {}
    for n in range(1, max_n + 1):
        weights = tuple([1.0 / n] * n + [0.0] * (max_n - n))
        result[f"bleu_{n}"] = corpus_bleu(
            refs_tok, preds_tok, weights=weights, smoothing_function=smooth
        )
    # Cumulative BLEU-4 (equal weights)
    result["bleu"] = corpus_bleu(
        refs_tok, preds_tok, weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smooth,
    )
    return result


def rouge_score(
    predictions: List[str], references: List[str]
) -> Dict[str, float]:
    """
    Corpus-level ROUGE-1, ROUGE-2, ROUGE-L using google rouge-score library.
    Returns dict with rouge_{1,2,l}_{precision,recall,f1}.
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)

    keys = ["rouge1", "rouge2", "rougeL"]
    totals = {k: {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0} for k in keys}
    n = len(predictions)

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for k in keys:
            totals[k]["precision"] += scores[k].precision
            totals[k]["recall"] += scores[k].recall
            totals[k]["fmeasure"] += scores[k].fmeasure

    denom = max(n, 1)
    name_map = {"rouge1": "rouge_1", "rouge2": "rouge_2", "rougeL": "rouge_l"}
    result: Dict[str, float] = {}
    for k in keys:
        prefix = name_map[k]
        result[f"{prefix}_precision"] = totals[k]["precision"] / denom
        result[f"{prefix}_recall"] = totals[k]["recall"] / denom
        result[f"{prefix}_f1"] = totals[k]["fmeasure"] / denom

    return result


def compute_all_metrics(
    predictions: List[str], references: List[str]
) -> Dict[str, float]:
    """
    Full metrics dict for Task 3:
        char_acc, word_acc, avg_levenshtein,
        bleu_1..4, bleu,
        rouge_1/2/l precision/recall/f1
    """
    metrics: Dict[str, float] = {}
    metrics.update(compute_decryption_metrics(predictions, references))
    metrics.update(bleu_score(predictions, references))
    metrics.update(rouge_score(predictions, references))
    return metrics
