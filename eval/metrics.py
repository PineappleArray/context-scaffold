import re
from collections import Counter
 
 
# ─── Partial-Match F1 (LoCoMo's primary metric) ──────────
 
def normalize_text(text: str) -> str:
    """Lowercase, remove punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
 
 
def compute_f1(predicted: str, gold: str) -> float:
    """
    Token-level F1 between predicted and gold answers.
    This is the primary metric used in the LoCoMo paper.
    
    F1 = 2 * (precision * recall) / (precision + recall)
    where precision = correct_tokens / predicted_tokens
          recall    = correct_tokens / gold_tokens
    """
    pred_tokens = normalize_text(predicted).split()
    gold_tokens = normalize_text(gold).split()
 
    if not pred_tokens or not gold_tokens:
        return 0.0
 
    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)
 
    # Count overlapping tokens
    overlap = 0
    for token, count in pred_counts.items():
        overlap += min(count, gold_counts.get(token, 0))
 
    if overlap == 0:
        return 0.0
 
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1
 
 
# ─── Exact Match ──────────────────────────────────────────
 
def compute_exact_match(predicted: str, gold: str) -> float:
    """Binary: 1.0 if normalized strings match, 0.0 otherwise."""
    return 1.0 if normalize_text(predicted) == normalize_text(gold) else 0.0
 
 
# ─── BERTScore ────────────────────────────────────────────
 
def compute_bert_score(predicted: str, gold: str) -> float:
    """
    Semantic similarity between predicted and gold using BERTScore.
    Returns F1 component of BERTScore.
    
    Higher = more semantically similar, even if wording differs.
    This catches cases where F1 misses paraphrases:
        gold: "moved to Denver"
        pred: "relocated to Denver, Colorado"
        F1 would be low, but BERTScore would be high.
    """
    try:
        from bert_score import score
        P, R, F1 = score(
            [predicted],
            [gold],
            model_type="microsoft/deberta-xlarge-mnli",
            lang="en",
            verbose=False
        )
        return F1.item()
    except ImportError:
        print("bert-score not installed, skipping. pip install bert-score")
        return 0.0
 
 
# ─── ROUGE-L ──────────────────────────────────────────────
 
def compute_rouge(predicted: str, gold: str) -> float:
    """
    ROUGE-L score (longest common subsequence).
    Measures structural similarity between predicted and gold.
    """
    try:
        from rouge_score.rouge_scorer import RougeScorer
        scorer = RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(gold, predicted)
        return scores['rougeL'].fmeasure
    except ImportError:
        print("rouge-score not installed, skipping. pip install rouge-score")
        return 0.0
 
 
# ─── Aggregate Results ────────────────────────────────────
 
def aggregate_results(results: dict) -> dict:
    """
    Compute overall and per-category averages.
    Returns summary dict with avg_f1, avg_bert, avg_rouge, avg_latency.
    """
    all_results = []
    for conv in results["conversations"]:
        all_results.extend(conv["results"])
 
    if not all_results:
        return {}
 
    summary = {
        "total_questions": len(all_results),
        "avg_f1": sum(r["f1"] for r in all_results) / len(all_results),
        "avg_bert": sum(r["bert_score"] for r in all_results) / len(all_results),
        "avg_rouge": sum(r["rouge_l"] for r in all_results) / len(all_results),
        "avg_latency": sum(r["latency"] for r in all_results) / len(all_results),
        "per_category": {}
    }
 
    for category in ["single_hop", "multi_hop", "temporal", "open_domain", "adversarial"]:
        cat_results = results["by_category"].get(category, [])
        if cat_results:
            summary["per_category"][category] = {
                "count": len(cat_results),
                "avg_f1": sum(r["f1"] for r in cat_results) / len(cat_results),
                "avg_bert": sum(r["bert_score"] for r in cat_results) / len(cat_results),
                "avg_rouge": sum(r["rouge_l"] for r in cat_results) / len(cat_results),
                "avg_latency": sum(r["latency"] for r in cat_results) / len(cat_results),
            }
 
    return summary