"""Top-K ranking metrics."""
import math


def recall_at_k(recommended, relevant, k=10):
    """Share of relevant items found in the top-k recommendations."""
    if not relevant:
        return 0.0
    hits = len(set(recommended[:k]) & relevant)
    return hits / len(relevant)


def hit_rate_at_k(recommended, relevant, k=10):
    """1 if any relevant item is in the top-k, else 0."""
    return float(bool(set(recommended[:k]) & relevant))


def ndcg_at_k(recommended, relevant, k=10):
    """Normalized discounted cumulative gain with binary relevance."""
    if not relevant:
        return 0.0
    dcg = sum(1 / math.log2(i + 2) for i, item in enumerate(recommended[:k]) if item in relevant)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg


def evaluate(recommendations, truth, k=10):
    """Average metrics over all visitors in truth.

    recommendations: visitor_id -> ranked list of item_ids
    truth: visitor_id -> set of relevant item_ids
    """
    totals = {"recall": 0.0, "ndcg": 0.0, "hit_rate": 0.0}
    for visitor, relevant in truth.items():
        recs = recommendations.get(visitor, [])
        totals["recall"] += recall_at_k(recs, relevant, k)
        totals["ndcg"] += ndcg_at_k(recs, relevant, k)
        totals["hit_rate"] += hit_rate_at_k(recs, relevant, k)
    n = len(truth)
    return {f"{name}@{k}": value / n for name, value in totals.items()} | {"n_users": n}
