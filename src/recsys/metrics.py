import numpy as np


def recall_at_k(recs, truth, k):
    if not truth:
        return 0.0
    hits = sum(1 for r in recs[:k] if r in truth)
    return hits / len(truth)


def ndcg_at_k(recs, truth, k):
    if not truth:
        return 0.0
    dcg = sum(1.0 / np.log2(i + 2) for i, r in enumerate(recs[:k]) if r in truth)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(truth), k)))
    return dcg / ideal


def ap_at_k(recs, truth, k):
    if not truth:
        return 0.0
    hits, score = 0, 0.0
    for i, r in enumerate(recs[:k]):
        if r in truth:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(truth), k)


def catalog_coverage(all_recs, n_items, k):
    shown = set()
    for recs in all_recs:
        shown.update(recs[:k])
    return len(shown) / n_items