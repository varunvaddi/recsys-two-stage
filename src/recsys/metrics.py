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

def evaluate(rec_matrix, users, truth_by_user, n_items, k=10):
    recs_list = [list(row[:k]) for row in rec_matrix]
    recall, ndcg, ap = [], [], []
    for u, recs in zip(users, recs_list):
        t = truth_by_user[u]
        recall.append(recall_at_k(recs, t, k))
        ndcg.append(ndcg_at_k(recs, t, k))
        ap.append(ap_at_k(recs, t, k))
    return {
        f"recall@{k}": float(np.mean(recall)),
        f"ndcg@{k}": float(np.mean(ndcg)),
        f"map@{k}": float(np.mean(ap)),
        f"coverage@{k}": catalog_coverage(recs_list, n_items, k),
    }