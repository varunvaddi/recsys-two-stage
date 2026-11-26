import numpy as np

from recsys import metrics


def per_user(rec, users, truth, k=10):
    """Per-user recall@k and ndcg@k arrays, in the order of `users`."""
    recall = np.array([metrics.recall_at_k(list(r[:k]), truth[u], k) for u, r in zip(users, rec)])
    ndcg = np.array([metrics.ndcg_at_k(list(r[:k]), truth[u], k) for u, r in zip(users, rec)])
    return recall, ndcg


def bootstrap_ci(values, n_boot=1000, alpha=0.05, seed=0):
    """Mean and a (1-alpha) percentile interval from resampling users with replacement."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.array([values[rng.integers(0, len(values), len(values))].mean() for _ in range(n_boot)])
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(values.mean()), float(lo), float(hi)


def paired_lift_ci(a, b, n_boot=1000, alpha=0.05, seed=0):
    """Relative lift of mean(a) over mean(b) on the SAME users, with a bootstrap interval.
    Resampling users jointly keeps the pairing, so it measures the difference, not two noisy levels."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    lifts = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(a), len(a))
        lifts.append(a[idx].mean() / b[idx].mean() - 1)
    lo, hi = np.quantile(lifts, [alpha / 2, 1 - alpha / 2])
    return float(a.mean() / b.mean() - 1), float(lo), float(hi)