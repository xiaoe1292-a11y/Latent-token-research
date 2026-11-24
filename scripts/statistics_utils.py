"""Descriptive Pareto and paired user bootstrap for single-target recall."""


def pareto(rows, quality='recall@100', cost='latency_p50_ms'):
    return [r['point'] for r in rows if not any(
        q[quality] >= r[quality] and q[cost] <= r[cost]
        and (q[quality] > r[quality] or q[cost] < r[cost]) for q in rows)]


def paired(a, b, seed=20261123, draws=20000):
    import numpy as np
    x, y = np.asarray(a['ranks']) <= 100, np.asarray(b['ranks']) <= 100
    if not len(x) or len(x) != len(y):
        raise ValueError('Paired populations must be nonempty and aligned')
    full_a, full_b = np.asarray(a['returned']) == 100, np.asarray(b['returned']) == 100
    if len(full_a) != len(x) or len(full_b) != len(x):
        raise ValueError('Coverage population mismatch')
    gained = ~x & y
    lost = x & ~y
    counts = [int(gained.sum()), int(lost.sum()), int((x == y).sum())]
    samples = np.random.default_rng(seed).multinomial(len(x), np.asarray(counts) / len(x), size=draws)
    differences = (samples[:, 0] - samples[:, 1]) / len(x)
    interval = np.quantile(differences, [.025, .975])
    both = full_a & full_b
    return dict(users=len(x), gained_hits100=counts[0], lost_hits100=counts[1],
                gained_from_previously_short=int((gained & ~full_a).sum()),
                gained_from_previously_full=int((gained & full_a).sum()),
                delta_recall100=(counts[0] - counts[1]) / len(x),
                bootstrap_95_low=float(interval[0]), bootstrap_95_high=float(interval[1]),
                bootstrap_draws=draws, bootstrap_seed=seed,
                both_full_users=int(both.sum()),
                delta_recall100_both_full=float((y[both].astype(int) - x[both].astype(int)).mean()) if both.any() else None,
                bootstrap_method='paired user percentile bootstrap, exact three-category multinomial representation')
