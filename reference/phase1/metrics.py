"""Item-level, single-target metrics with explicit missing candidate slots."""
import math


KEYS = ('recall@10', 'recall@50', 'recall@100', 'ndcg@10')


def measure(rows):
    sums = dict.fromkeys(KEYS, 0.0)
    valid = 0
    short = {str(k): 0 for k in (10, 50, 100)}
    users = set()
    for row in rows:
        if row['user'] in users:
            raise ValueError('Duplicate user')
        users.add(row['user'])
        items = row['candidate_slots']
        if len(items) != 100:
            raise ValueError('Exactly 100 candidate slots required')
        present = [x for x in items if x is not None]
        if len(present) != len(set(present)):
            raise ValueError('Duplicate recommended item')
        rank = next((i + 1 for i, item in enumerate(items) if item == row['target']), None)
        if rank != row['rank']:
            raise ValueError('Stored rank disagrees with item candidates')
        valid += len(present)
        for k in (10, 50, 100):
            short[str(k)] += len(present) < k
            if rank is not None and rank <= k:
                sums['recall@' + str(k)] += 1
        if rank is not None and rank <= 10:
            sums['ndcg@10'] += 1 / math.log2(rank + 1)
    if not rows:
        raise ValueError('Empty evaluation')
    return dict(n=len(rows), sums=sums, valid_candidates_sum=valid, insufficient=short)


def combine(results, users):
    identity = {(r['checkpoint_sha256'], r['split'], r['protocol_sha256']) for r in results}
    if len(identity) != 1:
        raise ValueError('Mixed checkpoint, split or protocol')
    position = 0
    sums = dict.fromkeys(KEYS, 0.0)
    valid = 0
    short = {str(k): 0 for k in (10, 50, 100)}
    for result in sorted(results, key=lambda r: r['start']):
        if result['start'] != position or result['end'] <= position or result['n'] != result['end'] - position:
            raise ValueError('Missing, overlapping or invalid user shard')
        position = result['end']
        for key in KEYS:
            sums[key] += result['sums'][key]
        valid += result['valid_candidates_sum']
        for key in short:
            short[key] += result['insufficient'][key]
    if position != users:
        raise ValueError('Incomplete user population')
    return dict(users=users, metrics={k: v / users for k, v in sums.items()},
                mean_unique_candidates=valid / users, insufficient_users=short)
