"""Recalculate published metrics and costs on CPU; no GPU or raw IDs needed."""
import argparse
import csv
import json
import math
from pathlib import Path
import platform

import numpy as np

from common import ROOT, COUNTS, BEAMS, read_json, rows, sha256
from statistics_utils import paired, pareto


def close(actual, expected, label, tolerance=1e-10):
    if not math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=tolerance):
        raise ValueError('%s: calculated %r, published %r' % (label, actual, expected))


def verify(root=ROOT):
    root = Path(root)
    data, cohorts = rows(root), {}
    if len(data) != 16 or {(int(r['n']), int(r['beam'])) for r in data} != {(n,b) for n in COUNTS for b in BEAMS}:
        raise ValueError('Expected exactly the16 distinct N x Beam configurations')
    config = read_json(root / 'configs/joint16.json')
    if config['automatic_beam_retry'] or config['generate_calls_per_request'] != 1:
        raise ValueError('Joint matrix must use a single generation with the fixed actual beam')
    paths = read_json(root / 'data/joint16/path-counts.json')
    profiles = read_json(root / 'data/joint16/profiles.json')
    timed_requests = 0
    with np.load(root / 'data/joint16/paired-records.npz', allow_pickle=False) as records, np.load(root / 'data/joint16/timings.npz', allow_pickle=False) as timing:
        for row in data:
            point, beam = row['point'], int(row['beam'])
            rank, returned = records[point + '_rank'], records[point + '_returned']
            if rank.shape != (94762,) or returned.shape != rank.shape or not np.all((rank >= 1) & (rank <= 101)) or not np.all(returned <= 100):
                raise ValueError(point + ': invalid aligned rank/coverage arrays')
            cohorts[point] = dict(ranks=rank.copy(), returned=returned.copy())
            for k in (10, 50, 100):
                close((rank <= k).mean(), row['recall@' + str(k)], point + ' Recall@' + str(k), 1e-12)
            # Float64 matches scalar math.log2; int16 ufunc promotion can use float32.
            discount = np.where(rank <= 10, 1 / np.log2(rank.astype(np.float64) + 1), 0)
            close(discount.mean(), row['ndcg@10'], point + ' NDCG@10', 1e-12)
            close(returned.mean(), row['mean_returned'], point + ' mean_returned')
            close((returned == 100).mean(), row['fraction_full100'], point + ' fraction_full100')
            close((returned < 100).sum(), row['users_below100'], point + ' users_below100')
            p = paths[point]
            if p['paths'] != beam * 94762 or p['invalid_latent'] + p['invalid_sid'] + p['valid_paths'] != p['paths']:
                raise ValueError(point + ': path-count conservation failed')
            if p['duplicate_valid_paths'] != p['valid_paths'] - p['unique_beam_items']:
                raise ValueError(point + ': duplicate counts inconsistent')
            for key, numerator, denominator in (
                ('valid_rate','valid_paths','paths'), ('invalid_sid_rate','invalid_sid','paths'),
                ('invalid_latent_rate','invalid_latent','paths'),
                ('duplicate_rate_valid','duplicate_valid_paths','valid_paths'),
                ('duplicate_rate_all','duplicate_valid_paths','paths'),
                ('unique_yield_rate','unique_beam_items','paths')):
                close(p[numerator] / p[denominator], row[key], point + ' ' + key)
            close(p['unique_beam_items'] / 94762, row['mean_unique_beam_items'], point + ' unique candidates')
            t, profile = timing[point], profiles[point]
            if t.shape != (768,) or not np.all(np.isfinite(t) & (t > 0)):
                raise ValueError(point + ': expected768 positive raw timings')
            if profile['actual_beam'] != beam or profile['retry_calls'] != 0 or profile['batch_size'] != 1:
                raise ValueError(point + ': changed beam/retry/batch settings')
            if profile['profile_mode'] != row['profile_mode'] or profile['profile_mode'] != 'shared_gpu_observation' or profile['exclusive_gpu']:
                raise ValueError(point + ': cost scope must remain shared GPU observation')
            close(np.quantile(t, .5) * 1000, row['latency_p50_ms'], point + ' P50')
            close(np.quantile(t, .95) * 1000, row['latency_p95_ms'], point + ' P95')
            close(len(t) / t.sum(), row['throughput_users_per_second'], point + ' throughput')
            close(profile['peak_allocated_bytes'] / 1048576, row['peak_allocated_mib'], point + ' memory MiB')
            close(profile['peak_reserved_bytes'] / 1048576, row['peak_reserved_mib'], point + ' reserved MiB')
            for actual, expected in zip(np.median(t.reshape(3,256), axis=1) * 1000, json.loads(row['repeat_p50_ms'])):
                close(actual, expected, point + ' repeat P50')
            timed_requests += len(t)
    comparisons = read_json(root / 'data/joint16/paired-bootstrap.json')
    if len(comparisons) != 36 or sum(c['kind'] == 'beam' for c in comparisons) != 12:
        raise ValueError('Expected12 adjacent-beam and24 within-beam latent comparisons')
    for c in comparisons:
        reproduced = paired(cohorts[c['before']], cohorts[c['after']], c['bootstrap_seed'], c['bootstrap_draws'])
        for key, value in reproduced.items():
            if isinstance(value, (int, float)):
                close(value, c[key], 'bootstrap ' + c['before'] + ' ' + key, 1e-12)
            elif value != c[key]:
                raise ValueError('Bootstrap metadata or conditional coverage differs: ' + key)
    for budget in read_json(root / 'data/joint16/budget-recommendations.json'):
        available = [r for r in data if r['latency_p50_ms'] <= budget['p50_budget_ms'] and r['peak_allocated_mib'] <= budget['allocated_budget_mib'] and r['fraction_full100'] >= budget['minimum_fraction_full100']]
        selected = max(available, key=lambda r: (r['recall@100'], -r['latency_p50_ms'])) if available else None
        if budget['selected'] != (selected['point'] if selected else None):
            raise ValueError('Published conditional budget recommendation is inconsistent')
    with (root / 'data/phase2/results.csv').open(encoding='utf-8-sig', newline='') as stream:
        previous = list(csv.DictReader(stream))
    if len(previous) != 9:
        raise ValueError('Expected9 historical fixed-beam points')
    index = {r['point']: r for r in data}
    for row in previous:
        for metric in ('recall@10','recall@50','recall@100','ndcg@10'):
            close(row[metric], index[row['point']][metric], 'reused quality ' + row['point'] + ' ' + metric, 1e-12)
    with (root / 'data/phase1/results.csv').open(encoding='utf-8-sig', newline='') as stream:
        first = list(csv.DictReader(stream))
    if len(first) != 10 or {int(r['n_latent_tokens']) for r in first} != {1,2,4,8,16}:
        raise ValueError('Expected5 phase1 models with batch1 and4 profiles')
    for n in (1,2,4,8,16):
        a, b = [r for r in first if int(r['n_latent_tokens']) == n]
        for metric in ('recall@10','recall@50','recall@100','ndcg@10'):
            close(a[metric], b[metric], 'phase1 duplicated quality row', 1e-12)
    for name, digest in read_json(root / 'provenance/data-sha256.json').items():
        if sha256(root / name) != digest:
            raise ValueError('Published input hash mismatch: ' + name)
    for name, digest in read_json(root / 'provenance/reference-source-sha256.json').items():
        if sha256(root / name) != digest:
            raise ValueError('Archived reference source changed: ' + name)
    return dict(status='passed', points=16, user_point_records=1516192,
        raw_timed_requests=timed_requests, paired_bootstrap_comparisons=36,
        historical_quality_points_checked=9, phase1_models_archived=5,
        observed_pareto=pareto(data), cost_mode='shared_gpu_observation',
        python=platform.python_version(), numpy=np.__version__,
        scope='CPU recalculation of published ranks, path sums, raw times, paired bootstrap and budgets. Full candidate lists/checkpoints are not included.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = verify(args.root)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
