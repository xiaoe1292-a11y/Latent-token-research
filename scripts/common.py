"""Portable readers for the published aggregate results."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COUNTS = (2, 4, 8, 16)
BEAMS = (250, 500, 1000, 2000)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(root=ROOT):
    with (Path(root) / 'data/joint16/results.csv').open(encoding='utf-8-sig', newline='') as stream:
        raw = list(csv.DictReader(stream))
    numeric = ('n', 'beam', 'recall@10', 'recall@50', 'recall@100', 'ndcg@10',
        'mean_returned', 'min_returned', 'mean_unique_beam_items', 'users_below100',
        'fraction_full100', 'valid_rate', 'duplicate_rate_valid', 'duplicate_rate_all',
        'invalid_sid_rate', 'invalid_latent_rate', 'unique_yield_rate',
        'latency_p50_ms', 'latency_p95_ms', 'throughput_users_per_second',
        'peak_allocated_mib', 'peak_reserved_mib')
    return [dict(row, **{key: float(row[key]) for key in numeric}) for row in raw]
