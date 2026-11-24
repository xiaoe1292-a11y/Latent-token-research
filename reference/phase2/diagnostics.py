"""Counts of decoded beam paths; these are counts, not posterior probabilities."""
import math

SCALARS = ('paths', 'invalid_latent', 'invalid_sid', 'valid_paths', 'unique_beam_items',
           'duplicate_valid_paths', 'effective_latents')


def inspect_paths(paths, n, inverse):
    latent, valid = [0] * n, [0] * n
    items = set()
    bad_latent = bad_sid = good = 0
    for path in paths:
        z = path[0] if path else 0
        if not 1 <= z <= n:
            bad_latent += 1
            continue
        latent[z - 1] += 1
        item = inverse.get(tuple(path[1:]))
        if item is None:
            bad_sid += 1
            continue
        valid[z - 1] += 1
        good += 1
        items.add(item)
    total = sum(latent)
    effective = math.exp(-sum((v / total) * math.log(v / total) for v in latent if v)) if total else 0.0
    return dict(paths=len(paths), invalid_latent=bad_latent, invalid_sid=bad_sid, valid_paths=good,
                unique_beam_items=len(items), duplicate_valid_paths=good - len(items),
                effective_latents=effective, latent_beams=latent, latent_valid_paths=valid)


def summarize(rows, n, beam):
    sums = dict.fromkeys(SCALARS, 0.0)
    latent, valid = [0] * n, [0] * n
    for row in rows:
        d = row['path_diagnostics']
        assert d['paths'] == beam
        assert d['invalid_latent'] + d['invalid_sid'] + d['valid_paths'] == beam
        assert d['duplicate_valid_paths'] == d['valid_paths'] - d['unique_beam_items']
        assert sum(d['latent_beams']) == beam - d['invalid_latent']
        assert sum(d['latent_valid_paths']) == d['valid_paths']
        assert len(d['latent_beams']) == len(d['latent_valid_paths']) == n
        assert 0 <= d['effective_latents'] <= n + 1e-10
        assert len([x for x in row['candidate_slots'] if x is not None]) == min(100, d['unique_beam_items'])
        for key in SCALARS:
            sums[key] += d[key]
        latent = [a + b for a, b in zip(latent, d['latent_beams'])]
        valid = [a + b for a, b in zip(valid, d['latent_valid_paths'])]
    return dict(path_sums=sums, latent_beam_totals=latent, latent_valid_totals=valid)


def combine(results, users):
    identity = {(r['point_id'], r['count'], r['beam'], r['checkpoint_sha256'], r['protocol_sha256']) for r in results}
    if len(identity) != 1:
        raise ValueError('Mixed latent count, fixed beam, checkpoint or protocol')
    n, beam = results[0]['count'], results[0]['beam']
    sums = {key: sum(r['path_sums'][key] for r in results) for key in SCALARS}
    assert sums['paths'] == beam * users
    latent = [sum(r['latent_beam_totals'][i] for r in results) for i in range(n)]
    valid = [sum(r['latent_valid_totals'][i] for r in results) for i in range(n)]
    return dict(mean_unique_items_in_beam=sums['unique_beam_items'] / users,
                mean_count_effective_latents=sums['effective_latents'] / users,
                invalid_latent_fraction=sums['invalid_latent'] / sums['paths'],
                invalid_sid_fraction=sums['invalid_sid'] / sums['paths'],
                duplicate_fraction_of_valid_paths=sums['duplicate_valid_paths'] / sums['valid_paths'] if sums['valid_paths'] else 0,
                valid_path_fraction=sums['valid_paths'] / sums['paths'], path_sums=sums,
                latent_beam_totals=latent, latent_valid_totals=valid)
