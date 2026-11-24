"""Expanded inference matrix using the unchanged, audited Phase 2 worker."""
import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import traceback

CODE = Path('/share1/home/fengziyu/fzycode/latte_reproduction_20261111')
PREVIOUS = CODE / 'phase2_inference_20261123_v1'
OLD = CODE / 'runs/20261123-inference-scaling-v1'
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(PREVIOUS))
import inference2 as s2
s1, FIRST = s2.s1, s2.FIRST
verify, combine_diagnostics = s2.verify, s2.combine_diagnostics


def initialize(root):
    if root.exists():
        raise FileExistsError('Preserve previous run; use a new directory')
    old, prior = s1.load(OLD / 'state.json'), s1.load(OLD / 'protocol.json')
    first = s1.load(FIRST / 'state.json')
    assert old['status'] == first['status'] == 'completed'
    assert s1.load(OLD / 'completion-audit.json')['status'] == 'passed'
    s2.verify(prior)
    s1.verify(s1.load(FIRST / 'protocol.json'))
    assert prior['evaluation_batch_size'] == 2 and prior['test_users'] == 94762
    assert prior['automatic_beam_retry'] is False
    root.mkdir(parents=True)
    points, missing = [], []
    for n in (2, 4, 8, 16):
        cp = first['cases'].get(str(n), {}).get('checkpoint')
        if not cp or not Path(cp['path']).exists():
            missing.append(dict(n=n, checkpoint=cp, reason='missing; no training authorized'))
            continue
        assert s1.sha(cp['path']) == cp['sha256']
        points.extend(dict(id='n%02d-b%04d' % (n, b), count=n, beam=b, checkpoint=cp) for b in (250, 500, 1000, 2000))
    s1.atomic(root / 'missing-checkpoints.json', missing)
    if not points:
        raise RuntimeError('No available checkpoint; no training was started')
    reused = [p['id'] for p in points if p['id'] in old['points']]
    order = [p['id'] for p in points]
    random.Random(20261123).shuffle(order)
    repeat_points = [p for p in ('n02-b0500', 'n04-b0500', 'n08-b1000', 'n16-b2000') if p in order]
    plan = dict(stage='joint16', requested_counts=[2, 4, 8, 16], counts=sorted({p['count'] for p in points}),
        beams=[250, 500, 1000, 2000], points=points, created=time.time(), seed=2024,
        training='none; frozen Phase1 epoch30 / step62160', new_training_seeds=False,
        test_users=94762, split='test', evaluation_batch_size=2, shard_size=4096,
        automatic_beam_retry=False, generate_calls_per_request=1, return_items=100,
        profile=dict(batch_size=1, users=256, repeats=3, warmup_batches=5, split='validation', exclusive_gpu=True),
        profile_device_policy='choose one exclusive RTX4090 once, save UUID, use same physical device for all 16; wait when shared',
        profile_order=order, repeat_points=repeat_points, repeat_users=256, repeat_passes=3,
        bootstrap=dict(draws=20000, seed=20261123, unit='paired test user, not training seed',
                       comparisons='adjacent beam within N and all N pairs at each beam', multiplicity='exploratory unadjusted intervals'),
        budget_scenarios=[dict(p50_ms=t, allocated_mib=m) for t, m in ((60, 512), (75, 600), (100, 1100), (150, 2100))],
        budget_minimum_full100_fractions=[0, .99, 1],
        quality_reuse=reused, reused_root=str(OLD), reused_protocol_sha256=s1.sha(OLD / 'protocol.json'),
        reused_state_sha256=s1.sha(OLD / 'state.json'),
        first_protocol_sha256=prior['first_protocol_sha256'],
        first_checkpoint_audit_sha256=prior['first_checkpoint_audit_sha256'],
        source_hashes={**prior['source_hashes'], **{str(p): s1.sha(p) for p in HERE.glob('*.py')}},
        imported_phase1_hashes=prior['imported_phase1_hashes'],
        resource_reserve_gib=dict(code=10, data=100, per_gpu_free=7),
        inference='unchanged Phase2 predict/benchmark and official agg_max; fixed actual beam; no fill or seen filter',
        test_regime='exploratory reused test population; matrix and checkpoints frozen before new evaluation',
        missing_checkpoints=missing)
    s1.atomic(root / 'protocol.json', plan)
    state = dict(tasks={}, points={p['id']: {} for p in points}, status='initialized')
    for key, entry in old['tasks'].items():
        if entry['task']['point']['id'] not in reused or entry['task']['kind'] not in ('preflight', 'eval'):
            continue
        assert entry['status'] == 'completed'
        item = copy.deepcopy(entry)
        item['origin'] = 'reused'
        out = Path(item['task']['output'])
        item['evidence_result_sha256'] = s1.sha(out / 'result.json')
        if item['task']['kind'] == 'eval':
            item['evidence_progress_sha256'] = s1.sha(out / 'progress.json')
        state['tasks'][key] = item
    for name in reused:
        state['points'][name]['test'] = copy.deepcopy(old['points'][name]['test'])
    s1.atomic(root / 'state.json', state)
    print(json.dumps(dict(status='initialized', available_points=len(points), reused_quality_points=len(reused),
                          missing_checkpoints=missing, new_quality_points=len(points) - len(reused))))


def gpu_identity(gpu):
    values = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=uuid,name',
                                      '--format=csv,noheader'], text=True).strip().split(',', 1)
    assert 'RTX 4090' in values[1]
    return dict(index=gpu, uuid=values[0].strip(), name=values[1].strip())


def worker(taskfile, gpu):
    task = s1.load(taskfile)
    root, out = Path(task['root']), Path(task['output'])
    plan = s1.load(root / 'protocol.json')
    verify(plan)
    identity = gpu_identity(gpu)
    if task['kind'] == 'profile':
        assert identity == s1.load(root / 'state.json')['profile_device']
    original_build, original_atomic = s1.build, s1.atomic

    def checked_build(*args, **kwargs):
        import torch
        ctx = original_build(*args, **kwargs)
        assert ctx['tokenizer'].base_latent_token == 1
        assert ctx['tokenizer'].n_latent_tokens == task['point']['count']
        assert all(p.dtype == torch.float32 for p in ctx['model'].parameters())
        return ctx

    def logged_atomic(path, result):
        if Path(path) == out / 'result.json':
            result['physical_gpu'] = identity
            result['driver_source_sha256'] = s1.sha(__file__)
        return original_atomic(path, result)

    s1.build, s1.atomic = checked_build, logged_atomic
    try:
        if task['kind'] != 'repeat':
            return s2.worker(taskfile, gpu)
        assert s1.sha(root / 'protocol.json') == task['protocol_sha256']
        cp = task['point']['checkpoint']
        assert s1.sha(cp['path']) == cp['sha256']
        sys.path[:0] = [str(s1.BASE), str(s1.FLOW)]
        from flow_models import prepare_runtime
        admission = prepare_runtime(gpu)
        started = time.time()
        ctx = s1.build(s1.load(FIRST / 'protocol.json'), task['point']['count'], cp['path'])
        users = list(ctx['raw'])[:plan['repeat_users']]
        reference, hashes, measurements = None, [], []
        for _ in range(plan['repeat_passes']):
            rows = []
            for start in range(0, len(users), 2):
                rows.extend(s2.predict(ctx, users[start:start + 2], 'validation', task['point']['beam'], bulk=True, diagnostic=True))
            encoded = json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()
            hashes.append(hashlib.sha256(encoded).hexdigest())
            measurements.append(s1.measure(rows))
            if reference is None:
                reference = rows
            else:
                assert rows == reference, 'Repeated inference candidates or path diagnostics changed'
        assert len(set(hashes)) == 1
        expected = plan['repeat_passes'] * len(users) // 2
        assert ctx['beam_calls'] == {str(task['point']['beam']): expected}
        out.mkdir(parents=True, exist_ok=True)
        s1.atomic(out / 'result.json', dict(status='completed', task=task, admission=admission,
            checkpoint_sha256=cp['sha256'], protocol_sha256=task['protocol_sha256'], split='validation',
            repeat_hashes=hashes, metrics_by_repeat=measurements, reference_rows=reference,
            all_candidates_and_diagnostics_equal=True, actual_generate_calls=expected, retry_calls=0,
            started=started, finished=time.time()))
    finally:
        s1.build, s1.atomic = original_build, original_atomic


# The controller below is derived from the backed-up Phase 2 controller.
# Model generation, aggregation, checkpoints and stored quality blocks stay intact.

from resource_worker import verify_policy, required_memory


def manager(root):
    verify_policy()
    import fcntl
    import shutil
    lock = (root / 'manager.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = s1.load(root / 'protocol.json')
    verify(plan)
    digest = s1.sha(root / 'protocol.json')
    statefile = root / 'state.json'
    state = s1.load(statefile) if statefile.exists() else dict(tasks={}, points={})
    live = {}
    for key, entry in state['tasks'].items():
        result = Path(entry['task']['output']) / 'result.json'
        if result.exists():
            saved = s1.load(result)
            assert saved['status'] == 'completed' and saved['task'] == entry['task']
            entry['status'] = 'completed'
        elif entry['status'] == 'running':
            cmd = Path('/proc', str(entry['pid']), 'cmdline')
            if cmd.exists() and str(root / 'tasks' / (key + '.json')).encode() in cmd.read_bytes():
                live[key] = dict(pid=entry['pid'], gpu=entry['gpu'], process=None)
            else:
                entry['status'] = 'pending'

    def add(point, kind, suffix='', **extra):
        key = point['id'] + '-' + kind + suffix
        if key not in state['tasks']:
            task = dict(id=key, root=str(root), point=point, kind=kind, protocol_sha256=digest,
                        output=str(s1.DATA / 'runs' / root.name / 'work' / key), **extra)
            s1.atomic(root / 'tasks' / (key + '.json'), task)
            state['tasks'][key] = dict(status='pending', attempts=0, task=task, origin='new')
        return key

    while True:
        for key, process in list(live.items()):
            entry = state['tasks'][key]
            out = Path(entry['task']['output'])
            native = process['process']
            ended = native.poll() is not None if native else not Path('/proc', str(process['pid'])).exists()
            if (out / 'result.json').exists():
                result = s1.load(out / 'result.json')
                assert result['status'] == 'completed' and result['task'] == entry['task']
                entry['status'] = 'completed'
            elif ended:
                error = s1.load(out / 'error.json') if (out / 'error.json').exists() else dict(error='worker exited')
                resource = error.get('error_type') == 'ResourceUnavailable' or 'PROFILE_BUSY' in error.get('error', '')
                if resource:
                    entry['attempts'] -= 1
                entry.update(status='failed' if entry['attempts'] >= 3 else 'pending', last_error=error, retry_after=time.time() + 60)
            else:
                continue
            del live[key]
        all_done = True
        for point in plan['points']:
            ps = state['points'].setdefault(point['id'], {})
            key = add(point, 'preflight')
            if state['tasks'][key]['status'] != 'completed':
                all_done = False
                continue
            keys = [add(point, 'eval', '-%05d' % start, start=start, end=min(start + plan['shard_size'], plan['test_users']))
                    for start in range(0, plan['test_users'], plan['shard_size'])]
            if 'test' not in ps and all(state['tasks'][key]['status'] == 'completed' for key in keys):
                results = [s1.load(Path(state['tasks'][key]['task']['output']) / 'result.json') for key in keys]
                assert len({(r['count'], r['beam'], r['point_id']) for r in results}) == 1
                ps['test'] = dict(**s1.combine(results, plan['test_users']), **combine_diagnostics(results, plan['test_users']))
                s1.atomic(root / (point['id'] + '-test.json'), ps['test'])
            if 'test' not in ps:
                all_done = False
                continue
            key = add(point, 'profile')
            if state['tasks'][key]['status'] == 'completed':
                ps['profile'] = s1.load(Path(state['tasks'][key]['task']['output']) / 'result.json')
            else:
                all_done = False
            if point['id'] in plan['repeat_points']:
                key = add(point, 'repeat')
                if state['tasks'][key]['status'] == 'completed':
                    ps['repeat'] = s1.load(Path(state['tasks'][key]['task']['output']) / 'result.json')
                else:
                    all_done = False
        counts = Counter(entry['status'] for entry in state['tasks'].values())
        state.update(status='experiments_completed' if all_done else 'running', updated=time.time(), counts=dict(counts))
        s1.atomic(statefile, state)
        summary = dict(status=state['status'], updated=state['updated'], counts=dict(counts), running=list(live),
                       points={key: dict(test=value.get('test'), profiled='profile' in value, repeated='repeat' in value) for key, value in state['points'].items()},
                       reused_tasks=sum(e.get('origin')=='reused' for e in state['tasks'].values()),
                       new_task_counts=dict(Counter(e['status'] for e in state['tasks'].values() if e.get('origin')!='reused')),
                       profile_device=state.get('profile_device'))
        s1.atomic(root / 'summary.json', summary)
        if all_done:
            with (root / 'analysis.log').open('a') as stream:
                completed = subprocess.run([s1.PREFIX + '/bin/python3.12', '-B', str(HERE / 'joint_report.py'), str(root)],
                    env=s1.env(''), stdout=stream, stderr=subprocess.STDOUT)
            state['status'] = 'completed' if completed.returncode == 0 else 'analysis_failed'
            s1.atomic(statefile, state)
            summary.update(status=state['status'], audit_passed=(root / 'completion-audit.json').exists())
            s1.atomic(root / 'summary.json', summary)
            break
        available_disk = shutil.disk_usage(s1.DATA).free >= 100 * 1024**3 and shutil.disk_usage(CODE).free >= 10 * 1024**3
        if not (root / 'STOP').exists() and available_disk:
            raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'], text=True)
            cards = [tuple(map(int, line.split(','))) for line in raw.splitlines()]
            occupied = {value['gpu'] for value in live.values()}
            pending = [(key, value) for key, value in state['tasks'].items()
                       if value['status'] == 'pending' and time.time() >= value.get('retry_after', 0)]
            pending.sort(key=lambda item: ({'preflight': 0, 'eval': 1, 'repeat': 2, 'profile': 3}[item[1]['task']['kind']],
                item[1]['task'].get('start', 0), plan['profile_order'].index(item[1]['task']['point']['id']) if item[1]['task']['kind']=='profile' else item[0]))
            ready_profile = all('test' in state['points'][p['id']] for p in plan['points']) and all('repeat' in state['points'][p] for p in plan['repeat_points'])
            if ready_profile and 'profile_device' not in state:
                eligible = [gpu for gpu, free in cards if free >= 7168 and gpu not in occupied and not s1.active_pids(gpu)]
                if eligible:
                    state['profile_device'] = gpu_identity(min(eligible))
                    s1.atomic(statefile, state)
            pending = [(k,e) for k,e in pending if e['task']['kind']!='profile' or ready_profile and 'profile_device' in state]
            for gpu, free in sorted(cards, key=lambda value: -value[1]):
                if gpu in occupied:
                    continue
                chosen = next(((key, value) for key, value in pending if free >= required_memory(value['task']) and (value['task']['kind'] != 'profile' or gpu == state['profile_device']['index'] and not s1.active_pids(gpu))), None)
                if chosen is None:
                    continue
                key, entry = chosen
                pending.remove(chosen)
                out = Path(entry['task']['output'])
                out.mkdir(parents=True, exist_ok=True)
                with (out / ('attempt-%d.log' % (entry['attempts'] + 1))).open('a') as stream:
                    proc = subprocess.Popen([s1.PREFIX + '/bin/python3.12', '-B', str(HERE / ('joint16.py' if entry['task']['kind']=='profile' else 'resource_worker.py')), 'worker',
                        str(root / 'tasks' / (key + '.json')), '--gpu', str(gpu)], env=s1.env(gpu),
                        stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                entry.update(status='running', attempts=entry['attempts'] + 1, pid=proc.pid, gpu=gpu, started=time.time())
                live[key] = dict(pid=proc.pid, gpu=gpu, process=proc)
            s1.atomic(statefile, state)
        if not live and any(value['status'] == 'failed' for value in state['tasks'].values()):
            state['status'] = 'needs_attention'
            s1.atomic(statefile, state)
            break
        time.sleep(15)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['init', 'manager', 'worker'])
    parser.add_argument('path', type=Path)
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()
    if args.mode == 'init':
        initialize(args.path)
    elif args.mode == 'manager':
        manager(args.path)
    else:
        try:
            worker(args.path, args.gpu)
        except BaseException as error:
            task = s1.load(args.path)
            s1.atomic(Path(task['output']) / 'error.json', dict(error_type=type(error).__name__, error=str(error),
                traceback=traceback.format_exc(), time=time.time()))
            raise
