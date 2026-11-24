"""Finite fixed-beam verification on frozen stage-one Latte checkpoints."""
import argparse
from collections import Counter
import gc
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

CODE = Path('/share1/home/fengziyu/fzycode/latte_reproduction_20261111')
PHASE1 = CODE / 'phase1_20261123_v1'
FIRST = CODE / 'runs/20261123-latent-ablation-v1'
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(PHASE1))
import stage1 as s1
from diagnostics import inspect_paths, summarize, combine as combine_diagnostics


def initialize(root):
    if root.exists():
        raise FileExistsError('Use a fresh output directory')
    first = s1.load(FIRST / 'state.json')
    assert first['status'] == 'completed'
    assert s1.load(FIRST / 'final-checkpoint-audit.json')['status'] == 'passed'
    plan1 = s1.load(FIRST / 'protocol.json')
    s1.verify(plan1)
    points = [dict(id='n%02d-b%04d' % (n, beam), count=n, beam=beam,
                   checkpoint=first['cases'][str(n)]['checkpoint'])
              for n in (4, 8, 16) for beam in (500, 1000, 2000)]
    plan = dict(stage=2, direction='Inference Scaling and decoded-path diagnostics', created=time.time(),
        points=points, counts=[4, 8, 16], beams=[500, 1000, 2000], seed=2024,
        training='none; frozen stage1 epoch30 / step62160', new_training_seeds=False,
        automatic_beam_retry=False, generate_calls_per_request=1, return_items=100,
        evaluation_batch_size=2, test_users=94762, split='test', shard_size=4096,
        profile=dict(batch_size=1, users=256, repeats=3, warmup_batches=5, split='validation', exclusive_gpu=True),
        first_protocol_sha256=s1.sha(FIRST / 'protocol.json'),
        first_checkpoint_audit_sha256=s1.sha(FIRST / 'final-checkpoint-audit.json'),
        diagnostics='decoded beam path counts; effective latent count is count entropy, not posterior entropy',
        test_regime='exploratory reuse of stage1 full test population; matrix frozen before this stage; no checkpoint selection',
        nominal_path_budget='B/n is a proxy; beam allocation across latent types is not constrained to be equal',
        source_hashes={str(p): s1.sha(p) for p in HERE.glob('*.py')},
        imported_phase1_hashes={str(PHASE1 / name): s1.sha(PHASE1 / name) for name in ['stage1.py', 'metrics.py']})
    root.mkdir(parents=True)
    s1.atomic(root / 'protocol.json', plan)
    return plan


def verify(plan):
    for group in ('source_hashes', 'imported_phase1_hashes'):
        for path, digest in plan[group].items():
            assert s1.sha(path) == digest, 'Frozen source changed: ' + path
    assert s1.sha(FIRST / 'protocol.json') == plan['first_protocol_sha256']
    assert s1.sha(FIRST / 'final-checkpoint-audit.json') == plan['first_checkpoint_audit_sha256']


def predict(ctx, users, split, beam, bulk=False, diagnostic=False):
    import torch
    from profile_full_models import make_batch
    model = ctx['model']
    model.eval()
    examples = [dict(user=u, item_seq=ctx['raw'][u][:-1] if split == 'validation' else ctx['raw'][u]) for u in users]
    batch = make_batch(ctx['tokenizer'], examples, torch.device('cuda'), False)
    inverse = ctx['tokenizer'].tokens_to_item
    generate, aggregate = model.t5.generate, model._aggregate_predictions
    calls, details = [], []

    def counted(*args, **kwargs):
        assert kwargs['num_beams'] == kwargs['num_return_sequences'] == beam
        calls.append(beam)
        ctx.setdefault('beam_calls', Counter())[str(beam)] += 1
        return generate(*args, **kwargs)

    def cpu_aggregate(tokens, scores, k):
        device = tokens.device
        cpu_tokens, cpu_scores = tokens.detach().cpu(), scores.detach().cpu()
        if diagnostic:
            assert cpu_tokens.shape[1] == beam
            raw_paths = cpu_tokens.tolist()
            details.extend(inspect_paths(row, ctx['count'], inverse) for row in raw_paths)
            if ctx.get('capture_raw_paths'):
                ctx.setdefault('raw_path_rows', []).extend(raw_paths)
        return aggregate(cpu_tokens, cpu_scores, k).to(device)

    model.t5.generate = counted
    if bulk:
        model._aggregate_predictions = cpu_aggregate
    try:
        with torch.inference_mode():
            # Exactly one official generation/aggregation helper call; bypass only the retry wrapper.
            codes = model._generate_and_aggregate(batch, beam, 100, ctx['tokenizer'].n_digit).cpu().tolist()
    finally:
        model.t5.generate, model._aggregate_predictions = generate, aggregate
    assert calls == [beam], 'A fixed-beam request must never silently retry'
    assert not diagnostic or len(details) == len(users)
    rows = []
    for index, (user, example, values) in enumerate(zip(users, examples, codes)):
        items = [inverse.get(tuple(code)) for code in values]
        target = example['item_seq'][-1]
        rank = next((i + 1 for i, item in enumerate(items) if item == target), None)
        row = dict(user=user, target=target, candidate_slots=items, rank=rank)
        if diagnostic:
            row['path_diagnostics'] = details[index]
        rows.append(row)
    return rows


def benchmark(ctx, plan, gpu, beam):
    import numpy as np
    import torch
    spec = plan['profile']
    if any(pid != os.getpid() for pid in s1.active_pids(gpu)):
        raise RuntimeError('PROFILE_BUSY: another compute process is present')
    users = list(ctx['raw'])[:spec['users']]
    for i in range(spec['warmup_batches']):
        predict(ctx, users[i:i + 1], 'validation', beam)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    before = ctx.get('beam_calls', {}).get(str(beam), 0)
    times = []
    for repeat in range(spec['repeats']):
        for user in users:
            torch.cuda.synchronize()
            started = time.perf_counter()
            predict(ctx, [user], 'validation', beam)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - started)
            if any(pid != os.getpid() for pid in s1.active_pids(gpu)):
                raise RuntimeError('PROFILE_BUSY: device became shared')
    calls = ctx['beam_calls'][str(beam)] - before
    assert calls == len(times) == spec['users'] * spec['repeats']
    return dict(batch_size=1, users=spec['users'], repeats=spec['repeats'], measured_calls=calls,
        actual_beam=beam, retry_calls=0, timed_batch_seconds=times,
        latency_mean_ms=float(np.mean(times) * 1000), latency_p50_ms=float(np.percentile(times, 50) * 1000),
        latency_p95_ms=float(np.percentile(times, 95) * 1000), throughput_users_per_second=calls / sum(times),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        baseline_allocated_bytes=baseline, exclusive_gpu=True, precision='float32',
        torch_version=torch.__version__, cuda_version=torch.version.cuda, gpu_name=torch.cuda.get_device_name(),
        timed_scope='preprocessing, H2D, one official fixed-beam generation, GPU scalar aggregation, ASIN decoding',
        memory_scope='PyTorch allocator; excludes CUDA context', parameters=sum(p.numel() for p in ctx['model'].parameters()))


def worker(taskfile, gpu):
    task = s1.load(taskfile)
    root, out = Path(task['root']), Path(task['output'])
    plan = s1.load(root / 'protocol.json')
    verify(plan)
    assert s1.sha(root / 'protocol.json') == task['protocol_sha256']
    point = task['point']
    cp = point['checkpoint']
    assert s1.sha(cp['path']) == cp['sha256']
    out.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(s1.BASE), str(s1.FLOW)]
    from flow_models import prepare_runtime
    admission = prepare_runtime(gpu)
    ctx = s1.build(s1.load(FIRST / 'protocol.json'), point['count'], cp['path'])
    beam, n = point['beam'], point['count']
    stop, started = [], time.time()
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM):
        signal.signal(sig, lambda number, frame: stop.append(number))
    signal.alarm(7200)
    if task['kind'] == 'preflight':
        long = [u for u, seq in ctx['raw'].items() if len(seq) >= 24][:2]
        short = [u for u, seq in ctx['raw'].items() if len(seq) == 5][:2]
        assert len(long) == len(short) == 2
        users = long + short
        ctx['capture_raw_paths'] = True
        reference, actual = [], []
        for start in (0, 2):
            reference.extend(predict(ctx, users[start:start + 2], 'validation', beam))
            actual.extend(predict(ctx, users[start:start + 2], 'validation', beam, bulk=True, diagnostic=True))
        assert reference == [{k: v for k, v in row.items() if k != 'path_diagnostics'} for row in actual]
        assert any(item is not None for row in actual for item in row['candidate_slots'])
        assert ctx['beam_calls'] == {str(beam): 4}
        summarize(actual, n, beam)
        for row, paths in zip(actual, ctx['raw_path_rows']):
            row['raw_beam_paths'] = paths
        token_map = out / 'token-map.json'
        s1.atomic(token_map, [dict(item=item, tokens=list(tokens)) for item, tokens in sorted(ctx['tokenizer'].item2tokens.items())])
        result = dict(actual_gpu_parity=True, official_fixed_beam_candidate_equality=True,
                      one_generation_per_batch=True, long_history_users=2, short_history_users=2, rows=actual,
                      token_map=dict(path=str(token_map), sha256=s1.sha(token_map)),
                      peak_allocated_bytes=__import__('torch').cuda.max_memory_allocated())
    elif task['kind'] == 'eval':
        users = list(ctx['raw'])[task['start']:task['end']]
        progressfile = out / 'progress.json'
        progress = s1.load(progressfile) if progressfile.exists() else dict(committed=0, parts=[],
            checkpoint_sha256=cp['sha256'], protocol_sha256=task['protocol_sha256'], point_id=point['id'])
        assert progress['checkpoint_sha256'] == cp['sha256'] and progress['protocol_sha256'] == task['protocol_sha256']
        assert progress['point_id'] == point['id']
        position = progress['committed']
        while position < len(users) and not stop:
            end, rows = min(position + 256, len(users)), []
            for offset in range(position, end, 2):
                rows.extend(predict(ctx, users[offset:min(offset + 2, end)], 'test', beam, bulk=True, diagnostic=True))
            s1.measure(rows)
            summarize(rows, n, beam)
            path = out / ('rows-%05d-%05d.json' % (position, end))
            s1.atomic(path, rows)
            progress['parts'].append(dict(path=str(path), sha256=s1.sha(path)))
            position = end
            progress.update(committed=position, time=time.time())
            s1.atomic(progressfile, progress)
        if position != len(users):
            raise InterruptedError('Committed evaluation blocks retained for resumption')
        rows = []
        for part in progress['parts']:
            assert s1.sha(part['path']) == part['sha256']
            rows.extend(s1.load(part['path']))
        assert [row['user'] for row in rows] == users
        result = dict(start=task['start'], end=task['end'], split='test', point_id=point['id'], count=n, beam=beam,
            checkpoint_sha256=cp['sha256'], protocol_sha256=task['protocol_sha256'],
            actual_generate_calls_this_attempt=dict(ctx.get('beam_calls', {})), **s1.measure(rows), **summarize(rows, n, beam))
    elif task['kind'] == 'profile':
        result = benchmark(ctx, plan, gpu, beam)
    else:
        raise ValueError(task['kind'])
    result.update(status='completed', task=task, admission=admission, started=started, finished=time.time(),
                  elapsed_seconds=time.time() - started)
    s1.atomic(out / 'result.json', result)


def manager(root):
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
            state['tasks'][key] = dict(status='pending', attempts=0, task=task)
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
        counts = Counter(entry['status'] for entry in state['tasks'].values())
        state.update(status='experiments_completed' if all_done else 'running', updated=time.time(), counts=dict(counts))
        s1.atomic(statefile, state)
        summary = dict(status=state['status'], updated=state['updated'], counts=dict(counts), running=list(live),
                       points={key: dict(test=value.get('test'), profiled='profile' in value) for key, value in state['points'].items()})
        s1.atomic(root / 'summary.json', summary)
        if all_done:
            with (root / 'analysis.log').open('a') as stream:
                completed = subprocess.run([s1.PREFIX + '/bin/python3.12', '-B', str(HERE / 'report2.py'), str(root)],
                    env=s1.env(''), stdout=stream, stderr=subprocess.STDOUT)
            state['status'] = 'completed' if completed.returncode == 0 else 'analysis_failed'
            s1.atomic(statefile, state)
            summary.update(status=state['status'], audit_passed=(root / 'completion-audit.json').exists())
            s1.atomic(root / 'summary.json', summary)
            break
        available_disk = shutil.disk_usage(s1.DATA).free >= 100 * 1024**3 and shutil.disk_usage(CODE).free >= 20 * 1024**3
        if not (root / 'STOP').exists() and available_disk:
            raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'], text=True)
            cards = [tuple(map(int, line.split(','))) for line in raw.splitlines()]
            occupied = {value['gpu'] for value in live.values()}
            pending = [(key, value) for key, value in state['tasks'].items()
                       if value['status'] == 'pending' and time.time() >= value.get('retry_after', 0)]
            pending.sort(key=lambda item: ({'preflight': 0, 'eval': 1, 'profile': 2}[item[1]['task']['kind']],
                                           item[1]['task'].get('start', 0), item[0]))
            for gpu, free in sorted(cards, key=lambda value: -value[1]):
                if gpu in occupied or free < 7168:
                    continue
                chosen = next(((key, value) for key, value in pending if value['task']['kind'] != 'profile' or not s1.active_pids(gpu)), None)
                if chosen is None:
                    continue
                key, entry = chosen
                pending.remove(chosen)
                out = Path(entry['task']['output'])
                out.mkdir(parents=True, exist_ok=True)
                with (out / ('attempt-%d.log' % (entry['attempts'] + 1))).open('a') as stream:
                    proc = subprocess.Popen([s1.PREFIX + '/bin/python3.12', '-B', str(HERE / 'inference2.py'), 'worker',
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
