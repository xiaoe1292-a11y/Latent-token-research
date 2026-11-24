"""Finite Latte latent-vocabulary ablation; no second-stage experiments."""
import argparse
from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from metrics import measure, combine

CODE = Path('/share1/home/fengziyu/fzycode/latte_reproduction_20261111')
DATA = Path('/share2/data/fengziyu/latte_reproduction_20261111')
BASE = CODE / 'adapter_20261119'
FLOW = CODE / 'reproflow_20261123_v7'
CACHE = DATA / 'cache/official_lastout_20261119_v1'
PROCESSED = CACHE / 'AmazonReviews2023/Video_Games/processed'
HERE = Path(__file__).resolve().parent
PREFIX = '/tmp/fengziyu-latte-python-cache-20261119-v2/prefix'


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    pending.replace(path)


def env(gpu):
    result = os.environ.copy()
    result.update(PYTHONHOME=PREFIX, PYTHONPATH=':'.join([
        '/tmp/fengziyu-latte-source-cache-20261119-v2/site-packages',
        '/tmp/fengziyu-latte-wheel-cache-20261119-v1/site-packages',
        str(CODE / 'env/lib/python3.12/site-packages')]),
        LD_LIBRARY_PATH=PREFIX + '/lib:' + str(CODE / 'env/lib'),
        PYTHONPYCACHEPREFIX='/tmp/fengziyu-latte-pycache-20261119',
        TMPDIR='/tmp/fengziyu-latte-runtime-20261119', CUBLAS_WORKSPACE_CONFIG=':4096:8',
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
        HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        WANDB_MODE='disabled', TOKENIZERS_PARALLELISM='false',
        CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1')
    return result


def initialize(root):
    if root.exists():
        raise FileExistsError('New run requires a new directory')
    old = load(CODE / 'runs/20261120-reproflow-v2/state.json')
    cp = old['cases']['Latte-s2024']['checkpoints']['30']
    plan = dict(stage=1, created=time.time(), latent_counts=[1, 2, 4, 8, 16], seed=2024,
                epochs=30, steps=62160, scheduler_horizon_steps=310800, warmup_steps=10000,
                batch_size=256, history=20, beam=500, upstream_retry_beam=1000,
                return_items=100, aggregation='agg_max', precision='float32',
                validation_users=4096, test_users=94762, shard_size=4096,
                evaluation_batch_size=4, seen_filter=False, popularity_fill=False,
                checkpoint_rule='fixed epoch30; all configurations frozen before any test',
                profile=dict(batch_sizes=[1, 4], warmup_batches=5, measured_users=256,
                             repeats=3, split='validation', exclusive_gpu_required=True),
                reference=dict(path=cp, sha256=sha(cp), n_latent_tokens=8, epoch=30),
                scope='Latte reproduction and latent vocabulary ablation only; stage2 requires user confirmation',
                upstream_commit='05e4e6d983225bcb7172f148a076890e80c524d1',
                source_hashes={str(p): sha(p) for p in sorted((CODE / 'source/genrec').rglob('*.py'))},
                helper_hashes={str(BASE / n): sha(BASE / n) for n in
                    ['train_bounded.py', 'profile_full_models.py', 'training_runtime.py', 'run_controlled.py']},
                execution_hashes={str(p): sha(p) for p in HERE.glob('*.py')})
    root.mkdir(parents=True)
    atomic(root / 'protocol.json', plan)
    return plan


def verify(plan):
    for group in ('source_hashes', 'helper_hashes', 'execution_hashes'):
        for path, digest in plan[group].items():
            assert sha(path) == digest, 'Frozen source changed: ' + path
    assert sha(plan['reference']['path']) == plan['reference']['sha256']


def build(plan, count, checkpoint=None, training=False):
    sys.path[:0] = [str(BASE), str(CODE / 'source'), str(FLOW)]
    import torch
    from accelerate import Accelerator
    from genrec.utils import get_config
    from genrec.datasets.AmazonReviews2023.dataset import AmazonReviews2023
    from genrec.models.Latte.model import Latte
    from genrec.models.Latte.tokenizer import LatteTokenizer
    from flow_models import seed_all
    from training_runtime import ShuffledBatches, restore_state
    verify(plan)
    seed_all(plan['seed'])
    state = torch.load(checkpoint, map_location='cpu', weights_only=False) if checkpoint else None
    template = state or torch.load(plan['reference']['path'], map_location='cpu', weights_only=False)
    provenance = template['provenance']
    config = dict(provenance['training_config'])
    config.update(n_latent_tokens=count, rand_seed=plan['seed'])
    config = get_config('Latte', 'AmazonReviews2023', None, config)
    config.update(device=torch.device('cuda'), use_ddp=False, accelerator=Accelerator())
    for name, digest in load(CACHE / 'protocol-audit.json')['cache_hashes'].items():
        assert sha(PROCESSED / name) == digest
    sid = PROCESSED / 'sentence-t5-base_rqkmeans_3x256_psid.sem_ids'
    assert sha(sid) == provenance['sid_sha256']
    data = AmazonReviews2023(config)
    tokenizer = LatteTokenizer(config, data)
    model = Latte(config, data, tokenizer).cuda()
    expected = dict(provenance, training_config=dict(provenance['training_config'], n_latent_tokens=count))
    assert tokenizer.n_latent_tokens == count
    assert len(tokenizer.item2tokens) == len(tokenizer.tokens_to_item) == 25612
    assert tokenizer.base_sem_token == count + 1
    if state:
        assert state['provenance'] == expected, 'Checkpoint latent count/data/training mismatch'
        model.load_state_dict(state['model'], strict=True)
    ctx = dict(model=model, tokenizer=tokenizer, raw=load(PROCESSED / 'all_item_seqs.json'),
               provenance=expected, step=0, count=count)
    if training:
        from transformers.optimization import get_scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        scheduler = get_scheduler('cosine', optimizer, num_warmup_steps=10000, num_training_steps=310800)
        sampler = ShuffledBatches(530300, 256, plan['seed'])
        if state:
            sampler, ctx['step'] = restore_state(state, model, optimizer, scheduler, expected)
        ctx.update(optimizer=optimizer, scheduler=scheduler, sampler=sampler)
    del template, state
    # Inference top-k changes neither weights nor the training provenance.
    model.config.update(topk=[10, 50, 100], num_beams=plan['beam'])
    return ctx


def save(ctx, path):
    import torch
    from training_runtime import capture_state
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = capture_state(ctx['model'], ctx['optimizer'], ctx['scheduler'], ctx['sampler'],
                          ctx['step'], ctx['provenance'], cuda=True)
    pending = path.with_suffix('.pending')
    torch.save(state, pending)
    pending.replace(path)
    return dict(path=str(path), sha256=sha(path), step=ctx['step'])


def predict(ctx, users, split):
    import torch
    from profile_full_models import make_batch
    examples = [dict(user=u, item_seq=ctx['raw'][u][:-1] if split == 'validation' else ctx['raw'][u]) for u in users]
    batch = make_batch(ctx['tokenizer'], examples, torch.device('cuda'), False)
    model = ctx['model']
    model.eval()
    original = model.t5.generate
    def counted(*args, **kwargs):
        ctx.setdefault('beam_calls', Counter())[str(kwargs['num_beams'])] += 1
        return original(*args, **kwargs)
    model.t5.generate = counted
    try:
        with torch.inference_mode():
            codes = model.generate(batch, n_return_sequences=100).cpu().tolist()
    finally:
        model.t5.generate = original
    inverse = ctx['tokenizer'].tokens_to_item
    rows = []
    for user, example, values in zip(users, examples, codes):
        items = [inverse.get(tuple(code)) for code in values]
        target = example['item_seq'][-1]
        rank = next((i + 1 for i, item in enumerate(items) if item == target), None)
        rows.append(dict(user=user, target=target, candidate_slots=items, rank=rank))
    return rows


def active_pids(gpu):
    output = subprocess.check_output(['nvidia-smi', '--id=' + str(gpu), '--query-compute-apps=pid',
                                      '--format=csv,noheader,nounits'], text=True)
    return [int(x.strip()) for x in output.splitlines() if x.strip().isdigit()]


def profile(ctx, plan, gpu):
    import torch
    import numpy as np
    # Measurements include input construction, H2D, generation, aggregation and item mapping.
    if any(pid != os.getpid() for pid in active_pids(gpu)):
        raise RuntimeError('PROFILE_BUSY: another compute process uses this GPU')
    users = list(ctx['raw'])[:plan['profile']['measured_users']]
    results = []
    for batch in plan['profile']['batch_sizes']:
        for i in range(plan['profile']['warmup_batches']):
            predict(ctx, users[i * batch:(i + 1) * batch], 'validation')
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        timings = []
        calls_before = dict(ctx.get('beam_calls', {}))
        for repeat in range(plan['profile']['repeats']):
            for start in range(0, len(users), batch):
                torch.cuda.synchronize()
                begun = time.perf_counter()
                predict(ctx, users[start:start + batch], 'validation')
                torch.cuda.synchronize()
                timings.append(time.perf_counter() - begun)
                if any(pid != os.getpid() for pid in active_pids(gpu)):
                    raise RuntimeError('PROFILE_BUSY: GPU became shared during measurement')
        results.append(dict(batch_size=batch, users=len(users), repeats=plan['profile']['repeats'],
            batches=len(timings), latency_batch_mean_ms=float(np.mean(timings) * 1000),
            latency_batch_p50_ms=float(np.percentile(timings, 50) * 1000),
            latency_batch_p95_ms=float(np.percentile(timings, 95) * 1000),
            throughput_users_per_second=len(users) * plan['profile']['repeats'] / sum(timings),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(), model_baseline_allocated_bytes=before,
            beam_calls={k: v - calls_before.get(k, 0) for k, v in ctx['beam_calls'].items()},
            synchronized_batch_seconds=timings))
    return dict(measurements=results, parameters=sum(p.numel() for p in ctx['model'].parameters()),
                torch_version=torch.__version__, cuda_version=torch.version.cuda,
                device=torch.cuda.get_device_name(0), precision='float32', timed_scope='request preprocessing through item candidates',
                memory_scope='PyTorch allocator; excludes CUDA context and other processes', exclusive_gpu=True)


def worker(taskfile, gpu):
    task = load(taskfile)
    root, out = Path(task['root']), Path(task['output'])
    plan = load(root / 'protocol.json')
    assert task['protocol_sha256'] == sha(root / 'protocol.json')
    out.mkdir(parents=True, exist_ok=True)
    begun = time.time()
    sys.path[:0] = [str(BASE), str(FLOW)]
    from flow_models import prepare_runtime
    admission = prepare_runtime(gpu)
    stop = []
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
        signal.signal(sig, lambda number, frame: stop.append(number))
    signal.alarm(7200)
    kind, count = task['kind'], task['count']
    progressfile = out / 'progress.json'
    progress = load(progressfile) if progressfile.exists() else {}
    checkpoint = progress.get('checkpoint', task.get('checkpoint'))
    if isinstance(checkpoint, dict):
        assert sha(checkpoint['path']) == checkpoint['sha256']
        checkpoint = checkpoint['path']
    if kind in ('eval', 'profile'):
        assert sha(checkpoint) == task['checkpoint_sha256']
    ctx = build(plan, count, checkpoint, kind in ('train', 'preflight'))
    import torch
    if kind == 'preflight':
        from flow_models import update
        from profile_full_models import make_batch
        from training_runtime import capture_state, restore_state
        ctx.update(case=dict(family='gen'))
        before = capture_state(ctx['model'], ctx['optimizer'], ctx['scheduler'], ctx['sampler'], 0, ctx['provenance'], cuda=True)
        samples = [dict(user=u, item_seq=s[:-2]) for u, s in list(ctx['raw'].items())[:256]]
        labels = make_batch(ctx['tokenizer'], samples, torch.device('cuda'), True)['labels'].cpu()
        assert labels.shape == (256, 5)
        assert int(labels[:, 0].min()) >= 1 and int(labels[:, 0].max()) <= count
        assert len(set(labels[:, 0].tolist())) == count
        for row, sample in zip(labels.tolist(), samples):
            assert row[1:4] == list(ctx['tokenizer'].item2tokens[sample['item_seq'][-1]])
            assert row[-1] == ctx['tokenizer'].eos_token
        path = out / 'before.pt'
        torch.save(before, path)
        before = torch.load(path, map_location='cpu', weights_only=False)
        ctx['sampler'], ctx['step'] = restore_state(before, ctx['model'], ctx['optimizer'], ctx['scheduler'], ctx['provenance'])
        first = update(ctx)
        expected = {k: v.detach().cpu().clone() for k, v in ctx['model'].state_dict().items()}
        ctx['sampler'], ctx['step'] = restore_state(before, ctx['model'], ctx['optimizer'], ctx['scheduler'], ctx['provenance'])
        second = update(ctx)
        assert first['loss'] == second['loss']
        assert all(torch.equal(v, ctx['model'].state_dict()[k].cpu()) for k, v in expected.items())
        result = dict(loss=first['loss'], actual_gpu_replay=True, bitwise_parameter_equal=True,
                      checkpoint_roundtrip_sha256=sha(path), labels_verified=256,
                      vocabulary=ctx['tokenizer'].vocab_size, latent_count=count)
    elif kind == 'train':
        from flow_models import update
        ctx.update(case=dict(family='gen'))
        cpdir = DATA / 'runs' / root.name / 'checkpoints' / ('n' + str(count))
        with (out / 'training.jsonl').open('a') as stream:
            while ctx['step'] < plan['steps'] and not stop:
                values = update(ctx)
                stream.write(json.dumps(dict(step=ctx['step'], **values)) + '\n')
                if ctx['step'] % 100 == 0:
                    stream.flush()
                    atomic(progressfile, dict(step=ctx['step'], checkpoint=progress.get('checkpoint'), time=time.time()))
                if ctx['step'] % 2072 == 0 or ctx['step'] % 1000 == 0:
                    saved = save(ctx, cpdir / ('epoch-%03d.pt' % (ctx['step'] // 2072) if ctx['step'] % 2072 == 0 else 'recovery.pt'))
                    progress = dict(step=ctx['step'], checkpoint=saved, time=time.time())
                    atomic(progressfile, progress)
        if ctx['step'] != plan['steps']:
            atomic(progressfile, dict(checkpoint=save(ctx, cpdir / 'interrupted.pt'), step=ctx['step']))
            raise InterruptedError('Checkpoint saved; resume required')
        result = dict(checkpoint=progress['checkpoint'], end_step=ctx['step'], epoch=30)
    elif kind == 'eval':
        users = list(ctx['raw'])[task['start']:task['end']]
        parts = progress.get('parts', [])
        position = progress.get('committed', 0)
        while position < len(users) and not stop:
            end = min(position + 256, len(users))
            rows = []
            for offset in range(position, end, 4):
                rows.extend(predict(ctx, users[offset:min(offset + 4, end)], task['split']))
            measure(rows)
            part = out / ('rows-%05d-%05d.json' % (position, end))
            atomic(part, rows)
            parts.append(dict(path=str(part), sha256=sha(part)))
            position = end
            atomic(progressfile, dict(committed=position, parts=parts, checkpoint_sha256=task['checkpoint_sha256']))
        if position != len(users):
            raise InterruptedError('Evaluation blocks retained; resume required')
        rows = []
        for part in parts:
            assert sha(part['path']) == part['sha256']
            rows.extend(load(part['path']))
        assert [r['user'] for r in rows] == users
        result = dict(start=task['start'], end=task['end'], split=task['split'],
                      checkpoint_sha256=task['checkpoint_sha256'], protocol_sha256=task['protocol_sha256'],
                      actual_beam_calls_this_attempt=dict(ctx.get('beam_calls', {})), **measure(rows))
    elif kind == 'profile':
        result = profile(ctx, plan, gpu)
    else:
        raise ValueError(kind)
    result.update(status='completed', task=task, admission=admission, started=begun, finished=time.time(),
                  elapsed_seconds=time.time() - begun)
    atomic(out / 'result.json', result)


def manager(root):
    import fcntl
    import shutil
    lock = (root / 'manager.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = load(root / 'protocol.json')
    verify(plan)
    statefile = root / 'state.json'
    state = load(statefile) if statefile.exists() else dict(tasks={}, cases={})
    digest = sha(root / 'protocol.json')
    live = {}
    for key, entry in state['tasks'].items():
        if entry['status'] == 'running':
            proc = Path('/proc') / str(entry['pid']) / 'cmdline'
            if proc.exists() and str(root / 'tasks' / (key + '.json')).encode() in proc.read_bytes():
                live[key] = dict(pid=entry['pid'], gpu=entry['gpu'], process=None)
            else:
                entry['status'] = 'pending'

    def add(count, kind, suffix='', **extra):
        key = 'n%d-%s%s' % (count, kind, suffix)
        if key not in state['tasks']:
            task = dict(id=key, root=str(root), kind=kind, count=count, protocol_sha256=digest,
                        output=str(DATA / 'runs' / root.name / 'work' / key), **extra)
            atomic(root / 'tasks' / (key + '.json'), task)
            state['tasks'][key] = dict(status='pending', attempts=0, task=task)
        return key

    while True:
        for key, proc in list(live.items()):
            entry = state['tasks'][key]
            out = Path(entry['task']['output'])
            result = out / 'result.json'
            process = proc['process']
            ended = process.poll() is not None if process else not Path('/proc', str(proc['pid'])).exists()
            if result.exists():
                entry['status'] = 'completed'
            elif ended:
                error = load(out / 'error.json') if (out / 'error.json').exists() else dict(error='worker exited')
                busy = 'PROFILE_BUSY' in error.get('error', '') or error.get('error_type') == 'ResourceUnavailable'
                if busy:
                    entry['attempts'] -= 1
                entry.update(status='failed' if entry['attempts'] >= 3 else 'pending', last_error=error, retry_after=time.time() + 60)
            else:
                continue
            del live[key]
        for count in plan['latent_counts']:
            case = state['cases'].setdefault(str(count), {})
            pre = add(count, 'preflight')
            if state['tasks'][pre]['status'] != 'completed':
                continue
            if 'checkpoint' not in case:
                if count == 8:
                    case['checkpoint'] = plan['reference']
                else:
                    train = add(count, 'train')
                    if state['tasks'][train]['status'] != 'completed':
                        continue
                    case['checkpoint'] = load(Path(state['tasks'][train]['task']['output']) / 'result.json')['checkpoint']
                atomic(root / ('frozen-n%d.json' % count), dict(rule=plan['checkpoint_rule'], checkpoint=case['checkpoint'], protocol_sha256=digest))
            cp = case['checkpoint']
            for split, users in [('validation', plan['validation_users']), ('test', plan['test_users'])]:
                keys = [add(count, 'eval', '-%s-%05d' % (split, start), split=split, start=start,
                            end=min(users, start + plan['shard_size']), checkpoint=cp['path'], checkpoint_sha256=cp['sha256'])
                        for start in range(0, users, plan['shard_size'])]
                if split not in case and all(state['tasks'][key]['status'] == 'completed' for key in keys):
                    case[split] = combine([load(Path(state['tasks'][key]['task']['output']) / 'result.json') for key in keys], users)
                    atomic(root / ('n%d-%s.json' % (count, split)), case[split])
            # Profile after evaluation, on a GPU with no other compute process.
            if 'test' in case and 'validation' in case:
                key = add(count, 'profile', checkpoint=cp['path'], checkpoint_sha256=cp['sha256'])
                if state['tasks'][key]['status'] == 'completed':
                    case['profile'] = load(Path(state['tasks'][key]['task']['output']) / 'result.json')
        complete = all('profile' in state['cases'].get(str(n), {}) for n in plan['latent_counts'])
        counts = Counter(entry['status'] for entry in state['tasks'].values())
        state.update(status='completed' if complete else 'running', updated=time.time(), counts=dict(counts))
        atomic(statefile, state)
        atomic(root / 'summary.json', dict(status=state['status'], counts=dict(counts), running=list(live),
                    cases={k: dict(checkpoint=v.get('checkpoint'), validation=v.get('validation'), test=v.get('test'),
                                   profiled='profile' in v) for k, v in state['cases'].items()}))
        if complete:
            with (root / 'analysis.log').open('a') as stream:
                result = subprocess.run([PREFIX + '/bin/python3.12', '-B', str(HERE / 'report.py'), str(root)],
                    env=env(''), stdout=stream, stderr=subprocess.STDOUT)
            state['status'] = 'completed' if result.returncode == 0 else 'analysis_failed'
            atomic(statefile, state)
            atomic(root / 'summary.json', dict(status=state['status'], counts=dict(counts),
                        report=str(root / 'REPORT.md'), audit_passed=(root / 'completion-audit.json').exists()))
            break
        if not (root / 'STOP').exists() and shutil.disk_usage(DATA).free > 100 * 1024**3:
            output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'], text=True)
            cards = [tuple(map(int, row.split(','))) for row in output.splitlines()]
            occupied = {p['gpu'] for p in live.values()}
            pending = [(k, v) for k, v in state['tasks'].items() if v['status'] == 'pending' and time.time() >= v.get('retry_after', 0)]
            # Prioritize fresh training before large test shards.
            pending.sort(key=lambda kv: ({'preflight': 0, 'train': 1, 'eval': 2, 'profile': 3}[kv[1]['task']['kind']], kv[0]))
            for gpu, free in sorted(cards, key=lambda x: -x[1]):
                if gpu in occupied or free < 7168:
                    continue
                choice = next(((k, v) for k, v in pending if v['task']['kind'] != 'profile' or not active_pids(gpu)), None)
                if choice is None:
                    continue
                key, entry = choice
                pending.remove(choice)
                out = Path(entry['task']['output'])
                out.mkdir(parents=True, exist_ok=True)
                with (out / ('attempt-%d.log' % (entry['attempts'] + 1))).open('a') as stream:
                    proc = subprocess.Popen([PREFIX + '/bin/python3.12', '-B', str(HERE / 'stage1.py'), 'worker',
                        str(root / 'tasks' / (key + '.json')), '--gpu', str(gpu)], env=env(gpu), stdin=subprocess.DEVNULL,
                        stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                entry.update(status='running', attempts=entry['attempts'] + 1, pid=proc.pid, gpu=gpu, started=time.time())
                live[key] = dict(pid=proc.pid, gpu=gpu, process=proc)
            atomic(statefile, state)
        if not live and any(v['status'] == 'failed' for v in state['tasks'].values()):
            state['status'] = 'needs_attention'
            atomic(statefile, state)
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
            task = load(args.path)
            atomic(Path(task['output']) / 'error.json', dict(error_type=type(error).__name__, error=str(error),
                       traceback=traceback.format_exc(), time=time.time()))
            raise
