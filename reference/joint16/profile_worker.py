"""Uniform same-card measurements with explicit shared-server telemetry."""
import argparse
import gc
import os
from pathlib import Path
import subprocess
import time
import traceback

import joint16 as original

HERE = Path(__file__).resolve().parent


def verify_cost_policy():
    policy = original.s1.load(HERE / 'cost-policy-v1.json')
    assert policy['allow_shared_observational_cost']
    for name, digest in policy['sources'].items():
        assert original.s1.sha(HERE / name) == digest
    return policy


def background(gpu):
    return sorted(pid for pid in original.s1.active_pids(gpu) if pid != os.getpid())


def utilization(gpu):
    return dict(time=time.time(), raw=subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-gpu=utilization.gpu,memory.free,clocks.sm,temperature.gpu', '--format=csv,noheader,nounits'], text=True).strip())


def worker(path, gpu):
    import numpy as np
    import torch
    task = original.s1.load(path)
    assert task['kind'] == 'profile'
    root, out = Path(task['root']), Path(task['output'])
    plan = original.s1.load(root / 'protocol.json')
    original.verify(plan)
    policy = verify_cost_policy()
    assert original.s1.sha(root / 'protocol.json') == policy['original_protocol_sha256'] == task['protocol_sha256']
    point, identity = task['point'], original.gpu_identity(gpu)
    assert identity == original.s1.load(root / 'state.json')['profile_device']
    assert original.s1.sha(point['checkpoint']['path']) == point['checkpoint']['sha256']
    import sys
    sys.path[:0] = [str(original.s1.BASE), str(original.s1.FLOW)]
    from flow_models import prepare_runtime
    admission = prepare_runtime(gpu)
    started = time.time()
    ctx = original.s1.build(original.s1.load(original.FIRST / 'protocol.json'), point['count'], point['checkpoint']['path'])
    assert all(p.dtype == torch.float32 for p in ctx['model'].parameters())
    assert ctx['tokenizer'].n_latent_tokens == point['count']
    spec, beam = plan['profile'], point['beam']
    users = list(ctx['raw'])[:spec['users']]
    initial = background(gpu)
    telemetry = [utilization(gpu)]
    for i in range(spec['warmup_batches']):
        original.s2.predict(ctx, users[i:i + 1], 'validation', beam)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    before = ctx.get('beam_calls', {}).get(str(beam), 0)
    times, observed = [], []
    for repeat in range(spec['repeats']):
        for user in users:
            torch.cuda.synchronize()
            begin = time.perf_counter()
            original.s2.predict(ctx, [user], 'validation', beam)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - begin)
            observed.append(background(gpu))
        telemetry.append(utilization(gpu))
    calls = ctx['beam_calls'][str(beam)] - before
    assert calls == len(times) == spec['users'] * spec['repeats'] == 768
    exclusive = not initial and not any(observed)
    result = dict(status='completed', task=task, admission=admission,
        batch_size=1, users=256, repeats=3, measured_calls=calls, actual_beam=beam, retry_calls=0,
        timed_batch_seconds=times, latency_mean_ms=float(np.mean(times) * 1000),
        latency_p50_ms=float(np.percentile(times, 50) * 1000), latency_p95_ms=float(np.percentile(times, 95) * 1000),
        throughput_users_per_second=calls / sum(times), peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(), baseline_allocated_bytes=baseline,
        exclusive_gpu=exclusive, profile_mode='exclusive_process_checks' if exclusive else 'shared_gpu_observation',
        background_pids_at_start=initial, background_pids_by_batch=observed, utilization_snapshots=telemetry,
        physical_gpu=identity, precision='float32', torch_version=torch.__version__, cuda_version=torch.version.cuda,
        gpu_name=torch.cuda.get_device_name(), parameters=sum(p.numel() for p in ctx['model'].parameters()),
        cost_policy_sha256=original.s1.sha(HERE / 'cost-policy-v1.json'), driver_source_sha256=original.s1.sha(__file__),
        timed_scope='preprocessing, H2D, one official fixed-beam generation, GPU scalar aggregation, ASIN decoding',
        memory_scope='PyTorch allocator only; excludes CUDA context and other processes',
        started=started, finished=time.time(), elapsed_seconds=time.time() - started)
    out.mkdir(parents=True, exist_ok=True)
    original.s1.atomic(out / 'result.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['worker'])
    parser.add_argument('path', type=Path)
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()
    try:
        worker(args.path, args.gpu)
    except BaseException as error:
        task = original.s1.load(args.path)
        original.s1.atomic(Path(task['output']) / 'error.json', dict(error_type=type(error).__name__, error=str(error),
            traceback=traceback.format_exc(), time=time.time()))
        raise
