"""Explicitly budgeted full-catalog training, optional validation, never test.

Prepared entrypoint, not an authorization to run long training. Scheduler horizon
stays at the paper's 150 epochs across separately approved segments.
"""
import argparse
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import sys
import time

from profile_full_models import train_step, evaluate, verify_windows
from run_controlled import emit, gpu_guard, load_json, sha, verify_cache
from training_runtime import PrefixWindows, ShuffledBatches, capture_state, restore_state


def main(args):
    if args.max_steps < 0 or args.compute_seconds <= 0 or args.batch_size <= 0:
        raise ValueError('Nonnegative steps and positive time/batch budgets required')
    if args.max_steps == 0 and args.resume is None:
        raise ValueError('Evaluation-only runs require a checkpoint')
    if args.validation_users < 0:
        raise ValueError('Validation sample size must be nonnegative')
    audit, processed = verify_cache(args.full_cache)
    sid = processed / 'sentence-t5-base_rqkmeans_3x256_psid.sem_ids'
    sid_audit = load_json(processed / 'semantic-audit.json')
    assert sha(sid) == sid_audit['artifacts'][str(sid)]['sha256']
    assert load_json(args.tokenizer_audit)['status'] == 'passed'
    gpu_guard(args.physical_gpu)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'invocation.json').write_text(json.dumps(vars(args), default=str, indent=2))
    code_hashes = {}
    for name in ['train_bounded.py', 'profile_full_models.py', 'training_runtime.py', 'run_controlled.py']:
        source = Path(__file__).with_name(name)
        shutil.copy2(source, args.output / name)
        code_hashes[name] = sha(source)
    sys.path.insert(0, str(args.source))
    emit('import_start')
    import_start = time.monotonic()
    import torch
    from accelerate import Accelerator
    from transformers.optimization import get_scheduler
    from genrec.utils import get_config, init_seed
    from genrec.datasets.AmazonReviews2023.dataset import AmazonReviews2023
    if args.model == 'PSID':
        from genrec.models.PSID.tokenizer import PSIDTokenizer as Tokenizer
        from genrec.models.PSID.model import PSID as Model
    else:
        from genrec.models.Latte.tokenizer import LatteTokenizer as Tokenizer
        from genrec.models.Latte.model import Latte as Model
    emit('imports_ready', seconds=time.monotonic() - import_start)
    gpu_guard(args.physical_gpu)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_per_process_memory_fraction(.25, 0)
    started = time.monotonic()
    stop = {'reason': None}

    def request_stop(signum, frame):
        stop['reason'] = signal.Signals(signum).name

    previous_handlers = {sig: signal.getsignal(sig) for sig in [signal.SIGALRM, signal.SIGTERM, signal.SIGINT]}
    for sig in previous_handlers:
        signal.signal(sig, request_stop)
    signal.setitimer(signal.ITIMER_REAL, args.compute_seconds)
    overrides = {'category': 'Video_Games', 'metadata': 'sentence', 'split': 'last_out', 'kcore': '5core',
        'cache_dir': str(args.full_cache),
        'sent_emb_model': str(args.data_root / 'assets/sentence-transformers/sentence-t5-base'),
        'vq_method': 'rqkmeans', 'vq_codebook_size': 256, 'vq_n_codebooks': 3, 'sent_emb_pca': 192,
        'rand_seed': 2024, 'epochs': 150, 'patience': 20, 'train_batch_size': args.batch_size,
        'eval_batch_size': 4, 'num_beams': args.eval_beam, 'warmup_steps': 10000,
        'dataloader_num_workers': 0}
    config = get_config(args.model, 'AmazonReviews2023', None, overrides)
    (args.output / 'config.json').write_text(json.dumps(config, indent=2))
    # Evaluation settings and device may change across segments; training may not.
    training_config = {k: v for k, v in config.items()
                       if k not in ['num_beams', 'eval_batch_size', 'run_local_time', 'run_id']}
    config.update(device=torch.device('cuda'), use_ddp=False, accelerator=Accelerator())
    init_seed(2024, True)
    dataset = AmazonReviews2023(config)
    tokenizer = Tokenizer(config, dataset)
    windows = PrefixWindows(dataset.all_item_seqs)
    assert len(windows) == audit['model_train_examples']
    if args.validation_users > len(windows.users):
        raise ValueError('Requested validation sample is larger than the validation population')
    verify_windows(windows, tokenizer)
    epoch_steps = math.ceil(len(windows) / args.batch_size)
    total_steps = epoch_steps * 150
    provenance = {'model': args.model, 'cache_hashes': audit['cache_hashes'], 'sid_sha256': sha(sid),
        'training_config': training_config, 'scheduler_steps': total_steps, 'code_hashes': code_hashes}
    model = Model(config, dataset, tokenizer).to(config['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = get_scheduler('cosine', optimizer, num_warmup_steps=10000, num_training_steps=total_steps)
    batches = ShuffledBatches(len(windows), args.batch_size, 2024)
    step = 0
    if args.resume:
        state = torch.load(args.resume, map_location='cpu', weights_only=False)
        batches, step = restore_state(state, model, optimizer, scheduler, provenance)
        del state
    if step + args.max_steps > total_steps:
        raise ValueError('Requested segment exceeds the fixed 150-epoch scheduler horizon')
    first_step = step
    checkpoints = []

    def save_checkpoint():
        path = args.output / f'step-{step}.pt'
        if path.exists():
            return
        pending = path.with_suffix('.pending')
        torch.save(capture_state(model, optimizer, scheduler, batches, step, provenance, cuda=True), pending)
        os.replace(pending, path)
        checkpoints.append({'path': str(path), 'step': step, 'sha256': sha(path)})
        emit('checkpoint_saved', **checkpoints[-1])

    torch.cuda.reset_peak_memory_stats()
    model.train()
    with (args.output / 'training.jsonl').open('x') as log:
        while step < first_step + args.max_steps and not stop['reason']:
            sample = train_step(model, optimizer, scheduler, batches, windows, tokenizer, config['device'])
            step += 1
            row = {'step': step, **sample}
            log.write(json.dumps(row) + '\n')
            log.flush()
            if step % 100 == 0 or step == first_step + 1:
                emit('train_step', **row)
            if step % epoch_steps == 0:
                save_checkpoint()
    save_checkpoint()
    training_seconds = time.monotonic() - started
    result = {'status': 'stopped' if stop['reason'] else 'training_segment_complete',
        'stop_reason': stop['reason'], 'model': args.model, 'start_step': first_step, 'end_step': step,
        'requested_additional_steps': args.max_steps, 'epoch_steps': epoch_steps,
        'checkpoints': checkpoints, 'training_and_setup_seconds': training_seconds,
        'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(), 'test_evaluated': False,
        'checkpoint_selection': 'fixed step; no best-validation selection',
        'evaluation': None, 'engine': 'lazy prefixes, explicit saved shuffle; not upstream DataLoader RNG'}
    # Save successful training evidence before optional, potentially expensive decoding.
    (args.output / 'training-result.json').write_text(json.dumps(result, indent=2))
    if args.validation_users and not stop['reason']:
        users = (windows.users if args.validation_users == len(windows.users)
                 else random.Random(2024).sample(windows.users, args.validation_users))
        original_generate = model.generate

        def budgeted_generate(*positional, **keyword):
            if stop['reason']:
                raise InterruptedError('Validation stopped at a batch boundary')
            predictions = original_generate(*positional, **keyword)
            if stop['reason']:
                raise InterruptedError('Validation stopped at a batch boundary')
            return predictions

        model.generate = budgeted_generate
        try:
            result['evaluation'], _ = evaluate(model, tokenizer, dataset.all_item_seqs, users,
                                               args.eval_beam, 4, bulk=False)
            result['evaluation']['scope'] = 'full validation' if len(users) == len(windows.users) else 'fixed validation sample'
        except InterruptedError:
            result.update(status='stopped_after_training', stop_reason=stop['reason'],
                          evaluation=None, incomplete_validation_discarded=True)
        finally:
            model.generate = original_generate
    signal.setitimer(signal.ITIMER_REAL, 0)
    for sig, handler in previous_handlers.items():
        signal.signal(sig, handler)
    if stop['reason']:
        result.update(status='stopped', stop_reason=stop['reason'])
    result['total_post_import_seconds'] = time.monotonic() - started
    (args.output / 'result.json').write_text(json.dumps(result, indent=2))
    emit('complete', end_step=step, status=result['status'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['source', 'data-root', 'full-cache', 'tokenizer-audit', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--physical-gpu', choices=['0', '1'], required=True)
    parser.add_argument('--model', choices=['PSID', 'Latte'], required=True)
    parser.add_argument('--max-steps', type=int, required=True)
    parser.add_argument('--compute-seconds', type=int, required=True)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--validation-users', type=int, default=0)
    parser.add_argument('--eval-beam', type=int, choices=[50, 500], default=500)
    parser.add_argument('--resume', type=Path)
    main(parser.parse_args())
