"""Bounded full-configuration timing and checkpoint recovery, not a quality experiment."""
import argparse
from collections import Counter
import gc
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import statistics
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[name] = '4'

from run_controlled import abort_on_deadline, emit, gpu_guard, load_json, sha, verify_cache
from training_runtime import PrefixWindows, ShuffledBatches, capture_state, restore_state, aggregate_bulk_transfer, cpu_copy


def make_batch(tokenizer, examples, device, training):
    import torch
    from torch.utils.data import default_collate
    records = []
    for example in examples:
        ids, mask, labels = tokenizer._tokenize_once(example)
        row = {'input_ids': torch.tensor(ids), 'attention_mask': torch.tensor(mask)}
        if training and tokenizer.collate_fn['train'] is not None:
            row['target_item'] = example['item_seq'][-1]
        else:
            row['labels'] = torch.tensor(labels)
        records.append(row)
    collate = tokenizer.collate_fn['train'] if training else None
    batch = (collate or default_collate)(records)
    return {key: value.to(device) for key, value in batch.items()}


def verify_windows(windows, tokenizer):
    users = windows.users
    long_user = next(user for user in users if 24 <= len(windows.sequences[user]) <= 64)
    last_short = next(user for user in reversed(users) if len(windows.sequences[user]) <= 64)
    checked = 0
    for user in dict.fromkeys([users[0], users[1], long_user, last_short]):
        position = users.index(user)
        start = windows.ends[position - 1] if position else 0
        sequence = windows.sequences[user]
        gold = tokenizer.tokenize_function({'user': [user], 'item_seq': [sequence[:-2]]}, 'train')
        for local_index in [0, len(sequence) - 4]:
            example = windows.example(start + local_index)
            ids, mask, labels = tokenizer._tokenize_once(example)
            assert ids == gold['input_ids'][local_index]
            assert mask == gold['attention_mask'][local_index]
            if 'labels' in gold:
                assert labels == gold['labels'][local_index]
            else:
                assert example['item_seq'][-1] == gold['target_item'][local_index]
            checked += 1
    return checked


def train_step(model, optimizer, scheduler, sampler, windows, tokenizer, device):
    import torch
    started = time.monotonic()
    indices = sampler.next()
    batch = make_batch(tokenizer, [windows.example(index) for index in indices], device, True)
    torch.cuda.synchronize()
    prepared = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    loss = model(batch).loss
    assert torch.isfinite(loss)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    torch.cuda.synchronize()
    ended = time.monotonic()
    return {'loss': loss.item(), 'compute_seconds': ended - prepared,
            'end_to_end_seconds': ended - started, 'lr': optimizer.param_groups[0]['lr']}


def evaluate(model, tokenizer, sequences, users, beam, batch_size, bulk=False):
    import torch
    from genrec.evaluator import Evaluator
    inverse = {tuple(value): item for item, value in tokenizer.item2tokens.items()}
    parent = getattr(model, '_aggregate_predictions', None)
    if bulk:
        assert parent is not None
        model._aggregate_predictions = lambda tokens, scores, k: aggregate_bulk_transfer(parent, tokens, scores, k)
    model.config['num_beams'] = beam
    original_generate = model.t5.generate
    calls = Counter()

    def counted(*positional, **keyword):
        calls[str(keyword['num_beams'])] += 1
        return original_generate(*positional, **keyword)

    model.t5.generate = counted
    evaluator = Evaluator(model.config, tokenizer)
    values, predicted_all, rows = {}, [], []
    model.eval()
    torch.cuda.synchronize()
    started = time.monotonic()
    try:
        with torch.no_grad():
            for start in range(0, len(users), batch_size):
                selected = users[start:start + batch_size]
                batch = make_batch(tokenizer, [{'user': user, 'item_seq': sequences[user][:-1]} for user in selected],
                                   model.config['device'], False)
                predicted = model.generate(batch, n_return_sequences=10)
                if predicted.shape[-1] < 3:
                    predicted = torch.nn.functional.pad(predicted, (0, 3 - predicted.shape[-1]))
                metrics = evaluator.calculate_metrics(predicted, batch['labels'])
                for key, value in metrics.items():
                    values.setdefault(key, []).append(value)
                codes = predicted.detach().cpu().tolist()
                predicted_all.extend(codes)
                for user, candidates in zip(selected, codes):
                    items = [inverse.get(tuple(code)) for code in candidates]
                    target = sequences[user][-2]
                    rank = next((index + 1 for index, item in enumerate(items) if item == target), None)
                    rows.append({'user': user, 'target': target, 'candidate_slots': items, 'rank': rank})
        torch.cuda.synchronize()
        seconds = time.monotonic() - started
    finally:
        model.t5.generate = original_generate
        if bulk:
            model._aggregate_predictions = parent
    metrics = {key: torch.cat(parts).mean().item() for key, parts in values.items()}
    for k in [5, 10]:
        direct_recall = sum(row['rank'] is not None and row['rank'] <= k for row in rows) / len(rows)
        direct_ndcg = sum(1 / math.log2(row['rank'] + 1) for row in rows if row['rank'] and row['rank'] <= k) / len(rows)
        assert math.isclose(direct_recall, metrics[f'recall@{k}'], abs_tol=1e-7)
        assert math.isclose(direct_ndcg, metrics[f'ndcg@{k}'], abs_tol=1e-7)
    return {'seconds': seconds, 'requested_beam': beam, 'actual_beam_calls': dict(calls),
            'batch_size': batch_size, 'metrics': metrics, 'rows': rows,
            'mean_unique_valid_candidates': statistics.mean(len({i for i in row['candidate_slots'] if i}) for row in rows),
            'asid_metrics_independently_recomputed': True}, predicted_all


def aggregation_fixture(model):
    import torch
    parent = model._aggregate_predictions
    tokens = list(model.tokenizer.item2tokens.values())[:3]
    paths = [[1, *tokens[0]], [2, *tokens[0]], [1, *tokens[1]], [1, *tokens[2]],
             [0, *tokens[2]], [1, 0, 0, 0]]
    tensor = torch.tensor([paths], device=model.config['device'])
    scores = torch.tensor([[-1., -.5, -.7, -.7, 0., 0.]], device=model.config['device'])
    expected = torch.tensor([[tokens[0], tokens[1], tokens[2], (0, 0, 0)]], device=model.config['device'])
    previous = model.aggregation_method
    try:
        for method in ['agg_max', 'agg_sum']:
            model.aggregation_method = method
            reference = parent(tensor, scores, 4)
            transferred = aggregate_bulk_transfer(parent, tensor, scores, 4)
            assert torch.equal(reference, expected)
            assert torch.equal(reference, transferred)
    finally:
        model.aggregation_method = previous
    return True


def main(args):
    if min(args.steps, args.batch_size, args.validation_users, args.compute_seconds) <= 0:
        raise ValueError('Positive step, batch, validation and time budgets are required')
    assert load_json(args.tokenizer_audit)['status'] == 'passed'
    audit, processed = verify_cache(args.full_cache)
    sid = processed / 'sentence-t5-base_rqkmeans_3x256_psid.sem_ids'
    sid_report = load_json(processed / 'semantic-audit.json')
    assert sha(sid) == sid_report['artifacts'][str(sid)]['sha256']
    gpu_guard(args.physical_gpu)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'invocation.json').write_text(json.dumps(vars(args), default=str, indent=2))
    for filename in [__file__, Path(__file__).with_name('training_runtime.py')]:
        shutil.copy2(filename, args.output / Path(filename).name)
    if args.runtime_cache:
        cache = load_json(args.runtime_cache / 'result.json')
        assert cache['status'] == 'passed'
        expected_paths = [cache['site_packages']]
        if args.source_cache:
            sources = load_json(args.source_cache / 'result.json')
            assert sources['status'] == 'passed'
            expected_paths.insert(0, sources['site_packages'])
            shutil.copy2(args.source_cache / 'result.json', args.output / 'source-cache.json')
        if args.python_cache:
            runtime = load_json(args.python_cache / 'result.json')
            assert runtime['status'] == 'passed'
            assert sys.executable == runtime['runtime']['executable']
            assert os.environ.get('PYTHONHOME') == runtime['runtime']['prefix']
            expected_paths.append(str(Path(runtime['source_env']) / 'lib/python3.12/site-packages'))
            shutil.copy2(args.python_cache / 'result.json', args.output / 'python-cache.json')
        assert os.environ.get('PYTHONPATH') == os.pathsep.join(expected_paths)
        shutil.copy2(args.runtime_cache / 'result.json', args.output / 'runtime-cache.json')
    else:
        assert not args.source_cache and not args.python_cache
    sys.path.insert(0, str(args.source))
    emit('import_start')
    import_start = time.monotonic()
    import torch
    emit('import_ready', module='torch', elapsed_seconds=time.monotonic() - import_start)
    from accelerate import Accelerator
    emit('import_ready', module='accelerate', elapsed_seconds=time.monotonic() - import_start)
    from transformers.optimization import get_scheduler
    emit('import_ready', module='transformers.optimization', elapsed_seconds=time.monotonic() - import_start)
    from genrec.utils import get_config, init_seed
    from genrec.datasets.AmazonReviews2023.dataset import AmazonReviews2023
    emit('import_ready', module='genrec.dataset', elapsed_seconds=time.monotonic() - import_start)
    from genrec.models.PSID.tokenizer import PSIDTokenizer
    from genrec.models.Latte.tokenizer import LatteTokenizer
    from genrec.models.PSID.model import PSID
    from genrec.models.Latte.model import Latte
    emit('imports_ready', seconds=time.monotonic() - import_start)
    emit('runtime_paths', torch=torch.__file__, torch_version=torch.__version__,
         python=sys.executable, python_version=sys.version, accelerator_module=Accelerator.__module__)
    gpu_guard(args.physical_gpu)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_per_process_memory_fraction(.25, 0)
    signal.signal(signal.SIGALRM, abort_on_deadline)
    signal.setitimer(signal.ITIMER_REAL, args.compute_seconds)
    started = time.monotonic()
    results = []
    overrides = {'category': 'Video_Games', 'metadata': 'sentence', 'split': 'last_out', 'kcore': '5core',
        'cache_dir': str(args.full_cache), 'sent_emb_model': str(args.data_root / 'assets/sentence-transformers/sentence-t5-base'),
        'vq_method': 'rqkmeans', 'vq_codebook_size': 256, 'vq_n_codebooks': 3, 'sent_emb_pca': 192,
        'rand_seed': 2024, 'epochs': 150, 'patience': 20, 'train_batch_size': args.batch_size,
        'eval_batch_size': 4, 'num_beams': 500, 'warmup_steps': 10000, 'dataloader_num_workers': 0}
    for name, model_class, tokenizer_class in [('PSID', PSID, PSIDTokenizer), ('Latte', Latte, LatteTokenizer)]:
        config = get_config(name, 'AmazonReviews2023', None, overrides)
        (args.output / f'{name}-config.json').write_text(json.dumps(config, indent=2))
        config.update(device=torch.device('cuda'), use_ddp=False, accelerator=Accelerator())
        init_seed(2024, True)
        dataset = AmazonReviews2023(config)
        tokenizer = tokenizer_class(config, dataset)
        windows = PrefixWindows(dataset.all_item_seqs)
        assert len(windows) == audit['model_train_examples']
        equivalent = verify_windows(windows, tokenizer)
        total_steps = math.ceil(len(windows) / args.batch_size) * 150

        def create():
            model = model_class(config, dataset, tokenizer).to(config['device'])
            optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
            scheduler = get_scheduler('cosine', optimizer, num_warmup_steps=10000, num_training_steps=total_steps)
            return model, optimizer, scheduler

        model, optimizer, scheduler = create()
        sampler = ShuffledBatches(len(windows), args.batch_size, 2024)
        provenance = {'model': name, 'cache_hashes': audit['cache_hashes'], 'sid_sha256': sha(sid),
                      'batch_size': args.batch_size, 'total_scheduler_steps': total_steps, 'seed': 2024}
        torch.cuda.reset_peak_memory_stats()
        samples = []
        for step in range(args.steps):
            samples.append(train_step(model, optimizer, scheduler, sampler, windows, tokenizer, config['device']))
            if step % 10 == 0 or step == args.steps - 1:
                emit('train_step', model=name, step=step + 1, **samples[-1])
        checkpoint = args.output / f'{name}-step{args.steps}.pt'
        torch.save(capture_state(model, optimizer, scheduler, sampler, args.steps, provenance, cuda=True), checkpoint)
        reference_losses = [train_step(model, optimizer, scheduler, sampler, windows, tokenizer, config['device'])['loss']
                            for _ in range(3)]
        reference = cpu_copy(model.state_dict())
        sampler_reference = sampler.state_dict()
        del model, optimizer, scheduler
        gc.collect()
        torch.cuda.empty_cache()
        model, optimizer, scheduler = create()
        restored = torch.load(checkpoint, map_location='cpu', weights_only=False)
        sampler, restored_step = restore_state(restored, model, optimizer, scheduler, provenance)
        replay_losses = [train_step(model, optimizer, scheduler, sampler, windows, tokenizer, config['device'])['loss']
                         for _ in range(3)]
        actual = cpu_copy(model.state_dict())
        bitwise = all(torch.equal(reference[key], actual[key]) for key in reference)
        maximum_difference = max((reference[key] - actual[key]).abs().max().item() for key in reference)
        assert maximum_difference <= 1e-6
        assert all(abs(a - b) <= 1e-6 for a, b in zip(reference_losses, replay_losses))
        assert sampler.position == sampler_reference['position'] and torch.equal(sampler.order, sampler_reference['order'])
        del restored, reference, actual, optimizer, scheduler
        users = random.Random(2024).sample(windows.users, args.validation_users)
        evaluations = {}
        model.eval()
        fixture = aggregation_fixture(model) if name == 'Latte' else None
        for beam in [50, 500]:
            original, predictions = evaluate(model, tokenizer, dataset.all_item_seqs, users, beam, 4)
            evaluations[str(beam)] = {'original': original}
            if name == 'Latte':
                bulk, bulk_predictions = evaluate(model, tokenizer, dataset.all_item_seqs, users, beam, 4, bulk=True)
                assert predictions == bulk_predictions
                evaluations[str(beam)].update(bulk_transfer=bulk, predictions_identical=True)
            emit('evaluation_complete', model=name, beam=beam, seconds=original['seconds'])
        stable = samples[min(10, len(samples) - 1):]
        report = {'model': name, 'scope': 'timing only; not a converged recommendation result',
            'parameters': sum(p.numel() for p in model.parameters()), 'profile_steps': args.steps,
            'evaluation_at_step': args.steps + 3, 'train_examples': len(windows),
            'lazy_tokenization_matches_upstream_cases': equivalent, 'samples': samples,
            'stable_end_to_end_median_seconds': statistics.median(value['end_to_end_seconds'] for value in stable),
            'stable_compute_median_seconds': statistics.median(value['compute_seconds'] for value in stable),
            'full_epoch_steps': math.ceil(len(windows) / args.batch_size),
            'full_epoch_train_only_estimate_seconds': math.ceil(len(windows) / args.batch_size) * statistics.median(value['end_to_end_seconds'] for value in stable),
            'checkpoint': str(checkpoint), 'checkpoint_sha256': sha(checkpoint),
            'resume_three_steps': {'checkpoint_step': restored_step, 'reference_losses': reference_losses,
                'replayed_losses': replay_losses, 'parameters_bitwise_equal': bitwise,
                'maximum_parameter_difference': maximum_difference, 'sampler_position_equal': True},
            'validation_users': args.validation_users, 'evaluations': evaluations,
            'bulk_aggregation_hand_fixture_max_and_sum': fixture,
            'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(),
            'optimizer_schedule': 'upstream AdamW all parameters; 10000-step warmup; cosine to 150 epochs',
            'training_engine': 'lazy windows and explicit saved permutation; differs from upstream eager HF map/DataLoader RNG',
            'deterministic_algorithms': True, 'test_evaluated': False}
        (args.output / f'{name}-result.json').write_text(json.dumps(report, indent=2))
        results.append(report)
        del model, tokenizer, dataset, sampler
        gc.collect()
        torch.cuda.empty_cache()
    signal.setitimer(signal.ITIMER_REAL, 0)
    summary = {'status': 'passed', 'compute_seconds': time.monotonic() - started, 'results': results,
               'test_evaluated': False, 'formal_training_started': False}
    (args.output / 'result.json').write_text(json.dumps(summary, indent=2))
    emit('complete', compute_seconds=summary['compute_seconds'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--physical-gpu', choices=['0', '1'], default='0')
    parser.add_argument('--runtime-cache', type=Path)
    parser.add_argument('--source-cache', type=Path)
    parser.add_argument('--python-cache', type=Path)
    for name in ['source', 'data-root', 'full-cache', 'tokenizer-audit', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--validation-users', type=int, default=128)
    parser.add_argument('--compute-seconds', type=int, default=300)
    main(parser.parse_args())
