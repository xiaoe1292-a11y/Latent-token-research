"""Local-assets, validation-only entrypoint for pinned Latte/PSID source.

smoke: a separate 512-user / reduced-size SID pilot, exactly N steps per model.
train-val: upstream Trainer.fit with pre-existing full-size SIDs; never test.
"""
import argparse
from collections import Counter
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def emit(stage, **values):
    print(json.dumps({'stage': stage, **values}, default=str), flush=True)


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify_cache(root):
    audit = load_json(root / 'protocol-audit.json')
    processed = root / 'AmazonReviews2023/Video_Games/processed'
    for name, expected in audit['cache_hashes'].items():
        assert sha(processed / name) == expected, f'Cache changed: {name}'
    assert audit['official_hf_hash_match']
    return audit, processed


def pilot_cache(full_root, pilot_root, users_limit):
    audit, processed = verify_cache(full_root)
    full_mapping = load_json(processed / 'id_mapping.json')
    sequences = load_json(processed / 'all_item_seqs.json')
    metadata = load_json(processed / 'metadata.sentence.json')
    users = full_mapping['id2user'][1:users_limit + 1]
    selected = {user: sequences[user] for user in users}
    items = {item for seq in selected.values() for item in seq}
    mapping = {'id2user': ['[PAD]'] + users,
               'id2item': [item for item in full_mapping['id2item'] if item == '[PAD]' or item in items]}
    mapping.update({key: {value: index for index, value in enumerate(mapping['id2' + key[:-3]])}
                    for key in ['user2id', 'item2id']})
    # A new namespace prevents smoke SIDs from contaminating the full experiment.
    output = pilot_root / 'AmazonReviews2023/Video_Games/processed'
    output.mkdir(parents=True, exist_ok=False)
    values = {'id_mapping.json': mapping, 'all_item_seqs.json': selected,
              'metadata.sentence.json': {item: metadata[item] for item in metadata if item in items}}
    for name, value in values.items():
        (output / name).write_text(json.dumps(value), encoding='utf-8')
    info = {'scope': 'smoke-only, not paper protocol result', 'users': len(users), 'items': len(items),
            'full_cache_hashes': audit['cache_hashes'], 'selection': 'first N upstream mapped users',
            'train_examples': sum(len(seq) - 3 for seq in selected.values())}
    (pilot_root / 'pilot-manifest.json').write_text(json.dumps(info, indent=2))
    return info


def gpu_guard(physical_gpu='0'):
    assert physical_gpu in ('0', '1'), 'Only explicitly selected GPU 0 or 1 is supported'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == physical_gpu, 'Visible GPU differs from explicitly selected physical GPU'
    text = subprocess.check_output(['nvidia-smi', '--id=' + physical_gpu, '--query-compute-apps=pid,used_gpu_memory',
                                    '--format=csv,noheader,nounits'], text=True)
    others = [line for line in text.splitlines() if line.strip() and int(line.split(',')[0]) != os.getpid()]
    if others:
        raise RuntimeError('GPU ' + physical_gpu + ' has another compute process; stopping: ' + '; '.join(others))
    snapshot = subprocess.check_output(['nvidia-smi', '--id=' + physical_gpu,
        '--query-gpu=index,name,memory.used,memory.total', '--format=csv,noheader'], text=True).strip()
    emit('gpu_available', snapshot=snapshot)
    return snapshot


def abort_on_deadline(signum, frame):
    raise TimeoutError('Authorized compute time limit reached')


def train_smoke(name, config, dataset, tokenizer_class, model_class, output, args):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from genrec.evaluator import Evaluator
    from genrec.utils import init_seed

    init_seed(config['rand_seed'], True)
    tokenizer = tokenizer_class(config, dataset)
    inverse = {tuple(tokens): item for item, tokens in tokenizer.item2tokens.items()}
    assert len(inverse) == dataset.n_items - 1, 'Semantic IDs are not unique'
    selected_splits = dataset.split()
    # Test data is never tokenized or sent to an evaluator.
    validation = selected_splits['val'].select(range(min(args.validation_users, len(selected_splits['val']))))
    tokenized = tokenizer.tokenize({'train': selected_splits['train'], 'val': validation})
    assert len(tokenized['train']) == sum(len(seq) - 3 for seq in dataset.all_item_seqs.values())
    assert all(train == full[:-2] for train, full in zip(selected_splits['train']['item_seq'],
                                                      dataset.all_item_seqs.values()))
    generator = torch.Generator().manual_seed(config['rand_seed'])
    train_loader = DataLoader(tokenized['train'], batch_size=config['train_batch_size'], shuffle=True,
                              generator=generator, num_workers=0, collate_fn=tokenizer.collate_fn['train'])
    val_loader = DataLoader(tokenized['val'], batch_size=config['eval_batch_size'], shuffle=False,
                            num_workers=0, collate_fn=tokenizer.collate_fn['val'])
    model = model_class(config, dataset, tokenizer).to(config['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    emit('model_ready', model=name, parameters=sum(p.numel() for p in model.parameters()),
         train_examples=len(tokenized['train']), validation_users=len(validation))
    iterator = iter(train_loader)
    losses, step_times = [], []
    model.train()
    for step in range(args.steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        batch = {key: value.to(config['device']) for key, value in batch.items()}
        assert batch['input_ids'].shape[1] == 62
        torch.cuda.synchronize()
        start = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        loss = model(batch).loss
        assert torch.isfinite(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
        optimizer.step()
        torch.cuda.synchronize()
        losses.append(loss.item())
        step_times.append(time.monotonic() - start)
        emit('train_step', model=name, step=step + 1, loss=loss.item(), seconds=step_times[-1])

    # Save full pilot state, but only claim verified model-weight reload, not exact continuation.
    checkpoint = output / f'{name}-step{args.steps}.pt'
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save({'model': state, 'optimizer': optimizer.state_dict(), 'step': args.steps,
                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                'numpy_rng': np.random.get_state(), 'python_rng': random.getstate(),
                'config': {key: str(value) if key == 'device' else value for key, value in config.items()
                           if key != 'accelerator'}}, checkpoint)
    restored = torch.load(checkpoint, map_location='cpu', weights_only=False)
    clone = model_class(config, dataset, tokenizer)
    clone.load_state_dict(restored['model'], strict=True)
    assert all(torch.equal(value, clone.state_dict()[key]) for key, value in state.items())
    del clone, restored, state

    calls = []
    original_generate = model.t5.generate

    def observed_generate(*positional, **keyword):
        result = original_generate(*positional, **keyword)
        offset = 2 if name == 'Latte' else 1
        raw_codes = result.sequences[:, offset:offset + tokenizer.n_digit].detach().cpu().tolist()
        valid = [tuple(code) in inverse for code in raw_codes]
        if name == 'Latte':
            latent = result.sequences[:, 1].detach().cpu().tolist()
            valid = [ok and 1 <= code <= tokenizer.n_latent_tokens for ok, code in zip(valid, latent)]
        calls.append({'beam': keyword['num_beams'], 'returned_per_user': keyword['num_return_sequences'],
                      'raw_sequences': len(raw_codes), 'raw_valid_sequences': sum(valid),
                      'first_raw_sequences': result.sequences[:3].detach().cpu().tolist() if not calls else []})
        return result

    model.t5.generate = observed_generate
    model.eval()
    evaluator = Evaluator(config, tokenizer)
    values, predictions, labels_all = {}, [], []
    torch.cuda.synchronize()
    evaluation_start = time.monotonic()
    with torch.no_grad():
        for batch in val_loader:
            batch = {key: value.to(config['device']) for key, value in batch.items()}
            predicted = model.generate(batch, n_return_sequences=max(config['topk']))
            if predicted.shape[-1] < tokenizer.n_digit:
                # Incomplete codes remain invalid, never mapped to a real item by padding.
                predicted = torch.nn.functional.pad(predicted, (0, tokenizer.n_digit - predicted.shape[-1]))
            metrics = evaluator.calculate_metrics(predicted, batch['labels'])
            for key, value in metrics.items():
                values.setdefault(key, []).append(value)
            predictions.extend(predicted.detach().cpu().tolist())
            labels_all.append(batch['labels'].detach().cpu())
    torch.cuda.synchronize()
    eval_seconds = time.monotonic() - evaluation_start
    model.t5.generate = original_generate

    real_candidates = [[inverse[tuple(code)] for code in row if tuple(code) in inverse] for row in predictions]
    counts = [len(set(row)) for row in real_candidates]
    popularity = Counter(item for seq in dataset.all_item_seqs.values() for item in seq[:-2])
    ranked = sorted(dataset.id_mapping['id2item'][1:], key=lambda item: (-popularity[item], dataset.item2id[item]))
    pop_preds = torch.tensor([[list(tokenizer.item2tokens[item]) for item in ranked[:10]]] * len(validation))
    pop_metrics = evaluator.calculate_metrics(pop_preds, torch.cat(labels_all))
    semantic_path = Path(dataset.cache_dir) / f'processed/sentence-t5-base_rqkmeans_3x{config["vq_codebook_size"]}_psid.sem_ids'
    codes = load_json(semantic_path)
    report = {
        'model': name, 'scope': 'smoke only; not converged; not paper reproduction metrics',
        'steps': args.steps, 'parameters': sum(p.numel() for p in model.parameters()),
        'losses': losses, 'step_seconds': step_times, 'evaluation_seconds': eval_seconds,
        'pilot_optimizer': 'AdamW, constant lr, no scheduler; differs from paper training schedule',
        'validation_users': len(validation), 'train_examples': len(tokenized['train']),
        'validation_metrics': {key: torch.cat(parts).mean().item() for key, parts in values.items()},
        'popularity_validation_same_users': {key: value.mean().item() for key, value in pop_metrics.items()},
        'candidate_diagnostics': {'unique_valid_per_user': counts, 'users_below_top10': sum(n < 10 for n in counts),
            'users_with_valid_candidates': sum(n > 0 for n in counts),
            'duplicate_valid_slots': sum(len(row) - len(set(row)) for row in real_candidates),
            'invalid_final_slots': len(validation) * 10 - sum(len(row) for row in real_candidates),
            'catalog_coverage': len({item for row in real_candidates for item in row}) / (dataset.n_items - 1),
            'generation_calls': calls},
        'semantic_ids_sha256': sha(semantic_path), 'semantic_unique_items': len({tuple(v) for v in codes.values()}),
        'codebook_utilization': [len({code[digit] for code in codes.values()}) / config['vq_codebook_size'] for digit in range(3)],
        'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(),
        'checkpoint': str(checkpoint), 'checkpoint_sha256': sha(checkpoint), 'model_reload_equal': True,
        'exact_optimizer_resume_verified': False, 'test_evaluated': False,
        'generation_verified': any(n > 0 for n in counts),
        'examples': [{'user': validation[index]['user'], 'history': validation[index]['item_seq'][:-1],
                      'target': validation[index]['item_seq'][-1], 'predicted_items': real_candidates[index]}
                     for index in range(min(3, len(validation)))],
    }
    (output / f'{name}-result.json').write_text(json.dumps(report, indent=2))
    emit('model_complete', model=name, validation_metrics=report['validation_metrics'],
         mean_unique_candidates=sum(counts) / len(counts), peak_cuda=report['peak_cuda_allocated_bytes'])
    return report


def main(args):
    if min(args.users, args.validation_users, args.steps, args.compute_seconds) <= 0:
        raise ValueError('Users, validation users, training steps and compute limit must be positive')
    if not 0 < args.memory_fraction <= 1:
        raise ValueError('Memory fraction must be in (0,1]')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'invocation.json').write_text(json.dumps(vars(args), default=str, indent=2))
    shutil.copy2(__file__, args.output / 'run_controlled.py')
    sys.path.insert(0, str(args.source))
    emit('starting', mode=args.mode, gpu_visible=os.environ.get('CUDA_VISIBLE_DEVICES'))
    gpu_snapshot = gpu_guard()
    audit, _ = verify_cache(args.full_cache)
    cache_root = args.full_cache
    pilot = None
    if args.mode == 'smoke':
        cache_root = args.data_root / 'cache' / args.output.name
        pilot = pilot_cache(args.full_cache, cache_root, args.users)
        if args.reuse_embedding_cache:
            previous = args.reuse_embedding_cache / 'AmazonReviews2023/Video_Games/processed'
            current = cache_root / 'AmazonReviews2023/Video_Games/processed'
            for name in ['id_mapping.json', 'metadata.sentence.json']:
                assert sha(previous / name) == sha(current / name), 'Embedding order/content changed'
            embedding = previous / 'sentence-t5-base.sent_emb'
            previous_config = load_json(args.output.parent / args.reuse_embedding_cache.name / 'PSID-config.json')
            assert previous_config['sent_emb_model'] == str(args.data_root / 'assets/sentence-transformers/sentence-t5-base')
            assert previous_config['sent_emb_dim'] == 768 and previous_config['metadata'] == 'sentence'
            assert embedding.stat().st_size == pilot['items'] * 768 * 4
            shutil.copy2(embedding, current / embedding.name)
            pilot['reused_raw_embedding_sha256'] = sha(embedding)
        emit('pilot_cache_ready', **pilot)
    model_path = args.data_root / 'assets/sentence-transformers/sentence-t5-base'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if not model_path.exists():
        model_path.symlink_to(args.data_root / 'assets/sentence-t5-base', target_is_directory=True)
    assert model_path.resolve() == (args.data_root / 'assets/sentence-t5-base').resolve()

    import_start = time.monotonic()
    emit('import_start')
    import torch
    from accelerate import Accelerator
    from genrec.datasets.AmazonReviews2023.dataset import AmazonReviews2023
    from genrec.models.PSID.model import PSID
    from genrec.models.PSID.tokenizer import PSIDTokenizer
    from genrec.models.Latte.model import Latte
    from genrec.models.Latte.tokenizer import LatteTokenizer
    from genrec.utils import get_config, init_seed
    emit('imports_ready', seconds=time.monotonic() - import_start)
    gpu_guard()
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, 0)
    signal.signal(signal.SIGALRM, abort_on_deadline)
    signal.setitimer(signal.ITIMER_REAL, args.compute_seconds)
    compute_start = time.monotonic()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    accelerator = Accelerator(log_with='tensorboard', project_dir=str(args.output / 'tensorboard'))
    overrides = {
        'category': 'Video_Games', 'metadata': 'sentence', 'split': 'last_out', 'kcore': '5core',
        'cache_dir': str(cache_root), 'sent_emb_model': str(model_path), 'vq_method': 'rqkmeans',
        'vq_n_codebooks': 3, 'vq_codebook_size': args.pilot_codebook_size if args.mode == 'smoke' else 256,
        'sent_emb_pca': 32 if args.mode == 'smoke' else 192,
        'sent_emb_batch_size': 16, 'faiss_omp_num_threads': 4, 'dataloader_num_workers': 0,
        'rand_seed': 2024, 'train_batch_size': 16 if args.mode == 'smoke' else 256,
        'eval_batch_size': 2 if args.mode == 'smoke' else 4,
        'num_beams': 50 if args.mode == 'smoke' else 500,
        'patience': 20, 'epochs': 1 if args.mode == 'smoke' else 150,
        'warmup_steps': 0 if args.mode == 'smoke' else 10000,
        'steps': args.steps if args.mode == 'smoke' else None,
        'run_id': args.output.name, 'log_dir': str(args.output / 'logs'),
        'ckpt_dir': str(args.output / 'checkpoints'), 'tensorboard_log_dir': str(args.output / 'tensorboard'),
    }
    models = [('PSID', PSID, PSIDTokenizer), ('Latte', Latte, LatteTokenizer)]
    reports = []
    for name, model_class, tokenizer_class in models:
        if args.mode == 'train-val' and name != args.model:
            continue
        config = get_config(name, 'AmazonReviews2023', None, overrides)
        (args.output / f'{name}-config.json').write_text(json.dumps(config, indent=2))
        config['device'], config['use_ddp'], config['accelerator'] = torch.device('cuda'), False, accelerator
        init_seed(config['rand_seed'], True)
        dataset = AmazonReviews2023(config)
        split = dataset.split()
        if args.mode == 'smoke':
            torch.cuda.reset_peak_memory_stats()
            reports.append(train_smoke(name, config, dataset, tokenizer_class, model_class, args.output, args))
            gc.collect()
            torch.cuda.empty_cache()
        else:
            # Explicitly refuse to create full-catalog embeddings or train a tokenizer implicitly.
            processed = Path(dataset.cache_dir) / 'processed'
            expected = processed / 'sentence-t5-base_rqkmeans_3x256_psid.sem_ids'
            if not expected.exists():
                raise RuntimeError('Full-size SIDs must be prepared and audited before a long model training run')
            from torch.utils.data import DataLoader
            from genrec.trainer import Trainer
            tokenizer = tokenizer_class(config, dataset)
            tokenized = tokenizer.tokenize({key: split[key] for key in ['train', 'val']})
            model = model_class(config, dataset, tokenizer)
            trainer = Trainer(config, model, tokenizer)
            loaders = {key: DataLoader(tokenized[key], batch_size=config['train_batch_size' if key == 'train' else 'eval_batch_size'],
                        shuffle=key == 'train', num_workers=0, collate_fn=tokenizer.collate_fn[key]) for key in ['train', 'val']}
            best_epoch, best_score = trainer.fit(loaders['train'], loaders['val'])
            reports.append({'model': name, 'best_epoch': int(best_epoch), 'best_val_ndcg@10': best_score,
                            'test_evaluated': False, 'checkpoint': trainer.saved_model_ckpt})
            trainer.end()
    signal.setitimer(signal.ITIMER_REAL, 0)
    if args.mode == 'smoke':
        assert reports[0]['semantic_ids_sha256'] == reports[1]['semantic_ids_sha256']
        assert reports[0]['popularity_validation_same_users'] == reports[1]['popularity_validation_same_users']
    passed = args.mode != 'smoke' or all(item['generation_verified'] for item in reports)
    result = {'status': 'passed' if passed else 'needs_generation_debugging', 'mode': args.mode,
              'compute_seconds': time.monotonic() - compute_start,
              'gpu_before': gpu_snapshot, 'pilot': pilot, 'full_protocol_users': audit['users'],
              'results': reports, 'test_evaluated': False,
              'source_hashes': {str(p.relative_to(args.source)): sha(p) for p in args.source.rglob('*')
                                if p.is_file() and p.suffix in {'.py', '.yaml'}}}
    (args.output / 'result.json').write_text(json.dumps(result, indent=2))
    emit('complete', mode=args.mode, compute_seconds=result['compute_seconds'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['smoke', 'train-val'], default='smoke')
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--full-cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', choices=['PSID', 'Latte'], default='Latte')
    parser.add_argument('--users', type=int, default=512)
    parser.add_argument('--validation-users', type=int, default=32)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--compute-seconds', type=int, default=300)
    parser.add_argument('--memory-fraction', type=float, default=0.25)
    parser.add_argument('--pilot-codebook-size', type=int, default=64)
    parser.add_argument('--reuse-embedding-cache', type=Path)
    parsed = parser.parse_args()
    output_existed = parsed.output.exists()
    try:
        main(parsed)
    except Exception as error:
        emit('failed', error_type=type(error).__name__, error=str(error))
        if not output_existed and parsed.output.is_dir():
            (parsed.output / 'failure.json').write_text(json.dumps({'status': 'failed',
                'error_type': type(error).__name__, 'error': str(error), 'test_evaluated': False}, indent=2))
        raise
