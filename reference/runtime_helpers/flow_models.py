"""Reuses pinned models and training primitives; no edits to historical runners."""
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
from types import SimpleNamespace, MethodType

from flow_common import CODE, DATA, BASE, CACHE, PROCESSED, SAS_SOURCE, SAS_SHA, load, sha


def prepare_runtime(gpu):
    import os
    import subprocess
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == str(gpu)
    text = subprocess.check_output(['nvidia-smi', '--id=' + str(gpu),
        '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], text=True).strip()
    uuid, free = [x.strip() for x in text.split(',')]
    if int(free) < 7168:
        raise ResourceUnavailable('Less than 7GiB free; skip this device without disturbing existing jobs')
    sys.path.insert(0, str(BASE))
    sys.path.insert(0, str(CODE / 'source'))
    import torch
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_per_process_memory_fraction(.25, 0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return {'physical_gpu': int(gpu), 'uuid': uuid, 'free_mib_at_admission': int(free)}


class ResourceUnavailable(RuntimeError):
    pass


class ScientificFailure(RuntimeError):
    pass


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dataset():
    return load(PROCESSED / 'all_item_seqs.json'), load(PROCESSED / 'id_mapping.json')


def config_seed(case):
    seed_all(case['seed'])


def build(case, checkpoint=None, optimizer_needed=False):
    import torch
    from training_runtime import ShuffledBatches
    state = torch.load(checkpoint, map_location='cpu', weights_only=False) if checkpoint else None
    config_seed(case)
    raw, mapping = dataset()
    for name, expected_hash in load(CACHE / 'protocol-audit.json')['cache_hashes'].items():
        assert sha(PROCESSED / name) == expected_hash, 'Frozen data changed: ' + name
    if case['family'] == 'gen':
        from accelerate import Accelerator
        from genrec.utils import get_config
        from genrec.datasets.AmazonReviews2023.dataset import AmazonReviews2023
        template = state or torch.load(case['template'], map_location='cpu', weights_only=False)
        config = dict(template['provenance']['training_config'])
        config.update(rand_seed=case['seed'], cache_dir=case.get('cache', str(CACHE)),
                      vq_method=case.get('loader_method', 'rqkmeans'))
        if case.get('variant') not in (None, 'rqkmeans'):
            config['experiment_variant'] = case['variant']
        config.update(num_beams=500, eval_batch_size=4)
        config = get_config(case['model'], 'AmazonReviews2023', None, config)
        proc = Path(config['cache_dir']) / 'AmazonReviews2023/Video_Games/processed'
        sid = proc / f"sentence-t5-base_{config['vq_method']}_3x256_psid.sem_ids"
        if not sid.exists():
            raise ScientificFailure('SID prerequisite incomplete; implicit encoder/quantizer training is forbidden')
        ids = load(sid)
        assert len(ids) == len({tuple(x) for x in ids.values()}) == 25612
        config.update(device=torch.device('cuda'), use_ddp=False, accelerator=Accelerator())
        data = AmazonReviews2023(config)
        if case['model'] == 'PSID':
            from genrec.models.PSID.model import PSID as Model
            from genrec.models.PSID.tokenizer import PSIDTokenizer as Tokenizer
        else:
            from genrec.models.Latte.model import Latte as Model
            from genrec.models.Latte.tokenizer import LatteTokenizer as Tokenizer
        tokenizer = Tokenizer(config, data)
        model = Model(config, data, tokenizer).cuda()
        training_config = {k: v for k, v in config.items()
            if k not in ['device', 'accelerator', 'use_ddp', 'num_beams', 'eval_batch_size', 'run_local_time', 'run_id']}
        expected = {'model': case['model'], 'cache_hashes': load(CACHE / 'protocol-audit.json')['cache_hashes'],
            'sid_sha256': sha(sid), 'training_config': training_config, 'scheduler_steps': 310800,
            'code_hashes': {name: sha(BASE / name) for name in
                ['train_bounded.py', 'profile_full_models.py', 'training_runtime.py', 'run_controlled.py']}}
        context = {'case': case, 'model': model, 'tokenizer': tokenizer, 'raw': raw, 'mapping': mapping,
                   'epoch_steps': 2072, 'provenance': expected, 'state': state}
        if optimizer_needed:
            from transformers.optimization import get_scheduler
            optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
            scheduler = get_scheduler('cosine', optimizer, num_warmup_steps=10000, num_training_steps=310800)
            sampler = ShuffledBatches(530300, 256, case['seed'])
    elif case['family'] == 'sasrec':
        assert sha(SAS_SOURCE) == SAS_SHA
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        spec = importlib.util.spec_from_file_location('reproflow_sasrec', SAS_SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        recipe = case['recipe']
        config = SimpleNamespace(device='cuda', norm_first=recipe['norm_first'], maxlen=recipe['history'],
            hidden_units=recipe['dim'], num_blocks=2, num_heads=recipe['heads'], dropout_rate=recipe['dropout'])
        model = module.SASRec(len(raw), 25612, config).cuda()
        if not state and recipe['init'] == 'xavier':
            with torch.no_grad():
                for value in model.parameters():
                    if value.ndim > 1:
                        torch.nn.init.xavier_normal_(value)
                model.item_emb.weight[0].zero_()
                model.pos_emb.weight[0].zero_()
        epoch_steps = math.ceil(len(raw) / 128) if recipe['objective'] == 'BCE' else 2072
        expected = {'model': case['id'], 'source_sha256': SAS_SHA, 'recipe': recipe, 'seed': case['seed'],
                    'cache_hashes': load(CACHE / 'protocol-audit.json')['cache_hashes']}
        # Existing CE checkpoints keep their original provenance, independently checked.
        if state and case.get('legacy_ce'):
            expected = state['provenance']
            assert expected['source_sha256'] == SAS_SHA and expected['seed'] == case['seed']
            assert expected['config'] == vars(config)
            assert expected['cache_hashes'] == load(CACHE / 'protocol-audit.json')['cache_hashes']
        context = {'case': case, 'model': model, 'raw': raw, 'mapping': mapping,
                   'epoch_steps': epoch_steps, 'provenance': expected, 'state': state}
        if optimizer_needed:
            optimizer = torch.optim.Adam(model.parameters(), lr=.001,
                betas=(.9, .98) if recipe['objective'] == 'BCE' else (.9, .999))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
            sampler = ShuffledBatches(epoch_steps * 128 if recipe['objective'] == 'BCE' else 530300,
                                     128 if recipe['objective'] == 'BCE' else 256, case['seed'])
    else:
        raise ValueError('Unsupported model family')
    if state:
        assert state['provenance'] == expected, 'Model/data/recipe provenance mismatch'
        model.load_state_dict(state['model'], strict=True)
    if optimizer_needed:
        step = 0
        if state:
            from training_runtime import restore_state
            sampler, step = restore_state(state, model, optimizer, scheduler, expected)
        context.update(optimizer=optimizer, scheduler=scheduler, sampler=sampler, step=step)
    context['state'] = None
    del state
    return context


def sas_batch(context, examples):
    import numpy as np
    import torch
    length = context['case']['recipe']['history']
    rows = np.zeros((len(examples), length), dtype=np.int64)
    labels = []
    for index, values in enumerate(examples):
        values = [context['mapping']['item2id'][x] for x in values]
        rows[index, -(len(values) - 1):] = values[:-1]
        labels.append(values[-1])
    return rows, torch.tensor(labels, device='cuda')


def update(context):
    import numpy as np
    import torch
    from training_runtime import PrefixWindows
    model, optimizer = context['model'], context['optimizer']
    model.train()
    if context['case']['family'] == 'gen':
        from profile_full_models import train_step
        if 'windows' not in context:
            context['windows'] = PrefixWindows(context['raw'])
        result = train_step(model, optimizer, context['scheduler'], context['sampler'], context['windows'],
                            context['tokenizer'], torch.device('cuda'))
        context['step'] += 1
        return result
    recipe = context['case']['recipe']
    indices = context['sampler'].next()
    optimizer.zero_grad(set_to_none=True)
    if recipe['objective'] == 'CE':
        if 'windows' not in context:
            context['windows'] = PrefixWindows(context['raw'])
        examples = [context['windows'].example(i, recipe['history'])['item_seq'] for i in indices]
        inputs, labels = sas_batch(context, examples)
        features = model.log2feats(inputs)[:, -1, :]
        loss = torch.nn.functional.cross_entropy(features @ model.item_emb.weight[1:].T, labels - 1)
    else:
        if 'train_ids' not in context:
            context['train_ids'] = [[context['mapping']['item2id'][x] for x in s[:-2]] for s in context['raw'].values()]
            context['train_sets'] = [set(x) for x in context['train_ids']]
        size, length = len(indices), recipe['history']
        seq = np.zeros((size, length), dtype=np.int64)
        pos, neg = np.zeros_like(seq), np.zeros_like(seq)
        chosen = np.random.randint(0, len(context['train_ids']), size=size)
        for row, user in enumerate(chosen):
            values = context['train_ids'][user]
            excluded = context['train_sets'][user]
            nxt, column = values[-1], length - 1
            for item in reversed(values[:-1]):
                seq[row, column], pos[row, column] = item, nxt
                draw = np.random.randint(1, 25613)
                while draw in excluded:
                    draw = np.random.randint(1, 25613)
                neg[row, column] = draw
                nxt, column = item, column - 1
                if column < 0:
                    break
        positive, negative = model(chosen, seq, pos, neg)
        mask = np.where(pos != 0)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(positive[mask], torch.ones_like(positive[mask]))
        loss += torch.nn.functional.binary_cross_entropy_with_logits(negative[mask], torch.zeros_like(negative[mask]))
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    context['scheduler'].step()
    context['step'] += 1
    torch.cuda.synchronize()
    return {'loss': loss.item(), 'lr': optimizer.param_groups[0]['lr']}


def checkpoint(context, path):
    import torch
    from training_runtime import capture_state
    state = capture_state(context['model'], context['optimizer'], context['scheduler'],
                          context['sampler'], context['step'], context['provenance'], cuda=True)
    state['execution_adapter_sha256'] = sha(Path(__file__))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix('.pending')
    torch.save(state, pending)
    pending.replace(path)
    return {'path': str(path), 'sha256': sha(path), 'step': context['step']}


def predictions(context, users, split, beam=500, batch_size=4):
    import numpy as np
    import torch
    from flow_common import example
    context['model'].eval()
    raw, model = context['raw'], context['model']
    examples = [example(raw[u], split, None if context['case']['family'] == 'gen'
                         else context['case']['recipe']['history']) for u in users]
    with torch.no_grad():
        if context['case']['family'] == 'gen':
            from profile_full_models import make_batch
            batch = make_batch(context['tokenizer'], [{'user': u, 'item_seq': s} for u, s in zip(users, examples)],
                               torch.device('cuda'), False)
            model.config['num_beams'] = beam
            original = model.t5.generate
            def counted(*args, **kwargs):
                actual = kwargs['num_beams']
                # Preserve Latte's upstream agg_max retry for insufficient candidates.
                assert actual in ([beam, beam * 2] if context['case']['model'] == 'Latte' else [beam])
                calls = context.setdefault('beam_calls', {})
                calls[str(actual)] = calls.get(str(actual), 0) + 1
                return original(*args, **kwargs)
            model.t5.generate = counted
            try:
                codes = model.generate(batch, n_return_sequences=10)
            finally:
                model.t5.generate = original
            if codes.shape[-1] < 3:
                codes = torch.nn.functional.pad(codes, (0, 3 - codes.shape[-1]))
            if 'inverse' not in context:
                context['inverse'] = {tuple(code): item for item, code in context['tokenizer'].item2tokens.items()}
            candidates = [[context['inverse'].get(tuple(code)) for code in row] for row in codes.cpu().tolist()]
        else:
            inputs, _ = sas_batch(context, examples)
            scores = model.predict(None, inputs, np.arange(1, 25613))
            assert torch.isfinite(scores).all()
            ranked = (torch.argsort(scores, dim=-1, descending=True, stable=True)[:, :10] + 1).cpu().tolist()
            candidates = [[context['mapping']['id2item'][i] for i in row] for row in ranked]
    rows = []
    for user, items, values in zip(users, candidates, examples):
        target = values[-1]
        rank = next((i + 1 for i, item in enumerate(items) if item == target), None)
        rows.append({'user': user, 'target': target, 'candidate_slots': items, 'rank': rank})
    return rows
