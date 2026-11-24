"""Lazy upstream-equivalent training windows and complete single-device state."""
from bisect import bisect_right
import random


class PrefixWindows:
    def __init__(self, sequences):
        self.sequences = sequences
        self.users, self.ends = [], []
        total = 0
        for user, sequence in sequences.items():
            count = max(0, len(sequence) - 3)
            if count:
                self.users.append(user)
                total += count
                self.ends.append(total)
        if not total:
            raise ValueError('No training targets remain after holding out the last two events')

    def __len__(self):
        return self.ends[-1]

    def locate(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        user_index = bisect_right(self.ends, index)
        start = self.ends[user_index - 1] if user_index else 0
        return self.users[user_index], index - start + 1

    def example(self, index, max_history=20):
        user, target = self.locate(index)
        return {'user': user, 'item_seq': self.sequences[user][max(0, target - max_history):target + 1]}


class ShuffledBatches:
    def __init__(self, size, batch_size, seed):
        import torch
        if min(size, batch_size) <= 0:
            raise ValueError('Dataset size and batch size must be positive')
        self.size, self.batch_size = size, batch_size
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(size, generator=self.generator)
        self.position, self.epoch = 0, 0

    def next(self):
        import torch
        if self.position == self.size:
            self.order = torch.randperm(self.size, generator=self.generator)
            self.position, self.epoch = 0, self.epoch + 1
        end = min(self.position + self.batch_size, self.size)
        indices = self.order[self.position:end].tolist()
        self.position = end
        return indices

    def state_dict(self):
        return {'size': self.size, 'batch_size': self.batch_size, 'position': self.position,
                'epoch': self.epoch, 'order': self.order.clone(), 'generator': self.generator.get_state()}

    @classmethod
    def from_state(cls, state):
        import torch
        obj = cls.__new__(cls)
        for key in ['size', 'batch_size', 'position', 'epoch']:
            setattr(obj, key, state[key])
        assert 0 <= obj.position <= obj.size and len(state['order']) == obj.size
        obj.order = state['order'].clone()
        obj.generator = torch.Generator()
        obj.generator.set_state(state['generator'])
        return obj


def cpu_copy(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(cpu_copy(item) for item in value)
    if isinstance(value, list):
        return [cpu_copy(item) for item in value]
    return value


def capture_state(model, optimizer, scheduler, batches, step, provenance, cuda=False):
    import numpy as np
    import torch
    return {'format': 'latte-single-device-v1', 'model': cpu_copy(model.state_dict()),
            'optimizer': cpu_copy(optimizer.state_dict()), 'scheduler': scheduler.state_dict(),
            'sampler': batches.state_dict(), 'step': step, 'provenance': provenance,
            'torch_rng': torch.get_rng_state(), 'numpy_rng': np.random.get_state(),
            'python_rng': random.getstate(), 'cuda_rng': torch.cuda.get_rng_state_all() if cuda else None}


def restore_state(state, model, optimizer, scheduler, expected_provenance):
    import numpy as np
    import torch
    if state['format'] != 'latte-single-device-v1' or state['provenance'] != expected_provenance:
        raise ValueError('Checkpoint data/model/training provenance differs')
    model.load_state_dict(state['model'], strict=True)
    scheduler.load_state_dict(state['scheduler'])
    optimizer.load_state_dict(state['optimizer'])
    batches = ShuffledBatches.from_state(state['sampler'])
    # Model/optimizer creation must precede RNG restoration.
    torch.set_rng_state(state['torch_rng'])
    np.random.set_state(state['numpy_rng'])
    random.setstate(state['python_rng'])
    if state['cuda_rng'] is not None:
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    return batches, state['step']


def aggregate_bulk_transfer(parent_aggregation, tokens, scores, top_k):
    """Keep upstream sorting/filtering math; transfer whole tensors, rather than each beam."""
    device = tokens.device
    return parent_aggregation(tokens.detach().cpu(), scores.detach().cpu(), top_k).to(device)
