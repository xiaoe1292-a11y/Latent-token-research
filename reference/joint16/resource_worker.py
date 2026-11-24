"""Resource-only admission adjustment based on measured batch2 preflight peaks."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time
import traceback

import joint16 as original

HERE = Path(__file__).resolve().parent
POLICY = None


def verify_policy():
    global POLICY
    POLICY = original.s1.load(HERE / 'resource-policy-v1.json')
    for name, digest in POLICY['sources'].items():
        assert original.s1.sha(HERE / name) == digest
    assert original.s1.sha(HERE / 'joint16.py') == POLICY['original_controller_sha256']
    return POLICY


def required_memory(task):
    policy = POLICY or verify_policy()
    if task['kind'] in ('profile', 'preflight'):
        return 7168
    return policy['points'][task['point']['id']]['required_free_mib']


def worker(path, gpu):
    task = original.s1.load(path)
    policy = verify_policy()
    assert original.s1.sha(Path(task['root']) / 'protocol.json') == policy['protocol_sha256']
    sys.path[:0] = [str(original.s1.BASE), str(original.s1.FLOW)]
    import flow_models
    native = flow_models.prepare_runtime
    source = inspect.getsource(native)
    old = 'int(free) < 7168'
    assert source.count(old) == 1
    minimum = required_memory(task)
    updated = source.replace(old, 'int(free) < ' + str(minimum)).replace('Less than 7GiB free', 'Below measured-peak admission reserve')
    namespace = dict(flow_models.__dict__)
    exec(compile(updated, 'measured-peak-resource-guard', 'exec'), namespace)
    admitted = namespace['prepare_runtime']

    def tracked(gpu):
        result = admitted(gpu)
        result.update(required_free_mib=minimum,
                      calibration=policy['points'][task['point']['id']],
                      resource_policy_sha256=original.s1.sha(HERE / 'resource-policy-v1.json'),
                      guard_source_sha256=hashlib.sha256(updated.encode()).hexdigest())
        return result

    flow_models.prepare_runtime = tracked
    try:
        original.worker(path, gpu)
    finally:
        flow_models.prepare_runtime = native


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
