"""Frozen scientific protocol, parity-checked evaluation and verified report units."""
import inspect
import stage1
from stage1 import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    record = load(args.root / 'delivery-runtime-v3.json')
    assert record['scientific_protocol_sha256'] == sha(args.root / 'protocol.json')
    for path, digest in record['source_hashes'].items():
        assert sha(path) == digest, 'Delivery runtime changed'
    source = inspect.getsource(stage1.manager)
    replacements = {"str(HERE / 'stage1.py'), 'worker'": "str(HERE / 'fast_worker.py'), 'worker'",
                    "str(HERE / 'report.py')": "str(HERE / 'report_checked.py')"}
    for old, new in replacements.items():
        assert source.count(old) == 1
        source = source.replace(old, new)
    exec(compile(source, str(HERE / 'final_manager.py'), 'exec'), globals())
    manager(args.root)
