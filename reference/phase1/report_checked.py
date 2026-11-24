"""Correct and verify MiB plot units without editing frozen execution sources."""
import argparse
import inspect
from pathlib import Path

import report
import stage1


def verify_axis(axis, expected):
    import numpy as np
    assert len(axis.lines) == len(expected)
    checks = []
    for line, values in zip(axis.lines, expected):
        actual = np.asarray(line.get_ydata(), dtype=float)
        np.testing.assert_allclose(actual, values, rtol=0, atol=1e-9,
                                   err_msg='Memory chart does not match allocator bytes / 1048576')
        checks.append(dict(actual_mib=actual.tolist(), expected_mib=list(values)))
    return checks


def run(root):
    state, plan = stage1.load(root / 'state.json'), stage1.load(root / 'protocol.json')
    assert state['status'] == 'completed'
    expected = [[next(row for row in state['cases'][str(n)]['profile']['measurements']
                     if row['batch_size'] == batch)['peak_allocated_bytes'] / 1048576
                 for n in plan['latent_counts']] for batch in (1, 4)]
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    source = inspect.getsource(report.create)
    assert source.count('1024**-20') == 1, 'Review original plot conversion before applying correction'
    source = source.replace('1024**-20', '1024**-2')
    exec(compile(source, str(Path(__file__)), 'exec'), report.__dict__)
    original = Figure.savefig
    checks = []

    def checked_save(figure, filename, *args, **kwargs):
        if Path(filename).name == 'efficiency.png':
            assert len(figure.axes) == 3
            checks.extend(verify_axis(figure.axes[2], expected))
        return original(figure, filename, *args, **kwargs)

    Figure.savefig = checked_save
    try:
        report.create(root)
    finally:
        Figure.savefig = original
    assert len(checks) == 2
    stage1.atomic(root / 'figure-units-audit.json', dict(status='passed', memory_chart_unit='MiB',
        divisor_bytes=1048576, series=checks, correction='1024**-20 corrected to 1024**-2',
        original_source_sha256=stage1.sha(report.__file__), corrected_runtime_sha256=stage1.sha(__file__)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    run(parser.parse_args().root)
