"""Independent full-user metric audit, raw-path spot audit and fixed-beam report."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

from inference2 import s1, FIRST

MIB = 1048576
KEYS = ('recall@10', 'recall@50', 'recall@100', 'ndcg@10')


def audit_preflight(result):
    point = result['task']['point']
    assert result['actual_gpu_parity'] and result['official_fixed_beam_candidate_equality']
    assert s1.sha(result['token_map']['path']) == result['token_map']['sha256']
    inverse = {tuple(row['tokens']): row['item'] for row in s1.load(result['token_map']['path'])}
    assert len(inverse) == 25612
    for row in result['rows']:
        paths = row['raw_beam_paths']
        assert len(paths) == point['beam']
        accepted = [p for p in paths if p and 1 <= p[0] <= point['count']]
        known = [(p[0], inverse[tuple(p[1:])]) for p in accepted if tuple(p[1:]) in inverse]
        d = row['path_diagnostics']
        assert d['invalid_latent'] == len(paths) - len(accepted)
        assert d['invalid_sid'] == len(accepted) - len(known)
        assert d['valid_paths'] == len(known)
        assert d['unique_beam_items'] == len({item for _, item in known})
        assert d['duplicate_valid_paths'] == len(known) - len({item for _, item in known})
        assert d['latent_beams'] == [sum(p[0] == i for p in accepted) for i in range(1, point['count'] + 1)]
        assert d['latent_valid_paths'] == [sum(z == i for z, _ in known) for i in range(1, point['count'] + 1)]
    return len(result['rows'])


def create(root):
    plan, state = s1.load(root / 'protocol.json'), s1.load(root / 'state.json')
    assert state['status'] in ('experiments_completed', 'completed')
    raw = s1.load(s1.PROCESSED / 'all_item_seqs.json')
    users = list(raw)
    catalogue = set(s1.load(s1.PROCESSED / 'id_mapping.json')['item2id'])
    audit, data, paired, raw_users, distinct_raw_users = [], [], {}, 0, set()
    for point in plan['points']:
        name, n, beam = point['id'], point['count'], point['beam']
        ps = state['points'][name]
        assert 'test' in ps and 'profile' in ps
        pre = next(v['task'] for v in state['tasks'].values() if v['task']['kind'] == 'preflight' and v['task']['point']['id'] == name)
        pre_result = s1.load(Path(pre['output']) / 'result.json')
        raw_users += audit_preflight(pre_result)
        distinct_raw_users.update(row['user'] for row in pre_result['rows'])
        assert all(row['target'] == raw[row['user']][-2] for row in pre_result['rows'])
        tasks = [v['task'] for v in state['tasks'].values() if v['task']['kind'] == 'eval' and v['task']['point']['id'] == name]
        sums, position, path_totals, returned, ranks = [0.0] * 4, 0, 0, [], []
        unique_total = valid_paths = duplicate_paths = bad_latent = bad_sid = 0
        for task in sorted(tasks, key=lambda value: value['start']):
            assert task['start'] == position
            progress = s1.load(Path(task['output']) / 'progress.json')
            for part in progress['parts']:
                assert s1.sha(part['path']) == part['sha256']
                for row in s1.load(part['path']):
                    assert row['user'] == users[position] and row['target'] == raw[users[position]][-1]
                    slots = row['candidate_slots']
                    assert len(slots) == 100
                    present = [item for item in slots if item is not None]
                    assert len(present) == len(set(present)) and all(item in catalogue for item in present)
                    found = [i + 1 for i, item in enumerate(slots) if item == row['target']]
                    rank = found[0] if found else None
                    assert rank == row['rank']
                    for i, k in enumerate((10, 50, 100)):
                        sums[i] += int(rank is not None and rank <= k)
                    if rank is not None and rank <= 10:
                        sums[3] += 1 / math.log2(rank + 1)
                    d = row['path_diagnostics']
                    assert d['paths'] == beam == d['invalid_latent'] + d['invalid_sid'] + d['valid_paths']
                    assert d['unique_beam_items'] - d['valid_paths'] + d['duplicate_valid_paths'] == 0
                    assert len(present) == min(100, d['unique_beam_items'])
                    assert sum(d['latent_beams']) == beam - d['invalid_latent']
                    assert sum(d['latent_valid_paths']) == d['valid_paths']
                    total_latent = sum(d['latent_beams'])
                    entropy = -sum((count / total_latent) * math.log(count / total_latent) for count in d['latent_beams'] if count) if total_latent else 0
                    expected_effective = math.exp(entropy) if total_latent else 0
                    assert math.isclose(expected_effective, d['effective_latents'], abs_tol=1e-10)
                    path_totals += d['paths']
                    unique_total += d['unique_beam_items']
                    valid_paths += d['valid_paths']
                    duplicate_paths += d['duplicate_valid_paths']
                    bad_latent += d['invalid_latent']
                    bad_sid += d['invalid_sid']
                    ranks.append(rank or 101)
                    returned.append(len(present))
                    position += 1
        assert position == plan['test_users'] == ps['test']['users']
        for i, key in enumerate(KEYS):
            assert math.isclose(sums[i] / position, ps['test']['metrics'][key], abs_tol=1e-12, rel_tol=0)
        expected_sums = ps['test']['path_sums']
        for key, value in [('paths', path_totals), ('unique_beam_items', unique_total), ('valid_paths', valid_paths),
                           ('duplicate_valid_paths', duplicate_paths), ('invalid_latent', bad_latent), ('invalid_sid', bad_sid)]:
            assert value == expected_sums[key]
        profile = ps['profile']
        times = profile['timed_batch_seconds']
        assert profile['actual_beam'] == beam and profile['retry_calls'] == 0 and profile['exclusive_gpu']
        assert profile['measured_calls'] == len(times) == 768 and all(t > 0 for t in times)
        assert math.isclose(statistics.mean(times) * 1000, profile['latency_mean_ms'], abs_tol=1e-8)
        assert math.isclose(statistics.median(times) * 1000, profile['latency_p50_ms'], abs_tol=1e-8)
        ordered, index = sorted(times), (len(times) - 1) * .95
        lo, hi = math.floor(index), math.ceil(index)
        p95 = ordered[lo] if lo == hi else ordered[lo] * (hi - index) + ordered[hi] * (index - lo)
        assert math.isclose(p95 * 1000, profile['latency_p95_ms'], abs_tol=1e-8)
        assert math.isclose(len(times) / sum(times), profile['throughput_users_per_second'], abs_tol=1e-10)
        assert profile['peak_reserved_bytes'] >= profile['peak_allocated_bytes'] >= profile['baseline_allocated_bytes'] > 0
        paired[name] = dict(ranks=ranks, returned=returned)
        data.append(dict(point=name, n=n, beam=beam, nominal_beams_per_latent=beam / n,
            **ps['test']['metrics'], mean_returned=ps['test']['mean_unique_candidates'],
            mean_unique_beam_items=ps['test']['mean_unique_items_in_beam'],
            users_below100=ps['test']['insufficient_users']['100'],
            duplicate_fraction=ps['test']['duplicate_fraction_of_valid_paths'],
            invalid_latent_fraction=ps['test']['invalid_latent_fraction'], invalid_sid_fraction=ps['test']['invalid_sid_fraction'],
            mean_count_effective_latents=ps['test']['mean_count_effective_latents'],
            latency_p50_ms=profile['latency_p50_ms'], latency_p95_ms=profile['latency_p95_ms'],
            throughput_users_per_second=profile['throughput_users_per_second'],
            peak_allocated_mib=profile['peak_allocated_bytes'] / MIB, peak_reserved_mib=profile['peak_reserved_bytes'] / MIB))
        audit.append(dict(point=name, independently_verified_users=position, diagnostic_consistency_users=position,
                          profile_batches=len(times), fixed_beam=beam, retries=0))
    changes = []
    for n in plan['counts']:
        for before, after in [(500, 1000), (1000, 2000)]:
            a, b = paired['n%02d-b%04d' % (n, before)], paired['n%02d-b%04d' % (n, after)]
            gained = [i for i, (x, y) in enumerate(zip(a['ranks'], b['ranks'])) if x > 100 and y <= 100]
            lost = sum(x <= 100 and y > 100 for x, y in zip(a['ranks'], b['ranks']))
            before_metric = next(row['recall@100'] for row in data if row['n'] == n and row['beam'] == before)
            after_metric = next(row['recall@100'] for row in data if row['n'] == n and row['beam'] == after)
            assert abs((len(gained) - lost) / plan['test_users'] - (after_metric - before_metric)) < 1e-12
            changes.append(dict(n=n, before_beam=before, after_beam=after, gained_hits100=len(gained), lost_hits100=lost,
                gained_from_previously_short=sum(a['returned'][i] < 100 for i in gained),
                gained_from_previously_full=sum(a['returned'][i] == 100 for i in gained),
                net_recall100_change=(len(gained) - lost) / plan['test_users']))
    s1.atomic(root / 'completion-audit.json', dict(status='passed', points=audit,
        metric_user_point_records=sum(row['independently_verified_users'] for row in audit),
        diagnostic_raw_path_audit_user_point_records=raw_users, diagnostic_raw_path_distinct_users=len(distinct_raw_users),
        note='all metrics independently recomputed; all diagnostic count consistency checked; raw paths spot-audited on preflight cohorts',
        protocol_sha256=s1.sha(root / 'protocol.json')))
    s1.atomic(root / 'paired-beam-changes.json', changes)
    with (root / 'results.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)
    render(root, data, plan)
    table = ['| n | Fixed beam | R@10 | R@50 | R@100 | NDCG@10 | Mean candidates | <100 users | p50 ms | Users/s | Peak MiB |',
             '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in data:
        table.append('| %d | %d | %.6f | %.6f | %.6f | %.6f | %.2f | %d | %.2f | %.2f | %.1f |' % (
            row['n'], row['beam'], row['recall@10'], row['recall@50'], row['recall@100'], row['ndcg@10'],
            row['mean_returned'], row['users_below100'], row['latency_p50_ms'], row['throughput_users_per_second'], row['peak_allocated_mib']))
    lines = ['# 第二阶段：固定Beam预算与潜在路径验证', '',
        '状态：9组全量测试、单请求推理测量与独立审计已完成。保持seed2024和第一阶段epoch30权重，没有新增训练种子或重新训练。', '',
        '## 协议与比较口径', '',
        'Amazon Reviews 2023 / Video_Games，94762用户完整test，目录25612商品；原SID、用户顺序和留出目标不变。'
        'n=4/8/16 × beam=500/1000/2000，每次请求只调用一次指定beam，关闭自动扩beam重试，agg_max保持官方实现。'
        '各beam独立执行搜索，较大beam的结果并不保证包含较小beam结果。', '',
        '质量评估batch2，使用已与官方GPU聚合逐项核对的批量CPU传输；推理基准使用官方GPU标量聚合、batch1、前256验证用户、5次预热、3次重复，CUDA前后同步。'
        '计时含预处理/H2D/生成/聚合/ASIN反解，不含加载、I/O与GPU占用检查。共享节点未锁GPU时钟，推理测量要求无其他计算进程。显存为PyTorch峰值allocated/reserved，不含CUDA context。', '',
        '这是复用第一阶段已查看测试集的扩展分析；配置在本轮开始前冻结，测试结果不用于选择checkpoint。'
        '结论范围是这三份冻结模型与当前数据/搜索策略，未开展多种子稳定性验证，不表述为Scaling Law。', '',
        '## 全量结果', '', *table, '', '![Quality](quality-vs-beam.png)', '', '![Coverage](coverage.png)', '',
        '![Cost](cost.png)', '', '![Tradeoff](tradeoff.png)', '', '## 相同模型内增大beam', '',
        '| n | Beam change | Gained R@100 hits | Lost hits | Gain from short list | Gain from full list | Net R@100 change |',
        '|---:|---|---:|---:|---:|---:|---:|']
    for row in changes:
        lines.append('| %d | %d→%d | %d | %d | %d | %d | %+.6f |' % (row['n'], row['before_beam'], row['after_beam'],
            row['gained_hits100'], row['lost_hits100'], row['gained_from_previously_short'], row['gained_from_previously_full'], row['net_recall100_change']))
    lines += ['', '新增命中不全等于补齐空槽：beam变化还会改变搜索剪枝和top100排序。上表区分此前候选不足与已满100的用户，避免把全部收益归因于单一机制。', '',
        '## 名义每潜在种类搜索预算相近的对角线', '',
        '以下B/n均为125，但beam并未强制平均分配到各latent；它是名义预算对照，不是实际相同路径分配。', '', *table[:2]]
    for row in data:
        if row['beam'] / row['n'] == 125:
            lines.append(table[2 + data.index(row)])
    lines += ['', '## 路径诊断的解释', '',
        '每条解码路径分为无效latent、有效latent但SID不在目录、有效商品路径。有效路径中的重复商品数量与全beam不同商品数分别记录。'
        'effective_latents由返回beam的latent计数熵计算，反映当前搜索结果中的类型分布，不是模型完整后验概率或真实表示容量。'
        '全体用户均检查统计守恒；预检4个不同用户×9组共36条用户—配置记录另保存原始beam路径与token映射进行独立复算，未声称全量原始路径已逐条独立复算。', '',
        '完整指标独立核验、逐点计时和样本诊断证据见completion-audit.json、state.json、preflight结果与results.csv。']
    (root / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(dict(status='passed', points=len(data), users_per_point=94762, raw_diagnostic_users=raw_users)))


def render(root, data, plan):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    colors = {4: '#1765a1', 8: '#db7624', 16: '#16806a'}
    figures = []
    for filename, keys, labels, shape in [
        ('quality-vs-beam.png', list(KEYS), ['Recall@10', 'Recall@50', 'Recall@100', 'NDCG@10'], (2, 2)),
        ('coverage.png', ['mean_returned', 'mean_unique_beam_items', 'duplicate_fraction', 'mean_count_effective_latents'],
         ['Returned unique items (max 100)', 'Unique items in all decoded beams', 'Duplicate / valid decoded paths', 'Count-entropy effective latent types'], (2, 2)),
        ('cost.png', ['latency_p50_ms', 'throughput_users_per_second', 'peak_allocated_mib'],
         ['Batch1 p50 latency (ms)', 'Throughput (users/s)', 'Peak allocated (MiB)'], (1, 3))]:
        fig, axes = plt.subplots(*shape, figsize=(12, 7 if shape[0] == 2 else 4), constrained_layout=True)
        for axis, key, label in zip(np.asarray(axes).flat, keys, labels):
            for n in plan['counts']:
                rows = [row for row in data if row['n'] == n]
                line, = axis.plot([row['beam'] for row in rows], [row[key] for row in rows], marker='o', color=colors[n], label='n=' + str(n))
                np.testing.assert_allclose(line.get_ydata(), [row[key] for row in rows], rtol=0, atol=1e-12)
                if key == 'peak_allocated_mib':
                    for row, value in zip(rows, line.get_ydata()):
                        assert abs(value - s1.load(root / 'state.json')['points'][row['point']]['profile']['peak_allocated_bytes'] / MIB) < 1e-9
            axis.set_xticks(plan['beams'])
            axis.set(xlabel='Fixed actual beam size (no retry)', ylabel=label)
            axis.grid(alpha=.25)
            axis.legend()
        fig.suptitle('Latte / Video_Games: frozen epoch30, seed2024')
        fig.savefig(root / filename, dpi=180)
        plt.close(fig)
        figures.append(filename)
    fig, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for row in data:
        axis.scatter(row['latency_p50_ms'], row['recall@100'], color=colors[row['n']])
        offset = {4: (5, 6), 8: (5, -14), 16: (5, 6)}[row['n']]
        axis.annotate('n%d/B%d' % (row['n'], row['beam']), (row['latency_p50_ms'], row['recall@100']), xytext=offset, textcoords='offset points', fontsize=8)
    axis.set(xlabel='Batch1 p50 latency (ms)', ylabel='Full-test Recall@100', title='Fixed-beam quality and inference cost')
    axis.margins(x=.18, y=.18)
    axis.grid(alpha=.25)
    fig.savefig(root / 'tradeoff.png', dpi=180)
    plt.close(fig)
    figures.append('tradeoff.png')
    s1.atomic(root / 'figure-audit.json', dict(status='passed', memory_divisor=MIB,
        image_sha256={name: s1.sha(root / name) for name in figures}, source_sha256=s1.sha(__file__)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    create(parser.parse_args().root)
