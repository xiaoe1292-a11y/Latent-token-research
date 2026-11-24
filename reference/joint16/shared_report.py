"""Full candidate audit and expanded-matrix analysis; no model modification."""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

from joint16 import s1, s2, PREVIOUS, OLD
from joint_stats import paired, pareto
import report2 as previous_report

KEYS = ('recall@10', 'recall@50', 'recall@100', 'ndcg@10')
MIB = 1048576


def close(a, b, tolerance=1e-10):
    assert math.isclose(a, b, abs_tol=tolerance, rel_tol=0), (a, b)


def create(root):
    import numpy as np
    from profile_worker import verify_cost_policy
    verify_cost_policy()
    plan, state = s1.load(root / 'protocol.json'), s1.load(root / 'state.json')
    assert state['status'] in ('experiments_completed', 'completed')
    s2.verify(plan)
    assert s1.sha(OLD / 'protocol.json') == plan['reused_protocol_sha256']
    assert s1.sha(OLD / 'state.json') == plan['reused_state_sha256']
    raw = s1.load(s1.PROCESSED / 'all_item_seqs.json')
    users = list(raw)
    catalogue = set(s1.load(s1.PROCESSED / 'id_mapping.json')['item2id'])
    data, audit, cohorts, arrays, repeat_audits = [], [], {}, {}, []
    sample_records = sample_paths = 0
    versions = set()
    for point in plan['points']:
        name, n, beam = point['id'], point['count'], point['beam']
        ps = state['points'][name]
        entries = [e for e in state['tasks'].values() if e['task']['point']['id'] == name]
        for e in entries:
            assert e['status'] == 'completed'
            if e.get('origin') == 'reused':
                out = Path(e['task']['output'])
                assert s1.sha(out / 'result.json') == e['evidence_result_sha256']
                if e['task']['kind'] == 'eval':
                    assert s1.sha(out / 'progress.json') == e['evidence_progress_sha256']
        pre_task = next(e['task'] for e in entries if e['task']['kind'] == 'preflight')
        pre = s1.load(Path(pre_task['output']) / 'result.json')
        assert len({r['user'] for r in pre['rows']}) == 4
        assert all(r['target'] == raw[r['user']][-2] for r in pre['rows'])
        sample_records += previous_report.audit_preflight(pre)
        sample_paths += sum(len(r['raw_beam_paths']) for r in pre['rows'])
        sums, position, ranks, returned = [0.0] * 4, 0, [], []
        path_sums = dict.fromkeys(('paths', 'invalid_latent', 'invalid_sid', 'valid_paths', 'unique_beam_items', 'duplicate_valid_paths'), 0)
        tasks = [e['task'] for e in entries if e['task']['kind'] == 'eval']
        for task in sorted(tasks, key=lambda t: t['start']):
            assert task['start'] == position
            result = s1.load(Path(task['output']) / 'result.json')
            assert result['task'] == task and result['checkpoint_sha256'] == point['checkpoint']['sha256']
            progress = s1.load(Path(task['output']) / 'progress.json')
            assert progress['committed'] == task['end'] - task['start']
            for part in progress['parts']:
                assert s1.sha(part['path']) == part['sha256']
                for row in s1.load(part['path']):
                    assert row['user'] == users[position] and row['target'] == raw[users[position]][-1]
                    slots = row['candidate_slots']
                    assert len(slots) == 100
                    present = [x for x in slots if x is not None]
                    assert len(present) == len(set(present)) and all(x in catalogue for x in present)
                    rank = next((i + 1 for i, item in enumerate(slots) if item == row['target']), None)
                    assert rank == row['rank']
                    for i, k in enumerate((10, 50, 100)):
                        sums[i] += int(rank is not None and rank <= k)
                    if rank is not None and rank <= 10:
                        sums[3] += 1 / math.log2(rank + 1)
                    d = row['path_diagnostics']
                    assert d['paths'] == beam == d['invalid_latent'] + d['invalid_sid'] + d['valid_paths']
                    assert d['duplicate_valid_paths'] == d['valid_paths'] - d['unique_beam_items']
                    assert len(present) == min(100, d['unique_beam_items'])
                    assert len(d['latent_beams']) == len(d['latent_valid_paths']) == n
                    total = sum(d['latent_beams'])
                    assert total == beam - d['invalid_latent']
                    assert sum(d['latent_valid_paths']) == d['valid_paths']
                    effective = math.exp(-sum(v / total * math.log(v / total) for v in d['latent_beams'] if v)) if total else 0
                    close(effective, d['effective_latents'])
                    for k in path_sums:
                        path_sums[k] += d[k]
                    ranks.append(rank or 101)
                    returned.append(len(present))
                    position += 1
            assert position == task['end']
        assert position == plan['test_users'] == ps['test']['users']
        for i, key in enumerate(KEYS):
            close(sums[i] / position, ps['test']['metrics'][key], 1e-12)
        for k, v in path_sums.items():
            assert v == ps['test']['path_sums'][k]
        ranks, returned = np.asarray(ranks, dtype=np.int16), np.asarray(returned, dtype=np.uint8)
        cohorts[name] = dict(ranks=ranks, returned=returned)
        arrays[name + '_rank'], arrays[name + '_returned'] = ranks, returned
        p = ps['profile']
        assert p['physical_gpu'] == state['profile_device']
        versions.add((p['torch_version'], p['cuda_version'], p['gpu_name'], p['precision']))
        times = p['timed_batch_seconds']
        assert p['actual_beam'] == beam and p['retry_calls'] == 0
        assert len(p['background_pids_by_batch']) == 768
        assert p['exclusive_gpu'] == (not p['background_pids_at_start'] and not any(p['background_pids_by_batch']))
        assert p['cost_policy_sha256'] == s1.sha(root / 'cost-policy-v1.json')
        assert p['measured_calls'] == len(times) == 768 and p['batch_size'] == 1
        close(statistics.mean(times) * 1000, p['latency_mean_ms'])
        close(statistics.median(times) * 1000, p['latency_p50_ms'])
        ordered, index = sorted(times), (len(times) - 1) * .95
        lo, hi = math.floor(index), math.ceil(index)
        p95 = ordered[lo] if lo == hi else ordered[lo] * (hi - index) + ordered[hi] * (index - lo)
        close(p95 * 1000, p['latency_p95_ms'])
        close(len(times) / sum(times), p['throughput_users_per_second'])
        assert p['peak_reserved_bytes'] >= p['peak_allocated_bytes'] >= p['baseline_allocated_bytes'] > 0
        full = returned == 100
        data.append(dict(point=name, n=n, beam=beam, quality_origin='reused' if name in plan['quality_reuse'] else 'new',
            checkpoint_sha256=point['checkpoint']['sha256'], evaluation_batch_size=2, profile_batch_size=1,
            profile_mode=p['profile_mode'], background_pids=sorted(set(p['background_pids_at_start']).union(*map(set,p['background_pids_by_batch']))),
            precision='float32', profile_gpu_uuid=p['physical_gpu']['uuid'], **ps['test']['metrics'],
            mean_returned=float(returned.mean()), min_returned=int(returned.min()),
            mean_unique_beam_items=path_sums['unique_beam_items'] / position,
            users_below100=int((~full).sum()), fraction_full100=float(full.mean()),
            recall100_on_full100=float((ranks[full] <= 100).mean()) if full.any() else None,
            valid_rate=path_sums['valid_paths'] / path_sums['paths'],
            unique_yield_rate=path_sums['unique_beam_items'] / path_sums['paths'],
            duplicate_rate_valid=path_sums['duplicate_valid_paths'] / path_sums['valid_paths'] if path_sums['valid_paths'] else 0,
            duplicate_rate_all=path_sums['duplicate_valid_paths'] / path_sums['paths'],
            invalid_sid_rate=path_sums['invalid_sid'] / path_sums['paths'],
            invalid_latent_rate=path_sums['invalid_latent'] / path_sums['paths'],
            mean_count_effective_latents=ps['test']['mean_count_effective_latents'],
            latency_p50_ms=p['latency_p50_ms'], latency_p95_ms=p['latency_p95_ms'],
            throughput_users_per_second=p['throughput_users_per_second'],
            peak_allocated_mib=p['peak_allocated_bytes'] / MIB, peak_reserved_mib=p['peak_reserved_bytes'] / MIB,
            repeat_p50_ms=[statistics.median(times[i:i + 256]) * 1000 for i in range(0, 768, 256)]))
        if name in plan['repeat_points']:
            repeat = ps['repeat']
            rows = repeat['reference_rows']
            digest = hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            assert repeat['repeat_hashes'] == [digest] * 3 and repeat['all_candidates_and_diagnostics_equal']
            assert [r['user'] for r in rows] == users[:256]
            assert all(r['target'] == raw[r['user']][-2] for r in rows)
            measure = s1.measure(rows)
            assert repeat['metrics_by_repeat'] == [measure] * 3
            assert repeat['actual_generate_calls'] == 384 and repeat['retry_calls'] == 0
            repeat_audits.append(dict(point=name, validation_users=256, passes=3, candidates_and_diagnostics_equal=True))
        audit.append(dict(point=name, users=position, quality_origin=data[-1]['quality_origin'], profile_batches=len(times)))
    assert len(versions) == 1
    np.savez_compressed(root / 'paired-records.npz', **arrays)
    comparisons = []
    pairs = [('beam', 'n%02d-b%04d' % (n, a), 'n%02d-b%04d' % (n, b))
             for n in plan['counts'] for a, b in zip(plan['beams'], plan['beams'][1:])]
    pairs += [('latent', 'n%02d-b%04d' % (a, b), 'n%02d-b%04d' % (c, b))
              for b in plan['beams'] for a, c in itertools.combinations(plan['counts'], 2)]
    for i, (kind, a, b) in enumerate(pairs):
        result = paired(cohorts[a], cohorts[b], plan['bootstrap']['seed'] + i, plan['bootstrap']['draws'])
        qa, qb = next(r for r in data if r['point'] == a), next(r for r in data if r['point'] == b)
        close(result['delta_recall100'], qb['recall@100'] - qa['recall@100'])
        comparisons.append(dict(kind=kind, before=a, after=b, **result,
            p50_ratio=qb['latency_p50_ms'] / qa['latency_p50_ms'], allocated_ratio=qb['peak_allocated_mib'] / qa['peak_allocated_mib']))
    s1.atomic(root / 'paired-bootstrap.json', comparisons)
    front = pareto(data)
    budgets = []
    for milliseconds, memory in ((60, 512), (75, 600), (100, 1100), (150, 2100)):
        for coverage in (0, .99, 1):
            eligible = [r for r in data if r['latency_p50_ms'] <= milliseconds and r['peak_allocated_mib'] <= memory and r['fraction_full100'] >= coverage]
            selected = max(eligible, key=lambda r: (r['recall@100'], -r['latency_p50_ms'])) if eligible else None
            budgets.append(dict(p50_budget_ms=milliseconds, allocated_budget_mib=memory,
                minimum_fraction_full100=coverage, selected=selected['point'] if selected else None,
                recall100=selected['recall@100'] if selected else None))
    s1.atomic(root / 'budget-recommendations.json', budgets)
    with (root / 'results.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)
    s1.atomic(root / 'completion-audit.json', dict(status='passed', points=audit,
        metric_user_point_records=sum(r['users'] for r in audit), profile_batches=sum(r['profile_batches'] for r in audit),
        raw_path_sample_records=sample_records, raw_path_sample_paths=sample_paths,
        repeat_audits=repeat_audits, profile_device=state['profile_device'], versions=list(versions)[0],
        cost_policy_sha256=s1.sha(root / 'cost-policy-v1.json'),
        cost_exclusive_points=sum(r['profile_mode']=='exclusive_process_checks' for r in data),
        cost_shared_points=sum(r['profile_mode']=='shared_gpu_observation' for r in data),
        pareto_recall100_p50=front, user_order_sha256=hashlib.sha256(json.dumps(users).encode()).hexdigest(),
        paired_records_sha256=s1.sha(root / 'paired-records.npz'), protocol_sha256=s1.sha(root / 'protocol.json')))
    render(root, data, front, plan)
    write_report(root, data, comparisons, front, budgets, plan)
    print(json.dumps(dict(status='passed', points=len(data), user_point_records=len(data) * plan['test_users'], profile_batches=768 * len(data))))


def render(root, data, front, plan):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    colors = {2: '#8455a5', 4: '#1765a1', 8: '#db7624', 16: '#16806a'}
    for filename, key, label in [('recall-vs-beam.png', 'recall@100', 'Full-test Recall@100'),
                                  ('unique-vs-beam.png', 'mean_unique_beam_items', 'Mean unique valid items in all beams')]:
        fig, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
        for n in plan['counts']:
            selected = [r for r in data if r['n'] == n]
            line, = axis.plot([r['beam'] for r in selected], [r[key] for r in selected], marker='o', color=colors[n], label='N=' + str(n))
            np.testing.assert_array_equal(line.get_ydata(), [r[key] for r in selected])
        if key == 'mean_unique_beam_items':
            axis.axhline(100, ls='--', color='#555555', lw=1, label='100 unique items')
        axis.set_xticks(plan['beams'])
        axis.set(xlabel='Fixed actual beam size (one generation)', ylabel=label, title='Latte: frozen epoch30 / seed2024')
        axis.legend()
        axis.grid(alpha=.25)
        fig.savefig(root / filename, dpi=180)
        plt.close(fig)
    fig, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    markers = {250: 'o', 500: 's', 1000: '^', 2000: 'D'}
    for r in data:
        axis.scatter(r['latency_p50_ms'], r['recall@100'], c=colors[r['n']], marker=markers[r['beam']], s=55,
                     edgecolors='black' if r['point'] in front else 'none', linewidths=1.2)
    selected = sorted([r for r in data if r['point'] in front], key=lambda r: r['latency_p50_ms'])
    axis.plot([r['latency_p50_ms'] for r in selected], [r['recall@100'] for r in selected], 'k--', alpha=.5, lw=1, label='Measured Pareto frontier')
    for i, r in enumerate(selected):
        axis.annotate('N%d/B%d' % (r['n'], r['beam']), (r['latency_p50_ms'], r['recall@100']),
                      xytext=(8, 12 if i % 2 == 0 else -20), textcoords='offset points', fontsize=8,
                      arrowprops=dict(arrowstyle='-', lw=.5))
    legend = [Line2D([], [], marker='o', ls='', color=colors[n], label='N=' + str(n)) for n in plan['counts']]
    legend += [Line2D([], [], marker=markers[b], ls='', color='gray', label='B=' + str(b)) for b in plan['beams']]
    axis.legend(handles=legend, loc='lower right', ncol=2, fontsize=8)
    axis.set(xlabel='Batch1 p50 latency (ms)', ylabel='Full-test Recall@100', title='Observed recall and latency under recorded GPU load (black: frontier)')
    axis.margins(x=.15, y=.18)
    axis.grid(alpha=.25)
    fig.savefig(root / 'pareto.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, key, title, fmt in zip(axes.flat, ['recall@100', 'ndcg@10', 'latency_p50_ms', 'mean_returned'],
            ['Recall@100', 'NDCG@10', 'Batch1 p50 latency (ms)', 'Returned unique items (max 100)'], ['.4f', '.5f', '.1f', '.1f']):
        matrix = np.array([[next(r[key] for r in data if r['n'] == n and r['beam'] == b) for b in plan['beams']] for n in plan['counts']])
        image = axis.imshow(matrix, aspect='auto', cmap='viridis')
        axis.set_xticks(range(len(plan['beams'])), plan['beams'])
        axis.set_yticks(range(len(plan['counts'])), plan['counts'])
        axis.set(xlabel='Fixed beam size', ylabel='Latent token types N', title=title)
        for i, j in np.ndindex(matrix.shape):
            color = 'white' if image.norm(matrix[i, j]) < .55 else 'black'
            axis.text(j, i, format(matrix[i, j], fmt), ha='center', va='center', color=color, fontsize=10)
        fig.colorbar(image, ax=axis, shrink=.85)
    fig.savefig(root / 'heatmap.png', dpi=180)
    plt.close(fig)
    files = ['recall-vs-beam.png', 'unique-vs-beam.png', 'pareto.png', 'heatmap.png']
    s1.atomic(root / 'figure-audit.json', dict(status='passed', memory_divisor=MIB, csv_sha256=s1.sha(root / 'results.csv'),
        source_sha256=s1.sha(__file__), image_sha256={name: s1.sha(root / name) for name in files}))


def write_report(root, data, comparisons, front, budgets, plan):
    findings = ['## 搜索收益、饱和与N=16的解释', '']
    for n in plan['counts']:
        cases = [r for r in data if r['n'] == n]
        best = max(cases, key=lambda r: r['recall@100'])
        deltas = [c for c in comparisons if c['kind'] == 'beam' and int(c['before'][1:3]) == n]
        normalized = [100 * c['delta_recall100'] * 1000 / (int(c['after'][-4:]) - int(c['before'][-4:])) for c in deltas]
        findings += ['N=%d：B250→500、500→1000、1000→2000的R@100边际收益折算为每新增1000 Beam分别为%+.4f、%+.4f、%+.4f个百分点。'
            '本轮该N最高R@100=%.6f，首次达到该点估计最高值的Beam为%d；这是当前矩阵观测，不是可外推的普适饱和点。' % (
                n, *normalized, best['recall@100'], best['beam']), '']
    lookup = {r['point']: r for r in data}
    if all(k in lookup for k in ('n08-b1000', 'n16-b1000', 'n08-b2000', 'n16-b2000')):
        a, b = lookup['n16-b1000'], lookup['n16-b2000']
        low_gap = lookup['n08-b1000']['recall@100'] - a['recall@100']
        high_gap = lookup['n08-b2000']['recall@100'] - b['recall@100']
        closure = 100 * (1 - high_gap / low_gap) if low_gap > 0 else float('nan')
        findings += ['N=16从B1000扩到2000，平均返回数由%.3f到%.3f，R@100由%.6f到%.6f。'
            '在两侧均匹配Beam的比较中，N8–N16的R@100差距从%.4f降到%.4f个百分点，缩小约%.2f%%。'
            '这支持低搜索预算和候选不足是先前高K差距的重要因素；它不是对“补空槽”机制的单独因果分解。'
            'N16的NDCG@10从%.6f到%.6f，仍须结合剩余质量差距、重复路径和成本解释，不能将候选恢复等同于更多latent更优。' % (
                a['mean_returned'], b['mean_returned'], a['recall@100'], b['recall@100'],
                low_gap*100, high_gap*100, closure, a['ndcg@10'], b['ndcg@10']), '']
    table = ['| N | B | R@10 | R@50 | R@100 | NDCG@10 | 返回唯一数 | 满100比例 | P50 ms | P95 ms | 用户/s | allocated MiB |',
             '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in data:
        table.append('| %d | %d | %.6f | %.6f | %.6f | %.6f | %.2f | %.2f%% | %.2f | %.2f | %.2f | %.1f |' % (
            r['n'], r['beam'], r['recall@10'], r['recall@50'], r['recall@100'], r['ndcg@10'], r['mean_returned'],
            100 * r['fraction_full100'], r['latency_p50_ms'], r['latency_p95_ms'], r['throughput_users_per_second'], r['peak_allocated_mib']))
    lines = ['# Latent Token × Beam Size 联合消融报告', '',
        '完成可用%d组配置，每组94762完整测试用户。N=2/4/8/16，B=250/500/1000/2000；冻结第一阶段epoch30 / step62160、seed2024，无重新训练或新增训练种子。' % len(data), '',
        '## 审查与可比性', '',
        '成本测量协议补充见cost-policy-v1.json：依据用户允许按现有资源同时运行的指令，优先选择空闲卡，否则固定一张卡完成16组测量。'
        '记录每批其他计算进程PID及每轮GPU利用率、空闲显存、SM时钟和温度。CSV的profile_mode区分通过独占进程检查与共享GPU观测。'
        '模型、实际Beam、输入、batch及精度保持一致，但共享负载无法完全控制，成本可比性受限；Pareto及预算表是该批次观测，不能解释为严格隔离性能或线上SLA。', '',
        '执行前审查见AUDIT.md与AUDIT-REMOTE.json；原代码和两阶段结果ZIP已备份。复用先前9组batch2/float32固定实际beam质量结果，补充7组；完整%d组候选均再次从原始ASIN目标独立复算。'
        '所有%d组成本在本轮同一物理RTX4090、统一batch1/float32和前256验证用户上重测，5次预热、3次测量重复，CUDA前后同步。'
        '每批只生成一次指定beam，沿用官方agg_max；质量评估只批量传输聚合输入，成本计时仍用官方GPU标量聚合。' % (len(data), len(data)), '',
        '推理计时含输入构造、H2D、生成、聚合与商品反解；不含模型加载、I/O及GPU占用检查。'
        '显存为PyTorch峰值allocated/reserved，MiB=1048576字节，不含CUDA context。GPU时钟未锁，微小延迟差异不视为稳定优势。', '',
        '固定低B无法保证每用户都有100个不同商品。本轮保留空槽和全体用户分母，不自动扩beam、不热门补齐。'
        '同时记录全beam不同商品数、截断后返回数及覆盖率；Bootstrap附两侧都满100的同用户子集诊断，子集为空则N/A。'
        '该诊断有条件筛选，不替代全量指标或作跨配置无偏排名。', '',
        *findings, '## 完整结果', '', *table, '', '![Recall curves](recall-vs-beam.png)', '', '![Heatmap](heatmap.png)', '',
        '## 路径质量', '',
        '| N | B | 全Beam平均唯一商品 | Valid Rate | 重复/有效路径 | 重复/全部Beam | 无效SID/全部Beam | 不足100用户 |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in data:
        lines.append('| %d | %d | %.2f | %.2f%% | %.2f%% | %.2f%% | %.2f%% | %d |' % (
            r['n'], r['beam'], r['mean_unique_beam_items'], 100*r['valid_rate'], 100*r['duplicate_rate_valid'],
            100*r['duplicate_rate_all'], 100*r['invalid_sid_rate'], r['users_below100']))
    lines += ['', 'Valid Rate=有效商品路径/全部Beam；重复路径=有效商品路径数−不同商品数，包含跨latent重复。'
        'Unique Yield=不同有效商品数/全部Beam，完整数值见CSV。计数熵只描述返回Beam中的latent类型分布，不代表完整后验或真实表示容量。', '',
        '![Unique candidates](unique-vs-beam.png)', '', '## 搜索收益与配对Bootstrap', '',
        '| N | B变化 | R@100变化/百分点 | 95%用户Bootstrap区间/百分点 | 新增/丢失命中 | P50倍数 | 两侧满100用户 | 子集差值/百分点 |',
        '|---:|---|---:|---|---|---:|---:|---:|']
    for c in comparisons:
        if c['kind'] != 'beam':
            continue
        n = int(c['before'][1:3])
        a, b = int(c['before'][-4:]), int(c['after'][-4:])
        subset = 'N/A' if c['delta_recall100_both_full'] is None else '%+.4f' % (100*c['delta_recall100_both_full'])
        lines.append('| %d | %d→%d | %+.4f | [%+.4f, %+.4f] | %d/%d | %.2f | %d | %s |' % (
            n, a, b, 100*c['delta_recall100'], 100*c['bootstrap_95_low'], 100*c['bootstrap_95_high'],
            c['gained_hits100'], c['lost_hits100'], c['p50_ratio'], c['both_full_users'], subset))
    lines += ['', '每对20000次用户配对Bootstrap，二元命中差使用与逐用户重采样等价的三类别多项抽样。'
        '区间条件于冻结模型，只描述用户抽样不确定性，不验证训练种子稳定性；36个预定比较为探索性分析，未作整体多重比较显著性声明。'
        '同B的全部N比较见paired-bootstrap.json。Beam结果不嵌套，新命中来自原先不足列表也不能证明收益完全由补空槽导致。', '',
        '## 资源约束下的配置', '', '记录负载条件下的Recall@100–p50观测Pareto集合：' + '、'.join(front) + '。', '',
        '![Pareto frontier](pareto.png)', '',
        '| P50预算 ms | allocated预算 MiB | 至少满100用户比例 | 本轮最高R@100可行配置 | R@100 |',
        '|---:|---:|---:|---|---:|']
    for b in budgets:
        lines.append('| %d | %d | %.0f%% | %s | %s |' % (b['p50_budget_ms'], b['allocated_budget_mib'],
            b['minimum_fraction_full100']*100, b['selected'] or '无可行配置', 'N/A' if b['recall100'] is None else '%.6f' % b['recall100']))
    lines += ['', '预算表只在本次16组测量内筛选，共享模式下为条件性参考；p50不等于p95或线上SLA；100%覆盖要求依据本次测试人群，不能保证新用户。'
        'Pareto依赖质量指标和成本维度，未将微小计时差异解释为稳定优势。', '',
        '## 证据与范围', '',
        '关键4组各256验证用户连续3遍推理的完整候选和诊断相同；所有16组都有3遍独立计时重复。'
        '全部候选指标独立重算、全量诊断计数守恒、样本原始路径与计时复算见completion-audit.json。'
        'paired-records.npz保留每用户rank及返回数，以便本地重算指标和Bootstrap；它不包含完整候选ASIN列表。'
        'Checkpoint缺失记录见missing-checkpoints.json，本轮无训练。', '',
        '本轮复用已查看的测试集，属于冻结模型上的扩展分析。只讨论当前矩阵的搜索收益、局部饱和与资源权衡；'
        '不开展Data Scaling、复杂曲线拟合或多训练种子验证，不将局部边际收益递减称为Scaling Law。', '']
    (root / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    create(parser.parse_args().root)
