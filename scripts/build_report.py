# -*- coding: utf-8 -*-
"""Generate the portable Chinese report from verified published records."""
import argparse
import os
from pathlib import Path

from common import ROOT, COUNTS, BEAMS, read_json, rows
from statistics_utils import pareto
from verify_results import verify


def build(output):
    verify(ROOT)
    data=rows(ROOT)
    comparisons=read_json(ROOT/'data/joint16/paired-bootstrap.json')
    selected=next(c for c in comparisons if c['before']=='n04-b1000' and c['after']=='n08-b1000')
    lines=['# Latte Latent Token × Beam Size 实验报告','',
        '实验完成日期：2026-11-24。本报告从公开的数值记录生成，不启动训练或GPU推理。', '',
        '## 研究问题与数据', '',
        '研究固定训练权重下，latent token种类数与实际搜索预算如何共同影响商品召回、候选覆盖和推理成本。'
        '上游为hyp1231/Latte，固定commit `05e4e6d983225bcb7172f148a076890e80c524d1`。'
        '数据为Amazon Reviews2023 / Video_Games，5-core、全评分，94762用户、25612商品；逐用户last-out，'
        '验证目标为倒数第二项，测试目标为最后一项。训练前缀530300条、历史20；每组seed2024、固定30 epoch /62160 updates。', '',
        'Semantic ID由Sentence-T5-base、PCA192及3×256 RQKMeans和冲突修复构造。PCA使用预先可用的完整商品内容目录；'
        '量化仅拟合25527个训练商品，85个heldout-only商品仅编码。模型为encoder4/decoder4、d_model128、d_ff1024。'
        '固定预算不保证收敛，也不等于论文最终数字复现。', '',
        '## 审查与实验协议', '',
        'N=2/4/8/16，实际B=250/500/1000/2000，共16点；每点94762完整test用户，质量batch2、FP32。'
        '复用四份Checkpoint及历史9组可比质量，补7组质量；本轮无训练、新训练种子或模型结构改动。'
        '每次请求只执行一次指定Beam，关闭官方不足候选时的自动扩大重试。agg_max按商品保留最高分路径并去重。'
        '不足100时保留空槽，全体用户仍在分母中；不补热门或过滤已见商品。', '',
        '每用户单目标，因此Recall@K=HitRate@K；NDCG@10未命中为0，命中排名r时为1/log2(r+1)。'
        'Valid Rate为有效商品路径/全部Beam；重复率同时报告重复路径/有效路径、重复路径/全部Beam。'
        '全Beam唯一商品数与截断后返回商品数分别记录。', '',
        '所有16组成本在同一物理RTX4090重新测量：batch1、FP32、5次预热、前256验证用户×3遍，CUDA前后同步，'
        '共12288次计时。计时含输入构造、H2D、一次生成、官方GPU标量聚合和商品反解，不含模型加载、I/O及GPU占用检查。'
        '显存为PyTorch峰值allocated/reserved，MiB=1048576字节，不含CUDA context和其他进程。', '',
        '**全部16组均为shared_gpu_observation。** 逐批背景进程检查证明存在其他计算进程；时钟没有锁定。'
        '因此成本、Pareto和预算建议属于本批条件性观测，严格隔离性能与线上SLA均未验证。原始背景PID不随公开包分发。', '',
        '## 全部16组结果', '',
        '| N | B | R@10 | R@50 | R@100 | NDCG@10 | 平均返回数 | 满100用户 | P50 ms | P95 ms | 用户/s | allocated MiB |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in data:
        lines.append('| %d | %d | %.6f | %.6f | %.6f | %.6f | %.3f | %.3f%% | %.2f | %.2f | %.2f | %.1f |'%(
            r['n'],r['beam'],r['recall@10'],r['recall@50'],r['recall@100'],r['ndcg@10'],r['mean_returned'],
            100*r['fraction_full100'],r['latency_p50_ms'],r['latency_p95_ms'],r['throughput_users_per_second'],r['peak_allocated_mib']))
    lines+=['','![结果矩阵](../assets/figures/heatmap.png)','',
        '## 搜索收益与局部饱和','',
        '相同实际Beam下，R@100点估计最高的N依次为：B250→N2、B500→N4、B1000→N8、B2000→N8。'
        '这些是四份冻结权重的点估计排名，不代表跨训练种子的稳定优势。', '',
        '| N | Beam变化 | R@100变化/百分点 | 配对95%区间/百分点 | 新增/丢失命中 | 观测P50倍数 | 两侧均满100用户 |',
        '|---:|---|---:|---|---|---:|---:|']
    for c in comparisons:
        if c['kind']=='beam':
            lines.append('| %d | %d→%d | %+.4f | [%+.4f, %+.4f] | %d/%d | %.3f | %d |'%(
                int(c['before'][1:3]),int(c['before'][-4:]),int(c['after'][-4:]),100*c['delta_recall100'],
                100*c['bootstrap_95_low'],100*c['bootstrap_95_high'],c['gained_hits100'],c['lost_hits100'],c['p50_ratio'],c['both_full_users']))
    lines+=['','N2在B500之后只净增1个命中，B1000→2000无净提升；N4在B1000→2000保持相同R@100。'
        'N8的B1000→2000净增28个命中、R@100增0.0295个百分点，观测P50增加31.85%、allocated增加95.38%。'
        '这是当前四档搜索预算中的局部边际收益递减，不能拟合或外推为普适Scaling Law。','',
        'N16的B1000→2000使R@100从0.215603升至0.235865，平均返回数86.173→99.981；'
        '与匹配相同Beam的N8的差距从2.3607缩小至0.3641个百分点，约缩小84.58%。'
        '搜索预算/覆盖不足是重要影响因素，但Beam结果不嵌套，改变搜索也会改变原有候选与排名；'
        '不能把收益全部归因于补空槽。N16的NDCG@10仍约0.034939，低于N8约0.035778。','',
        '![召回曲线](../assets/figures/recall-vs-beam.png)','',
        '## Bootstrap的解释边界','',
        '每对比较使用20000次配对用户有放回重采样；对二元命中差采用数学上等价的“新增、丢失、不变”三类别多项抽样。'
        '抽样种子是统计重采样种子，不是新增训练种子。36项对照包含12项相邻Beam及24项同Beam的N比较。'
        '区间条件于冻结模型，为未作整体多重校正的探索性用户抽样区间。', '',
        '特别地，在B1000下，N8−N4的R@100差为%+.4f个百分点，95%%区间[%+.4f,%+.4f]个百分点，包含0；'
        '新增/丢失命中为%d/%d。N8是点估计更高，不能宣称其可靠优于N4。'%(100*selected['delta_recall100'],
            100*selected['bootstrap_95_low'],100*selected['bootstrap_95_high'],selected['gained_hits100'],selected['lost_hits100']), '',
        '## 有效路径、重复商品与覆盖','',
        '| N | B | 全Beam平均唯一商品 | Valid Rate | 重复/有效路径 | 无效SID/全部Beam | 不足100用户 |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for r in data:
        lines.append('| %d | %d | %.3f | %.3f%% | %.3f%% | %.3f%% | %d |'%(
            r['n'],r['beam'],r['mean_unique_beam_items'],100*r['valid_rate'],100*r['duplicate_rate_valid'],100*r['invalid_sid_rate'],r['users_below100']))
    lines+=['','N16/B1000的有效路径比例约99.77%，重复/有效路径约91.36%；主要现象是重复商品多，而非大量无效SID。'
        '均值接近100不等于人人满100；条件筛选“两侧均满100”的子集诊断不能替代全量比较。','',
        '![唯一候选](../assets/figures/unique-vs-beam.png)','',
        '## 观测Pareto与预算选择','',
        '未加覆盖约束的R@100–P50观测Pareto集合：'+', '.join(pareto(data))+'。','',
        '![观测Pareto](../assets/figures/pareto.png)','',
        '默认建议至少99%测试用户满100商品。按本轮观测，N4/B1000可作为均衡候选；约600 MiB allocated预算可考虑N4/B500；'
        '原定100 ms P50预算内选择N2/B500，150 ms内选择N8/B1000。追求最高点估计则为N8/B2000。'
        '若要求本次全体测试用户均满100，150 ms预算的选择变为N2/B1000。完整预设预算见[预算JSON](../data/joint16/budget-recommendations.json)。', '',
        'N2/B250的观测P50=95.63 ms，N2/B500反而为87.23 ms；N16/B1000三遍P50约107.87/110.03/137.97 ms。'
        '这些非单调成本和轮间波动不能直接归因于算法。相同硬件、稳定PID或相同三遍前沿成员都不足以证明背景负载完全一致。'
        '不得把本轮结果称为严格隔离部署最优或线上延迟保证。','',
        '## 历史、证据与限制','',
        '第一阶段N=1/2/4/8/16使用初始B500及不足时一次B1000重试；N1的R@100=0.240360，'
        '高于本轮四个N中的最高点。由于解码和计时批次不同，不能混合成本来计算同一前沿。'
        '第二阶段历史9组固定Beam质量被本轮复用，但成本重新测量。[完整历史说明](RESULTS_HISTORY.md)。','',
        '原始全量审计覆盖420项任务证据（225复用、195新增）、1516192条用户—配置记录、5936份候选块清单、'
        '60000条样本原始路径与12288次计时。4组关键配置各256验证用户×3遍完整候选和诊断一致。', '',
        '公开包可在CPU上独立复算16组指标、路径总计、计时、36项Bootstrap与预算，原始ASIN候选、交互、完整逐任务日志、'
        'Checkpoint及集群进程信息不分发。第一阶段只有汇总结果，不能用本包重新生成其全部原始候选。'
        '[验证说明](VERIFICATION.md) · [复现与迁移](REPRODUCIBILITY.md)。', '',
        '所有结论限于当前数据、单训练种子、固定30 epoch和有限搜索预算；测试集是已查看后的扩展分析。'
        '不声称全局最优、SOTA、Scaling Law，未开展Data Scaling、复杂拟合或新增训练种子。','']
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    text='\n'.join(lines)
    for name in ('RESULTS_HISTORY.md','VERIFICATION.md','REPRODUCIBILITY.md'):
        target=os.path.relpath(ROOT/'docs'/name,output.parent).replace('\\','/')
        text=text.replace(']('+name+')',']('+target+')')
    for folder in ('assets','data'):
        target=os.path.relpath(ROOT/folder,output.parent).replace('\\','/')
        text=text.replace('](../'+folder+'/',']('+target+'/')
    output.write_text(text,encoding='utf-8')
    print('Report written:',output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/REPORT.md')
    build(parser.parse_args().output)
