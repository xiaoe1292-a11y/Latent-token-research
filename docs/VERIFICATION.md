# 验证证据与边界

## 公开包的本地验收

```bash
python scripts/verify_results.py
python -m unittest discover -s tests -v
python scripts/plot_results.py
python scripts/build_report.py
```

- 16 组全部通过 CPU 数值复核：1,516,192 条用户—配置排名、12,288 次原始计时、36 项用户配对 Bootstrap。
- 与历史九组复用点的四项质量指标一致；第一阶段五组的 batch1/batch4 行数及重复质量字段一致。
- 7 项测试包含 4 项统计边界测试，以及在真实公开数据副本上的 3 项错误注入：错误 Recall、把字节误写成 MiB、把共享测量改写成独占测量。三种错误均必须被拒绝。
- 重绘使用已核验 CSV，召回/候选曲线数组与源数据核对；四张图的可读性另做人工视觉检查。
- 已发布数据、配置与历史参考源码的精确 SHA256 在 `provenance/` 中保留。

当前本地复核记录见 [`public-verification.json`](../reports/public-verification.json)，命令输出见 [`checks.log`](../reports/checks.log)。GitHub Actions 是准备好的后续自动核验配置，实际远端状态以你上传后的运行记录为准。

## 原始实验中的审计

原始完整批次审计覆盖：420 项任务证据（225 复用、195 新增）、384 个质量分片、5,936 份候选块清单、60,000 条样本原始生成路径和 12,288 次计时。

所有 16 组原始 ASIN 候选与真实目标独立重算、检查合法性与去重、排名、计数守恒。N2/B500、N4/B500、N8/B1000、N16/B2000 各 256 验证用户连续三遍完整候选及诊断相同。这是同一权重的推理重放，不是多训练种子稳定性实验。

原审计器曾在本地 NDCG 重算时发现 int16 的 `log2` 选择 float32，产生约 3.6×10⁻⁹ 的数值差异。显式转 float64 后按原 1×10⁻¹² 容差通过，没有放宽阈值或改实验指标。公开核验器直接采用该 float64 口径。

原始审计范围与三项错误注入摘要见 [`original-audit-summary.json`](../provenance/original-audit-summary.json)和 [`original-negative-controls.json`](../provenance/original-negative-controls.json)。这些摘要说明原批次做了什么，不替代对未发布原始候选的独立验证。

## 未覆盖的事项

1. 本轮 16 组成本全部来自共享 GPU，未验证严格隔离成本、锁频性能或线上 SLA；计时三遍稳定也不能排除跨配置背景负载差异。
2. 无多训练种子验证、数据规模扩展、复杂拟合或 Scaling Law 验证。Bootstrap 只描述冻结模型下的用户抽样不确定性。
3. 原始交互、完整 ASIN 候选、权重、集群运行日志及进程信息未公开；CPU 核验不能重建全部 GPU 生成过程，也不能对第一阶段重新生成原始候选。
4. 测试集曾用于前期分析，当前为冻结权重的扩展探索。固定 30 epoch 不保证收敛，不能把点估计最高称为全局最优或论文最终性能复现。
