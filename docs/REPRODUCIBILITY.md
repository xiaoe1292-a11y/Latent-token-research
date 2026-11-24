# 复核、复现与运行边界

## 1. CPU 复核：可以直接执行

建议 Python 3.10–3.12。本地验收环境为 Python 3.9.16、NumPy 1.26.0、Matplotlib 3.9.4；GPU 实验使用 Python 3.12、Torch 2.8.0+cu128、Transformers 4.56.1、NumPy 2.2.6。

```bash
python -m pip install -r requirements.txt
python scripts/verify_results.py
python -m unittest discover -s tests -v
python scripts/summarize.py
python scripts/plot_results.py
python scripts/build_report.py
```

后两条命令分别把重绘图片写入 `outputs/figures/`，把重建报告写入 `outputs/REPORT.md`，不改动首页已审阅图和实验 CSV。

核验器从 `paired-records.npz` 重算 Recall@10/50/100、NDCG@10 和候选覆盖；从 `timings.npz` 重算 P50/P95、吞吐和每遍 P50；从原始字节值核对 allocated/reserved MiB；复算路径计数、36 项配对 Bootstrap 和条件预算筛选。最后验证发布数据与历史参考源码的 SHA256。

所有 NPZ 使用 `allow_pickle=False`。排名按同一测试用户顺序对齐，未命中记为 101；文件只保留排名与返回数量，不包含用户标识或历史。计时每组 768 条，依次为相同 256 名验证用户的三遍推理。吞吐为请求数/计时总和，不包含计时范围之外的 GPU 占用检查。

NDCG 折扣显式采用 float64，避免 int16 数组的 log2 自动选择 float32 带来的核验误差。

## 2. GPU 实验：历史实现与环境迁移

`reference/` 是历史训练/推理实现，供审查实际工作与调用路径；它保留原运行目录和哈希依赖，不能直接当成当前仓库的 GPU 快速入口。CPU 依赖中也没有安装完整训练环境。

要在新服务器重做训练/推理，需要：

1. 获取官方 [Latte](https://github.com/hyp1231/Latte)，固定 commit `05e4e6d983225bcb7172f148a076890e80c524d1`，安装其训练依赖。
2. 按 `configs/common.json` 取得 Amazon Reviews 2023 / Video_Games 官方 5-core、全评分、逐用户 last-out 数据。验证用户顺序、序列、商品映射和内容目录是否一致。
3. 准备同版本 Semantic ID、模型配置及四份冻结 Checkpoint。它们未随包发布；重新生成 SID 或从头训练通常不能保证逐项等于本包结果。
4. 迁移 `reference/` 中的路径、运行时、资源准入与缓存配置，并重新审计 Checkpoint 的数据/SID/词表/训练预算。Checkpoint SHA256 保存在 `configs/joint16.json`；配置种类改变会改变 token 编号，不能混用不同 N 的权重。
5. 先执行真实 GPU 预检，确认固定实际 Beam、只生成一次、官方 `agg_max` 和候选一致性，再评估全体测试用户。质量 batch2，FP32；不自动补 Beam、不热门填充。
6. 性能测量保持相同硬件、输入、batch1、精度、预热与同步口径；如要验证严格隔离成本，需要独占资源重新测量全部可比点。

本包没有把“运行分析脚本”描述为“重新训练了官方模型”。训练、完整候选生成及严格隔离基准需要上述资产和环境，不能由已发布排名数组替代。

## 3. 数据与协议文件

- `configs/common.json`：训练和数据口径、SID、模型与上游 commit。
- `configs/phase1.json`：第一阶段五组模型及自动重试协议。
- `configs/phase2.json`：历史九组固定 Beam 结果的口径。
- `configs/joint16.json`：最终 16 组矩阵、权重哈希、冻结统计设置和计时顺序。
- `provenance/inputs.json`：原输入文件哈希及公开导出说明。联合 CSV 仅去掉物理 GPU UUID、背景 PID 两列，其余已发布字段逐项保持原值。
- `provenance/data-sha256.json`：公开数据与配置的精确字节哈希。
- `provenance/reference-source-sha256.json`：历史参考源码字节哈希。

`.gitattributes` 保留 `data/`、`configs/`、`reference/` 的精确字节，避免 Git 自动换行转换使历史哈希失效。修改数据后核验失败是预期行为；新实验应使用新的结果目录与协议，不覆盖这些文件。

## 4. 公开范围

本仓库分发数值结果、匿名排名、原始计时、路径汇总、统计比较、历史实现与审计摘要。原始用户交互、商品元数据、完整 ASIN 候选块、Checkpoint、集群连接脚本、进程 PID 和认证信息不包含在分发清单中。

原研究的完整逐任务证据与结果包另行本地封存；公开包不能凭摘要独立重做全部原始候选合法性审计。`provenance/original-audit-summary.json` 记录原审计范围，公开包当前核验范围见 `reports/public-verification.json`。
