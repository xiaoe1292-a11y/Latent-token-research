# Semantic ID、Latent Token 与搜索预算

## 商品表示

商品文本经 Sentence-T5-base 得到 768 维向量，PCA 降至 192 维，再由三层、每层 256 中心的残差量化给出三位 SID，最后修复冲突。本项目 25,612 个商品 SID 均唯一。

PCA 使用预先可用的完整内容目录；量化只拟合 25,527 个训练商品，85 个 heldout-only 商品仅编码。内容目录可用性与交互 last-out 划分是两个不同假设，不能把整个处理流程描述为只访问训练商品内容。

## N 表示种类数

当 latent 种类数为 N 时，本项目 tokenizer 的布局为：

```text
PAD                 0
latent              1 .. N
SID 第1位           N+1   .. N+256
SID 第2位           N+257 .. N+512
SID 第3位           N+513 .. N+768
user token          N+769
EOS                 N+770
vocab_size          N+771
```

一条生成输出始终为 `[latent, SID1, SID2, SID3, EOS]`，长度为 5。N=16 不是依次输出 16 个 latent，也不是把主体网络参数增大 16 倍。第一阶段从 N=1 到 16，参数量只增加 1,920 个，约 0.0421%。

改变 N 会改变后续 token 编号及词表尺寸，所以训练对应模型并重建 tokenizer，不能直接复用不同 N 的词表权重。

## 随机潜在标签与训练恢复

上游训练为每个样本随机采样一个 latent 标签，再接上商品三位 SID 与 EOS。下次遇到同一商品可以使用不同 latent；这些标签不是预先命名的兴趣类别。

恢复需要保存模型、优化器、scheduler、样本排列/位置及 Python、NumPy、Torch CPU/CUDA 随机状态。只读回权重不足以精确继续动态标签与 dropout。原第一阶段包含真实 GPU 保存/读回后同一步损失和参数的一致性预检。

随机标签会引入随 N 变化的熵项，固定训练预算下每条潜在路径获得的监督也会变化。因此不直接用不同 N 的原始训练 loss 判断推荐质量；相同训练种子也不保证不同词表尺寸下逐参数初始化一致。

## Beam 路径如何变成商品

Beam search 输出 `[z, SID1, SID2, SID3, EOS]`。不同 z 可以对应同一 SID，因此 B 条路径通常少于 B 个不同商品。

本实验使用官方 `agg_max`：过滤无效 latent / SID，按分数排序，相同商品保留最高分路径，去重后截断到最多 100 个商品。它不是把潜在路径概率相加；`agg_sum` 未在本项目中消融。

第一阶段保留官方初始 Beam500、不足时一次完整 Beam1000 重试。因此其实际搜索成本依赖候选不足率。后续固定 Beam 实验只执行一次指定 B，并记录所有候选不足用户；它回答与第一阶段不同的解码问题。

## 指标

- 单用户单目标，Recall@K 等于命中率；不因候选不足删掉用户。
- NDCG@10：目标排名 r≤10 时贡献 `1/log2(r+1)`，否则为 0。
- Valid Rate：有效商品路径数 / 全部 Beam 路径数。
- Duplicate Rate：`有效路径数 − 唯一商品数`，分别除以有效路径数和全部路径数。
- Unique Candidates：全 Beam 的不同有效商品数；返回数另按最多 100 个统计。
- 满 100 比例：实际返回 100 个不同商品的用户比例；均值四舍五入为 100.0 不代表人人满 100。

路径计数熵得到的 effective latent count 只描述返回 Beam 中的类型分布，不是完整后验熵或真实表示容量。

## 代码对应

官方固定版本：[tokenizer.py](https://github.com/hyp1231/Latte/blob/05e4e6d983225bcb7172f148a076890e80c524d1/genrec/models/Latte/tokenizer.py)、[model.py](https://github.com/hyp1231/Latte/blob/05e4e6d983225bcb7172f148a076890e80c524d1/genrec/models/Latte/model.py)。

本项目历史实现见 [`reference/`](../reference/README.md)，公开结果复算入口为 [`verify_results.py`](../scripts/verify_results.py)。
