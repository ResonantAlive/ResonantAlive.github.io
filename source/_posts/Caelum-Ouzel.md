---
title: "Caelum Ouzel : 分星、积云、循轨"
date: 2026-10-01 22:30:00
tags: [project开源,模型训练]
---

我们很高兴开源 **Caelum Ouzel（苍穹乌鸫）** 的训练代码。这是一个紧凑型 decoder-only MoE（混合专家）语言模型：总参数约 17 亿，每个 token 只激活约 3.5 亿。

你也可以：总成本 200～300 元，从零训练一个属于自己的家庭小模型。实际花费取决于训练时的 MFU、GPU 型号和租卡单价。

仓库：https://github.com/ResonantAlive/Cealum-Ouzel （Apache-2.0）


## 两三百块，真的够吗？

我们算一笔账。

README 里预训练的默认预算是 **20 亿（2B）token**。在 RTX PRO 6000 上，FP8 路径实测吞吐约 **4.57 万 tok/s**（原生 BF16 约 4.27 万）。2B ÷ 45.7k ≈ 43,800 秒，也就是 **12 小时出头**；再加上编译、验证和存 checkpoint 的开销，乘以你租卡的小时单价，就是总成本的大头。

不想自己算？`train.py` 可以直接按预算反推要训多少 token：

```bash
python train.py \
  --manifest dataset/pretrain_manifest.yaml \
  --out-dir All-checkpoints/caelum-ouzel-pretrain \
  --budget-cny 250 \
  --price-per-hour <租卡单价> \
  --measured-tokens-per-second <实测端到端吞吐> \
  --reserve-cny 20
```

公式很朴素：（预算 − 预留）÷ 单价 × 3600 × 实测吞吐 × 0.95（安全系数）。

有两点需要注意：

- 这三个参数必须一起给。
- 吞吐请填**端到端**实测值（包含验证、存档、编译和数据等待），不要填纯计算的 tok/s。否则算出来的 token 数偏多，钱花完了，训练还没跑完。不放心的话，可以再加一个 `--max-wall-hours` 兜底。

## 模型卡

| 项目 | 数值 |
|---|---|
| 架构 | Decoder-only 稀疏 MoE Transformer |
| 总参数 / 激活参数 | ~1.699B / ~0.350B（每 token） |
| 层数 / 隐藏维度 | 28 / 1024 |
| 注意力 | GQA：8 个 query 头，4 个 KV 头，head_dim 128 |
| 上下文窗口 | 32,768 tokens |
| 分词器 | 32,768 词表的 byte-level BPE |
| MoE | 每层 8 个路由专家（top-1）+ 1 个共享专家 |
| 路由 / 共享专家维度 | 2240 / 416 |
| 位置编码 | RoPE，theta = 1,000,000 |
| 归一化 / 激活 | RMSNorm / SwiGLU |
| 权重绑定 | 输入嵌入与 LM head 绑定 |

完整的架构和训练默认参数以仓库里的 `model hyperparameters.yaml` 为准。

## 开源了什么？

**开源：**

- 预训练源码（含 CPT 持续预训练）
- SFT pipeline
- DPO 源码
- tokenizer 训练源码
- 环境检查、benchmark 与测试
- 数据集来源 / 推荐数据集

**不开源：**

- GRPO 源码、PPO 训练源码
- DPO 数据集、部分 SFT 数据集
- 预训练 code 数据集
- 模型权重与 checkpoint（训练进行中）

按 README 走，大概率可以复现 base 模型。复现不出来先别急着怪代码：训练本身是概率过程，FP8 的精度本来就差，换张卡、换个驱动，曲线都可能有出入。（作者不保证复现结果）

## 设计要点

**文档打包，但不串门。** 预训练把多篇文档打包进同一条序列，减少 padding 浪费；同时用 FlashAttention 2 的 varlen 接口，保证注意力不会跨文档。

**断点续训精确到 token。** checkpoint 同时保存采样器状态和随机数状态。`--resume` 之后从上次的位置接着读，不重不漏。

**不保存完整 logits。** 用 Liger 的分块 linear cross-entropy 计算 32K 词表投影，不需要保存一整块 [tokens, vocab] 的 logits 张量。

**FP8 可选，BF16 兜底。** 默认走 NVIDIA Transformer Engine 的 FP8（DelayedScaling）；遇到不支持的环境，加 `--no-fp8 --no-te` 回退原生 BF16。

**序列长度逐步拉长。** 预训练按 512 → 1024 → 2048 分三段增加序列长度，32K 长上下文交给 CPT 阶段。

**后训练也要省显存。** SFT 只监督 assistant 片段；DPO 只预计算一次冻结参考模型的 log-prob，并按参考模型哈希缓存，不用全程再驻留一份 1.7B 的参考模型。

**花钱之前先体检。** `check_environment.py --strict --full-model` 会在付费训练前检查一遍 CUDA、FlashAttention 和 TE，并按正式模型形状跑数值与反向检查。

**实时面板。** 默认在 http://127.0.0.1:6006 异步绘制 Loss、Val Loss、PPL 与 Val PPL，不占用 CUDA stream。

## 推荐数据集

| 数据集 | 许可证 |
|---|---|
| CCI-3.0-HQ | Apache-2.0 |
| FineWeb-Edu | ODC-By |
| chinese-cosmopedia | Apache-2.0，另需遵守 OpenCSG 社区许可协议 |
| LongData-Corpus | CC-BY-NC-4.0 |

最推荐的组合：**FineWeb-Edu + chinese-cosmopedia + LongData-Corpus**。

两点提醒：

- chinese-cosmopedia 是大模型合成的教科书式数据，用多了，模型容易学到教师模型的口癖。
- LongData-Corpus 是非商用（NC）许可。用它训练出来的模型，商用或公开发布权重前，请自己确认许可证是否兼容。

以上仅为推荐，如有侵权请联系我们或提交 PR。请确认训练数据都有适当授权，且不含私密、机密或非法收集的内容。

## 不建议的做法

**1. 拿中文为主的 tokenizer 去处理英文。** 分词器的压缩率受主训练语言影响很大。按我们的语料训练出来的 tokenizer，中文平均每个 token 能装下 1.5～2.5 个汉字；英文却可能要 3～5 个 token 才能拼出一个单词，模型处理英文的能力会大幅下降。如果你主要处理英文，建议用 `train_tokenizer.py` 重新训练一个适配英文的 tokenizer。

**2. 买付费数据集。** 模型太小了，付费数据集带来的提升很有限。

**3. 直接复用大模型的 tokenizer。** embedding 会吃掉大量参数：以 15 万词表、1024 隐藏维度为例，光 embedding 就约 1.5 亿参数，接近我们每个 token 激活参数（3.5 亿）的一半，而 32K 词表的 embedding 只有约 3,355 万。留给 Transformer block 的参数变少，模型能力下降。同时 logits 也跟着放大近 5 倍，显存吃紧只能把 micro batch 调小，tensor 喂不满，MFU 下降，训练变慢。

**4. 做 logits / 隐状态蒸馏。** 需要和教师模型使用同一个 tokenizer、相同的超参数，门槛较高。

**5. 接入 Agent 工具。** 模型的工作记忆连 system prompt 都不一定能完整存下，极易出错，甚至造成实际损失。而且说实话，你觉得 1.7B 能有多少指令遵循能力？

## 快速上手

需要 Python 3.12，并安装与机器 CUDA runtime 匹配的 PyTorch（我们的实测环境是 PyTorch 2.8.0+cu128）。

```bash
git clone https://github.com/ResonantAlive/Cealum-Ouzel.git
cd Cealum-Ouzel

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 正式 CUDA 训练：装 FlashAttention 2 和 Transformer Engine，然后先体检
python -m pip install -r requirements-server.txt
python check_environment.py --strict --full-model
```

花钱之前，先用一个迷你模型跑通整条链路（smoke run）。`smoke/corpus.jsonl` 自己准备一个小的 JSONL 即可，每行一个带 `text` 字段的对象：

```bash
python train_tokenizer.py --input smoke/corpus.jsonl --output smoke/tokenizer.json \
  --vocab-size 512 --allow-smaller-vocab
python build_pretrain_ids.py --tokenizer smoke/tokenizer.json --input smoke/corpus.jsonl \
  --output smoke/encoded --fresh --workers 2
python train.py --smoke --no-fp8 --no-te
```

正式流程：训练 tokenizer → 逐个编码语料 → 写 manifest → 预训练 → CPT → SFT → DPO。

```bash
# 1. 训练分词器
python train_tokenizer.py --input /path/to/tokenizer-corpus \
  --output tokenizer/tokenizer.json --vocab-size 32768

# 2. 逐个编码语料（corpus.json 里出现 complete: true 才算完成）
python build_pretrain_ids.py --tokenizer tokenizer/tokenizer.json \
  --input /path/to/raw-jsonl --output dataset/encoded/my-corpus \
  --name my-corpus --workers 20

# 3. 从模板创建本地 manifest
cp dataset/pretrain_manifest.example.yaml dataset/pretrain_manifest.yaml

# 4. 预训练（balanced profile）
python train.py --manifest dataset/pretrain_manifest.yaml \
  --out-dir All-checkpoints/caelum-ouzel-pretrain \
  --target-tokens 2000000000 --batch-profile balanced --compile

# 中断了？从最近的 checkpoint 精确续训
python train.py --resume All-checkpoints/caelum-ouzel-pretrain/last.pt \
  --manifest dataset/pretrain_manifest.yaml \
  --out-dir All-checkpoints/caelum-ouzel-pretrain

# 5. 持续预训练（新阶段用 --init，只加载权重）
cp dataset/cpt_manifest.example.yaml dataset/cpt_manifest.yaml
python trainCPT.py --init All-checkpoints/caelum-ouzel-pretrain/last.pt \
  --manifest dataset/cpt_manifest.yaml \
  --out-dir All-checkpoints/caelum-ouzel-cpt --compile

# 6. SFT 与 DPO（做了 CPT 的话，SFT 的 --init 换成 CPT 的 checkpoint）
python trainSFT.py --init All-checkpoints/caelum-ouzel-pretrain/last.pt \
  --data /path/to/sft.jsonl --out-dir All-checkpoints/caelum-ouzel-sft
python trainDPO.py --init All-checkpoints/caelum-ouzel-sft/last.pt \
  --data /path/to/dpo.jsonl --out-dir All-checkpoints/caelum-ouzel-dpo
```

几个注意事项：

- `--init` 和 `--resume` 别混用：新阶段用 `--init`，优化器、学习率进度、随机状态和语料游标全部重新开始；`--resume` 只用于同一阶段的精确续训。
- 默认用 balanced profile。extreme profile（每步 40,960 × 3 tokens）只有在对应机器上完成长时间的 OOM、验证、checkpoint 和续训测试后再用。
- 一个完整 checkpoint（模型 + AdamW 状态）约 19 GiB，请按磁盘空间设置 `--keep-last-checkpoints`。
- 个人路径、原始数据、编码语料、checkpoint 和实验日志不要提交进仓库。

更详细的说明见仓库里的 README。

## 现状与局限

- 目前只在 RTX PRO 6000（sm_120）+ PyTorch 2.8.0+cu128 + Transformer Engine 2.15 上完整验证过。FlashAttention 和 TE 与 ABI、GPU 架构强相关，换环境请先跑 `check_environment.py` 和 `benchmark_pro6000.py`。
- 文中的吞吐与成本都基于我们的机器实测，换卡、换镜像都会变化。
- 小模型能力有限：可能生成错误、有偏见或不当的内容，不适合用于高风险决策。

## 免责声明

本项目源代码以 Apache-2.0 协议按"现状"（as is）提供，不对适用性、稳定性、无缺陷或满足特定用途作任何明示或暗示的保证。

数据合规

训练数据的授权由使用者自行确认。请遵守各数据集的许可证，确保数据不含私密、机密或非法收集的内容。

成本与风险

租用 GPU 会实际产生费用。请先完成 smoke run 和环境检查，再开始付费训练。因配置错误、OOM、中断等造成的费用损失，由使用者自行承担，作者不为开销打包票，不为任何开销负责。

发布模型

如果你要公开发布自己训练的权重，请同时公开数据来源、预期用途、评测结果和已知限制，并确认模型许可证与所有训练数据的许可证兼容。作者不为其负责。

欢迎试用、提 Issue 和 PR。

---

*Caelum Ouzel 训练代码以 Apache-2.0 协议开源。*
