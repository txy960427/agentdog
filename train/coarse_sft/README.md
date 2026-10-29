# AgentDoG 1.5粗粒度0.8B SFT

当前是**未做influence-function净化的SFT基线**：数据已完成清洗、分层选择、CoT和审核，但没有论文3.2.2节的梯度价值筛选。训练1000条只是规模对齐，不代表样本选择方法对齐。当前数据保留作为baseline；若补净化，应从更大的训练侧候选池筛选，冻结validation/test及其group/pair，另建净化数据版本后再训练。LLaMA-Factory本身不会自动补这一步。

从GPU机器的项目根目录运行。使用已有Python/PyTorch环境，不需要UV。

```bash
python -m pip install -e ./LLaMA-Factory
llamafactory-cli env
```

## 1. 看cutoff_len应该设多少

```bash
bash train/coarse_sft/run.sh check
```

默认底座为`Qwen/Qwen3.5-0.8B`。也可直接运行Python并指定本地底座：

```bash
python train/coarse_sft/run.py check --model /你的模型目录/Qwen3.5-0.8B
```

只加载tokenizer/processor，不加载模型权重，不占用训练显存。读取全部1500条，调用LLaMA-Factory实际ShareGPT转换器和SFT编码器，统计完整prompt+assistant+结束符的token数，校验框架输入前缀与官方`enable_thinking=False`推理前缀一致、prompt被mask且四问Analysis/Judgment完整参与loss。

终端给出各split的min/p50/p95/p99/max和推荐值。推荐值按全量最大长度向上取整到256，不使用test标签或分数选超参。例如最大完整长度为9,100时，推荐9,216；这只是示例，真实值以运行结果为准。

2026-10-29本地用官方0.8B tokenizer和Jinja、对齐后的LF前缀拼接做了CPU对照：train/validation/test最大完整长度分别为10,295/9,257/9,204，推荐10,496；8,192会影响11条，12,288和16,384均无超长。这是轻量tokenizer对照，不是完整框架或GPU训练验证；GPU上的`check`仍会调用实际框架编码器复验。

`run.sh`顶部只改`MODEL`、`GPU`、`CUTOFF_LEN`和`RESUME_FROM`。`CUTOFF_LEN=0`自动采用建议值；手动填写如8192时会统计超长记录，若有截断风险则停止，不删除或截断样本。自动覆盖只是长度覆盖，不代表显存一定够用。

## 2. 小批试训，然后正式训练

```bash
bash train/coarse_sft/run.sh smoke
bash train/coarse_sft/run.sh train
```

两个命令都会重新预检当前模型和数据。`smoke`只做5个优化步骤，梯度累积为1，包含一次验证和保存；采用按长度分组，让长样本尽早暴露显存问题。`train`按`train.yaml`完整训练，1000条train和250条validation，250条test只做编码/长度检查，不参与梯度或checkpoint评分。

模型起点是Qwen3.5底座；语言部分全参SFT，视觉模块冻结。学习率`1e-5`来自项目[论文依据E6](../../docs/agentdog15_reproduction/evidence.md#e6)。3 epoch、batch size 1、梯度累积16、warmup 0.1是当前工程起跑值，不声称为论文完整超参。默认BF16、梯度检查点、SDPA、不开packing；每epoch验证并保存，用验证loss选择checkpoint，benchmark仍通过`judge_eval`评测。

0.8B配置面向先前计划的4090 24GB起跑，但实际显存受最长样本、PyTorch/Transformers和内核影响。若试训OOM，不要缩小cutoff后直接截断数据；先在同一配置下选择更充足的GPU资源，或再验证Liger/优化器卸载方案。

## 数据直接用Parquet

`data/1.5/coarse_sft/step4/final/dataset_info.json`将三个Parquet注册为ShareGPT的`messages/role/content`。不用转换JSON，不改写原有Parquet。顶层`gold/id/split/enable_thinking`是元数据，只有`messages`进入模型；非思考模式由训练配置显式控制。`<Analysis>`是要训练的普通回答正文，不是被移除的原生`<think>`。

官方Qwen Jinja会对user正文做`strip()`，本版本LLaMA-Factory的默认格式器不做；现有user末尾换行会导致一个token的差异。`lf_entry.py`仅在我们自己的训练进程中对齐这一格式化行为，并保留底座官方Jinja用于保存后的推理。预检与训练使用同一适配，逐条检查token一致；框架源码和原始数据保持不变。训练通过官方`run_exp`接口完成，请使用本目录入口，避免裸CLI绕过适配和长度检查。

## 每次运行后看哪里

```text
train/coarse_sft/runs/<时间>_<模式>/
├── report.json       # token统计、推荐长度、输入hash/版本、训练结束状态
├── lengths.jsonl     # 每条id对应的prompt/answer/total长度
├── sample.txt        # 一条实际模型输入与参与loss的回答，便于手工核对
├── train.yaml        # 本次实际配置，已写入选定cutoff和输出路径
├── train.log         # smoke/train才有
└── model/            # smoke/train的模型、checkpoint、trainer_state等
```

正式训练中断后，将`run.sh`里的`RESUME_FROM`设为已有`model/checkpoint-*`目录，再运行`train`。保持底座、数据、batch/累积与训练配置一致；新run继续保存后续模型。运行目录被Git忽略，源码、注册文件和基础配置正常追踪。
