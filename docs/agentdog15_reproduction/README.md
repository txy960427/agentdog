# AgentDoG 1.5 完整复现设计

## 1. 复现目标

本项目最终复现四个模型产物：两个AgentDoG Judge模型，以及两个主执行Agent模型。

| 目标模型 | 基座与规模 | 训练路线 | 模型功能 |
|---|---|---|---|
| AgentDoG粗粒度模型 | Qwen3.5 0.8B、2B、4B | 粗粒度SFT | 输入Agent执行轨迹，输出四问`Analysis`和`safe/unsafe` |
| AgentDoG细粒度模型 | Qwen3.5 0.8B、2B、4B | 细粒度SFT初始化，再做GDPO | 输入Agent执行轨迹，输出Analysis及RS/FM/RWH三个标签 |
| 主执行Agent SFT模型 | Qwen3.5-4B | 安全轨迹SFT | 学习在工具环境中安全完成任务 |
| 主执行Agent RL模型 | Qwen3.5-4B | 在主执行SFT模型上继续安全RL | 同时优化任务utility与AgentDoG安全奖励 |

细粒度SFT checkpoint是细粒度GDPO的初始化产物，不单独算第五个最终目标模型。

## 2. 模型关系

```text
AgentDoG Judge
├── 粗粒度SFT ─────────────────────────→ AgentDoG粗粒度模型
└── 细粒度SFT → GDPO ──────────────────→ AgentDoG细粒度模型

主执行Agent
└── 安全轨迹SFT ───────────────────────→ 主执行Agent SFT模型
                 └── 在线安全RL ───────→ 主执行Agent RL模型
                         ↑
                 AgentDoG提供安全奖励
```

粗粒度AgentDoG主要用于轨迹级`safe/unsafe`判断。细粒度AgentDoG负责RS、FM、RWH诊断，并在主执行模型的在线RL阶段提供更细的安全信号。

## 3. 复现阶段

| 阶段 | 训练输入 | 阶段产物 | 当前规划状态 |
|---|---|---|---|
| 1. AgentDoG粗粒度SFT | 粗粒度轨迹、CoT、safe/unsafe | AgentDoG粗粒度模型 | Step 0至Step 2已产出；Step 3代码待运行；Step 4待实现 |
| 2. AgentDoG细粒度SFT | 细粒度轨迹、CoT、RS/FM/RWH | 细粒度SFT初始化模型 | 待规划 |
| 3. AgentDoG细粒度GDPO | 细粒度query与可验证三维奖励 | AgentDoG细粒度模型 | 待规划 |
| 4. 主执行Agent SFT | AgentDoG筛选后的安全轨迹与良性工具轨迹 | 主执行Agent SFT模型 | 待规划 |
| 5. 主执行Agent RL | 环境任务、工具、rollout与安全/utility奖励 | 主执行Agent RL模型 | 待规划 |

阶段1先完成。其他阶段只保留章节位置，等各自的数据来源、规模和格式确认后再写，不在主文档提前假设。

## 4. 训练数据设计

根目录下的`data/`按版本、模型类型和训练阶段组织，与`model/agentdog/1.5/<size>/<coarse-grained|fine-grained>`保持同一语义层次。训练数据由不同模型规模共享，所以数据目录不再复制0.8B、2B、4B三级。`verl/`只保存训练框架代码，不再承载本项目的数据、模型、benchmark或外层评测代码：

```text
data/agentdog/
├── 1.0/
│   └── source/AgentDoG1.0-Training-Data/      # 只读公开原料
└── 1.5/
    ├── coarse-grained/
    │   └── sft/                               # 当前阶段
    ├── fine-grained/
    │   ├── sft/                               # 待规划
    │   └── gdpo/                              # 待规划
    └── main-agent/
        ├── sft/                               # 待规划
        └── rl/                                # 待规划
```

目录约定同时记录在[data/agentdog/README.md](../../data/agentdog/README.md)。

### 4.1 AgentDoG粗粒度SFT阶段数据设计

#### 4.1.1 数据集目标

数据集名称：`AgentDoG1.5-Coarse-SFT-Reproduction`。

训练目标是让模型读取Profile、Available Tools和完整执行轨迹，生成四问CoT，并输出一个`safe`或`unsafe`判断。

| Split | 条数 | 比例 | 用途 |
|---|---:|---:|---|
| Train | 1,000 | 66.7% | 参数训练 |
| Validation | 250 | 16.7% | 选择epoch、checkpoint和数据版本 |
| Internal Test | 250 | 16.7% | 所有规则冻结后做一次同分布测试 |
| **合计** | **1,500** | **100%** |  |

论文披露的是roughly 1k训练样本，所以1,000对应Train。Validation和Internal Test各250条，整体按8:2:2构建。正式论文指标另外在R-Judge和ATBench上测试。

#### 4.1.2 数据组成

| 数据组件 | Train | Validation | Internal Test | 来源与构造方式 |
|---|---:|---:|---:|---|
| 无显式工具列表safe | 100 | 25 | 25 | v1.0 BinarySafety去重、清洗和分层抽样 |
| 无显式工具列表unsafe | 100 | 25 | 25 | v1.0 BinarySafety去重、清洗和分层抽样 |
| 带工具unsafe | 400 | 100 | 100 | v1.0 FineGrainedTaxonomy清洗和分层抽样，保留RS/FM/RWH |
| 带工具safe | 400 | 100 | 100 | 对带工具unsafe逐条生成一一对应safe twin |
| **合计** | **1,000** | **250** | **250** |  |

三个split内部都保持safe/unsafe严格平衡。FineGrained的RS、FM、RWH三个标签同时保留，用于本阶段分层、审计和后续阶段的候选种子；这600条不能直接视为尚未规划的细粒度SFT/GDPO完整训练集。

#### 4.1.3 数据构建步骤

1. Step 0统一解析BinarySafety和FineGrainedTaxonomy，跨来源去重和grouping，排除benchmark近重复，冻结900条parent及其split；
2. Step 1按冻结清单转换300条BinarySafety，构造无显式工具列表的safe/unsafe部分，不再抽样；
3. Step 2按冻结清单转换600条FineGrainedTaxonomy，构造带工具unsafe部分并保留三维标签，不再抽样；
4. Step 3对每条带工具unsafe轨迹生成一个同场景、同工具、同split的safe twin；
5. Step 4对Train、Validation和Internal Test统一生成四问CoT，组装相同结构的完整参考答案；测试时只取user消息推理，不把参考assistant消息送给模型。

Step 0是本阶段唯一允许采样和分配split的阶段。它必须从模型训练、算法工程、数据工程和数据分析四个视角完成验收；任何跨阶段近重复落入不同split时都不得进入下游。

下一级设计文档：

1. [阶段1总设计](stages/01_judge_coarse_sft_data/README.md)
2. [全局候选池、采样与切分](stages/01_judge_coarse_sft_data/00_global_selection.md)
3. [按清单转换BinarySafety](stages/01_judge_coarse_sft_data/01_binary_no_tool.md)
4. [按清单转换带工具unsafe](stages/01_judge_coarse_sft_data/02_tool_unsafe.md)
5. [带工具safe twin生成](stages/01_judge_coarse_sft_data/03_tool_safe.md)
6. [CoT生成、占位符、解析与SFT导出](stages/01_judge_coarse_sft_data/04_cot_and_sft_export.md)
7. [最终粗粒度CoT Teacher Prompt](stages/01_judge_coarse_sft_data/prompts/coarse_cot_teacher.txt)

#### 4.1.4 最终数据格式

构建过程保存canonical JSONL；训练时导出`verl`可直接读取的Parquet：

```json
{
  "id": "tool_unsafe_000123",
  "split": "train",
  "messages": [
    {"role": "user", "content": "[粗粒度Judge输入]"},
    {
      "role": "assistant",
      "content": "<Analysis>\n1. ...\n\n2. ...\n\n3. ...\n\n4. ...\n</Analysis>\n\n<Judgment> unsafe </Judgment>"
    }
  ],
  "gold": "unsafe",
  "enable_thinking": false
}
```

Train、Validation和Internal Test都保留assistant参考输出。Train和Validation交给SFT trainer；Internal Test不交给trainer，推理评测时使用`messages[0]`作为输入、`gold`作为标签，`messages[1]`只作为冻结参考答案保留。

#### 4.1.5 阶段测试数据

| 测试类型 | 数据 | 作用 |
|---|---|---|
| Validation | 内部250条Validation | 训练中选择checkpoint |
| Internal Test | 内部250条Test | 冻结后检查同分布泛化 |
| External Test | R-Judge、ATBench | 对齐论文粗粒度核心指标 |
| Adaptation Test | ATBench-Codex、ATBench-Claw | 单独进行格式适配后报告，不与主粗粒度测试混合 |

同一`group_id`和`pair_id`不得跨split。外部benchmark只能用于近重复排除和最终测试，不能用于抽样、CoT生成、checkpoint选择或阈值调节。

### 4.2 AgentDoG细粒度SFT阶段数据设计

> 状态：待规划。

### 4.3 AgentDoG细粒度GDPO阶段数据设计

> 状态：待规划。

### 4.4 主执行Agent SFT阶段数据设计

> 状态：待规划。

### 4.5 主执行Agent RL阶段数据设计

> 状态：待规划。

## 5. 训练方案设计

### 5.1 AgentDoG粗粒度SFT训练方案

> 状态：待数据集设计冻结后规划。

### 5.2 AgentDoG细粒度SFT训练方案

> 状态：待规划。

### 5.3 AgentDoG细粒度GDPO训练方案

> 状态：待规划。

### 5.4 主执行Agent SFT训练方案

> 状态：待规划。

### 5.5 主执行Agent RL训练方案

> 状态：待规划。

## 6. 评测方案设计

### 6.1 AgentDoG粗粒度模型评测

> 状态：已有官方checkpoint复现基线，待训练方案确定后整理成独立阶段文档。

### 6.2 AgentDoG细粒度模型评测

> 状态：待规划。

### 6.3 主执行Agent SFT模型评测

> 状态：待规划。

### 6.4 主执行Agent RL模型评测

> 状态：待规划。

## 7. 论文依据

论文证据、公开数据事实和无法严格复原的内容统一记录在[证据索引](evidence.md)。阶段文档只引用对应证据，不在主文档重复展开。
