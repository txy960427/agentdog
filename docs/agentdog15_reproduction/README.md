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

粗粒度AgentDoG主要用于轨迹级`safe/unsafe`判断，也是当前公开主执行模型RL代码中安全奖励`S`的来源。细粒度AgentDoG负责RS、FM、RWH诊断；它通过细粒度SFT学习输出格式和诊断知识，再由GDPO直接优化三维标签准确率。论文的主执行模型RL只解析`safe/unsafe`，没有消费RS/FM/RWH，因此细粒度模型是独立的诊断Judge产物，不是主执行模型训练的必要前置。

## 3. 复现阶段

| 阶段 | 训练输入 | 阶段产物 | 当前规划状态 |
|---|---|---|---|
| 1. AgentDoG粗粒度SFT | 粗粒度轨迹、CoT、safe/unsafe | AgentDoG粗粒度模型 | Step 4已通过并导出1,500条Parquet（1,000/250/250）；当前为未做影响函数净化的SFT基线，训练尚未开始 |
| 2. AgentDoG细粒度SFT | 细粒度unsafe轨迹、CoT、RS/FM/RWH | 细粒度SFT初始化模型 | Step 0/1已按全量4,000条通过；Step 2已有6条pilot，其余待续跑 |
| 3. AgentDoG细粒度GDPO | 细粒度query与可验证三维奖励 | AgentDoG细粒度模型 | 待规划 |
| 4. 主执行Agent SFT | AgentDoG筛选后的安全轨迹与良性工具轨迹 | 主执行Agent SFT模型 | 待规划 |
| 5. 主执行Agent RL | 环境任务、工具、rollout与安全/utility奖励 | 主执行Agent RL模型 | 待规划 |

阶段1先完成。其他阶段只保留章节位置，等各自的数据来源、规模和格式确认后再写，不在主文档提前假设。

## 4. 训练数据设计

`data/`按数据版本和任务组织；管线与训练代码按任务组织，分别位于`data_pipeline/coarse_sft/`和`train/coarse_sft/`。仓库已经叫agentdog，不再重复嵌套同名目录，代码也不再增加`1.5`目录。训练数据由0.8B、2B、4B共享。`verl/`只保存训练框架代码：

```text
data/
├── 1.0/
│   └── source/AgentDoG1.0-Training-Data/      # 只读公开原料
└── 1.5/
    ├── coarse_sft/                            # 阶段1，已完成数据构建
    └── fine_sft/                              # 阶段2，Step 0/1已全量完成
```

代码同样隔离为`data_pipeline/coarse_sft/`和`data_pipeline/fine_sft/`。`fine_gdpo`、`main_sft`、`main_rl`仍待设计，不提前创建空目录。

目录约定同时记录在[data/README.md](../../data/README.md)。

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

#### 4.2.1 阶段目标与边界

数据集名称：`AgentDoG1.5-Fine-SFT-Reproduction`。

本阶段与`coarse_sft`完全隔离，使用v1.0 `AgentDoG-FineGrainedTaxonomy`全部4,000条唯一unsafe轨迹重新分组和切分；不读取`coarse_sft/step0/final/selection.jsonl`，也不复用粗粒度Step 4生成的四问CoT。两个阶段可以使用相同公开原料，因为训练的是不同模型，但各自的split、manifest、CoT和最终Parquet互不依赖。

专用细粒度模型只诊断unsafe轨迹。safe样本没有RS/FM/RWH gold，不进入本阶段；safe与三标签联合学习属于Unified模型，不属于本设计。

#### 4.2.2 数据规模

| Split | 条数 | 用途 |
|---|---:|---|
| Train | 2,701 | 细粒度SFT参数训练，后续派生GDPO query |
| Validation | 678 | 选择epoch和checkpoint |
| Internal Test | 621 | 冻结规则后的同分布生成测试 |
| **合计** | **4,000** | 全部为unsafe且具有RS/FM/RWH，并全部补充CoT |

论文只披露影响函数净化后roughly 1k训练样本，没有公开细粒度SFT的精确条数和split。为避免不可复现的抽样选择，本版本明确改为全量4,000条基线；它不是论文影响函数净化后1k数据的严格复现，后续可只从Train内部另建筛选实验。

第一版面向论文Table 3和官方Fine模型卡，固定使用ATBench基础标签空间：8个RS、14个FM、10个RWH。论文1.5整体的15/21/11包含Claw/Codex扩展类别，后续如复现完整ATBench Family再建立扩展数据版本，不能把新增类别静默混入本数据集。

#### 4.2.3 三步数据流水线

1. Step 0重新解析全部4,000条FineGrainedTaxonomy，建立细粒度专用group，按8:2:2确定性冻结为2,701/678/621，不做标签重采样；
2. Step 1按清单物化全部4,000条canonical row，提取Profile、可见工具JSON、原始轨迹和RS/FM/RWH，不再抽样或修改split；
3. Step 2用论文Oracle Fine-Grained CoT思路为全部4,000条生成`<Analysis>`，最终三行固定为`Failure Mode → Risk Consequence → Risk Source`，审核后导出Parquet。

下一级设计文档：

1. [阶段2总设计](stages/02_judge_fine_sft_data/README.md)
2. [Step 0：全量清洗与切分](stages/02_judge_fine_sft_data/00_global_selection.md)
3. [Step 1：细粒度canonical转换](stages/02_judge_fine_sft_data/01_canonical_conversion.md)
4. [Step 2：Fine CoT、审核与SFT导出](stages/02_judge_fine_sft_data/02_cot_and_sft_export.md)
5. [Fine CoT Teacher Prompt](stages/02_judge_fine_sft_data/prompts/fine_cot_teacher.txt)

#### 4.2.4 最终数据格式

```json
{
  "id": "fine_unsafe_000123",
  "split": "train",
  "messages": [
    {"role": "user", "content": "[已验证的Fine-Grained Judgment Prompt + tools + trajectory]"},
    {
      "role": "assistant",
      "content": "<Analysis>\nWhat happened: ...\nFailure Mode: ...\nRisk Consequence: ...\nRisk Source: ...\n</Analysis>\n\nFailure Mode: Tool Misuse in Specific Context\nRisk Consequence: Functional & Opportunity Harm\nRisk Source: Indirect Prompt Injection"
    }
  ],
  "gold": {
    "risk_source": "Indirect Prompt Injection",
    "failure_mode": "Tool Misuse in Specific Context",
    "risk_consequence": "Functional & Opportunity Harm"
  },
  "enable_thinking": false
}
```

Student输入使用训练专用`fine_sft_student.txt`与`build_prompt_fields(..., task="fine")`。该模板与benchmark版本只有最后三行顺序不同，训练统一为`FM → Risk Consequence → RS`。Train、Validation和Internal Test均保留assistant参考答案；测试推理只发送user消息。

### 4.3 AgentDoG细粒度GDPO阶段数据设计

> 状态：待规划。

### 4.4 主执行Agent SFT阶段数据设计

> 状态：待规划。

### 4.5 主执行Agent RL阶段数据设计

> 状态：待规划。

## 5. 训练方案设计

### 5.1 AgentDoG粗粒度SFT训练方案

已提供LLaMA-Factory 0.8B全参SFT基线、完整长度预检与小样本试训入口，见[训练说明](../../train/coarse_sft/README.md)。GPU训练尚未开始。

补充候选入口`data_pipeline/coarse_sft/pipeline.py cot_pool`可为未采样的5100条v1.0轨迹预生成粗粒度CoT，供后续审核及影响函数候选筛选。它继承原split并作为Git忽略的本地缓存独立保存，不改变当前1000/250/250基线；后续训练只能使用其中的Train候选。调用方式见[管线说明](../../data_pipeline/README.md#补充未采样原始数据的cot)。

### 5.2 AgentDoG细粒度SFT训练方案

> 状态：Step 0/1已按全量4,000条完成，Step 2已有6条pilot并支持断点续跑；完成其余CoT和最终Parquet后再补LLaMA-Factory训练参数与命令。论文公开学习率为`1e-5`，细粒度SFT-only结果未披露，因此本项目必须保存自己的SFT中间评测基线。

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

Fine-SFT和Fine-SFT+GDPO都在ATBench的497条unsafe子集上报告RS、FM、RWH及三者平均准确率，并额外报告严格解析率和非平凡Analysis通过率。Table 3的gold标签空间为8/14/10；Internal Test只用于同分布回归，不能替代ATBench。

### 6.3 主执行Agent SFT模型评测

> 状态：待规划。

### 6.4 主执行Agent RL模型评测

> 状态：待规划。

## 7. 论文依据

论文证据、公开数据事实和无法严格复原的内容统一记录在[证据索引](evidence.md)。阶段文档只引用对应证据，不在主文档重复展开。
