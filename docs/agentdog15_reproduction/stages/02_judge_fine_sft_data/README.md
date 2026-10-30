# 阶段2：AgentDoG细粒度SFT数据总设计

> 当前状态：Step 0/1已按全量4,000条重建并验收；Step 2 dry-run已通过，Teacher API正在按checkpoint断点续跑。只有4,000条全部完成硬检查后才导出最终Parquet。

## 1. 目标与模型关系

本阶段训练专用细粒度Judge的SFT初始化模型。输入为可用工具和完整unsafe轨迹，输出非平凡`<Analysis>`及一个Risk Source、一个Failure Mode和一个Risk Consequence。

```text
Qwen3.5 Base
  → Fine SFT（本阶段）
  → GDPO（三维可验证奖励）
  → AgentDoG Fine最终模型
```

它不接在粗粒度SFT后，也不直接参与公开主执行Agent RL。主执行Agent RL当前只解析粗粒度`safe/unsafe`作为安全奖励`S`；Fine模型的目的，是提升Judge自身的三标签诊断、原因定位和解释能力。

## 2. 来源、规模与标签空间

唯一原始来源：

```text
data/1.0/source/AgentDoG1.0-Training-Data/
└── AgentDoG-FineGrainedTaxonomy/train.json
```

源数据有4,000条唯一unsafe轨迹，覆盖8个RS、14个FM、10个RWH和1,044个三维组合。本版本不再抽样，全部进入Fine-SFT数据：

| Split | 数量 | 用途 |
|---|---:|---|
| Train | 2,701 | SFT训练；后续派生GDPO query |
| Validation | 678 | checkpoint选择 |
| Internal Test | 621 | 冻结后同分布测试 |
| **合计** | **4,000** | 全部补充CoT |

所有正式样本均为unsafe。Table 3和官方Fine模型使用ATBench基础8/14/10类别；v1.5的15/21/11是ATBench Family整体扩展，不进入本版本。

## 3. 三步构建流程

```text
Step 0  4,000条候选统一解析、grouping和8:2:2确定性split冻结
        → selection.jsonl（2,701/678/621，全量）

Step 1  按selection回读原料、清洗工具构建字段、转换canonical schema
        → train/validation/test.jsonl

Step 2  Fine Oracle Teacher生成三问CoT、规则检查、语义审核、组装Student messages
        → train/validation/test.parquet
```

只有Step 0允许分配split；本版本不做标签重采样。Step 1/2不得补抽、重切或因为Teacher输出不好直接丢弃记录。原始标签争议保留在审核清单中解决，不能用未入选后备集替换，因为当前4,000条已全部入选。

## 4. Step 1 canonical格式

```json
{
  "id": "fine_unsafe_000123",
  "group_id": "scenario_abcd1234",
  "split": "train",
  "source": {
    "dataset": "AgentDoG-FineGrainedTaxonomy",
    "row_id": "123",
    "raw_hash": "..."
  },
  "profile": "You are a helpful assistant.",
  "tools": [
    {
      "name": "verify_order_payment",
      "description": "Verify whether an order can be paid.",
      "parameters": {"type": "object", "properties": {}}
    }
  ],
  "trajectory": [
    {"role": "user", "content": "Verify order ORD-789."},
    {"role": "agent", "thought": null, "action": "{\"name\": \"verify_order_payment\", \"arguments\": {\"uid\": \"ORD-789\"}}"},
    {"role": "environment", "content": "{\"status\": \"success\"}"}
  ],
  "target": {
    "risk_source": "Tool Description Injection",
    "failure_mode": "Failure to Validate Tool Outputs",
    "risk_consequence": "Financial & Economic Harm",
    "cot": null
  }
}
```

canonical只保存可复现训练输入所需字段。抽样特征、confuser、Teacher响应、审核结论和API用量放在对应Step的中间文件或报告，不扩张每条正式训练记录。

## 5. 最终SFT数据格式

```json
{
  "id": "fine_unsafe_000123",
  "split": "train",
  "messages": [
    {
      "role": "user",
      "content": "[fine_sft_student.txt经公共Fine renderer渲染后的完整输入]"
    },
    {
      "role": "assistant",
      "content": "<Analysis>\nWhat happened: ...\nFailure Mode: ...\nRisk Consequence: ...\nRisk Source: ...\n</Analysis>\n\nFailure Mode: Failure to Validate Tool Outputs\nRisk Consequence: Financial & Economic Harm\nRisk Source: Tool Description Injection"
    }
  ],
  "gold": {
    "risk_source": "Tool Description Injection",
    "failure_mode": "Failure to Validate Tool Outputs",
    "risk_consequence": "Financial & Economic Harm"
  },
  "enable_thinking": false
}
```

SFT loss只计算assistant消息。Internal Test保留两条message，但生成评测只发送`messages[:1]`。顶层不再重复`tools`，防止chat template二次注入。

## 6. Prompt冻结规则

- Student输入读取[prompts/fine_sft_student.txt](prompts/fine_sft_student.txt)；它与benchmark版本只有最后三行顺序不同；
- 占位符调用`judge_eval.prompt_builder.render_prompt`生成，类别和轨迹调用`build_prompt_fields(..., task="fine")`；
- 工具使用缩进JSON，只保留`name/description/parameters`；
- trajectory不重复工具块，Agent action保留数据集原文，不加入空thought；
- 最终三行顺序固定为`Failure Mode → Risk Consequence → Risk Source`；
- Qwen chat template固定`enable_thinking=false`，显式`<Analysis>`作为普通assistant token训练。

Teacher使用[Fine CoT Teacher Prompt](prompts/fine_cot_teacher.txt)。它可以读取gold和confuser生成监督答案，但gold/confuser不得进入Student user消息。

## 7. 计划目录

```text
data/1.5/fine_sft/
├── step0/
│   ├── intermediate/candidates.jsonl
│   ├── final/selection.jsonl
│   └── report.json
├── step1/
│   ├── final/train.jsonl
│   ├── final/validation.jsonl
│   ├── final/test.jsonl
│   └── report.json
└── step2/
    ├── intermediate/teacher_outputs.jsonl
    ├── intermediate/reviews.jsonl
    ├── final/train.parquet
    ├── final/validation.parquet
    ├── final/test.parquet
    └── report.json

data_pipeline/fine_sft/
├── pipeline.py                       # 唯一命令入口与Step 0
├── step1.py                          # canonical转换
└── step2.py                          # CoT生成、断点与导出
```

第一版只保留selection、canonical split、Teacher检查点、语义审核和最终Parquet这些关键产物。

## 8. 下一级文档

1. [Step 0：全量清洗与切分](00_global_selection.md)
2. [Step 1：细粒度canonical转换](01_canonical_conversion.md)
3. [Step 2：Fine CoT、审核与SFT导出](02_cot_and_sft_export.md)
4. [Fine CoT Teacher Prompt](prompts/fine_cot_teacher.txt)

论文依据见[证据索引](../../evidence.md)。
