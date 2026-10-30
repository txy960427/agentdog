# Step 2：Fine CoT、审核与SFT导出

> 状态：生成、断点、硬检查与Parquet导出代码已实现，4,000条dry-run已通过；已有6条pilot，其余待断点续跑。

## 1. 两套Prompt的职责

### Teacher Prompt

直接读取[prompts/fine_cot_teacher.txt](prompts/fine_cot_teacher.txt)。它依据论文Appendix B.1 Oracle Fine-Grained CoT设计，可以读取：

- 8/14/10类别；
- 可见工具JSON；
- Student实际可见的trajectory；
- 三个正确标签；
- 三个可混淆标签。

这些oracle信息只用于生成assistant监督答案，不进入Student输入。

### Student Prompt

训练使用[prompts/fine_sft_student.txt](prompts/fine_sft_student.txt)。它完整复用[judge_eval/prompt_finegrained.txt](../../../../judge_eval/prompt_finegrained.txt)的类别、工具、轨迹与三问构建，唯一改动是将最后三行顺序调整为`FM → Risk Consequence → RS`。渲染仍必须调用：

```python
prompt_row = canonical_to_atbench_prompt_row(row)
fields = build_prompt_fields("atbench", prompt_row, task="fine")
student_user = render_prompt(template, fields)
```

`canonical_to_atbench_prompt_row`只做字段适配：`tools → tool_used`、canonical事件列表→`contents`，并复制`profile`；不得读取target或改变事件文本。这样实际格式化仍由已经验证的公共renderer完成，而不是在Fine-SFT管线里复制一份实现。

训练、Validation、Internal Test和ATBench外部评测共享同一模板、类别列表和renderer。

## 2. 占位符构建

| 占位符 | 来源与规则 |
|---|---|
| `{failure_mode_categories}` | `prompt_builder.FAILURE_MODES`的14类编号列表 |
| `{risk_consequence_categories}` | `REAL_WORLD_HARMS`的10类编号列表 |
| `{risk_source_categories}` | 8个RS名称及官方叶定义 |
| `{tools_json}` | canonical tools删除隐藏字段后，以`ensure_ascii=False, indent=2`输出 |
| `{trajectory}` | Fine renderer：Profile + Conversation；不重复工具块，不输出thought，action保留原字符串 |
| 三个gold | canonical target，仅Teacher Internal Guidance可见 |
| 三个confuser | Step 0候选统计确定，仅Teacher可见 |

Teacher与Student必须使用同一份`tools_json`和`trajectory`，避免Teacher依赖Student看不到的thought、原始HTML或被压缩字段。

## 3. Confuser规则

论文使用每个维度一个plausible alternative，但未公开选择算法。第一版使用确定性规则：

1. 固定另外两个gold维度，在Train候选池中选择最常见的非gold替代类别；
2. 若不存在，放宽为固定其中一个维度，选择共现最多的非gold类别；
3. 仍不存在时，使用版本化静态同类映射；
4. Validation/Test的confuser映射只能由Train统计生成，不能读取本split分布调优；
5. confuser只帮助Teacher解释边界，不进入最终SFT row。

## 4. Teacher输出契约

```text
<Analysis>
What happened: [2-3 sentence summary]
Failure Mode: [reasoning]
Risk Consequence: [reasoning]
Risk Source: [reasoning]
</Analysis>

Failure Mode: <gold FM>
Risk Consequence: <gold RWH>
Risk Source: <gold RS>
```

上述Teacher输出结构和标签顺序严格保持论文附录B.1。导出细粒度SFT数据时，Student Prompt和assistant目标也使用`FM → Risk Consequence → RS`；三个标签内容必须来自Teacher实际输出，管线不代写标签。

每问1–3句，必须引用可见轨迹证据。不得提及oracle、gold、hint、confuser或“给定标签”。

## 5. 解析与审核

### 自动硬检查

- `<Analysis>`唯一且闭合，外部无多余文本；
- 分析恰有`What happened / Failure Mode / Risk Consequence / Risk Source`四个非空字段；
- 最终恰有顺序为FM、Risk Consequence、RS的三行；
- 三个类别属于各自标签空间并与canonical gold完全相同；
- `finish_reason`不是length，正文非空；
- 不包含oracle/gold/confuser/hint泄漏词。

Teacher省略任一CoT字段或任一最终标签时均为格式失败；管线不使用gold补写Teacher未输出的标签。

### 语义审核（生成后独立执行）

- FM解释具体行为失败，而不是风险来源或后果；
- RWH解释受影响对象和现实后果，不把动作名称当成危害；
- RS追踪最早引入风险的因素，不把最终失败表现当成来源；
- 只引用Student可见证据，不编造成功执行、授权或工具结果；
- error/Complete不能自动抹掉此前不安全行为；
- 分析与三个gold一致，但不能只是重复标签定义。

格式/API错误可重试；标签冲突进入`reviews.jsonl`，不通过反复调用模型强行刷到gold。当前生成脚本只做格式、标签空间、gold一致性与泄漏词硬检查，不把这些检查冒充成语义审核。证据支持和维度边界必须在生成后另行审核；若人工确认原gold错误，记录裁决并修正或剔除该训练目标，不能用Reserve替换，因为4,000条已全部入选。

## 6. 最终导出

```json
{
  "id": "fine_unsafe_000123",
  "split": "train",
  "messages": [
    {"role": "user", "content": "[student_user]"},
    {
      "role": "assistant",
      "content": "<Analysis>\nWhat happened: ...\nFailure Mode: ...\nRisk Consequence: ...\nRisk Source: ...\n</Analysis>\n\nFailure Mode: ...\nRisk Consequence: ...\nRisk Source: ..."
    }
  ],
  "gold": {
    "risk_source": "...",
    "failure_mode": "...",
    "risk_consequence": "..."
  },
  "enable_thinking": false
}
```

```text
data/1.5/fine_sft/step2/
├── intermediate/teacher_outputs.jsonl   # 断点、原始响应、解析状态
├── intermediate/reviews.jsonl           # 语义审核与人工结论
├── final/train.parquet                   # 2,701
├── final/validation.parquet              # 678
├── final/test.parquet                    # 621
└── report.json
```

Internal Test推理只发送`messages[:1]`，assistant参考答案不进上下文。最终Parquet不写顶层tools，避免LLaMA-Factory再次注入工具。

## 7. 验收与后续GDPO

- 4,000条全部有通过审核的非平凡Analysis；
- 三个Parquet精确为2,701/678/621；
- Student user prompt与ATBench Fine评测renderer逐字同源；
- strict parse、gold一致和标签空间检查全部通过；
- 使用目标Qwen tokenizer报告prompt/response token分布，无静默截断；
- 保存模板hash、renderer Git commit、Teacher模型、API用量和输出hash。

Fine-SFT训练后先在Internal Test与ATBench 497条unsafe上建立中间基线。GDPO数据由本阶段Train派生：删除assistant target，保留同一user prompt，把三个gold放入reward metadata；无需重新采样或重新生成CoT。
