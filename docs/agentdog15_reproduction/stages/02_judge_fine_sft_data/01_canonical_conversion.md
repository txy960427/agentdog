# Step 1：细粒度canonical转换

> 状态：已实现并验收。本Step转换Step 0冻结的全部4,000条记录，不再抽样。

## 1. 目标与输出

Step 1按`source_row_id + raw_hash`回读v1.0原料，解析出Profile、可见工具、轨迹和三个gold标签，保存：

```text
data/1.5/fine_sft/step1/
├── final/train.jsonl          # 2,701
├── final/validation.jsonl     # 678
├── final/test.jsonl           # 621
└── report.json
```

## 2. canonical字段

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
  "tools": [],
  "trajectory": [],
  "target": {
    "risk_source": "Indirect Prompt Injection",
    "failure_mode": "Tool Misuse in Specific Context",
    "risk_consequence": "Functional & Opportunity Harm",
    "cot": null
  }
}
```

不增加粗粒度label、pair_id、confuser、采样特征或审核字段。全部记录天然unsafe，`target`只保留细粒度训练需要的三标签和待生成CoT。

## 3. 字段映射

| canonical字段 | v1.0来源 | 规则 |
|---|---|---|
| `id/group_id/split` | Step 0 selection | 原样继承 |
| `source.row_id/raw_hash` | selection + 原始行 | 回读一致性校验 |
| `profile` | instruction中的Agent Profile | 去首尾空白；默认`You are a helpful assistant.`不重复 |
| `tools` | instruction中的Available tools | 解析JSON，保持源顺序 |
| `trajectory` | BEGIN/END TRAJECTORY中的Conversation | 严格保持事件顺序 |
| `target.risk_source` | output第1维 | 规范成官方类别名 |
| `target.failure_mode` | output第2维 | 规范成官方类别名 |
| `target.risk_consequence` | output第3维 | `Real World Harm`统一命名为`risk_consequence` |
| `target.cot` | 无 | 固定为`null`，由Step 2生成 |

## 4. 工具与轨迹规则

1. 工具只保留当前可见的`name`、`description`和`parameters`；
2. 删除构建元数据`_source`与`_original_description`；工具说明注入样本保留已经注入后的当前`description`；
3. 工具顺序不排序，参数schema不删减；
4. user/environment保存`role + content`，agent保存`role + thought + action`；
5. canonical保留原始thought供审计，但Step 2的Teacher和Student渲染均使用已验证Fine renderer，不把thought写入可见输入；
6. action保持源字符串，渲染时使用ATBench Fine的raw-action规则，不改成粗粒度normalized action；
7. 不删除末尾Complete、error或重复action；
8. environment的HTML可见化和超大字段head-tail压缩只在统一renderer中发生，canonical不覆盖原内容。

## 5. 标签规则

允许的gold固定为：

- 8个Risk Source；
- 14个Failure Mode；
- 10个Risk Consequence。

标签规范化只处理大小写、snake_case与论文展示名的确定性映射，不允许模型重标。标签非法、缺失或出现多个值时直接失败并返回Step 0替换，不在Step 1猜测。

## 6. 验收

- 三个文件精确为2,701/678/621，总计4,000条；
- 每行字段集合固定，ID/group/split与selection一致；
- raw hash全部匹配；
- tools、trajectory和三标签均可解析；
- `_source/_original_description`不进入final；
- `cot`全部为`null`；
- 三维分布、tuple数、输入长度和工具数量与Step 0报告一致；
- 本Step不调用API、不生成prompt、不写Parquet。
