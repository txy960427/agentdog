# Step 0：全量清洗与切分

> 状态：已实现并验收。Step 0保留全部4,000条，只负责清洗、分组和分配split。

## 1. 输入与输出

输入：

```text
data/1.0/source/AgentDoG1.0-Training-Data/
└── AgentDoG-FineGrainedTaxonomy/train.json   # 4,000条唯一unsafe

```

外部benchmark不参与本阶段数据构建、split分配或Teacher生成。公开benchmark只在训练完成后用于独立测评。

关键输出：

```text
data/1.5/fine_sft/step0/
├── intermediate/candidates.jsonl
├── final/selection.jsonl
└── report.json
```

`candidates.jsonl`只保存最小索引和结构特征；`selection.jsonl`冻结全部4,000个ID及split；完整轨迹由Step 1按`row_id + raw_hash`从只读原料回读。

## 2. 规模与split

先计算全局`group_id`，再用`seed + group_id`将4,000条候选按8:2:2确定性分入Train/Validation/Test。哈希划分保持group完整，因此实际数量接近而不强制凑成精确比例：

| Split | 正式数量 | 标签要求 |
|---|---:|---|
| Train | 2,701 | 覆盖8 RS、14 FM、10 RWH |
| Validation | 678 | 覆盖8 RS、14 FM、10 RWH |
| Internal Test | 621 | 覆盖8 RS、14 FM、10 RWH |

所有记录均为unsafe，没有Reserve或未采样记录。

## 3. 分组与切分隔离

1. 解析instruction中的Profile、Available Tools和Conversation；
2. 计算原始记录、规范化轨迹、用户任务、工具集合和action序列hash；
3. 完整轨迹相同或用户任务与action骨架高度相似的记录进入同一group；
4. 一个group只能属于一个split；
5. CoT、gold文字顺序、Teacher模型和审核结论不参与grouping；
6. 允许细粒度模型与粗粒度模型使用同一公开原料，但两个阶段的selection和split完全独立。

## 4. 三维分布规则

本版本不重采样、不上采样稀有标签，也不压低高频标签。RS、FM、RWH边际和联合tuple均保持v1.0全部4,000条的真实分布；报告按split记录三维边际、联合tuple与结构特征，用于训练后分层分析。

## 5. 结构分层

三个split分别监控：

- 轨迹事件数和输入字符/token长度；
- 可用工具数、实际调用工具数；
- tool-description injection、环境注入、用户输入和内部失败等风险入口；
- environment error、schema warning、重复工具名；
- Complete是否存在及其位置；
- 三维tuple数量及单例tuple数量。

结构特征只用于保持代表性，不能修改轨迹，也不能把error、Complete或短轨迹自动视为脏数据。

## 6. 选择清单示例

```json
{
  "id": "fine_unsafe_019e5ca802a7d90a",
  "source_row_id": "2922",
  "raw_hash": "...",
  "group_id": "scenario_c5530b5b4a13f79f",
  "split": "train",
  "risk_source": "Indirect Prompt Injection",
  "failure_mode": "Incorrect Tool Parameters",
  "risk_consequence": "Physical & Health Harm"
}
```

## 7. 硬验收

- selection精确为4,000条，当前split为2,701/678/621；
- 每个split覆盖8/14/10，标签必须属于基础ATBench空间；
- Train、Validation、Test之间`group_id`交集为0；
- 选中ID、source row和raw hash全部唯一且可回读；
- 报告三维边际、联合tuple、长度和工具结构；
- 同一seed和输入hash重跑时selection hash完全相同；
- 不生成CoT，不改工具，不改轨迹，不调用模型API。

论文影响函数净化缺少公开`Q_safe`、参考checkpoint和最终样本ID，本版本不伪装已经复现；当前产物是“全量v1.0 Fine-SFT基线”。如以后补做影响函数筛选，只能在Train 2,701条内部生成一个独立训练子集，不能使用Validation/Test或外部benchmark。
