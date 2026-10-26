# Step 0：全局候选池、采样与切分

## 1. 当前阶段的目标、来源和目标数据

### 1.1 当前目标

Step 0是数据的唯一采样和split分配阶段。它先用统一parser检查两个公开来源，再跨来源去重、分组和排除benchmark近重复，最后冻结900条原始parent：

| 数据组件 | Train | Validation | Internal Test | 合计 |
|---|---:|---:|---:|---:|
| BinarySafety | 200 | 50 | 50 | 300 |
| FineGrained unsafe | 400 | 100 | 100 | 600 |
| **全部parent** | **600** | **150** | **150** | **900** |

Step 1和Step 2只能读取本Step的选择清单，不再自行抽样或切分。FineGrained记录同时保留RS、FM、RWH，既可降级成粗粒度`unsafe`，也可供后续细粒度训练使用。Step 3由600条FineGrained parent逐条生成safe twin；Step 4只增加CoT并导出训练格式。

### 1.2 原始数据来源

```text
data/agentdog/1.0/source/AgentDoG1.0-Training-Data/
├── AgentDoG-BinarySafety/train.json
└── AgentDoG-FineGrainedTaxonomy/train.json

benchmark/agentdog/
├── atbench/test.json
└── rjudge/data/
```

BinarySafety有4,000条物理记录，精确去重后为2,000条唯一记录，safe和unsafe各1,000条。FineGrainedTaxonomy有4,000条唯一unsafe记录，覆盖8个Risk Source、14个Failure Mode、10个Real-World Harm和1,044个三维组合。

外部benchmark只参与排重，不参与配额、模型选择或阈值调节。ATBench-Codex和ATBench-Claw属于格式适配测试，不是本阶段主粗粒度测试集。

### 1.3 候选索引示例

`intermediate/candidates.jsonl`保存用于复盘采样的最小索引，不复制完整轨迹：

```json
{
  "stage": "step2",
  "id": "tool_unsafe_4c04a28b1213a699",
  "source_row_id": "0",
  "raw_hash": "a7bef845...",
  "trajectory_hash": "4c04a28b...",
  "group_id": "scenario_516b2f24de488d1b",
  "coarse": "unsafe",
  "risk_source": "Tool Description Injection",
  "failure_mode": "Failure to Validate Tool Outputs",
  "real_world_harm": "Fairness, Equity, and Allocative Harm",
  "features": {
    "length_quartile": 1,
    "event_bin": "09-16",
    "available_tool_count_bin": "4-6",
    "has_environment_error": false,
    "has_schema_warning": false,
    "has_duplicate_tool_name": false
  },
  "eligible": true
}
```

### 1.4 最终选择清单示例

`final/selection.jsonl`是后续阶段的唯一选择依据：

```json
{
  "stage": "step2",
  "id": "tool_unsafe_4c04a28b1213a699",
  "source_row_id": "0",
  "raw_hash": "a7bef845...",
  "group_id": "scenario_516b2f24de488d1b",
  "pair_id": "pair_2d39e21c0d71c50b",
  "split": "train",
  "coarse": "unsafe",
  "risk_source": "Tool Description Injection",
  "failure_mode": "Failure to Validate Tool Outputs",
  "real_world_harm": "Financial & Economic Harm"
}
```

清单不保存训练prompt、CoT或完整trajectory。Step 1和Step 2通过`source_row_id + raw_hash`回读只读原料，并拒绝任何hash不一致的记录。

## 2. 文件结构

```text
data/agentdog/1.5/coarse-grained/sft/
└── step0/
    ├── intermediate/
    │   └── candidates.jsonl       # 两个来源的最小候选索引、特征和全局group
    ├── final/
    │   └── selection.jsonl        # 冻结的900条parent
    └── report.json                # 四视角审查、配额、泄漏与输出hash
```

`selection.jsonl`中的900条记录全部构成正式数据。后续确需替换时，必须返回Step 0重新选择同split、同抽样层记录并重新运行全部验收。

## 3. 处理流程

1. 用共享parser读取两个原始文件，校验字段、标签、轨迹边界、角色顺序和工具调用结构；
2. 计算raw、trajectory、规范化用户任务和工具调用序列hash；
3. 完成来源内部及跨来源的精确去重；
4. 使用规范化用户任务、工具调用序列和调用骨架建立跨来源全局`group_id`；
5. 与ATBench、R-Judge做精确及近重复排除；
6. 为每个全局group确定性分配唯一split，比例为8:2:2；
7. Step 1候选只能从对应split池中按safe/unsafe和结构分布选择；
8. Step 2候选只能从对应split池中按Risk Source配额及RS内部真实分布选择；
9. 生成稳定`pair_id`并冻结900条parent；
10. 完成四视角审查，只有全部通过才写出正式选择清单。

Step 0只做解析、索引、分组、采样和切分，不清洗或改写轨迹内容，不生成safe twin，也不生成CoT。

### 已知源数据偏差

Step 0的硬验收通过表示去重、配额、三标签覆盖、split隔离和benchmark排重正确，不表示公开原料无偏。当前v1.0有两个必须记录的源数据风险：

- BinarySafety的safe只出现`json/text` action，unsafe全部是`mixed` action，格式本身可以区分标签；
- FineGrainedTaxonomy没有1–4事件轨迹，5–8事件只有277条，其余3,723条均为9–16事件。

复现阶段不在Step 0中改写公开轨迹。最终数据依靠Step 2/3的unsafe-safe配对降低格式捷径，但短unsafe覆盖仍是已知限制。

## 4. 全局分组与split规则

1. 两个来源必须在同一候选空间中分组，不能由Step 1和Step 2分别生成互不知情的split；
2. 完整轨迹相同的记录必须属于同一group；
3. 相同或高度相似的用户任务只有在工具调用序列相同时才合并，避免把通用短句误当成同一场景；
4. `group_id`先于抽样确定，同一group只能属于一个split；
5. `pair_id`只能属于一个split；Step 3 child继承parent的group、pair和split；
6. split由`seed + group_id`确定，不能依赖文件遍历顺序；
7. CoT、teacher输出和人工审核结论不得参与grouping或split。

## 5. 抽样规则

### 5.1 Step 1配额

| Split | safe | unsafe | 合计 |
|---|---:|---:|---:|
| Train | 100 | 100 | 200 |
| Validation | 25 | 25 | 50 |
| Internal Test | 25 | 25 | 50 |

每个split在label内部独立匹配长度四分位、事件数、工具调用数、action格式、environment error和末尾Complete。不得使用跨split累计计数器补偿分布。

### 5.2 Step 2配额

Train中8个Risk Source各50条。Validation和Test各100条，每个RS分别为12或13条，两个split的额外配额互补。

每个split独立匹配对应Risk Source候选池中的：

- Failure Mode和Real-World Harm；
- 长度四分位、事件数和可用工具数；
- environment error；
- 参数schema warning；
- 同名工具；
- 上述风险结构的并集。

这些是风险情境和行为信号，不称为“脏数据”。真正无法解析、轨迹损坏或标签非法的记录进入quarantine，不参与抽样。

### 5.3 Validation和Test代表性

100条Step 2 Validation和100条Test不可能精确复刻全部联合分布，但必须同时覆盖8个RS、14个FM、10个RWH，也不能让一个split承担另一个split缺少的异常结构。参考目标如下：

| Split | Environment error | Schema warning | 重名工具 | 任一风险结构 |
|---|---:|---:|---:|---:|
| Train 400 | 18–20 | 36–38 | 10–12 | 60–63 |
| Validation 100 | 3–6 | 7–11 | 1–4 | 12–18 |
| Test 100 | 3–6 | 7–11 | 1–4 | 12–18 |

实际目标应从冻结源文件重新计算并使用确定性最大余数法取整；表中数字是当前源文件的审计参考，不作为脱离源hash的常量。

## 6. 四视角审查与硬验收

### 6.1 模型训练视角

- 最终预期Train为500 safe、500 unsafe；
- 有完整工具schema的数据与Binary数据比例符合设计；
- 8个RS、14个FM、10个RWH和长度区间均有覆盖；
- 稀有风险结构被保留，但没有被人为放大；
- Train、Validation和Test不存在明显任务难度或表面格式偏移。

### 6.2 算法工程视角

- 配额全部满足；每个split使用独立分布目标；
- 同一seed重跑时`selection.jsonl`内容hash完全一致；
- 抽样相对同配额随机基线确实降低FM、RWH和结构边际分布误差；
- 任一group、pair或稳定ID跨split时直接失败。

### 6.3 数据工程视角

- 记录两个输入文件、benchmark文件和parser版本hash；
- quarantine、精确重复、近重复和benchmark匹配均可追溯；
- Step 1/2能够由`selection.jsonl`稳定回读同一原始记录；
- 跨BinarySafety和FineGrained的近重复检查必须进入正式报告。

### 6.4 数据分析视角

- 按split报告标签、RS、FM、RWH、三维组合、长度、事件数、工具数和风险结构；
- 同时报告源分布、目标分布、实际分布和绝对偏差；
- Validation和Test分别验收，不能只检查900条或600条总体；
- Internal Test只用于同分布管线检查；论文核心指标仍使用ATBench和R-Judge。

### 6.5 必须为零或精确满足的条件

- Train与Validation/Test之间的跨阶段group泄漏为0；
- 选中记录与ATBench、R-Judge的精确或近重复为0；
- Step 1为200/50/50，Step 2为400/100/100；
- Step 1标签配额和Step 2 Risk Source配额精确满足；
- 记录ID、pair ID均唯一；
- 输入hash或同seed重跑输出hash发生变化时停止下游阶段。

## 7. 后续阶段契约

- Step 1只物化清单中`stage=step1`的300条记录；
- Step 2只物化清单中`stage=step2`的600条记录；
- Step 3不抽样，逐条继承Step 2 parent的split；
- Step 4不抽样，合并Step 1、Step 2、Step 3后生成CoT和SFT messages；
- 任何记录替换都必须更新Step 0清单版本并重新执行四视角审查。
