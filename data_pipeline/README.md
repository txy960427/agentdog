# AgentDoG 数据构建管线

当前只实现Step 0。它统一读取BinarySafety和FineGrainedTaxonomy，完成跨来源grouping、benchmark排重、split分配、抽样和四视角审查。Step 1、Step 2代码暂不保留。

## 运行

在仓库根目录执行：

```bash
python3 data_pipeline/agentdog/1.5/coarse-grained/sft/pipeline.py step0
```

也可以使用Bash包装：

```bash
bash data_pipeline/agentdog/1.5/coarse-grained/sft/run_pipeline.sh
```

入口还接受`all`，目前等价于`step0`。脚本没有路径、数量或阈值参数，固定配置写在Step 0代码顶部。

## 输入

```text
data/agentdog/1.0/source/AgentDoG1.0-Training-Data/
├── AgentDoG-BinarySafety/train.json
└── AgentDoG-FineGrainedTaxonomy/train.json

benchmark/agentdog/
├── atbench/test.json
└── rjudge/data/
```

ATBench和R-Judge只用于排除近重复。它们不参与抽样目标或模型选择。

## 输出

```text
data/agentdog/1.5/coarse-grained/sft/step0/
├── intermediate/candidates.jsonl  # 6,000条最小候选索引
├── final/selection.jsonl          # 900条冻结parent
└── report.json                    # 四视角审查和输出hash
```

`selection.jsonl`包含300条Binary parent和600条FineGrained parent。FineGrained记录保留RS、FM、RWH三个标签。后续步骤只能按该清单读取原始记录，不得重新抽样或改变split。

## 查看结果

```bash
wc -l data/agentdog/1.5/coarse-grained/sft/step0/intermediate/candidates.jsonl
wc -l data/agentdog/1.5/coarse-grained/sft/step0/final/selection.jsonl
python3 -m json.tool data/agentdog/1.5/coarse-grained/sft/step0/report.json
```

详细规则见[Step 0设计](../docs/agentdog15_reproduction/stages/01_judge_coarse_sft_data/00_global_selection.md)。
