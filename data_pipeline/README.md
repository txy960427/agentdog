# AgentDoG 数据构建管线

## 运行

在仓库根目录执行同一个入口，并用阶段名决定运行范围：

```bash
# BinarySafety无工具数据
python3 data_pipeline/agentdog/1.5/coarse-grained/sft/pipeline.py step1

# FineGrained带工具unsafe数据
python3 data_pipeline/agentdog/1.5/coarse-grained/sft/pipeline.py step2

# 按顺序运行当前已实现的Step 1和Step 2
python3 data_pipeline/agentdog/1.5/coarse-grained/sft/pipeline.py all
```

Bash包装命令完全相同：

```bash
bash data_pipeline/agentdog/1.5/coarse-grained/sft/run_pipeline.sh step1
bash data_pipeline/agentdog/1.5/coarse-grained/sft/run_pipeline.sh step2
bash data_pipeline/agentdog/1.5/coarse-grained/sft/run_pipeline.sh all
```

配置固定放在各阶段自己的`config.json`中，命令行不需要传路径、数量或阈值。

## 输入与输出

两个阶段只读以下公开原料和外部benchmark：

```text
data/agentdog/1.0/source/AgentDoG1.0-Training-Data/
├── AgentDoG-BinarySafety/train.json
└── AgentDoG-FineGrainedTaxonomy/train.json

benchmark/agentdog/
├── atbench/test.json
└── rjudge/data/
```

每个阶段只保留一个关键中间文件、三个最终切分和一个汇总报告：

```text
data/agentdog/1.5/coarse-grained/sft/
├── step1/
│   ├── intermediate/cleaned.jsonl  # 去重后的2,000条Binary候选
│   ├── final/
│   │   ├── train.jsonl             # 200条
│   │   ├── validation.jsonl        # 25条
│   │   └── test.jsonl              # 25条
│   └── report.json
└── step2/
    ├── intermediate/cleaned.jsonl  # 清洗后的4,000条带工具unsafe候选
    ├── final/
    │   ├── train.jsonl             # 400条
    │   ├── validation.jsonl        # 50条
    │   └── test.jsonl              # 50条
    └── report.json
```

重跑某个阶段只覆盖该阶段目录。`report.json`集中保存来源hash、去重、类别分布、抽样配额、benchmark排重、泄漏检查和输出文件hash。

## 查看结果

```bash
wc -l data/agentdog/1.5/coarse-grained/sft/step2/final/*.jsonl
python3 -m json.tool data/agentdog/1.5/coarse-grained/sft/step2/report.json
head -n 1 data/agentdog/1.5/coarse-grained/sft/step2/final/train.jsonl | python3 -m json.tool
```

Step 2的最终数据严格保持统一9字段；`_source`和`_original_description`不会进入训练数据。8个Risk Source采用硬均匀配额，RS桶内的FM、RWH、长度、工具数量及异常结构按4,000条原料的真实分布匹配。

官方unsafe轨迹中故意存在环境error、错误参数类型、缺少必填参数和同名工具，这些属于风险行为或场景内容，管线会按原始比例保留，不将其当作脏数据删除。`report.json`的`source_distribution_preservation`记录原始比例、入选比例和差值；四项异常分布任一偏差超过2个百分点时，硬验收失败。
