# AgentDoG data layout

数据按`版本 → 模型类型 → 训练阶段`组织；同一阶段的数据由0.8B、2B、4B共享，不按模型规模复制。

```text
data/agentdog/
├── 1.0/
│   └── source/AgentDoG1.0-Training-Data/
└── 1.5/
    ├── coarse-grained/
    │   └── sft/
    ├── fine-grained/
    │   ├── sft/
    │   └── gdpo/
    └── main-agent/
        ├── sft/
        └── rl/
```

`1.0/source`只保存公开原料。每个1.5训练阶段使用独立的`stepN/`目录，目录内只保留关键中间结果`intermediate/`、可直接交给下一阶段的`final/`和单一汇总文件`report.json`。

粗粒度SFT先由`step0/`对BinarySafety与FineGrained进行跨来源分组、统一采样和split冻结。`step1/`、`step2/`只按选择清单转换数据，`step3/`和`step4/`继续继承split，不得重新抽样。

当前只展开`1.5/coarse-grained/sft`：Step 0至Step 2已经实现并产出数据；Step 3代码和Prompt已完成但尚未调用API；Step 4只有数据契约与Prompt。详细规范见[阶段1数据总设计](../../docs/agentdog15_reproduction/stages/01_judge_coarse_sft_data/README.md)。

数据构建命令和管线入口见[`data_pipeline/README.md`](../../data_pipeline/README.md)。
