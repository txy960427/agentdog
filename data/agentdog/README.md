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

`1.0/source`只保存公开原料。每个1.5训练阶段内部统一使用：`source/`、`intermediate/`、`canonical/`、`metadata/`、`reports/`、`final/`和`manifest.json`。

当前只设计`1.5/coarse-grained/sft`，详细规范见[阶段1数据总设计](../../docs/agentdog15_reproduction/stages/01_judge_coarse_sft_data/README.md)。
