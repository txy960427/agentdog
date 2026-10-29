# 数据目录

仓库本身就是agentdog，不再重复嵌套同名目录。数据保留版本号，用来区分v1.0公开原料和v1.5复现数据；同一份数据由0.8B、2B、4B共享。

```text
data/
├── README.md
├── 1.0/source/AgentDoG1.0-Training-Data/  # 原始公开数据，只读，本地保存
└── 1.5/coarse_sft/                      # 本项目重建，纳入Git
    ├── step0/                           # 候选索引、冻结选择与split
    ├── step1/                           # 无工具列表safe/unsafe
    ├── step2/                           # 带工具unsafe
    ├── step3/                           # 带工具safe twin与审核血缘
    └── step4/                           # CoT、审查记录与最终SFT数据
        └── final/
            ├── dataset_info.json
            ├── train.parquet            # 1,000条
            ├── validation.parquet       # 250条
            └── test.parquet             # 250条
```

Step 0至Step 4均已完成。三个split的safe/unsafe均为1:1，group和pair不跨split。当前是未做影响函数净化的SFT基线。

原始文件和最终Parquet不因目录迁移重新生成。后续任务也按任务命名，例如`fine_sft`、`fine_gdpo`、`main_sft`、`main_rl`，待设计确定后再建目录。

- [数据管线与运行命令](../data_pipeline/README.md)
- [粗粒度SFT训练入口](../train/coarse_sft/README.md)
- [阶段1数据设计](../docs/agentdog15_reproduction/stages/01_judge_coarse_sft_data/README.md)
