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
    ├── step4/                           # CoT、审查记录与最终SFT数据
    │   └── final/
    │       ├── dataset_info.json
    │       ├── train.parquet            # 1,000条
    │       ├── validation.parquet       # 250条
    │       └── test.parquet             # 250条
    ├── step5/                           # 长度去偏成品及独立cot/check数据，不覆盖基线
    └── cot_pool/                        # 可重建的临时CoT候选池，Git忽略
```

Step 0至Step 4均已完成。三个split的safe/unsafe均为1:1，group和pair不跨split。当前是未做影响函数净化的SFT基线。Step 5的100进/100出长度去偏变体也已完成CoT、独立审核和Parquet导出。

最终CoT的模型内容审核复用统一入口，结果写入对应Target的`check/`目录且不改变训练数据。生成检查通过与内容审核通过分别记录；命令见[管线说明](../data_pipeline/README.md#粗粒度cot语义审核)。

2026-10-30首审9条疑点均已逐条处置：2条定点修订并经模型重审pass，1条按项目口径保留原safe和CoT；其余6条按论文的可靠性/功能风险范围保留unsafe并重写CoT。累计修改Train 6、Validation 2条assistant内容，输入、标签、划分和其余1492条不变，Test文件未重写。外部模型记录和本地裁决分开保存：1493条模型pass，另7条本地接受，综合未决0；新增6条未追加API复审。原响应、修订清单和审核快照均保留。

`cot_pool`可为剩余5100条原始轨迹预生成粗粒度CoT，保留原split；Train 3421、Validation 877、Test 802。它仅是可重建的API断点缓存，默认不提交Git；Step 5已将选中的100条及其完整血缘独立固化，因此不依赖该目录。

原始文件和最终Parquet不因目录迁移重新生成。`fine_sft/step0`保存全部4,000条候选索引、冻结清单和报告，`fine_sft/step1`保存2,701/678/621条canonical JSONL和转换报告，`fine_sft/step2`为全部4,000条生成CoT并最终导出Parquet；`fine_gdpo`、`main_sft`、`main_rl`待设计确定后再建目录。

- [数据管线与运行命令](../data_pipeline/README.md)
- [粗粒度SFT训练入口](../train/coarse_sft/README.md)
- [阶段1数据设计](../docs/agentdog15_reproduction/stages/01_judge_coarse_sft_data/README.md)
