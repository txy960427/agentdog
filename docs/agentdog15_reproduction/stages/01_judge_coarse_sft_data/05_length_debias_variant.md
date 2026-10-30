# Step 5：长度去偏Train变体

## 1. 目标

CK180中gold-safe误报率随事件数显著上升：2–4事件为2.1%，5–8事件为46.2%，9–12事件为90.9%，13–20事件为93.9%。因此本实验只改变Train中safe轨迹的长度分布，暂不改变标签先验、总条数、来源配额或外部评测数据。

## 2. 替换方案

- 从Step 0中未进入基线的Train/BinarySafety/gold-safe候选取100条；完整选择已固化在Step 5自身目录，不依赖临时`cot_pool`；
- 事件数至少5，按事件数降序、轨迹字符数降序、ID升序固定排序；
- 要求ID、group、整轨内容唯一，且轨迹以完整`Complete`回答结束；
- 一对一移除基线Train中现有100条Binary safe；
- 保留100条Binary unsafe、400条tool unsafe、400条tool safe；
- Train仍为1,000条和safe:unsafe=500:500；Validation/Test完全冻结。

最终选中5事件45条、6–8事件47条、9–11事件8条，并排除2条Teacher已判为标签冲突的候选。Train全部safe的长度桶从`346/141/13/0`变为`274/206/20/0`（1–4/5–8/9–12/13+）。公开未使用池没有13+事件的safe，因此本实验主要检验5–8事件误报，不能单独解决13+区间。

## 3. 文件与运行

```powershell
python data_pipeline/coarse_sft/pipeline.py step5
```

```text
data/1.5/coarse_sft/step5/
├── final/
│   ├── replacement_manifest.jsonl
│   ├── train.parquet
│   ├── validation.parquet
│   ├── test.parquet
│   └── dataset_info.json
├── report.json
└── cot/
    ├── intermediate/teacher_outputs.jsonl
    ├── final/candidates.jsonl
    ├── check/{records.jsonl,report.json}
    └── report.json
```

清单共200行：100行`remove`和100行`add`。新增行记录排序、事件数、字符数、轨迹hash及CoT状态；报告记录基线与变体分布、所有输入hash和冻结的评测文件。

## 4. 当前状态

100条CoT已全部生成并通过独立审核，`step5/report.json`状态为`pass`。最终Train仍为1,000条和safe:unsafe=500:500；Validation/Test各250条，文件hash与Step 4基线完全相同。只有Train发生100条remove和100条add。

复用与基线完全相同的粗粒度CoT生成和审核脚本，只切换到Step 5数据目录：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\data_pipeline\coarse_sft\run_cot.ps1 -Target step5 -DryRun
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\data_pipeline\coarse_sft\run_cot.ps1 -Target step5
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\data_pipeline\coarse_sft\run_cot_check.ps1 -Target step5
python data_pipeline/coarse_sft/pipeline.py step5
```
