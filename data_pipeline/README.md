# AgentDoG 数据构建管线

当前实现Step 0至Step 4：统一抽样和冻结split → 转换BinarySafety → 转换带工具unsafe → 生成并审核safe twin → 补充四问CoT并导出SFT数据。

## 运行

在仓库根目录执行：

```bash
python3 data_pipeline/agentdog/1.5/coarse-grained/sft/pipeline.py step0
python3 data_pipeline/agentdog/1.5/coarse-grained/sft/pipeline.py step1
python3 data_pipeline/agentdog/1.5/coarse-grained/sft/pipeline.py step2
python3 data_pipeline/agentdog/1.5/coarse-grained/sft/pipeline.py all
```

也可以使用Bash包装：

```bash
bash data_pipeline/agentdog/1.5/coarse-grained/sft/run_pipeline.sh
bash data_pipeline/agentdog/1.5/coarse-grained/sft/run_pipeline.sh step1
bash data_pipeline/agentdog/1.5/coarse-grained/sft/run_pipeline.sh step2
bash data_pipeline/agentdog/1.5/coarse-grained/sft/run_pipeline.sh all
```

`all`只运行不产生API费用的Step 0、Step 1和Step 2；Step 3和Step 4需要显式运行。

## Step 3：Safe Twin

Step 3已完成。600条safe twin全部通过人工审核、三模型语义复核和最终验收，并按400/100/100冻结到`final/`。生成期的pilot、draft、API流水和失败队列已清理，只保留训练必需数据与紧凑血缘：

```text
step3/
├── final/{train,validation,test}.jsonl
├── intermediate/
│   ├── lineage.jsonl                    # 600条最终候选的parent、策略、改写点和hash
│   ├── reviews.jsonl                    # 600条人工审核结论
│   ├── model_semantic_audit.jsonl       # 600条三模型聚合结论
│   ├── model_semantic_audit_report.json
│   ├── final_quality_audit.json
│   └── final_disputed_decisions.jsonl
└── report.json
```

如需从头重建，使用`run_step3.sh`依次执行`pilot`和`full`，再运行`steps/step3/audit.py`；审核失败时运行`steps/step3/repair.py`。全部通过后运行`steps/step3/verify.py`，最后以`MODE=finalize`导出。生成、审核和修复均支持断点续跑，但最终验收后不保留体积较大的历史API响应。

## Step 4：CoT生成与SFT导出

不需要GPU。先安装唯一新增依赖：

```powershell
python -m pip install pyarrow
```

本机Windows运行：

```powershell
# 只预览输入和检查格式，不调用API
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\data_pipeline\agentdog\1.5\coarse-grained\sft\run_step4_company.local.ps1 -DryRun
# 先生成6条，交错覆盖三种数据来源
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\data_pipeline\agentdog\1.5\coarse-grained\sft\run_step4_company.local.ps1 -Limit 6
# 全量生成；再次执行同一命令会断点续跑
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\data_pipeline\agentdog\1.5\coarse-grained\sft\run_step4_company.local.ps1
```

脚本顶部直接修改`Model`、`BaseUrl`、`MaxTokens`、`Workers`和`RequestInterval`。默认模型仍为`gpt-6.1-sol`、high推理强度、最大生成8,192 token；现改为4并发、全局请求启动间隔8秒。2026-10-29实测接口能接受8个同时在途的短请求，但返回每分钟最多10次请求的限额；6.5秒间隔的连续调用仍在第11次触发429，因此默认保留8秒并加入统一退避。所有worker及重试共用同一个间隔，不是每个worker分别每8秒发一次请求；正常启动速度上限约7.5次/分钟，遇限流等待会更低。实测结论保留在`step4/api_concurrency_summary.json`，原始短请求流水和一次性探测脚本已清理，不能把测试解释为硬并发上限。本地脚本自动读取已有company脚本中的Key；脚本本身不进Git。

换电脑时也可以直接使用统一入口：

```powershell
$env:STEP4_API_KEY = "你的Key"
$env:STEP4_MODEL = "gpt-6.1-sol"
python data_pipeline/agentdog/1.5/coarse-grained/sft/pipeline.py step4 --dry-run
python data_pipeline/agentdog/1.5/coarse-grained/sft/pipeline.py step4 --limit 6
python data_pipeline/agentdog/1.5/coarse-grained/sft/pipeline.py step4
```

正式生成要求Step 1/2/3均已验收且存在final；Step 3仍有failed/pending时会在API调用之前停止。dry-run可读取Step 3 draft作预览，且不写生成文件。

运行后仅保留这些核心输出：

```text
data/agentdog/1.5/coarse-grained/sft/step4/
├── intermediate/
│   ├── teacher_outputs.jsonl           # 原始响应、修订历史和断点
│   ├── independent_reviews.jsonl       # 双模型复核原文
│   └── unresolved_cases.jsonl          # 当前为空，保留复核入口
├── report.json                        # 批次结束或中断时的状态、数量和输出hash
├── final_audit.json                   # 全量离线来源、划分、CoT和导出复验
├── manual_conflict_review.json        # 35条冲突的修正和裁决血缘
├── review_report.json                 # 最近一轮双模型复核摘要
├── pilot_review_50.json               # 首批人工修订来源
├── api_concurrency_summary.json       # API限流实测结论
└── final/                             # 全部1,500条完成后才导出
    ├── train.parquet                  # 1,000条
    ├── validation.parquet             # 250条
    └── test.parquet                   # 250条
```

单条格式错误或暂时性API错误最多共尝试3次；429会让所有worker统一等待65秒再按全局间隔重试；502/503默认冷却30秒，错误体给出更长`retry_after`时按其要求等待（本次网关曾要求60秒）。401等非暂时性错误、重试耗尽或生成截断会停止派发新请求，已经发出的请求仍收取并保存结果。标签冲突记为`review_required`并保留原文，下次不会自动重试到一致。只有主线程写checkpoint，返回顺序可以不同但每条响应只写一次；按一次Ctrl+C会停止新派发，等待在途结果保存后退出。模型、可见输入、gold或模板变化时，对应检查点自动失效；只改并发数/请求间隔不影响已完成结果。`generated`仅表示四问格式和标签一致性通过，脚本没有新增自动语义过滤。

历史试跑：2026-10-29逐条处理首批50条，其中8条CoT局部人工修订，其余结果复用；原始响应保留在同一checkpoint，新增记录通过`response_source`和`manual_review.source_key`注明来源。检查明细见`step4/pilot_review_50.json`，当前全量状态见下文。

输入适配集中在`steps/step4/inputs.py`：统一历史Complete包装；工具定义缺失时明确标为未知并列出实际观察到的工具名；去除safe twin末尾已确认的返修说明，保留其他thought。两个有问题的safe twin在`input_corrections.json`中按源轨迹hash记录提前停止点和人工改写，Teacher与Student使用相同的适配结果。Step 0的抽样/split和Step 1–3原始final保留不动，benchmark renderer不修改。

2026-10-29已完成全量导出，`step4/report.json`为`pass`：Train 1,000、Validation 250、Test 250，各split的safe/unsafe均为1:1。1500条均含四问Analysis和一致的Judgment，当前没有failed、pending或review_required。训练前仍需使用目标模型tokenizer检查实际训练长度。

全量生成曾出现35条标签冲突：先定点修复27条；剩余8条经过双模型盲判，2条unsafe获支持，另6条safe进一步前移停止点，消除凭据来源和授权/隐私争议后重新盲判，12次判断全部为safe。最终33条轨迹修正记录在`steps/step4/conflict_repairs.json`（5条无工具safe、28条safe twin），另2条unsafe未改输入。没有替换父样本、修改gold或split，也没有补造授权或工具结果。原始final和API响应保留，Teacher与Student共同读取`inputs.py`应用后的修正。离线修正和模型复核的来源分别记录，不把人工修订冒充成模型响应。详见`step4/manual_conflict_review.json`、`review_report.json`和`intermediate/independent_reviews.jsonl`。

当前无须再次生成或复核。需要重导出时运行原Step 4命令，检查点一致则不调用API。争议复核入口仍保留：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\data_pipeline\agentdog\1.5\coarse-grained\sft\run_step4_company.local.ps1 -Review
Get-Content .\data\agentdog\1.5\coarse-grained\sft\step4\review_report.json
```

`ReviewModels`默认是`gpt-6-sol,gpt-6-astra`，可直接修改脚本顶部变量。每条由两种模型分别读取同一Teacher输入，不发送gold、旧CoT或争议原因，每条正常2次请求；本次累计28次（原8条16次，修正后6条12次）。结果逐次保存到`step4/intermediate/independent_reviews.jsonl`；重跑只续未完成的调用，标签分歧不会自动重复，其他已完成数据不重跑。

两种模型均与gold一致时，选取第一种模型的新CoT写入主检查点并记录复核来源；否则保持原gold和`review_required`。`review_report.json`中的`accepted`表示收录，`models_disagree`表示两模型分歧，`gold_disputed`表示两模型均反对原gold，`api_incomplete`表示调用未完成。模型分歧或共同反对gold需要人工判断，不能循环调用直到碰巧一致。

当主`report.json`的`review_required`清零后，再执行一次不带`-Review`的普通Step 4命令：此时不再调用API，直接导出`final/train.parquet`、`validation.parquet`和`test.parquet`，条数为1,000/250/250，成功状态为`pass`。训练前另做目标模型tokenizer长度检查。

查看进度：

```powershell
Get-Content .\data\agentdog\1.5\coarse-grained\sft\step4\report.json
Get-Content .\data\agentdog\1.5\coarse-grained\sft\step4\intermediate\teacher_outputs.jsonl -Tail 1 -Wait
```

仅当当前`report.json`为`pass`且final文件hash匹配时，才是本轮完整导出；旧final文件即使仍存在，也不能代替当前报告。实现见`steps/step4/pipeline.py`，详细Prompt和数据契约见[Step 4设计](../docs/agentdog15_reproduction/stages/01_judge_coarse_sft_data/04_cot_and_sft_export.md)。

全量离线复验（不调用API）：

```powershell
python data_pipeline/agentdog/1.5/coarse-grained/sft/pipeline.py step4 --verify
```

会从本地v1.0逐条重做900条parent的转换，复验600条safe twin血缘、全部1500条CoT与导出内容、文件hash、比例、重复输入及group/pair隔离，结果写入`step4/final_audit.json`。通用拒绝句在不同工具和完整轨迹中复用属于已记录的源数据特征，不作为整条轨迹重复。当前检查不是对1500条重新独立语义标注。`.gitattributes`和写文件代码统一LF，保证Windows/Linux检出后的hash一致。

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

data/agentdog/1.5/coarse-grained/sft/step1/
├── final/
│   ├── train.jsonl                 # 200条
│   ├── validation.jsonl            # 50条
│   └── test.jsonl                  # 50条
└── report.json                     # 清单核验和输出hash

data/agentdog/1.5/coarse-grained/sft/step2/
├── final/
│   ├── train.jsonl                 # 400条
│   ├── validation.jsonl            # 100条
│   └── test.jsonl                  # 100条
└── report.json                     # 标签、工具和输出审查

data/agentdog/1.5/coarse-grained/sft/step3/
├── intermediate/
│   ├── lineage.jsonl                # 最终候选的紧凑血缘
│   ├── reviews.jsonl                # 人工审核结论
│   └── model_semantic_audit.jsonl   # 三模型审核结论
├── final/                           # 400/100/100
└── report.json
```

`selection.jsonl`包含300条Binary parent和600条FineGrained parent。FineGrained记录保留RS、FM、RWH三个标签。后续步骤只能按该清单读取原始记录，不得重新抽样或改变split。

## 查看结果

```bash
wc -l data/agentdog/1.5/coarse-grained/sft/step0/intermediate/candidates.jsonl
wc -l data/agentdog/1.5/coarse-grained/sft/step0/final/selection.jsonl
python3 -m json.tool data/agentdog/1.5/coarse-grained/sft/step0/report.json
python3 -m json.tool data/agentdog/1.5/coarse-grained/sft/step1/report.json
python3 -m json.tool data/agentdog/1.5/coarse-grained/sft/step2/report.json
```

完整阶段契约见[粗粒度SFT数据总设计](../docs/agentdog15_reproduction/stages/01_judge_coarse_sft_data/README.md)，当前抽样规则见[Step 0设计](../docs/agentdog15_reproduction/stages/01_judge_coarse_sft_data/00_global_selection.md)。
