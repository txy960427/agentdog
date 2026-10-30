# AgentDoG 数据构建管线

## PowerShell脚本速查

| 脚本 | 用途 | 当前是否常用 |
|---|---|---|
| `coarse_sft/run_cot.ps1` | 粗粒度CoT统一生成；Target可选baseline/pool/step5 | **当前使用** |
| `coarse_sft/run_cot_check.ps1` | 粗粒度CoT统一审核；Target可选baseline/step5 | Step 5生成后使用 |
| `fine_sft/run_step2.ps1` | 为4,000条细粒度样本生成三标签CoT | 细粒度阶段使用 |
| `coarse_sft/run_step3_company_full.local.ps1` | 从断点重建600条safe twin | Step 3重建时使用 |

每个脚本开头都写明了输入、输出、是否调用API及常用命令。`run_step3_company_full.local.ps1`被Git忽略并保存本机接口配置；其余脚本可提交。

## 细粒度 SFT

当前已实现Step 0–2：从v1.0全部4,000条细粒度原料完成清洗、分组切分、canonical转换和CoT生成/导出。Step 0/1不调用API：

```powershell
python data_pipeline/fine_sft/pipeline.py step0
python data_pipeline/fine_sft/pipeline.py step1
python data_pipeline/fine_sft/pipeline.py step2 --dry-run
python data_pipeline/fine_sft/pipeline.py step2 --limit 6
python data_pipeline/fine_sft/pipeline.py step2
```

也可执行`all`，按顺序重建Step 0和Step 1：

```powershell
python data_pipeline/fine_sft/pipeline.py all
```

Step 0会重建`data/1.5/fine_sft/step0/`，保留4,000条候选索引、4,000条全量清单和审计报告。Step 1按这份清单重建`step1/final/{train,validation,test}.jsonl`，不重新抽样、不生成CoT、不调用API。正式划分为Train 2,701、Validation 678、Test 621；每份均覆盖8个RS、14个FM和10个RWH。

Step 2需要API且必须显式运行，`all`不会调用它。Windows可使用本地包装脚本：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\data_pipeline\fine_sft\run_step2.ps1 -DryRun
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\data_pipeline\fine_sft\run_step2.ps1 -Limit 6
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\data_pipeline\fine_sft\run_step2.ps1
```

`-Limit`只限制本次待处理数量，小批顺序会交错覆盖Train/Validation/Test；再次执行同一命令会从`step2/intermediate/teacher_outputs.jsonl`续跑。Teacher必须按论文附录B.1实际输出`What happened / FM / RWH / RS`四段CoT和`FM → RWH → RS`三个标签；三标签必须合法且与gold一致，脚本不会代替Teacher补写。格式/API错误最多尝试3次；标签冲突时不通过循环调用强行刷成一致。训练专用Student Prompt与benchmark版本只有输出顺序不同，Teacher、Student Prompt和assistant目标统一为`FM → Risk Consequence → RS`。仅4,000条全部通过硬检查时才导出三个Parquet。

Step 2当前仍在断点续跑。`teacher_outputs.jsonl`每条响应立即落盘，是续跑依据；`report.json`只在一次运行正常结束或受控中断后刷新，因此进程运行期间可能落后于checkpoint，不能用旧报告覆盖checkpoint。

---

当前基线实现Step 0至Step 4：统一抽样和冻结split → 转换BinarySafety → 转换带工具unsafe → 生成并审核safe twin → 补充四问CoT并导出SFT数据。Step 5是独立的长度去偏实验，不覆盖基线。

代码按任务放在`data_pipeline/coarse_sft/`，不再嵌套`agentdog/1.5`；输入输出的数据版本在脚本中明确指定，当前分别为`data/1.0/`和`data/1.5/coarse_sft/`。配套训练入口为`train/coarse_sft/`。

## 运行

在仓库根目录执行：

```bash
python3 data_pipeline/coarse_sft/pipeline.py step0
python3 data_pipeline/coarse_sft/pipeline.py step1
python3 data_pipeline/coarse_sft/pipeline.py step2
python3 data_pipeline/coarse_sft/pipeline.py all
```

也可以使用Bash包装：

```bash
bash data_pipeline/coarse_sft/run_pipeline.sh
bash data_pipeline/coarse_sft/run_pipeline.sh step1
bash data_pipeline/coarse_sft/run_pipeline.sh step2
bash data_pipeline/coarse_sft/run_pipeline.sh all
```

`all`只运行不产生API费用的Step 0、Step 1和Step 2；Step 3、Step 4和实验性Step 5需要显式运行。

## Step 5：长度去偏替换实验

GPU复现实验显示gold-safe误报率随事件数从2.1%升至90%以上，因此本阶段先隔离验证长度捷径，不调整safe/unsafe比例。它从未进入基线的Train/BinarySafety/gold-safe中固定选择100条较长轨迹，一对一替换基线Train现有100条Binary safe；Train仍为1,000条和500:500，Validation/Test继续使用Step 4原文件。

```powershell
python data_pipeline/coarse_sft/pipeline.py step5
Get-Content .\data\1.5\coarse_sft\step5\report.json
```

当前选择为5事件45条、6–8事件47条、9–11事件8条；已排除2条Teacher标签冲突样本。输出只保留替换清单和报告：

```text
data/1.5/coarse_sft/step5/
├── final/replacement_manifest.jsonl  # remove 100 + add 100，含长度、hash和CoT状态
├── report.json                       # 选择与长度分布报告
└── cot/                              # 复用通用CoT生成/审核逻辑，数据独立落盘
    ├── intermediate/teacher_outputs.jsonl
    ├── final/candidates.jsonl
    ├── check/{records.jsonl,report.json}
    └── report.json
```

当前100条CoT和独立审核均已全部通过，已导出Step 5训练变体：Train 1,000条，Validation/Test各250条且与Step 4基线逐字节相同。详细契约见[Step 5设计](../docs/agentdog15_reproduction/stages/01_judge_coarse_sft_data/05_length_debias_variant.md)。

继续使用粗粒度统一CoT脚本，但将断点和结果写入Step 5自己的目录：

```powershell
# 预检应显示selected=100，不调用API
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\data_pipeline\coarse_sft\run_cot.ps1 -Target step5 -DryRun
# 正式续跑：只请求当前断点中尚未完成的样本
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\data_pipeline\coarse_sft\run_cot.ps1 -Target step5
# 生成完成后，用同一审核器检查Step 5目录
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\data_pipeline\coarse_sft\run_cot_check.ps1 -Target step5
# 审核全部pass后重新执行Step 5导出（不调用API）
python data_pipeline/coarse_sft/pipeline.py step5
```

## 粗粒度CoT语义审核

统一入口`run_cot_check.ps1`复用同一审核Prompt和代码：`-Target baseline`读取原1,500条最终Parquet，`-Target step5`读取`step5/cot/final/candidates.jsonl`。两者使用各自的`check/`断点，不改CoT、gold或训练文件，也不审核细粒度数据。默认使用与生成模型不同的`gpt-6-astra`，4并发、全局8秒间隔；无需GPU。baseline读取Parquet时需要`pyarrow`。

```powershell
# 预检，不调用API、不写审核文件
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\data_pipeline\coarse_sft\run_cot_check.ps1 -Target baseline -DryRun
# 先审6条，查看审核器的具体意见
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\data_pipeline\coarse_sft\run_cot_check.ps1 -Target baseline -Limit 6
# 审核全部剩余记录；再次运行同一命令即续跑
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\data_pipeline\coarse_sft\run_cot_check.ps1 -Target baseline
```

脚本顶部可改`Model`、`BaseUrl`、`MaxTokens`、`Workers`、`RequestInterval`。审核Prompt单独放在[coarse_cot_checker.txt](../docs/agentdog15_reproduction/stages/01_judge_coarse_sft_data/prompts/coarse_cot_checker.txt)：说明安全裁判背景、safe/unsafe规则，检查四问内部矛盾、证据虚构、授权/执行状态混淆及决定性遗漏。审核模型只看到最终Student输入和Analysis，不看到gold或原Judgment；模型返回轨迹判断、Analysis支持的标签和带原文引用的问题，程序本地比对隐藏标签。

结果只写两个文件：

```text
data/1.5/coarse_sft/step4/check/
├── records.jsonl    # 每次API响应、原文引用、模型来源、用量和断点，逐条刷盘
└── report.json      # 当前pass/review/fail/failed/pending数量及具体疑点
```

`pass`为未发现实质问题；`fail`为CoT明确有缺陷或与原Judgment矛盾；`review`为证据不确定、解释有争议或轨迹判断反对gold。`failed`表示API或审核响应格式/原文引用不合格，不能据此认定训练样本有错。所有已完成意见（包括review/fail）都会保留，不重复抽取到通过，不自动剔除或重新生成。审核器仍然看到了旧CoT，这不是完全独立盲判，疑点留待后续复核。

相同配置重跑只补未完成；换审核模型、接口、Prompt或样本内容会使对应审核key变化。运行时看终端逐条结果；结束或按一次Ctrl+C收取在途结果后更新汇总：

```powershell
Get-Content .\data\1.5\coarse_sft\step4\check\report.json
```

Linux也可以设置`STEP4_API_KEY`后运行`python data_pipeline/coarse_sft/pipeline.py step4 --check --limit 6`，省略`--limit`即全量；`STEP4_CHECK_MODEL`控制审核模型。`--check`不能与`--review`、`--verify`或`--repair-cot`混用。若相同key还在跑其他生成任务，请错开执行或调大间隔；8秒限流只在各自进程内生效。

2026-10-30首审1500条后，已对两条明确的CoT表述缺陷应用定点补丁：删除未展示的Apple来源归属，以及没有证据支持的注入执行归因。只修改Train中这两条assistant文本，输入、gold、split和其他1498条不变；Validation/Test文件未重写。原API响应保留，新记录明确标为`manual_cot_revision_after_check`。只重审修订后的2条，`gpt-6-astra`均判pass；当前模型汇总为1493 pass、7 review。

补丁及9条逐条处理依据见[check_repairs.json](coarse_sft/steps/step4/check_repairs.json)，复现应用命令如下，不调用API且重复执行不追加重复响应：

```powershell
python data_pipeline/coarse_sft/pipeline.py step4 --repair-cot
python data_pipeline/coarse_sft/pipeline.py step4 --verify
```

`step4/check/report_before_cot_repair.json`保留原审核快照，`manual_review.json`记录本助手的逐条复核，不冒充外部模型意见。承包商身份核验案例按项目口径保留safe及原CoT。

随后已按论文§3.1和附录A.2/A.3直接裁决其余6条：均保留unsafe，重写四问，将依据落在已发生的错误执行、错误状态记录或误导性行动建议；不再用未证实的注入执行、外泄或发信来支撑判断。论文的功能/机会风险范围见[证据E13](../docs/agentdog15_reproduction/evidence.md#e13)。累计修改8条CoT（Train 6、Validation 2），其余1492条和所有输入、标签、split保持不变，Test文件未重写；未替换样本。

本轮6条由本助手离线裁决，没有调用API。可只刷新审核汇总：

```powershell
python data_pipeline/coarse_sft/pipeline.py step4 --check --report-only
```

`counts`只显示当前内容对应的外部模型记录：1493 pass、1 review、6 pending；新6条的旧模型意见仍在原始记录中，未冒充新CoT的审核结果。`local_review`显示7条有效本地裁决、综合未决0，且逐条校验输入/回答hash；样本变化会使本地裁决失效。已有checkpoint使用修订前数据，后续重载文件训练才使用新CoT。

## 补充未采样原始数据的CoT

原1500条已经完成，不需要再跑。`cot_pool`为Step 0中未选中的原始轨迹补充**粗粒度四问CoT**，复用Step 1/2转换器、Step 4 Teacher/Student提示词和输入适配，独立保存在`data/1.5/coarse_sft/cot_pool/`。它是可重建且被Git忽略的临时API断点缓存，不改原Step 0–4，也不额外生成safe twin。只用Python标准库，不需要GPU或pyarrow。

当前共5100条：无工具列表safe 850条、无工具列表unsafe 850条、带工具unsafe 3400条。继承原split：Train 3421、Validation 877、Test 802；保留原group/pair和RS/FM/RWH，排除已采样ID、不可用记录和benchmark近重复，不重新均衡抽样。

Windows从项目根目录运行；自动读取本机原company脚本里的key：

```powershell
# 预览5100条，不调用API、不写文件
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\data_pipeline\coarse_sft\run_cot.ps1 -Target pool -DryRun
# 全量生成；再次运行同一命令即续跑
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\data_pipeline\coarse_sft\run_cot.ps1 -Target pool
# 若时间有限，可优先只处理3421条训练侧候选
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\data_pipeline\coarse_sft\run_cot.ps1 -Target pool -Split train
```

顶部可改`Model`、`BaseUrl`、`MaxTokens`、`Workers`、`RequestInterval`，也可加`-Limit 30`只处理30条待完成记录。默认沿用`gpt-6.1-sol`、high、8192 token、4并发和全局8秒启动间隔；5100个首次请求仅按间隔就至少需要约11小时20分钟，重试和响应耗时会延长。新电脑可设置`STEP4_API_KEY`，或在Linux设置同样的`STEP4_*`变量后使用统一入口：

```bash
python data_pipeline/coarse_sft/pipeline.py cot_pool --dry-run
python data_pipeline/coarse_sft/pipeline.py cot_pool
# 也支持 --split train、--limit 30
```

本地运行时只生成四个核心文件：

```text
data/1.5/coarse_sft/cot_pool/
├── intermediate/inputs.jsonl          # 全量canonical输入，保留来源、split、group/pair和三维标签
├── intermediate/teacher_outputs.jsonl # 每次响应立即刷盘，包含原文、用量、状态和模型来源
├── final/candidates.jsonl             # 每次结束或Ctrl+C都更新已通过格式及标签检查的部分
└── report.json                        # 数量、状态、分布、累计用量与文件hash
```

`final/candidates.jsonl`与现有SFT成品使用同样的`id/split/messages/gold/enable_thinking`字段。标签冲突记为`review_required`，失败和冲突不进入final，普通续跑不反复抽取冲突直到与gold一致。混合模型可以续跑：key只依赖当前输入和gold，换模型/接口不重做已完成结果；每条实际来源单独保留。输入或提示词变化时，对应结果失效。`--split`只控制本次请求，不清除其他split已生成的结果。

这是未做独立语义审核和influence-function筛选的候选池，不能直接替换原1000条训练集。后续训练候选只能使用`split=train`，Validation/Test不能混入。断电后重跑同一命令，恢复最后完整记录；按一次Ctrl+C会等待在途结果落盘。查看进度：

```powershell
Get-Content .\data\1.5\coarse_sft\cot_pool\intermediate\teacher_outputs.jsonl -Tail 1 -Wait
Get-Content .\data\1.5\coarse_sft\cot_pool\report.json
```

`report.json`在本次结束或正常中断时更新，运行期间以逐条进度和checkpoint为准。

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
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\data_pipeline\coarse_sft\run_cot.ps1 -Target baseline -DryRun
# 先生成6条，交错覆盖三种数据来源
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\data_pipeline\coarse_sft\run_cot.ps1 -Target baseline -Limit 6
# 全量生成；再次执行同一命令会断点续跑
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\data_pipeline\coarse_sft\run_cot.ps1 -Target baseline
```

脚本顶部直接修改`Model`、`BaseUrl`、`MaxTokens`、`Workers`和`RequestInterval`。默认模型仍为`gpt-6.1-sol`、high推理强度、最大生成8,192 token；现改为4并发、全局请求启动间隔8秒。2026-10-29实测接口能接受8个同时在途的短请求，但返回每分钟最多10次请求的限额；6.5秒间隔的连续调用仍在第11次触发429，因此默认保留8秒并加入统一退避。所有worker及重试共用同一个间隔，不是每个worker分别每8秒发一次请求；正常启动速度上限约7.5次/分钟，遇限流等待会更低。实测结论保留在`step4/api_concurrency_summary.json`，原始短请求流水和一次性探测脚本已清理，不能把测试解释为硬并发上限。本地脚本自动读取已有company脚本中的Key；脚本本身不进Git。

换电脑时也可以直接使用统一入口：

```powershell
$env:STEP4_API_KEY = "你的Key"
$env:STEP4_MODEL = "gpt-6.1-sol"
python data_pipeline/coarse_sft/pipeline.py step4 --dry-run
python data_pipeline/coarse_sft/pipeline.py step4 --limit 6
python data_pipeline/coarse_sft/pipeline.py step4
```

正式生成要求Step 1/2/3均已验收且存在final；Step 3仍有failed/pending时会在API调用之前停止。dry-run可读取Step 3 draft作预览，且不写生成文件。

运行后仅保留这些核心输出：

```text
data/1.5/coarse_sft/step4/
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
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\data_pipeline\coarse_sft\run_cot.ps1 -Target baseline -Review
Get-Content .\data\1.5\coarse_sft\step4\review_report.json
```

`ReviewModels`默认是`gpt-6-sol,gpt-6-astra`，可直接修改脚本顶部变量。每条由两种模型分别读取同一Teacher输入，不发送gold、旧CoT或争议原因，每条正常2次请求；本次累计28次（原8条16次，修正后6条12次）。结果逐次保存到`step4/intermediate/independent_reviews.jsonl`；重跑只续未完成的调用，标签分歧不会自动重复，其他已完成数据不重跑。

两种模型均与gold一致时，选取第一种模型的新CoT写入主检查点并记录复核来源；否则保持原gold和`review_required`。`review_report.json`中的`accepted`表示收录，`models_disagree`表示两模型分歧，`gold_disputed`表示两模型均反对原gold，`api_incomplete`表示调用未完成。模型分歧或共同反对gold需要人工判断，不能循环调用直到碰巧一致。

当主`report.json`的`review_required`清零后，再执行一次不带`-Review`的普通Step 4命令：此时不再调用API，直接导出`final/train.parquet`、`validation.parquet`和`test.parquet`，条数为1,000/250/250，成功状态为`pass`。训练前另做目标模型tokenizer长度检查。

查看进度：

```powershell
Get-Content .\data\1.5\coarse_sft\step4\report.json
Get-Content .\data\1.5\coarse_sft\step4\intermediate\teacher_outputs.jsonl -Tail 1 -Wait
```

仅当当前`report.json`为`pass`且final文件hash匹配时，才是本轮完整导出；旧final文件即使仍存在，也不能代替当前报告。实现见`steps/step4/pipeline.py`，详细Prompt和数据契约见[Step 4设计](../docs/agentdog15_reproduction/stages/01_judge_coarse_sft_data/04_cot_and_sft_export.md)。

全量离线复验（不调用API）：

```powershell
python data_pipeline/coarse_sft/pipeline.py step4 --verify
```

会从本地v1.0逐条重做900条parent的转换，复验600条safe twin血缘、全部1500条CoT与导出内容、文件hash、比例、重复输入及group/pair隔离，结果写入`step4/final_audit.json`。通用拒绝句在不同工具和完整轨迹中复用属于已记录的源数据特征，不作为整条轨迹重复。当前检查不是对1500条重新独立语义标注。`.gitattributes`和写文件代码统一LF，保证Windows/Linux检出后的hash一致。

## 输入

```text
data/1.0/source/AgentDoG1.0-Training-Data/
├── AgentDoG-BinarySafety/train.json
└── AgentDoG-FineGrainedTaxonomy/train.json

benchmark/agentdog/
├── atbench/test.json
└── rjudge/data/
```

ATBench和R-Judge只用于排除近重复。它们不参与抽样目标或模型选择。

## 输出

```text
data/1.5/coarse_sft/step0/
├── intermediate/candidates.jsonl  # 6,000条最小候选索引
├── final/selection.jsonl          # 900条冻结parent
└── report.json                    # 四视角审查和输出hash

data/1.5/coarse_sft/step1/
├── final/
│   ├── train.jsonl                 # 200条
│   ├── validation.jsonl            # 50条
│   └── test.jsonl                  # 50条
└── report.json                     # 清单核验和输出hash

data/1.5/coarse_sft/step2/
├── final/
│   ├── train.jsonl                 # 400条
│   ├── validation.jsonl            # 100条
│   └── test.jsonl                  # 100条
└── report.json                     # 标签、工具和输出审查

data/1.5/coarse_sft/step3/
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
wc -l data/1.5/coarse_sft/step0/intermediate/candidates.jsonl
wc -l data/1.5/coarse_sft/step0/final/selection.jsonl
python3 -m json.tool data/1.5/coarse_sft/step0/report.json
python3 -m json.tool data/1.5/coarse_sft/step1/report.json
python3 -m json.tool data/1.5/coarse_sft/step2/report.json
```

完整阶段契约见[粗粒度SFT数据总设计](../docs/agentdog15_reproduction/stages/01_judge_coarse_sft_data/README.md)，当前抽样规则见[Step 0设计](../docs/agentdog15_reproduction/stages/01_judge_coarse_sft_data/00_global_selection.md)。
