# AgentDoG 数据构建管线

当前实现Step 0至Step 3：Step 0负责统一抽样和冻结split，Step 1转换BinarySafety，Step 2转换带工具的FineGrained unsafe数据，Step 3通过DeepSeek生成并审核safe twin。Step 3代码已就绪但尚未调用API。

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

`all`只运行不产生API费用的Step 0、Step 1和Step 2；Step 3必须显式运行，避免误触发付费请求。普通阶段没有路径、数量或阈值参数。

## Step 3：DeepSeek Safe Twin

不要把API Key写进脚本、README或Git。先在GPU机器设置环境变量：

```bash
export DEEPSEEK_API_KEY="替换为新生成的Key"
```

当前Key已经在聊天中暴露，正式调用前应在DeepSeek控制台撤销并重新生成。

Step 3使用[生成Prompt](../docs/agentdog15_reproduction/stages/01_judge_coarse_sft_data/prompts/safe_twin_generator.txt)和[检查Prompt](../docs/agentdog15_reproduction/stages/01_judge_coarse_sft_data/prompts/safe_twin_checker.txt)，分三步执行：

```bash
# 1. 生成16条pilot；完成后人工编辑pilot/intermediate/reviews.jsonl
MODE=pilot bash data_pipeline/agentdog/1.5/coarse-grained/sft/run_step3.sh

# 2. 16条全部approved后，生成600条正式草稿
MODE=full bash data_pipeline/agentdog/1.5/coarse-grained/sft/run_step3.sh

# 3. 完成人工审核后，只做本地验收和final导出，不调用API
MODE=finalize bash data_pipeline/agentdog/1.5/coarse-grained/sft/run_step3.sh
```

默认配置集中在`run_step3.sh`：`deepseek-v4-pro`、思考模式、JSON Output、生成上限65,536 token、检查上限16,384 token、单并发、每批8条、900秒超时、无HTTP重试、每条只生成一轮。每条完成后立即追加并刷盘；再次执行同一命令只处理尚未成功的记录。生成器一次返回两个候选，本地规则先排序，只检查优先候选；优先候选失败才调用API检查第二候选。失败记录不会在同一轮自动重新生成，必须先审查后再显式运行。可用`BATCH_SIZE=50`调整单批数量。

`full`会直接复用Prompt和模型配置完全一致、且已经审核通过的16条pilot，不会重复生成或重复检查它们。

生成器和检查器当前使用同一个`deepseek-v4-pro` checkpoint，因此600条正式草稿全部需要人工审核。`reviews.jsonl`至少要把`status`改为`approved`并填写`reviewer`与`reviewed_at`；存在`rejected`或未完成审核时，`finalize`会硬失败。

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
├── intermediate/materialized.jsonl # 300条统一格式记录
├── final/
│   ├── train.jsonl                 # 200条
│   ├── validation.jsonl            # 50条
│   └── test.jsonl                  # 50条
└── report.json                     # 清单核验和输出hash

data/agentdog/1.5/coarse-grained/sft/step2/
├── intermediate/materialized.jsonl # 600条带工具unsafe记录
├── final/
│   ├── train.jsonl                 # 400条
│   ├── validation.jsonl            # 100条
│   └── test.jsonl                  # 100条
└── report.json                     # 标签、工具和输出审查

data/agentdog/1.5/coarse-grained/sft/step3/
├── pilot/                           # 16条prompt冻结样本
├── intermediate/
│   ├── candidates.jsonl             # 全部候选、检查结果和调用元数据
│   └── reviews.jsonl                # 人工审核文件
├── draft/                           # 600条待审核safe twin
├── final/                           # finalize后生成400/100/100
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
