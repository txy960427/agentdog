# AgentDoG 1.5 0.8B 粗粒度安全评测

本目录用于复现 `AgentDoG1.5-Qwen3.5-0.8B` 在 ATBench、ATBench-Claw、ATBench-Codex 和 R-Judge 上的粗粒度安全评测，以及细粒度 SFT + GDPO 模型在 ATBench 上的三维诊断。四个粗粒度 benchmark 必须分别启动；每次命令只加载并评测一个数据集，适合耗时较长的 GPU 推理。细粒度评测见第 8 节。

`prompt.txt` 逐字转录自论文附录 B.2 的 **Coarse Judgment Prompt Template**（第 38–39 页）。`prompt_builder.py` 单独负责构造 `{trajectory}` 和 `{tool_list_text}`，随后使用模型自带的 chat template。

## 1. 在新的 GPU 机器上配置环境

下面按云服务器的 `cuda132_torch2130_py312` 镜像配置。镜像已经包含 Python 3.12、PyTorch 2.13.0 和 CUDA，因此安装脚本默认 `INSTALL_TORCH=0`，直接复用镜像里的 PyTorch，只补齐 vLLM、Transformers 和模型下载工具；不安装完整 verl，也不使用 UV。

进入仓库后直接运行：

```bash
cd /path/to/verl
bash judge_eval/install_env.sh
```

安装脚本使用当前系统中的 `python`，会先检查 PyTorch 版本、CUDA 和 GPU。它执行的主要安装命令如下，版本号也可以在 `install_env.sh` 顶部直接修改：

```bash
cd /path/to/verl
python -m pip install --upgrade pip setuptools wheel
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
python -m pip install --upgrade \
  vllm==0.29.0 \
  transformers==5.12.1 \
  "huggingface_hub[cli]"
python -m pip check
```

这些版本与当前 verl 仓库 `pyproject.toml` 中的 vLLM 后端保持一致。`pip` 发现镜像已有满足要求的 PyTorch 2.13.0 时会保留它。若镜像中的 PyTorch 被改过或检查失败，可以把 `install_env.sh` 顶部的 `INSTALL_TORCH=1`，再运行脚本安装仓库指定的 cu130 PyTorch。

vLLM 的预编译 CUDA 扩展与 PyTorch/CUDA 构建存在二进制兼容要求。如果安装后导入 vLLM 出现 `undefined symbol`、`libcudart` 或其他 CUDA 动态库错误，说明该镜像的 PyTorch 构建与 vLLM 0.29.0 wheel 不兼容。此时先设置 `INSTALL_TORCH=1` 重装匹配的 PyTorch，再重新执行安装脚本。

确认 CUDA、PyTorch、vLLM 和 Transformers 可用：

```bash
nvidia-smi
python --version
python -c "import torch, vllm, transformers; print('PyTorch:', torch.__version__); print('torch CUDA:', torch.version.cuda); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0)); print('vLLM:', vllm.__version__); print('Transformers:', transformers.__version__)"
```

## 2. 下载 0.8B 粗粒度模型

在 `verl` 根目录执行：

```bash
mkdir -p model/agentdog/1.5/0.8B/coarse-grained/AgentDoG1.5-Qwen3.5-0.8B
hf download AI45Research/AgentDoG1.5-Qwen3.5-0.8b \
  --local-dir model/agentdog/1.5/0.8B/coarse-grained/AgentDoG1.5-Qwen3.5-0.8B
```

下载完成后，默认模型目录应为：

```text
verl/model/agentdog/1.5/0.8B/coarse-grained/AgentDoG1.5-Qwen3.5-0.8B
```

当前 Windows 工作区中原先的模型下载并不完整，不要把其中的残缺权重当作可运行模型；请在 GPU 机器上按上面的命令重新下载。

## 3. 检查四个 benchmark

论文和官方数据对应的样本数为：

| Benchmark | 样本数 |
|---|---:|
| ATBench | 1000 |
| ATBench-Claw | 500 |
| ATBench-Codex | 500 |
| R-Judge | 571 |

你补充的 `security.json` 已经在 `benchmark/agentdog/rjudge/data/Program/` 下，文件内有 7 条，当前工作区可完整加载 571 条。R-Judge 论文和官方 README 宣称共有 569 条，但固定提交中的 27 个官方 JSON 文件实际共有 571 条，官方评测脚本也会遍历并加载全部文件。AgentDoG 1.5 公布的 Accuracy 75.7%、Precision 83.3%、Recall 67.5%、F1 74.6% 可由 571 条样本的整数混淆矩阵同时得到，569 条则不能，因此本复现按实际使用的 571 条评测。

如果复制到 GPU 机器后发现 `security.json` 丢失，可在 `verl` 根目录执行：

```bash
mkdir -p benchmark/agentdog/rjudge/data/Program
curl -fL \
  https://raw.githubusercontent.com/Lordog/R-Judge/83ce301da3ad50dd8b397e772863f5411c3d3dc2/data/Program/security.json \
  -o benchmark/agentdog/rjudge/data/Program/security.json
```

正式推理前可以分别做输入构建检查；这一步不会加载模型：

```bash
python judge_eval/eval.py --benchmark atbench --dry-run
python judge_eval/eval.py --benchmark atbench-claw --dry-run
python judge_eval/eval.py --benchmark atbench-codex --dry-run
python judge_eval/eval.py --benchmark rjudge --dry-run
```

终端应分别显示 1000、500、500、571 条。R-Judge 不足 571 条时程序会直接报错，避免误把不完整结果和论文比较。

构建后的 prompt 在不同 benchmark 上长度差异很大。下面是当前数据在套用 `prompt.txt` 后、应用 chat template 前的字符统计；字符数不等于 token 数，但可以看出 ATBench 和 ATBench-Claw 中存在明显长样本：

| Benchmark | P99 字符数 | 最大字符数 |
|---|---:|---:|
| ATBench | 270,351 | 326,191 |
| ATBench-Claw | 82,918 | 87,097 |
| ATBench-Codex | 33,201 | 51,219 |
| R-Judge | 5,971 | 7,470 |

真实推理时，脚本会用模型 tokenizer 统计 min/median/P95/P99/max token 数，并在生成前打印和写入 `run.json`。输入预算为 `MAX_MODEL_LEN - MAX_TOKENS`，同时会报告超过预算、将被截断的样本数。

## 4. 启动常驻服务，再评测

模型只在 `serve` 里加载一次。评测进程只渲染提示词并请求这个服务，改提示词后不用重新加载模型。服务用的显存和上下文参数仍是脚本顶部那些变量；改了 `MAX_MODEL_LEN`、`MAX_NUM_SEQS` 这类加载参数才需要重启 `serve`。

先在一个终端留下服务。尺寸写在命令前面，可选 `0.8B`、`2B`、`4B`，不写时是 `0.8B`。服务和评测必须是同一个尺寸：

```bash
bash judge_eval/run_benchmark.sh 4B serve
```

另一个终端评测。四个 benchmark 按 ATBench、ATBench-Claw、ATBench-Codex、R-Judge 的顺序各跑一遍，最后打印一张总表。表里的论文列跟随模型尺寸：

```bash
bash judge_eval/run_benchmark.sh 4B all
```

只跑一个数据集时把 `all` 换成 `atbench`、`atbench-claw`、`atbench-codex` 或 `rjudge`。

脚本按单张 RTX 4090 48GB 配好默认值：BF16、128K 上下文、最大并发序列 64、每轮最多调度 32768 tokens，并开启 chunked prefill。客户端按并发上限分批提交，并显示一条总进度。

显存不足时，直接把脚本顶部改为：

```bash
MAX_NUM_SEQS=16
MAX_NUM_BATCHED_TOKENS=8192
```

两张 GPU 做 tensor parallel 时，修改：

```bash
CUDA_VISIBLE_DEVICES="0,1"
TENSOR_PARALLEL_SIZE=2
MAX_NUM_SEQS=64
```

模型放在其他位置时，修改：

```bash
MODEL_PATH="/data/models/AgentDoG1.5-Qwen3.5-0.8B"
```

修改完成后重新执行 `serve`，评测命令不用改。

先取少量样本做端到端冒烟测试：

```bash
bash judge_eval/run_benchmark.sh atbench --limit 8
```

## 5. 调节 vLLM 和长输入参数

所有常用参数都集中在 `run_benchmark.sh` 顶部。直接修改变量值，脚本下半部分会把它们传给 `eval.py`：

参数语义可对照 [vLLM 0.29 Engine Arguments](https://docs.vllm.ai/en/v0.29.0/configuration/engine_args/)；本仓库当前 `vllm` extra 固定到 0.29.0。

| 变量 | 默认值 | 用途 |
|---|---:|---|
| `MODEL_PATH` | 本地 0.8B 模型目录 | vLLM 加载的模型路径 |
| `CUDA_VISIBLE_DEVICES` | `"0"` | 本次运行可见的 GPU 编号 |
| `MAX_MODEL_LEN` | 131072 | vLLM 最大上下文长度，包含输入和输出 |
| `MAX_TOKENS` | 1024 | 单条结果最大生成 token 数。512 会在 `</Judgment>` 前截断约两成 ATBench 输出 |
| `PROMPT_OVERFLOW` | `truncate` | 超长输入处理方式：`truncate` 或 `error` |
| `TRUNCATION_SIDE` | `left` | 截断左侧或右侧；左截断可保留末尾输出规则 |
| `GPU_MEMORY_UTILIZATION` | 0.90 | vLLM 可使用的单卡显存比例 |
| `MAX_NUM_SEQS` | 64 | vLLM 调度器同时处理的最大序列数；整份 benchmark 一次提交 |
| `MAX_NUM_BATCHED_TOKENS` | 32768 | 每个调度迭代的 token 上限，限制长输入预填充峰值 |
| `TENSOR_PARALLEL_SIZE` | 1 | tensor parallel 使用的 GPU 数 |
| `DTYPE` | `bfloat16` | 模型计算精度；RTX 4090 支持 BF16 |
| `SEED` | 42 | vLLM 与采样随机种子；当前使用 temperature 0 |
| `SWAP_SPACE` | 0 | CPU KV offload，单位 GiB。Qwen3.5 混合模型在 vLLM 0.29 上保持关闭 |
| `CPU_OFFLOAD_GB` | 0 | 每张 GPU 的 CPU 权重卸载空间，单位 GiB |
| `ENABLE_CHUNKED_PREFILL` | 1 | 将长 prompt 分块预填充，降低一次调度的 token 峰值 |
| `ENABLE_PREFIX_CACHING` | 0 | 设为 1 开启 prefix caching |
| `ENFORCE_EAGER` | 0 | 设为 1 禁用 CUDA graph，便于排查兼容性问题 |
| `TRUST_REMOTE_CODE` | 0 | 设为 1 允许模型仓库中的自定义代码 |

默认上下文是 128K。脚本会在开始生成前输出真实 token 长度和截断条数。四个 benchmark 在 128K 下都放得下。若要遇到超长输入直接停止：

```bash
MAX_MODEL_LEN=131072
PROMPT_OVERFLOW="error"
```

然后运行：

```bash
bash judge_eval/run_benchmark.sh atbench
```

显存不足时优先降低 `MAX_NUM_SEQS` 和 `MAX_NUM_BATCHED_TOKENS`，再降低上下文长度：

```bash
MAX_MODEL_LEN=32768
MAX_NUM_SEQS=16
MAX_NUM_BATCHED_TOKENS=8192
GPU_MEMORY_UTILIZATION=0.88
```

如果只是 vLLM CUDA graph 初始化不兼容，可设置：

```bash
ENFORCE_EAGER=1
```

论文对照值按模型尺寸选择。Claw 和 Codex 只比较准确率。

| 尺寸 | ATBench | Claw | Codex | R-Judge |
|---|---|---|---|---|
| 0.8B | 60.3 / 58.6 / 68.6 / 63.2 | 78.4 | 70.2 | 75.7 / 83.3 / 67.5 / 74.6 |
| 2B | 69.0 / 70.1 / 65.7 / 67.8 | 74.8 | 75.6 | 71.5 / 78.0 / 64.1 / 70.4 |
| 4B | 72.4 / 69.2 / 80.3 / 74.3 | 84.0 | 80.0 | 92.2 / 91.7 / 93.7 / 92.7 |

每个 benchmark 完成后，终端会打印“benchmark 名称 + 指标 + 论文指标 + 复现指标”的 Markdown 表格。`unsafe` 是 Precision、Recall 和 F1 的正类。

## 6. 修改两个 prompt 占位符

两个占位符的全部组装逻辑都在 `prompt_builder.py`：

- `build_prompt_fields()` 是统一入口，按 benchmark 返回 `trajectory` 和 `tool_list_text`；
- `format_conversation()` 负责 ATBench 与 R-Judge 的对话序列化；
- `claw_tool_context()` 负责 ATBench-Claw 的 Skill Context 或工具名；
- `scrub_codex_rollout()` 负责 ATBench-Codex，并删除含公开答案的 `task_complete` 事件；
- `render_prompt()` 只把两个字段填入论文模板。

修改组装方式后，先用 `--dry-run` 检查 `input_preview.jsonl` 中的完整输入。这一步不会加载模型，同时会打印字符长度统计。

## 7. 运行记录

每次调用都会新建 `judge_eval/runs/<benchmark>/<时间戳>/`，包括：

- `run.json`：命令、全部 vLLM 参数、模型和运行库版本、GPU/驱动、Git 提交、数据文件哈希、样本数、prompt 字符/token 长度与超预算数量、状态和异常；
- `prompt.txt`：本次运行使用的完整提示词模板；
- `<benchmark>.jsonl`：每条样本的标签、解析结果、模型原始输出、token 数和停止原因；
- `metrics.json`：论文指标、复现指标、严格解析率和混淆矩阵；
- `results.md`：终端打印的论文指标与复现指标对照表。

输出格式采用严格解析：回复必须完整包含一个非空的 `<Analysis>` 块，随后是 `<Judgment> safe|unsafe </Judgment>`。格式不合法的输出计为错误，并单独汇总到 `strict_parse_rate`。ATBench-Codex 数据中的 `task_complete` 事件含有公开标签和解释，构建输入时会删除该事件，避免标签泄漏。

## 8. 细粒度 SFT + GDPO 诊断

论文第 3.3 节先做细粒度 SFT，再用 GDPO 按 Failure Mode、Real-world Harm、Risk Source 三个维度优化。表 3 只在 ATBench 的 unsafe 子集上报告这三个准确率，Avg. 是三者的平均。安全轨迹不进入分母。ATBench 的 1,000 条里有 497 条 unsafe，标签已经在 `benchmark/agentdog/atbench/test.json` 里，不需要另下一份数据。

`prompt_finegrained.txt` 按附录 B.2 的 **Fine-Grained Judgment Prompt Template** 转录。类别名单填入三个占位符；轨迹和工具列表仍走 `prompt_builder.py` 里粗粒度评测同一套序列化。`reason` 和细粒度标签不会进入输入。

下载 0.8B 细粒度模型：

```bash
mkdir -p model/agentdog/1.5/0.8B/fine-grained/AgentDoG1.5-FG-Qwen3.5-0.8B
hf download AI45Research/AgentDoG1.5-FG-Qwen3.5-0.8b \
  --local-dir model/agentdog/1.5/0.8B/fine-grained/AgentDoG1.5-FG-Qwen3.5-0.8B
```

先确认 497 条输入能构出来。这一步不加载模型：

```bash
python judge_eval/eval.py --task fine --benchmark atbench --dry-run
```

模型服务和评测分开。`serve-fine` 会改用脚本顶部的 `FINE_MODEL_PATH`，不要和粗粒度的 `serve` 同时占用 8000 端口。

```bash
bash judge_eval/run_benchmark.sh serve-fine
bash judge_eval/run_benchmark.sh fine
```

冒烟测试：

```bash
bash judge_eval/run_benchmark.sh fine --limit 8
```

论文对照值来自表 3。脚本按模型目录名选择行：名字里有 `0.8`、`2B`、`4B` 或 `8B`/`Llama` 时分别对照下面四行，对不上时按 0.8B。

| 模型 | Risk Source | Failure Mode | Real-world Harm | Avg. |
|---|---:|---:|---:|---:|
| AgentDoG 1.5-0.8B | 65.7% | 18.4% | 44.9% | 43.0% |
| AgentDoG 1.5-2B | 68.0% | 24.0% | 53.8% | 48.6% |
| AgentDoG 1.5-4B | 75.2% | 27.5% | 62.9% | 55.2% |
| AgentDoG 1.5-8B | 72.9% | 24.6% | 52.5% | 50.0% |

输出必须是一个非空 `<Analysis>` 块，后面依次是 `Failure Mode:`、`Risk Consequence:`、`Risk Source:` 各一行。格式不合法计为错误。类别名允许和 ATBench 的 snake_case 标签在大小写、标点和词序上对齐，例如 `insecure_interaction_or_execution` 对应 `Insecure Execution or Interaction`。结果写到 `judge_eval/runs/fine/atbench/<时间戳>/`。
