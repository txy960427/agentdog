# AgentDoG 1.5 粗粒度安全评测数据

对应论文：[AgentDoG 1.5](https://arxiv.org/pdf/2605.29801)，第 2.3、3.4.1 节、表 2。

| Benchmark | 本地入口 | 样本数 | safe / unsafe | 状态 |
| --- | --- | ---: | --- | --- |
| ATBench | `atbench/test.json` | 1,000 | 503 / 497 | 完整，官方哈希校验通过 |
| ATBench-Claw | `atbench-claw/test.json` | 500 | 204 / 296 | 完整，官方哈希校验通过 |
| ATBench-Codex | `atbench-codex/test.json` | 500 | 250 / 250 | 完整，官方哈希校验通过 |
| R-Judge | `rjudge/data/**/*.json` | 571（27 个文件） | 270 / 301 | 完整；见下方数量说明 |

## 版本与来源

- ATBench：<https://huggingface.co/datasets/AI45Research/ATBench>，选择 `ATBench` config 的 `ATBench/test.json`，本地统一为 `atbench/test.json`。此处只使用 1.5 对应的 1,000 条版本。
- ATBench-Claw：<https://huggingface.co/datasets/AI45Research/ATBench-Claw>，`test.json`。
- ATBench-Codex：<https://huggingface.co/datasets/AI45Research/ATBench-Codex>，`test.json`。
- R-Judge：<https://github.com/Lordog/R-Judge>。R-Judge 原论文和 README 宣称共有 569 条，但固定提交 `83ce301` 的 27 个官方 JSON 文件实际共有 571 条，官方评测脚本会遍历并加载全部文件。AgentDoG 1.5 的已发表指标也与 571 条的整数混淆矩阵一致，因此本地保留并评测全部 571 条。

固定提交、文件大小、SHA-256、校验结果和数量说明见 `download-manifest.json`。各目录保留官方 README、来源信息和文件树。Hugging Face 文件通过 hf-mirror.com 下载，并对照固定提交的官方哈希验证。

## 粗粒度评测口径

按照 1.5 论文，输入完整 agent trajectory，输出 safe/unsafe；R-Judge 和 ATBench 报告 Accuracy、Precision、Recall、F1，Claw/Codex 用于跨执行环境泛化评测。论文图 1 将 ATBench 标成 ATBench-Pro；正文第 2.3 节明确为 1,000 条 ATBench。

标签字段不同，加载时须分别处理：

- ATBench：`label`，0 = safe，1 = unsafe。
- ATBench-Claw：`labels.is_safe`，true = safe，false = unsafe。
- ATBench-Codex：`is_safe`，true = safe，false = unsafe。
- R-Judge：`label`，0 = safe，1 = unsafe。

计算二分类 Precision/Recall/F1 时将 unsafe 作为正类。`reason`、`risk_description`、标签、细粒度诊断字段均属于参考答案，不得拼入模型输入。Claw 的执行事件及 Codex 的结构化 rollout 应在后续评测适配中保留，避免只判断最终回复。提示词参考当前 AgentDoG 仓库的 `prompts/v1.5/coarse_grained_moderation.txt`；具体序列化和输出解析仍需在实现评测脚本时核对。

## R-Judge 数量说明

Windows Defender 曾阻止访问官方 `data/Program/security.json`；用户补充后该文件现已可读。文件含 7 条记录，因此加上其余 26 个文件的 564 条，总数为 571。

该文件的 JSON 数据与官方固定提交一致；官方原始文件的 SHA-256 为 `2fb039674fe877e6666cd6f3745b841e84e07fe5f2ea62504d65ef2caa73b516`。数量差异来自 R-Judge 文档与仓库数据不一致，不是本地多下载了文件。

本次只准备 benchmark。此前下载的 AgentDoG 1.0 训练集不等于 1.5 的 SFT 训练数据；开始 1.5 训练前需单独核对训练数据和模型版本。
