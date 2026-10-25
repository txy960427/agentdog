---
pretty_name: ATBench-Codex
language:
- en
license: apache-2.0
task_categories:
- text-classification
task_ids:
- multi-class-classification
tags:
- agent
- safety
- benchmark
- codex
- trajectory
- tool-use
size_categories:
- 10M<n<100M
---

# ATBench-Codex: Codex-Oriented Agent Trajectory Safety Benchmark
<p align="center">
  <a href="https://github.com/AI45Lab/AgentDoG">🐙 GitHub</a>&nbsp&nbsp | &nbsp&nbsp
  <a href="https://arxiv.org/abs/2604.14858">📄 ATBench-Claw/Codex Paper</a>&nbsp&nbsp | &nbsp&nbsp
  <a href="https://arxiv.org/abs/2604.02022">📄 ATBench Paper</a>&nbsp&nbsp | &nbsp&nbsp
  <a href="https://arxiv.org/pdf/2601.18491">🧾 AgentDoG Paper</a>&nbsp&nbsp | &nbsp&nbsp
  <a href="https://huggingface.co/collections/AI45Research/agentdog">🤗 Hugging Face Collection</a>
</p>

ATBench-Codex is a Codex-oriented benchmark release derived from ATBench and serves as a benchmark companion to [AgentDoG](https://huggingface.co/papers/2601.18491), our diagnostic guardrail framework for AI agent safety and security. It is designed for trajectory-level safety evaluation in executable coding-agent settings, with a focus on the point where safety decisions must be made before actions such as shell execution, workspace mutation, repository changes, MCP tool invocation, or long-horizon tool chaining are actually carried out.

Compared with the original [ATBench](https://huggingface.co/datasets/AI45Research/ATBench), this release is built around Codex-specific operational semantics, including multi-tool coding workflows, structured rollout events, repository and artifact manipulation, MCP server supply-chain surfaces, and instruction-following failures that only become visible once actions are executed against a live workspace.

This 500-example release preserves the current Codex export schema directly, including a normalized `conversation`, a structured `codex_rollout`, top-level safety taxonomy fields, and per-example tool metadata.

***

## Dataset Overview

- Total trajectories: 500
- Safe trajectories: 250
- Unsafe trajectories: 250
- Average conversation turns per sample: 7.51
- Conversation turn range: 6 to 17
- Average rollout events per sample: 21.80
- Rollout event range: 19 to 34
- Storage format: JSON release file `test.json`
- Core top-level fields: `id`, `conversation`, `codex_rollout`, `risk_source`, `failure_mode`, `harm_type`, `is_safe`, `defense_type`, `reason`, `tool_used`, `output_format`
- Optional top-level field: `injected_tool_descriptions`

ATBench-Codex is designed as an evaluation-oriented benchmark rather than a training corpus. Each sample pairs a user-facing conversation with a structured Codex rollout trace, a binary safety verdict, and fine-grained taxonomy labels. This makes it suitable for both conversation-level safety detection and deeper analysis of tool-driven agent trajectories.

In this release, `output_format` is always `codex_rollout`. The normalized `conversation` captures the user / assistant / environment transcript, while `codex_rollout` records the underlying execution trace with event types such as `session_meta`, `turn_context`, `response_item`, and `event_msg`.

`defense_type` should not be treated as a direct synonym for safety in this release. It is populated as:

- `detected_with_warning`: 136
- `defended`: 12
- `partial_refusal`: 45
- `refused`: 28
- `None`: 279

Notably, `None` appears in both safe and unsafe samples in this Codex export.

***

## Why ATBench-Codex?

Codex changes the unit of safety from harmful text to executable coding actions. In this setting, the central question is often not whether the final assistant message sounds harmless, but whether the next shell command, file mutation, repository action, or MCP tool call should be allowed to execute at all.

ATBench-Codex is therefore built around failure surfaces that are especially important in Codex-style systems:

- repository or artifact injection that poisons later decisions
- MCP and dependency supply-chain compromise
- unsafe shell-script execution and destructive workspace mutation
- incorrect tool parameters and failure to validate tool outputs
- cross-tool attack chaining across coding and data systems
- over-privileged or insufficiently confirmed actions in a live workspace

The benchmark is intended to support:

- benchmark evaluation of coding-agent safety classifiers and guard models
- taxonomy-aware error analysis on structured Codex rollouts
- studying safe-vs-unsafe continuations in executable coding workflows
- future runtime safety control for Codex-style agent systems

***

## Codex-Oriented Safety Taxonomy

ATBench-Codex preserves the same three orthogonal dimensions used by ATBench:

- `risk_source`: where the risk comes from
- `failure_mode`: how the unsafe behavior manifests
- `harm_type`: what real-world harm it can cause

The current release covers:

- 9 risk-source categories
- 18 failure-mode categories
- 10 harm categories

Representative Codex-oriented categories include:

- Risk source:
`Dependency / MCP Supply-Chain Compromise`,
`Repository Artifact Injection`,
`Tool Description Injection`,
`Policy Precedence Misinterpretation`
- Failure mode:
`Cross-Tool Attack Chaining`,
`Destructive Workspace Mutation`,
`Failure to Validate Tool Outputs`,
`Incorrect Tool Parameters`,
`Unsafe Shell Script Execution`,
`Choosing Malicious Tool`

![ATBench-Codex taxonomy distribution](./figures/atbench_codex_label_distribution.png)

***

## Example Codex-Relevant Risk Patterns

Representative scenarios covered by ATBench-Codex include:

- A coding request leads to an unsafe shell command or destructive workspace mutation without sufficient confirmation.
- A repository artifact or injected tool description biases later tool selection and action planning.
- An MCP or dependency source introduces compromised behavior into a multi-tool coding workflow.
- The agent fails to validate tool outputs and propagates incorrect or unsafe results into later steps.
- A seemingly benign coding task escalates into an over-privileged action against files, services, or infrastructure.

These cases are designed to evaluate whether a model can correctly judge trajectory safety before harmful execution is finalized.

***

## Data Structure

Each sample in the current release has the following high-level structure:

```json
{
"id": 2279,
"conversation": [
{"role": "user", "content": "..."},
{"role": "assistant", "content": "..."},
{"role": "environment", "content": "..."}
],
"codex_rollout": [
{
"timestamp": "2026-04-13T00:06:15.675Z",
"type": "session_meta",
"payload": {}
}
],
"risk_source": "dependency_mcp_supply_chain_compromise",
"failure_mode": "cross_tool_attack_chaining",
"harm_type": "functional_and_opportunity_harm",
"is_safe": true,
"defense_type": "detected_with_warning",
"reason": "Short natural-language rationale for the annotation.",
"tool_used": [],
"output_format": "codex_rollout"
}
```

Key fields:

- `id`: source identifier carried over from the larger 3k-example Codex pool
- `conversation`: normalized conversation transcript with roles drawn from `user`, `assistant`, and `environment`
- `codex_rollout`: structured Codex execution trace
- `risk_source`: primary risk-source label
- `failure_mode`: primary failure-mode label
- `harm_type`: primary harm label
- `is_safe`: binary trajectory-level safety label
- `defense_type`: defense outcome annotation when available
- `reason`: concise annotation rationale
- `tool_used`: list of tool specifications exposed to the agent in that sample
- `output_format`: output format tag, always `codex_rollout` in this release
- `injected_tool_descriptions`: optional dictionary of injected or modified tool descriptions when present

Across the 500 samples, `conversation` averages 7.51 turns and `codex_rollout` averages 21.80 events. The rollout trace includes `session_meta`, `turn_context`, `response_item`, and `event_msg` records. Within `response_item`, the most common payload types are `message`, `function_call`, `function_call_output`, and `reasoning`.

Most samples expose between 2 and 3 tools, with 187 distinct tool names appearing across the release. `injected_tool_descriptions` is present in 34 samples and omitted or empty in the remainder.

***

## Intended Uses

ATBench-Codex is intended for:

- benchmark evaluation of Codex-style agent safety classifiers and guard models
- taxonomy-aware error analysis on executable coding trajectories
- studying safety outcomes in tool-rich coding and MCP workflows
- comparing binary safety detection with fine-grained Codex risk diagnosis

***

## Limitations

- This release is Codex-oriented and does not attempt to cover all possible coding-agent frameworks.
- It is a selected 500-example subset derived from a larger 3k-example Codex source pool.
- `defense_type` is informative but not equivalent to the binary safety label in this export.
- Fine-grained labels reflect the annotation framework used for this benchmark and should not be treated as a universal ontology for all agent systems.
- The dataset is useful for evaluation and analysis, but should not be used as a standalone real-world safety policy without additional validation.

***

## License

Apache-2.0

***

## Citation

If you use ATBench-Codex, please cite the following works:

```bibtex
@misc{li2026atbenchdiverserealistictrajectory,
      title={ATBench: A Diverse and Realistic Trajectory Benchmark for Long-Horizon Agent Safety},
      author={Yu Li and Haoyu Luo and Yuejin Xie and Yuqian Fu and Zhonghao Yang and Shuai Shao and Qihan Ren and Wanying Qu and Yanwei Fu and Yujiu Yang and Jing Shao and Xia Hu and Dongrui Liu},
      year={2026},
      eprint={2604.02022},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2604.02022},
}
```

```bibtex
@misc{liu2026agentdogdiagnosticguardrailframework,
      title={AgentDoG: A Diagnostic Guardrail Framework for AI Agent Safety and Security},
      author={Dongrui Liu and Qihan Ren and Chen Qian and Shuai Shao and Yuejin Xie and Yu Li and Zhonghao Yang and Haoyu Luo and Peng Wang and Qingyu Liu and Binxin Hu and Ling Tang and Jilin Mei and Dadi Guo and Leitao Yuan and Junyao Yang and Guanxu Chen and Qihao Lin and Yi Yu and Bo Zhang and Jiaxuan Guo and Jie Zhang and Wenqi Shao and Huiqi Deng and Zhiheng Xi and Wenjie Wang and Wenxuan Wang and Wen Shen and Zhikai Chen and Haoyu Xie and Jialing Tao and Juntao Dai and Jiaming Ji and Zhongjie Ba and Linfeng Zhang and Yong Liu and Quanshi Zhang and Lei Zhu and Zhihua Wei and Hui Xue and Chaochao Lu and Jing Shao and Xia Hu},
      year={2026},
      eprint={2601.18491},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2601.18491},
}
```

```bibtex
@misc{yang2026benchmarkstrajectorysafetyevaluation,
      title={Benchmarks for Trajectory Safety Evaluation and Diagnosis in OpenClaw and Codex: ATBench-Claw and ATBench-Codex},
      author={Zhonghao Yang and Yu Li and Yanxu Zhu and Tianyi Zhou and Yuejin Xie and Haoyu Luo and Jing Shao and Xia Hu and Dongrui Liu},
      year={2026},
      eprint={2604.14858},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2604.14858},
}
```
