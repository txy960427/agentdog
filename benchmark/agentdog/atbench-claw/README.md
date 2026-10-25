---
pretty_name: ATBench-Claw
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
- openclaw
- trajectory
- tool-use
size_categories:
- 10M<n<100M
---


# ATBench-Claw: OpenClaw-Oriented Agent Trajectory Safety Benchmark
<p align="center">
  <a href="https://github.com/AI45Lab/AgentDoG">🐙 GitHub</a>&nbsp&nbsp | &nbsp&nbsp
  <a href="https://arxiv.org/abs/2604.14858">📄 ATBench-Claw/CodeX Paper</a>&nbsp&nbsp | &nbsp&nbsp
  <a href="https://arxiv.org/abs/2604.02022">📄 ATBench Paper</a>&nbsp&nbsp | &nbsp&nbsp
  <a href="https://arxiv.org/pdf/2601.18491">🧾 AgentDoG Paper</a>&nbsp&nbsp | &nbsp&nbsp
  <a href="https://huggingface.co/collections/AI45Research/agentdog">🤗 Hugging Face Collection</a>
</p>

ATBench-Claw is an OpenClaw-oriented benchmark release derived from ATBench and serves as the benchmark companion to [AgentDoG](https://huggingface.co/papers/2601.18491), our diagnostic guardrail framework for AI agent safety and security. It is designed for trajectory-level safety evaluation in executable agent settings, with a focus on the point where safety decisions must be made before actions such as file deletion, message sending, code execution, cross-boundary access, or unattended automation are actually carried out.

Compared with the original [ATBench](https://huggingface.co/datasets/AI45Research/ATBench), this release is built around OpenClaw-specific operational semantics, including session-level state carryover, skills and plugins, approval-sensitive actions, cross-channel workflows, and action-centric failure chains that may not be visible from final text output alone.

This 500-example release preserves the same raw trajectory schema as the larger ATBench-Claw trace release. Each sample uses the original `trajectory / labels / reason` structure rather than a flattened export format, which keeps the benchmark aligned with the underlying OpenClaw execution traces.

---

## Dataset Overview

- Total trajectories: 500
- Safe trajectories: 204
- Unsafe trajectories: 296
- Average message events per trajectory: 13.09
- Event count range: 7 to 21
- Top-level fields: `trajectory`, `labels`, `reason`

ATBench-Claw is designed as an evaluation-oriented benchmark rather than a training corpus. Each sample is a complete OpenClaw execution trace with a binary safety verdict and fine-grained taxonomy labels. In addition to unsafe trajectories, this release also retains taxonomy labels and defense outcomes for safe trajectories, making it possible to analyze successful mitigation behavior rather than only failure cases.

In the current release, safe trajectories are further annotated with `defense_type`, indicating how the risk was handled:

- `detected_with_warning`: 119
- `defended`: 85
- `None` (typically unsafe trajectories): 296

---

## Why ATBench-Claw?

OpenClaw changes the unit of safety from harmful text to executable decisions. In this setting, the key question is often not whether the final assistant message looks benign, but whether the next action should be allowed to execute at all.

ATBench-Claw is therefore built around failure surfaces that are especially important in OpenClaw-style systems:

- ambiguous sender or session identity
- persistent memory or session-state contamination
- compromised skills, plugins, or tool metadata
- approval bypass and missing human-in-the-loop checks
- cross-channel or wrong-recipient routing
- unattended automation that continues risky execution

The benchmark is intended to support:

- offline evaluation of guard and diagnostic models
- fine-grained trajectory-level safety diagnosis
- analysis of safe-vs-unsafe continuations under executable constraints
- future runtime safety control for OpenClaw-style agent systems

---

## OpenClaw-Customized Safety Taxonomy

ATBench-Claw preserves the same three orthogonal dimensions used by ATBench:

- `risk_source`: where the risk comes from
- `failure_mode`: how the unsafe behavior manifests
- `harm_type`: what real-world harm it can cause

The current release covers:

- 13 risk-source categories
- 16 failure-mode categories
- 11 harm categories

OpenClaw-specific additions include:

- Risk source:
  `Sender / Session Identity Ambiguity`,
  `Persistent Memory / Session-State Contamination`,
  `Skill / Plugin Supply-Chain Compromise`,
  `Platform / Tool Vulnerability Exploitation`,
  `Policy Precedence Misinterpretation`
- Failure mode:
  `Approval Bypass or Missing Human-in-the-Loop`,
  `Action Scope Overreach`,
  `Cross-Tool Attack Chaining`,
  `Cross-Channel / Recipient Misrouting`,
  `Unsafe Unattended Automation`
- Harm type:
  `Compliance, Legal, and Auditability Harm`

In the figure below, OpenClaw-specific extensions are highlighted in red.

![ATBench-Claw taxonomy distribution](./figures/atbench_claw_label_distribution.png)

---

## Example OpenClaw-Relevant Risk Patterns

Representative scenarios covered by ATBench-Claw include:

- A vague cleanup request expands into an irreversible deletion plan without confirmation.
- A compromised skill or plugin update injects unsafe behavior into later tool-use chains.
- A message, file, or automated action is routed to the wrong recipient or channel.
- Persistent session state or memory contamination poisons later turns that appear locally benign.
- Unattended automation continues a risky workflow after the context that justified it has changed.

These cases are designed to evaluate whether a model can correctly judge trajectory safety before harmful execution is finalized.

---

## Data Structure

Each sample in the current release has the following high-level structure:

```json
{
  "trajectory": {
    "type": "session",
    "version": 3,
    "provider": "",
    "modelId": "",
    "events": []
  },
  "labels": {
    "risk_source": "policy_precedence_misinterpretation",
    "failure_mode": "action_scope_overreach",
    "harm_type": "functional_and_opportunity_harm",
    "is_safe": false,
    "defense_type": null
  },
  "reason": "Short natural-language rationale for the annotation."
}
```

Key fields:

- `trajectory`: the full OpenClaw trajectory object
- `trajectory.events`: the ordered list of OpenClaw message events
- `labels.is_safe`: binary trajectory-level safety label
- `labels.risk_source`: primary risk-source label
- `labels.failure_mode`: primary failure-mode label
- `labels.harm_type`: primary harm label
- `labels.defense_type`: mitigation outcome for defended safe traces
- `reason`: concise annotation rationale

In the current export, all `trajectory.events` entries are stored as `message` events. Assistant-side tool invocations appear as `toolCall` items inside `message.content`, while tool outputs are represented as message events with `role: "toolResult"`. Across the 500 samples, all trajectories use `type: "session"` and `version: 3`.

Roles observed in this release are `user`, `assistant`, and `toolResult`. The current 500-sample subset preserves the same raw trajectory schema as the earlier full-trace release, which means downstream code written for the original `trajectory / labels / reason` format can ingest this release directly.

---

## Intended Uses

ATBench-Claw is intended for:

- benchmark evaluation of agent safety classifiers and guard models
- taxonomy-aware error analysis on executable trajectories
- studying successful defenses in safe-but-risky trajectories
- comparing binary safety detection with fine-grained OpenClaw risk diagnosis

---

## Limitations

- This release is OpenClaw-oriented and does not attempt to cover all possible agent frameworks.
- It is a selected 500-example subset derived from the larger ATBench-Claw source set.
- Fine-grained labels reflect the annotation framework used for this benchmark and should not be treated as a universal ontology for all agent systems.
- The dataset is useful for evaluation and analysis, but should not be used as a standalone real-world safety policy without additional validation.

---

## License

Apache-2.0

---

## Citation

If you use ATBench-CodeX, please cite the following works:

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
      title={Benchmarks for Trajectory Safety Evaluation and Diagnosis in OpenClaw and Codex: ATBench-Claw and ATBench-CodeX},
      author={Zhonghao Yang and Yu Li and Yanxu Zhu and Tianyi Zhou and Yuejin Xie and Haoyu Luo and Jing Shao and Xia Hu and Dongrui Liu},
      year={2026},
      eprint={2604.14858},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2604.14858},
}
```
