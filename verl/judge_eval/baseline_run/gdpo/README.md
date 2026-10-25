# GDPO fine baseline (current judge_eval config)

Config:
- Prompt: `prompt_finegrained.txt` — final labels Risk Source → Failure Mode → Risk Consequence
- `{risk_source_categories}`: appendix leaf definitions
- Tools in `{tools_json}` (JSON schemas); trajectory has no tool block
- ATBench ACTION: raw dataset strings (no thought)
- Model: `AgentDoG1.5-FG-Qwen3.5-4B`；`enable_thinking=False`；**seed=42**

Source run (2026-10-25):
- atbench: `runs/fine/all/20261025_022454_906837`

| Dataset | Risk Source | Failure Mode | Real-world Harm | Avg. | Parse |
|---|---:|---:|---:|---:|---:|
| atbench (497 unsafe) | 74.2 | 26.4 | 62.0 | 54.2 | 99.2 |

Paper (Table 3, 4B): 75.2 / 27.5 / 62.9 / 55.2
