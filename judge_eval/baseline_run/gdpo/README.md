# GDPO fine baseline (current judge_eval config)

Config:
- Prompt: `prompt_finegrained.txt` — labels Risk Source → Failure Mode → Risk Consequence
- `{risk_source_categories}`: appendix leaf definitions
- Rules: each dimension must use its own category list (RS ≠ FM name)
- Tools in `{tools_json}` (JSON); trajectory has no tool block; raw ACTION strings
- Parse: partial triples still score resolved dims; `strict_parse` needs all three
- Model: `AgentDoG1.5-FG-Qwen3.5-4B`；`enable_thinking=False`

Seed sweep (2026-10-25, same prompt/parse):

| Seed | RS | FM | RWH | Avg | Parse | Run |
|---:|---:|---:|---:|---:|---:|---|
| **42** | **74.4** | 26.0 | **63.0** | **54.5** | 99.4 | `runs/fine/all/20261025_024249_560945` |
| 7 | 74.2 | 26.0 | 63.0 | 54.4 | 99.2 | `runs/fine/all/20261025_024854_510377` |
| 123 | 73.6 | 26.2 | 62.6 | 54.1 | 99.4 | `runs/fine/all/20261025_024551_929321` |

Baseline snapshot = **seed 42** → `baseline_run/gdpo/atbench/`.

Paper (Table 3, 4B): 75.2 / 27.5 / 62.9 / 55.2
