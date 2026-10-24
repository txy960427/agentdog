# SFT coarse baseline (current judge_eval config)

Config:
- P1 layout: Profile → Tools → Conversation
- ATBench: prose tools; R-Judge: `Tool: name` from calls
- No trailing Available-tools section in `prompt.txt`
- thought + env 1g bulky head-tail; `enable_thinking=False`
- **seed=42**

Source runs (seed sweep, 2026-10-24):
- atbench: `runs/seed_sweep/atbench/20261024_123711_238158`
- rjudge: `runs/seed_sweep/rjudge/20261024_124406_283703`

| Dataset | Acc | P | R | F1 |
|---|---:|---:|---:|---:|
| atbench | 73.1 | 70.3 | 79.5 | 74.6 |
| rjudge | 93.0 | 93.1 | 93.7 | 93.4 |
