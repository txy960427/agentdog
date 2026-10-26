# AgentDoG reproduction workspace

## Repository

- Root: `D:/agentdog`
- Git remote: `https://github.com/txy960427/agentdog.git`
- `AgentDoG/`: AgentDoG implementation and released prompts
- `verl/`: training framework dependency

Reproduction assets are kept beside `verl/` so updating the framework does not mix framework code with local datasets, models, pipelines, or evaluation runs:

```text
agentdog/
├── AgentDoG/                  # AgentDoG implementation
├── verl/                      # training framework only
├── data/agentdog/             # source, canonical and final training data
├── data_pipeline/agentdog/    # deterministic data-building pipelines
├── benchmark/agentdog/        # external evaluation datasets
├── model/agentdog/            # local model snapshots
├── judge_eval/                # external benchmark evaluation
└── docs/agentdog15_reproduction/
```

## Training data

- Location: `data/agentdog/1.0/source/AgentDoG1.0-Training-Data/`
- Source: `https://huggingface.co/datasets/AI45Research/AgentDoG1.0-Training-Data`
- Revision: `f4da9da000e1bde5dce851e522bcb1e3c50684ed`
- Version: v1.0

The public source contains:

- `AgentDoG-BinarySafety/train.json`: 4,000 physical records; 2,000 unique records after exact deduplication.
- `AgentDoG-FineGrainedTaxonomy/train.json`: 4,000 records.

Both source files retain their original `instruction/input/output` schema. Checksums and source metadata are stored beside the downloaded files.

## Data pipeline

Run from the repository root:

```bash
python3 data_pipeline/agentdog/1.5/coarse-grained/sft/pipeline.py step1
python3 data_pipeline/agentdog/1.5/coarse-grained/sft/pipeline.py step2
python3 data_pipeline/agentdog/1.5/coarse-grained/sft/pipeline.py all
```

The generated canonical data and audit reports are written to:

```text
data/agentdog/1.5/coarse-grained/sft/
```

The v1.0 public dataset is source material for the reproduction design; it is not a complete release of the AgentDoG 1.5 training set.
