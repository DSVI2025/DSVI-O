# DSVI-O

This repository contains the data-generation and experiment code used for the
DSVI-O Section 6 reproducibility study. It provides a self-contained pipeline
for generating the synthetic source and target cohorts, converting them to the
MAT layout used by the experiments, and running experiments S6-E1 through
S6-E7.

Generated datasets and experiment outputs are intentionally not committed to
Git. The scripts create them under `data/` and `results/` by default.

## Experiment suite

| ID | Experiment | Entry point |
| --- | --- | --- |
| S6-E0 | Source/target cohort generation | `scripts/generate_section6_data.sh` |
| S6-E1 | Source-domain held-out benchmark | `scripts/run_s6_e1_source_domain.sh` |
| S6-E2 | Target-domain full-recomputation baseline | `scripts/run_s6_e2_target_full_recomputation.sh` |
| S6-E2a | Target response-density ablation | `scripts/run_s6_e2_target_response_density.sh` |
| S6-E3 | Sensor-noise robustness | `scripts/run_s6_e3_noise_robustness.sh` |
| S6-E4 | Source-target similarity | `scripts/run_s6_e4_source_target_similarity.sh` |
| S6-E5 | Similarity-weighted Top-K transfer | `scripts/run_s6_e5_similarity_weighted_topk.sh` |
| S6-E6 | Transfer-method latency comparison | `scripts/run_s6_e6_transfer_latency.sh` |
| S6-E7 | Delayed response reuse | `scripts/run_s6_e7_delayed_response_reuse.sh` |

See [EXPERIMENTS.md](EXPERIMENTS.md) for the complete experiment and data
inventory, and [REPRODUCING.md](REPRODUCING.md) for inputs, outputs, defaults,
and runtime overrides. A standalone mathematical description of the mixed-noise
construction used in S6-E3 is available in
[Mixed-noise definition](docs/mixed_noise_definition.pdf).

## Requirements

- Python 3.9 or newer
- Bash
- CUDA-capable PyTorch is recommended for the full experiment suite; CPU runs
  are supported but substantially slower

Create an isolated environment and install the Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch installation differs by platform. If the default package does not
match your CUDA runtime, install the appropriate build from the official
PyTorch package index before running the experiments.

## Quick start

Run a reduced data-generation smoke test:

```bash
NUM_VERSIONS=1 \
NUM_USERS=1 \
PROCESSES=1 \
SOURCE_OUTPUT_ROOT=data/smoke/source \
TARGET_OUTPUT_ROOT=data/smoke/target \
scripts/generate_section6_data.sh
```

Generate the complete source and target cohorts with the manuscript defaults:

```bash
scripts/generate_section6_data.sh
```

Then run experiments individually, for example:

```bash
DEVICE=cuda scripts/run_s6_e1_source_domain.sh
DEVICE=cuda scripts/run_s6_e2_target_full_recomputation.sh
DEVICE=cuda scripts/run_s6_e3_noise_robustness.sh
```

To run the complete pipeline from data generation through S6-E7:

```bash
DEVICE=cuda scripts/run_section6_all.sh
```

`run_section6_all.sh` defaults to `CLEAN=1`, which removes the repository's
generated `data/` and `results/` directories before starting. Set `CLEAN=0` to
retain existing artifacts.

All runners accept environment-variable overrides for data paths, output
paths, devices, batching, and experiment-specific settings. These options are
documented in [REPRODUCING.md](REPRODUCING.md).

## Repository layout

```text
.
├── scripts/           # Data-generation and experiment launchers
├── src/               # Python implementation grouped by experiment
├── docs/              # Supplementary experiment documentation
├── EXPERIMENTS.md     # Section 6 experiment/data registry
├── REPRODUCING.md     # Detailed execution guide
└── requirements.txt   # Python dependencies
```

## Reproducibility checks

The source tree can be checked without generating data:

```bash
python -m compileall -q src
for script in scripts/*.sh; do bash -n "$script"; done
```

The full suite is computationally and storage intensive: the default data
configuration creates 10 versions for each of 20 users, with 10 days per
version sampled every five seconds.
