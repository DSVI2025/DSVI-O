# DSVI-O

DSVI-O is a computational framework for dynamic stochastic systems in which an
upper-level state evolves continuously while a lower-level stochastic
variational inequality or optimization problem produces feedback responses.
The framework is motivated by systems whose decisions depend on time-varying
random inputs, equilibrium constraints, and historical observations.

This repository accompanies the manuscript **“Transfer Learning in
Differential Stochastic Variational Inequalities with History-Dependent
Responses.”** The paper extends the DSVI-O elderly-health benchmark to study
history-dependent response trajectories and their transfer across related
stochastic environments.

## Research overview

The history-dependent DSVI model couples two components:

- an upper-level projected differential system for the evolving state; and
- a lower-level stochastic variational inequality, represented in the
  application by an optimization problem, whose response depends on the
  stopped history of the stochastic process.

The associated paper studies well-posedness, sample average approximation, and
stability under changes in the initial state and the law of the exogenous
process. Its computational study uses an elderly-health embodied-intelligence
benchmark to examine two practical questions:

1. How robust is the predicted health-state trajectory to perturbations in
   wearable and clinical sensor streams?
2. Can precomputed response trajectories be transferred from source users to
   related target users without recomputing the target response online?

## Elderly-health benchmark

The benchmark contains synthetic multimodal records for source and target
cohorts. Each user has 100 days of observations sampled every five seconds,
with three data modalities:

- smartwatch physiological and activity signals;
- intelligent-insole measurements; and
- electronic medical record features.

The source cohort contains ten users from the DSVI-O benchmark. The transfer
study generates a separate ten-user target cohort under related demographic,
health, and living-environment conditions. Source and target users do not share
trajectories, health-state labels, or precomputed response trajectories.

Generated datasets and experiment outputs are not committed to Git. The code
creates them under `data/` and `results/`.

## Code and experiments

The repository contains the data generators and the computational experiments
from the manuscript’s elderly-health application.

| ID | Study | Main entry point |
| --- | --- | --- |
| S6-E0 | Generate independent source and target cohorts | `scripts/generate_section6_data.sh` |
| S6-E1 | Evaluate source-domain held-out performance | `scripts/run_s6_e1_source_domain.sh` |
| S6-E2 | Compute the target-domain full-response reference | `scripts/run_s6_e2_target_full_recomputation.sh` |
| S6-E2a | Study sparse target-response updates | `scripts/run_s6_e2_target_response_density.sh` |
| S6-E3 | Evaluate robustness under sensor perturbations | `scripts/run_s6_e3_noise_robustness.sh` |
| S6-E4 | Measure source-target user similarity | `scripts/run_s6_e4_source_target_similarity.sh` |
| S6-E5 | Evaluate nearest-source and similarity-weighted transfer | `scripts/run_s6_e5_similarity_weighted_topk.sh` |
| S6-E6 | Compare online latency across transfer methods | `scripts/run_s6_e6_transfer_latency.sh` |
| S6-E7 | Study delayed reuse of transferred responses | `scripts/run_s6_e7_delayed_response_reuse.sh` |

The main transfer method ranks source users by multimodal historical
similarity, combines the response trajectories of the top-K source users, and
uses the weighted response in the target user’s online state update. The target
observations remain active in every update; only the second-stage response is
transferred or reused.

## Documentation

- [Experiment and data inventory](EXPERIMENTS.md)
- [Execution details](REPRODUCING.md)
- [Mathematical definition of the mixed-noise construction](docs/mixed_noise_definition.pdf)

## Repository layout

```text
.
├── scripts/           # Data-generation and experiment launchers
├── src/               # Implementations grouped by experiment
├── docs/              # Supplementary mathematical documentation
├── EXPERIMENTS.md     # Experiment and data registry
├── REPRODUCING.md     # Environment and execution details
└── requirements.txt   # Python dependencies
```

## Software requirements

The implementation requires Python 3.9 or newer, Bash, and the packages listed
in `requirements.txt`. CUDA-capable PyTorch is recommended for the full
experiment suite; CPU execution is supported but substantially slower.

For detailed commands, input/output conventions, and runtime configuration,
see [REPRODUCING.md](REPRODUCING.md).
