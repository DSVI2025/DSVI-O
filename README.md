# DSVI-O

This repository provides computational materials for differential stochastic
variational inequalities with parametric optimization (DSVI-O) and their
extension to history-dependent response transfer.

## Mathematical scope

DSVI-O describes a class of dynamic stochastic systems in which a continuously
evolving upper-level state is coupled with the solution of a lower-level
parametric optimization problem. The lower-level solution acts as an endogenous
response in the state dynamics, allowing equilibrium and optimization
constraints to enter a stochastic dynamical model.

The history-dependent extension replaces the instantaneous lower-level
response by a response map defined on the stopped history of the exogenous
process. The resulting formulation couples a projected differential system
with history-dependent stochastic variational inequalities. Its analysis
concerns well-posedness of the closed-loop system, sample average approximation,
and stability with respect to the initial state and the probability law of the
stochastic environment.

The transfer-learning setting considers related source and target
environments. Response trajectories constructed in the source domain are
transferred to the target domain through a similarity-weighted multi-source
rule, while target observations remain in the upper-level state update. This
separates transfer of the lower-level response from direct substitution of the
target trajectory.

## Related work

1. X. Chen, J. Guo, and G. Wang, **“Differential Stochastic Variational
   Inequalities with Parametric Optimization,”** 2025.
   [[arXiv](https://arxiv.org/abs/2508.15241)]
   [[PDF](https://arxiv.org/pdf/2508.15241)]

2. **“Transfer Learning in Differential Stochastic Variational Inequalities
   with History-Dependent Responses.”**

## Elderly-health application

The accompanying application concerns an elderly-health embodied-intelligence
system with synthetic multimodal observations. Each individual is represented
by smartwatch signals, intelligent-insole measurements, electronic medical
record features, and a time-dependent health state. Observations are generated
at five-second resolution over 100 days per individual.

The source and target cohorts contain ten users each and are generated under
related demographic, health, and living-environment conditions. They remain
disjoint at the level of user trajectories, health-state labels, and
precomputed response trajectories. This construction supports the study of
stability under sensor perturbations and transfer of history-dependent
responses across related individuals.

## Repository contents

The repository includes:

- generators for the synthetic multimodal source and target cohorts;
- numerical implementations of the source- and target-domain DSVI systems;
- constructions for sensor perturbations and response-update sparsification;
- source-target similarity and similarity-weighted response transfer methods;
- delayed response-reuse and computational-cost analyses; and
- supplementary mathematical and implementation documentation.

The source code is organized by computational component under `src/`, with
corresponding command-line entry points under `scripts/`. A detailed inventory
is provided in [EXPERIMENTS.md](EXPERIMENTS.md). Environment requirements,
input/output conventions, and execution details are collected separately in
[REPRODUCING.md](REPRODUCING.md).

The mathematical construction of the mixed sensor perturbation used in the
robustness study is documented in
[mixed_noise_definition.pdf](docs/mixed_noise_definition.pdf).

Generated datasets and numerical outputs are intentionally excluded from the
repository. By default, they are written to `data/` and `results/`,
respectively.

## Software

The implementation requires Python 3.9 or newer and the packages listed in
`requirements.txt`. CUDA-capable PyTorch is recommended for the full numerical
study; CPU execution is supported but substantially slower.
