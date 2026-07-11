# Reproducing the Section 6 Experiments

This document describes the computational workflows used for the Section 6
evaluation. Run all commands from the repository root.

## Data

The source and target CSV cohorts are committed under:

```text
data/source
data/target
```

The committed data use the following configuration:

- `NUM_VERSIONS=10`
- `NUM_USERS=10`
- source user IDs: `1-10`
- target user IDs: `11-20`
- each version/user contains 10 days of 5-second data, `172800` rows per time-series file

Large health and insole CSV files are managed with Git LFS. Run `git lfs pull`
after cloning if the LFS objects were not downloaded automatically.

## S6-E1 Source-Domain Experiment

S6-E1 code:

- `src/s6_e1_source_domain/run_source_domain_v10_test.py`
- `src/s6_e1_source_domain/run_old_matlab_equivalent.py`
- `src/s6_e1_source_domain/generate_linear_test_mat.py`

Run script:

```bash
scripts/run_s6_e1_source_domain.sh
```

Default MAT input root:

```text
data/mat/source
```

If the MAT files are missing but `data/source` exists, the run script
automatically converts the source CSV cohort into MAT files first.

The input root must contain:

- `healthData.mat`
- `insoleData.mat`
- `EMRData.mat`

Additional Python dependencies for S6-E1 are `scipy` and `torch`.

Override inputs/outputs and runtime with environment variables:

```bash
SOURCE_DATA_ROOT=data/source \
DATA_DIR=/path/to/generated_linear_test_mat \
OUT_ROOT=results/s6_e1_source_domain \
DEVICE=cuda \
TARGET_DAY_BATCH_SIZE=4 \
HISTORY_CHUNK_SIZE=256 \
scripts/run_s6_e1_source_domain.sh
```

## S6-E2 Target-Domain Full-Recomputation Baseline

S6-E2 code:

- `src/s6_e2_target_full_recomputation/run_old_matlab_equivalent.py`
- `src/s6_e2_target_full_recomputation/generate_linear_test_mat.py`
- `src/s6_e2_target_full_recomputation/merge_source_target_mat.py`
- `src/s6_e2_target_full_recomputation/summarize_target_full_recomputation.py`

Run script:

```bash
scripts/run_s6_e2_target_full_recomputation.sh
```

Default MAT input root:

```text
data/mat/source_target_shared_emr
```

By default, `TARGET_EMR_MODE=source_shared`, matching the MAT layout used by
the manuscript Table 2 result. If the MAT files are missing, the run script
converts the source and target CSV cohorts and merges them so target users
reuse the source shared EMR matrix. To use each target user's own EMR instead,
set `TARGET_EMR_MODE=target`.

The default target users are `11-20`, and the fixed target evaluation days are
defined in the runner as `[66, 4, 100, 72, 61, 48, 85, 36, 52, 46]`.

Default outputs:

- `results/s6_e2_target_full_recomputation/all_users_summary.json`
- `results/s6_e2_target_full_recomputation/table2_target_full_recomputation.csv`
- `results/s6_e2_target_full_recomputation/table2_target_full_recomputation_latex_rows.tex`

Additional Python dependencies for S6-E2 are `scipy` and `torch`.

Override runtime settings with environment variables:

```bash
TARGET_DATA_ROOT=data/target \
SOURCE_DATA_ROOT=data/source \
DATA_DIR=data/mat/source_target_shared_emr \
OUT_ROOT=results/s6_e2_target_full_recomputation \
DEVICE=cuda \
TARGET_DAY_BATCH_SIZE=4 \
HISTORY_CHUNK_SIZE=256 \
scripts/run_s6_e2_target_full_recomputation.sh
```

## S6-E3 Noise Robustness

S6-E3 code:

- `src/s6_e3_noise_robustness/run_noise_sweep_suite.py`
- `src/s6_e3_noise_robustness/run_old_matlab_equivalent.py`
- `src/s6_e3_noise_robustness/generate_linear_test_mat.py`
- `src/s6_e3_noise_robustness/summarize_noise_robustness.py`
- `src/s6_e3_noise_robustness/plot_noise_robustness_publication.py`

Run script:

```bash
scripts/run_s6_e3_noise_robustness.sh
```

Default MAT input root:

```text
data/mat/source_legacy_shared_emr
```

If the MAT files are missing but `data/source` exists, the run script
automatically converts the source CSV cohort into the legacy shared-EMR MAT
layout used by the manuscript noise-robustness result.

Default settings match the manuscript Section 6.2 case study:

- representative source user: `1`
- noise families: additive, drift, multiplicative, cumulative, Laplace, impulse, mixed
- strength values: `0.5,1,1.5,2,5,10,20`
- reference sampling: `50` reference days, `20` trials, seed `20260423`
- noise seed: `20260423`

Default outputs:

- `results/s6_e3_noise_robustness/suite_results.json`
- `results/s6_e3_noise_robustness/noise_accuracy_by_type_vs_strength.csv`
- `results/s6_e3_noise_robustness/table3_noise_robustness.csv`
- `results/s6_e3_noise_robustness/table3_noise_robustness_latex_rows.tex`
- `results/s6_e3_noise_robustness/application_noise_robustness.png`

Additional Python dependencies for S6-E3 are `scipy`, `torch`, and
`matplotlib`.

Override runtime settings with environment variables:

```bash
SOURCE_DATA_ROOT=data/source \
DATA_DIR=data/mat/source_legacy_shared_emr \
OUT_ROOT=results/s6_e3_noise_robustness \
DEVICE=cuda \
MAX_PARALLEL=2 \
scripts/run_s6_e3_noise_robustness.sh
```

## S6-E4 Source-Target Similarity

S6-E4 code:

- `src/s6_e4_source_target_similarity/compute_source_target_similarity.py`
- `src/s6_e4_source_target_similarity/run_old_matlab_equivalent.py`
- `src/s6_e4_source_target_similarity/generate_linear_test_mat.py`
- `src/s6_e4_source_target_similarity/merge_source_target_mat.py`

Run script:

```bash
scripts/run_s6_e4_source_target_similarity.sh
```

Default MAT input root:

```text
data/mat/source_target_shared_emr
```

By default, `TARGET_EMR_MODE=source_shared`, matching the shared-EMR MAT layout
used by the paper-similarity transfer runs. If the MAT files are missing, the
run script converts the committed source and target CSV cohorts and merges them
before computing similarity.

Default settings match manuscript Section 6.3.1:

- source users: `1-10`
- target users: `11-20`
- historical rows: first `90` days
- modality weights: `watch=2`, `insole=2`, `emr=1`
- pairwise pooled standardization within each target-source pair

Default outputs:

- `results/s6_e4_source_target_similarity/pairwise_similarity.csv`
- `results/s6_e4_source_target_similarity/pairwise_similarity.json`
- `results/s6_e4_source_target_similarity/source_target_transfer_weights.csv`
- `results/s6_e4_source_target_similarity/target_similarity_summary.csv`
- `results/s6_e4_source_target_similarity/similarity_config.json`
- `results/s6_e4_source_target_similarity/summary.md`

Additional Python dependencies for S6-E4 are `scipy`, `torch`, and `numpy`.

Override runtime settings with environment variables:

```bash
TARGET_DATA_ROOT=data/target \
SOURCE_DATA_ROOT=data/source \
DATA_DIR=data/mat/source_target_shared_emr \
OUT_ROOT=results/s6_e4_source_target_similarity \
SIMILARITY_CHUNK_SIZE=256 \
scripts/run_s6_e4_source_target_similarity.sh
```

## S6-E5 Similarity-Weighted Top-K Transfer

S6-E5 code:

- `src/s6_e5_similarity_weighted_topk/export_transfer_response_cache.py`
- `src/s6_e5_similarity_weighted_topk/run_transfer_learning_suite_paper_similarity.py`
- `src/s6_e5_similarity_weighted_topk/_transfer_learning_common.py`
- `src/s6_e5_similarity_weighted_topk/summarize_transfer_learning_results.py`
- `src/s6_e5_similarity_weighted_topk/plot_target_transfer_k_accuracy.py`
- `src/s6_e5_similarity_weighted_topk/plot_target_transfer_k_boxplot.py`
- `src/s6_e5_similarity_weighted_topk/summarize_topk_transfer.py`
- `src/s6_e5_similarity_weighted_topk/run_old_matlab_equivalent.py`
- `src/s6_e5_similarity_weighted_topk/generate_linear_test_mat.py`
- `src/s6_e5_similarity_weighted_topk/merge_source_target_mat.py`

Run script:

```bash
scripts/run_s6_e5_similarity_weighted_topk.sh
```

Default MAT input root:

```text
data/mat/source_target_shared_emr
```

Default baseline root:

```text
results/s6_e2_target_full_recomputation
```

Default source-response cache:

```text
results/s6_e5_similarity_weighted_topk/source_response_cache
```

If the source-response cache is missing, the run script exports it first from
the stable MAT data. The Top-K transfer stage then uses the manuscript
paper-similarity setting from S6-E4: first `90` historical rows,
pairwise pooled standardization, and modality weights `watch=2`, `insole=2`,
`emr=1`.

Default outputs:

- `results/s6_e5_similarity_weighted_topk/pairwise_similarity.csv`
- `results/s6_e5_similarity_weighted_topk/method1_all_pairs.csv`
- `results/s6_e5_similarity_weighted_topk/method2_k_sweep.csv`
- `results/s6_e5_similarity_weighted_topk/method2_best_k_summary.json`
- `results/s6_e5_similarity_weighted_topk/target_transfer_k_accuracy_mean.csv`
- `results/s6_e5_similarity_weighted_topk/table4_transfer_topk_latex_rows.tex`
- `results/s6_e5_similarity_weighted_topk/target_transfer_k_accuracy.png`
- `results/s6_e5_similarity_weighted_topk/target_transfer_k_accuracy_boxplot.png`

Additional Python dependencies for S6-E5 are `scipy`, `torch`, `numpy`, and
`matplotlib`.

Override runtime settings with environment variables:

```bash
DATA_DIR=data/mat/source_target_shared_emr \
BASELINE_ROOT=results/s6_e2_target_full_recomputation \
OUT_ROOT=results/s6_e5_similarity_weighted_topk \
CACHE_ROOT=results/s6_e5_similarity_weighted_topk/source_response_cache \
DEVICE=cuda \
HISTORY_CHUNK_SIZE=1024 \
scripts/run_s6_e5_similarity_weighted_topk.sh
```
