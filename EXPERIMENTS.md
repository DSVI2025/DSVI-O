# Section 6 Experiment and Data Inventory

Source: the reproducible configuration used for the paper's Section 6 experiments.

This document lists the data and experiments used in Section 6. Data generation is treated as a single preparation experiment.

## Data used in Section 6

| Data object | Contents | Scale and structure | Purpose |
| --- | --- | --- | --- |
| Source cohort | Ten source users from the DSVI-O synthetic elderly-health benchmark | Ten versions per user and ten days per version, for 100 days per user; one sample every five seconds and 17,280 time points per day | Source-domain held-out evaluation, noise robustness, source-response precomputation, and source-to-target transfer |
| Target cohort | Ten newly generated target users following the same DSVI-O benchmark protocol | Disjoint from the source cohort; ten versions and ten days per version, for 100 days per user and 17,280 time points per day | Target-domain baselines and cross-domain transfer evaluation |
| Smartwatch modality | Smartwatch time-series features | Feature dimension `m1=14` | Dynamic health-state input used in source-target similarity and noise perturbation |
| Intelligent-insole modality | Intelligent-insole time-series features | Feature dimension `m2=17` | Dynamic health-state input used in source-target similarity and noise perturbation |
| EMR modality | Static electronic medical record and clinical features | Feature dimension `m3=30` | Third input modality; not perturbed in the noise-robustness experiment |
| Health-state labels | Health state for every user, day, and time point | Generated as a continuous `status in [0,2]`, then discretized into healthy `(0)`, weak `(1)`, and ill `(2)` | Accuracy, precision, recall, specificity, and F1 evaluation |
| Historical reference blocks | Historical matrices sampled from reference days | At every time point and for every modality, sample 50 reference rows and repeat for 20 trials | Construction of the history-dependent lower-level response |
| Source response trajectories | Precomputed second-stage response trajectories for source users | Organized by source user, held-out day, modality, and time point | Reused during transfer learning instead of recomputing responses online in the target domain |

Generated datasets use the following English-only layout:

```text
data/{source,target}/vN/generated_data/
├── user_profiles.csv
├── medical_records.csv
├── electronic_medical_records/
├── health_data/
└── insole_data/
```

## Data-generation entry points

All paths are relative to the repository root.

| Purpose | Script | Default output |
| --- | --- | --- |
| Generate both Section 6 cohorts | `scripts/generate_section6_data.sh` | `data/source`, `data/target` |
| Generate only the source cohort | `scripts/generate_source_data.sh` | `data/source` |
| Generate only the target cohort | `scripts/generate_target_data.sh` | `data/target` |

## Experiment code and launchers

| ID | Main source files | Launcher | Default output |
| --- | --- | --- | --- |
| S6-E1 | `src/s6_e1_source_domain/run_source_domain_v10_test.py`; `run_old_matlab_equivalent.py`; `generate_linear_test_mat.py` | `scripts/run_s6_e1_source_domain.sh` | `results/s6_e1_source_domain` |
| S6-E2 | `src/s6_e2_target_full_recomputation/run_old_matlab_equivalent.py`; `generate_linear_test_mat.py`; `merge_source_target_mat.py`; `summarize_target_full_recomputation.py` | `scripts/run_s6_e2_target_full_recomputation.sh` | `results/s6_e2_target_full_recomputation` |
| S6-E2a | `src/s6_e2_target_response_density/run_target_observation_density_baseline.py` | `scripts/run_s6_e2_target_response_density.sh` | `results/s6_e2_target_response_density` |
| S6-E3 | `src/s6_e3_noise_robustness/run_noise_sweep_suite.py`; `run_old_matlab_equivalent.py`; `generate_linear_test_mat.py`; `summarize_noise_robustness.py`; `plot_noise_robustness_publication.py` | `scripts/run_s6_e3_noise_robustness.sh` | `results/s6_e3_noise_robustness` |
| S6-E4 | `src/s6_e4_source_target_similarity/compute_source_target_similarity.py`; `run_old_matlab_equivalent.py`; `generate_linear_test_mat.py`; `merge_source_target_mat.py` | `scripts/run_s6_e4_source_target_similarity.sh` | `results/s6_e4_source_target_similarity` |
| S6-E5 | Files under `src/s6_e5_similarity_weighted_topk/` | `scripts/run_s6_e5_from_scratch.sh`; `scripts/run_s6_e5_similarity_weighted_topk.sh` | `results/s6_e5_similarity_weighted_topk` |
| S6-E6 | `src/s6_e6_transfer_latency/summarize_transfer_method_latency.py` | `scripts/run_s6_e6_transfer_latency.sh` | `results/s6_e6_transfer_latency` |
| S6-E7 | Files under `src/s6_e7_delayed_response_reuse/` | `scripts/run_s6_e7_delayed_response_reuse.sh` | `results/s6_e7_delayed_response_reuse` |

## Experiment overview

| ID | Experiment | Paper location | Objective | Core configuration | Main output |
| --- | --- | --- | --- | --- | --- |
| S6-E0 | Data generation and cohort construction | Start of Section 6 | Construct the source and target synthetic cohorts used by the elderly-health transfer study | Ten source users and ten disjoint target users; ten versions and ten days per version; three modalities and continuous health status | Unified source/target multimodal datasets and the reference/evaluation split |
| S6-E1 | Source-domain held-out benchmark | Section 6.1; Table 1 | Compare original DSVI-O with history-dependent DSVI on source held-out days | Ten source users; reference days 1-90; held-out days 91-100; three health states | Class-wise accuracy, precision, recall, specificity, and F1 |
| S6-E2 | Target-domain full-recomputation baseline | Section 6.1; Table 2 | Establish a target-domain baseline that fully recomputes second-stage responses | Ten target users; reference/training days 1-90 and evaluation days 91-100 | Overall baseline accuracy `97.934%` and macro-F1 `0.982` |
| S6-E2a | Target response-density ablation | Section 6.1 companion baseline | Measure the effect of recomputing target responses less frequently while retaining five-second target observations | Response strides `120,240,720,1440`, corresponding to 10 minutes, 20 minutes, 1 hour, and 2 hours | Comparison with `Full (5 sec)` and four sparse-response settings |
| S6-E3 | Noise robustness | Section 6.2; Figure 4; Table 3 | Evaluate source-domain prediction under sensor perturbations | One representative source user; smartwatch and insole perturbations; EMR unchanged; seven noise families and strengths `0.5,1,1.5,2,5,10,20` | Clean accuracy `99.429%`; mixed-noise accuracy `89.685%` at `s=20` |
| S6-E4 | Source-target similarity | Section 6.3.1 | Compute historical similarity between source and target users | First 90 historical days; per-modality standardization; weights `gamma=(2,2,1)` | Pairwise distances, transfer kernels, and normalized source weights |
| S6-E5 | Similarity-weighted Top-K transfer | Section 6.3.2; Figures 5-6; Table 4 | Test whether weighted multi-source transfer outperforms single-source transfer | `K=1,...,10`; similarity and response references use days 1-90; evaluation uses days 91-100 | Mean accuracy increases from `91.657%` at `K=1` to `96.700%` at `K=8` |
| S6-E6 | Transfer-method latency | Section 6.3.2; Table 5 | Compare full recomputation, random-source, nearest-source, and weighted transfer | Weighted transfer uses `K=8`; latency is measured on the five-second update scale | Weighted transfer: `96.700%` accuracy and `0.725 s` batch runtime; full recomputation: `97.934%` and `70.325 s` |
| S6-E7 | Delayed response reuse | Section 6.4; Figures 7-8; Table 6 | Measure the accuracy-cost tradeoff when transferred responses are refreshed less often | Weighted transfer with `K=8`; refresh intervals `0,0.5,1,5,10,30,60,240,720` minutes | Accuracy from `96.700%` at 0 minutes to `67.285%` at 720 minutes |

After adopting the strict `v1`-`v9` reference/training split and `v10` evaluation split, results produced with older splits must be regenerated before they are used in the paper.

## Experiment dependencies

| Relationship | Description |
| --- | --- |
| S6-E0 -> all | Cohort construction supplies the common data foundation |
| S6-E1 -> S6-E2 | Source-domain validation precedes the target full-recomputation baseline |
| S6-E2 -> S6-E2a | The response-density ablation uses S6-E2 as its `Full (5 sec)` reference |
| S6-E2 -> S6-E5/S6-E6/S6-E7 | Full recomputation is the primary target-domain reference |
| S6-E3 | Independent source-domain sensor-noise robustness evaluation |
| S6-E4 -> S6-E5 | Similarity determines Top-K source selection and transfer weights |
| S6-E5 -> S6-E6 | The representative `K=8` setting enters the latency comparison |
| S6-E5 -> S6-E7 | The delayed-response experiment fixes `K=8` and varies the refresh interval |
