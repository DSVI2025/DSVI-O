# DSVI-O Source-Domain Dataset

This repository provides the source-domain synthetic elderly-health dataset
associated with **“Differential Stochastic Variational Inequalities with
Parametric Optimization.”**
[[arXiv](https://arxiv.org/abs/2508.15241)]
[[PDF](https://arxiv.org/pdf/2508.15241)]

The dataset represents ten synthetic individuals through multimodal records
from smartwatch sensors, intelligent insoles, electronic medical records, and
individual profiles. It contains no observations from real patients or
identifiable individuals.

The complete source-domain cohort is stored under [`data/source/`](data/source/).
A description of the cohort, directory structure, temporal resolution, and
record types is provided in [`data/README.md`](data/README.md).

Large smartwatch and insole CSV files are stored with Git LFS. After cloning,
retrieve the complete dataset with:

```bash
git lfs pull
```

This repository contains the original source-domain data and its documentation
only. It does not include target-domain data, experimental implementations, or
data-generation scripts.
