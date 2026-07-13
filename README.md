# DSVI-O Dataset

This repository provides the synthetic elderly-health dataset associated with
**“Differential Stochastic Variational Inequalities with Parametric
Optimization.”**
[[arXiv](https://arxiv.org/abs/2508.15241)]
[[PDF](https://arxiv.org/pdf/2508.15241)]

The dataset represents ten synthetic individuals through multimodal records
from smartwatch sensors, intelligent insoles, electronic medical records, and
individual profiles. It contains no observations from real patients or
identifiable individuals.

The complete dataset is stored under [`data/`](data/). A description of the
cohort, directory structure, temporal resolution, and record types is provided
in [`data/README.md`](data/README.md).

Large smartwatch and insole CSV files are stored with Git LFS. After cloning,
retrieve the complete dataset with:

```bash
git lfs pull
```

This repository contains the original data and its documentation only. It does
not include experimental implementations or data-generation scripts.

## Citation

If you use this dataset, please cite the dataset and the associated paper using
the metadata in [`CITATION.cff`](CITATION.cff). The preferred paper citation is:

> X. Chen, J. Guo, and G. Wang, “Differential Stochastic Variational
> Inequalities with Parametric Optimization,” 2025,
> [https://doi.org/10.48550/arXiv.2508.15241](https://doi.org/10.48550/arXiv.2508.15241).

## License

The dataset and accompanying documentation are licensed under the
[Creative Commons Attribution 4.0 International License](LICENSE).
