# Source and target cohorts

This directory contains the source and target cohorts used in the
elderly-health computational study.

- `source/`: users 1--10
- `target/`: users 11--20
- ten versions (`v1`--`v10`) per cohort
- ten days per version and user
- 17,280 observations per day at five-second resolution

Each version contains user profiles, electronic medical records, smartwatch
health data, and intelligent-insole data. The source and target cohorts are
disjoint at the level of user trajectories, health-state labels, and response
trajectories.

The large smartwatch and insole CSV files are stored with Git LFS. A clone
without automatic LFS download can retrieve them with:

```bash
git lfs pull
```

`SHA256SUMS` records checksums for all cohort files.
