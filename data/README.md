# DSVI-O Dataset

The dataset contains synthetic multimodal elderly-health records for users
1--10. The data are organized into ten versions (`v1`--`v10`). Each version
contains ten days per user, sampled every five seconds, with 17,280 time points
per day. The version directories represent separate ten-day data realizations
rather than one continuous calendar-time series; together they provide 100
day-level trajectories for each user.

The dataset is fully synthetic. It contains no observations from real patients
or identifiable individuals and is intended for methodological research on
dynamic stochastic systems and multimodal health-state modelling.

## Directory structure

Each version follows the same layout:

```text
data/vN/
|-- user_profiles.csv
|-- medical_records.csv
|-- electronic_medical_records/
|   `-- <user_id>_EMR.csv
|-- health_data/
|   `-- <user_id>_health_data.csv
`-- insole_data/
    `-- <user_id>_insole_data.csv
```

## Record types

- `user_profiles.csv` contains user identifiers and demographic and
  anthropometric attributes, including age, gender, height, weight, and BMI.
- `medical_records.csv` contains the cohort-level electronic medical records;
  `electronic_medical_records/` provides the corresponding per-user records.
- `health_data/` contains five-second smartwatch measurements together with
  activity and health-state fields.
- `insole_data/` contains five-second plantar-pressure, gait, balance, motion,
  and exercise-related measurements.

All tabular files are UTF-8 CSV files with a header row. Timestamps use the
format `YYYY-MM-DD HH:MM:SS`. Measurement units are included in the electronic
medical-record column names; the remaining variable names are given directly
in the CSV headers.

The large smartwatch and insole CSV files are stored with Git LFS. A clone
without automatic LFS download can retrieve them with:

```bash
git lfs pull
```
