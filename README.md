# DSVI-O
## Data Statistics
- Data volume: 200
- Individual types (20 for each type):
    - Always normal
    - Always abnormal
    - Always warning
    - Normal -> warning
    - Warning -> normal
    - Warning -> abnormal
    - Normal -> warning -> abnormal
    - Abnormal -> warning -> normal
    - Normal -> warning -> normal
    - Abnormal -> warning -> abnormal

## Data Description
### 1. user_profiles.csv
- **Content**: Basic user information
- **Fields**:
  - user_id: User ID
  - age: Age
  - gender: Gender (Male/Female)
  - height_cm: Height (cm)
  - weight_kg: Weight (kg)
  - BMI: Body Mass Index
- **Note**: Each row corresponds to the basic attributes of a user.

### 2. user_groups.csv
- **Content**: User grouping information
- **Fields**:
  - user_id: User ID
  - abnormal_group: Abnormal group (e.g., heart rate abnormal, SpO2 abnormal, normal, etc.)
  - individual_type: Individual type (e.g., always normal, always abnormal, normal to warning, etc.)
- **Note**: Used for subsequent group analysis and anomaly detection.

### 3. medical_records.csv
- **Content**: User electronic medical records
- **Fields**:
  - user_id: User ID
  - WBC, RBC, HGB, HCT, PLT, NEUT%, LYM%, GLU, CRP, ALT, AST, UA, CREA, BUN, TC, TG, HDL, LDL, SBP, DBP, TP, ALB, SpO2, Temp, BMI, etc.
- **Note**: Each row corresponds to a user's medical examination results, including blood routine, metabolism, blood lipids, blood pressure, nutrition, and other indicators.

### 4. daily_physiology.csv
- **Content**: User's daily physiological monitoring data
- **Fields**:
  - time: Timestamp (to the second)
  - HR: Heart rate
  - SpO2: Blood oxygen saturation
  - step: Steps per 5 seconds
  - step_cum: Cumulative steps
  - calories: Calorie consumption
  - activity: Activity type (e.g., sleep, active, leisure, etc.)
  - weather: Weather
  - status: Health status (0=normal, 1=warning, 2=abnormal)
- **Note**: Each row records the user's physiological and activity status at a certain moment.

### 5. insole_data.csv
- **Content**: User's daily plantar pressure and gait data
- **Fields**:
  - time: Timestamp (to the second)
  - LF, LH, RF, RH: Left front, left heel, right front, right heel plantar pressure
  - step_freq: Step frequency
  - gait_type: Gait type (e.g., normal, abnormal, fast walk, slow walk, rest)
  - center_x, center_y: Plantar pressure center coordinates
- **Note**: Each row records the user's plantar pressure and gait characteristics at a certain moment.

### 6. medical_records_abnormal.csv
- **Content**: User electronic medical records (including abnormal and warning states)
- **Fields**:
  - user_id: User ID
  - Other medical indicators (same as medical_records.csv)
  - status: Health status (0=normal, 1=warning, 2=abnormal)
- **Note**:
  - This file is an extension of medical_records.csv based on the individual_type and abnormal_group in user_groups.csv.
  - When the user is in a "warning" or "abnormal" state, relevant medical indicators are adjusted according to the abnormal type. The "warning" amplitude is half of the "abnormal" amplitude.
  - For example:
    - Heart rate abnormal group:
      - Warning: CRP +1, WBC +0.75, NEUT% +4, LYM% -4
      - Abnormal: CRP +2, WBC +1.5, NEUT% +8, LYM% -8
    - SpO2 abnormal group:
      - Warning: SpO2 -1, RBC +0.15, HCT +0.015
      - Abnormal: SpO2 -2, RBC +0.3, HCT +0.03
  - Each user may have multiple records, corresponding to different health states.
