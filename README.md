
## Main Data Files and Field Descriptions

1. The current dataset includes 10 versions of data, with each version containing 10 users and 10 days of data per user.
2. The same 10 users are consistent across different versions.
3. Each user generates 10 days of data, with status changes occurring randomly. Each status lasts no more than 12 hours. Available status patterns:

```python
# Pattern Types
CHANGE_PATTERNS = {
    'Always Healthy': {'from': 0, 'to': 0, 'type': 'linear'},
    'Always Sick': {'from': 2, 'to': 2, 'type': 'linear'},
    'Sudden Warning': {'from': 0, 'to': 1, 'type': 'sudden'},
    'Linear Gradual Warning': {'from': 0, 'to': 1, 'type': 'linear'},
    'Other Gradual Warning': {'from': 0, 'to': 1, 'type': 'exponential'},
    'Sudden Recovery': {'from': 1, 'to': 0, 'type': 'sudden'},
    'Linear Gradual Recovery': {'from': 1, 'to': 0, 'type': 'linear'},
    'Other Gradual Recovery': {'from': 1, 'to': 0, 'type': 'exponential'},
    'Sudden Warning to Sick': {'from': 1, 'to': 2, 'type': 'sudden'},
    'Sudden Sick to Warning': {'from': 2, 'to': 1, 'type': 'sudden'},
}
```


  
```


### 1. user_profiles.csv (User Basic Information)
- user_id: Unique user identifier
- age: Age
- gender: Gender (Male/Female)
- height_cm: Height (centimeters)
- weight_kg: Weight (kilograms)
- BMI: Body Mass Index

### 2. medical_records.csv (Electronic Medical Records)
- PatientID: Unique user identifier
- Age: Age
- Gender: Gender (Male/Female)
- WBC(10^9/L): White Blood Cell count
- RBC(10^12/L): Red Blood Cell count
- HGB(g/L): Hemoglobin
- HCT(%): Hematocrit
- MCV(fL): Mean Corpuscular Volume
- MCH(pg): Mean Corpuscular Hemoglobin
- MCHC(g/L): Mean Corpuscular Hemoglobin Concentration
- PLT(10^9/L): Platelet count
- LYM%(%): Lymphocyte percentage
- NEUT%(%): Neutrophil percentage
- GLU(mmol/L): Glucose
- ALT(U/L): Alanine Aminotransferase
- AST(U/L): Aspartate Aminotransferase
- TP(g/L): Total Protein
- ALB(g/L): Albumin
- BUN(mmol/L): Blood Urea Nitrogen
- CREA(μmol/L): Creatinine
- UA(μmol/L): Uric Acid
- TC(mmol/L): Total Cholesterol
- TG(mmol/L): Triglycerides
- HDL(mmol/L): High-Density Lipoprotein
- LDL(mmol/L): Low-Density Lipoprotein
- CRP(mg/L): C-Reactive Protein
- SBP(mmHg): Systolic Blood Pressure
- DBP(mmHg): Diastolic Blood Pressure
- BMI(kg/m^2): Body Mass Index
- SpO2(%): Blood Oxygen Saturation
- Temp(℃): Body Temperature

### 4. Health Data (health_data.csv)
- time: Timestamp (format: YYYY-MM-DD HH:MM:SS)
- heart_rate: Heart rate (beats/minute)
- spo2: Blood oxygen saturation (%)
- sleep_quality: Sleep quality (0: awake, 1: light sleep, 2: deep sleep)
- steps: Cumulative step count (steps)
- calories: Cumulative calories burned (kcal)
- skin_temperature: Skin temperature (℃)
- hrv: Heart rate variability (ms)
- stress_index: Stress index (dimensionless)
- sbp: Systolic blood pressure (mmHg)
- dbp: Diastolic blood pressure (mmHg)
- activity_intensity: Activity intensity (dimensionless)
- active_minutes: Cumulative active minutes (minutes)
- floors_climbed: Cumulative floors climbed (floors)
- activity: Current activity type (such as sleep, activity, leisure, etc.)
- status: Health status (0: normal, 1: abnormal, 2: sick, or continuous values)

### 5. Electronic Insole Data (insole_data.csv)
- time: Timestamp (format: YYYY-MM-DD HH:MM:SS)
- LF: Left forefoot pressure (kg)
- LH: Left hindfoot pressure (kg)
- RF: Right forefoot pressure (kg)
- RH: Right hindfoot pressure (kg)
- step_freq: Step frequency (steps/minute)
- stride_length: Stride length (meters)
- contact_time: Ground contact time (milliseconds)
- symmetry: Gait symmetry (%)
- gait_type: Gait type (such as "normal", "abnormal", etc.)
- motion_type: Motion type (such as "stationary", "walking", etc.)
- force_estimate: Plantar pressure estimate (N)
- balance_score: Balance score (0-100)
- lr_ratio: Left-right foot pressure ratio (0-1)
- center_x: Lateral center of pressure offset (mm)
- center_y: Longitudinal center of pressure offset (mm)  