import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class UserGenerator:

    HEALTH_STATUS = {
        'Normal': 0,
        'Warning': 1,
        'Critical Illness': 2
    }

    CHANGE_PATTERNS = {
        'Sudden Warning': {'from': 0, 'to': 2, 'type': 'sudden'},
        'Linear Gradual Warning': {'from': 0, 'to': 1, 'type': 'linear'},
        'Exponential Gradual Warning': {'from': 0, 'to': 1, 'type': 'exponential'},
        'Sudden Recovery': {'from': 1, 'to': 0, 'type': 'sudden'},
        'Linear Gradual Recovery': {'from': 1, 'to': 0, 'type': 'linear'},
        'Exponential Gradual Recovery': {'from': 1, 'to': 0, 'type': 'exponential'},
        'Sudden Warning to Illness': {'from': 1, 'to': 2, 'type': 'sudden'},
        'Sudden Illness to Warning': {'from': 2, 'to': 1, 'type': 'sudden'},
    }

    ABNORMAL_RULES = {
        'Normal': {},
        'Heart Rate Abnormality': {'CRP': 2, 'WBC': 1.5, 'NEUT%': 8, 'LYM%': -8},
        'Oxygen Saturation Abnormality': {'SpO2': -2, 'RBC': 0.3, 'HCT': 0.03},
        'Sleep Abnormality': {'SBP': 10, 'GLU': 0.7, 'CRP': 1.5},
        'Step Count Abnormality': {'HGB': -10, 'TC': -0.5, 'TG': -0.2, 'GLU': -0.5},
        'Calorie Abnormality': {'GLU': -0.5, 'ALT': 10, 'UA': 50, 'CREA': 10, 'SBP': 8},
    }

    INDICATOR_CONFIG = {

        'WBC': {'mu': 7.0, 'sigma': 1.0, 'clip': (4.0, 10.0)},
        'RBC': {
            'mu_male': 5.0, 'mu_female': 4.4, 'sigma': 0.25,
            'clip_male': (4.3, 5.8),
            'clip_female': (3.8, 5.1)
        },
        'HGB': {
            'mu_male': 150, 'mu_female': 130, 'sigma': 7,
            'clip_male': (130, 175),
            'clip_female': (115, 150)
        },
        'HCT': {
            'mu_male': 0.45, 'mu_female': 0.40, 'sigma': 0.025,
            'clip_male': (0.40, 0.50),
            'clip_female': (0.35, 0.45)
        },
        'PLT': {'mu': 200, 'sigma': 33, 'clip': (100, 300)},
        'NEUT%': {'mu': 60, 'sigma': 3.3, 'clip': (50, 70)},
        'LYM%': {'mu': 30, 'sigma': 3.3, 'clip': (20, 40)},

        'GLU': {'mu': 5.2, 'sigma': 0.37, 'clip': (3.9, 6.1)},
        'CRP': {'mu': 2, 'sigma': 1.3, 'clip': (0, 8)},
        'ALT': {'mu': 25, 'sigma': 6.8, 'clip': (9, 50)},
        'AST': {'mu': 27, 'sigma': 4.1, 'clip': (15, 40)},
        'UA': {
            'mu_male': 315, 'mu_female': 250, 'sigma': 35,
            'clip_male': (210, 420),
            'clip_female': (150, 350)
        },
        'CREA': {
            'mu_male': 88, 'mu_female': 75, 'sigma': 10,
            'clip_male': (62, 115),
            'clip_female': (53, 97)
        },
        'BUN': {'mu': 5.5, 'sigma': 0.88, 'clip': (2.9, 8.2)},
        'TC': {'mu': 4.4, 'sigma': 0.43, 'clip': (3.1, 5.7)},
        'TG': {'mu': 1.05, 'sigma': 0.22, 'clip': (0.4, 1.7)},
        'HDL': {'mu': 1.45, 'sigma': 0.18, 'clip': (0.9, 2.0)},
        'LDL': {'mu': 2.6, 'sigma': 0.3, 'clip': (1.7, 3.5)},
        'SBP': {'mu': 120, 'sigma': 8.3, 'clip': (90, 140)},
        'DBP': {'mu': 75, 'sigma': 5, 'clip': (60, 90)},
        'TP': {'mu': 70, 'sigma': 3.3, 'clip': (60, 80)},
        'ALB': {'mu': 45, 'sigma': 3.3, 'clip': (35, 55)},
        'SpO2': {'mu': 98, 'sigma': 1, 'clip': (90, 100)},
        'Temp': {'mu': 36.6, 'sigma': 0.2, 'clip': (36.0, 37.2)}
    }

    WEARABLE_CONFIG = {
        'heart_rate': {'mu': 75, 'sigma': 10, 'clip': (50, 120)},
        'sleep_hours': {'mu': 7.5, 'sigma': 1.0, 'clip': (4, 12)},
        'sleep_quality': {'mu': 80, 'sigma': 15, 'clip': (20, 100)},
        'blood_oxygen': {'mu': 98, 'sigma': 1, 'clip': (90, 100)},
        'skin_temp': {'mu': 32.5, 'sigma': 1.5, 'clip': (28, 37)},
        'hrv': {'mu': 35, 'sigma': 10, 'clip': (10, 80)},
        'stress_index': {'mu': 40, 'sigma': 20, 'clip': (0, 100)},
        'blood_pressure_sys': {'mu': 120, 'sigma': 15, 'clip': (90, 180)},
        'blood_pressure_dia': {'mu': 80, 'sigma': 10, 'clip': (60, 120)},
        'sedentary_alerts': {'mu': 8, 'sigma': 3, 'clip': (0, 20)},
        'body_movement': {'mu': 50, 'sigma': 20, 'clip': (0, 200)},
        'activity_intensity': {'mu': 30, 'sigma': 15, 'clip': (0, 100)},
        'calories_burned': {'mu': 2000, 'sigma': 400, 'clip': (800, 4000)},
        'active_minutes': {'mu': 60, 'sigma': 30, 'clip': (0, 300)},
        'floors_climbed': {'mu': 5, 'sigma': 3, 'clip': (0, 50)}
    }

    INSOLE_CONFIG = {
        'step_frequency': {'mu': 90, 'sigma': 10, 'clip': (60, 120)},
        'stride_length': {'mu': 65, 'sigma': 8, 'clip': (40, 90)},
        'ground_contact_time': {'mu': 250, 'sigma': 30, 'clip': (180, 350)},
        'left_right_symmetry': {'mu': 95, 'sigma': 5, 'clip': (70, 100)},
        'plantar_pressure': {'mu': 80, 'sigma': 15, 'clip': (30, 150)},
        'foot_contact_time': {'mu': 200, 'sigma': 25, 'clip': (150, 300)},
        'posture_score': {'mu': 85, 'sigma': 10, 'clip': (50, 100)},
        'activity_type': {'walking': 0.4, 'running': 0.2, 'standing': 0.3, 'sitting': 0.1},
        'force_estimation': {'mu': 600, 'sigma': 100, 'clip': (300, 1200)},
        'balance_status': {'mu': 80, 'sigma': 15, 'clip': (30, 100)},
        'weight_distribution': {'mu': 50, 'sigma': 5, 'clip': (35, 65)}
    }

    def __init__(self, seed=42):
        self.seed = seed
        self.master_rng = np.random.default_rng(self.seed)

    def generate_user_profile(self, n):
        rng = self.master_rng
        age_bins = [(60, 69), (70, 79), (80, 100)]
        age_probs = [0.5, 0.3, 0.2]
        age_bin_choices = rng.choice(len(age_bins), size=n, p=age_probs)
        ages = [rng.integers(age_bins[i][0], age_bins[i][1] + 1) for i in age_bin_choices]
        genders = []
        for age in ages:
            if age < 80:
                gender = rng.choice(['Male', 'Female'], p=[0.5, 0.5])
            else:
                gender = rng.choice(['Male', 'Female'], p=[0.4, 0.6])
            genders.append(gender)
        heights = []
        for age, gender in zip(ages, genders):
            if gender == 'Male':
                mean_height = 170 - (age - 60) * 0.1
                std_height = 5
            else:
                mean_height = 160 - (age - 60) * 0.1
                std_height = 5
            height = rng.normal(mean_height, std_height)
            heights.append(round(np.clip(height, 145, 190), 1))
        bmis = rng.normal(loc=22, scale=2.5, size=n)
        bmis = np.clip(bmis, 18.5, 30)
        weights = [round(bmi * (h/100)**2, 1) for bmi, h in zip(bmis, heights)]
        df = pd.DataFrame({
            'age': ages,
            'gender': genders,
            'height_cm': heights,
            'weight_kg': weights,
            'BMI': [round(bmi, 1) for bmi in bmis]
        })
        return df

    def generate_single_medical_record(self, gender, age, bmi, user_id,seed):
        cfg = self.INDICATOR_CONFIG

        rng = np.random.default_rng(seed)

        WBC = rng.normal(cfg['WBC']['mu'], cfg['WBC']['sigma'])
        if gender == 'Male':
            RBC = rng.normal(cfg['RBC']['mu_male'], cfg['RBC']['sigma'])
            HGB = rng.normal(cfg['HGB']['mu_male'], cfg['HGB']['sigma'])
            HCT = rng.normal(cfg['HCT']['mu_male'], cfg['HCT']['sigma'])
            UA = rng.normal(cfg['UA']['mu_male'], cfg['UA']['sigma'])
            CREA = rng.normal(cfg['CREA']['mu_male'], cfg['CREA']['sigma'])
        else:
            RBC = rng.normal(cfg['RBC']['mu_female'], cfg['RBC']['sigma'])
            HGB = rng.normal(cfg['HGB']['mu_female'], cfg['HGB']['sigma'])
            HCT = rng.normal(cfg['HCT']['mu_female'], cfg['HCT']['sigma'])
            UA = rng.normal(cfg['UA']['mu_female'], cfg['UA']['sigma'])
            CREA = rng.normal(cfg['CREA']['mu_female'], cfg['CREA']['sigma'])
        PLT = rng.normal(cfg['PLT']['mu'], cfg['PLT']['sigma'])
        NEUT = rng.normal(cfg['NEUT%']['mu'], cfg['NEUT%']['sigma'])
        LYM = rng.normal(cfg['LYM%']['mu'], cfg['LYM%']['sigma'])

        glu_mu = cfg['GLU']['mu'] + 0.02*(age-60) + 0.03*(bmi-22)
        GLU = rng.normal(glu_mu, cfg['GLU']['sigma'])
        CRP = rng.normal(cfg['CRP']['mu'], cfg['CRP']['sigma'])
        ALT = rng.normal(cfg['ALT']['mu'], cfg['ALT']['sigma'])
        AST = rng.normal(cfg['AST']['mu'], cfg['AST']['sigma'])
        BUN = rng.normal(cfg['BUN']['mu'], cfg['BUN']['sigma'])

        tc_mu = cfg['TC']['mu'] + 0.01*(age-60)
        tg_mu = cfg['TG']['mu'] + 0.01*(age-60)
        TC = rng.normal(tc_mu, cfg['TC']['sigma'])
        TG = rng.normal(tg_mu, cfg['TG']['sigma'])
        HDL = rng.normal(cfg['HDL']['mu'], cfg['HDL']['sigma'])
        LDL = rng.normal(cfg['LDL']['mu'], cfg['LDL']['sigma'])

        sbp_mu = cfg['SBP']['mu'] + 0.2*(age-60) + 0.5*(bmi-22)
        dbp_mu = cfg['DBP']['mu'] + 0.1*(age-60) + 0.2*(bmi-22)
        SBP = rng.normal(sbp_mu, cfg['SBP']['sigma'])
        DBP = rng.normal(dbp_mu, cfg['DBP']['sigma'])

        TP = rng.normal(cfg['TP']['mu'], cfg['TP']['sigma'])
        ALB = rng.normal(cfg['ALB']['mu'], cfg['ALB']['sigma'])

        SpO2 = rng.normal(cfg['SpO2']['mu'], cfg['SpO2']['sigma'])
        Temp = rng.normal(cfg['Temp']['mu'], cfg['Temp']['sigma'])
        BMI = bmi

        rec = {
            'WBC': WBC, 'RBC': RBC, 'HGB': HGB, 'HCT': HCT, 'UA': UA, 'CREA': CREA, 'PLT': PLT,
            'NEUT%': NEUT, 'LYM%': LYM, 'GLU': GLU, 'CRP': CRP, 'ALT': ALT, 'AST': AST, 'BUN': BUN,
            'TC': TC, 'TG': TG, 'HDL': HDL, 'LDL': LDL, 'SBP': SBP, 'DBP': DBP, 'TP': TP, 'ALB': ALB,
            'SpO2': SpO2, 'Temp': Temp, 'BMI': BMI
        }
        for k, v in rec.items():
            if k in cfg:

                if gender == 'Male' and f"clip_male" in cfg[k]:
                    low, high = cfg[k]["clip_male"]
                elif gender == 'Female' and f"clip_female" in cfg[k]:
                    low, high = cfg[k]["clip_female"]
                else:
                    low, high = cfg[k]["clip"]

                rec[k] = np.clip(rec[k], low, high)

        rec["SpO2"] = rng.choice([98, 99])

        HCT_percent = rec['HCT'] * 100

        MCV = HCT_percent * 10 / rec['RBC'] if rec['RBC'] > 0 else 0
        MCH = rec['HGB'] * 10 / rec['RBC'] if rec['RBC'] > 0 else 0
        MCHC = rec['HGB'] * 100 / HCT_percent if HCT_percent > 0 else 0

        result = {
            'PatientID': user_id,
            'Age': age,
            'Gender': gender,
            'WBC(10^9/L)': round(rec['WBC'], 1),
            'RBC(10^12/L)': round(rec['RBC'], 1),
            'HGB(g/L)': round(rec['HGB']),
            'HCT(%)': round(HCT_percent, 1),
            'MCV(fL)': round(MCV, 1),
            'MCH(pg)': round(MCH, 1),
            'MCHC(g/L)': round(MCHC, 1),
            'PLT(10^9/L)': round(rec['PLT']),
            'LYM%(%)': round(rec['LYM%'], 1),
            'NEUT%(%)': round(rec['NEUT%'], 1),
            'GLU(mmol/L)': round(rec['GLU'], 1),
            'ALT(U/L)': round(rec['ALT']),
            'AST(U/L)': round(rec['AST']),
            'TP(g/L)': round(rec['TP'], 1),
            'ALB(g/L)': round(rec['ALB'], 1),
            'BUN(mmol/L)': round(rec['BUN'], 1),
            'CREA(μmol/L)': round(rec['CREA']),
            'UA(μmol/L)': round(rec['UA']),
            'TC(mmol/L)': round(rec['TC'], 1),
            'TG(mmol/L)': round(rec['TG'], 1),
            'HDL(mmol/L)': round(rec['HDL'], 1),
            'LDL(mmol/L)': round(rec['LDL'], 1),
            'CRP(mg/L)': round(rec['CRP'], 1),
            'SBP(mmHg)': round(rec['SBP']),
            'DBP(mmHg)': round(rec['DBP']),
            'BMI(kg/m^2)': round(rec['BMI'], 1),
            'SpO2(%)': round(rec['SpO2']),
            'Temp(℃)': round(rec['Temp'], 1)
        }
        return result

    def generate_medical_record(self, user_df):
        records = []
        user_seeds = self.master_rng.integers(0, 1e9, size=len(user_df))
        for idx, (row, seed) in enumerate(zip(user_df.itertuples(), user_seeds)):
            rec = self.generate_single_medical_record(row.gender, row.age, row.BMI, idx+1, seed)
            records.append(rec)
        df = pd.DataFrame(records)
        return df

    def apply_change_pattern(self, from_state, to_state, pattern_type, duration_minutes=1440):
        """
        Generate continuous health-state labels from a change pattern
        duration_minutes: Number of minutes in one day (1440)
        Returns one continuous health-state value in [0, 2] per minute.
        """
        if pattern_type == 'sudden':

            change_point = self.master_rng.integers(100, duration_minutes - 100)
            states = np.full(duration_minutes, from_state, dtype=float)
            states[change_point:] = to_state
            return states

        elif pattern_type == 'linear':

            change_start = self.master_rng.integers(100, duration_minutes // 2)
            change_duration = self.master_rng.integers(60, 360)
            change_end = min(change_start + change_duration, duration_minutes - 100)

            states = np.full(duration_minutes, from_state, dtype=float)

            for i in range(change_start, change_end):
                progress = (i - change_start) / (change_end - change_start)
                states[i] = from_state + (to_state - from_state) * progress
            states[change_end:] = to_state
            return states

        elif pattern_type == 'exponential':

            change_start = self.master_rng.integers(100, duration_minutes // 2)
            change_duration = self.master_rng.integers(60, 360)
            change_end = min(change_start + change_duration, duration_minutes - 100)

            states = np.full(duration_minutes, from_state, dtype=float)
            for i in range(change_start, change_end):
                progress = (i - change_start) / (change_end - change_start)

                sigmoid_progress = 1 / (1 + np.exp(-6 * (progress - 0.5)))
                states[i] = from_state + (to_state - from_state) * sigmoid_progress
            states[change_end:] = to_state
            return states

        else:

            return np.full(duration_minutes, from_state, dtype=float)

    def generate_wearable_data(self, gender, age, health_states, abnormal_group, user_id, seed):
        """Generate wearable-device data"""
        rng = np.random.default_rng(seed)
        cfg = self.WEARABLE_CONFIG
        n_points = len(health_states)

        data = {}
        for metric, config in cfg.items():
            if metric == 'activity_type':
                continue

            base_values = rng.normal(config['mu'], config['sigma'], n_points)

            for i, state in enumerate(health_states):
                if state >= 1.5:
                    if metric in ['heart_rate', 'stress_index']:
                        base_values[i] *= 1.2
                    elif metric in ['sleep_quality', 'hrv']:
                        base_values[i] *= 0.8
                elif state >= 0.5:
                    if metric in ['heart_rate', 'stress_index']:
                        base_values[i] *= 1.1
                    elif metric in ['sleep_quality']:
                        base_values[i] *= 0.9

            data[metric] = np.clip(base_values, config['clip'][0], config['clip'][1])

        activity_probs = list(cfg['activity_type'].values())
        activity_types = list(cfg['activity_type'].keys())
        data['activity_type'] = rng.choice(activity_types, size=n_points, p=activity_probs)

        return data

    def generate_insole_data(self, gender, age, health_states, abnormal_group, user_id, seed):
        """Generate intelligent-insole data"""
        rng = np.random.default_rng(seed)
        cfg = self.INSOLE_CONFIG
        n_points = len(health_states)

        data = {}
        for metric, config in cfg.items():
            if metric == 'activity_type':
                continue

            base_values = rng.normal(config['mu'], config['sigma'], n_points)

            for i, state in enumerate(health_states):
                if state >= 1.5:
                    if metric in ['left_right_symmetry', 'balance_status', 'posture_score']:
                        base_values[i] *= 0.85
                    elif metric in ['ground_contact_time', 'foot_contact_time']:
                        base_values[i] *= 1.1
                elif state >= 0.5:
                    if metric in ['left_right_symmetry', 'balance_status']:
                        base_values[i] *= 0.95

            data[metric] = np.clip(base_values, config['clip'][0], config['clip'][1])

        return data

    def generate_daily_health_data(self, user_df, individual_types, abnormal_labels):
        """
        Generate one day of health-monitoring data for each user
        Includes time-series health-state labels and monitoring-device data
        """
        all_data = []
        user_seeds = self.master_rng.integers(0, 1e9, size=len(user_df))

        for idx, (row, individual_type, abnormal_group, seed) in enumerate(zip(
            user_df.itertuples(), individual_types, abnormal_labels, user_seeds)):

            rng = np.random.default_rng(seed)

            if individual_type == 'Always Healthy for One Day':
                health_states = np.zeros(1440, dtype=float)
            elif individual_type == 'Always Ill for One Day':
                health_states = np.full(1440, 2.0, dtype=float)
            elif individual_type in self.CHANGE_PATTERNS:
                pattern = self.CHANGE_PATTERNS[individual_type]
                health_states = self.apply_change_pattern(
                    pattern['from'], pattern['to'], pattern['type']
                )
            else:

                health_states = np.zeros(1440, dtype=float)

            wearable_data = self.generate_wearable_data(
                row.gender, row.age, health_states, abnormal_group, idx+1, seed
            )
            insole_data = self.generate_insole_data(
                row.gender, row.age, health_states, abnormal_group, idx+1, seed
            )

            user_data = {
                'user_id': idx + 1,
                'age': row.age,
                'gender': row.gender,
                'individual_type': individual_type,
                'abnormal_group': abnormal_group,
                'health_states': health_states,
                **wearable_data,
                **insole_data
            }

            all_data.append(user_data)

        return all_data

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--nums", default=200, type=int, help="num of data")
    parser.add_argument("--seed", default=53, type=int, help="seed")
    parser.add_argument("--workdir", default = "./data", help = "work directory")
    parser.add_argument("--version", default="v1", help="version of data")

    args = parser.parse_args()
    n_users = args.nums
    workdir = args.workdir
    version = args.version
    print(args)
    generator = UserGenerator(seed=args.seed)
    user_df = generator.generate_user_profile(n_users)
    user_df = user_df.copy()
    user_df['user_id'] = np.arange(1, len(user_df)+1)
    file_path = f'{workdir}/{version}/generated_data'
    os.makedirs(file_path, exist_ok=True)

    user_df[['user_id', 'age', 'gender', 'height_cm', 'weight_kg', 'BMI']].to_csv(os.path.join(file_path,'user_profiles.csv'), index=False)

    os.makedirs(file_path, exist_ok=True)

    med_df = generator.generate_medical_record(user_df)

    os.makedirs(os.path.join(file_path, 'electronic_medical_records'), exist_ok=True)
    for row in med_df.itertuples():
        patient_id = row.PatientID
        patient_df = med_df[med_df['PatientID'] == patient_id]
        patient_df.to_csv(os.path.join(file_path,f'electronic_medical_records/{patient_id}_EMR.csv'), index=False)
    med_df.to_csv(os.path.join(file_path,'medical_records.csv'), index=False)
    print(user_df.head())
    print(med_df.head())
