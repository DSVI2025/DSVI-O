#!/usr/bin/env python3
"""
Health data generator entry point
Uses the refactored modular design
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from health_gen.health_patterns import HealthPatternGenerator
from health_gen.activity import ActivityCalculator, circadian_wave
from health_gen.physiological_indicators import PhysiologicalCalculator
from health_gen.insole_data import InsoleDataGenerator
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count
from functools import partial

from health_gen import plot_health_states
from health_gen import config

class HealthDataGenerator:
    """Main health data generator class"""

    def __init__(self, user_profile, medical_record, days=1, start_date='2025-01-01', seed=42, weather='Sunny'):
        self.user_profile = user_profile
        self.medical_record = medical_record
        self.start_date = start_date
        self.days = days
        self.seed = seed
        self.weather = weather
        self.rng = np.random.default_rng(seed)

        self.age = user_profile['age']
        self.gender = user_profile['gender']
        self.height = user_profile['height_cm']
        self.weight = user_profile['weight_kg']
        self.bmi = user_profile['BMI']

        self.spo2_base = medical_record['SpO2(%)']
        self.sbp_base = medical_record['SBP(mmHg)']
        self.dbp_base = medical_record['DBP(mmHg)']
        self.hgb_base = medical_record['HGB(g/L)']

        # Calculate baseline heart rate
        self.hr_base = self._calculate_base_heart_rate()

        # Calculate basal metabolic rate
        self.bmr = self._calc_bmr()

        self.pattern_generator = HealthPatternGenerator(self.rng)
        self.activity_calculator = ActivityCalculator(self.rng, self.age)
        self.physiological_calculator = PhysiologicalCalculator(
            self.rng, self.age, self.sbp_base, self.dbp_base
        )
        self.insole_generator = InsoleDataGenerator(self.rng, self.age, self.weight)

        total_intervals = self.days * config.INTERVALS_PER_DAY
        self.health_states = self.pattern_generator.generate_multi_day_health_states(total_intervals)
        self._health_states = self.health_states['health_states']

    def _calculate_base_heart_rate(self):
        """Calculate baseline heart rate"""
        return (70 - 0.2 * (self.age - 60) +
                (2 if self.gender == 'Male' else 0) +
                0.05 * (self.sbp_base - 120) +
                0.01 * (self.hgb_base - 140))

    def _calc_bmr(self):
        """Calculate basal metabolic rate (BMR)"""
        s = 5 if self.gender == 'Male' else -161
        return 10 * self.weight + 6.25 * self.height - 5 * self.age + s

    def generate_data(self):
        """Generate health data"""
        all_rows = []

        weather_hr_delta, weather_spo2_delta = self.activity_calculator.calculate_weather_effects(self.weather)

        current_interval = 0
        for day in range(self.days):
            current_date = (datetime.strptime(self.start_date, "%Y-%m-%d") +
                          timedelta(days=day)).strftime("%Y-%m-%d")

            day_health_states = self._health_states[current_interval:current_interval + config.INTERVALS_PER_DAY]

            daily_rows = self._generate_day_data(current_date, day_health_states, weather_hr_delta, weather_spo2_delta)
            all_rows.extend(daily_rows)

            current_interval += config.INTERVALS_PER_DAY

        return pd.DataFrame(all_rows)

    def _generate_day_data(self, date, day_health_states, weather_hr_delta, weather_spo2_delta):
        """Generate one day of data"""
        schedule = self.activity_calculator.generate_daily_schedule()

        health_states = day_health_states
        current_abnormal_group = self.pattern_generator.select_abnormal_group_for_states(health_states)

        rows = []
        step_cum = 0
        active_minutes_cum = 0
        floors_cum = 0
        max_status = float(np.max(health_states)) if len(health_states) else 0.0
        daily_step_limit, daily_active_minutes_limit = self.activity_calculator.generate_daily_activity_limits(
            self.weather, max_status, current_abnormal_group
        )

        for t in range(0, 86400, 5):
            activity = self.activity_calculator.get_activity_state(t, schedule)

            interval_index = min(int(t // config.TIME_GRANULARITY_SECONDS), len(health_states) - 1)
            status_value = float(health_states[interval_index])
            rounded_status = round(status_value, 3)

            hr = (self.hr_base +
                  circadian_wave(t) +
                  self.activity_calculator.calculate_activity_hr_delta(activity) +
                  weather_hr_delta +
                  self.rng.normal(0, 1))

            spo2 = (self.spo2_base +
                   self.activity_calculator.calculate_activity_spo2_delta(activity) +
                   weather_spo2_delta +
                   self.rng.normal(0, 0.1))

            row_abnormal_group = current_abnormal_group if rounded_status > 0.5 else 'Normal'
            if row_abnormal_group != 'Normal':
                hr, spo2 = self.physiological_calculator.apply_abnormal_effects(
                    row_abnormal_group, status_value, hr, spo2
                )

            hr = np.clip(hr, 55, 120)
            spo2 = np.clip(spo2, 85, 100)

            step = self.activity_calculator.get_step_increment(activity, self.weather)
            step = self.activity_calculator.apply_step_budget(step, step_cum, daily_step_limit)
            step_cum += step

            if activity == 'Sleeping':

                calories = self.bmr/17280 * 0.9
            else:
                calories = (self.bmr/17280 +
                           self.activity_calculator.calculate_activity_calorie_delta(activity) +
                           (0.1 if self.weather in ['Stormy', 'Snowy'] else 0))

            dt = datetime.strptime(date, "%Y-%m-%d") + timedelta(seconds=t)
            time_str = dt.strftime("%Y-%m-%d %H:%M:%S")

            sleep_quality = self.physiological_calculator.calculate_sleep_quality(
                activity, status_value
            )

            skin_temp = self.physiological_calculator.calculate_skin_temp(t, activity, status_value)
            hrv = self.physiological_calculator.calculate_hrv(activity, status_value)
            stress = self.physiological_calculator.calculate_stress(activity, status_value)
            sbp_val, dbp_val = self.physiological_calculator.calculate_blood_pressure(t, activity, status_value)
            activity_intensity = self.activity_calculator.calculate_activity_intensity(activity)

            if activity == 'Active' and activity_intensity >= 30 and active_minutes_cum < daily_active_minutes_limit:
                active_minutes_cum = min(daily_active_minutes_limit, active_minutes_cum + 5/60)

            if activity == 'Active' and self.rng.random() < 0.002:
                floors_cum += 1

            rows.append({
                'time': time_str,
                'heart_rate': round(hr, 1),
                'spo2': round(spo2, 1),
                'sleep_quality': sleep_quality,
                'steps': step_cum,
                'calories': round(calories, 4),
                'skin_temperature': skin_temp,
                'hrv': hrv,
                'stress_index': stress,
                'sbp': sbp_val,
                'dbp': dbp_val,
                'activity_intensity': activity_intensity,
                'active_minutes': round(active_minutes_cum, 1),
                'floors_climbed': floors_cum,
                'activity': activity,
                'status': rounded_status,
                'abnormal_group': row_abnormal_group,
            })

        return rows

    def generate_insole_data(self, daily_df):
        """Generate intelligent-insole data"""
        return self.insole_generator.generate_insole_data(daily_df)

def process_user_data(user_data, workdir, version, base_seed, plot=False):
    """Generate data for one user in a worker process"""
    user_id, user_profile, medical_record = user_data

    print(f"Processing user {user_id}")

    generator = HealthDataGenerator(
        user_profile,
        medical_record,
        days=10,
        start_date='2025-01-01',
        seed=user_id + base_seed,
        weather='Sunny'
    )

    if plot:
        # Plot the health-state distribution
        plot_health_states(generator.health_states)
    # Generate health data
    daily_data = generator.generate_data()

    # Generate intelligent-insole data
    insole_data = generator.generate_insole_data(daily_data)

    os.makedirs(f'{workdir}/{version}/generated_data/health_data', exist_ok=True)
    daily_data.to_csv(f'{workdir}/{version}/generated_data/health_data/{user_id}_health_data.csv', index=False)

    os.makedirs(f'{workdir}/{version}/generated_data/insole_data', exist_ok=True)
    insole_data.to_csv(f'{workdir}/{version}/generated_data/insole_data/{user_id}_insole_data.csv', index=False)

    return {
        'user_id': user_id,
        'daily_shape': daily_data.shape,
        'insole_shape': insole_data.shape,
        'health_states': generator.health_states,
        'status_range': (daily_data['status'].min(), daily_data['status'].max()),
        'abnormal_groups': sorted(daily_data['abnormal_group'].unique().tolist())
    }

def main():
    """Main function demonstrating the health data generator with multiprocessing"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=53, type=int, help="seed")
    parser.add_argument("--workdir", default="./data", help="work directory")
    parser.add_argument("--version", default="v0", help="version of data")
    parser.add_argument("--processes", default=None, type=int, help="number of processes (default: CPU count)")
    parser.add_argument("--plot", action='store_true', help="whether to plot health states")

    args = parser.parse_args()
    workdir = args.workdir
    version = args.version
    num_processes = args.processes or cpu_count()

    print(f"Arguments: {args}")
    print(f"Using {num_processes} processes for parallel generation")

    file_path = f'{workdir}/{version}/generated_data'
    user_df = pd.read_csv(os.path.join(file_path, 'user_profiles.csv'))
    medical_df = pd.read_csv(os.path.join(file_path, 'medical_records.csv'))
    print("=== Health Data Generator v2.0 (multiprocessing) ===")

    user_data_list = []
    for i in range(len(user_df)):
        user_id = user_df.iloc[i]['user_id']
        user_profile = user_df.iloc[i].to_dict()
        medical_record = medical_df[medical_df['PatientID'] == user_id].iloc[0].to_dict()
        user_data_list.append((user_id, user_profile, medical_record))

    if num_processes == 1:

        results = []
        for user_data in user_data_list:
            result = process_user_data(user_data, workdir, version, args.seed, args.plot)
            results.append(result)
    else:

        with Pool(processes=num_processes) as pool:

            worker_func = partial(process_user_data, workdir=workdir, version=version, base_seed=args.seed)
            results = pool.map(worker_func, user_data_list)

    print("\n=== Generation Summary ===")
    total_daily_points = 0
    total_insole_points = 0

    for result in results:
        user_id = result['user_id']
        daily_shape = result['daily_shape']
        insole_shape = result['insole_shape']
        status_range = result['status_range']

        print(f"\nUser {user_id}:")
        print(f"  Health data: {daily_shape[0]} data points, {daily_shape[1]} metrics")
        print(f"  Insole data: {insole_shape[0]} data points, {insole_shape[1]} metrics")
        print(f"  Health-state range: {status_range[0]:.3f} to {status_range[1]:.3f}")

        total_daily_points += daily_shape[0]
        total_insole_points += insole_shape[0]

    print(f"\nTotal:")
    print(f"  Users processed: {len(results)}")
    print(f"  Total health-data points: {total_daily_points}")
    print(f"  Total insole-data points: {total_insole_points}")
    print(f"  Processes used: {num_processes}")

if __name__ == '__main__':
    main()
