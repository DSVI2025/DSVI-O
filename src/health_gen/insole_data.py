"""
Intelligent-insole data generation module
Handles all intelligent-insole calculations and data generation
"""

import numpy as np
import pandas as pd
from .config import INSOLE_CONFIG

class InsoleDataGenerator:
    """Intelligent-insole data generator"""

    def __init__(self, rng, age, weight):
        """
        Initialize the intelligent-insole data generator

        Args:
            rng (np.random.Generator): random number generator
            age (int): user age
            weight (float): user weight
        """
        self.rng = rng
        self.age = age
        self.weight = weight

    def generate_insole_data(self, daily_df):
        """
        Generate intelligent-insole data from daily_df

        Args:
            daily_df (pd.DataFrame): Daily physiological data

        Returns:
            pd.DataFrame: Intelligent-insole data
        """
        channels = ['LF', 'LH', 'RF', 'RH']
        base_pressures = {'LF': 0.25, 'LH': 0.25, 'RF': 0.25, 'RH': 0.25}

        insole_rows = []

        for idx in range(len(daily_df)):
            row = daily_df.iloc[idx]
            t = idx * 5
            activity = row['activity']
            status_value = row.get('status', 0.0)
            abnormal_group = row.get('abnormal_group', 'Normal')

            step_freq = self._calculate_step_frequency(activity, status_value, abnormal_group)
            step_period = 60 / step_freq if step_freq > 0 else 60

            pressures = self._generate_activity_specific_pressures(
                activity, channels, base_pressures, t, step_period,
                status_value, abnormal_group, step_freq
            )

            pressure_asym = abs(pressures['LF'] + pressures['LH'] - pressures['RF'] - pressures['RH']) / (self.weight + 1e-3)
            gait_type = self._get_gait_type(step_freq, pressure_asym)
            center_x, center_y = self._get_center_offset(status_value, t, step_period, activity, step_freq)

            time_str = row['time']

            stride_length = self._calculate_stride_length(step_freq, status_value, activity)
            contact_time = self._calculate_contact_time(step_freq, status_value, activity)
            symmetry = self._calculate_symmetry(status_value, activity)
            balance_score = self._calculate_balance(status_value, activity)
            force_estimate = self._calculate_force_estimate(activity, pressures)
            motion_type = self._get_motion_type(step_freq)

            left_weight = pressures['LF'] + pressures['LH']
            right_weight = pressures['RF'] + pressures['RH']
            total_weight = left_weight + right_weight
            lr_ratio = round(left_weight / (total_weight + 0.01), 3) if total_weight > 0 else 0.5

            exercise_intensity = self._calculate_exercise_intensity(activity, step_freq)
            plantar_fatigue = self._calculate_plantar_fatigue(activity, t, status_value)

            insole_rows.append({
                'time': time_str,
                'LF': pressures['LF'],
                'LH': pressures['LH'],
                'RF': pressures['RF'],
                'RH': pressures['RH'],
                'step_freq': round(step_freq, 1),
                'stride_length': stride_length,
                'contact_time': contact_time,
                'symmetry': symmetry,
                'gait_type': gait_type,
                'motion_type': motion_type,
                'force_estimate': force_estimate,
                'balance_score': balance_score,
                'lr_ratio': lr_ratio,
                'center_x': center_x,
                'center_y': center_y,
                'exercise_intensity': exercise_intensity,
                'plantar_fatigue': plantar_fatigue
            })

        return pd.DataFrame(insole_rows)

    def _generate_activity_specific_pressures(self, activity, channels, base_pressures, t, step_period, status_value, abnormal_group, step_freq):
        """
        Generate activity-specific pressure patterns

        Args:
            activity (str): activity type
            channels (list): pressure-sensor channels
            base_pressures (dict): baseline pressure distribution
            t (int): time
            step_period (float): step period
            status_value (float): Health-state value
            abnormal_group (str): abnormal group

        Returns:
            dict: pressure values by channel
        """
        pressures = {}

        if activity == 'Sleeping':

            for ch in channels:
                pressures[ch] = self.rng.normal(0, 0.5)
                pressures[ch] = max(0, pressures[ch])

        elif activity == 'Resting':

            for ch in channels:
                if ch in ['LH', 'RH']:
                    base_pressure = base_pressures[ch] * self.weight * 0.6
                else:
                    base_pressure = base_pressures[ch] * self.weight * 0.3

                pressures[ch] = base_pressure + self.rng.normal(0, 1.5)
                pressures[ch] = max(0, pressures[ch])

        elif activity in ['Active', 'Waking']:

            for ch in channels:

                if ch in ['LF', 'LH']:
                    phase = (t / step_period) % (2 * np.pi)
                else:
                    phase = (t / step_period + np.pi) % (2 * np.pi)

                base = base_pressures[ch] * self.weight

                if activity == 'Active':
                    pressure_multiplier = 1.0 + 0.6 * np.sin(phase)
                    noise_level = 3.0
                else:
                    pressure_multiplier = 1.0 + 0.3 * np.sin(phase)
                    noise_level = 2.0

                pressure = base * pressure_multiplier

                if ch in ['LF', 'RF'] and activity == 'Active':
                    pressure *= 1.2

                pressure += self.rng.normal(0, noise_level)
                pressures[ch] = max(0, pressure)

        elif activity in ['Lunch', 'Dinner', 'Leisure']:

            for ch in channels:
                if ch in ['LH', 'RH']:
                    base_pressure = base_pressures[ch] * self.weight * 0.5
                else:
                    base_pressure = base_pressures[ch] * self.weight * 0.4

                if step_freq > 0:
                    phase = (t / (step_period * 2)) % (2 * np.pi)
                    pressure_variation = 0.1 * np.sin(phase)
                else:
                    pressure_variation = 0

                pressure = base_pressure * (1 + pressure_variation)
                pressure += self.rng.normal(0, 1.0)
                pressures[ch] = max(0, pressure)

        if abnormal_group == 'Step Count Abnormality':
            for ch in channels:
                pressures[ch] *= 0.8

        if status_value >= 1.5:

            for ch in channels:
                pressures[ch] *= self.rng.uniform(0.7, 1.3)

        for ch in channels:
            pressures[ch] = round(np.clip(pressures[ch], 0, self.weight * 1.5), 1)

        return pressures

    def _calculate_exercise_intensity(self, activity, step_freq):
        """
        Calculate exercise intensity

        Args:
            activity (str): activity type
            step_freq (float): step frequency

        Returns:
            float: exercise intensity (0-100)
        """
        base_intensity = {
            'Sleeping': 0,
            'Resting': 5,
            'Waking': 15,
            'Lunch': 10,
            'Dinner': 10,
            'Leisure': 20,
            'Active': 60
        }

        intensity = base_intensity.get(activity, 30)

        if step_freq > 100:
            intensity += 20
        elif step_freq > 80:
            intensity += 10
        elif step_freq < 50:
            intensity = max(0, intensity - 15)

        intensity += self.rng.normal(0, 5)
        return round(np.clip(intensity, 0, 100), 1)

    def _calculate_plantar_fatigue(self, activity, t, status_value):
        """
        Calculate plantar fatigue

        Args:
            activity (str): activity type
            t (int): Current time in seconds
            status_value (float): Health-state value

        Returns:
            float: fatigue (0-100)
        """

        time_hours = t / 3600
        base_fatigue = min(time_hours * 2, 30)

        activity_fatigue = {
            'Sleeping': -10,
            'Resting': -5,
            'Waking': 5,
            'Lunch': 0,
            'Dinner': 0,
            'Leisure': 3,
            'Active': 15
        }

        fatigue = base_fatigue + activity_fatigue.get(activity, 0)

        if status_value >= 1.5:
            fatigue *= 1.3
        elif status_value >= 0.5:
            fatigue *= 1.1

        fatigue += self.rng.normal(0, 3)
        return round(np.clip(fatigue, 0, 100), 1)

    def _calculate_step_frequency(self, activity, status_value, abnormal_group):
        """
        Calculate step frequency

        Args:
            activity (str): activity type
            status_value (float): Health-state value
            abnormal_group (str): abnormal group

        Returns:
            float: Step-frequency value
        """
        if activity in ['Sleeping', 'Resting']:
            return 0

        cfg = INSOLE_CONFIG['step_frequency']

        if activity in ['Lunch', 'Dinner']:
            if self.rng.random() < 0.75:
                return 0
            base = self.rng.normal(42, 8)
            clip = (20, 58)
        elif activity == 'Leisure':
            if self.rng.random() < 0.6:
                return 0
            base = self.rng.normal(48, 9)
            clip = (20, 58)
        elif activity == 'Waking':
            if self.rng.random() < 0.35:
                return 0
            base = self.rng.normal(58, 10)
            clip = (25, 75)
        else:
            base = cfg['mu'] - 0.35 * max(self.age - 60, 0) + 5
            clip = (45, cfg['clip'][1])

        if status_value >= 1.5:
            base *= 0.7
        elif status_value >= 0.5:
            base *= 0.85

        if abnormal_group == 'Step Count Abnormality':
            base *= 0.7

        step_freq = base + self.rng.normal(0, cfg['sigma'])
        return float(np.clip(step_freq, clip[0], clip[1]))

    def _get_gait_type(self, step_freq, pressure_asym):
        """
        Determine gait type

        Args:
            step_freq (float): step frequency
            pressure_asym (float): pressure asymmetry

        Returns:
            str: gait type
        """
        if step_freq == 0:
            return 'Resting'
        elif pressure_asym > 0.3:
            return 'Abnormal'
        elif step_freq > 110:
            return 'Brisk'
        elif step_freq < 60:
            return 'Slow'
        else:
            return 'Normal'

    def _get_center_offset(self, status_value, t, step_period, activity, step_freq):
        """
        Calculate center-of-pressure offset

        Args:
            status_value (float): Health-state value
            t (int): time
            step_period (float): step period
            activity (str): activity type

        Returns:
            tuple: (center_x, center_y)
        """
        if activity == 'Sleeping':
            center_x = self.rng.normal(0, 1)
            center_y = self.rng.normal(0, 1)
        elif step_freq <= 0 or activity == 'Resting':
            center_x = self.rng.normal(0, 1.5)
            center_y = self.rng.normal(0, 1.5)
        else:
            if status_value >= 1.5 or self.age > 80:
                center_x = self.rng.normal(0, 8)
                center_y = self.rng.normal(0, 8)
            else:
                center_x = self.rng.normal(0, 3)
                center_y = self.rng.normal(0, 3)

            center_x += 5 * np.sin(2 * np.pi * t / step_period)
            center_y += 5 * np.cos(2 * np.pi * t / step_period)

        return round(center_x, 2), round(center_y, 2)

    def _calculate_stride_length(self, step_freq, status_value, activity):
        """
        Calculate stride length

        Args:
            step_freq (float): step frequency
            status_value (float): Health-state value
            activity (str): activity type

        Returns:
            float: stride length
        """
        if step_freq <= 0:
            return 0.0

        cfg = INSOLE_CONFIG['stride_length']
        stride_base = cfg['mu']

        activity_modifiers = {
            'Sleeping': 0,
            'Resting': 0,
            'Waking': 0.8,
            'Lunch': 0.7,
            'Dinner': 0.7,
            'Leisure': 0.9,
            'Active': 1.2
        }

        stride_base *= activity_modifiers.get(activity, 1.0)

        if status_value >= 1.5:
            stride_base *= 0.85
        elif status_value >= 0.5:
            stride_base *= 0.95

        stride_length = stride_base + self.rng.normal(0, cfg['sigma'])
        return round(np.clip(stride_length, cfg['clip'][0], cfg['clip'][1]), 1)

    def _calculate_contact_time(self, step_freq, status_value, activity):
        """
        Calculate ground-contact time

        Args:
            step_freq (float): step frequency
            status_value (float): Health-state value
            activity (str): activity type

        Returns:
            float: ground-contact time
        """
        if step_freq <= 0:
            return 0.0

        cfg = INSOLE_CONFIG['ground_contact_time']
        contact_base = cfg['mu']

        activity_modifiers = {
            'Sleeping': 0,
            'Resting': 0,
            'Waking': 1.1,
            'Lunch': 1.2,
            'Dinner': 1.2,
            'Leisure': 1.0,
            'Active': 0.8
        }

        contact_base *= activity_modifiers.get(activity, 1.0)

        if status_value >= 1.5:
            contact_base *= 1.1
        elif status_value >= 0.5:
            contact_base *= 1.05

        contact_time = contact_base + self.rng.normal(0, cfg['sigma'])
        return round(np.clip(contact_time, cfg['clip'][0], cfg['clip'][1]), 0)

    def _calculate_symmetry(self, status_value, activity):
        """
        Calculate left-right symmetry

        Args:
            status_value (float): Health-state value
            activity (str): activity type

        Returns:
            float: symmetry value
        """
        cfg = INSOLE_CONFIG['left_right_symmetry']
        symmetry_base = cfg['mu']
        noise_sigma = cfg['sigma']
        upper_clip = cfg['clip'][1]

        if activity == 'Sleeping':
            symmetry_base = 96.5
            noise_sigma = 1.0
            upper_clip = 99.5
        elif activity == 'Resting':
            symmetry_base = 95.5
            noise_sigma = 1.5
            upper_clip = 99.5
        elif activity == 'Active':
            symmetry_base *= 0.95
            noise_sigma = 3.5
        elif activity in ['Lunch', 'Dinner']:
            symmetry_base *= 0.98
            noise_sigma = 2.0
            upper_clip = 99.5
        elif activity == 'Leisure':
            symmetry_base *= 0.97
            noise_sigma = 2.5
            upper_clip = 99.5

        if status_value >= 1.5:
            symmetry_base *= 0.85
        elif status_value >= 0.5:
            symmetry_base *= 0.95

        symmetry = symmetry_base + self.rng.normal(0, noise_sigma)
        return round(np.clip(symmetry, cfg['clip'][0], upper_clip), 1)

    def _calculate_balance(self, status_value, activity):
        """
        Calculate balance score

        Args:
            status_value (float): Health-state value
            activity (str): activity type

        Returns:
            float: balance score
        """
        cfg = INSOLE_CONFIG['balance_status']
        balance_base = cfg['mu']
        noise_sigma = cfg['sigma']
        upper_clip = cfg['clip'][1]

        if activity == 'Sleeping':
            balance_base = 96
            noise_sigma = 1.5
            upper_clip = 99.5
        elif activity == 'Resting':
            balance_base = 92
            noise_sigma = 3.0
            upper_clip = 99.5
        elif activity == 'Active':
            balance_base *= 0.9
            noise_sigma = 8.0
        elif activity in ['Waking']:
            balance_base *= 0.85
            noise_sigma = 7.0
        elif activity in ['Lunch', 'Dinner', 'Leisure']:
            balance_base *= 0.92
            noise_sigma = 5.0
            upper_clip = 99.5

        if status_value >= 1.5:
            balance_base *= 0.85
        elif status_value >= 0.5:
            balance_base *= 0.95

        balance_score = balance_base + self.rng.normal(0, noise_sigma)
        return round(np.clip(balance_score, cfg['clip'][0], upper_clip), 1)

    def _calculate_force_estimate(self, activity, pressures):
        """
        Calculate force estimate

        Args:
            activity (str): activity type

        Returns:
            float: force estimate
        """
        cfg = INSOLE_CONFIG['force_estimation']
        pressure_total = sum(pressures.values())
        force_estimate = pressure_total * 9.8 + self.rng.normal(0, cfg['sigma'] * 0.15)

        if activity == 'Sleeping':
            low, high = 0, 80
        elif activity == 'Resting':
            low, high = 0, 500
        elif activity in ['Lunch', 'Dinner', 'Leisure']:
            low, high = 0, 800
        elif activity == 'Waking':
            low, high = 0, 900
        else:
            low, high = cfg['clip']

        return round(np.clip(force_estimate, low, high), 1)

    def _get_motion_type(self, step_freq):
        """
        Determine motion type

        Args:
            step_freq (float): step frequency

        Returns:
            str: motion type
        """
        if step_freq <= 0:
            return 'Stationary'
        elif step_freq < 60:
            return 'Slow Walking'
        elif step_freq < 90:
            return 'Normal Walking'
        elif step_freq < 120:
            return 'Fast Walking'
        else:
            return 'Running'
