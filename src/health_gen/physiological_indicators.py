"""
Physiological indicator calculation module
Handles physiological calculations and abnormal-group effects
"""

import numpy as np
from .config import WEARABLE_CONFIG, ABNORMAL_RULES
from .activity import circadian_wave

class PhysiologicalCalculator:
    """Physiological indicator calculator"""

    def __init__(self, rng, age, sbp_base, dbp_base):
        """
        Initialize the physiological indicator calculator

        Args:
            rng (np.random.Generator): random number generator
            age (int): user age
            sbp_base (float): baseline systolic pressure
            dbp_base (float): baseline diastolic pressure
        """
        self.rng = rng
        self.age = age
        self.sbp_base = sbp_base
        self.dbp_base = dbp_base

    def apply_abnormal_effects(self, abnormal_group, status_value, hr, spo2):
        """
        Apply abnormal-group effects to physiological indicators

        Args:
            abnormal_group (str): abnormal-group name
            status_value (float): current health-state value
            hr (float): current heart rate
            spo2 (float): current oxygen saturation

        Returns:
            tuple: Heart rate and oxygen saturation after abnormal-group effects
        """
        if abnormal_group == 'Normal':
            return hr, spo2

        rules = ABNORMAL_RULES.get(abnormal_group, {})

        for indicator, adjustment in rules.items():
            effect_magnitude = status_value * adjustment
            noise = self.rng.normal(0, abs(adjustment) * 0.1)

            if indicator == 'CRP':

                hr += effect_magnitude + noise
            elif indicator == 'WBC':

                hr += effect_magnitude * 2 + noise
            elif indicator == 'NEUT%':

                hr += effect_magnitude * 0.5 + noise
            elif indicator == 'LYM%':

                hr += effect_magnitude * 0.3 + noise
            elif indicator == 'SpO2':

                spo2 += effect_magnitude + noise
            elif indicator == 'RBC':

                spo2 += effect_magnitude * 2 + noise
            elif indicator == 'HCT':

                spo2 += effect_magnitude * 10 + noise
            elif indicator == 'SBP':

                hr += effect_magnitude * 0.2 + noise
            elif indicator == 'GLU':

                hr += effect_magnitude * 3 + noise
            elif indicator == 'HGB':

                spo2 += effect_magnitude * 0.1 + noise
                hr += effect_magnitude * 0.05 + noise
            elif indicator in ['TC', 'TG', 'ALT', 'UA', 'CREA']:

                hr += effect_magnitude * 0.1 + noise

        return hr, spo2

    def calculate_skin_temp(self, t, activity, status_value):
        """
        Calculate skin temperature

        Args:
            t (int): time (seconds)
            activity (str): activity type
            status_value (float): Health-state value

        Returns:
            float: skin temperature
        """
        cfg = WEARABLE_CONFIG['skin_temp']
        base_temp = cfg['mu'] + 0.3 * np.sin(2 * np.pi * t / 86400)

        if activity == 'Active':
            base_temp += self.rng.normal(0.5, 0.1)
        elif activity == 'Sleeping':
            base_temp -= self.rng.normal(0.3, 0.1)

        if status_value >= 1.5:
            base_temp *= 1.1
        elif status_value >= 0.5:
            base_temp *= 1.05

        return round(np.clip(base_temp, cfg['clip'][0], cfg['clip'][1]), 1)

    def calculate_hrv(self, activity, status_value):
        """
        Calculate heart-rate variability

        Args:
            activity (str): activity type
            status_value (float): Health-state value

        Returns:
            float: heart-rate variability
        """
        cfg = WEARABLE_CONFIG['hrv']
        base_hrv = 42 - 0.18 * max(self.age - 60, 0) - status_value * 6

        if activity == 'Sleeping':
            base_hrv += 8
        elif activity == 'Active':
            base_hrv -= 8
        elif activity in ['Lunch', 'Dinner', 'Leisure']:
            base_hrv -= 2

        if status_value >= 1.5:
            base_hrv *= 0.9
        elif status_value >= 0.5:
            base_hrv *= 0.95

        hrv = base_hrv + self.rng.normal(0, 6)
        return round(np.clip(hrv, cfg['clip'][0], cfg['clip'][1]), 1)

    def calculate_stress(self, activity, status_value):
        """
        Calculate stress index

        Args:
            activity (str): activity type
            status_value (float): Health-state value

        Returns:
            float: Stress index
        """
        cfg = WEARABLE_CONFIG['stress_index']
        base_stress = 34 + status_value * 16

        if activity == 'Active':
            base_stress += 14
        elif activity == 'Sleeping':
            base_stress -= 18
        elif activity in ['Lunch', 'Dinner', 'Leisure']:
            base_stress += 4

        if status_value >= 1.5:
            base_stress *= 1.12
        elif status_value >= 0.5:
            base_stress *= 1.06

        stress = base_stress + self.rng.normal(0, 12)
        return round(np.clip(stress, cfg['clip'][0], cfg['clip'][1]), 1)

    def calculate_blood_pressure(self, t, activity, status_value):
        """
        Calculate blood pressure

        Args:
            t (int): time (seconds)
            activity (str): activity type
            status_value (float): Health-state value

        Returns:
            tuple: (systolic pressure, diastolic pressure)
        """
        cfg_sys = WEARABLE_CONFIG['blood_pressure_sys']
        cfg_dia = WEARABLE_CONFIG['blood_pressure_dia']

        sbp_base = self.sbp_base + status_value * 10 + circadian_wave(t, amp=5)
        dbp_base = self.dbp_base + status_value * 5 + circadian_wave(t, amp=3)

        if activity == 'Active':
            sbp_base += 15
            dbp_base += 8

        sbp = sbp_base + self.rng.normal(0, cfg_sys['sigma'])
        dbp = dbp_base + self.rng.normal(0, cfg_dia['sigma'])

        sbp_val = round(np.clip(sbp, cfg_sys['clip'][0], cfg_sys['clip'][1]), 0)
        dbp_val = round(np.clip(dbp, cfg_dia['clip'][0], cfg_dia['clip'][1]), 0)

        return sbp_val, dbp_val

    def calculate_sleep_quality(self, activity, status_value):
        """
        Calculate sleep quality

        Args:
            activity (str): current activity
            status_value (float): Health-state value

        Returns:
            int: Sleep quality (0: awake, 1: light sleep, 2: deep sleep)
        """
        if activity != 'Sleeping':
            return 0

        if status_value < 0.5:
            return self.rng.choice([0, 1, 2], p=[0.05, 0.70, 0.25])
        elif status_value < 1.5:
            return self.rng.choice([0, 1, 2], p=[0.15, 0.75, 0.10])
        else:  # Illness
            return self.rng.choice([0, 1, 2], p=[0.40, 0.55, 0.05])
