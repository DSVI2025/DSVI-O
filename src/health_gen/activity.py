"""
Activity calculation module
Handles daily activities, schedule generation, and activity-related physiological calculations
"""

import numpy as np
from .config import (
    DEFAULT_SCHEDULE_TEMPLATE, ACTIVITY_DELTAS, MET_VALUES, ACTIVITY_INTENSITY_MAP,
    SECONDS_PER_DAY
)

class ActivityCalculator:
    """Activity calculator"""

    def __init__(self, rng, age):
        """
        Initialize the activity calculator

        Args:
            rng (np.random.Generator): random number generator
            age (int): user age
        """
        self.rng = rng
        self.age = age

    def generate_daily_schedule(self):
        """
        Generate a personalized daily activity schedule

        Returns:
            list: Tuples of (start_time, end_time, activity_name)
        """
        schedule = []
        last_end = 0

        for name, start, end in DEFAULT_SCHEDULE_TEMPLATE:
            if start == 0:
                schedule.append((start, end, name))
                last_end = end
                continue

            if name in ['Active', 'Leisure']:
                if self.rng.random() < 0.8:
                    name = 'Leisure'
                else:
                    name = 'Active'

            start_shift = int(self.rng.normal(0, 600))
            end_shift = int(self.rng.normal(0, 600))

            real_start = max(last_end, start + start_shift)
            real_end = min(SECONDS_PER_DAY, end + end_shift)

            if real_end <= real_start:
                real_end = real_start + 1800

            schedule.append((real_start, real_end, name))
            last_end = real_end

        schedule[-1] = (schedule[-1][0], SECONDS_PER_DAY, schedule[-1][2])
        return schedule

    def get_activity_state(self, t, schedule=None):
        """
        Return the activity at the specified time

        Args:
            t (int): time (seconds)
            schedule (list, optional): Custom schedule

        Returns:
            str: Activity state
        """
        if schedule is None:

            h = t / 3600
            if 0 <= h < 7 or 22 <= h <= 24:
                return 'Sleeping'
            elif 7 <= h < 8:
                return 'Waking'
            elif 8 <= h < 12:
                return 'Active'
            elif 12 <= h < 13:
                return 'Lunch'
            elif 13 <= h < 17:
                return 'Active'
            elif 17 <= h < 19:
                return 'Dinner'
            elif 19 <= h < 22:
                return 'Leisure'
            else:
                return 'Resting'

        for start, end, name in schedule:
            if start <= t <= end:

                if name == 'Leisure' and self.rng.random() < 0.1:
                    return 'Resting'
                if name == 'Lunch' and self.rng.random() < 0.1:
                    return 'Active'
                return name
        return 'Resting'

    def calculate_activity_hr_delta(self, activity):
        """
        Calculate the heart-rate delta for an activity

        Args:
            activity (str): activity type

        Returns:
            float: heart-rate delta
        """
        base_delta = ACTIVITY_DELTAS['heart_rate'].get(activity, 0)
        return self.rng.normal(base_delta, 0.1)

    def calculate_activity_spo2_delta(self, activity):
        """
        Calculate the oxygen-saturation delta for an activity

        Args:
            activity (str): activity type

        Returns:
            float: oxygen-saturation delta
        """
        base_delta = ACTIVITY_DELTAS['spo2'].get(activity, 0)
        return self.rng.normal(base_delta, 0.1)

    def get_step_increment(self, activity, weather='Sunny'):
        """
        Calculate the step increment for an activity

        Args:
            activity (str): activity type
            weather (str): weather

        Returns:
            int: step increment
        """

        if activity == 'Sleeping':
            return 0

        base = ACTIVITY_DELTAS['steps'].get(activity, 0)

        if activity == 'Active':

            base = self.rng.uniform(8, 15)
        elif activity == 'Waking':
            base = self.rng.uniform(3, 8)
        elif activity in ['Lunch', 'Dinner']:

            if self.rng.random() < 0.3:
                base = self.rng.uniform(1, 4)
            else:
                base = 0
        elif activity == 'Leisure':

            if self.rng.random() < 0.2:
                base = self.rng.uniform(0.5, 3)
            else:
                base = 0
        elif activity == 'Resting':

            if self.rng.random() < 0.05:
                base = self.rng.uniform(0, 1)
            else:
                base = 0

        if self.age >= 80:
            base *= 0.5
        elif self.age >= 70:
            base *= 0.7
        elif self.age <= 30:
            base *= 1.2

        if weather in ['Rainy', 'Stormy', 'Snowy']:
            base *= 0.6
        elif weather == 'Sunny':
            base *= 1.1

        if base <= 0:
            return 0

        return int(np.round(base + self.rng.uniform(-0.5, 0.5)))

    def generate_daily_activity_limits(self, weather='Sunny', max_status=0.0, abnormal_group='Normal'):
        """
        Generate daily step and active-minute budgets for an older adult.

        Returns:
            tuple: Daily step limit and daily active-minute limit
        """
        if self.age >= 80:
            step_low, step_high = 1200, 4500
            active_low, active_high = 20, 100
        elif self.age >= 70:
            step_low, step_high = 2500, 6500
            active_low, active_high = 35, 150
        else:
            step_low, step_high = 3500, 8500
            active_low, active_high = 45, 180

        step_limit = float(self.rng.integers(step_low, step_high + 1))
        active_limit = float(self.rng.integers(active_low, active_high + 1))

        status_factor = 1.0
        if max_status >= 1.5:
            status_factor = 0.55
        elif max_status >= 0.5:
            status_factor = 0.75

        if abnormal_group == 'Step Count Abnormality':
            status_factor *= 0.75

        if weather in ['Rainy', 'Stormy', 'Snowy']:
            status_factor *= 0.75

        step_limit *= status_factor
        active_limit *= status_factor

        return max(500, int(round(step_limit))), max(10.0, active_limit)

    def apply_step_budget(self, step, current_steps, daily_step_limit):
        """
        Scale or cap the step increment using the remaining daily budget.
        """
        remaining_steps = daily_step_limit - current_steps
        if step <= 0 or remaining_steps <= 0:
            return 0

        progress = current_steps / max(daily_step_limit, 1)
        if progress >= 0.8:
            fatigue_ratio = min((progress - 0.8) / 0.2, 1.0)
            if self.rng.random() < fatigue_ratio * 0.75:
                return 0
            step = int(round(step * (1.0 - 0.6 * fatigue_ratio)))

        return max(0, min(step, remaining_steps))

    def calculate_activity_calorie_delta(self, activity):
        """
        Calculate the calorie delta for an activity

        Args:
            activity (str): activity type

        Returns:
            float: calorie delta
        """
        met = MET_VALUES.get(activity, 1)

        if activity == 'Active':
            met = self.rng.uniform(2.0, 3.5)
        elif activity == 'Sleeping':
            met = self.rng.uniform(0.8, 1.0)
        elif activity in ['Lunch', 'Dinner']:
            met = self.rng.uniform(1.3, 1.8)
        elif activity == 'Leisure':
            met = self.rng.uniform(1.0, 1.5)
        else:
            met = self.rng.normal(met, 0.1)

        return (met - 1) * 0.01

    def calculate_activity_intensity(self, activity):
        """
        Calculate activity intensity

        Args:
            activity (str): activity type

        Returns:
            float: Activity-intensity value
        """
        from .config import WEARABLE_CONFIG
        cfg = WEARABLE_CONFIG['activity_intensity']
        base_intensity = ACTIVITY_INTENSITY_MAP.get(activity, cfg['mu'])
        intensity = base_intensity + self.rng.normal(0, 2)

        return round(np.clip(intensity, cfg['clip'][0], cfg['clip'][1]), 1)

    def calculate_weather_effects(self, weather):
        """
        Calculate weather effects on physiological indicators

        Args:
            weather (str): weather

        Returns:
            tuple: Heart-rate and oxygen-saturation deltas
        """
        if weather == 'Sunny':
            return 0, 0
        elif weather == 'Cloudy':
            hr_delta = self.rng.choice([-1, 1]) * self.rng.uniform(1, 2)
            spo2_delta = self.rng.choice([-1, 1]) * 0.1
            return hr_delta, spo2_delta
        elif weather == 'Rainy':
            return self.rng.uniform(5, 10), -self.rng.uniform(0.2, 0.5)
        elif weather == 'Stormy':
            return self.rng.uniform(12, 18), -self.rng.uniform(0.5, 1.0)
        elif weather == 'Snowy':
            return self.rng.uniform(8, 15), -self.rng.uniform(0.5, 0.8)
        else:
            return 0, 0

def circadian_wave(t, peak=15*3600, amp=7):
    """
    Generate a circadian rhythm waveform

    Args:
        t (int): time (seconds)
        peak (int): Peak time in seconds (default: 15:00)
        amp (float): Amplitude

    Returns:
        float: Circadian-rhythm value
    """
    return amp * np.sin(2 * np.pi * (t - peak) / SECONDS_PER_DAY)
