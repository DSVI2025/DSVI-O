"""
Health-pattern generation module
Handles health-state pattern generation and transition logic
"""

import numpy as np
from .config import (CHANGE_PATTERNS, ABNORMAL_RULES, MAX_PATTERN_DURATION_MINUTES,
                     TIME_GRANULARITY_SECONDS, INTERVALS_PER_MINUTE, INTERVALS_PER_HOUR,
                     INTERVALS_PER_DAY, seconds_to_intervals, intervals_to_seconds)

class HealthPatternGenerator:
    """Health-pattern generator"""

    def __init__(self, rng):
        """
        Initialize the health-pattern generator

        Args:
            rng (np.random.Generator): random number generator
        """
        self.rng = rng

    def generate_random_group_info(self):
        """
        Generate group metadata by sampling from CHANGE_PATTERNS

        Returns:
            dict: Mapping containing individual_type and abnormal_group
        """
        individual_type = self.rng.choice(list(CHANGE_PATTERNS.keys()))

        pattern = CHANGE_PATTERNS[individual_type]
        needs_abnormal = pattern['to'] > 0

        if needs_abnormal:
            abnormal_group = self.rng.choice(list(ABNORMAL_RULES.keys()))
        else:
            abnormal_group = 'Normal'

        return {
            'individual_type': individual_type,
            'abnormal_group': abnormal_group
        }

    def generate_multi_day_health_states(self, total_intervals):
        """
        Generate a multi-day health-state series with pattern transitions
        Each CHANGE_PATTERNS event lasts at most 12 hours (8640 intervals).

        Args:
            total_intervals (int): Total number of five-second intervals

        Returns:
            dict: Health-state series and pattern metadata
        """
        health_states = np.zeros(total_intervals, dtype=float)
        current_interval = 0
        last_state = 0.0
        results = {
            "health_states": health_states,
            "patterns": []
        }

        max_pattern_duration_intervals = MAX_PATTERN_DURATION_MINUTES * INTERVALS_PER_MINUTE
        min_pattern_duration_intervals = 30 * INTERVALS_PER_MINUTE

        while current_interval < total_intervals:

            pattern_name = self._select_next_pattern_based_on_state(last_state)
            pattern = CHANGE_PATTERNS[pattern_name]

            remaining_intervals = total_intervals - current_interval
            max_duration = min(max_pattern_duration_intervals, remaining_intervals)

            if max_duration <= min_pattern_duration_intervals:
                health_states[current_interval:] = last_state
                break

            min_duration = min(min_pattern_duration_intervals, max_duration)
            duration = self.rng.integers(min_duration, max_duration + 1)

            adjusted_pattern = self._adjust_pattern_for_continuity(pattern, last_state, pattern_name)

            segment_states = self.apply_change_pattern(
                adjusted_pattern['from'],
                adjusted_pattern['to'],
                adjusted_pattern['type'],
                duration
            )

            actual_end = min(current_interval + len(segment_states), total_intervals)
            actual_segment = segment_states[:actual_end - current_interval]

            health_states[current_interval:actual_end] = actual_segment

            results["patterns"].append({
                "pattern_name": pattern_name,
                "from": current_interval,
                "to": actual_end,
                "type": adjusted_pattern['type'],
                "duration": len(actual_segment)
            })

            current_interval = actual_end
            last_state = actual_segment[-1] if len(actual_segment) > 0 else last_state

        results["health_states"] = health_states
        return results

    def _adjust_pattern_for_continuity(self, pattern, last_state, pattern_name):
        """
        Adjust a pattern to preserve continuity with the previous state
        Preserve the pattern semantics while aligning its starting state

        Args:
            pattern (dict): Original pattern configuration
            last_state (float): Ending state of the previous pattern
            pattern_name (str): pattern name

        Returns:
            dict: Adjusted pattern configuration
        """
        adjusted_pattern = pattern.copy()

        adjusted_pattern['from'] = last_state

        if 'Recovery' in pattern_name:

            adjusted_pattern['to'] = 0.0

        elif 'Warning' in pattern_name and ' to ' not in pattern_name:

            adjusted_pattern['to'] = 1.0

        elif 'Warning to Illness' in pattern_name:

            adjusted_pattern['to'] = 2.0

        elif 'Illness to Warning' in pattern_name:

            adjusted_pattern['to'] = 1.0

        elif 'Always Healthy' == pattern_name:

            adjusted_pattern['to'] = 0.0

            if last_state > 0.1:
                adjusted_pattern['type'] = 'linear'

        elif 'Always Ill' == pattern_name:

            adjusted_pattern['to'] = 2.0

            if last_state < 1.9:
                adjusted_pattern['type'] = 'linear'

        else:

            if abs(adjusted_pattern['from'] - pattern['to']) < 0.1:
                if last_state < 0.5:
                    adjusted_pattern['to'] = self.rng.choice([1.0, 2.0])
                elif last_state > 1.5:
                    adjusted_pattern['to'] = self.rng.choice([0.0, 1.0])
                else:
                    adjusted_pattern['to'] = self.rng.choice([0.0, 2.0])
            else:
                adjusted_pattern['to'] = pattern['to']

        return adjusted_pattern

    def _select_next_pattern_based_on_state(self, current_state):
        """
        Select the next pattern from the current health state
        Keep state transitions continuous and plausible

        Args:
            current_state (float): current health state (0=Normal, 1=Warning, 2=Illness)

        Returns:
            str: Selected pattern name
        """

        if current_state <= 0.5:
            suitable_patterns = [
                'Always Healthy',
                'Sudden Warning',
                'Linear Gradual Warning',
                'Exponential Gradual Warning'
            ]

            weights = [0.6, 0.15, 0.15, 0.1]

        elif 0.5 < current_state <= 1.5:
            suitable_patterns = [
                'Sudden Recovery',
                'Linear Gradual Recovery',
                'Exponential Gradual Recovery',
                'Sudden Warning to Illness',
            ]

            weights = [0.3, 0.2, 0.2, 0.3]

        else:
            suitable_patterns = [
                'Sudden Illness to Warning',
                'Always Ill'
            ]

            weights = [0.7, 0.3]

        return self.rng.choice(suitable_patterns, p=weights)

    def generate_health_status_schedule(self, individual_type):
        """
        Generate a one-day health-state schedule for an individual type.

        Args:
            individual_type (str): Individual type

        Returns:
            np.array: One health-state value per five-second interval
        """

        if individual_type == 'Always Healthy':
            return np.zeros(INTERVALS_PER_DAY, dtype=float)
        elif individual_type == 'Always Ill':
            return np.full(INTERVALS_PER_DAY, 2.0, dtype=float)
        elif individual_type in CHANGE_PATTERNS:
            pattern = CHANGE_PATTERNS[individual_type]
            return self.apply_change_pattern(
                pattern['from'], pattern['to'], pattern['type'], INTERVALS_PER_DAY
            )
        else:

            return np.zeros(INTERVALS_PER_DAY, dtype=float)

    def apply_change_pattern(self, from_state, to_state, pattern_type, duration_intervals):
        """
        Generate continuous health-state labels from a change pattern

        Args:
            from_state (float): starting state
            to_state (float): target state
            pattern_type (str): Change-pattern type
            duration_intervals (int): Duration in five-second intervals

        Returns:
            np.array: Continuous health-state values in [0, 2]
        """
        if duration_intervals <= 0:
            return np.array([])

        if abs(from_state - to_state) < 1e-6:
            return np.full(duration_intervals, from_state, dtype=float)

        hour_intervals = INTERVALS_PER_HOUR
        min_change_intervals = 10 * INTERVALS_PER_MINUTE

        if pattern_type == 'sudden':

            if duration_intervals < hour_intervals:
                change_point = duration_intervals // 2
            else:

                min_change = 30 * INTERVALS_PER_MINUTE
                max_change = min(duration_intervals - min_change, 2 * hour_intervals)
                change_point = self.rng.integers(min_change, max_change)

            states = np.full(duration_intervals, from_state, dtype=float)
            states[change_point:] = to_state
            return states

        elif pattern_type == 'linear':

            if duration_intervals < 2 * hour_intervals:
                change_start = min_change_intervals
                change_duration = duration_intervals - 2 * min_change_intervals
            else:
                change_start = self.rng.integers(min_change_intervals,
                                               min(hour_intervals, duration_intervals // 4))
                max_change_duration = min(6 * hour_intervals,
                                        duration_intervals - change_start - min_change_intervals)
                change_duration = self.rng.integers(hour_intervals, max_change_duration)

            change_end = min(change_start + change_duration, duration_intervals - min_change_intervals)

            states = np.full(duration_intervals, from_state, dtype=float)

            if change_end > change_start:
                for i in range(change_start, change_end):
                    progress = (i - change_start) / (change_end - change_start)
                    states[i] = from_state + (to_state - from_state) * progress
                states[change_end:] = to_state
            return states

        elif pattern_type == 'exponential':

            if duration_intervals < 2 * hour_intervals:
                change_start = min_change_intervals
                change_duration = duration_intervals - 2 * min_change_intervals
            else:
                change_start = self.rng.integers(min_change_intervals,
                                               min(hour_intervals, duration_intervals // 4))
                max_change_duration = min(6 * hour_intervals,
                                        duration_intervals - change_start - min_change_intervals)
                change_duration = self.rng.integers(hour_intervals, max_change_duration)

            change_end = min(change_start + change_duration, duration_intervals - min_change_intervals)

            states = np.full(duration_intervals, from_state, dtype=float)
            if change_end > change_start:
                for i in range(change_start, change_end):
                    progress = (i - change_start) / (change_end - change_start)

                    sigmoid_progress = 1 / (1 + np.exp(-6 * (progress - 0.5)))
                    states[i] = from_state + (to_state - from_state) * sigmoid_progress
                states[change_end:] = to_state
            return states

        else:

            return np.full(duration_intervals, from_state, dtype=float)

    def select_abnormal_group_for_states(self, health_states):
        """
        Select an abnormal group from the health-state series
        Choose a random abnormal group when warning or abnormal states occur

        Args:
            health_states (np.array): Health-state series

        Returns:
            str: abnormal-group name
        """
        has_abnormal = np.any(health_states > 0.5)

        if has_abnormal:
            return self.rng.choice(list(ABNORMAL_RULES.keys()))
        else:
            return 'Normal'
