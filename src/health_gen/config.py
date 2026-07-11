"""
Health data generator configuration
Contains constants, configuration parameters, and rule definitions
"""

HEALTH_STATUS = {
    'Normal': 0,
    'Warning': 1,
    'Critical Illness': 2
}

# pattern type
CHANGE_PATTERNS = {
    'Always Healthy': {'from': 0, 'to': 0, 'type': 'linear'},
    'Always Ill': {'from': 2, 'to': 2, 'type': 'linear'},
    'Sudden Warning': {'from': 0, 'to': 1, 'type': 'sudden'},
    'Linear Gradual Warning': {'from': 0, 'to': 1, 'type': 'linear'},
    'Exponential Gradual Warning': {'from': 0, 'to': 1, 'type': 'exponential'},
    'Sudden Recovery': {'from': 1, 'to': 0, 'type': 'sudden'},
    'Linear Gradual Recovery': {'from': 1, 'to': 0, 'type': 'linear'},
    'Exponential Gradual Recovery': {'from': 1, 'to': 0, 'type': 'exponential'},
    'Sudden Warning to Illness': {'from': 1, 'to': 2, 'type': 'sudden'},
    'Sudden Illness to Warning': {'from': 2, 'to': 1, 'type': 'sudden'},
}

ABNORMAL_RULES = {
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

ACTIVITY_DELTAS = {
    'heart_rate': {
        'Sleeping': -8, 'Waking': 5, 'Active': 9,
        'Lunch': 3, 'Dinner': 3, 'Leisure': 2, 'Resting': 0
    },
    'spo2': {
        'Sleeping': -0.2, 'Waking': 0, 'Active': -0.1,
        'Lunch': 0, 'Dinner': 0, 'Leisure': 0, 'Resting': 0
    },
    'steps': {
        'Sleeping': 0, 'Waking': 5, 'Active': 12,
        'Lunch': 2, 'Dinner': 2, 'Leisure': 1, 'Resting': 0
    }
}

MET_VALUES = {
    'Sleeping': 0.9, 'Waking': 2, 'Active': 2.35,
    'Lunch': 1.5, 'Dinner': 1.5, 'Leisure': 1.2, 'Resting': 1
}

ACTIVITY_INTENSITY_MAP = {
    'Active': 45,
    'Sleeping': 9,
    'Resting': 10,
    'Waking': 20,
    'Lunch': 15,
    'Dinner': 15,
    'Leisure': 18
}

SECONDS_PER_DAY = 86400
MINUTES_PER_DAY = 1440
HOURS_PER_DAY = 24
MAX_PATTERN_DURATION_HOURS = 12
MAX_PATTERN_DURATION_MINUTES = MAX_PATTERN_DURATION_HOURS * 60

TIME_GRANULARITY_SECONDS = 5
INTERVALS_PER_MINUTE = 60 // TIME_GRANULARITY_SECONDS
INTERVALS_PER_HOUR = INTERVALS_PER_MINUTE * 60
INTERVALS_PER_DAY = INTERVALS_PER_HOUR * 24

DEFAULT_SCHEDULE_TEMPLATE = [
    ('Sleeping', 0, 7*3600),              # 0:00-7:00
    ('Waking', 7*3600, 8*3600),         # 7:00-8:00
    ('Active', 8*3600, 12*3600),        # 8:00-12:00
    ('Lunch', 12*3600, 13*3600),       # 12:00-13:00
    ('Active', 13*3600, 17*3600),       # 13:00-17:00
    ('Dinner', 17*3600, 19*3600),       # 17:00-19:00
    ('Leisure', 19*3600, 22*3600),       # 19:00-22:00
    ('Sleeping', 22*3600, 24*3600)        # 22:00-24:00
]

def seconds_to_intervals(seconds):
    """Convert seconds to five-second intervals."""
    return seconds // TIME_GRANULARITY_SECONDS

def intervals_to_seconds(intervals):
    """Convert five-second intervals to seconds."""
    return intervals * TIME_GRANULARITY_SECONDS

def hours_to_intervals(hours):
    """Convert hours to five-second intervals."""
    return hours * INTERVALS_PER_HOUR

def intervals_to_hours(intervals):
    """Convert five-second intervals to hours."""
    return intervals / INTERVALS_PER_HOUR
