"""
Health data generator package
Provides modular health data generation
"""

from .health_patterns import HealthPatternGenerator
from .activity import ActivityCalculator, circadian_wave
from .physiological_indicators import PhysiologicalCalculator
from .insole_data import InsoleDataGenerator
from . import config

def plot_health_states(*args, **kwargs):
    from .utils import plot_health_states as _plot_health_states

    return _plot_health_states(*args, **kwargs)

__all__ = [
    'HealthPatternGenerator',
    'ActivityCalculator',
    'PhysiologicalCalculator',
    'InsoleDataGenerator',
    'circadian_wave',
    'config',
    'plot_health_states'
]

__version__ = '2.0.0'
