import pandas as pd
import numpy as np
from datetime import datetime

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib

    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
except ModuleNotFoundError:
    plt = None
    mdates = None

def _require_matplotlib():
    if plt is None or mdates is None:
        raise RuntimeError("Plotting requires matplotlib. Install matplotlib or run without plot options.")

def plot_health_states(health_states, save_path='health_states_plot.png'):
    """
    Plot health states and pattern transitions
    Supports the health-pattern generator with 8-to-16-hour pattern durations.

    Args:
        health_states: Mapping contain　ing health_states and patterns
    """
    _require_matplotlib()
    from .config import INTERVALS_PER_HOUR

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12))

    health_values = health_states['health_states']
    patterns = health_states['patterns']

    time_hours = np.arange(len(health_values)) / INTERVALS_PER_HOUR

    ax1.plot(time_hours, health_values, linewidth=2, color='blue', alpha=0.7, label='health state')
    ax1.set_ylabel('Health-state value', fontsize=12)
    ax1.set_title('Health-State Time Series (8-16 Hour Patterns)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.1, 2.1)

    ax1.axhline(y=0, color='green', linestyle='--', alpha=0.6, label='Normal State (0)')
    ax1.axhline(y=1, color='orange', linestyle='--', alpha=0.6, label='Warning State (1)')
    ax1.axhline(y=2, color='red', linestyle='--', alpha=0.6, label='Illness State (2)')
    ax1.legend(loc='upper right')

    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightpink', 'lightcyan',
              'wheat', 'lavender', 'lightcoral', 'lightgray', 'lightsteelblue']

    for i, pattern in enumerate(patterns):
        start_hour = pattern["from"] / INTERVALS_PER_HOUR
        end_hour = pattern["to"] / INTERVALS_PER_HOUR
        duration_hours = pattern["duration"] / INTERVALS_PER_HOUR

        ax1.axvspan(start_hour, end_hour, alpha=0.3, color=colors[i % len(colors)])

        if i > 0:
            ax1.axvline(x=start_hour, color='black', linestyle='-', alpha=0.8, linewidth=1)

        mid_hour = (start_hour + end_hour) / 2
        ax1.text(mid_hour, 2.05, f"Pattern {i+1}", ha='center', va='bottom',
                fontsize=10, fontweight='bold')
        ax1.text(mid_hour, 1.9, f"{duration_hours:.1f}h", ha='center', va='bottom',
                fontsize=8, alpha=0.7)

    pattern_names = [p["pattern_name"] for p in patterns]
    pattern_durations = [p["duration"] / INTERVALS_PER_HOUR for p in patterns]
    pattern_types = [p["type"] for p in patterns]

    type_colors = {'sudden': 'red', 'linear': 'blue', 'exponential': 'green'}
    bar_colors = [type_colors.get(ptype, 'gray') for ptype in pattern_types]

    bars = ax2.bar(range(len(patterns)), pattern_durations, color=bar_colors, alpha=0.7)
    ax2.set_xlabel('Pattern index', fontsize=12)
    ax2.set_ylabel('Duration (hours)', fontsize=12)
    ax2.set_title('Pattern Durations and Types (8-16 Hour Validation)', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(patterns)))
    ax2.set_xticklabels([f"Pattern {i+1}" for i in range(len(patterns))])
    ax2.grid(True, alpha=0.3, axis='y')

    ax2.axhline(y=8, color='orange', linestyle='--', alpha=0.7, label='Minimum duration (8 hours)')
    ax2.axhline(y=16, color='red', linestyle='--', alpha=0.7, label='Maximum duration (16 hours)')

    for i, (bar, duration, name) in enumerate(zip(bars, pattern_durations, pattern_names)):

        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{duration:.1f}h', ha='center', va='bottom', fontsize=9, fontweight='bold')

        display_name = name if len(name) <= 8 else name[:6] + '..'
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                display_name, ha='center', va='center', fontsize=8, rotation=90,
                fontweight='bold', color='white')

    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.7, label=f'{ptype.capitalize()}')
                      for ptype, color in type_colors.items()]
    ax2.legend(handles=legend_elements, loc='upper right', title='Change type')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    print(f"\n=== Health Pattern Validation ===")
    print(f"Pattern count: {len(patterns)}")
    print(f"Pattern duration range: {min(pattern_durations):.1f} - {max(pattern_durations):.1f} hours")

    consecutive_same = sum(1 for i in range(1, len(pattern_names))
                          if pattern_names[i] == pattern_names[i-1])
    print(f"Consecutive identical patterns: {consecutive_same} times {'✓' if consecutive_same == 0 else '✗'}")

    duration_violations = sum(1 for d in pattern_durations if d < 8 or d > 16)
    print(f"Duration violations: {duration_violations} times {'✓' if duration_violations == 0 else '✗'}")

    unique_states = set(np.round(health_values).astype(int))
    has_all_states = {0, 1, 2}.issubset(unique_states)
    print(f"Contains all states (0, 1, 2): {'✓' if has_all_states else '✗'} - observed states: {sorted(unique_states)}")

    print(f"Plot saved to: {save_path}")

def plot_health_data(data, columns=None, save_path='health_data_plot.png',
                    title='Health Data Over Time', figsize=(15, 10)):
    """
    Plot health-data time series

    Args:
        data (pd.DataFrame): Health data containing a time column
        columns (list): Columns to plot; all numeric columns when None
        save_path (str): output path
        title (str): plot title
        figsize (tuple): figure size
    """
    _require_matplotlib()
    if 'time' not in data.columns:
        raise ValueError("Data must contain a 'time' column")

    data_copy = data.copy()
    data_copy['time'] = pd.to_datetime(data_copy['time'])

    if columns is None:
        exclude_cols = ['time', 'activity', 'abnormal_group']
        numeric_cols = data_copy.select_dtypes(include=[np.number]).columns
        columns = [col for col in numeric_cols if col not in exclude_cols]

    valid_columns = [col for col in columns if col in data_copy.columns]
    if not valid_columns:
        raise ValueError(f"Requested columns {columns} are not present in the data")

    n_cols = len(valid_columns)
    n_rows = (n_cols + 2) // 3

    fig, axes = plt.subplots(n_rows, 3, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for i, column in enumerate(valid_columns):
        row = i // 3
        col = i % 3
        ax = axes[row, col] if n_rows > 1 else axes[col]

        ax.plot(data_copy['time'], data_copy[column], linewidth=1.5, alpha=0.8)
        ax.set_title(f'{column}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel(column)
        ax.grid(True, alpha=0.3)

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))

        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    total_subplots = n_rows * 3
    for i in range(n_cols, total_subplots):
        row = i // 3
        col = i % 3
        if n_rows > 1:
            axes[row, col].set_visible(False)
        else:
            axes[col].set_visible(False)

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_insole_data(data, columns=None, save_path='insole_data_plot.png',
                    title='Smart Insole Data Over Time', figsize=(15, 10)):
    """
    Plot intelligent-insole time series

    Args:
        data (pd.DataFrame): Insole data containing a time column
        columns (list): Columns to plot; all numeric columns when None
        save_path (str): output path
        title (str): plot title
        figsize (tuple): figure size
    """
    _require_matplotlib()
    if 'time' not in data.columns:
        raise ValueError("Data must contain a 'time' column")

    data_copy = data.copy()
    data_copy['time'] = pd.to_datetime(data_copy['time'])

    if columns is None:
        exclude_cols = ['time', 'activity_type']
        numeric_cols = data_copy.select_dtypes(include=[np.number]).columns
        columns = [col for col in numeric_cols if col not in exclude_cols]

    valid_columns = [col for col in columns if col in data_copy.columns]
    if not valid_columns:
        raise ValueError(f"Requested columns {columns} are not present in the data")

    n_cols = len(valid_columns)
    n_rows = (n_cols + 2) // 3

    fig, axes = plt.subplots(n_rows, 3, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for i, column in enumerate(valid_columns):
        row = i // 3
        col = i % 3
        ax = axes[row, col] if n_rows > 1 else axes[col]

        ax.plot(data_copy['time'], data_copy[column], linewidth=1.5, alpha=0.8, color='green')
        ax.set_title(f'{column}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel(column)
        ax.grid(True, alpha=0.3)

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))

        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    total_subplots = n_rows * 3
    for i in range(n_cols, total_subplots):
        row = i // 3
        col = i % 3
        if n_rows > 1:
            axes[row, col].set_visible(False)
        else:
            axes[col].set_visible(False)

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_combined_data(health_data, insole_data, health_columns=None, insole_columns=None,
                      save_path='combined_data_plot.png', figsize=(20, 12)):
    """
    Plot combined health and insole data

    Args:
        health_data (pd.DataFrame): health data
        insole_data (pd.DataFrame): insole data
        health_columns (list): health-data columns to plot
        insole_columns (list): insole-data columns to plot
        save_path (str): output path
        figsize (tuple): figure size
    """
    _require_matplotlib()

    if health_columns is None:
        health_columns = ['heart_rate', 'spo2', 'steps', 'calories']

    if insole_columns is None:
        insole_columns = ['step_frequency', 'stride_length', 'plantar_pressure', 'posture_score']

    valid_health_cols = [col for col in health_columns if col in health_data.columns]
    valid_insole_cols = [col for col in insole_columns if col in insole_data.columns]

    if not valid_health_cols and not valid_insole_cols:
        raise ValueError("No valid columns are available to plot")

    health_copy = health_data.copy()
    insole_copy = insole_data.copy()
    health_copy['time'] = pd.to_datetime(health_copy['time'])
    insole_copy['time'] = pd.to_datetime(insole_copy['time'])

    total_cols = len(valid_health_cols) + len(valid_insole_cols)
    n_rows = (total_cols + 1) // 2

    fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    plot_idx = 0

    for column in valid_health_cols:
        row = plot_idx // 2
        col = plot_idx % 2
        ax = axes[row, col] if n_rows > 1 else axes[col]

        ax.plot(health_copy['time'], health_copy[column],
                linewidth=1.5, alpha=0.8, color='blue', label='Health Data')
        ax.set_title(f'Health: {column}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel(column)
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plot_idx += 1

    for column in valid_insole_cols:
        row = plot_idx // 2
        col = plot_idx % 2
        ax = axes[row, col] if n_rows > 1 else axes[col]

        ax.plot(insole_copy['time'], insole_copy[column],
                linewidth=1.5, alpha=0.8, color='green', label='Insole Data')
        ax.set_title(f'Insole: {column}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel(column)
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plot_idx += 1

    total_subplots = n_rows * 2
    for i in range(plot_idx, total_subplots):
        row = i // 2
        col = i % 2
        if n_rows > 1:
            axes[row, col].set_visible(False)
        else:
            axes[col].set_visible(False)

    plt.suptitle('Combined Health and Insole Data', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_health_status_distribution(data, save_path='health_status_distribution.png'):
    """
    Plot the health-state distribution

    Args:
        data (pd.DataFrame): Health data containing a status column
        save_path (str): output path
    """
    _require_matplotlib()
    if 'status' not in data.columns:
        raise ValueError("Data must contain a 'status' column")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.hist(data['status'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_title('Health Status Value Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Health Status Value')
    ax1.set_ylabel('Frequency')
    ax1.axvline(x=0.5, color='orange', linestyle='--', label='Warning Threshold')
    ax1.axvline(x=1.5, color='red', linestyle='--', label='Emergency Threshold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    status_categories = pd.cut(data['status'],
                              bins=[-0.1, 0.5, 1.5, 2.1],
                              labels=['Normal', 'Warning', 'Emergency'])
    status_counts = status_categories.value_counts()

    colors = ['lightgreen', 'orange', 'red']
    ax2.pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90)
    ax2.set_title('Health Status Category Distribution', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
