"""
Group 2: Aerodynamic Performance features.
Captures aero setup philosophy and grip characteristics.

NOTE: dirty_air_sensitivity and floor_load_sensitivity are RETIRED in 2026.
"""

import pandas as pd
import numpy as np
from scipy.signal import find_peaks


def classify_corners(speed_trace: np.ndarray, distance_trace: np.ndarray,
                      min_distance: int = 50, min_prominence: float = 20) -> dict:
    """
    Classify corners from a speed trace into fast (>180 km/h) and slow (<100 km/h).
    Returns dict with corner_speeds, fast_corners, slow_corners arrays.
    """
    inv_speed = -speed_trace
    peaks, properties = find_peaks(inv_speed, distance=min_distance,
                                    prominence=min_prominence)
    corner_speeds = speed_trace[peaks]

    return {
        'corner_speeds': corner_speeds,
        'corner_indices': peaks,
        'fast_corners': corner_speeds[corner_speeds > 180],
        'slow_corners': corner_speeds[corner_speeds < 100],
        'medium_corners': corner_speeds[(corner_speeds >= 100) & (corner_speeds <= 180)],
    }


def compute_corner_vs_straight(telemetry: pd.DataFrame, clean_laps: pd.DataFrame,
                                 driver: str) -> dict:
    """
    Downforce/drag tradeoff proxy.
    avg_min_corner_speed / v_max — higher ratio = higher downforce setup.
    """
    driver_laps = clean_laps[clean_laps['Driver'] == driver]
    if driver_laps.empty:
        return {'corner_speed_vs_straight_speed': np.nan, 'v_max': np.nan}

    all_corner_speeds = []
    all_vmax = []

    for _, lap in driver_laps.iterrows():
        try:
            lap_tel = telemetry.slice_by_lap(lap)
            if lap_tel is None or lap_tel.empty:
                continue

            speed = lap_tel['Speed'].values
            distance = lap_tel['Distance'].values

            corners = classify_corners(speed, distance)
            if len(corners['corner_speeds']) > 0:
                all_corner_speeds.extend(corners['corner_speeds'])
            all_vmax.append(speed.max())
        except Exception:
            continue

    if not all_corner_speeds or not all_vmax:
        return {'corner_speed_vs_straight_speed': np.nan, 'v_max': np.nan}

    avg_corner = np.mean(all_corner_speeds)
    v_max = np.mean(all_vmax)

    return {
        'corner_speed_vs_straight_speed': avg_corner / v_max if v_max > 0 else np.nan,
        'v_max': v_max,
        'avg_min_corner_speed': avg_corner,
    }


def compute_high_speed_corner_grip(telemetry: pd.DataFrame,
                                     clean_laps: pd.DataFrame,
                                     driver: str) -> float:
    """
    Aerodynamic grip in fast corners (>180 km/h apex speed).
    Mean minimum speed across corners classified as fast.
    """
    driver_laps = clean_laps[clean_laps['Driver'] == driver]
    if driver_laps.empty:
        return np.nan

    fast_corner_speeds = []
    for _, lap in driver_laps.iterrows():
        try:
            lap_tel = telemetry.slice_by_lap(lap)
            if lap_tel is None or lap_tel.empty:
                continue

            speed = lap_tel['Speed'].values
            distance = lap_tel['Distance'].values
            corners = classify_corners(speed, distance)
            if len(corners['fast_corners']) > 0:
                fast_corner_speeds.extend(corners['fast_corners'])
        except Exception:
            continue

    return np.mean(fast_corner_speeds) if fast_corner_speeds else np.nan


def compute_low_speed_corner_grip(telemetry: pd.DataFrame,
                                    clean_laps: pd.DataFrame,
                                    driver: str) -> float:
    """
    Mechanical grip in slow corners (<100 km/h apex speed).
    Mean minimum speed across corners classified as slow.
    """
    driver_laps = clean_laps[clean_laps['Driver'] == driver]
    if driver_laps.empty:
        return np.nan

    slow_corner_speeds = []
    for _, lap in driver_laps.iterrows():
        try:
            lap_tel = telemetry.slice_by_lap(lap)
            if lap_tel is None or lap_tel.empty:
                continue

            speed = lap_tel['Speed'].values
            distance = lap_tel['Distance'].values
            corners = classify_corners(speed, distance)
            if len(corners['slow_corners']) > 0:
                slow_corner_speeds.extend(corners['slow_corners'])
        except Exception:
            continue

    return np.mean(slow_corner_speeds) if slow_corner_speeds else np.nan


def compute_sector_delta(session, clean_laps: pd.DataFrame, driver: str) -> float:
    """
    High-speed vs. technical sector balance.
    Normalized gap to pole per sector — S1 typically aero-sensitive,
    S3 typically mechanical-sensitive.
    """
    driver_laps = clean_laps[clean_laps['Driver'] == driver]
    if driver_laps.empty:
        return np.nan

    s1_times = driver_laps['Sector1Time'].dropna().dt.total_seconds()
    s3_times = driver_laps['Sector3Time'].dropna().dt.total_seconds()

    if s1_times.empty or s3_times.empty:
        return np.nan

    # Get field-best sector times for normalization
    all_s1 = clean_laps['Sector1Time'].dropna().dt.total_seconds()
    all_s3 = clean_laps['Sector3Time'].dropna().dt.total_seconds()

    if all_s1.empty or all_s3.empty:
        return np.nan

    s1_gap_pct = ((s1_times.median() - all_s1.min()) / all_s1.min()) * 100
    s3_gap_pct = ((s3_times.median() - all_s3.min()) / all_s3.min()) * 100

    return s1_gap_pct - s3_gap_pct


def compute_dirty_air_sensitivity(clean_laps: pd.DataFrame, driver: str,
                                    gap_threshold: float = 2.0) -> float:
    """
    Pace loss when following another car within gap_threshold seconds.
    RETIRE in 2026.
    Requires interval/gap data (from OpenF1 or computed from positions).
    """
    # This requires gap data that must be merged in during pipeline
    # Placeholder — computed when interval data is available
    return np.nan


def compute_all_aero_features(session, clean_laps: pd.DataFrame,
                                driver: str, is_2026: bool = False) -> dict:
    """Compute all Group 2 features for a driver."""
    try:
        telemetry = session.car_data
    except Exception:
        telemetry = None

    features = {}

    if telemetry is not None:
        cs = compute_corner_vs_straight(telemetry, clean_laps, driver)
        features['corner_speed_vs_straight_speed'] = cs['corner_speed_vs_straight_speed']
        features['high_speed_corner_grip'] = compute_high_speed_corner_grip(
            telemetry, clean_laps, driver)
        features['low_speed_corner_grip'] = compute_low_speed_corner_grip(
            telemetry, clean_laps, driver)

    features['sector1_vs_sector3_delta'] = compute_sector_delta(session, clean_laps, driver)

    if not is_2026:
        features['dirty_air_sensitivity'] = compute_dirty_air_sensitivity(
            clean_laps, driver)
        features['floor_load_sensitivity'] = np.nan  # Requires track surface data

    return features
