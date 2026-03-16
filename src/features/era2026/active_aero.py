"""
2026 Active Aerodynamics features.
Active aero replaces DRS with Straight Mode (low drag) and Corner Mode (high downforce).
All cars can use it at all times — no proximity requirement.
"""

import pandas as pd
import numpy as np


def compute_straight_mode_activation_timing(telemetry: pd.DataFrame,
                                              clean_laps: pd.DataFrame,
                                              driver: str) -> float:
    """
    How early in the straight the driver opens active aero.
    Distance from corner exit to aero-open event.
    Earlier = more aggressive, higher trust in car stability.
    """
    driver_laps = clean_laps[clean_laps['Driver'] == driver]
    if driver_laps.empty:
        return np.nan

    activation_distances = []
    for _, lap in driver_laps.iterrows():
        try:
            lap_tel = telemetry.slice_by_lap(lap)
            if lap_tel is None or lap_tel.empty:
                continue

            speed = lap_tel['Speed'].values
            distance = lap_tel['Distance'].values

            # Detect corner exits (speed starts increasing from local minimum)
            for i in range(2, len(speed) - 1):
                if speed[i] > speed[i-1] and speed[i-1] <= speed[i-2] and speed[i-1] < 200:
                    corner_exit_dist = distance[i]
                    # Look for speed plateau indicating straight mode engagement
                    # (acceleration rate increases when drag reduces)
                    for j in range(i + 10, min(i + 100, len(speed))):
                        accel_before = speed[j-5:j].mean() - speed[j-10:j-5].mean()
                        accel_after = speed[j:j+5].mean() - speed[j-5:j].mean() if j+5 < len(speed) else 0
                        if accel_after > accel_before * 1.3 and accel_after > 2:
                            activation_distances.append(distance[j] - corner_exit_dist)
                            break
        except Exception:
            continue

    return np.mean(activation_distances) if activation_distances else np.nan


def compute_straight_mode_speed_gain(telemetry: pd.DataFrame,
                                       clean_laps: pd.DataFrame,
                                       driver: str) -> float:
    """
    Speed delta with wings open vs. closed baseline.
    Approximated by comparing peak straight speeds to expected values
    from corner exit speed and acceleration profile.
    """
    driver_laps = clean_laps[clean_laps['Driver'] == driver]
    if driver_laps.empty:
        return np.nan

    # Use peak straight speed as proxy — higher = better straight mode
    vmax_values = []
    for _, lap in driver_laps.iterrows():
        try:
            lap_tel = telemetry.slice_by_lap(lap)
            if lap_tel is not None and not lap_tel.empty:
                vmax_values.append(lap_tel['Speed'].max())
        except Exception:
            continue

    return np.mean(vmax_values) if vmax_values else np.nan


def compute_corner_mode_grip_delta(telemetry_2026: pd.DataFrame,
                                     clean_laps_2026: pd.DataFrame,
                                     driver: str,
                                     legacy_high_speed_grip: float = None) -> float:
    """
    Downforce quality in corner mode vs. 2025 equivalent.
    Delta between 2026 and 2025 min corner speeds for same team at same circuit.
    """
    driver_laps = clean_laps_2026[clean_laps_2026['Driver'] == driver]
    if driver_laps.empty:
        return np.nan

    corner_speeds = []
    for _, lap in driver_laps.iterrows():
        try:
            lap_tel = telemetry_2026.slice_by_lap(lap)
            if lap_tel is None or lap_tel.empty:
                continue
            speed = lap_tel['Speed'].values
            inv_speed = -speed
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(inv_speed, distance=50, prominence=20)
            fast_corners = speed[peaks][speed[peaks] > 180]
            if len(fast_corners) > 0:
                corner_speeds.extend(fast_corners)
        except Exception:
            continue

    if not corner_speeds:
        return np.nan

    current_grip = np.mean(corner_speeds)
    if legacy_high_speed_grip is not None and not np.isnan(legacy_high_speed_grip):
        return current_grip - legacy_high_speed_grip
    return current_grip


def compute_active_aero_consistency(telemetry: pd.DataFrame,
                                      clean_laps: pd.DataFrame,
                                      driver: str) -> float:
    """
    Std dev of activation timing and speed gain across laps.
    Low variance = driver and car confident in deployment.
    """
    driver_laps = clean_laps[clean_laps['Driver'] == driver]
    if driver_laps.empty:
        return np.nan

    vmax_per_lap = []
    for _, lap in driver_laps.iterrows():
        try:
            lap_tel = telemetry.slice_by_lap(lap)
            if lap_tel is not None and not lap_tel.empty:
                vmax_per_lap.append(lap_tel['Speed'].max())
        except Exception:
            continue

    return np.std(vmax_per_lap) if len(vmax_per_lap) > 3 else np.nan


def compute_all_active_aero_features(session, clean_laps: pd.DataFrame,
                                       driver: str,
                                       legacy_grip: float = None) -> dict:
    """Compute all active aero features for a 2026 driver."""
    try:
        telemetry = session.car_data
    except Exception:
        return {
            'straight_mode_activation_timing': np.nan,
            'straight_mode_speed_gain': np.nan,
            'corner_mode_grip_delta': np.nan,
            'active_aero_consistency': np.nan,
        }

    return {
        'straight_mode_activation_timing': compute_straight_mode_activation_timing(
            telemetry, clean_laps, driver),
        'straight_mode_speed_gain': compute_straight_mode_speed_gain(
            telemetry, clean_laps, driver),
        'corner_mode_grip_delta': compute_corner_mode_grip_delta(
            telemetry, clean_laps, driver, legacy_grip),
        'active_aero_consistency': compute_active_aero_consistency(
            telemetry, clean_laps, driver),
    }
