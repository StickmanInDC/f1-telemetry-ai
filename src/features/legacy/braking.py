"""
Group 4: Braking & Corner Entry features.
Braking mechanics are fully STABLE across eras.
rotation_balance requires recalibration in 2026.
"""

import pandas as pd
import numpy as np
from scipy.signal import find_peaks


def _get_braking_zones(speed: np.ndarray, brake: np.ndarray,
                        distance: np.ndarray) -> list[dict]:
    """
    Identify braking zones from telemetry.
    A braking zone starts when brake > 20% and ends when brake drops below 10%.
    """
    zones = []
    in_zone = False
    zone_start = 0

    for i in range(len(brake)):
        if brake[i] > 20 and not in_zone:
            in_zone = True
            zone_start = i
        elif brake[i] < 10 and in_zone:
            in_zone = False
            if i - zone_start > 5:  # Minimum zone length
                zones.append({
                    'start_idx': zone_start,
                    'end_idx': i,
                    'start_distance': distance[zone_start],
                    'end_distance': distance[i],
                    'entry_speed': speed[zone_start],
                    'min_speed': speed[zone_start:i].min(),
                    'brake_trace': brake[zone_start:i],
                    'speed_trace': speed[zone_start:i],
                })

    return zones


def compute_brake_point_consistency(telemetry: pd.DataFrame,
                                      clean_laps: pd.DataFrame,
                                      driver: str) -> float:
    """
    Std dev of braking point distance across identical corners.
    Lower = more consistent braking.
    """
    driver_laps = clean_laps[clean_laps['Driver'] == driver]
    if driver_laps.empty:
        return np.nan

    # Collect brake point distances per corner (binned by approximate distance)
    corner_brake_points = {}

    for _, lap in driver_laps.iterrows():
        try:
            lap_tel = telemetry.slice_by_lap(lap)
            if lap_tel is None or lap_tel.empty:
                continue

            speed = lap_tel['Speed'].values
            brake = lap_tel['Brake'].values if 'Brake' in lap_tel.columns else None
            distance = lap_tel['Distance'].values

            if brake is None:
                continue

            zones = _get_braking_zones(speed, brake, distance)
            for zone in zones:
                # Bin corners by approximate track position (100m bins)
                corner_bin = int(zone['end_distance'] / 100) * 100
                if corner_bin not in corner_brake_points:
                    corner_brake_points[corner_bin] = []
                corner_brake_points[corner_bin].append(zone['start_distance'])
        except Exception:
            continue

    # Compute std dev of brake points across laps for each corner
    consistency_values = []
    for corner, points in corner_brake_points.items():
        if len(points) >= 3:
            consistency_values.append(np.std(points))

    return np.mean(consistency_values) if consistency_values else np.nan


def compute_trail_braking_index(telemetry: pd.DataFrame,
                                  clean_laps: pd.DataFrame,
                                  driver: str) -> float:
    """
    Degree of brake-steer overlap at corner entry.
    Higher = more trail braking (brake and turn simultaneously).
    Approximated as fraction of braking zone where speed is still decreasing
    while below corner entry speed threshold.
    """
    driver_laps = clean_laps[clean_laps['Driver'] == driver]
    if driver_laps.empty:
        return np.nan

    trail_ratios = []
    for _, lap in driver_laps.iterrows():
        try:
            lap_tel = telemetry.slice_by_lap(lap)
            if lap_tel is None or lap_tel.empty:
                continue

            speed = lap_tel['Speed'].values
            brake = lap_tel['Brake'].values if 'Brake' in lap_tel.columns else None

            if brake is None:
                continue

            distance = lap_tel['Distance'].values
            zones = _get_braking_zones(speed, brake, distance)

            for zone in zones:
                bt = zone['brake_trace']
                total_len = len(bt)
                if total_len < 5:
                    continue
                # Trail braking: brake pressure in final 40% of braking zone
                final_40_start = int(total_len * 0.6)
                final_brake = bt[final_40_start:]
                # Fraction still braking (>5%) in final phase
                trail_frac = np.mean(final_brake > 5)
                trail_ratios.append(trail_frac)
        except Exception:
            continue

    return np.mean(trail_ratios) if trail_ratios else np.nan


def compute_brake_release_rate(telemetry: pd.DataFrame,
                                 clean_laps: pd.DataFrame,
                                 driver: str) -> float:
    """
    Speed of brake pressure drop approaching apex.
    Gradient of brake pressure in final 30% of braking zone.
    Steeper drop = sharper release.
    """
    driver_laps = clean_laps[clean_laps['Driver'] == driver]
    if driver_laps.empty:
        return np.nan

    release_rates = []
    for _, lap in driver_laps.iterrows():
        try:
            lap_tel = telemetry.slice_by_lap(lap)
            if lap_tel is None or lap_tel.empty:
                continue

            speed = lap_tel['Speed'].values
            brake = lap_tel['Brake'].values if 'Brake' in lap_tel.columns else None

            if brake is None:
                continue

            distance = lap_tel['Distance'].values
            zones = _get_braking_zones(speed, brake, distance)

            for zone in zones:
                bt = zone['brake_trace']
                total_len = len(bt)
                if total_len < 5:
                    continue
                # Final 30% of braking zone
                final_start = int(total_len * 0.7)
                final_brake = bt[final_start:]
                if len(final_brake) >= 3:
                    gradient = np.gradient(final_brake).mean()
                    release_rates.append(abs(gradient))  # Magnitude of release
        except Exception:
            continue

    return np.mean(release_rates) if release_rates else np.nan


def compute_all_braking_features(session, clean_laps: pd.DataFrame,
                                   driver: str) -> dict:
    """Compute all Group 4 features for a driver."""
    try:
        telemetry = session.car_data
    except Exception:
        telemetry = None

    features = {}
    if telemetry is not None:
        features['brake_point_consistency'] = compute_brake_point_consistency(
            telemetry, clean_laps, driver)
        features['trail_braking_index'] = compute_trail_braking_index(
            telemetry, clean_laps, driver)
        features['brake_release_rate'] = compute_brake_release_rate(
            telemetry, clean_laps, driver)
    else:
        features['brake_point_consistency'] = np.nan
        features['trail_braking_index'] = np.nan
        features['brake_release_rate'] = np.nan

    # rotation_balance requires steering angle data — not always available
    features['rotation_balance'] = np.nan

    return features
