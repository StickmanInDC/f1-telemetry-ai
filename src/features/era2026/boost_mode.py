"""
2026 Boost Mode features.
Boost Mode allows manual ERS deployment at any point on circuit.
No proximity requirement. Effectiveness depends on harvested energy.
"""

import pandas as pd
import numpy as np


def detect_boost_events(telemetry_lap: pd.DataFrame,
                         speed_threshold: float = 5.0) -> list[dict]:
    """
    Detect boost deployment events from telemetry.
    Boost events are identified as sudden speed increases above the
    expected acceleration profile on straights.

    A boost event shows as an anomalous acceleration spike while
    already at high speed (>200 km/h).
    """
    if telemetry_lap is None or telemetry_lap.empty:
        return []

    speed = telemetry_lap['Speed'].values
    distance = telemetry_lap['Distance'].values
    throttle = telemetry_lap['Throttle'].values if 'Throttle' in telemetry_lap.columns else None

    events = []

    if throttle is None:
        return events

    # Look for acceleration spikes while already at high speed on full throttle
    for i in range(10, len(speed) - 10):
        if speed[i] > 200 and throttle[i] > 95:
            # Compare acceleration in this window vs. surrounding
            accel_here = speed[i+5] - speed[i] if i + 5 < len(speed) else 0
            accel_before = speed[i] - speed[i-5]
            accel_jump = accel_here - accel_before

            if accel_jump > speed_threshold:
                events.append({
                    'distance': distance[i],
                    'speed_at_activation': speed[i],
                    'speed_gain': accel_here,
                    'track_position_pct': distance[i] / distance[-1] if distance[-1] > 0 else 0,
                })

    return events


def compute_boost_deployment_pattern(telemetry: pd.DataFrame,
                                       clean_laps: pd.DataFrame,
                                       driver: str) -> float:
    """
    Where on the lap Boost is typically deployed.
    Shannon entropy of deployment positions — high = spread deployment,
    low = concentrated at one zone.
    """
    driver_laps = clean_laps[clean_laps['Driver'] == driver]
    if driver_laps.empty:
        return np.nan

    all_positions = []
    for _, lap in driver_laps.iterrows():
        try:
            lap_tel = telemetry.slice_by_lap(lap)
            events = detect_boost_events(lap_tel)
            all_positions.extend([e['track_position_pct'] for e in events])
        except Exception:
            continue

    if len(all_positions) < 3:
        return np.nan

    # Bin positions into 10 track zones and compute Shannon entropy
    hist, _ = np.histogram(all_positions, bins=10, range=(0, 1))
    hist = hist / hist.sum() if hist.sum() > 0 else hist
    # Add small epsilon to avoid log(0)
    hist = hist + 1e-10
    entropy = -np.sum(hist * np.log2(hist))

    return entropy


def compute_boost_frequency(telemetry: pd.DataFrame,
                              clean_laps: pd.DataFrame,
                              driver: str) -> float:
    """Average number of boost activations per lap."""
    driver_laps = clean_laps[clean_laps['Driver'] == driver]
    if driver_laps.empty:
        return np.nan

    counts = []
    for _, lap in driver_laps.iterrows():
        try:
            lap_tel = telemetry.slice_by_lap(lap)
            events = detect_boost_events(lap_tel)
            counts.append(len(events))
        except Exception:
            continue

    return np.mean(counts) if counts else np.nan


def compute_boost_energy_per_activation(telemetry: pd.DataFrame,
                                          clean_laps: pd.DataFrame,
                                          driver: str,
                                          max_mguk_power_kw: float = 350) -> float:
    """
    Average MJ deployed per boost event.
    Proxy: event duration x estimated power output.
    """
    driver_laps = clean_laps[clean_laps['Driver'] == driver]
    if driver_laps.empty:
        return np.nan

    energies = []
    for _, lap in driver_laps.iterrows():
        try:
            lap_tel = telemetry.slice_by_lap(lap)
            events = detect_boost_events(lap_tel)
            for event in events:
                # Approximate duration from speed gain / acceleration
                duration_s = event['speed_gain'] / 20  # rough estimate
                energy_mj = max_mguk_power_kw * duration_s / 1000
                energies.append(energy_mj)
        except Exception:
            continue

    return np.mean(energies) if energies else np.nan


def compute_all_boost_features(session, clean_laps: pd.DataFrame,
                                 driver: str) -> dict:
    """Compute all Boost Mode features for a 2026 driver."""
    try:
        telemetry = session.car_data
    except Exception:
        return {
            'boost_deployment_pattern': np.nan,
            'boost_frequency_per_lap': np.nan,
            'boost_energy_per_activation': np.nan,
            'boost_defensive_vs_offensive_ratio': np.nan,
        }

    return {
        'boost_deployment_pattern': compute_boost_deployment_pattern(
            telemetry, clean_laps, driver),
        'boost_frequency_per_lap': compute_boost_frequency(
            telemetry, clean_laps, driver),
        'boost_energy_per_activation': compute_boost_energy_per_activation(
            telemetry, clean_laps, driver),
        'boost_defensive_vs_offensive_ratio': np.nan,  # Requires position data merge
    }
