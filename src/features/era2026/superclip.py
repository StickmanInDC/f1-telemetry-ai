"""
2026 Recharge & Super-Clipping features.
Super-clipping: harvesting at end of straight while at full throttle.
The 2026-defining harvesting behaviour.
"""

import pandas as pd
import numpy as np


def detect_superclip_events(telemetry_lap: pd.DataFrame,
                              throttle_threshold: float = 95,
                              min_duration: float = 0.5,
                              speed_drop_threshold: float = -0.5) -> list[dict]:
    """
    Detect super-clipping: speed plateau at straight end while at full throttle.
    Returns list of (onset_distance, duration, speed_loss) per event.
    """
    if telemetry_lap is None or telemetry_lap.empty:
        return []

    tel = telemetry_lap.copy()

    if 'Throttle' not in tel.columns or 'Speed' not in tel.columns:
        return []

    throttle = tel['Throttle'].values
    speed = tel['Speed'].values
    distance = tel['Distance'].values

    # Compute speed changes
    speed_diff = np.diff(speed, prepend=speed[0])
    full_throttle = throttle >= throttle_threshold

    events = []
    in_event = False
    onset_dist = 0
    speed_at_onset = 0

    for i in range(len(speed)):
        if full_throttle[i] and speed_diff[i] < speed_drop_threshold:
            if not in_event:
                onset_dist = distance[i]
                speed_at_onset = speed[i]
                in_event = True
        elif in_event:
            # Estimate duration from distance and speed
            dist_covered = distance[i] - onset_dist
            avg_speed = (speed_at_onset + speed[i]) / 2
            duration = dist_covered / (avg_speed / 3.6) if avg_speed > 0 else 0

            if duration >= min_duration:
                events.append({
                    'onset_dist': onset_dist,
                    'duration': duration,
                    'speed_loss': speed_at_onset - speed[i],
                    'end_dist': distance[i],
                })
            in_event = False

    return events


def compute_superclip_frequency(telemetry: pd.DataFrame,
                                  clean_laps: pd.DataFrame,
                                  driver: str) -> float:
    """Super-clipping events per lap."""
    driver_laps = clean_laps[clean_laps['Driver'] == driver]
    if driver_laps.empty:
        return np.nan

    counts = []
    for _, lap in driver_laps.iterrows():
        try:
            lap_tel = telemetry.slice_by_lap(lap)
            events = detect_superclip_events(lap_tel)
            counts.append(len(events))
        except Exception:
            continue

    return np.mean(counts) if counts else np.nan


def compute_superclip_harvest_rate(telemetry: pd.DataFrame,
                                     clean_laps: pd.DataFrame,
                                     driver: str,
                                     car_mass_kg: float = 768) -> float:
    """
    Energy recovered per super-clip event (proxy).
    Speed loss x car mass x event duration — approximate yield.
    """
    driver_laps = clean_laps[clean_laps['Driver'] == driver]
    if driver_laps.empty:
        return np.nan

    harvest_rates = []
    for _, lap in driver_laps.iterrows():
        try:
            lap_tel = telemetry.slice_by_lap(lap)
            events = detect_superclip_events(lap_tel)
            for event in events:
                # KE delta = 0.5 * m * (v1^2 - v2^2) in Joules
                v1 = (event['speed_loss'] + 300) / 3.6  # approximate original speed
                v2 = 300 / 3.6
                energy_j = 0.5 * car_mass_kg * abs(v1**2 - v2**2)
                harvest_rates.append(energy_j / 1e6)  # Convert to MJ
        except Exception:
            continue

    return np.mean(harvest_rates) if harvest_rates else np.nan


def compute_superclip_duration(telemetry: pd.DataFrame,
                                 clean_laps: pd.DataFrame,
                                 driver: str) -> float:
    """Average duration of super-clip window per straight (seconds)."""
    driver_laps = clean_laps[clean_laps['Driver'] == driver]
    if driver_laps.empty:
        return np.nan

    durations = []
    for _, lap in driver_laps.iterrows():
        try:
            lap_tel = telemetry.slice_by_lap(lap)
            events = detect_superclip_events(lap_tel)
            durations.extend([e['duration'] for e in events])
        except Exception:
            continue

    return np.mean(durations) if durations else np.nan


def compute_lift_coast_vs_superclip(telemetry: pd.DataFrame,
                                      clean_laps: pd.DataFrame,
                                      driver: str) -> float:
    """
    Preference: Lift-and-Coast vs. super-clip ratio.
    Lift-and-coast: throttle drop <50% without braking at high speed.
    Super-clip: speed drop at full throttle.
    Ratio > 1 means more L&C than superclip.
    """
    driver_laps = clean_laps[clean_laps['Driver'] == driver]
    if driver_laps.empty:
        return np.nan

    lc_count = 0
    sc_count = 0

    for _, lap in driver_laps.iterrows():
        try:
            lap_tel = telemetry.slice_by_lap(lap)
            if lap_tel is None or lap_tel.empty:
                continue

            throttle = lap_tel['Throttle'].values
            speed = lap_tel['Speed'].values
            brake = lap_tel['Brake'].values if 'Brake' in lap_tel.columns else np.zeros(len(throttle))

            # Count lift-and-coast events
            for i in range(len(throttle)):
                if throttle[i] < 50 and brake[i] < 10 and speed[i] > 200:
                    lc_count += 1

            # Count superclip events
            events = detect_superclip_events(lap_tel)
            sc_count += len(events)
        except Exception:
            continue

    if sc_count == 0:
        return np.nan

    return lc_count / sc_count


def compute_all_superclip_features(session, clean_laps: pd.DataFrame,
                                     driver: str) -> dict:
    """Compute all superclip/recharge features for a 2026 driver."""
    try:
        telemetry = session.car_data
    except Exception:
        return {
            'superclip_frequency': np.nan,
            'superclip_harvest_rate': np.nan,
            'superclip_duration': np.nan,
            'lift_coast_vs_superclip_ratio': np.nan,
            'brake_regen_efficiency': np.nan,
            'energy_balance_per_lap': np.nan,
        }

    return {
        'superclip_frequency': compute_superclip_frequency(
            telemetry, clean_laps, driver),
        'superclip_harvest_rate': compute_superclip_harvest_rate(
            telemetry, clean_laps, driver),
        'superclip_duration': compute_superclip_duration(
            telemetry, clean_laps, driver),
        'lift_coast_vs_superclip_ratio': compute_lift_coast_vs_superclip(
            telemetry, clean_laps, driver),
        'brake_regen_efficiency': np.nan,  # Requires field comparison
        'energy_balance_per_lap': np.nan,  # Requires ERS state data from OpenF1
    }
