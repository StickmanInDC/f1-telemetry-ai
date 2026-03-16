"""
Group 6: Track Characterization features.
These describe the circuit, not the car.
Used for K-Means clustering to group circuits by performance archetype.
"""

import pandas as pd
import numpy as np
from scipy.signal import find_peaks

# Manual street circuit labels
STREET_CIRCUITS = {
    'monaco', 'singapore', 'baku', 'las_vegas', 'miami',
    'jeddah', 'melbourne', 'albert_park',
    # FastF1 circuit names (may vary)
    'Monaco', 'Singapore', 'Baku', 'Las Vegas', 'Miami',
    'Jeddah', 'Melbourne', 'Albert Park',
}

# Approximate altitudes (meters above sea level)
CIRCUIT_ALTITUDES = {
    'monza': 162, 'spa': 400, 'silverstone': 153, 'monaco': 7,
    'singapore': 18, 'baku': -28, 'bahrain': 7, 'jeddah': 18,
    'melbourne': 5, 'suzuka': 45, 'austin': 275, 'mexico': 2240,
    'interlagos': 760, 'las_vegas': 610, 'miami': 2, 'montreal': 18,
    'barcelona': 101, 'hungaroring': 264, 'zandvoort': 0, 'imola': 47,
    'lusail': 12, 'yas_marina': 5, 'shanghai': 4, 'red_bull_ring': 677,
}

# Approximate lap lengths (km)
CIRCUIT_LAP_LENGTHS = {
    'monza': 5.793, 'spa': 7.004, 'silverstone': 5.891, 'monaco': 3.337,
    'singapore': 4.940, 'baku': 6.003, 'bahrain': 5.412, 'jeddah': 6.174,
    'melbourne': 5.278, 'suzuka': 5.807, 'austin': 5.513, 'mexico': 4.304,
    'interlagos': 4.309, 'las_vegas': 6.201, 'miami': 5.412, 'montreal': 4.361,
    'barcelona': 4.675, 'hungaroring': 4.381, 'zandvoort': 4.259, 'imola': 4.909,
    'lusail': 5.419, 'yas_marina': 5.281, 'shanghai': 5.451, 'red_bull_ring': 4.318,
}


def compute_pct_full_throttle(telemetry: pd.DataFrame,
                                clean_laps: pd.DataFrame) -> float:
    """Fraction of lap at >95% throttle. Averaged across representative laps."""
    throttle_fracs = []
    for _, lap in clean_laps.head(20).iterrows():
        try:
            lap_tel = telemetry.slice_by_lap(lap)
            if lap_tel is None or lap_tel.empty or 'Throttle' not in lap_tel.columns:
                continue
            throttle = lap_tel['Throttle'].values
            frac = np.mean(throttle > 95)
            throttle_fracs.append(frac)
        except Exception:
            continue
    return np.mean(throttle_fracs) if throttle_fracs else np.nan


def compute_pct_heavy_braking(telemetry: pd.DataFrame,
                                clean_laps: pd.DataFrame) -> float:
    """Fraction of lap at >80% brake pressure."""
    brake_fracs = []
    for _, lap in clean_laps.head(20).iterrows():
        try:
            lap_tel = telemetry.slice_by_lap(lap)
            if lap_tel is None or lap_tel.empty or 'Brake' not in lap_tel.columns:
                continue
            brake = lap_tel['Brake'].values
            frac = np.mean(brake > 80)
            brake_fracs.append(frac)
        except Exception:
            continue
    return np.mean(brake_fracs) if brake_fracs else np.nan


def compute_corner_counts(telemetry: pd.DataFrame,
                            clean_laps: pd.DataFrame) -> dict:
    """
    Count and classify corners from speed trace.
    Fast corners: apex >180 km/h. Slow corners: apex <100 km/h.
    """
    all_corner_speeds = []
    for _, lap in clean_laps.head(10).iterrows():
        try:
            lap_tel = telemetry.slice_by_lap(lap)
            if lap_tel is None or lap_tel.empty:
                continue

            speed = lap_tel['Speed'].values
            inv_speed = -speed
            peaks, _ = find_peaks(inv_speed, distance=50, prominence=20)
            corner_speeds = speed[peaks]
            all_corner_speeds.append(corner_speeds)
        except Exception:
            continue

    if not all_corner_speeds:
        return {'avg_corner_speed': np.nan, 'num_slow_corners': np.nan,
                'num_fast_corners': np.nan}

    # Use the most common lap structure (modal corner count)
    corner_counts = [len(cs) for cs in all_corner_speeds]
    if not corner_counts:
        return {'avg_corner_speed': np.nan, 'num_slow_corners': np.nan,
                'num_fast_corners': np.nan}

    # Average across laps
    avg_speeds = np.concatenate(all_corner_speeds)
    return {
        'avg_corner_speed': np.mean(avg_speeds),
        'num_slow_corners': np.mean([np.sum(cs < 100) for cs in all_corner_speeds]),
        'num_fast_corners': np.mean([np.sum(cs > 180) for cs in all_corner_speeds]),
    }


def compute_track_evolution(clean_laps: pd.DataFrame) -> float:
    """
    Lap time improvement across session from rubber lay-in.
    Linear regression slope of all-driver median lap time vs. lap number.
    """
    if clean_laps.empty:
        return np.nan

    clean = clean_laps.copy()
    clean['lt_seconds'] = clean['LapTime'].dt.total_seconds()

    # Median lap time per lap number across all drivers
    per_lap = clean.groupby('LapNumber')['lt_seconds'].median().reset_index()
    per_lap = per_lap.dropna()

    if len(per_lap) < 5:
        return np.nan

    coeffs = np.polyfit(per_lap['LapNumber'].values,
                         per_lap['lt_seconds'].values, deg=1)
    return coeffs[0]  # Negative = track is getting faster


def compute_energy_recovery_potential(telemetry: pd.DataFrame,
                                       clean_laps: pd.DataFrame,
                                       car_mass_kg: float = 798) -> float:
    """
    Braking energy available per lap.
    Sum of deceleration events x estimated mass x speed delta.
    """
    energy_values = []
    for _, lap in clean_laps.head(10).iterrows():
        try:
            lap_tel = telemetry.slice_by_lap(lap)
            if lap_tel is None or lap_tel.empty:
                continue

            speed = lap_tel['Speed'].values
            brake = lap_tel['Brake'].values if 'Brake' in lap_tel.columns else None

            if brake is None:
                continue

            # Sum kinetic energy change at each braking event
            total_energy = 0
            in_brake = False
            brake_start_speed = 0

            for i in range(len(brake)):
                if brake[i] > 20 and not in_brake:
                    in_brake = True
                    brake_start_speed = speed[i]
                elif brake[i] < 10 and in_brake:
                    in_brake = False
                    speed_end = speed[i]
                    # KE = 0.5 * m * (v1^2 - v2^2), speeds in m/s
                    v1 = brake_start_speed / 3.6
                    v2 = speed_end / 3.6
                    energy = 0.5 * car_mass_kg * (v1**2 - v2**2)
                    total_energy += max(0, energy)

            if total_energy > 0:
                energy_values.append(total_energy / 1e6)  # MJ
        except Exception:
            continue

    return np.mean(energy_values) if energy_values else np.nan


def compute_all_track_features(session, clean_laps: pd.DataFrame,
                                 circuit_name: str = '') -> dict:
    """Compute all Group 6 features for a circuit from a race session."""
    try:
        telemetry = session.car_data
    except Exception:
        telemetry = None

    features = {}
    circuit_key = circuit_name.lower().replace(' ', '_')

    if telemetry is not None:
        features['pct_full_throttle'] = compute_pct_full_throttle(
            telemetry, clean_laps)
        features['pct_heavy_braking'] = compute_pct_heavy_braking(
            telemetry, clean_laps)

        corner_info = compute_corner_counts(telemetry, clean_laps)
        features.update(corner_info)

        features['energy_recovery_potential'] = compute_energy_recovery_potential(
            telemetry, clean_laps)

    features['track_evolution_rate'] = compute_track_evolution(clean_laps)

    # Static features from lookup tables
    features['is_street_circuit'] = 1 if circuit_key in STREET_CIRCUITS or \
        circuit_name in STREET_CIRCUITS else 0
    features['altitude_m'] = CIRCUIT_ALTITUDES.get(circuit_key, np.nan)
    features['lap_length_km'] = CIRCUIT_LAP_LENGTHS.get(circuit_key, np.nan)

    # Surface abrasion proxy (computed from deg rates later in pipeline)
    features['surface_abrasion_proxy'] = np.nan

    return features
