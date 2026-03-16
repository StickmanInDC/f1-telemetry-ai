"""
2026 Turbo Architecture & Launch Characteristics features.
Captures the small vs. large turbo tradeoff between PU suppliers.
Features are PU supplier-level — propagate to all customer teams.
"""

import pandas as pd
import numpy as np


def compute_turbo_spool_proxy(telemetry: pd.DataFrame,
                                clean_laps: pd.DataFrame,
                                driver: str, grid_pos: int) -> float:
    """
    How quickly PU reaches full torque from standstill.
    Speed at 50m/100m/150m from start line normalized by grid position.
    """
    # Use lap 1 telemetry for race start analysis
    all_laps = clean_laps  # Need raw laps including lap 1
    lap1 = all_laps[
        (all_laps['Driver'] == driver) &
        (all_laps['LapNumber'] == 1)
    ]

    if lap1.empty:
        return np.nan

    try:
        lap_tel = telemetry.slice_by_lap(lap1.iloc[0])
        if lap_tel is None or lap_tel.empty:
            return np.nan

        speed = lap_tel['Speed'].values
        distance = lap_tel['Distance'].values

        speed_50 = speed[distance <= 50].max() if (distance <= 50).any() else np.nan
        speed_100 = speed[distance <= 100].max() if (distance <= 100).any() else np.nan
        speed_150 = speed[distance <= 150].max() if (distance <= 150).any() else np.nan

        # Average speed at checkpoints, normalized by grid position
        # Further back on grid = lower expected speed
        raw_score = np.nanmean([speed_50, speed_100, speed_150])
        # Grid position penalty: ~2 km/h per position back
        normalized = raw_score + (grid_pos - 1) * 2

        return normalized
    except Exception:
        return np.nan


def compute_launch_positions_gained(results_df: pd.DataFrame, team: str) -> float:
    """
    Grid positions gained or lost by end of lap 1.
    Averaged across season, excluding DNF-on-lap-1 events.
    """
    if results_df is None or results_df.empty:
        return np.nan

    team_results = results_df[results_df['constructor_name'] == team]
    if team_results.empty:
        # Try constructor_id
        team_results = results_df[results_df['constructor_id'] == team.lower()]

    if team_results.empty:
        return np.nan

    # Can only compute from grid vs position after lap 1
    # Use grid vs final position as proxy when lap 1 position not available
    valid = team_results[
        (team_results['grid'] > 0) &
        (team_results['laps_completed'] > 0)
    ]

    if valid.empty:
        return np.nan

    # Approximate lap 1 gains from grid vs final position is imprecise;
    # better data comes from OpenF1 position data
    positions_gained = valid['grid'] - valid['position'].astype(float)
    return positions_gained.mean()


def compute_low_speed_corner_exit_pace(telemetry: pd.DataFrame,
                                         clean_laps: pd.DataFrame,
                                         driver: str) -> float:
    """
    Acceleration out of slow corners — secondary turbo response signal.
    Speed delta from apex to 200m after apex for corners <80 km/h.
    """
    driver_laps = clean_laps[clean_laps['Driver'] == driver]
    if driver_laps.empty:
        return np.nan

    exit_deltas = []
    for _, lap in driver_laps.iterrows():
        try:
            lap_tel = telemetry.slice_by_lap(lap)
            if lap_tel is None or lap_tel.empty:
                continue

            speed = lap_tel['Speed'].values
            distance = lap_tel['Distance'].values

            # Find slow corner apexes
            for i in range(1, len(speed) - 1):
                if (speed[i] < speed[i-1] and speed[i] < speed[i+1] and
                        speed[i] < 80):
                    apex_dist = distance[i]
                    # Speed 200m after
                    mask_200m = (distance >= apex_dist + 180) & (distance <= apex_dist + 220)
                    if mask_200m.any():
                        speed_at_200m = speed[mask_200m].mean()
                        exit_deltas.append(speed_at_200m - speed[i])
        except Exception:
            continue

    return np.mean(exit_deltas) if exit_deltas else np.nan


def compute_high_speed_straight_deficit(telemetry: pd.DataFrame,
                                          clean_laps: pd.DataFrame,
                                          driver: str,
                                          field_vmax: float = None) -> float:
    """
    V-max shortfall on long straights vs. field.
    Expect Ferrari-powered cars lower at Monza/Spa (small turbo cost).
    """
    driver_laps = clean_laps[clean_laps['Driver'] == driver]
    if driver_laps.empty:
        return np.nan

    vmax_values = []
    for _, lap in driver_laps.iterrows():
        try:
            lap_tel = telemetry.slice_by_lap(lap)
            if lap_tel is not None and not lap_tel.empty:
                vmax_values.append(lap_tel['Speed'].max())
        except Exception:
            continue

    if not vmax_values:
        return np.nan

    car_vmax = np.mean(vmax_values)

    if field_vmax is not None:
        return car_vmax - field_vmax
    return car_vmax


def compute_superclip_onset_distance(telemetry: pd.DataFrame,
                                       clean_laps: pd.DataFrame,
                                       driver: str) -> float:
    """
    How early in a straight the car reaches super-clip threshold.
    Earlier onset may signal lower PU power ceiling.
    """
    from .superclip import detect_superclip_events

    driver_laps = clean_laps[clean_laps['Driver'] == driver]
    if driver_laps.empty:
        return np.nan

    onset_distances = []
    for _, lap in driver_laps.iterrows():
        try:
            lap_tel = telemetry.slice_by_lap(lap)
            events = detect_superclip_events(lap_tel)
            for event in events:
                onset_distances.append(event['onset_dist'])
        except Exception:
            continue

    return np.mean(onset_distances) if onset_distances else np.nan


def compute_all_turbo_features(session, clean_laps: pd.DataFrame,
                                 driver: str, team: str,
                                 grid_pos: int = 10,
                                 results_df: pd.DataFrame = None,
                                 field_vmax: float = None) -> dict:
    """Compute all turbo/launch features for a 2026 driver."""
    try:
        telemetry = session.car_data
    except Exception:
        telemetry = None

    features = {
        'turbo_spool_proxy': np.nan,
        'launch_positions_gained': np.nan,
        'launch_vs_quali_delta': np.nan,
        'pitstop_exit_acceleration': np.nan,
        'low_speed_corner_exit_pace': np.nan,
        'high_speed_straight_deficit': np.nan,
        'superclip_onset_distance': np.nan,
    }

    if telemetry is not None:
        features['turbo_spool_proxy'] = compute_turbo_spool_proxy(
            telemetry, clean_laps, driver, grid_pos)
        features['low_speed_corner_exit_pace'] = compute_low_speed_corner_exit_pace(
            telemetry, clean_laps, driver)
        features['high_speed_straight_deficit'] = compute_high_speed_straight_deficit(
            telemetry, clean_laps, driver, field_vmax)
        features['superclip_onset_distance'] = compute_superclip_onset_distance(
            telemetry, clean_laps, driver)

    if results_df is not None:
        features['launch_positions_gained'] = compute_launch_positions_gained(
            results_df, team)

    return features
