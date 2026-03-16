"""
2026 Overtake Mode features.
Primary passing tool, replacing DRS.
Requires being within 1s of car ahead at detection point.
Allows +0.5MJ and higher sustained electrical power.
"""

import pandas as pd
import numpy as np


def compute_overtake_mode_conversion_rate(results_df: pd.DataFrame,
                                            team: str,
                                            position_data: pd.DataFrame = None) -> float:
    """
    When OT mode is available, fraction resulting in a position gain.
    Requires position change data correlated with OT mode activation.

    Proxy: positions gained on laps where gap < 1s to car ahead.
    """
    if position_data is None or position_data.empty:
        return np.nan

    # Filter for this team's drivers
    team_data = position_data[position_data['team'] == team]
    if team_data.empty:
        return np.nan

    # Count laps within 1s (OT eligible) vs. position gains
    ot_eligible = team_data[team_data['gap_ahead'] < 1.0]
    if ot_eligible.empty:
        return np.nan

    gains = (ot_eligible['position_change'] > 0).sum()
    return gains / len(ot_eligible)


def compute_overtake_mode_speed_delta(telemetry: pd.DataFrame,
                                        clean_laps: pd.DataFrame,
                                        driver: str,
                                        gap_data: pd.DataFrame = None) -> float:
    """
    Peak speed gain when OT mode active vs. non-OT baseline.
    Proxy: compare peak speeds on laps near other cars vs. clear-air laps.
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

    # Without explicit OT mode flag, return the variability in vmax
    # as a proxy for OT mode speed delta
    return np.max(vmax_values) - np.median(vmax_values)


def compute_overtake_mode_availability(results_df: pd.DataFrame,
                                         team: str,
                                         interval_data: pd.DataFrame = None) -> float:
    """
    Fraction of laps car is within 1s at detection point.
    Measures how often OT mode is available.
    """
    if interval_data is None or interval_data.empty:
        return np.nan

    team_intervals = interval_data[interval_data['team'] == team]
    if team_intervals.empty:
        return np.nan

    within_1s = (team_intervals['gap_ahead'].abs() < 1.0).mean()
    return within_1s


def compute_overtake_mode_defense(results_df: pd.DataFrame,
                                    team: str,
                                    position_data: pd.DataFrame = None) -> float:
    """
    Fraction of OT mode attacks against this car that are repelled.
    Proxy: position holds when car behind is within 1s.
    """
    if position_data is None or position_data.empty:
        return np.nan

    team_data = position_data[position_data['team'] == team]
    if team_data.empty:
        return np.nan

    # Laps where car behind is within 1s (being attacked)
    under_attack = team_data[team_data['gap_behind'] < 1.0]
    if under_attack.empty:
        return np.nan

    held_position = (under_attack['position_change'] >= 0).sum()
    return held_position / len(under_attack)


def compute_all_overtake_features(session, clean_laps: pd.DataFrame,
                                    driver: str, team: str,
                                    results_df: pd.DataFrame = None,
                                    position_data: pd.DataFrame = None,
                                    interval_data: pd.DataFrame = None) -> dict:
    """Compute all Overtake Mode features."""
    try:
        telemetry = session.car_data
    except Exception:
        telemetry = None

    features = {
        'overtake_mode_conversion_rate': compute_overtake_mode_conversion_rate(
            results_df, team, position_data) if results_df is not None else np.nan,
        'overtake_mode_speed_delta': np.nan,
        'overtake_mode_availability_rate': compute_overtake_mode_availability(
            results_df, team, interval_data) if results_df is not None else np.nan,
        'overtake_mode_defense_success': compute_overtake_mode_defense(
            results_df, team, position_data) if results_df is not None else np.nan,
    }

    if telemetry is not None:
        features['overtake_mode_speed_delta'] = compute_overtake_mode_speed_delta(
            telemetry, clean_laps, driver)

    return features
