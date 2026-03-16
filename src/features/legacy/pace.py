"""
Group 5: Qualifying vs. Race Pace features.
All STABLE — carry into 2026 with recalculated values.
"""

import pandas as pd
import numpy as np


def compute_race_pace_gap(race_laps: pd.DataFrame) -> pd.DataFrame:
    """
    Normalized race pace gap to fastest team for each team.
    Primary target variable.
    Strips first 4 laps of each stint to remove outlap noise.
    """
    clean = race_laps[race_laps['IsAccurate'] == True].copy()
    if 'TyreLife' in clean.columns:
        clean = clean[clean['TyreLife'] > 4]
    elif 'LapNumber' in clean.columns:
        # Fallback: exclude first 4 laps of race
        clean = clean[clean['LapNumber'] > 4]

    if clean.empty:
        return pd.DataFrame()

    team_pace = (
        clean.groupby('Team')['LapTime']
        .apply(lambda x: x.dt.total_seconds().median())
        .reset_index(name='median_pace')
    )

    fastest = team_pace['median_pace'].min()
    team_pace['pace_gap_pct'] = ((team_pace['median_pace'] - fastest) / fastest) * 100

    return team_pace


def compute_quali_pace_gap(quali_session) -> pd.DataFrame:
    """
    Single-lap gap to pole as normalized % of pole time.
    Uses fastest Q3 time (or best overall) per team.
    """
    laps = quali_session.laps.copy()
    if laps.empty:
        return pd.DataFrame()

    # Get best lap per driver
    best_laps = (
        laps.dropna(subset=['LapTime'])
        .groupby('Driver')['LapTime']
        .min()
        .reset_index()
    )
    best_laps['LapTime_s'] = best_laps['LapTime'].dt.total_seconds()

    # Merge team info
    driver_info = laps[['Driver', 'Team']].drop_duplicates()
    best_laps = best_laps.merge(driver_info, on='Driver', how='left')

    # Team-average (average of both drivers' best)
    team_pace = (
        best_laps.groupby('Team')['LapTime_s']
        .mean()
        .reset_index(name='mean_quali_time')
    )

    pole_time = team_pace['mean_quali_time'].min()
    team_pace['quali_pace_gap_pct'] = (
        (team_pace['mean_quali_time'] - pole_time) / pole_time * 100
    )

    return team_pace


def compute_quali_vs_race_delta(quali_gap_pct: float, race_gap_pct: float) -> float:
    """
    Whether car is a quali specialist or a race car.
    Positive = better in race trim than qualifying.
    """
    if np.isnan(quali_gap_pct) or np.isnan(race_gap_pct):
        return np.nan
    return quali_gap_pct - race_gap_pct


def compute_setup_sensitivity(team_pace_gaps: list[float]) -> float:
    """
    Variance in performance weekend to weekend.
    Std dev of race_pace_gap_pct across all rounds in a season.
    High variance = sensitive or complex setup window.
    """
    valid_gaps = [g for g in team_pace_gaps if not np.isnan(g)]
    if len(valid_gaps) < 3:
        return np.nan
    return np.std(valid_gaps)


def compute_wet_vs_dry_delta(wet_gaps: list[float],
                               dry_gaps: list[float]) -> float:
    """
    Pace change in wet vs. dry conditions.
    Requires weather classification per session.
    """
    valid_wet = [g for g in wet_gaps if not np.isnan(g)]
    valid_dry = [g for g in dry_gaps if not np.isnan(g)]

    if not valid_wet or not valid_dry:
        return np.nan

    return np.mean(valid_wet) - np.mean(valid_dry)


def compute_all_pace_features(race_session, quali_session,
                                clean_race_laps: pd.DataFrame,
                                team: str) -> dict:
    """
    Compute all Group 5 features for a team at a specific event.
    """
    features = {}

    # Race pace gap
    race_gaps = compute_race_pace_gap(clean_race_laps)
    if not race_gaps.empty and team in race_gaps['Team'].values:
        team_row = race_gaps[race_gaps['Team'] == team].iloc[0]
        features['race_pace_gap_pct'] = team_row['pace_gap_pct']
    else:
        features['race_pace_gap_pct'] = np.nan

    # Qualifying pace gap
    if quali_session is not None:
        quali_gaps = compute_quali_pace_gap(quali_session)
        if not quali_gaps.empty and team in quali_gaps['Team'].values:
            team_row = quali_gaps[quali_gaps['Team'] == team].iloc[0]
            features['quali_pace_gap_pct'] = team_row['quali_pace_gap_pct']
        else:
            features['quali_pace_gap_pct'] = np.nan
    else:
        features['quali_pace_gap_pct'] = np.nan

    # Quali vs race delta
    features['quali_vs_race_delta'] = compute_quali_vs_race_delta(
        features['quali_pace_gap_pct'], features['race_pace_gap_pct'])

    # setup_sensitivity and wet_vs_dry_delta are season-level features
    # computed in the pipeline after all races are processed
    features['setup_sensitivity'] = np.nan
    features['wet_vs_dry_delta'] = np.nan

    return features
