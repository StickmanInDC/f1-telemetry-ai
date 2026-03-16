"""
Feature Pipeline Orchestrator.
Builds the complete feature table from raw data for both legacy and 2026 eras.
Applies team-averaging rule: all car-level features averaged across both drivers.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

from ..ingestion.fastf1_loader import enable_cache, load_session, get_clean_laps, load_season_races
from ..ingestion.ergast_client import build_results_with_dnf_classification
from .legacy.tires import compute_all_tire_features
from .legacy.aero import compute_all_aero_features
from .legacy.braking import compute_all_braking_features
from .legacy.pace import compute_race_pace_gap, compute_quali_pace_gap, compute_quali_vs_race_delta
from .legacy.power_unit import compute_all_pu_features
from .legacy.track import compute_all_track_features
from .track_clustering import cluster_circuits

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / 'data' / 'processed'


def build_legacy_features(years: list[int] = None,
                            save: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the complete legacy feature table for the specified seasons.

    Returns:
        (team_features_df, track_features_df)
        team_features_df: One row per team per circuit per season
        track_features_df: One row per circuit (aggregated across seasons)
    """
    if years is None:
        years = [2022, 2023, 2024, 2025]

    enable_cache()

    all_team_features = []
    all_track_features = []
    race_pace_by_team = {}  # {(season, team): [pace_gaps]} for setup_sensitivity

    # Get DNF data for reliability context
    results_df = build_results_with_dnf_classification(years)

    for year in years:
        logger.info(f"Processing {year} season...")
        races = load_season_races(year)

        for round_num, event_name in races:
            logger.info(f"  {year} R{round_num}: {event_name}")

            try:
                race_session = load_session(year, round_num, 'R')
                clean_laps = get_clean_laps(race_session)
            except Exception as e:
                logger.warning(f"  SKIP race: {e}")
                continue

            if clean_laps.empty:
                logger.warning(f"  No clean laps for {year} R{round_num}")
                continue

            # Load qualifying session
            quali_session = None
            try:
                quali_session = load_session(year, round_num, 'Q')
            except Exception:
                logger.warning(f"  Could not load qualifying for {year} R{round_num}")

            # --- Track features (circuit-level) ---
            circuit_name = event_name
            track_feats = compute_all_track_features(
                race_session, clean_laps, circuit_name)
            track_feats['circuit'] = circuit_name
            track_feats['season'] = year
            track_feats['round'] = round_num
            all_track_features.append(track_feats)

            # --- Team features (team-averaged) ---
            teams = clean_laps['Team'].unique()
            race_gaps = compute_race_pace_gap(clean_laps)
            quali_gaps = compute_quali_pace_gap(quali_session) if quali_session else pd.DataFrame()

            for team in teams:
                team_drivers = clean_laps[clean_laps['Team'] == team]['Driver'].unique()
                if len(team_drivers) == 0:
                    continue

                # Compute features per driver then average (team-averaging rule)
                driver_features = []
                for driver in team_drivers:
                    feats = {}

                    # Group 1: Power Unit
                    pu_feats = compute_all_pu_features(race_session, clean_laps, driver)
                    feats.update(pu_feats)

                    # Group 2: Aero
                    aero_feats = compute_all_aero_features(
                        race_session, clean_laps, driver)
                    feats.update(aero_feats)

                    # Group 3: Tires
                    tire_feats = compute_all_tire_features(clean_laps, driver)
                    feats.update(tire_feats)

                    # Group 4: Braking
                    brake_feats = compute_all_braking_features(
                        race_session, clean_laps, driver)
                    feats.update(brake_feats)

                    driver_features.append(feats)

                # Average across both drivers
                team_feats = _average_driver_features(driver_features)

                # Group 5: Pace (already team-level)
                if not race_gaps.empty and team in race_gaps['Team'].values:
                    team_row = race_gaps[race_gaps['Team'] == team].iloc[0]
                    team_feats['race_pace_gap_pct'] = team_row['pace_gap_pct']
                else:
                    team_feats['race_pace_gap_pct'] = np.nan

                if not quali_gaps.empty and team in quali_gaps['Team'].values:
                    q_row = quali_gaps[quali_gaps['Team'] == team].iloc[0]
                    team_feats['quali_pace_gap_pct'] = q_row['quali_pace_gap_pct']
                else:
                    team_feats['quali_pace_gap_pct'] = np.nan

                team_feats['quali_vs_race_delta'] = compute_quali_vs_race_delta(
                    team_feats.get('quali_pace_gap_pct', np.nan),
                    team_feats.get('race_pace_gap_pct', np.nan))

                # Metadata
                team_feats['team'] = team
                team_feats['circuit'] = circuit_name
                team_feats['season'] = year
                team_feats['round'] = round_num

                all_team_features.append(team_feats)

                # Track pace gaps for setup_sensitivity
                key = (year, team)
                if key not in race_pace_by_team:
                    race_pace_by_team[key] = []
                race_pace_by_team[key].append(team_feats.get('race_pace_gap_pct', np.nan))

    # Build DataFrames
    team_features_df = pd.DataFrame(all_team_features)
    track_features_df = pd.DataFrame(all_track_features)

    if team_features_df.empty:
        logger.warning("No team features computed!")
        return team_features_df, track_features_df

    # Compute season-level features
    for (season, team), pace_gaps in race_pace_by_team.items():
        valid_gaps = [g for g in pace_gaps if not np.isnan(g)]
        sensitivity = np.std(valid_gaps) if len(valid_gaps) >= 3 else np.nan
        mask = (team_features_df['season'] == season) & (team_features_df['team'] == team)
        team_features_df.loc[mask, 'setup_sensitivity'] = sensitivity

    # Cluster circuits
    if not track_features_df.empty:
        # Aggregate track features across seasons (median)
        track_agg = track_features_df.groupby('circuit').median(numeric_only=True).reset_index()
        track_agg, _, _ = cluster_circuits(track_agg, k=5)

        # Map cluster assignments back to team features
        circuit_cluster_map = dict(zip(track_agg['circuit'], track_agg['circuit_cluster']))
        team_features_df['circuit_cluster'] = team_features_df['circuit'].map(circuit_cluster_map)
        track_features_df['circuit_cluster'] = track_features_df['circuit'].map(circuit_cluster_map)

    if save:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        team_features_df.to_parquet(DATA_DIR / 'legacy_team_features.parquet', index=False)
        track_features_df.to_parquet(DATA_DIR / 'legacy_track_features.parquet', index=False)
        logger.info(f"Saved {len(team_features_df)} team feature rows, "
                    f"{len(track_features_df)} track feature rows")

    return team_features_df, track_features_df


def build_2026_features(team_features_legacy: pd.DataFrame = None,
                          races_available: int = None,
                          save: bool = True) -> pd.DataFrame:
    """
    Build 2026 feature table combining stable legacy features with new 2026 features.
    For early season, relies heavily on priors.

    Returns: DataFrame with combined feature matrix for 2026.
    """
    from .era2026.active_aero import compute_all_active_aero_features
    from .era2026.boost_mode import compute_all_boost_features
    from .era2026.overtake_mode import compute_all_overtake_features
    from .era2026.superclip import compute_all_superclip_features
    from .era2026.turbo_launch import compute_all_turbo_features
    from .era2026.reliability import compute_reliability_features
    from ..models.priors import get_pu_priors

    enable_cache()

    year = 2026
    all_features = []
    results_df = build_results_with_dnf_classification([2026])
    pu_priors = get_pu_priors()

    races = load_season_races(year)
    if races_available:
        races = races[:races_available]

    for round_num, event_name in races:
        logger.info(f"  2026 R{round_num}: {event_name}")

        try:
            race_session = load_session(year, round_num, 'R')
            clean_laps = get_clean_laps(race_session)
        except Exception as e:
            logger.warning(f"  SKIP: {e}")
            continue

        if clean_laps.empty:
            continue

        teams = clean_laps['Team'].unique()

        for team in teams:
            team_drivers = clean_laps[clean_laps['Team'] == team]['Driver'].unique()
            if len(team_drivers) == 0:
                continue

            driver_features = []
            for driver in team_drivers:
                feats = {}

                # Stable legacy features (recalculated on 2026 data)
                tire_feats = compute_all_tire_features(clean_laps, driver)
                feats.update(tire_feats)

                brake_feats = compute_all_braking_features(
                    race_session, clean_laps, driver)
                feats.update(brake_feats)

                aero_feats = compute_all_aero_features(
                    race_session, clean_laps, driver, is_2026=True)
                feats.update(aero_feats)

                pu_feats = compute_all_pu_features(
                    race_session, clean_laps, driver, is_2026=True)
                feats.update(pu_feats)

                # New 2026 features
                active_aero = compute_all_active_aero_features(
                    race_session, clean_laps, driver)
                feats.update(active_aero)

                boost = compute_all_boost_features(
                    race_session, clean_laps, driver)
                feats.update(boost)

                superclip = compute_all_superclip_features(
                    race_session, clean_laps, driver)
                feats.update(superclip)

                grid_pos = 10  # Default; override from results_df
                turbo = compute_all_turbo_features(
                    race_session, clean_laps, driver, team,
                    grid_pos, results_df)
                feats.update(turbo)

                driver_features.append(feats)

            team_feats = _average_driver_features(driver_features)

            # Reliability features (team-level)
            if not results_df.empty:
                reliability = compute_reliability_features(results_df, team)
                team_feats.update(reliability)

            # PU priors
            for supplier, info in pu_priors.items():
                if team in info['teams']:
                    team_feats['turbo_size_prior'] = info['turbo_size']
                    team_feats['reliability_prior'] = info['reliability_prior']
                    break

            # Track features for interaction terms
            track_feats = compute_all_track_features(
                race_session, clean_laps, event_name)
            team_feats['track_power_sensitivity_interaction'] = (
                team_feats.get('turbo_spool_proxy', 0) *
                track_feats.get('pct_full_throttle', 0.5)
            )

            # Target variable
            from .legacy.pace import compute_race_pace_gap
            race_gaps = compute_race_pace_gap(clean_laps)
            if not race_gaps.empty and team in race_gaps['Team'].values:
                team_feats['race_pace_gap_pct'] = race_gaps[
                    race_gaps['Team'] == team].iloc[0]['pace_gap_pct']
            else:
                team_feats['race_pace_gap_pct'] = np.nan

            team_feats['team'] = team
            team_feats['circuit'] = event_name
            team_feats['season'] = year
            team_feats['round'] = round_num

            all_features.append(team_feats)

    features_df = pd.DataFrame(all_features)

    if save and not features_df.empty:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        features_df.to_parquet(DATA_DIR / '2026_team_features.parquet', index=False)
        logger.info(f"Saved {len(features_df)} 2026 feature rows")

    return features_df


def _average_driver_features(driver_features: list[dict]) -> dict:
    """
    Average numeric features across drivers (team-averaging rule).
    Non-numeric values take the first driver's value.
    """
    if not driver_features:
        return {}
    if len(driver_features) == 1:
        return driver_features[0].copy()

    averaged = {}
    all_keys = set()
    for df in driver_features:
        all_keys.update(df.keys())

    for key in all_keys:
        values = [d.get(key) for d in driver_features if key in d]
        numeric_values = [v for v in values if isinstance(v, (int, float)) and not np.isnan(v)]

        if numeric_values:
            averaged[key] = np.mean(numeric_values)
        elif values:
            averaged[key] = values[0]
        else:
            averaged[key] = np.nan

    return averaged


def load_cached_features(era: str = 'legacy') -> pd.DataFrame:
    """Load pre-computed feature table from parquet."""
    if era == 'legacy':
        path = DATA_DIR / 'legacy_team_features.parquet'
    elif era == '2026':
        path = DATA_DIR / '2026_team_features.parquet'
    else:
        raise ValueError(f"Unknown era: {era}")

    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()
