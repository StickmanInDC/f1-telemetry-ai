"""
2026 Reliability & Systemic Failure Risk features.
Reliability is a first-class predictive feature in 2026.
Separates car/platform failures (predictive) from driver incidents (not predictive).
"""

import pandas as pd
import numpy as np

SYSTEMIC_CATEGORIES = {'power_unit', 'electrical', 'hydraulics', 'mechanical', 'chassis'}
INCIDENT_CATEGORIES = {'collision_fault', 'collision_racing', 'collision_other'}

# Track stress indices — higher = more mechanical stress on car
TRACK_STRESS_INDEX = {
    'baku': 0.90, 'singapore': 0.85, 'monaco': 0.80, 'jeddah': 0.75,
    'melbourne': 0.60, 'miami': 0.55, 'las_vegas': 0.50,
    'bahrain': 0.40, 'suzuka': 0.45, 'silverstone': 0.35,
    'monza': 0.30, 'spa': 0.40, 'austin': 0.45, 'mexico': 0.50,
    'interlagos': 0.50, 'barcelona': 0.35, 'hungaroring': 0.55,
    'zandvoort': 0.45, 'imola': 0.40, 'lusail': 0.50,
    'yas_marina': 0.35, 'shanghai': 0.40, 'red_bull_ring': 0.30,
}


def compute_reliability_features(results_df: pd.DataFrame, team: str,
                                   last_n: int = None) -> dict:
    """
    Compute all reliability features for a team.

    Args:
        results_df: DataFrame with columns: race_id/round, team/constructor_name,
                    driver, laps_completed, total_laps, dnf_category
        team: Team name to compute features for
        last_n: If set, only use the last N races

    Returns dict of reliability feature values.
    """
    # Find team data using multiple possible column names
    team_col = None
    for col in ['constructor_name', 'team', 'constructor_id']:
        if col in results_df.columns:
            team_col = col
            break

    if team_col is None:
        return _empty_reliability_features()

    df = results_df[results_df[team_col] == team].copy()
    if df.empty:
        return _empty_reliability_features()

    race_col = 'race_id' if 'race_id' in df.columns else 'round'

    if last_n:
        race_ids = df[race_col].unique()
        if len(race_ids) > last_n:
            race_ids = race_ids[-last_n:]
        df = df[df[race_col].isin(race_ids)]

    n_races = df[race_col].nunique()
    n_starts = len(df)

    if n_starts == 0:
        return _empty_reliability_features()

    # Systemic DNFs
    systemic = df[df['dnf_category'].isin(SYSTEMIC_CATEGORIES)]
    systemic_rate = len(systemic) / n_starts

    # Both cars DNF rate
    both_dnf = (
        df[df['dnf_category'].isin(SYSTEMIC_CATEGORIES)]
        .groupby(race_col).size()
        .ge(2).sum()
    )
    both_cars_rate = both_dnf / n_races if n_races > 0 else 0

    # Completion rate
    if 'total_laps' in df.columns:
        completion = (df['laps_completed'] / df['total_laps'].clip(lower=1)).mean()
    else:
        completion = 1.0 - systemic_rate

    # Reliability trend (slope of completion rate over last races)
    if 'total_laps' in df.columns:
        per_race_completion = {}
        for race_id, group in df.groupby(race_col):
            per_race_completion[race_id] = (
                group['laps_completed'] / group['total_laps'].clip(lower=1)
            ).mean()
        per_race = pd.DataFrame(
            list(per_race_completion.items()),
            columns=[race_col, 'completion']
        )
    else:
        per_race = df.groupby(race_col)['laps_completed'].mean().reset_index(name='completion')

    trend = 0.0
    if len(per_race) >= 3:
        coeffs = np.polyfit(range(len(per_race)), per_race['completion'].values, 1)
        trend = coeffs[0]

    # Per-failure-type rates
    pu_rate = len(df[df['dnf_category'] == 'power_unit']) / n_starts
    elec_rate = len(df[df['dnf_category'] == 'electrical']) / n_starts
    hyd_rate = len(df[df['dnf_category'] == 'hydraulics']) / n_starts

    # Driver incident rate
    incident_rate = len(df[df['dnf_category'].isin(INCIDENT_CATEGORIES)]) / n_starts

    # DNQ rate
    dnq_count = len(df[df['dnf_category'] == 'unknown'])  # Approximate
    dns_count = len(df[df['laps_completed'] == 0])

    # Precautionary retirement rate
    precautionary = len(df[df['dnf_category'] == 'precautionary']) / n_starts

    return {
        'systemic_dnf_rate': systemic_rate,
        'driver_incident_rate': incident_rate,
        'dnf_rate_rolling_5': compute_rolling_dnf_rate(df, race_col, window=5),
        'completion_rate': completion,
        'both_cars_dnf_rate': both_cars_rate,
        'dnq_rate': dns_count / n_starts,
        'precautionary_retirement_rate': precautionary,
        'pu_failure_rate': pu_rate,
        'electrical_failure_rate': elec_rate,
        'hydraulic_failure_rate': hyd_rate,
        'reliability_trend': trend,
    }


def compute_rolling_dnf_rate(df: pd.DataFrame, race_col: str,
                               window: int = 5) -> float:
    """Systemic DNF rate over the last N races."""
    race_ids = sorted(df[race_col].unique())
    if len(race_ids) < window:
        window = len(race_ids)

    recent_races = race_ids[-window:]
    recent = df[df[race_col].isin(recent_races)]
    n_starts = len(recent)

    if n_starts == 0:
        return 0.0

    systemic = recent[recent['dnf_category'].isin(SYSTEMIC_CATEGORIES)]
    return len(systemic) / n_starts


def compute_pu_supplier_failure_rate(results_df: pd.DataFrame,
                                       pu_supplier_teams: list[str]) -> float:
    """
    PU failures aggregated across ALL customer teams of a supplier.
    """
    team_col = None
    for col in ['constructor_name', 'team', 'constructor_id']:
        if col in results_df.columns:
            team_col = col
            break

    if team_col is None:
        return np.nan

    supplier_data = results_df[results_df[team_col].isin(pu_supplier_teams)]
    if supplier_data.empty:
        return np.nan

    n_starts = len(supplier_data)
    pu_failures = supplier_data['dnf_category'].isin({'power_unit', 'electrical'}).sum()

    return pu_failures / n_starts


def compute_reliability_risk_score(systemic_dnf_rate: float,
                                     circuit_name: str) -> float:
    """
    Interaction: systemic_dnf_rate x track_stress_index.
    Amplifies fragility at high-stress circuits.
    """
    circuit_key = circuit_name.lower().replace(' ', '_')
    stress_index = TRACK_STRESS_INDEX.get(circuit_key, 0.5)
    return systemic_dnf_rate * stress_index


def _empty_reliability_features() -> dict:
    """Return dict of NaN reliability features."""
    return {
        'systemic_dnf_rate': np.nan,
        'driver_incident_rate': np.nan,
        'dnf_rate_rolling_5': np.nan,
        'completion_rate': np.nan,
        'both_cars_dnf_rate': np.nan,
        'dnq_rate': np.nan,
        'precautionary_retirement_rate': np.nan,
        'pu_failure_rate': np.nan,
        'electrical_failure_rate': np.nan,
        'hydraulic_failure_rate': np.nan,
        'reliability_trend': np.nan,
    }
