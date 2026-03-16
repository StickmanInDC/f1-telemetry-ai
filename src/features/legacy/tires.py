"""
Group 3: Tire Degradation features.
The highest-value feature group. ALL STABLE across both eras.
Implement first and validate thoroughly.
"""

import pandas as pd
import numpy as np


def compute_deg_rate(clean_laps: pd.DataFrame, driver: str,
                      compound: str) -> dict | None:
    """
    Lap time loss per stint lap per compound.
    Linear regression: LapTime_seconds ~ StintLap.
    Slope = deg rate (seconds per lap). The key feature.

    Returns dict with deg_rate, base_pace, r_squared, num_laps.
    """
    stint = clean_laps[
        (clean_laps['Driver'] == driver) &
        (clean_laps['Compound'] == compound)
    ].copy()

    if len(stint) < 4:
        return None

    stint = stint.sort_values('LapNumber')
    stint['StintLap'] = range(len(stint))
    lt_seconds = stint['LapTime'].dt.total_seconds()

    # Remove extreme outliers (>3 std from mean)
    mean_lt = lt_seconds.mean()
    std_lt = lt_seconds.std()
    if std_lt > 0:
        mask = np.abs(lt_seconds - mean_lt) < 3 * std_lt
        stint = stint[mask.values]
        lt_seconds = lt_seconds[mask]

    if len(stint) < 4:
        return None

    stint_laps = np.arange(len(stint))
    coeffs = np.polyfit(stint_laps, lt_seconds.values, deg=1)

    # R-squared for quality check
    predicted = np.polyval(coeffs, stint_laps)
    ss_res = np.sum((lt_seconds.values - predicted) ** 2)
    ss_tot = np.sum((lt_seconds.values - lt_seconds.mean()) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return {
        'driver': driver,
        'compound': compound,
        'deg_rate': coeffs[0],
        'base_pace': coeffs[1],
        'r_squared': r_squared,
        'num_laps': len(stint),
    }


def compute_compound_delta(clean_laps: pd.DataFrame, driver: str) -> float:
    """
    Pace delta between soft and medium (setup/thermal sensitivity proxy).
    Normalized against field average.
    """
    soft_laps = clean_laps[
        (clean_laps['Driver'] == driver) &
        (clean_laps['Compound'] == 'SOFT')
    ]
    medium_laps = clean_laps[
        (clean_laps['Driver'] == driver) &
        (clean_laps['Compound'] == 'MEDIUM')
    ]

    if soft_laps.empty or medium_laps.empty:
        return np.nan

    soft_pace = soft_laps['LapTime'].dt.total_seconds().median()
    medium_pace = medium_laps['LapTime'].dt.total_seconds().median()

    return medium_pace - soft_pace


def compute_thermal_deg_phase(clean_laps: pd.DataFrame, driver: str,
                                compound: str) -> float:
    """
    Early-stint degradation rate (laps 1-5 of stint).
    Steeper slope = higher thermal deg sensitivity.
    """
    stint = clean_laps[
        (clean_laps['Driver'] == driver) &
        (clean_laps['Compound'] == compound)
    ].copy()

    if len(stint) < 5:
        return np.nan

    stint = stint.sort_values('LapNumber')
    early = stint.head(5)
    lt_seconds = early['LapTime'].dt.total_seconds()

    if len(lt_seconds) < 3:
        return np.nan

    stint_laps = np.arange(len(lt_seconds))
    coeffs = np.polyfit(stint_laps, lt_seconds.values, deg=1)
    return coeffs[0]


def compute_mechanical_deg_phase(clean_laps: pd.DataFrame, driver: str,
                                   compound: str) -> float:
    """
    Late-stint cliff behaviour.
    Pace delta on final 5 laps vs. median pace laps 6-15.
    Captures cliff not just linear deg.
    """
    stint = clean_laps[
        (clean_laps['Driver'] == driver) &
        (clean_laps['Compound'] == compound)
    ].copy()

    if len(stint) < 15:
        return np.nan

    stint = stint.sort_values('LapNumber')
    lt_seconds = stint['LapTime'].dt.total_seconds()

    mid_pace = lt_seconds.iloc[5:15].median()
    final_pace = lt_seconds.iloc[-5:].median()

    return final_pace - mid_pace


def compute_undercut_vulnerability(clean_laps: pd.DataFrame, driver: str,
                                     compound: str) -> float:
    """
    Time lost per lap on aged vs. fresh tires.
    Pace delta: last 3 laps of stint vs. first 3 laps.
    """
    stint = clean_laps[
        (clean_laps['Driver'] == driver) &
        (clean_laps['Compound'] == compound)
    ].copy()

    if len(stint) < 8:
        return np.nan

    stint = stint.sort_values('LapNumber')
    lt_seconds = stint['LapTime'].dt.total_seconds()

    fresh_pace = lt_seconds.iloc[:3].median()
    aged_pace = lt_seconds.iloc[-3:].median()

    return aged_pace - fresh_pace


def compute_tyre_warmup(clean_laps: pd.DataFrame, driver: str,
                          compound: str) -> float:
    """
    Laps to reach peak pace after pit stop.
    Measures laps until lap time stabilizes (std dev of rolling 3-lap window < threshold).
    """
    stint = clean_laps[
        (clean_laps['Driver'] == driver) &
        (clean_laps['Compound'] == compound)
    ].copy()

    if len(stint) < 5:
        return np.nan

    stint = stint.sort_values('LapNumber')
    lt_seconds = stint['LapTime'].dt.total_seconds()

    # Rolling std dev with window of 3
    rolling_std = lt_seconds.rolling(3, min_periods=3).std()

    # Find first lap where rolling std drops below 0.3s (stable)
    threshold = 0.3
    for i, std_val in enumerate(rolling_std.values):
        if not np.isnan(std_val) and std_val < threshold:
            return max(1, i - 1)  # Laps to stabilize

    return len(stint)  # Never stabilized


def compute_all_tire_features(clean_laps: pd.DataFrame, driver: str) -> dict:
    """Compute all Group 3 features for a driver."""
    features = {}

    # Deg rate per compound
    for compound in ['SOFT', 'MEDIUM', 'HARD']:
        result = compute_deg_rate(clean_laps, driver, compound)
        key = f'deg_rate_{compound.lower()}'
        features[key] = result['deg_rate'] if result else np.nan

    features['compound_delta'] = compute_compound_delta(clean_laps, driver)

    # Thermal and mechanical deg (use medium compound as primary)
    for compound in ['MEDIUM', 'SOFT', 'HARD']:
        thermal = compute_thermal_deg_phase(clean_laps, driver, compound)
        if not np.isnan(thermal):
            features['thermal_deg_phase'] = thermal
            break
    else:
        features['thermal_deg_phase'] = np.nan

    for compound in ['MEDIUM', 'HARD']:
        mech = compute_mechanical_deg_phase(clean_laps, driver, compound)
        if not np.isnan(mech):
            features['mechanical_deg_phase'] = mech
            break
    else:
        features['mechanical_deg_phase'] = np.nan

    features['undercut_vulnerability'] = compute_undercut_vulnerability(
        clean_laps, driver, 'MEDIUM')

    features['tyre_warm_up_laps'] = compute_tyre_warmup(
        clean_laps, driver, 'MEDIUM')

    return features
