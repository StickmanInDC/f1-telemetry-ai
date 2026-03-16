"""
Tests for tire degradation feature computation.
Validates deg_rate calculation against known data patterns.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import timedelta

from src.features.legacy.tires import (
    compute_deg_rate,
    compute_compound_delta,
    compute_thermal_deg_phase,
    compute_mechanical_deg_phase,
    compute_undercut_vulnerability,
    compute_tyre_warmup,
    compute_all_tire_features,
)


def _make_stint_laps(driver: str, compound: str, base_time: float,
                      deg_rate: float, num_laps: int,
                      start_lap: int = 1) -> pd.DataFrame:
    """Create synthetic stint laps with known degradation rate."""
    laps = []
    for i in range(num_laps):
        lap_time = base_time + deg_rate * i + np.random.normal(0, 0.02)
        laps.append({
            'Driver': driver,
            'Compound': compound,
            'LapNumber': start_lap + i,
            'LapTime': timedelta(seconds=lap_time),
            'IsAccurate': True,
            'TyreLife': i + 1,
            'Team': 'TestTeam',
        })
    return pd.DataFrame(laps)


class TestDegRate:
    def test_positive_degradation(self):
        """Deg rate should be positive when lap times increase."""
        laps = _make_stint_laps('VER', 'MEDIUM', 90.0, 0.08, 15)
        result = compute_deg_rate(laps, 'VER', 'MEDIUM')

        assert result is not None
        assert result['deg_rate'] > 0
        assert abs(result['deg_rate'] - 0.08) < 0.03  # Close to known rate

    def test_no_degradation(self):
        """Deg rate near zero when lap times are constant."""
        laps = _make_stint_laps('HAM', 'HARD', 91.5, 0.0, 20)
        result = compute_deg_rate(laps, 'HAM', 'HARD')

        assert result is not None
        assert abs(result['deg_rate']) < 0.05

    def test_high_degradation(self):
        """Correctly captures high deg rates (soft tires)."""
        laps = _make_stint_laps('LEC', 'SOFT', 89.0, 0.15, 12)
        result = compute_deg_rate(laps, 'LEC', 'SOFT')

        assert result is not None
        assert result['deg_rate'] > 0.10

    def test_insufficient_laps(self):
        """Returns None when fewer than 4 laps."""
        laps = _make_stint_laps('NOR', 'MEDIUM', 90.0, 0.08, 3)
        result = compute_deg_rate(laps, 'NOR', 'MEDIUM')
        assert result is None

    def test_wrong_driver(self):
        """Returns None for a driver not in the data."""
        laps = _make_stint_laps('VER', 'MEDIUM', 90.0, 0.08, 15)
        result = compute_deg_rate(laps, 'HAM', 'MEDIUM')
        assert result is None

    def test_r_squared_quality(self):
        """R-squared should be high for clean linear degradation."""
        laps = _make_stint_laps('VER', 'MEDIUM', 90.0, 0.10, 20)
        result = compute_deg_rate(laps, 'VER', 'MEDIUM')

        assert result is not None
        assert result['r_squared'] > 0.5


class TestCompoundDelta:
    def test_soft_faster_than_medium(self):
        """Compound delta should be positive (medium slower than soft)."""
        soft = _make_stint_laps('VER', 'SOFT', 89.0, 0.12, 10)
        medium = _make_stint_laps('VER', 'MEDIUM', 90.0, 0.08, 15, start_lap=11)
        all_laps = pd.concat([soft, medium])

        delta = compute_compound_delta(all_laps, 'VER')
        assert delta > 0  # Medium is slower


class TestThermalDeg:
    def test_early_stint_detection(self):
        """Should detect steep early-stint degradation."""
        # Create laps with steep early deg, then leveling off
        laps = []
        for i in range(20):
            if i < 5:
                lt = 90.0 + 0.3 * i  # Steep
            else:
                lt = 90.0 + 0.3 * 5 + 0.05 * (i - 5)  # Flat
            laps.append({
                'Driver': 'VER', 'Compound': 'SOFT',
                'LapNumber': i + 1,
                'LapTime': timedelta(seconds=lt),
                'IsAccurate': True,
            })
        df = pd.DataFrame(laps)

        thermal = compute_thermal_deg_phase(df, 'VER', 'SOFT')
        assert thermal > 0.2  # Steep early deg


class TestMechanicalDeg:
    def test_cliff_detection(self):
        """Should detect late-stint cliff."""
        laps = []
        for i in range(25):
            if i < 20:
                lt = 90.0 + 0.05 * i  # Gentle
            else:
                lt = 90.0 + 0.05 * 20 + 0.5 * (i - 20)  # Cliff
            laps.append({
                'Driver': 'VER', 'Compound': 'MEDIUM',
                'LapNumber': i + 1,
                'LapTime': timedelta(seconds=lt),
                'IsAccurate': True,
            })
        df = pd.DataFrame(laps)

        mech = compute_mechanical_deg_phase(df, 'VER', 'MEDIUM')
        assert mech > 0.5  # Significant cliff


class TestAllTireFeatures:
    def test_returns_all_keys(self):
        """Should return all expected feature keys."""
        soft = _make_stint_laps('VER', 'SOFT', 89.0, 0.12, 10)
        medium = _make_stint_laps('VER', 'MEDIUM', 90.0, 0.08, 20, start_lap=11)
        hard = _make_stint_laps('VER', 'HARD', 91.0, 0.04, 20, start_lap=31)
        all_laps = pd.concat([soft, medium, hard])

        features = compute_all_tire_features(all_laps, 'VER')

        expected_keys = [
            'deg_rate_soft', 'deg_rate_medium', 'deg_rate_hard',
            'compound_delta', 'thermal_deg_phase', 'mechanical_deg_phase',
            'undercut_vulnerability', 'tyre_warm_up_laps',
        ]
        for key in expected_keys:
            assert key in features, f"Missing key: {key}"

    def test_deg_ordering(self):
        """Soft deg should be higher than medium, which should be higher than hard."""
        soft = _make_stint_laps('VER', 'SOFT', 89.0, 0.15, 12)
        medium = _make_stint_laps('VER', 'MEDIUM', 90.0, 0.08, 20, start_lap=13)
        hard = _make_stint_laps('VER', 'HARD', 91.0, 0.04, 25, start_lap=33)
        all_laps = pd.concat([soft, medium, hard])

        features = compute_all_tire_features(all_laps, 'VER')

        if not np.isnan(features['deg_rate_soft']) and not np.isnan(features['deg_rate_medium']):
            assert features['deg_rate_soft'] > features['deg_rate_medium']
        if not np.isnan(features['deg_rate_medium']) and not np.isnan(features['deg_rate_hard']):
            assert features['deg_rate_medium'] > features['deg_rate_hard']
