"""
Tests for reliability feature computation and DNF classification.
"""

import pytest
import pandas as pd
import numpy as np

from src.ingestion.ergast_client import classify_dnf
from src.features.era2026.reliability import (
    compute_reliability_features,
    compute_pu_supplier_failure_rate,
    compute_reliability_risk_score,
    SYSTEMIC_CATEGORIES,
)


class TestDnfClassification:
    def test_engine_failure(self):
        assert classify_dnf('Engine') == 'power_unit'
        assert classify_dnf('Power Unit') == 'power_unit'
        assert classify_dnf('Turbo') == 'power_unit'

    def test_electrical_failure(self):
        assert classify_dnf('Battery') == 'electrical'
        assert classify_dnf('Electronics') == 'electrical'
        assert classify_dnf('Electrical') == 'electrical'

    def test_hydraulic_failure(self):
        assert classify_dnf('Hydraulics') == 'hydraulics'
        assert classify_dnf('Hydraulic') == 'hydraulics'

    def test_mechanical_failure(self):
        assert classify_dnf('Gearbox') == 'mechanical'
        assert classify_dnf('Suspension') == 'mechanical'
        assert classify_dnf('Brakes') == 'mechanical'

    def test_collision(self):
        assert classify_dnf('Collision') == 'collision_racing'
        assert classify_dnf('Spun off') == 'collision_fault'
        assert classify_dnf('Debris') == 'collision_other'

    def test_finished(self):
        assert classify_dnf('Finished') == 'finished'
        assert classify_dnf('+1 Lap') == 'finished'
        assert classify_dnf('+2 Laps') == 'finished'

    def test_case_insensitive(self):
        assert classify_dnf('ENGINE') == 'power_unit'
        assert classify_dnf('battery') == 'electrical'
        assert classify_dnf('GEARBOX') == 'mechanical'

    def test_unknown(self):
        assert classify_dnf('Something weird') == 'unknown'


def _make_results(team: str, statuses: list[str],
                   laps_per_race: int = 57) -> pd.DataFrame:
    """Create synthetic results data for testing."""
    rows = []
    for i, status in enumerate(statuses):
        race_num = (i // 2) + 1  # 2 drivers per race
        driver_num = i % 2
        laps = laps_per_race if classify_dnf(status) == 'finished' else int(laps_per_race * 0.5)

        rows.append({
            'round': race_num,
            'constructor_name': team,
            'driver_code': f'D{driver_num}',
            'laps_completed': laps,
            'total_laps': laps_per_race,
            'status': status,
            'dnf_category': classify_dnf(status),
        })
    return pd.DataFrame(rows)


class TestReliabilityFeatures:
    def test_perfect_reliability(self):
        """Team with no failures should have zero systemic DNF rate."""
        results = _make_results('PerfectTeam',
                                 ['Finished'] * 10)  # 5 races, 2 drivers
        features = compute_reliability_features(results, 'PerfectTeam')

        assert features['systemic_dnf_rate'] == 0.0
        assert features['completion_rate'] == 1.0
        assert features['both_cars_dnf_rate'] == 0.0

    def test_high_failure_rate(self):
        """Team with frequent failures should show high systemic DNF rate."""
        results = _make_results('BadTeam', [
            'Engine', 'Finished',    # Race 1: 1 PU failure
            'Battery', 'Hydraulics', # Race 2: 2 systemic (both cars)
            'Finished', 'Gearbox',   # Race 3: 1 mechanical
            'Finished', 'Finished',  # Race 4: clean
            'Engine', 'Finished',    # Race 5: 1 PU
        ])
        features = compute_reliability_features(results, 'BadTeam')

        assert features['systemic_dnf_rate'] > 0.3
        assert features['both_cars_dnf_rate'] > 0  # Race 2 had both cars
        assert features['pu_failure_rate'] > 0
        assert features['electrical_failure_rate'] > 0

    def test_driver_incidents_excluded(self):
        """Driver incidents should not count toward systemic DNF rate."""
        results = _make_results('IncidentTeam', [
            'Collision', 'Finished',  # Race 1: driver incident
            'Spun off', 'Finished',   # Race 2: driver error
            'Finished', 'Finished',   # Race 3: clean
        ])
        features = compute_reliability_features(results, 'IncidentTeam')

        assert features['systemic_dnf_rate'] == 0.0
        assert features['driver_incident_rate'] > 0

    def test_rolling_window(self):
        """Rolling 5-race window should only use recent data."""
        # 10 races: first 5 have failures, last 5 are clean
        statuses = (
            ['Engine', 'Finished'] * 5 +  # Races 1-5: failures
            ['Finished', 'Finished'] * 5   # Races 6-10: clean
        )
        results = _make_results('ImprovingTeam', statuses)
        features = compute_reliability_features(results, 'ImprovingTeam', last_n=5)

        # Last 5 races should show clean reliability
        assert features['systemic_dnf_rate'] == 0.0

    def test_reliability_trend(self):
        """Improving team should have positive reliability trend."""
        # Create data with improving completion rate
        statuses = (
            ['Engine', 'Battery'] +     # Race 1: both fail
            ['Finished', 'Gearbox'] +   # Race 2: 1 fail
            ['Finished', 'Finished'] +  # Race 3: clean
            ['Finished', 'Finished'] +  # Race 4: clean
            ['Finished', 'Finished']    # Race 5: clean
        )
        results = _make_results('ImprovingTeam', statuses)
        features = compute_reliability_features(results, 'ImprovingTeam')

        assert features['reliability_trend'] > 0  # Improving


class TestPuSupplierRate:
    def test_aggregates_across_teams(self):
        """Should aggregate failures across all customer teams."""
        team1 = _make_results('Team1', ['Engine', 'Finished', 'Finished', 'Finished'])
        team2 = _make_results('Team2', ['Finished', 'Battery', 'Finished', 'Finished'])
        results = pd.concat([team1, team2])

        rate = compute_pu_supplier_failure_rate(results, ['Team1', 'Team2'])
        assert rate > 0  # Both teams had PU-related failures


class TestReliabilityRiskScore:
    def test_high_stress_circuit(self):
        """High-stress circuit should amplify failure risk."""
        score_baku = compute_reliability_risk_score(0.3, 'baku')
        score_monza = compute_reliability_risk_score(0.3, 'monza')

        assert score_baku > score_monza  # Baku is higher stress

    def test_zero_failures(self):
        """Zero systemic rate should give zero risk regardless of circuit."""
        assert compute_reliability_risk_score(0.0, 'baku') == 0.0
        assert compute_reliability_risk_score(0.0, 'monaco') == 0.0
