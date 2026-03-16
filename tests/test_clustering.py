"""
Tests for circuit clustering.
"""

import pytest
import pandas as pd
import numpy as np

from src.features.track_clustering import (
    cluster_circuits,
    find_optimal_k,
    get_cluster_profiles,
)


def _make_track_features() -> pd.DataFrame:
    """Create synthetic track feature data for testing."""
    np.random.seed(42)
    circuits = {
        # High-speed
        'Monza': {'pct_full_throttle': 0.78, 'pct_heavy_braking': 0.12,
                   'avg_corner_speed': 190, 'num_slow_corners': 2,
                   'num_fast_corners': 5, 'energy_recovery_potential': 3.5,
                   'is_street_circuit': 0, 'altitude_m': 162},
        'Spa': {'pct_full_throttle': 0.72, 'pct_heavy_braking': 0.15,
                 'avg_corner_speed': 180, 'num_slow_corners': 3,
                 'num_fast_corners': 6, 'energy_recovery_potential': 4.0,
                 'is_street_circuit': 0, 'altitude_m': 400},
        # Technical
        'Monaco': {'pct_full_throttle': 0.42, 'pct_heavy_braking': 0.30,
                    'avg_corner_speed': 95, 'num_slow_corners': 8,
                    'num_fast_corners': 0, 'energy_recovery_potential': 6.5,
                    'is_street_circuit': 1, 'altitude_m': 7},
        'Hungary': {'pct_full_throttle': 0.48, 'pct_heavy_braking': 0.25,
                     'avg_corner_speed': 110, 'num_slow_corners': 6,
                     'num_fast_corners': 1, 'energy_recovery_potential': 5.5,
                     'is_street_circuit': 0, 'altitude_m': 264},
        # Mixed
        'Bahrain': {'pct_full_throttle': 0.62, 'pct_heavy_braking': 0.20,
                     'avg_corner_speed': 145, 'num_slow_corners': 4,
                     'num_fast_corners': 3, 'energy_recovery_potential': 5.0,
                     'is_street_circuit': 0, 'altitude_m': 7},
        'Suzuka': {'pct_full_throttle': 0.64, 'pct_heavy_braking': 0.18,
                    'avg_corner_speed': 155, 'num_slow_corners': 3,
                    'num_fast_corners': 4, 'energy_recovery_potential': 4.5,
                    'is_street_circuit': 0, 'altitude_m': 45},
        # Street
        'Singapore': {'pct_full_throttle': 0.45, 'pct_heavy_braking': 0.28,
                       'avg_corner_speed': 100, 'num_slow_corners': 7,
                       'num_fast_corners': 0, 'energy_recovery_potential': 7.0,
                       'is_street_circuit': 1, 'altitude_m': 18},
        'Baku': {'pct_full_throttle': 0.55, 'pct_heavy_braking': 0.22,
                  'avg_corner_speed': 120, 'num_slow_corners': 5,
                  'num_fast_corners': 1, 'energy_recovery_potential': 5.5,
                  'is_street_circuit': 1, 'altitude_m': -28},
    }

    rows = []
    for circuit, features in circuits.items():
        row = {'circuit': circuit}
        row.update(features)
        rows.append(row)

    return pd.DataFrame(rows)


class TestCircuitClustering:
    def test_produces_clusters(self):
        """Should assign a cluster to each circuit."""
        df = _make_track_features()
        result, km, scaler = cluster_circuits(df, k=3, save=False)

        assert 'circuit_cluster' in result.columns
        assert result['circuit_cluster'].nunique() <= 3
        assert not result['circuit_cluster'].isna().any()

    def test_similar_circuits_cluster_together(self):
        """Monza and Spa should be in the same cluster (high-speed)."""
        df = _make_track_features()
        result, _, _ = cluster_circuits(df, k=3, save=False)

        monza_cluster = result[result['circuit'] == 'Monza']['circuit_cluster'].iloc[0]
        spa_cluster = result[result['circuit'] == 'Spa']['circuit_cluster'].iloc[0]
        assert monza_cluster == spa_cluster

    def test_different_circuits_separate(self):
        """Monza and Monaco should be in different clusters."""
        df = _make_track_features()
        result, _, _ = cluster_circuits(df, k=3, save=False)

        monza_cluster = result[result['circuit'] == 'Monza']['circuit_cluster'].iloc[0]
        monaco_cluster = result[result['circuit'] == 'Monaco']['circuit_cluster'].iloc[0]
        assert monza_cluster != monaco_cluster

    def test_optimal_k_returns_scores(self):
        """Should return silhouette scores for each k value."""
        df = _make_track_features()
        scores = find_optimal_k(df, k_range=range(2, 5))

        assert len(scores) > 0
        for k, score in scores.items():
            assert -1 <= score <= 1

    def test_cluster_profiles_meaningful(self):
        """Cluster profiles should have different mean values."""
        df = _make_track_features()
        result, _, _ = cluster_circuits(df, k=3, save=False)
        profiles = get_cluster_profiles(result)

        assert len(profiles) == 3
        # Different clusters should have different mean throttle percentages
        throttle_range = profiles['pct_full_throttle'].max() - profiles['pct_full_throttle'].min()
        assert throttle_range > 0.1
