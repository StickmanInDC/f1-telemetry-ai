"""
Tests for time-aware cross-validation.
"""

import pytest
import numpy as np

from src.models.validation import walk_forward_cv, expanding_window_cv


class TestWalkForwardCV:
    def test_no_data_leak(self):
        """Training data should never contain samples from test season."""
        X = np.random.randn(40, 5)
        y = np.random.randn(40)
        seasons = np.array([2022] * 10 + [2023] * 10 + [2024] * 10 + [2025] * 10)

        for X_train, X_test, y_train, y_test, test_year in walk_forward_cv(X, y, seasons):
            train_seasons = seasons[np.isin(X, X_train).all(axis=1)]
            assert all(s < test_year for s in np.unique(train_seasons) if s != 0)

    def test_correct_split_count(self):
        """Should produce correct number of train/test splits."""
        X = np.random.randn(30, 3)
        y = np.random.randn(30)
        seasons = np.array([2022] * 10 + [2023] * 10 + [2024] * 10)

        splits = list(walk_forward_cv(X, y, seasons))
        assert len(splits) == 2  # Test on 2023, 2024

    def test_expanding_training_set(self):
        """Each subsequent fold should have more training data."""
        X = np.random.randn(40, 3)
        y = np.random.randn(40)
        seasons = np.array([2022] * 10 + [2023] * 10 + [2024] * 10 + [2025] * 10)

        train_sizes = []
        for X_train, _, _, _, _ in walk_forward_cv(X, y, seasons):
            train_sizes.append(len(X_train))

        for i in range(1, len(train_sizes)):
            assert train_sizes[i] > train_sizes[i-1]


class TestExpandingWindowCV:
    def test_within_season_splits(self):
        """Should create expanding window within a single season."""
        X = np.random.randn(50, 3)
        y = np.random.randn(50)
        rounds = np.array([1]*5 + [2]*5 + [3]*5 + [4]*5 + [5]*5 +
                           [6]*5 + [7]*5 + [8]*5 + [9]*5 + [10]*5)

        splits = list(expanding_window_cv(X, y, rounds, min_train_rounds=4))
        assert len(splits) > 0

        # First test round should be round 5 (after 4 training rounds)
        assert splits[0][4] == 5
