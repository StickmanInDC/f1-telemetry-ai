"""
Time-aware cross-validation utilities.
NEVER use standard k-fold. Always train on earlier data, test on later data.
"""

import numpy as np
import pandas as pd
from typing import Generator


def walk_forward_cv(X: np.ndarray, y: np.ndarray, seasons: np.ndarray,
                     n_test_seasons: int = 1) -> Generator:
    """
    Time-aware cross-validation for F1 data.
    Always trains on earlier seasons, tests on later ones.

    Args:
        X: Feature matrix
        y: Target vector
        seasons: Array of season years aligned with X rows
        n_test_seasons: Minimum number of training seasons before first test

    Yields:
        (X_train, X_test, y_train, y_test, test_season) tuples
    """
    unique_seasons = sorted(np.unique(seasons))

    for i in range(n_test_seasons, len(unique_seasons)):
        test_season = unique_seasons[i]
        train_seasons = unique_seasons[:i]

        train_mask = np.isin(seasons, train_seasons)
        test_mask = seasons == test_season

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        yield (X[train_mask], X[test_mask],
               y[train_mask], y[test_mask],
               test_season)


def expanding_window_cv(X: np.ndarray, y: np.ndarray,
                          rounds: np.ndarray,
                          min_train_rounds: int = 4) -> Generator:
    """
    Expanding window CV for within-season validation.
    Useful for 2026 model with limited data.

    Args:
        X: Feature matrix
        y: Target vector
        rounds: Array of race round numbers
        min_train_rounds: Minimum rounds before first test

    Yields:
        (X_train, X_test, y_train, y_test, test_round) tuples
    """
    unique_rounds = sorted(np.unique(rounds))

    for i in range(min_train_rounds, len(unique_rounds)):
        test_round = unique_rounds[i]
        train_rounds = unique_rounds[:i]

        train_mask = np.isin(rounds, train_rounds)
        test_mask = rounds == test_round

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        yield (X[train_mask], X[test_mask],
               y[train_mask], y[test_mask],
               test_round)


def compute_baseline_persistence(y_test: np.ndarray,
                                    y_prev: np.ndarray) -> float:
    """
    Compute persistence baseline: "same as last race" prediction.
    This is the minimum bar the model must beat.
    """
    from sklearn.metrics import mean_absolute_error
    # Trim to same length
    n = min(len(y_test), len(y_prev))
    if n == 0:
        return np.nan
    return mean_absolute_error(y_test[:n], y_prev[:n])
