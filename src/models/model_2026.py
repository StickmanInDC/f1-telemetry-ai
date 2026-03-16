"""
2026 Transfer Model.
XGBoost with transfer learning from stable legacy features.
Dynamic regularization that loosens as data accumulates through the season.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import shap
import joblib
from pathlib import Path
import logging

from .validation import expanding_window_cv, compute_baseline_persistence

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent.parent.parent / 'data' / 'processed'

# Stable features carried from legacy into 2026 model (transfer anchors)
LATENT_FEATURES = [
    'deg_rate_medium', 'thermal_deg_phase', 'mechanical_deg_phase',
    'high_speed_corner_grip', 'low_speed_corner_grip',
    'brake_point_consistency', 'trail_braking_index',
    'corner_speed_vs_straight_speed',
    'quali_vs_race_delta', 'setup_sensitivity',
]

# New 2026-specific features
ERA2026_FEATURES = [
    # Active aero
    'straight_mode_speed_gain', 'active_aero_consistency',
    # Energy management
    'superclip_frequency', 'superclip_harvest_rate',
    'lift_coast_vs_superclip_ratio', 'energy_balance_per_lap',
    # Boost and OT mode
    'boost_frequency_per_lap', 'boost_energy_per_activation',
    'overtake_mode_conversion_rate', 'overtake_mode_availability_rate',
    # Turbo architecture
    'turbo_spool_proxy', 'launch_positions_gained',
    'high_speed_straight_deficit', 'track_power_sensitivity_interaction',
    # Reliability
    'systemic_dnf_rate', 'dnf_rate_rolling_5',
    'completion_rate', 'reliability_trend',
    # Track (updated)
    'circuit_cluster', 'energy_richness_score', 'track_power_sensitivity_score',
]

ALL_2026_FEATURES = LATENT_FEATURES + ERA2026_FEATURES
TARGET_COL = 'race_pace_gap_pct'


def build_2026_feature_matrix(feature_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Prepare the 2026 feature matrix combining latent + new features.
    Returns (X, y, rounds, feature_names).
    """
    available = [f for f in ALL_2026_FEATURES if f in feature_df.columns]

    df = feature_df.dropna(subset=[TARGET_COL]).copy()
    if df.empty:
        return np.array([]), np.array([]), np.array([]), available

    X = df[available].copy()
    for col in X.columns:
        median_val = X[col].median()
        X[col] = X[col].fillna(median_val if not np.isnan(median_val) else 0)

    y = df[TARGET_COL].values
    rounds = df['round'].values

    return X.values, y, rounds, available


def train_2026_model(feature_df: pd.DataFrame,
                       n_races_available: int = None,
                       save: bool = True) -> tuple[xgb.XGBRegressor, list[tuple]]:
    """
    Train the 2026 XGBoost model with dynamic regularization.
    Early season: strong regularization, rely on priors.
    Late season: looser regularization as data accumulates.
    """
    X, y, rounds, feature_names = build_2026_feature_matrix(feature_df)

    if len(X) == 0:
        logger.error("No valid 2026 training data!")
        return None, []

    if n_races_available is None:
        n_races_available = len(np.unique(rounds))

    # Dynamic regularization — decays with data volume
    reg_strength = max(0.1, 2.0 - (n_races_available * 0.15))
    max_depth = 3 if n_races_available < 8 else 4

    logger.info(f"Training 2026 model: {len(X)} samples, {len(feature_names)} features, "
                f"reg={reg_strength:.2f}, depth={max_depth}")

    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=max_depth,
        learning_rate=0.05,
        reg_alpha=reg_strength,
        reg_lambda=reg_strength * 2,
        subsample=0.8,
        colsample_bytree=0.7,
        random_state=42,
    )

    # If enough data, use expanding window CV
    maes = []
    if n_races_available >= 6:
        for X_train, X_test, y_train, y_test, test_round in expanding_window_cv(
                X, y, rounds, min_train_rounds=4):
            model.fit(X_train, y_train, verbose=False)
            preds = model.predict(X_test)
            mae = mean_absolute_error(y_test, preds)
            maes.append((test_round, mae))
            logger.info(f"  Round {test_round}: MAE = {mae:.4f}%")

    # Final model on all data
    model.fit(X, y, verbose=False)

    if save:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_DIR / 'model_2026.joblib')
        joblib.dump(feature_names, MODEL_DIR / 'model_2026_features.joblib')
        logger.info("Saved 2026 model")

    return model, maes


def predict_2026(feature_df: pd.DataFrame,
                   model: xgb.XGBRegressor = None) -> pd.DataFrame:
    """Generate predictions for upcoming 2026 races."""
    if model is None:
        model = joblib.load(MODEL_DIR / 'model_2026.joblib')

    feature_names = joblib.load(MODEL_DIR / 'model_2026_features.joblib')
    available = [f for f in feature_names if f in feature_df.columns]

    X = feature_df[available].copy()
    for col in X.columns:
        median_val = X[col].median()
        X[col] = X[col].fillna(median_val if not np.isnan(median_val) else 0)

    for f in feature_names:
        if f not in X.columns:
            X[f] = 0
    X = X[feature_names]

    predictions = model.predict(X.values)

    result = feature_df[['team', 'circuit']].copy()
    if 'season' in feature_df.columns:
        result['season'] = feature_df['season']
    if 'round' in feature_df.columns:
        result['round'] = feature_df['round']
    result['predicted_pace_gap_pct'] = predictions

    return result


def explain_2026_model(model: xgb.XGBRegressor = None,
                         feature_df: pd.DataFrame = None,
                         top_n: int = 15) -> pd.DataFrame:
    """Compute SHAP feature importance for the 2026 model."""
    if model is None:
        model = joblib.load(MODEL_DIR / 'model_2026.joblib')

    feature_names = joblib.load(MODEL_DIR / 'model_2026_features.joblib')

    if feature_df is not None:
        X, _, _, _ = build_2026_feature_matrix(feature_df)
    else:
        X = np.random.randn(100, len(feature_names))

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    importance = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0),
    }).sort_values('mean_abs_shap', ascending=False)

    return importance.head(top_n)


def compare_to_persistence_baseline(predictions_df: pd.DataFrame,
                                       actuals_df: pd.DataFrame) -> dict:
    """
    Compare model predictions to the naive 'same as last race' baseline.
    The 2026 model must beat this by race 6 to be considered useful.
    """
    model_maes = []
    persist_maes = []

    rounds = sorted(predictions_df['round'].unique())

    for i, round_num in enumerate(rounds):
        pred = predictions_df[predictions_df['round'] == round_num]
        actual = actuals_df[actuals_df['round'] == round_num]

        if actual.empty or pred.empty:
            continue

        # Merge on team
        merged = pred.merge(actual[['team', TARGET_COL]], on='team', how='inner')
        if merged.empty:
            continue

        model_mae = mean_absolute_error(
            merged[TARGET_COL], merged['predicted_pace_gap_pct'])
        model_maes.append(model_mae)

        # Persistence: use previous round's actual as prediction
        if i > 0:
            prev_round = rounds[i - 1]
            prev_actual = actuals_df[actuals_df['round'] == prev_round]
            if not prev_actual.empty:
                persist_merged = actual.merge(
                    prev_actual[['team', TARGET_COL]],
                    on='team', how='inner', suffixes=('_actual', '_prev'))
                if not persist_merged.empty:
                    persist_mae = mean_absolute_error(
                        persist_merged[f'{TARGET_COL}_actual'],
                        persist_merged[f'{TARGET_COL}_prev'])
                    persist_maes.append(persist_mae)

    return {
        'model_avg_mae': np.mean(model_maes) if model_maes else np.nan,
        'persistence_avg_mae': np.mean(persist_maes) if persist_maes else np.nan,
        'model_beats_baseline': (
            np.mean(model_maes) < np.mean(persist_maes)
            if model_maes and persist_maes else None
        ),
    }
