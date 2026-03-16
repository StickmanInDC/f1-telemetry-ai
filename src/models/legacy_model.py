"""
Legacy (pre-2026) prediction model.
XGBoost regressor predicting normalized race pace gap to leader.
Time-aware cross-validation only.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import shap
import joblib
from pathlib import Path
import logging

from .validation import walk_forward_cv

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent.parent.parent / 'data' / 'processed'

LEGACY_FEATURES = [
    # Group 1: PU
    'pu_vmax_avg', 'ers_deployment_consistency', 'energy_harvest_efficiency',
    'battery_depletion_signature',
    # Group 2: Aero
    'corner_speed_vs_straight_speed', 'high_speed_corner_grip',
    'low_speed_corner_grip', 'sector1_vs_sector3_delta',
    # Group 3: Tires
    'deg_rate_soft', 'deg_rate_medium', 'deg_rate_hard',
    'thermal_deg_phase', 'mechanical_deg_phase', 'tyre_warm_up_laps',
    # Group 4: Braking
    'brake_point_consistency', 'trail_braking_index', 'brake_release_rate',
    # Group 5: Pace
    'quali_vs_race_delta', 'quali_pace_gap_pct', 'setup_sensitivity',
    # Track context
    'circuit_cluster',
]

TARGET_COL = 'race_pace_gap_pct'


def prepare_features(feature_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Prepare feature matrix, target, and season arrays from raw feature DataFrame.
    Returns (X, y, seasons, feature_names).
    """
    available_features = [f for f in LEGACY_FEATURES if f in feature_df.columns]

    df = feature_df.dropna(subset=[TARGET_COL]).copy()
    if df.empty:
        return np.array([]), np.array([]), np.array([]), available_features

    X = df[available_features].copy()
    # Fill NaN with column median
    for col in X.columns:
        median_val = X[col].median()
        X[col] = X[col].fillna(median_val if not np.isnan(median_val) else 0)

    y = df[TARGET_COL].values
    seasons = df['season'].values

    return X.values, y, seasons, available_features


def train_legacy_model(feature_df: pd.DataFrame,
                         save: bool = True) -> tuple[xgb.XGBRegressor, list[tuple]]:
    """
    Train the legacy XGBoost model with time-aware cross-validation.

    Returns:
        (trained_model, list of (test_season, MAE) tuples)
    """
    X, y, seasons, feature_names = prepare_features(feature_df)

    if len(X) == 0:
        logger.error("No valid training data!")
        return None, []

    logger.info(f"Training legacy model: {len(X)} samples, {len(feature_names)} features")

    maes = []
    best_model = None

    for X_train, X_test, y_train, y_test, test_year in walk_forward_cv(X, y, seasons):
        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        maes.append((test_year, mae))
        logger.info(f"  {test_year}: MAE = {mae:.4f}%")

        best_model = model  # Keep the last (most recent test) model

    if save and best_model is not None:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, MODEL_DIR / 'legacy_model.joblib')
        joblib.dump(feature_names, MODEL_DIR / 'legacy_feature_names.joblib')
        logger.info("Saved legacy model")

    return best_model, maes


def predict_legacy(feature_df: pd.DataFrame,
                     model: xgb.XGBRegressor = None) -> pd.DataFrame:
    """
    Generate predictions using the legacy model.
    Returns DataFrame with team, circuit, predicted_pace_gap_pct.
    """
    if model is None:
        model = joblib.load(MODEL_DIR / 'legacy_model.joblib')

    feature_names = joblib.load(MODEL_DIR / 'legacy_feature_names.joblib')
    available = [f for f in feature_names if f in feature_df.columns]

    X = feature_df[available].copy()
    for col in X.columns:
        median_val = X[col].median()
        X[col] = X[col].fillna(median_val if not np.isnan(median_val) else 0)

    # Pad missing features with zeros
    for f in feature_names:
        if f not in X.columns:
            X[f] = 0

    X = X[feature_names]
    predictions = model.predict(X.values)

    result = feature_df[['team', 'circuit', 'season', 'round']].copy()
    result['predicted_pace_gap_pct'] = predictions

    return result


def explain_legacy_model(model: xgb.XGBRegressor = None,
                           feature_df: pd.DataFrame = None,
                           top_n: int = 15) -> pd.DataFrame:
    """
    Compute SHAP feature importance for the legacy model.
    Returns DataFrame sorted by importance.
    """
    if model is None:
        model = joblib.load(MODEL_DIR / 'legacy_model.joblib')

    feature_names = joblib.load(MODEL_DIR / 'legacy_feature_names.joblib')

    if feature_df is not None:
        X, _, _, _ = prepare_features(feature_df)
    else:
        # Create sample data for explanation
        X = np.random.randn(100, len(feature_names))

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    importance = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0),
    }).sort_values('mean_abs_shap', ascending=False)

    return importance.head(top_n)


def get_top3_accuracy(predictions_df: pd.DataFrame,
                        actuals_df: pd.DataFrame) -> float:
    """
    Compute top-3 team rank accuracy across races.
    What fraction of races have the correct 3 fastest teams predicted?
    """
    correct = 0
    total = 0

    for (season, round_num), group in predictions_df.groupby(['season', 'round']):
        actual = actuals_df[
            (actuals_df['season'] == season) &
            (actuals_df['round'] == round_num)
        ]

        if actual.empty or group.empty:
            continue

        pred_top3 = set(group.nsmallest(3, 'predicted_pace_gap_pct')['team'])
        actual_top3 = set(actual.nsmallest(3, 'race_pace_gap_pct')['team'])

        if len(pred_top3 & actual_top3) >= 2:  # At least 2 of 3 correct
            correct += 1
        total += 1

    return correct / total if total > 0 else 0
