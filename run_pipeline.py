"""
Main pipeline runner.
Executes the full data ingestion -> feature engineering -> model training pipeline.
Run this to build/update the feature tables and train models.

Usage:
    python run_pipeline.py --era legacy --years 2022 2023 2024 2025
    python run_pipeline.py --era 2026 --races 6
    python run_pipeline.py --era all
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ingestion.fastf1_loader import enable_cache
from src.ingestion.ergast_client import build_results_with_dnf_classification
from src.features.pipeline import build_legacy_features, build_2026_features
from src.models.legacy_model import train_legacy_model, explain_legacy_model
from src.models.model_2026 import train_2026_model, explain_2026_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger('pipeline')

DATA_DIR = Path(__file__).parent / 'data' / 'processed'


def run_legacy_pipeline(years: list[int]):
    """Run the complete legacy pipeline: features + model."""
    logger.info(f"=== Legacy Pipeline: {years} ===")

    # Step 1: Build features
    logger.info("Step 1: Building legacy features...")
    team_features, track_features = build_legacy_features(years)
    logger.info(f"  Team features: {len(team_features)} rows")
    logger.info(f"  Track features: {len(track_features)} rows")

    if team_features.empty:
        logger.error("No features built. Check data availability.")
        return

    # Step 2: Save results data for reliability tracker
    logger.info("Step 2: Building results/DNF data...")
    results_df = build_results_with_dnf_classification(years)
    if not results_df.empty:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        results_df.to_parquet(DATA_DIR / 'results_with_dnf.parquet', index=False)
        logger.info(f"  Results: {len(results_df)} rows")

    # Step 3: Train model
    logger.info("Step 3: Training legacy model...")
    model, maes = train_legacy_model(team_features)

    if model is not None:
        # Step 4: SHAP explanation
        logger.info("Step 4: Computing SHAP importance...")
        shap_df = explain_legacy_model(model, team_features)
        shap_df.to_parquet(DATA_DIR / 'shap_importance.parquet', index=False)
        logger.info(f"  Top features: {list(shap_df.head(5)['feature'])}")

        # Step 5: Generate predictions (use latest season as "upcoming")
        logger.info("Step 5: Generating predictions...")
        from src.models.legacy_model import predict_legacy
        predictions = predict_legacy(team_features, model)
        predictions.to_parquet(DATA_DIR / 'predictions_latest.parquet', index=False)

        # Save metrics
        import pandas as pd
        metrics = pd.DataFrame(maes, columns=['test_season', 'mae'])
        metrics.to_parquet(DATA_DIR / 'model_metrics.parquet', index=False)

        logger.info("=== Legacy Pipeline Complete ===")
        for year, mae in maes:
            logger.info(f"  {year} MAE: {mae:.4f}%")
    else:
        logger.warning("Model training failed — insufficient data")


def run_2026_pipeline(races_available: int = None):
    """Run the 2026 pipeline: features + transfer model."""
    logger.info(f"=== 2026 Pipeline (races={races_available}) ===")

    # Step 1: Build 2026 features
    logger.info("Step 1: Building 2026 features...")
    features_2026 = build_2026_features(races_available=races_available)

    if features_2026.empty:
        logger.warning("No 2026 data available yet. Pipeline will activate "
                       "once race data is published via OpenF1/FastF1.")
        return

    logger.info(f"  2026 features: {len(features_2026)} rows")

    # Step 2: Update priors from race data
    logger.info("Step 2: Updating PU priors...")
    results_2026 = build_results_with_dnf_classification([2026])
    if not results_2026.empty:
        from src.models.priors import update_priors_from_race_data
        priors = update_priors_from_race_data(results_2026, features_2026)
        logger.info(f"  Updated priors for {len(priors)} PU suppliers")

    # Step 3: Train 2026 model
    n_races = races_available or features_2026['round'].nunique()
    if n_races >= 3:
        logger.info("Step 3: Training 2026 model...")
        model, maes = train_2026_model(features_2026, n_races)

        if model is not None:
            logger.info("Step 4: Computing SHAP importance...")
            shap_df = explain_2026_model(model, features_2026)
            shap_df.to_parquet(DATA_DIR / 'shap_importance_2026.parquet', index=False)

            from src.models.model_2026 import predict_2026
            predictions = predict_2026(features_2026, model)
            predictions.to_parquet(DATA_DIR / 'predictions_2026_latest.parquet', index=False)
    else:
        logger.info(f"Only {n_races} races available — need 3+ for model training. "
                    "Using priors only.")

    logger.info("=== 2026 Pipeline Complete ===")


def main():
    parser = argparse.ArgumentParser(description='F1 Telemetry AI Pipeline')
    parser.add_argument('--era', choices=['legacy', '2026', 'all'],
                         default='legacy', help='Which era to process')
    parser.add_argument('--years', nargs='+', type=int,
                         default=[2022, 2023, 2024, 2025],
                         help='Legacy seasons to process')
    parser.add_argument('--races', type=int, default=None,
                         help='Number of 2026 races available')

    args = parser.parse_args()

    enable_cache()

    if args.era in ('legacy', 'all'):
        run_legacy_pipeline(args.years)

    if args.era in ('2026', 'all'):
        run_2026_pipeline(args.races)


if __name__ == '__main__':
    main()
