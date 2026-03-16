"""
PU Supplier Architecture Priors and Reliability Priors for 2026 season.
Learnable priors — updated dynamically as race data accumulates.
Never hard-coded constants; always with explicit update mechanism.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

PRIORS_FILE = Path(__file__).parent.parent.parent / 'data' / 'processed' / 'pu_priors.json'

# Initial 2026 priors (from doc 03)
DEFAULT_PU_PRIORS = {
    'ferrari': {
        'teams': ['Ferrari', 'Haas F1 Team', 'Haas', 'Cadillac', 'MoneyGram Haas F1 Team'],
        'turbo_size': 1,  # Small
        'reliability_prior': 0.15,
        'notes': 'Confirmed small turbo; strong race 1 reliability',
    },
    'mercedes': {
        'teams': ['Mercedes', 'McLaren', 'Williams', 'Alpine',
                  'Mercedes-AMG PETRONAS F1 Team', 'McLaren F1 Team',
                  'Williams Racing', 'BWT Alpine F1 Team'],
        'turbo_size': 3,  # Large
        'reliability_prior': 0.15,
        'notes': 'Implied large turbo; clean race 1',
    },
    'red_bull_ford': {
        'teams': ['Red Bull Racing', 'RB', 'Racing Bulls',
                  'Oracle Red Bull Racing', 'Visa Cash App RB F1 Team'],
        'turbo_size': 2,  # Medium (unknown)
        'reliability_prior': 0.45,
        'notes': 'Architecture unknown; hydraulic failure race 1',
    },
    'honda': {
        'teams': ['Aston Martin', 'Aston Martin Aramco F1 Team'],
        'turbo_size': 2,  # Medium
        'reliability_prior': 0.85,
        'notes': 'Vibration/battery crisis since pre-season testing',
    },
    'audi': {
        'teams': ['Sauber', 'Kick Sauber', 'Audi', 'Stake F1 Team Kick Sauber'],
        'turbo_size': 2,  # Medium
        'reliability_prior': 0.70,
        'notes': 'New constructor; DNS race 1; no architecture data',
    },
}


def get_pu_priors() -> dict:
    """Load current PU priors. Returns defaults if no saved priors exist."""
    if PRIORS_FILE.exists():
        with open(PRIORS_FILE, 'r') as f:
            return json.load(f)
    return DEFAULT_PU_PRIORS.copy()


def save_pu_priors(priors: dict):
    """Save updated PU priors to disk."""
    PRIORS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PRIORS_FILE, 'w') as f:
        json.dump(priors, f, indent=2)


def update_priors_from_race_data(results_df: pd.DataFrame,
                                   feature_df: pd.DataFrame = None) -> dict:
    """
    Update PU priors using empirical race data.
    Updates turbo_size inference from turbo_spool_proxy and
    reliability_prior from systemic_dnf_rate.

    Args:
        results_df: Race results with dnf_category column
        feature_df: Feature table with turbo_spool_proxy column

    Returns:
        Updated priors dict
    """
    priors = get_pu_priors()

    for supplier, info in priors.items():
        teams = info['teams']

        # Update reliability prior from empirical data
        supplier_results = results_df[
            results_df['constructor_name'].isin(teams) |
            results_df['constructor_id'].isin([t.lower().replace(' ', '_') for t in teams])
        ]

        if not supplier_results.empty:
            n_starts = len(supplier_results)
            from ..features.era2026.reliability import SYSTEMIC_CATEGORIES
            systemic_dnfs = supplier_results['dnf_category'].isin(SYSTEMIC_CATEGORIES).sum()
            empirical_rate = systemic_dnfs / n_starts

            # Bayesian update: blend prior with empirical
            # Weight shifts toward empirical as data accumulates
            n_races = supplier_results['round'].nunique() if 'round' in supplier_results.columns else 1
            prior_weight = max(0.1, 1.0 - n_races * 0.1)
            empirical_weight = 1.0 - prior_weight

            info['reliability_prior'] = (
                prior_weight * info['reliability_prior'] +
                empirical_weight * empirical_rate
            )

        # Update turbo_size from turbo_spool_proxy if available
        if feature_df is not None and 'turbo_spool_proxy' in feature_df.columns:
            team_data = feature_df[feature_df['team'].isin(teams)]
            if not team_data.empty:
                avg_spool = team_data['turbo_spool_proxy'].mean()
                # Rank across all suppliers to infer relative turbo size
                info['empirical_spool_proxy'] = float(avg_spool)

    save_pu_priors(priors)
    return priors


def get_team_prior_features(team: str) -> dict:
    """
    Get prior-based features for a specific team.
    Returns turbo_size and reliability_prior.
    """
    priors = get_pu_priors()

    for supplier, info in priors.items():
        if team in info['teams']:
            return {
                'turbo_size_prior': info['turbo_size'],
                'reliability_prior': info['reliability_prior'],
                'pu_supplier': supplier,
            }

    # Unknown team
    return {
        'turbo_size_prior': 2,
        'reliability_prior': 0.5,
        'pu_supplier': 'unknown',
    }
