"""
Circuit archetype clustering using K-Means on track characterization features.
Produces cluster assignments used as categorical features in prediction models.

Expected clusters (~k=5):
1. High-speed/power (Monza, Spa, Silverstone)
2. Technical/mechanical (Monaco, Hungary, Singapore)
3. Mixed-character (Suzuka, Bahrain)
4. Street/bumpy (Baku, Jeddah)
5. Modern purpose-built (COTA, Losail)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib
from pathlib import Path


CLUSTERING_FEATURES = [
    'pct_full_throttle',
    'pct_heavy_braking',
    'avg_corner_speed',
    'num_slow_corners',
    'num_fast_corners',
    'energy_recovery_potential',
    'is_street_circuit',
    'altitude_m',
]

MODEL_DIR = Path(__file__).parent.parent.parent / 'data' / 'processed'


def cluster_circuits(track_features_df: pd.DataFrame, k: int = 5,
                      save: bool = True) -> tuple[pd.DataFrame, KMeans, StandardScaler]:
    """
    Cluster circuits into archetypes using K-Means.

    Args:
        track_features_df: DataFrame with one row per circuit, columns matching CLUSTERING_FEATURES
        k: Number of clusters
        save: Whether to save the fitted scaler and model

    Returns:
        (track_features_df with 'circuit_cluster' column, fitted KMeans, fitted StandardScaler)
    """
    available_features = [f for f in CLUSTERING_FEATURES if f in track_features_df.columns]

    X = track_features_df[available_features].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    track_features_df = track_features_df.copy()
    track_features_df['circuit_cluster'] = km.fit_predict(X_scaled)

    # Compute silhouette score for quality check
    if len(X_scaled) > k:
        sil_score = silhouette_score(X_scaled, track_features_df['circuit_cluster'])
        print(f"Circuit clustering silhouette score: {sil_score:.3f}")

    if save:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(km, MODEL_DIR / 'circuit_kmeans.joblib')
        joblib.dump(scaler, MODEL_DIR / 'circuit_scaler.joblib')
        joblib.dump(available_features, MODEL_DIR / 'circuit_features.joblib')

    return track_features_df, km, scaler


def predict_circuit_cluster(track_features: dict) -> int:
    """
    Predict the cluster for a new or upcoming circuit using saved model.
    """
    km = joblib.load(MODEL_DIR / 'circuit_kmeans.joblib')
    scaler = joblib.load(MODEL_DIR / 'circuit_scaler.joblib')
    feature_names = joblib.load(MODEL_DIR / 'circuit_features.joblib')

    X = np.array([[track_features.get(f, 0) for f in feature_names]])
    X_scaled = scaler.transform(X)
    return int(km.predict(X_scaled)[0])


def find_optimal_k(track_features_df: pd.DataFrame,
                    k_range: range = range(3, 8)) -> dict:
    """
    Find optimal number of clusters using silhouette score.
    Returns dict of {k: silhouette_score}.
    """
    available_features = [f for f in CLUSTERING_FEATURES if f in track_features_df.columns]
    X = track_features_df[available_features].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    scores = {}
    for k in k_range:
        if k >= len(X_scaled):
            break
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        scores[k] = silhouette_score(X_scaled, labels)

    return scores


def get_cluster_profiles(track_features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Get the mean feature values per cluster for interpretation.
    """
    if 'circuit_cluster' not in track_features_df.columns:
        return pd.DataFrame()

    available_features = [f for f in CLUSTERING_FEATURES if f in track_features_df.columns]
    return track_features_df.groupby('circuit_cluster')[available_features].mean()


def check_cluster_stability(track_features_by_year: dict[int, pd.DataFrame],
                              k: int = 5) -> float:
    """
    Check whether cluster assignments are stable year-on-year.
    Returns adjusted Rand index between consecutive years.
    """
    from sklearn.metrics import adjusted_rand_score

    years = sorted(track_features_by_year.keys())
    ari_scores = []

    prev_labels = None
    prev_circuits = None

    for year in years:
        df = track_features_by_year[year]
        available_features = [f for f in CLUSTERING_FEATURES if f in df.columns]
        X = df[available_features].fillna(0).values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)

        if prev_labels is not None and prev_circuits is not None:
            # Find common circuits
            current_circuits = set(df.index if df.index.name else range(len(df)))
            common = current_circuits & prev_circuits
            if len(common) > k:
                # Compare labels for common circuits only
                ari = adjusted_rand_score(
                    prev_labels[:len(common)], labels[:len(common)])
                ari_scores.append(ari)

        prev_labels = labels
        prev_circuits = set(df.index if df.index.name else range(len(df)))

    return np.mean(ari_scores) if ari_scores else np.nan
