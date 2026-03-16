# F1 Telemetry AI — Tools, Stack & Code Patterns

Free-tier infrastructure | Python / FastF1 | XGBoost | Streamlit | Hugging Face Spaces

---

## Data Sources

| Source | What it provides | Access | Notes |
|---|---|---|---|
| FastF1 (Python library) | Lap times, telemetry (speed/throttle/brake/gear/DRS), tire data, weather, session data — 2018 to present | `pip install fastf1` | Primary data source. Cache aggressively — raw API downloads are slow. Use `fastf1.Cache.enable_cache()` |
| OpenF1 API (REST) | Real-time and historical race data: lap times, positions, car data, stints, pit stops — 2023 to present | REST API, no auth required. `pip install requests` | Better for 2026 data; has a `/car_data` endpoint for per-sample telemetry. Supplements FastF1 for new features |
| Ergast API (REST) | Historical race results, standings, constructor data — 1950 to present. Being archived but data still accessible | REST API, no auth. `pip install requests` | Use for historical DNF/result data to populate reliability features. Mirror locally early — the API may go offline |
| Jolpica API | Unofficial Ergast mirror. Same schema, maintained post-Ergast sunset | REST API, no auth | Use as drop-in Ergast replacement if Ergast goes down |

> **Cache everything:** FastF1 downloads large session files per race. On first run, a full season load can take 20–40 minutes. Set up a persistent cache directory on Google Drive or Kaggle persistent storage. Never run without caching enabled in production or CI.

---

## Free Compute Options

| Platform | Free Tier | Best For | Limitations |
|---|---|---|---|
| Google Colab | CPU always free; T4 GPU ~4 hrs/day | Exploration, feature engineering, model training | Session disconnects after idle. Mount Google Drive for persistence |
| Kaggle Notebooks | 30 hrs/week GPU (P100/T4); persistent storage | Training runs; F1 community datasets already available | GPU quota resets weekly; no outbound internet for some packages |
| Hugging Face Spaces | Free CPU inference hosting (Streamlit/Gradio) | Deploying the dashboard as a persistent web app | Cold starts on free tier; limited RAM (~16GB); no GPU on free tier |
| GitHub Codespaces | 60 hrs/month free (2-core/4GB) | Development and testing; running the feature pipeline locally | Not ideal for training; better for code editing and light data work |

> **Recommended workflow:** Develop and explore in Google Colab with Drive-mounted cache. Run serious training jobs on Kaggle (better persistent storage and more reliable GPU quota). Deploy dashboard to Hugging Face Spaces. Keep all code in a GitHub repo connecting the three.

---

## Python Library Stack

| Library | Version | Role | Install |
|---|---|---|---|
| `fastf1` | >=3.3 | Primary telemetry and session data source | `pip install fastf1` |
| `pandas` | >=2.0 | Tabular data manipulation; core data structure throughout pipeline | `pip install pandas` |
| `numpy` | >=1.26 | Numerical operations; polyfit for deg rate, signal processing | `pip install numpy` |
| `scipy` | >=1.12 | `find_peaks` for corner detection; signal smoothing | `pip install scipy` |
| `scikit-learn` | >=1.4 | K-Means circuit clustering; StandardScaler; cross-validation utils | `pip install scikit-learn` |
| `xgboost` | >=2.0 | Primary prediction model for both legacy and 2026 models | `pip install xgboost` |
| `lightgbm` | >=4.3 | Alternative to XGBoost; faster on large feature tables | `pip install lightgbm` |
| `shap` | >=0.45 | Feature importance explanation; validates model logic against domain knowledge | `pip install shap` |
| `streamlit` | >=1.35 | Dashboard framework; simple to deploy on Hugging Face Spaces | `pip install streamlit` |
| `plotly` | >=5.20 | Interactive charts for the dashboard | `pip install plotly` |
| `requests` | >=2.31 | OpenF1 and Ergast/Jolpica API calls | `pip install requests` |
| `joblib` | >=1.4 | Model serialisation; save/load trained XGBoost models | `pip install joblib` |

---

## Recommended Project Structure

```
f1-telemetry-ai/
├── data/
│   ├── cache/                  # FastF1 cache (gitignore this)
│   ├── raw/                    # Cached OpenF1 / Ergast JSON responses
│   └── processed/              # Saved feature tables as parquet files
│
├── src/
│   ├── ingestion/
│   │   ├── fastf1_loader.py    # Session loading and lap cleaning
│   │   ├── openf1_client.py    # OpenF1 REST API wrapper
│   │   └── ergast_client.py    # Ergast/Jolpica result data
│   │
│   ├── features/
│   │   ├── legacy/             # Pre-2026 feature implementations
│   │   │   ├── power_unit.py
│   │   │   ├── aero.py
│   │   │   ├── tires.py
│   │   │   ├── braking.py
│   │   │   ├── pace.py
│   │   │   └── track.py
│   │   ├── era2026/            # 2026-specific feature implementations
│   │   │   ├── active_aero.py
│   │   │   ├── boost_mode.py
│   │   │   ├── overtake_mode.py
│   │   │   ├── superclip.py
│   │   │   ├── turbo_launch.py
│   │   │   └── reliability.py
│   │   ├── track_clustering.py # Circuit archetype clustering
│   │   └── pipeline.py         # Orchestrates full feature build
│   │
│   ├── models/
│   │   ├── legacy_model.py     # Train/evaluate pre-2026 XGBoost
│   │   ├── model_2026.py       # Train/evaluate 2026 XGBoost with transfer
│   │   ├── priors.py           # PU supplier and reliability priors
│   │   └── validation.py       # Time-aware CV utilities
│   │
│   └── dashboard/
│       ├── app.py              # Streamlit entrypoint
│       ├── pages/
│       │   ├── race_prediction.py
│       │   ├── circuit_profile.py
│       │   ├── team_heatmap.py
│       │   └── reliability_tracker.py
│       └── components/         # Reusable Plotly chart functions
│
├── notebooks/                  # Exploration and validation notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_validation.ipynb
│   ├── 03_legacy_model_training.ipynb
│   └── 04_2026_transfer_model.ipynb
│
├── tests/
│   ├── test_tires.py           # Validate deg_rate on known data
│   └── test_reliability.py     # Validate DNF classification
│
├── requirements.txt
├── .gitignore                  # Include: data/cache/, data/raw/, *.parquet
└── README.md
```

---

## Model Architecture Patterns

### Target Variable

Use normalized race pace gap to the fastest team (in %) as the primary target. More stable than finishing position (heavily influenced by safety cars/strategy) and more informative than absolute lap time (varies by circuit).

```python
def compute_race_pace_gap(race_laps):
    """
    Returns normalized race pace gap to fastest team for each team.
    Strips first 4 and last 4 laps of each stint to remove in/outlap noise.
    """
    clean = race_laps[race_laps['IsAccurate']].copy()
    clean = clean[clean['TyreLife'] > 4]
    team_pace = (
        clean.groupby('Team')['LapTime']
        .apply(lambda x: x.dt.total_seconds().median())
        .reset_index(name='median_pace')
    )
    fastest = team_pace['median_pace'].min()
    team_pace['pace_gap_pct'] = ((team_pace['median_pace'] - fastest) / fastest) * 100
    return team_pace
```

### Time-Aware Cross-Validation

Never use standard k-fold. Always train on earlier data and test on later data.

```python
def walk_forward_cv(X, y, seasons, n_test_seasons=1):
    """
    Time-aware cross-validation for F1 data.
    Always trains on earlier seasons, tests on later ones.
    seasons: array of season years aligned with X rows.
    """
    unique_seasons = sorted(np.unique(seasons))
    for i in range(n_test_seasons, len(unique_seasons)):
        test_season = unique_seasons[i]
        train_seasons = unique_seasons[:i]
        train_mask = np.isin(seasons, train_seasons)
        test_mask  = seasons == test_season
        yield X[train_mask], X[test_mask], y[train_mask], y[test_mask], test_season
```

### Legacy Model Training

```python
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

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
    'circuit_cluster',  # from K-Means
]

def train_legacy_model(feature_df, target_col='pace_gap_pct'):
    X = feature_df[LEGACY_FEATURES].fillna(feature_df[LEGACY_FEATURES].median())
    y = feature_df[target_col]
    seasons = feature_df['season'].values
    maes = []
    for X_train, X_test, y_train, y_test, test_year in walk_forward_cv(X.values, y.values, seasons):
        model = xgb.XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,  # L1+L2 regularisation
            random_state=42
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=30,
            verbose=False
        )
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        maes.append((test_year, mae))
        print(f'  {test_year}: MAE = {mae:.4f}%')
    return model, maes
```

### 2026 Transfer Model

```python
# Features from legacy model that are physics-stable — the transfer anchors
LATENT_FEATURES = [
    'deg_rate_medium', 'thermal_deg_phase', 'mechanical_deg_phase',
    'high_speed_corner_grip', 'low_speed_corner_grip',
    'brake_point_consistency', 'trail_braking_index',
    'corner_speed_vs_straight_speed',  # recalibrated on 2026 data
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

def build_2026_feature_matrix(legacy_df, era2026_df):
    """
    Combine stable legacy features with new 2026 features.
    legacy_df and era2026_df must share team + circuit index.
    """
    X_latent = legacy_df[LATENT_FEATURES]
    X_new    = era2026_df[ERA2026_FEATURES]
    return pd.concat([X_latent, X_new], axis=1)

def train_2026_model(X_2026, y_2026, n_races_available):
    """
    Early season: fewer races, stronger regularisation, rely more on priors.
    Later season: loosen regularisation as data accumulates.
    """
    reg_strength = max(0.1, 2.0 - (n_races_available * 0.15))  # decays with data
    model = xgb.XGBRegressor(
        n_estimators=200, max_depth=3,  # shallow early season
        learning_rate=0.05,
        reg_alpha=reg_strength,
        reg_lambda=reg_strength * 2,
        random_state=42
    )
    model.fit(X_2026, y_2026)
    return model
```

### SHAP Feature Importance Validation

Run SHAP after every training run. If tire degradation and corner speed features are not in the top 5, the model has likely overfit to noise or there is a data quality problem.

```python
import shap

def explain_model(model, X, feature_names, top_n=15):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    # Summary plot — run in a notebook
    shap.summary_plot(shap_values, X, feature_names=feature_names, max_display=top_n)
    # Return mean absolute SHAP per feature for logging
    importance = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False)
    return importance
```

---

## Dashboard Architecture

### Streamlit App Structure

The dashboard is a multi-page Streamlit app. The entry point loads pre-computed prediction data from parquet files — it does not run the model live. Predictions are updated by a separate pipeline script run after each race.

```python
# app.py — Streamlit entrypoint
import streamlit as st

st.set_page_config(page_title='F1 Telemetry AI', layout='wide')

pages = {
    'Race Prediction':     'pages/race_prediction.py',
    'Circuit Profile':     'pages/circuit_profile.py',
    'Team Heatmap':        'pages/team_heatmap.py',
    'Reliability Tracker': 'pages/reliability_tracker.py',
}

# Each page loads from pre-computed parquet files — no live model inference
# Update pipeline: python src/features/pipeline.py && python src/models/model_2026.py
# Output: data/processed/predictions_latest.parquet
```

### Deploying to Hugging Face Spaces

```
# Directory structure for HF Spaces deployment
hf-space/
├── app.py                  # Streamlit entrypoint
├── requirements.txt        # Must include: streamlit, plotly, pandas, xgboost, shap
├── data/
│   └── predictions_latest.parquet  # Pre-computed; commit after each update
└── README.md               # HF Spaces config in YAML frontmatter

# README.md frontmatter:
# ---
# title: F1 Telemetry AI
# emoji: 🏎️
# colorFrom: red
# colorTo: gray
# sdk: streamlit
# sdk_version: 1.35.0
# app_file: app.py
# pinned: false
# ---

# Deploy:
# pip install huggingface_hub
# huggingface-cli upload YOUR_USERNAME/f1-telemetry-ai . --repo-type space
```

---

## 2026 Early-Season Playbook

The first 6 races of 2026 are the highest-uncertainty period. Follow this race-by-race update protocol:

| After race... | Action |
|---|---|
| Race 1 (Australia) | Pull telemetry. Seed all 2026 features with whatever is available. Set reliability priors from DNF taxonomy (race 1 results: Hadjar engine, Bottas hydraulic, Alonso/AM Honda vibration, Hulkenberg DNS). Do NOT train model yet — insufficient data. |
| Race 2 (China) | Update rolling reliability features. If 2+ data points available per team, fit a first-pass `turbo_spool_proxy` from race starts. Begin computing super-clip frequencies. |
| Race 3–4 | Train first 2026 model with maximum regularisation. Use all priors. Validate SHAP — reliability features should dominate early due to high variance in completion rates. |
| Race 5–6 | Retrain with slightly looser regularisation. Empirical `systemic_dnf_rate` should start to outweigh priors. Check whether turbo architecture cluster features have become predictive. |
| Race 7+ | Full model retrains every 3–4 races. Priors diminish. Treat as a normal ML pipeline from this point. |

> **Early-season over-fitting warning:** With only 4–6 races of 2026 data, any model with more than ~15 features and no strong regularisation will overfit badly. Keep `max_depth` at 3 and use strong L1/L2 penalties until race 8. The priors are your regulariser for the first half of the season.

---

## Quick-Start Checklist

Follow this order to get from zero to a working legacy model. Do not skip steps — each validates the next.

| Step | Task | Validation |
|---|---|---|
| 1 | `pip install fastf1 pandas numpy scipy scikit-learn xgboost shap streamlit plotly` | `import fastf1; fastf1.__version__` — should be >=3.3 |
| 2 | Set up cache: `fastf1.Cache.enable_cache('./f1_cache')` | Load one session `(2024, 1, 'R')` — should complete in <2 min on second run |
| 3 | Implement `get_clean_laps()` and verify lap counts | 2024 Australian GP should yield ~850–900 clean laps across all drivers |
| 4 | Implement `compute_deg_rate()` for one team, one compound | Mercedes medium compound in 2024 should show ~0.05–0.15 s/lap deg rate |
| 5 | Implement `compute_aero_signature()` for one circuit | Monza should give `corner_vs_straight` ~0.35; Monaco ~0.70 |
| 6 | Build full feature table for 2022–2024 seasons | Should produce ~200–250 rows (team × circuit per season) |
| 7 | Run K-Means track clustering with k=5 | Monza/Spa should cluster together; Monaco/Hungary/Singapore should cluster together |
| 8 | Train legacy XGBoost with `walk_forward_cv` on 2022–2024 | MAE on 2024 test should be <0.5% pace gap to leader |
| 9 | Run `explain_model()` with SHAP | Top 5 features should include: `deg_rate_medium`, `corner_speed_vs_straight_speed`, `quali_vs_race_delta` |
| 10 | Begin 2026 feature implementation starting with `superclip` and launch features | Verified against race 1 Australia: Ferrari-powered cars should show highest `turbo_spool_proxy` |
