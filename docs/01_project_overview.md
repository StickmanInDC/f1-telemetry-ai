# F1 Telemetry AI — Project Overview & Development Phases

Hobby project | Free-tier infrastructure | 2026 regulation-aware two-era architecture

---

## Project Overview

This project uses AI/ML to analyze Formula One telemetry data, derive performance characteristics of each team and car across different circuit types, and predict which cars are likely to perform well at upcoming races. It is a hobby project designed to run entirely on free-tier infrastructure.

**Goal:** Build a full prediction pipeline — from raw FastF1/OpenF1 telemetry through feature engineering, model training, and a Streamlit/Gradio dashboard — using only free compute and open data sources.

---

## Why Two Separate Models?

The 2026 F1 regulations are the most sweeping in the sport's history, changing both chassis and power unit simultaneously. A single unified model trained across both eras would conflate two fundamentally different performance regimes. The correct architecture is two distinct models connected by a transfer learning layer.

| Era | Training Data | Key Characteristics |
|-----|--------------|-------------------|
| Pre-2026 (Legacy) | 2018–2025 seasons | DRS, MGU-H, ground effect floors (2022+), traditional hybrid deployment, dirty-air sensitivity |
| 2026 Era | 2026 season onward | Active aero (Straight/Corner Mode), Boost Mode, Overtake Mode, super-clipping, no MGU-H, flat floors, ~50/50 ICE/ERS power split |

Physics-grounded features — tire degradation curves, braking mechanics, corner speed ratios — are stable across regulation changes and serve as the transfer layer. ERS deployment patterns, DRS behaviour, and dirty-air sensitivity are era-specific and must be fully replaced in 2026.

---

## System Architecture

### Pipeline Overview

```
FastF1 / OpenF1 API / Ergast API
        |
        v
Data Ingestion Layer
  - Cache to local / Google Drive
  - Strip safety car laps, outlaps, inlaps
  - One row per driver per lap (consistent schema)
        |
        v
Feature Engineering Pipeline  <-- see Feature Matrix documents
  - Legacy feature set (Groups 1-6)
  - 2026 feature set (active aero, Boost, OT mode, super-clip,
    turbo architecture, reliability)
        |
   _____|______
  |            |
  v            v
Legacy      2026 Era
Model       Model
(XGBoost)   (XGBoost + transfer features)
  |            |
  |____________|
        |
        v
Prediction Output
  - Race pace gap to leader (normalized %)
  - Constructor ranking probability per circuit cluster
  - SHAP explanation per prediction
        |
        v
Streamlit / Gradio Dashboard
  - Circuit profile view
  - Team performance heatmap
  - Reliability tracker
  - Upcoming race prediction panel
Deployed: Hugging Face Spaces (free tier)
```

### Two-Era Model Bridge

The legacy model is trained first on 2018–2025 data. A subset of its most stable, physics-grounded features — the latent layer — are extracted and used as inputs to the 2026 model alongside the new 2026-specific features. As race data accumulates through the 2026 season, the 2026 model is retrained every 3–4 races, progressively replacing priors with empirical signals.

```python
# Stable features carried from legacy into 2026 model (the transfer anchors)
LATENT_FEATURES = [
    'deg_rate_medium',          # tire physics unchanged
    'high_speed_corner_grip',   # corner mode still high downforce
    'brake_point_consistency',  # braking mechanics unchanged
    'corner_speed_vs_straight_speed',  # recalibrated but conceptually stable
    'quali_vs_race_delta',      # car character unchanged
    'setup_sensitivity',        # variance profile carries over
]

# 2026 input = stable latent features + new 2026-specific features
X_2026 = pd.concat([legacy_df[LATENT_FEATURES], new_2026_features], axis=1)
```

---

## Development Phases

Seven phases from data plumbing to in-season model updates. Do not proceed to Phase 3 until Phase 2 features pass domain validation.

### Phase 1 — Data Acquisition & Infrastructure (Weeks 1–2)

- Install FastF1, OpenF1 API client, Ergast API wrapper: `pip install fastf1 requests pandas`
- Set up Google Colab or Kaggle Notebook as primary compute — verify GPU allocation
- Build a persistent cache layer: FastF1 downloads are slow; cache raw session data to Google Drive or `/kaggle/working`
- Pull 2022–2025 seasons as the core legacy training window (ground-effect era); 2018–2021 as supplementary
- Establish consistent schema: one row per driver per lap, with `session_key`, `circuit_key`, `compound`, `stint_lap`, `tyre_age`
- Write data quality filters: flag and remove safety car laps, formation laps, outlaps, inlaps, red-flag sessions
- Confirm 2026 data is available via OpenF1 API as the season progresses (verify endpoint coverage post-race 1)

### Phase 2 — Feature Engineering: Legacy Era (Weeks 3–5)

- Implement tire degradation model: linear regression slope (`deg_rate`) per compound per stint — this is the single highest-value feature
- Compute aero signature: `corner_speed_vs_straight_speed` ratio per circuit per team
- Extract braking features: brake point distance, trail braking index, brake release rate
- Build PU/ERS features: DRS delta speed, activation rate, battery depletion signature
- Derive quali vs. race pace delta per team per event
- Implement team-averaging logic: average both drivers' pace features to isolate car performance from driver skill
- Build track characterization feature set: `pct_full_throttle`, corner speed distribution, `energy_recovery_potential`
- Run K-Means clustering (k=4–6) on track features to assign circuit archetypes
- Validate all features against known ground truth (e.g., Mercedes tire deg advantage 2020–2021, Red Bull straight-line speed 2023)

### Phase 3 — Legacy Prediction Model (Weeks 6–7)

- Define target variable: normalized race pace gap to leader in % (preferred) or finishing position
- Train XGBoost baseline on full 2022–2025 feature matrix
- Use **time-aware cross-validation ONLY** — train on earlier seasons, test on later ones (e.g., train 2022–2023, test 2024)
- Never use standard k-fold: teams copy solutions and improve rapidly, making random splits data-leak prone
- Compute SHAP values to validate feature importance matches domain knowledge
- Identify and document the stable latent features that will serve as transfer anchors for the 2026 model
- Benchmark: target MAE <0.5% on qualifying gap; top-3 team rank correct >65% of races

### Phase 4 — 2026 Feature Engineering (Weeks 8–9)

- Implement super-clipping detection: speed plateau at straight end while throttle >95%
- Build Boost Mode features: deployment pattern (track position), frequency per lap, energy per activation
- Implement Overtake Mode features: availability rate, conversion rate, speed delta, defense success rate
- Compute active aero features: straight mode activation timing, speed gain, corner mode grip delta
- Build turbo architecture cluster: `turbo_spool_proxy`, `launch_positions_gained`, `high_speed_straight_deficit`
- Implement reliability feature cluster: classify all DNFs by taxonomy, compute `systemic_dnf_rate`, `both_cars_dnf_rate`, `reliability_trend`
- Update track characterization: `energy_richness_score`, `superclip_opportunity_count`, `track_power_sensitivity_score`
- Seed learnable priors for PU supplier architecture and reliability scores from pre-season/race 1 data

### Phase 5 — 2026 Transfer Model (Weeks 10–11)

- Combine latent legacy features with new 2026 feature set into unified `X_2026` input matrix
- Train initial 2026 model on first 4–6 races of 2026 season data
- Use prior-weighted initialization: start model weights biased toward known architectural facts
- Validate that transfer learning outperforms a cold-start model trained on 2026 data alone
- Implement dynamic prior update: after each race, re-run feature pipeline and retrain
- Track SHAP importance drift: watch which 2026 features gain or lose predictive power as season progresses

### Phase 6 — Dashboard (Weeks 12–14)

- Build Streamlit or Gradio app with upcoming race prediction panel
- Circuit profile view: track cluster assignment, key characterization features, energy richness
- Team performance heatmap: feature scores across circuit archetypes
- Reliability tracker: DNF history by failure mode, rolling trend chart
- SHAP explanation panel: "why does the model favour this team at this circuit?"
- Deploy to Hugging Face Spaces (free tier) for persistent hosting with no server management

### Phase 7 — In-Season Model Updates (Ongoing — after each race)

- Pull new telemetry via OpenF1 API, re-run feature pipeline, update rolling_5 features
- Update PU supplier architecture priors from empirical `turbo_spool_proxy` data
- Update reliability priors from current `systemic_dnf_rate` — by race 6, empirical data should dominate priors
- Retrain or fine-tune 2026 model every 3–4 races
- Monitor and log when precautionary retirement patterns normalize (e.g., Aston Martin/Honda stabilization)
- Flag circuit clusters where predictions consistently over- or under-perform for manual review

---

## Core Design Principles

### Feature Engineering is the Whole Game
Raw telemetry is high-frequency, high-noise data sampled at 4Hz. It is not a useful ML input directly. All predictive power comes from how well telemetry is compressed into meaningful scalar features. A well-engineered 30-feature table with XGBoost will outperform raw telemetry fed to an LSTM. Invest the majority of development time in Phases 2 and 4, not in model architecture.

### Always Average Across Both Drivers
Lap time mixes car performance with driver skill. Before computing any pace-based feature, average across both drivers of a team for that circuit. If both Red Bull drivers are fast at Spa, it is the car. If only Verstappen is fast, it is the driver. This single step dramatically improves car-level signal quality.

### Physics-Grounded Features Transfer; Regulation-Specific Features Do Not
Tire degradation physics, braking mechanics, and corner speed ratios are governed by laws that do not change with regulations — these are the transfer anchors. DRS behaviour, MGU-H turbo response, and ground-effect floor sensitivity are tied to specific technical rules and must be retired in 2026, not repurposed.

### Learnable Priors, Never Hard-Coded Constants
PU supplier turbo architecture (small/large), reliability risk scores, and team-level starting estimates should be encoded as priors with an explicit update mechanism driven by real race data. Hard-coding known facts degrades predictions as gaps close during the season and teams bring upgrades.

### Time-Aware Validation Only
Never use standard k-fold cross-validation. F1 teams study each other intensively between races and seasons — performance is strongly autocorrelated in time. Use walk-forward validation: train on seasons N through N+k, test on season N+k+1. Shuffling the data introduces severe lookahead bias.

### Regulate Model Complexity to Data Volume
In 2026, you will start with very few races of data. Keep the model simple early in the season — a shallow XGBoost with strong regularisation and feature priors. As data accumulates, complexity can increase. A model with 20 features and 100 training examples is already at risk of overfitting; prune ruthlessly until race 8+.

---

## Success Metrics

| Metric | Target | How to Measure |
|--------|--------|---------------|
| Legacy model quali gap MAE | <0.5% gap to pole | Time-aware CV on 2022–2024, test on 2025 |
| Race pace top-3 rank accuracy | >65% of races correct | Which 3 teams are fastest in race trim |
| Circuit cluster stability | >90% same cluster year-on-year | Assign clusters per year, compare with adjusted Rand index |
| 2026 early-season prediction | Beats naive "same as last race" by race 6 | Compare MAE vs. persistence baseline |
| Reliability flag lead time | Flags systemic issues before 2nd DNF | Qualitative: did it flag Aston Martin/Honda before race 2? |
| Feature importance sanity | Top SHAP features match domain knowledge | Manual review: tire deg, corner speed, PU features should dominate |
