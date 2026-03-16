# F1 Telemetry AI — Feature Matrix: Pre-2026 Era (2018–2025)

Legacy model training data | Ground-effect era primary window (2022–2025) | FastF1 / OpenF1 / Ergast

---

> **How to use this document:** This document defines every feature for the legacy (pre-2026) model. Implement all groups before training. Validate each group against known domain ground truth before proceeding to the next. All features use FastF1 telemetry unless noted. Strip safety car laps, outlaps, and inlaps before computing any feature.

> **Team-averaging rule (apply everywhere):** All car-level features must be computed as the average of both drivers of a team for a given circuit/session. This isolates car performance from driver skill. If only one driver is fast, the signal is driver-level and should not feed the car performance model.

---

## Group 1: Power Unit & Energy Deployment

Features capturing PU performance, ERS deployment quality, and DRS exploitation.

**NOTE:** `drs_delta_speed` and `drs_activation_rate` are RETIRED in 2026. `mgu_h_proxy` is RETIRED in 2026. All others are recalibrated.

| Feature Name | Description | How to Compute |
|---|---|---|
| `pu_vmax_avg` | Average peak speed per straight per race | Max speed within each DRS zone per lap; mean across all stint laps (exclude outlap/inlap) |
| `ers_deployment_consistency` | Std dev of speed gain after corner exit | Speed delta at 100m / 200m / 300m after each classified apex; std across all laps — low = consistent ERS |
| `mgu_h_proxy` | Turbo response quality — MGU-H era only. **RETIRE in 2026** | Time from throttle application to 90% of peak speed after corner exit; lower = faster turbo response |
| `energy_harvest_efficiency` | Estimated time cost of lift-and-coast harvesting events | Detect lift-and-coast events (throttle drop without braking); frequency × duration × speed loss estimate |
| `drs_delta_speed` | Speed gain when DRS opens. **RETIRE in 2026** | Speed at DRS detection point vs. speed 200m downstream on same straight; mean across race |
| `drs_activation_rate` | Fraction of laps where driver is within 1s to trigger DRS. **RETIRE in 2026** | DRS activation events / total race laps |
| `battery_depletion_signature` | Speed drop in final 20% of straights — early super-clip signal | Speed trace: detect plateau or decline while throttle >95% at straight end; frequency and magnitude |

---

## Group 2: Aerodynamic Performance

Features capturing the aero setup philosophy and grip characteristics of each car.

**NOTE:** `dirty_air_sensitivity` and `floor_load_sensitivity` are RETIRED in 2026 due to active aero and removal of Venturi tunnels.

| Feature Name | Description | How to Compute |
|---|---|---|
| `corner_speed_vs_straight_speed` | Downforce/drag tradeoff proxy. Recalibrate in 2026 | `avg_min_corner_speed / v_max` — higher ratio = higher downforce setup; compute per circuit |
| `high_speed_corner_grip` | Aerodynamic grip in fast corners (>180 km/h apex speed) | Mean minimum speed across corners classified as fast; team-averaged across both drivers |
| `low_speed_corner_grip` | Mechanical grip in slow corners (<100 km/h apex speed) | Mean minimum speed across corners classified as slow; team-averaged |
| `sector1_vs_sector3_delta` | High-speed vs. technical sector balance | Normalized gap to pole per sector per event; S1 = aero-sensitive, S3 = mechanical-sensitive for most circuits |
| `dirty_air_sensitivity` | Pace loss when following another car. **RETIRE in 2026** | Lap time delta when gap to car ahead <2s vs. laps in clean air; requires position data from OpenF1 |
| `floor_load_sensitivity` | How ride height affects pace. **RETIRE in 2026 — Venturi tunnels gone** | Correlate lap time variation with known track surface profile roughness per circuit |

---

## Group 3: Tire Degradation

The highest-value feature group. **All features here are STABLE across both eras** — tire physics does not change with regulations. Implement this group first and validate thoroughly before any other group.

| Feature Name | Description | How to Compute |
|---|---|---|
| `deg_rate_soft` / `deg_rate_medium` / `deg_rate_hard` | Lap time loss per stint lap per compound | Linear regression: `LapTime_seconds ~ StintLap` within clean stints; slope = deg rate. Separate model per compound. Exclude outlap, inlap, SC laps |
| `compound_delta` | Pace delta between soft and medium (setup/thermal sensitivity proxy) | Mean pace delta soft→medium for same team at same circuit, normalized against field average |
| `thermal_deg_phase` | Early-stint degradation rate (laps 1–5 of stint) | Separate linear regression on StintLap 1–5 only; steeper slope = higher thermal deg sensitivity |
| `mechanical_deg_phase` | Late-stint cliff behaviour | Pace delta on final 5 laps of typical stint vs. median pace laps 6–15; captures cliff not just linear deg |
| `undercut_vulnerability` | Time lost per lap on aged vs. fresh tires | Used tire pace delta vs. fresh tire baseline at same track; drives undercut strategy modelling |
| `tyre_warm_up_laps` | Laps to reach peak pace after pit stop | Laps until rolling std dev of lap time stabilizes post-pit; per compound per team per circuit type |

> **Implementation note:** Use `fastf1.laps.pick_quicklaps()` as a starting filter, but tune the threshold. The default may be too aggressive. Manually verify a sample of retained laps for 2–3 teams before trusting the pipeline.

---

## Group 4: Braking & Corner Entry

Braking mechanics are fully **STABLE** across eras. `rotation_balance` requires recalibration in 2026 due to narrower car and flat floor.

| Feature Name | Description | How to Compute |
|---|---|---|
| `brake_point_consistency` | Std dev of braking point distance across identical corners | Distance from corner entry to point where brake pressure first exceeds 20%; std across all laps at same corner. Lower = more consistent |
| `trail_braking_index` | Degree of brake-steer overlap at corner entry | Correlation coefficient between brake pressure trace and lateral-g trace in the entry phase of each corner; higher = more trail braking |
| `brake_release_rate` | Speed of brake pressure drop approaching apex | Derivative (gradient) of brake pressure trace in final 30% of braking zone; steeper drop = sharper release |
| `rotation_balance` | Oversteer/understeer signature. Recalibrate in 2026 | Steering angle required to hold racing line vs. theoretical Ackermann minimum for corner radius and speed; positive = understeer bias |

---

## Group 5: Qualifying vs. Race Pace

Features distinguishing single-lap pace from race management capability. **All STABLE** — carry into 2026 with recalculated values.

| Feature Name | Description | How to Compute |
|---|---|---|
| `quali_vs_race_delta` | Whether car is a quali specialist or a race car | `(quali_gap_to_pole_pct) - (race_pace_gap_to_leader_pct)`. Positive = better in race trim than qualifying. Key car character signal |
| `quali_pace_gap_pct` | Single-lap gap to pole as normalized % of pole time | `((team_best_quali_time - pole_time) / pole_time) * 100`; normalize per circuit to remove track length bias |
| `race_pace_gap_pct` | Race pace gap to the fastest team | Median lap time in laps 5 through (stint_end - 5) per stint, excluding all non-representative laps; normalized |
| `setup_sensitivity` | Variance in performance weekend to weekend | Std dev of `race_pace_gap_pct` across all rounds in a season; high variance = sensitive or complex setup window |
| `wet_vs_dry_delta` | Pace change in wet vs. dry conditions | Pace gap delta between wet-classified sessions and dry sessions at same circuit; requires Weather API data |

---

## Group 6: Track Characterization

These features describe the circuit, not the car. Used for unsupervised K-Means/DBSCAN clustering to group circuits by performance archetype (e.g., "high-power", "technical", "street", "mixed"). Feed cluster assignment as a categorical feature into the prediction model.

| Feature Name | Description | Source / How to Compute |
|---|---|---|
| `pct_full_throttle` | Fraction of lap at >95% throttle | FastF1 throttle channel; compute per session, average across representative laps |
| `pct_heavy_braking` | Fraction of lap at >80% brake pressure | FastF1 brake channel; same approach as throttle |
| `avg_corner_speed` | Mean apex speed across all classified corners | Classify corners as local minima in speed trace; average minimum speed per corner across laps |
| `num_slow_corners` | Count of corners with apex speed <100 km/h | Corner classification from speed trace local minima |
| `num_fast_corners` | Count of corners with apex speed >180 km/h | Corner classification from speed trace local minima |
| `track_evolution_rate` | Lap time improvement across session from rubber lay-in | Linear regression slope of all-driver median lap time vs. session lap number; steeper = more track evolution |
| `energy_recovery_potential` | Braking energy available per lap | Sum of all deceleration events × estimated car mass (798 kg in 2025) × speed delta per event |
| `surface_abrasion_proxy` | Track surface harshness estimate | Team-average `deg_rate_medium` normalized by ambient temperature and compound age; circuit-level aggregate |
| `is_street_circuit` | Boolean: is this a temporary street circuit? | Manual label: Monaco, Singapore, Baku, Las Vegas, Miami, Jeddah, Melbourne = 1 |
| `altitude_m` | Circuit altitude above sea level | Track metadata; affects PU cooling and fuel burn |
| `lap_length_km` | Circuit lap length in km | Track metadata; used to normalize energy and distance features |

> **Clustering guidance:** Start with K-Means k=5 on standardized track features. Expect clusters roughly matching: (1) high-speed/power (Monza, Spa, Silverstone), (2) technical/mechanical (Monaco, Hungary, Singapore), (3) mixed-character (Suzuka, Bahrain), (4) street/bumpy (Baku, Jeddah), (5) modern purpose-built (COTA, Losail). Validate that cluster assignments are stable year-on-year before using as a model feature.

---

## Feature Pipeline Skeleton

```python
import fastf1
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

fastf1.Cache.enable_cache('./f1_cache')  # critical — API calls are slow

# ─── 1. Load session data ────────────────────────────────────────────────────
def load_session(year, round_num, session_type='R'):
    session = fastf1.get_session(year, round_num, session_type)
    session.load(telemetry=True, weather=True, laps=True)
    return session

# ─── 2. Clean laps ───────────────────────────────────────────────────────────
def get_clean_laps(session):
    laps = session.laps.copy()
    laps = laps[laps['IsAccurate'] == True]       # FastF1 accurate lap flag
    laps = laps[laps['PitOutTime'].isna()]         # remove outlaps
    laps = laps[laps['PitInTime'].isna()]          # remove inlaps
    # Remove safety car / VSC laps via track status
    laps = laps[~laps['TrackStatus'].str.contains('4|5|6|7', na=False)]
    return laps

# ─── 3. Tire degradation ─────────────────────────────────────────────────────
def compute_deg_rate(clean_laps, driver, compound):
    stint = clean_laps[
        (clean_laps['Driver'] == driver) &
        (clean_laps['Compound'] == compound)
    ].copy()
    stint['StintLap'] = range(len(stint))
    if len(stint) < 4:
        return None
    lt_seconds = stint['LapTime'].dt.total_seconds()
    coeffs = np.polyfit(stint['StintLap'], lt_seconds, deg=1)
    return {
        'driver': driver,
        'compound': compound,
        'deg_rate': coeffs[0],   # seconds per lap — the key feature
        'base_pace': coeffs[1]   # intercept
    }

# ─── 4. Aero signature ───────────────────────────────────────────────────────
def compute_aero_signature(telemetry):
    v_max = telemetry['Speed'].max()
    from scipy.signal import find_peaks
    inv_speed = -telemetry['Speed'].values
    peaks, _ = find_peaks(inv_speed, distance=50, prominence=20)
    corner_speeds = telemetry['Speed'].values[peaks]
    corner_speeds = corner_speeds[corner_speeds < 200]
    return {
        'v_max': v_max,
        'avg_min_corner_speed': corner_speeds.mean() if len(corner_speeds) else np.nan,
        'corner_vs_straight': corner_speeds.mean() / v_max if len(corner_speeds) else np.nan
    }

# ─── 5. Track clustering ─────────────────────────────────────────────────────
def cluster_circuits(track_features_df, k=5):
    features = [
        'pct_full_throttle', 'pct_heavy_braking', 'avg_corner_speed',
        'num_slow_corners', 'energy_recovery_potential'
    ]
    X = StandardScaler().fit_transform(track_features_df[features].fillna(0))
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    track_features_df['circuit_cluster'] = km.fit_predict(X)
    return track_features_df, km

# ─── 6. Target variable ──────────────────────────────────────────────────────
def compute_race_pace_gap(race_laps):
    """Normalized race pace gap to fastest team for each team (primary target variable)."""
    clean = race_laps[race_laps['IsAccurate']].copy()
    clean = clean[clean['TyreLife'] > 4]  # remove in/outlap noise
    team_pace = (
        clean.groupby('Team')['LapTime']
        .apply(lambda x: x.dt.total_seconds().median())
        .reset_index(name='median_pace')
    )
    fastest = team_pace['median_pace'].min()
    team_pace['pace_gap_pct'] = ((team_pace['median_pace'] - fastest) / fastest) * 100
    return team_pace
```
