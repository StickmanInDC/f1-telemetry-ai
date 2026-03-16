# F1 Telemetry AI — Feature Matrix: 2026 Era

New regulations | Active aero | Boost & Overtake modes | Super-clipping | No MGU-H | Flat floors

---

> **How to use this document:** This document defines (1) which pre-2026 features to retire, recalibrate, or carry stable into 2026, and (2) every new feature cluster introduced for the 2026 era. Read `02_feature_matrix_legacy.md` first. This document is a delta on top of that baseline.

**Status definitions:**
- `STABLE` — Feature concept and values unchanged. Copy from legacy pipeline; recalculate on 2026 data only.
- `RECALIBRATE` — Same concept; baseline values shift with new regs. Retrain/refit on 2026 data; do not use pre-2026 fitted values.
- `RETIRE` — Feature is no longer valid under 2026 regulations. Remove from X_2026 input matrix entirely.
- `NEW` — Brand-new feature for 2026 era only. Implement from scratch; see feature groups below.

---

## Feature Transition Map

Full status of every pre-2026 feature when building the 2026 model.

| Feature (Pre-2026) | 2026 Status | Reason / Action |
|---|---|---|
| `pu_vmax_avg` | RECALIBRATE | Active aero changes straight-line speed profile; rescale baselines from 2026 data |
| `ers_deployment_consistency` | RECALIBRATE | ERS capacity doubles to 9MJ/lap — re-baseline entirely; pre-2026 std dev values not comparable |
| `mgu_h_proxy` | **RETIRE** | MGU-H removed from PU; turbo lag is now a hardware architecture problem not a deployment strategy |
| `energy_harvest_efficiency` | RECALIBRATE | Multiple simultaneous harvest modes in 2026; split into sub-features (see Recharge group) |
| `drs_delta_speed` | **RETIRE** | DRS no longer exists — replaced by active aero Straight Mode available to all cars |
| `drs_activation_rate` | **RETIRE** | DRS no longer exists — proximity passing now captured by Overtake Mode features |
| `battery_depletion_signature` | RECALIBRATE | Super-clipping is now the dominant signal; magnitude is much larger; threshold must be recalibrated |
| `corner_speed_vs_straight_speed` | RECALIBRATE | Active aero on both wings changes the ratio baseline; concept valid but retrain on 2026 data |
| `high_speed_corner_grip` | STABLE | Corner Mode is still high downforce; fast corner grip physics unchanged |
| `low_speed_corner_grip` | STABLE | Mechanical grip unchanged |
| `sector1_vs_sector3_delta` | STABLE | Concept unchanged; recalculate values from 2026 data |
| `dirty_air_sensitivity` | **RETIRE** | Active aero available to all cars changes following dynamics fundamentally; legacy values invalid |
| `floor_load_sensitivity` | **RETIRE** | Venturi tunnels removed; flat floors produce far less downforce — legacy values irrelevant |
| `deg_rate_soft/medium/hard` | STABLE | Tire physics unchanged; same computation on 2026 data |
| `compound_delta` | STABLE | Compound chemistry unchanged |
| `thermal_deg_phase` | STABLE | Unchanged |
| `mechanical_deg_phase` | STABLE | Unchanged |
| `undercut_vulnerability` | STABLE | Unchanged |
| `tyre_warm_up_laps` | STABLE | Unchanged |
| `brake_point_consistency` | STABLE | Braking physics unchanged |
| `trail_braking_index` | STABLE | Unchanged |
| `brake_release_rate` | STABLE | Unchanged |
| `rotation_balance` | RECALIBRATE | Narrower car (1900mm vs 2000mm) and flat floor change handling characteristics; retrain on 2026 data |
| `quali_vs_race_delta` | STABLE | Car character concept unchanged; recalculate from 2026 data |
| `quali_pace_gap_pct` | STABLE | Unchanged |
| `race_pace_gap_pct` | STABLE | Unchanged |
| `setup_sensitivity` | STABLE | Unchanged |
| `wet_vs_dry_delta` | STABLE | Unchanged |
| Track characterization (all) | RECALIBRATE + NEW | Existing features recalibrated; add: `energy_richness_score`, `superclip_opportunity_count`, `track_power_sensitivity_score`, `active_aero_zone_count` |

---

## NEW Features: Active Aerodynamics

Active aero replaces DRS with a full-wing system (front and rear) in two modes: **Straight Mode** (low drag, wings open on straights) and **Corner Mode** (high downforce, wings closed through corners). All cars can use it at all times in designated zones — there is no proximity requirement for Straight Mode.

| Feature Name | Description | How to Compute |
|---|---|---|
| `straight_mode_activation_timing` | How early in the straight the driver opens active aero | Distance from corner exit to aero-open event in telemetry; earlier opening = more aggressive, higher trust in car stability |
| `straight_mode_speed_gain` | Speed delta with wings open vs. closed baseline | Compare speed traces in active aero zones vs. equivalent laps without (e.g., wet lap or formation lap); mean across race |
| `corner_mode_grip_delta` | Downforce quality in corner mode vs. 2025 equivalent | Min corner speed in 2026 vs. same circuit 2025 for same team; delta quantifies net downforce change |
| `active_aero_consistency` | Std dev of activation timing and speed gain across laps | Low variance = driver and car confident in deployment; high variance = setup instability or driver hesitation |

---

## NEW Features: Boost Mode

Boost Mode allows the driver to manually deploy stored ERS energy at any point on the circuit — for attack, defense, or lap time. Unlike Overtake Mode, it has no proximity requirement. Effectiveness depends entirely on how much energy has been harvested in the lap to that point.

| Feature Name | Description | How to Compute |
|---|---|---|
| `boost_deployment_pattern` | Where on the lap Boost is typically deployed | Map boost activation events to track distance position; compute Shannon entropy — high entropy = spread deployment, low = concentrated at one zone |
| `boost_frequency_per_lap` | Average number of boost activations per lap | Count boost trigger events per race lap; mean across stint laps (excludes outlap/inlap) |
| `boost_energy_per_activation` | Average MJ deployed per boost event | Event duration × estimated power output (350kW max MGU-K) — proxy for energy spend per activation |
| `boost_defensive_vs_offensive_ratio` | Whether team deploys Boost for attack or defense | Compare boost activations to car-ahead proximity data from OpenF1; ratio reveals strategic philosophy |

---

## NEW Features: Overtake Mode

Overtake Mode is the primary passing tool in 2026, replacing DRS. Requires being within 1 second of the car ahead at a designated detection point (nominally the final corner). Allows an extra +0.5MJ and higher sustained electrical power on the following lap. Can only be used in designated activation zones.

| Feature Name | Description | How to Compute |
|---|---|---|
| `overtake_mode_conversion_rate` | When OT mode is available, fraction resulting in a position gain | Position changes on OT-mode-active laps / total OT mode activations; measures attacking quality |
| `overtake_mode_speed_delta` | Peak speed gain when OT mode active vs. non-OT baseline | Speed trace comparison: OT mode active lap vs. equivalent non-OT lap in same stint |
| `overtake_mode_availability_rate` | Fraction of laps car is within 1s at detection point | Proportion of laps meeting proximity requirement from OpenF1 gap data |
| `overtake_mode_defense_success` | Fraction of OT mode attacks against this car that are repelled | Position held / total OT mode activations by opponent against this car |

---

## NEW Features: Recharge & Super-Clipping

Cars harvest energy via: (1) braking regen, (2) part-throttle harvesting, (3) **lift-and-coast** (disables active aero), and (4) **super-clipping** — harvesting at the end of a straight while still at full throttle, which keeps active aero open.

Super-clipping is the 2026-defining harvesting behaviour. The choice between lift-and-coast and super-clipping reveals team philosophy on aero trust and energy management.

| Feature Name | Description | How to Compute |
|---|---|---|
| `superclip_frequency` | Super-clipping events per lap | Detect speed plateau or slight decline while throttle >95% at end of straights; count per lap, mean across race |
| `superclip_harvest_rate` | Energy recovered per super-clip event (proxy) | Speed loss × car mass (768 kg) × event duration — approximate harvest yield per event |
| `superclip_duration` | Duration of super-clip window per straight (seconds) | Timestamp of plateau onset to end of speed plateau or throttle reduction |
| `lift_coast_vs_superclip_ratio` | Preference: Lift-and-Coast (loses aero) vs. super-clip (keeps aero) | Frequency ratio of each mode; reveals PU philosophy and driver trust in active aero consistency |
| `brake_regen_efficiency` | Energy recovered in braking zones relative to peers | Speed delta × deceleration profile per classified corner vs. field average at same circuit |
| `energy_balance_per_lap` | Net energy state trend through a stint | If OpenF1 exposes ERS state: integral of harvest and deployment events. Otherwise proxy via late-stint pace degradation pattern |

```python
def detect_superclip_events(telemetry, throttle_threshold=95, min_duration=0.5):
    """
    Detect super-clipping: speed plateau at straight end while still at full throttle.
    Returns list of (onset_distance, duration, speed_loss) per event.
    """
    tel = telemetry.copy()
    tel['FullThrottle'] = tel['Throttle'] >= throttle_threshold
    tel['SpeedDiff'] = tel['Speed'].diff()  # negative = slowing
    events = []
    in_event = False
    for i, row in tel.iterrows():
        if row['FullThrottle'] and row['SpeedDiff'] < -0.5:
            if not in_event:
                onset = row['Distance']
                speed_at_onset = row['Speed']
                in_event = True
        elif in_event:
            duration = (row['Distance'] - onset) / row['Speed']  # approx seconds
            if duration >= min_duration:
                events.append({
                    'onset_dist': onset,
                    'duration': duration,
                    'speed_loss': speed_at_onset - row['Speed']
                })
            in_event = False
    return events
```

---

## NEW Features: Turbo Architecture & Launch Characteristics

Ferrari chose a smaller turbo for 2026, gaining reduced turbine inertia and faster spool-up. This gives Ferrari-powered cars a launch advantage but may cost top-end power at high-speed circuits. **This is a PU supplier-level characteristic** — it propagates to all customer teams. Features in this group must be encoded at the supplier level and updated dynamically as race data accumulates.

**The tradeoff:**
- Small turbo: faster spool from standstill, better low-speed corner exits, lower top-end power ceiling on long straights
- Large turbo: slower spool at launch, better top-end power at Monza/Spa/Silverstone/Albert Park
- Ferrari blocked early proposals to change the start procedure because this advantage was an intentional engineering choice

| Feature Name | Description | How to Compute |
|---|---|---|
| `turbo_spool_proxy` | How quickly PU reaches full torque from standstill | Speed at 50m / 100m / 150m from start line normalized by grid position; average across season starts |
| `launch_positions_gained` | Grid positions gained or lost by end of lap 1 | `grid_position − position_after_lap_1`; positive = gained. Average across season, exclude DNF-lap-1 events |
| `launch_vs_quali_delta` | Whether car outperforms or underperforms its quali position at the start | `(expected_position_from_quali) − (actual_position_after_lap_1)`; normalised against grid size |
| `pitstop_exit_acceleration` | Turbo response from near-standstill during race | Speed at 100m and 200m past pit exit line; rolling start from low speed — independent signal from race start |
| `low_speed_corner_exit_pace` | Acceleration out of slow corners — secondary turbo response signal | Speed delta from apex to 200m after apex for corners classified as <80 km/h; team-averaged |
| `high_speed_straight_deficit` | V-max shortfall on long straights vs. field (small turbo cost) | V-max on straights >1 km delta to field average per circuit; expect Ferrari-powered cars lower at Monza/Spa |
| `track_power_sensitivity_interaction` | Interaction term: `turbo_spool_proxy` × `track_power_sensitivity_score` | Product of car's turbo_spool_proxy and the circuit's track_power_sensitivity_score; key non-linear interaction |
| `superclip_onset_distance` | How early in a straight the car reaches super-clip threshold | Distance from corner exit to onset of speed plateau; earlier onset may signal lower PU power ceiling |

### PU Supplier Architecture Priors (2026 Season Start)

Encode `turbo_size` as an ordinal integer per PU supplier (1=small, 2=medium, 3=large). Treat all medium/unknown suppliers as 2 until empirical `turbo_spool_proxy` data overrides the prior. Update `reliability_prior` after each race using the `reliability_trend` slope.

| PU Supplier | Teams | Turbo Size Prior | Reliability Prior (0=clean, 1=crisis) | Notes |
|---|---|---|---|---|
| Ferrari | Ferrari, Haas, Cadillac | Small (1) | 0.15 | Confirmed small turbo; strong race 1 reliability |
| Mercedes | Mercedes, McLaren, Williams, Alpine | Large (3) | 0.15 | Implied large turbo from Russell comments; clean race 1 |
| Red Bull/Ford | Red Bull, Racing Bulls | Medium (2) | 0.45 | Architecture unknown; hydraulic failure race 1 |
| Honda | Aston Martin | Medium (2) | 0.85 | Vibration/battery crisis since pre-season testing |
| Audi | Audi (Sauber) | Medium (2) | 0.70 | New constructor; DNS race 1; no architecture data |

```python
def compute_launch_features(race_laps, race_telemetry, driver, grid_pos):
    lap1 = race_laps[race_laps['Driver'] == driver].iloc[0]
    pos_after_lap1 = lap1['Position']
    tel_start = race_telemetry.get_driver(driver).slice_by_lap(lap1)
    speed_50m  = tel_start[tel_start['Distance'] <= 50]['Speed'].max()
    speed_100m = tel_start[tel_start['Distance'] <= 100]['Speed'].max()
    speed_150m = tel_start[tel_start['Distance'] <= 150]['Speed'].max()
    return {
        'turbo_spool_proxy': (speed_50m + speed_100m + speed_150m) / 3,
        'launch_positions_gained': grid_pos - pos_after_lap1,
    }
```

---

## NEW Features: Reliability & Systemic Failure Risk

Reliability is a first-class predictive feature in 2026. Features in this group separate car/platform failures (predictive of future retirements) from driver incidents (not predictive of mechanical reliability).

| Feature Name | Description | How to Compute |
|---|---|---|
| `systemic_dnf_rate` | Fraction of races ending in a car/platform failure | `systemic_dnf_count / races_started` — include only: `power_unit`, `electrical`, `hydraulics`, `mechanical`, `chassis` categories |
| `driver_incident_rate` | Fraction of races ending in driver-attributed incidents | `collision_fault_count / races_started` — does NOT feed systemic reliability features |
| `dnf_rate_rolling_5` | Systemic DNF rate over the last 5 races | Rolling window of `systemic_dnf_rate`; improving slope = team is fixing problems |
| `completion_rate` | Mean fraction of race distance completed across all events | `mean(laps_completed / total_laps)`; penalizes precautionary retirements |
| `both_cars_dnf_rate` | Fraction of races where BOTH cars retire with systemic failures | Race IDs with 2+ systemic DNFs / total races; strong platform-level signal |
| `dnq_rate` | Fraction of weekends where car failed to qualify or start | `dnq_or_dns_count / weekends_entered` |
| `precautionary_retirement_rate` | Fraction of races team retired car protectively | Team-instructed retirements not driven by discrete failure; signals management of fragile platform |
| `pu_failure_rate` | Power unit specific failure rate | `power_unit` category DNFs / `races_started` |
| `electrical_failure_rate` | Electrical/battery/ERS failure rate | `electrical` category DNFs / `races_started` — closely linked to PU supplier vibration issues in 2026 |
| `hydraulic_failure_rate` | Hydraulic system failure rate | `hydraulics` category DNFs / `races_started` |
| `pu_supplier_failure_rate` | PU failures aggregated across ALL customer teams of a supplier | Sum of `power_unit` + `electrical` DNFs across all teams running that PU / total starts for that PU |
| `reliability_trend` | Whether reliability is improving or worsening | Slope of `completion_rate` over last 5 races; positive slope = improving |
| `reliability_risk_score` | Interaction: `systemic_dnf_rate` × `track_stress_index` | Amplifies fragility at high-stress circuits; `track_stress_index`: Baku=0.9, Singapore=0.85, Bahrain=0.4 |

### DNF Classification Taxonomy

Every DNF and DNS in the training dataset must be tagged with one of the following categories. Systemic categories feed reliability features; incident categories are tracked separately.

| Category | Type | Typical FIA retirement strings / examples |
|---|---|---|
| `power_unit` | SYSTEMIC | Engine, Power Unit, Turbo, Hybrid, Fuel system, Exhaust, Overheating, Power loss |
| `electrical` | SYSTEMIC | Battery, Electronics, ECU, Wiring, ERS, Sensor failure, Vibration damage to battery |
| `hydraulics` | SYSTEMIC | Hydraulics, Brake hydraulics, Gearbox hydraulics, Hydraulic leak |
| `mechanical` | SYSTEMIC | Gearbox, Driveshaft, Suspension, Wheel bearing, Brake failure (non-hydraulic) |
| `chassis` | SYSTEMIC | Floor damage (no collision), Structural failure, Bodywork failure (non-collision) |
| `collision_fault` | INCIDENT | Driver error, Spin, First-lap incident attributed to driver |
| `collision_racing` | INCIDENT | Racing incident, Contact with another car, Debris from racing contact |
| `collision_other` | INCIDENT | Track debris, Safety car incident, External cause |
| `precautionary` | AMBIGUOUS | Team retirement, Conserving parts, Data gathering, Preventive stop |
| `unknown` | AMBIGUOUS | Unspecified, Retired to garage, No official reason given |

```python
SYSTEMIC = {'power_unit', 'electrical', 'hydraulics', 'mechanical', 'chassis'}

def compute_reliability_features(results_df, team, last_n=None):
    """
    results_df: one row per car per race.
    Columns: race_id, team, driver, laps_completed, total_laps, dnf_category
    """
    df = results_df[results_df['team'] == team].copy()
    if last_n:
        race_ids = df['race_id'].unique()[-last_n:]
        df = df[df['race_id'].isin(race_ids)]
    n_races = df['race_id'].nunique()
    n_starts = len(df)
    systemic = df[df['dnf_category'].isin(SYSTEMIC)]
    both_dnf = (
        df[df['dnf_category'].isin(SYSTEMIC)]
        .groupby('race_id').size().ge(2).sum()
    )
    completion = (df['laps_completed'] / df['total_laps']).mean()
    per_race = (
        df.groupby('race_id')
        .apply(lambda g: (g['laps_completed'] / g['total_laps']).mean())
        .reset_index(name='completion')
    )
    trend = (
        np.polyfit(range(len(per_race)), per_race['completion'], 1)[0]
        if len(per_race) >= 3 else 0.0
    )
    return {
        'systemic_dnf_rate':       len(systemic) / n_starts,
        'both_cars_dnf_rate':      both_dnf / n_races,
        'completion_rate':         completion,
        'reliability_trend':       trend,   # positive = improving
        'pu_failure_rate':         len(df[df['dnf_category'] == 'power_unit']) / n_starts,
        'electrical_failure_rate': len(df[df['dnf_category'] == 'electrical']) / n_starts,
    }
```

---

## Updated 2026 Track Characterization Features

The following features are **added to** the existing track characterization set. Circuits are now formally classified as energy-rich or energy-poor.

| Feature Name | Description | How to Compute |
|---|---|---|
| `energy_richness_score` | How much ERS harvest a circuit enables per lap | Braking zones per lap × avg deceleration magnitude; high = many heavy braking zones = abundant harvest |
| `superclip_opportunity_count` | Number of straights where super-clipping is viable | Count straights where full-throttle duration consistently enables end-of-straight harvest (threshold: >4 seconds flat) |
| `overtake_mode_zone_count` | Number of designated OT mode activation zones per circuit | FIA circuit documentation per race round; manual label or OpenF1 API if exposed |
| `active_aero_zone_count` | Number of designated Straight Mode activation zones | FIA circuit documentation; correlates strongly with `pct_full_throttle` |
| `track_power_sensitivity_score` | How much raw PU top-end power matters at this circuit | Composite: `pct_full_throttle × avg_straight_length × avg_straight_speed`. Monza ≈ 0.95, Monaco ≈ 0.15 |
| `lift_coast_penalty` | Estimated lap time cost of required lift-and-coast events per lap | Duration × speed cost per mandatory lift event for energy balance; circuit-level aggregate |

### Example Circuit Power Sensitivity Values

| Circuit | Energy Richness | Track Power Score | Archetype |
|---|---|---|---|
| Monza | Low — few braking zones | 0.95 | Power-sensitive; maximises small-turbo top-end deficit |
| Spa | Medium — varied layout | 0.88 | Power important; some harvest from Eau Rouge/Pouhon sectors |
| Monaco | High — constant braking | 0.15 | Energy-rich; launch/mechanical advantage dominates |
| Singapore | Very high — street circuit | 0.20 | Maximum energy-rich; super-clip opportunities limited |
| Bahrain | Medium — standard hairpins | 0.60 | Balanced; good test case for early-season model validation |
