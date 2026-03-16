"""
Group 1: Power Unit & Energy Deployment features.
Captures PU performance, ERS deployment quality, and DRS exploitation.

NOTE: drs_delta_speed, drs_activation_rate, and mgu_h_proxy are RETIRED in 2026.
"""

import pandas as pd
import numpy as np


def compute_pu_vmax(telemetry: pd.DataFrame, clean_laps: pd.DataFrame,
                     driver: str) -> float:
    """
    Average peak speed per straight per race.
    Max speed within each lap; mean across all clean stint laps.
    """
    driver_laps = clean_laps[clean_laps['Driver'] == driver]
    if driver_laps.empty:
        return np.nan

    vmax_values = []
    for _, lap in driver_laps.iterrows():
        try:
            lap_tel = telemetry.slice_by_lap(lap)
            if lap_tel is not None and not lap_tel.empty:
                vmax_values.append(lap_tel['Speed'].max())
        except Exception:
            continue

    return np.mean(vmax_values) if vmax_values else np.nan


def compute_ers_deployment_consistency(telemetry: pd.DataFrame,
                                        clean_laps: pd.DataFrame,
                                        driver: str) -> float:
    """
    Std dev of speed gain after corner exit.
    Low value = consistent ERS deployment. Measured at 200m after apex.
    """
    driver_laps = clean_laps[clean_laps['Driver'] == driver]
    if driver_laps.empty:
        return np.nan

    speed_gains = []
    for _, lap in driver_laps.iterrows():
        try:
            lap_tel = telemetry.slice_by_lap(lap)
            if lap_tel is None or lap_tel.empty:
                continue

            speed = lap_tel['Speed'].values
            distance = lap_tel['Distance'].values

            # Find corner apexes (local minima in speed)
            for i in range(1, len(speed) - 1):
                if speed[i] < speed[i-1] and speed[i] < speed[i+1] and speed[i] < 200:
                    # Find speed 200m after apex
                    apex_dist = distance[i]
                    mask_200m = (distance >= apex_dist + 180) & (distance <= apex_dist + 220)
                    if mask_200m.any():
                        speed_at_200m = speed[mask_200m].mean()
                        speed_gains.append(speed_at_200m - speed[i])
        except Exception:
            continue

    return np.std(speed_gains) if len(speed_gains) > 3 else np.nan


def compute_mgu_h_proxy(telemetry: pd.DataFrame, clean_laps: pd.DataFrame,
                         driver: str) -> float:
    """
    Turbo response quality — MGU-H era only. RETIRE in 2026.
    Time from throttle application to 90% of peak speed after corner exit.
    """
    driver_laps = clean_laps[clean_laps['Driver'] == driver]
    if driver_laps.empty:
        return np.nan

    response_times = []
    for _, lap in driver_laps.iterrows():
        try:
            lap_tel = telemetry.slice_by_lap(lap)
            if lap_tel is None or lap_tel.empty:
                continue

            speed = lap_tel['Speed'].values
            throttle = lap_tel['Throttle'].values
            time_arr = lap_tel['Time'].dt.total_seconds().values if hasattr(lap_tel['Time'], 'dt') else np.arange(len(speed)) * 0.25

            peak_speed = speed.max()
            target_speed = peak_speed * 0.9

            # Find throttle-on events (transition from <20% to >80%)
            for i in range(1, len(throttle) - 1):
                if throttle[i-1] < 20 and throttle[i] > 80:
                    # Measure time to reach 90% peak speed
                    for j in range(i, min(i + 200, len(speed))):
                        if speed[j] >= target_speed:
                            dt = time_arr[j] - time_arr[i]
                            if 0.5 < dt < 10:
                                response_times.append(dt)
                            break
        except Exception:
            continue

    return np.mean(response_times) if response_times else np.nan


def compute_energy_harvest_efficiency(telemetry: pd.DataFrame,
                                       clean_laps: pd.DataFrame,
                                       driver: str) -> float:
    """
    Estimated time cost of lift-and-coast harvesting events.
    Detects lift-and-coast: throttle drop without braking.
    """
    driver_laps = clean_laps[clean_laps['Driver'] == driver]
    if driver_laps.empty:
        return np.nan

    harvest_costs = []
    for _, lap in driver_laps.iterrows():
        try:
            lap_tel = telemetry.slice_by_lap(lap)
            if lap_tel is None or lap_tel.empty:
                continue

            throttle = lap_tel['Throttle'].values
            brake = lap_tel['Brake'].values if 'Brake' in lap_tel.columns else np.zeros(len(throttle))
            speed = lap_tel['Speed'].values

            # Detect lift-and-coast: throttle drops below 50%, no braking, speed > 150
            in_event = False
            event_speed_loss = 0
            event_count = 0
            for i in range(len(throttle)):
                if throttle[i] < 50 and brake[i] < 10 and speed[i] > 150:
                    if not in_event:
                        in_event = True
                        event_start_speed = speed[i]
                elif in_event:
                    event_speed_loss += event_start_speed - speed[max(0, i-1)]
                    event_count += 1
                    in_event = False

            if event_count > 0:
                harvest_costs.append(event_speed_loss / event_count)
        except Exception:
            continue

    return np.mean(harvest_costs) if harvest_costs else np.nan


def compute_drs_delta_speed(telemetry: pd.DataFrame, clean_laps: pd.DataFrame,
                             driver: str) -> float:
    """
    Speed gain when DRS opens. RETIRE in 2026.
    Speed at DRS detection point vs. speed 200m downstream.
    """
    driver_laps = clean_laps[clean_laps['Driver'] == driver]
    if driver_laps.empty:
        return np.nan

    drs_deltas = []
    for _, lap in driver_laps.iterrows():
        try:
            lap_tel = telemetry.slice_by_lap(lap)
            if lap_tel is None or lap_tel.empty or 'DRS' not in lap_tel.columns:
                continue

            drs = lap_tel['DRS'].values
            speed = lap_tel['Speed'].values
            distance = lap_tel['Distance'].values

            # Find DRS activation events (transition to open)
            for i in range(1, len(drs)):
                # DRS values vary by year; typically 10-14 = open
                if drs[i] >= 10 and drs[i-1] < 10:
                    activation_speed = speed[i]
                    activation_dist = distance[i]
                    # Speed 200m downstream
                    mask_200m = (distance >= activation_dist + 180) & (distance <= activation_dist + 220)
                    if mask_200m.any():
                        downstream_speed = speed[mask_200m].mean()
                        drs_deltas.append(downstream_speed - activation_speed)
        except Exception:
            continue

    return np.mean(drs_deltas) if drs_deltas else np.nan


def compute_drs_activation_rate(clean_laps: pd.DataFrame, driver: str) -> float:
    """
    Fraction of laps where driver is within 1s to trigger DRS. RETIRE in 2026.
    Uses the DRS column presence as proxy.
    """
    driver_laps = clean_laps[clean_laps['Driver'] == driver]
    if driver_laps.empty:
        return np.nan

    # If we have gap data, compute fraction of laps within DRS range
    # Simplified: use the fact that DRS is available in telemetry
    total = len(driver_laps)
    if total == 0:
        return np.nan

    # Approximate from position data
    return np.nan  # Requires OpenF1 interval data; computed in pipeline


def compute_battery_depletion_signature(telemetry: pd.DataFrame,
                                         clean_laps: pd.DataFrame,
                                         driver: str) -> float:
    """
    Speed drop in final 20% of straights — early super-clip signal.
    Detects plateau or decline while throttle >95% at straight end.
    """
    driver_laps = clean_laps[clean_laps['Driver'] == driver]
    if driver_laps.empty:
        return np.nan

    depletion_events = []
    for _, lap in driver_laps.iterrows():
        try:
            lap_tel = telemetry.slice_by_lap(lap)
            if lap_tel is None or lap_tel.empty:
                continue

            throttle = lap_tel['Throttle'].values
            speed = lap_tel['Speed'].values

            # Find straights (sustained high throttle)
            in_straight = False
            straight_start = 0
            for i in range(len(throttle)):
                if throttle[i] > 95:
                    if not in_straight:
                        in_straight = True
                        straight_start = i
                elif in_straight:
                    straight_len = i - straight_start
                    if straight_len > 20:  # Minimum straight length in samples
                        # Check final 20% for speed plateau/decline
                        final_start = straight_start + int(straight_len * 0.8)
                        final_speeds = speed[final_start:i]
                        if len(final_speeds) > 3:
                            speed_change = final_speeds[-1] - final_speeds[0]
                            if speed_change < -1:  # Speed dropping while on full throttle
                                depletion_events.append(abs(speed_change))
                    in_straight = False
        except Exception:
            continue

    if depletion_events:
        return np.mean(depletion_events)
    return 0.0


def compute_all_pu_features(session, clean_laps: pd.DataFrame,
                              driver: str, is_2026: bool = False) -> dict:
    """
    Compute all Group 1 features for a driver.
    Set is_2026=True to skip retired features.
    """
    try:
        telemetry = session.car_data
    except Exception:
        telemetry = None

    features = {}

    if telemetry is not None:
        features['pu_vmax_avg'] = compute_pu_vmax(telemetry, clean_laps, driver)
        features['ers_deployment_consistency'] = compute_ers_deployment_consistency(
            telemetry, clean_laps, driver)
        features['energy_harvest_efficiency'] = compute_energy_harvest_efficiency(
            telemetry, clean_laps, driver)
        features['battery_depletion_signature'] = compute_battery_depletion_signature(
            telemetry, clean_laps, driver)

        if not is_2026:
            features['mgu_h_proxy'] = compute_mgu_h_proxy(telemetry, clean_laps, driver)
            features['drs_delta_speed'] = compute_drs_delta_speed(
                telemetry, clean_laps, driver)
            features['drs_activation_rate'] = compute_drs_activation_rate(
                clean_laps, driver)

    return features
