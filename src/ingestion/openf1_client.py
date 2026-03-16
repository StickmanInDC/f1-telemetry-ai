"""
OpenF1 REST API client for supplementary race data.
Provides real-time and historical data: positions, car data, stints, pit stops.
Available from 2023 onward. No authentication required.
"""

import requests
import pandas as pd
import json
from pathlib import Path

BASE_URL = "https://api.openf1.org/v1"
RAW_CACHE_DIR = Path(__file__).parent.parent.parent / 'data' / 'raw' / 'openf1'


def _get(endpoint: str, params: dict = None, use_cache: bool = True) -> list[dict]:
    """Make a cached GET request to OpenF1 API."""
    RAW_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Build cache key from endpoint and params
    cache_key = endpoint.strip('/').replace('/', '_')
    if params:
        param_str = '_'.join(f"{k}={v}" for k, v in sorted(params.items()))
        cache_key += f"__{param_str}"
    cache_file = RAW_CACHE_DIR / f"{cache_key}.json"

    if use_cache and cache_file.exists():
        with open(cache_file, 'r') as f:
            return json.load(f)

    url = f"{BASE_URL}/{endpoint}"
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    # Cache the result
    with open(cache_file, 'w') as f:
        json.dump(data, f)

    return data


def get_sessions(year: int) -> pd.DataFrame:
    """Get all sessions for a given year."""
    data = _get("sessions", {"year": year})
    return pd.DataFrame(data)


def get_session_key(year: int, round_num: int, session_type: str = 'Race') -> int | None:
    """Get the OpenF1 session key for a specific session."""
    sessions = get_sessions(year)
    if sessions.empty:
        return None
    match = sessions[
        (sessions['year'] == year) &
        (sessions['session_type'] == session_type)
    ]
    if len(match) >= round_num:
        return match.iloc[round_num - 1].get('session_key')
    return None


def get_lap_data(session_key: int, driver_number: int = None) -> pd.DataFrame:
    """Get lap data for a session, optionally filtered by driver."""
    params = {"session_key": session_key}
    if driver_number:
        params["driver_number"] = driver_number
    data = _get("laps", params)
    return pd.DataFrame(data)


def get_car_data(session_key: int, driver_number: int,
                  speed_gt: int = None) -> pd.DataFrame:
    """
    Get high-frequency car telemetry data.
    Useful for per-sample speed/throttle/brake analysis.
    """
    params = {
        "session_key": session_key,
        "driver_number": driver_number,
    }
    if speed_gt:
        params["speed>"] = speed_gt
    data = _get("car_data", params)
    return pd.DataFrame(data)


def get_position_data(session_key: int) -> pd.DataFrame:
    """Get position data for all drivers in a session."""
    data = _get("position", {"session_key": session_key})
    return pd.DataFrame(data)


def get_stints(session_key: int, driver_number: int = None) -> pd.DataFrame:
    """Get stint information (compound, lap range) for a session."""
    params = {"session_key": session_key}
    if driver_number:
        params["driver_number"] = driver_number
    data = _get("stints", params)
    return pd.DataFrame(data)


def get_pit_stops(session_key: int) -> pd.DataFrame:
    """Get pit stop data for a session."""
    data = _get("pit", {"session_key": session_key})
    return pd.DataFrame(data)


def get_intervals(session_key: int, driver_number: int = None) -> pd.DataFrame:
    """Get gap/interval data between drivers."""
    params = {"session_key": session_key}
    if driver_number:
        params["driver_number"] = driver_number
    data = _get("intervals", params)
    return pd.DataFrame(data)


def get_drivers(session_key: int) -> pd.DataFrame:
    """Get driver info for a session."""
    data = _get("drivers", {"session_key": session_key})
    return pd.DataFrame(data)
