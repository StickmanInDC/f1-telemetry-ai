"""
FastF1 session loading and lap cleaning utilities.
Primary data source for telemetry, lap times, tire data, and weather.
"""

import fastf1
import pandas as pd
import numpy as np
from pathlib import Path

# Default cache directory - set before any session loads
DEFAULT_CACHE_DIR = Path(__file__).parent.parent.parent / 'data' / 'cache'


def enable_cache(cache_dir: str | Path | None = None):
    """Enable FastF1 cache. Critical for performance — raw downloads are slow."""
    cache_path = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    cache_path.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_path))


def load_session(year: int, round_num: int, session_type: str = 'R'):
    """
    Load a FastF1 session with telemetry, weather, and lap data.

    Args:
        year: Season year (e.g., 2024)
        round_num: Race round number (1-based)
        session_type: 'R' (race), 'Q' (qualifying), 'FP1'/'FP2'/'FP3'

    Returns:
        Loaded FastF1 session object
    """
    session = fastf1.get_session(year, round_num, session_type)
    session.load(telemetry=True, weather=True, laps=True)
    return session


def get_clean_laps(session) -> pd.DataFrame:
    """
    Filter session laps to remove non-representative data.

    Removes: inaccurate laps, outlaps, inlaps, safety car / VSC laps.
    Returns clean laps suitable for feature computation.
    """
    laps = session.laps.copy()

    # FastF1 accurate lap flag
    laps = laps[laps['IsAccurate'] == True]

    # Remove outlaps (pit exit) and inlaps (pit entry)
    laps = laps[laps['PitOutTime'].isna()]
    laps = laps[laps['PitInTime'].isna()]

    # Remove safety car / VSC laps via track status
    # Status codes: 4=SC, 5=red flag, 6=VSC, 7=VSC ending
    if 'TrackStatus' in laps.columns:
        laps = laps[~laps['TrackStatus'].astype(str).str.contains('4|5|6|7', na=False)]

    # Ensure LapTime is valid
    laps = laps.dropna(subset=['LapTime'])

    return laps


def get_stint_laps(clean_laps: pd.DataFrame, driver: str, compound: str) -> pd.DataFrame:
    """
    Extract laps for a specific driver and tire compound stint.
    Adds StintLap counter (0-indexed) for degradation analysis.
    """
    stint = clean_laps[
        (clean_laps['Driver'] == driver) &
        (clean_laps['Compound'] == compound)
    ].copy()

    if stint.empty:
        return stint

    stint = stint.sort_values('LapNumber')
    stint['StintLap'] = range(len(stint))
    stint['LapTime_seconds'] = stint['LapTime'].dt.total_seconds()

    return stint


def load_season_races(year: int, session_type: str = 'R') -> list:
    """
    Get schedule of all races for a season.
    Returns list of (round_num, event_name) tuples.
    """
    schedule = fastf1.get_event_schedule(year)
    races = []
    for _, event in schedule.iterrows():
        if event['EventFormat'] != 'testing':
            races.append((event['RoundNumber'], event['EventName']))
    return [(r, n) for r, n in races if r > 0]


def load_all_sessions(years: list[int], session_type: str = 'R',
                       progress_callback=None) -> dict:
    """
    Load all race sessions for multiple years.

    Returns:
        Dict of {(year, round_num): session} for successfully loaded sessions.
    """
    sessions = {}
    for year in years:
        races = load_season_races(year, session_type)
        for round_num, event_name in races:
            key = (year, round_num)
            try:
                session = load_session(year, round_num, session_type)
                sessions[key] = session
                if progress_callback:
                    progress_callback(f"Loaded {year} R{round_num}: {event_name}")
            except Exception as e:
                if progress_callback:
                    progress_callback(f"SKIP {year} R{round_num} ({event_name}): {e}")
    return sessions
