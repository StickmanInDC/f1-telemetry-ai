"""
Ergast / Jolpica API client for historical race results and DNF data.
Used primarily for reliability feature construction (DNF classification).
Falls back to Jolpica mirror if Ergast is unavailable.
"""

import requests
import pandas as pd
import json
from pathlib import Path

ERGAST_BASE = "https://ergast.com/api/f1"
JOLPICA_BASE = "https://api.jolpi.ca/ergast/f1"
RAW_CACHE_DIR = Path(__file__).parent.parent.parent / 'data' / 'raw' / 'ergast'


def _get(endpoint: str, use_cache: bool = True) -> dict:
    """Make a cached GET request, trying Ergast first then Jolpica fallback."""
    RAW_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cache_key = endpoint.strip('/').replace('/', '_')
    cache_file = RAW_CACHE_DIR / f"{cache_key}.json"

    if use_cache and cache_file.exists():
        with open(cache_file, 'r') as f:
            return json.load(f)

    # Try Ergast first, then Jolpica
    for base_url in [ERGAST_BASE, JOLPICA_BASE]:
        try:
            url = f"{base_url}/{endpoint}.json?limit=1000"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            with open(cache_file, 'w') as f:
                json.dump(data, f)
            return data
        except (requests.RequestException, requests.HTTPError):
            continue

    raise ConnectionError(f"Failed to fetch {endpoint} from both Ergast and Jolpica")


def get_race_results(year: int, round_num: int = None) -> pd.DataFrame:
    """
    Get race results for a season or specific round.
    Returns one row per driver per race with finishing status.
    """
    if round_num:
        endpoint = f"{year}/{round_num}/results"
    else:
        endpoint = f"{year}/results"

    data = _get(endpoint)
    races = data['MRData']['RaceTable']['Races']

    rows = []
    for race in races:
        race_round = int(race['round'])
        circuit_id = race['Circuit']['circuitId']
        for result in race['Results']:
            rows.append({
                'season': year,
                'round': race_round,
                'circuit_id': circuit_id,
                'circuit_name': race['Circuit']['circuitName'],
                'driver_id': result['Driver']['driverId'],
                'driver_code': result['Driver'].get('code', ''),
                'constructor_id': result['Constructor']['constructorId'],
                'constructor_name': result['Constructor']['name'],
                'grid': int(result['grid']),
                'position': result.get('position', None),
                'position_text': result['positionText'],
                'points': float(result['points']),
                'laps_completed': int(result['laps']),
                'status': result['status'],
                'time_millis': result.get('Time', {}).get('millis'),
            })

    return pd.DataFrame(rows)


def get_season_results(year: int) -> pd.DataFrame:
    """Get all race results for an entire season."""
    return get_race_results(year)


def get_qualifying_results(year: int, round_num: int = None) -> pd.DataFrame:
    """Get qualifying results for a season or specific round."""
    if round_num:
        endpoint = f"{year}/{round_num}/qualifying"
    else:
        endpoint = f"{year}/qualifying"

    data = _get(endpoint)
    races = data['MRData']['RaceTable']['Races']

    rows = []
    for race in races:
        race_round = int(race['round'])
        for result in race.get('QualifyingResults', []):
            rows.append({
                'season': year,
                'round': race_round,
                'driver_id': result['Driver']['driverId'],
                'driver_code': result['Driver'].get('code', ''),
                'constructor_id': result['Constructor']['constructorId'],
                'constructor_name': result['Constructor']['name'],
                'position': int(result['position']),
                'q1': result.get('Q1'),
                'q2': result.get('Q2'),
                'q3': result.get('Q3'),
            })

    return pd.DataFrame(rows)


def classify_dnf(status: str) -> str:
    """
    Classify a retirement status string into the DNF taxonomy.
    See doc 03 for full taxonomy definition.

    Returns one of: power_unit, electrical, hydraulics, mechanical, chassis,
                    collision_fault, collision_racing, collision_other,
                    precautionary, unknown, finished
    """
    s = status.lower().strip()

    # Finished normally
    if s.startswith('finished') or s.startswith('+') or s.isdigit():
        return 'finished'

    # Power unit failures
    pu_keywords = ['engine', 'power unit', 'turbo', 'hybrid', 'fuel',
                   'exhaust', 'overheating', 'power loss', 'oil pressure',
                   'oil leak', 'water pressure', 'water leak', 'mgu-h',
                   'mgu-k', 'ers', 'internal combustion']
    if any(kw in s for kw in pu_keywords):
        return 'power_unit'

    # Electrical failures
    elec_keywords = ['battery', 'electronic', 'ecu', 'wiring', 'sensor',
                     'electrical', 'vibration', 'energy store']
    if any(kw in s for kw in elec_keywords):
        return 'electrical'

    # Hydraulic failures
    hyd_keywords = ['hydraulic', 'brake hydraulic']
    if any(kw in s for kw in hyd_keywords):
        return 'hydraulics'

    # Mechanical failures
    mech_keywords = ['gearbox', 'driveshaft', 'suspension', 'wheel bearing',
                     'brake failure', 'brakes', 'clutch', 'differential',
                     'wheel', 'puncture', 'tyre', 'tire']
    if any(kw in s for kw in mech_keywords):
        return 'mechanical'

    # Chassis / structural
    chassis_keywords = ['floor', 'structural', 'bodywork', 'wing',
                        'front wing', 'rear wing', 'chassis']
    if any(kw in s for kw in chassis_keywords):
        return 'chassis'

    # Driver incidents
    collision_fault_keywords = ['spun off', 'spin', 'driver error', 'off track']
    if any(kw in s for kw in collision_fault_keywords):
        return 'collision_fault'

    collision_racing_keywords = ['collision', 'contact', 'accident', 'crash',
                                 'racing incident']
    if any(kw in s for kw in collision_racing_keywords):
        return 'collision_racing'

    collision_other_keywords = ['debris', 'damage']
    if any(kw in s for kw in collision_other_keywords):
        return 'collision_other'

    # Precautionary
    precautionary_keywords = ['retired', 'withdrew', 'not classified',
                               'disqualified']
    if any(kw in s for kw in precautionary_keywords):
        return 'precautionary'

    return 'unknown'


def build_results_with_dnf_classification(years: list[int]) -> pd.DataFrame:
    """
    Build a consolidated results DataFrame with DNF classifications
    across multiple seasons. One row per driver per race.
    """
    all_results = []
    for year in years:
        results = get_season_results(year)
        if not results.empty:
            results['dnf_category'] = results['status'].apply(classify_dnf)
            all_results.append(results)

    if not all_results:
        return pd.DataFrame()

    combined = pd.concat(all_results, ignore_index=True)

    # Compute total laps per race for completion rate
    total_laps = combined.groupby(['season', 'round'])['laps_completed'].max()
    combined = combined.merge(
        total_laps.rename('total_laps'),
        on=['season', 'round'],
        how='left'
    )

    return combined
