"""Baseball Savant access: daily Statcast search CSVs and the ABS challenge
leaderboard CSV. Both verified 2026-10-21."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import requests

from .config import RAW_SAVANT, RAW_STATCAST, USER_AGENT

log = logging.getLogger(__name__)

_session = requests.Session()
_session.headers["User-Agent"] = USER_AGENT

STATCAST_CSV = "https://baseballsavant.mlb.com/statcast_search/csv"
ABS_LEADERBOARD = "https://baseballsavant.mlb.com/leaderboard/abs-challenges"


def _statcast_params(date: str, game_type: str = "R") -> dict:
    season = date[:4]
    return {
        "all": "true", "hfPT": "", "hfAB": "", "hfGT": f"{game_type}|", "hfPR": "", "hfZ": "",
        "hfStadium": "", "hfBBL": "", "hfNewZones": "", "hfPull": "", "hfC": "",
        "hfSea": f"{season}|", "hfSit": "", "player_type": "batter", "hfOuts": "",
        "hfOpponent": "", "pitcher_throws": "", "batter_stands": "", "hfSA": "",
        "game_date_gt": date, "game_date_lt": date, "hfMo": "", "hfTeam": "",
        "home_road": "", "hfRO": "", "position": "", "hfInfield": "", "hfOutfield": "",
        "hfInn": "", "hfBBT": "", "hfFlag": "", "metric_1": "", "group_by": "name",
        "min_pitches": "0", "min_results": "0", "min_pas": "0", "sort_col": "pitches",
        "player_event_sort": "api_p_release_speed", "sort_order": "desc", "type": "details",
    }


def statcast_day_path(date: str, cache_dir: Path = RAW_STATCAST) -> Path:
    return cache_dir / f"{date}.csv"


def fetch_statcast_day(date: str, game_type: str = "R", cache_dir: Path = RAW_STATCAST, retries: int = 4) -> Path:
    """One day of pitch-level Statcast data (all games), cached as CSV."""
    p = statcast_day_path(date, cache_dir)
    if p.exists():
        return p
    for attempt in range(retries):
        try:
            r = _session.get(STATCAST_CSV, params=_statcast_params(date, game_type), timeout=180)
            r.raise_for_status()
            text = r.text
            if text.startswith("<"):
                raise RuntimeError("HTML response (rate limited?)")
            tmp = p.with_suffix(".tmp")
            tmp.write_text(text, encoding="utf-8")
            tmp.replace(p)
            return p
        except Exception as e:  # noqa: BLE001
            wait = 5 * 2 ** attempt
            log.warning("statcast %s attempt %d failed: %s; retry in %ss", date, attempt + 1, e, wait)
            time.sleep(wait)
    raise RuntimeError(f"could not fetch statcast for {date}")


def load_statcast_days(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        if p.stat().st_size < 200:
            continue  # off day
        df = pd.read_csv(p, low_memory=False, encoding="utf-8-sig")
        if len(df):
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_abs_leaderboard(season: int, challenge_type: str = "batter", level: str = "mlb",
                          game_type: str = "R", cache_dir: Path = RAW_SAVANT) -> Path:
    """Savant ABS leaderboard as CSV. challenge_type in
    {batter, catcher, pitcher, batting-team, catching-team}; level in {mlb, aaa}."""
    p = cache_dir / f"abs_leaderboard_{level}_{season}_{challenge_type}_{game_type}.csv"
    if p.exists():
        return p
    r = _session.get(
        ABS_LEADERBOARD,
        params={"season": season, "challengeType": challenge_type, "level": level,
                "gameType": game_type, "csv": "true"},
        timeout=120,
    )
    r.raise_for_status()
    if r.text.lstrip().startswith("<"):
        raise RuntimeError("leaderboard returned HTML, not CSV")
    p.write_text(r.text, encoding="utf-8")
    return p
