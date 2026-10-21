"""Paths and season constants."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
RAW_STATSAPI = RAW / "statsapi"
RAW_STATCAST = RAW / "statcast"
RAW_SAVANT = RAW / "savant"

for _d in (RAW_STATSAPI, RAW_STATCAST, RAW_SAVANT, PROCESSED):
    _d.mkdir(parents=True, exist_ok=True)

# Verified from https://statsapi.mlb.com/api/v1/seasons?sportId=1&season=2026
SEASONS = {
    (2026, 1): {"start": "2026-03-25", "end": "2026-09-27", "game_type": "R", "label": "mlb_2026"},
    # 2025 MLB had no ABS; Statcast only, for year-over-year chase change.
    (2025, 1): {"start": "2025-03-27", "end": "2025-09-28", "game_type": "R", "label": "mlb_2025"},
    # Triple-A 2025 (sportId 11). Challenge events exist but omit the initiating player.
    (2025, 11): {"start": "2025-03-28", "end": "2025-09-21", "game_type": "R", "label": "aaa_2025"},
}

MLB_SPORT_ID = 1
AAA_SPORT_ID = 11

# ABS rules as observed in 2026 MLB feeds. Starting tokens per team per game.
TOKENS_PER_GAME = 2

USER_AGENT = "abs-judgment-research/0.1 (personal research; contact via GitHub)"
