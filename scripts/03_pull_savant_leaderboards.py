"""Pull Savant ABS leaderboard CSVs (per-player aggregates with Savant's run pricing).

Usage: python scripts/03_pull_savant_leaderboards.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from abs_judgment.savant import fetch_abs_leaderboard  # noqa: E402

PULLS = [
    (2026, "batter", "mlb"), (2026, "catcher", "mlb"), (2026, "pitcher", "mlb"),
    (2026, "batting-team", "mlb"), (2026, "catching-team", "mlb"),
    (2025, "batter", "aaa"), (2025, "catcher", "aaa"),
]

if __name__ == "__main__":
    for season, ctype, level in PULLS:
        p = fetch_abs_leaderboard(season, ctype, level)
        n = sum(1 for _ in open(p, encoding="utf-8")) - 1
        print(f"{p.name}: {n} rows", flush=True)
