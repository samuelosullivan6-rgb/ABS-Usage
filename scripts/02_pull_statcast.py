"""Pull daily Statcast search CSVs for a season into data/raw/statcast/.

Usage:
  python scripts/02_pull_statcast.py --season 2026
Sequential on purpose: Savant throttles bursts.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from abs_judgment.config import POSTSEASON, RAW_STATCAST, RAW_STATCAST_POST, SEASONS  # noqa: E402
from abs_judgment.savant import fetch_statcast_day, statcast_day_path  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--sleep", type=float, default=1.5)
    ap.add_argument("--postseason", action="store_true", help="Savant game type PO, cached in data/raw/statcast_post/")
    ap.add_argument("--end", default=None, help="last date to include (postseason only; default: the config end)")
    a = ap.parse_args()
    if a.postseason:
        cfg = POSTSEASON[a.season]
        cache, game_type, end = RAW_STATCAST_POST, cfg["statcast_game_type"], (a.end or cfg["end"])
    else:
        cfg = SEASONS[(a.season, 1)]
        cache, game_type, end = RAW_STATCAST, cfg["game_type"], cfg["end"]
    days = [d.strftime("%Y-%m-%d") for d in pd.date_range(cfg["start"], end)]
    todo = [d for d in days if not statcast_day_path(d, cache).exists()]
    print(f"{len(days)} days; {len(todo)} to fetch", flush=True)
    for i, d in enumerate(todo, 1):
        p = fetch_statcast_day(d, game_type=game_type, cache_dir=cache)
        n = sum(1 for _ in open(p, encoding="utf-8")) - 1
        print(f"[{i}/{len(todo)}] {d}: {n} rows", flush=True)
        time.sleep(a.sleep)
    print("done", flush=True)


if __name__ == "__main__":
    main()
