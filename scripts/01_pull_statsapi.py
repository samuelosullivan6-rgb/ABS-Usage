"""Pull every final game feed for a season into data/raw/statsapi/.

Usage:
  python scripts/01_pull_statsapi.py --season 2026 --sport 1 [--workers 6]
  python scripts/01_pull_statsapi.py --season 2025 --sport 11
"""
from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tqdm import tqdm  # noqa: E402

from abs_judgment.config import POSTSEASON, PROCESSED, SEASONS  # noqa: E402
from abs_judgment.statsapi import feed_path, fetch_feed, list_games  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--sport", type=int, default=1)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--postseason", action="store_true", help="Wild Card through World Series; outputs carry _post")
    ap.add_argument("--end", default=None, help="last date to include (postseason only; default: the config end)")
    a = ap.parse_args()

    cfg = POSTSEASON[a.season] if a.postseason else SEASONS[(a.season, a.sport)]
    end = a.end if (a.postseason and a.end) else cfg["end"]
    games = list_games(cfg["start"], end, sport_id=a.sport, game_type=cfg["game_type"])
    import pandas as pd

    pd.DataFrame(games).to_csv(PROCESSED / f"games_{cfg['label']}.csv", index=False)
    todo = [g["game_pk"] for g in games if not feed_path(g["game_pk"]).exists()]
    print(f"{len(games)} final games; {len(todo)} to fetch", flush=True)

    failed = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(fetch_feed, pk): pk for pk in todo}
        for f in tqdm(as_completed(futs), total=len(futs), mininterval=5, file=sys.stdout):
            pk = futs[f]
            try:
                f.result()
            except Exception as e:  # noqa: BLE001
                failed.append(pk)
                print(f"FAILED {pk}: {e}", flush=True)
    print(f"done; {len(failed)} failures: {failed[:20]}", flush=True)


if __name__ == "__main__":
    main()
