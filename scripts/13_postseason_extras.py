"""Two more model-free quantities for the postseason write-up, from the pitch tables.

  runs_left_post.csv     unchallenged umpire misses on hitters priced at their stake
                         (stakes_mlb_2026.csv), per team per 162 games, regular and postseason
  umpire_zone_post.csv   share of taken pitches the umpire called strikes, by side of the box
                         and inches outside it, regular season (how the called zone differs from the ABS box)
Usage: python scripts/13_postseason_extras.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from abs_judgment.config import PROCESSED  # noqa: E402
from abs_judgment.judgment import miss_direction  # noqa: E402
from abs_judgment.policy import attach_stakes  # noqa: E402

COLS = ["game_pk", "is_take", "orig_called_strike", "orig_called_ball", "bat_tokens_before_ruleB", "abs_edge_in",
        "batter_challenged", "strikes_before", "balls_before", "outs_before", "base_state", "mid_x", "mid_z", "sz_top", "sz_bot"]
st = pd.read_csv(PROCESSED / "stakes_mlb_2026.csv")
rows, zone = [], []
for label, season in (("mlb_2026", "regular"), ("mlb_2026_post", "postseason")):
    pt = pd.read_parquet(PROCESSED / f"pitch_table_{label}.parquet", columns=COLS)
    n_games = pt.game_pk.nunique()
    pt["direction"] = miss_direction(pt.mid_x, pt.mid_z, pt.sz_top, pt.sz_bot)
    m = pt[(pt.orig_called_strike & (pt.abs_edge_in > 0) & (pt.bat_tokens_before_ruleB >= 1)).fillna(False).astype(bool)].copy()
    m["stake"] = np.abs(attach_stakes(m, st))
    m["chal"] = m.batter_challenged.astype(bool)
    per = 162 / (2 * n_games)
    rows.append({"season": season, "games": n_games, "misses": len(m), "unchallenged": int((~m.chal).sum()),
                 "mean_stake_runs": m.stake.mean(), "runs_left_per_team_162": m.loc[~m.chal, "stake"].sum() * per,
                 "runs_left_per_team_162_two_strike": m.loc[~m.chal & (m.strikes_before == 2), "stake"].sum() * per,
                 "runs_won_per_team_162": m.loc[m.chal, "stake"].sum() * per})
    if season == "regular":
        tk = pt[((pt.orig_called_strike | pt.orig_called_ball) & pt.is_take & pt.abs_edge_in.notna()).astype(bool)]
        for k in ("above", "below", "side"):
            for band, bm in (("0-1 in outside", (tk.abs_edge_in > 0) & (tk.abs_edge_in <= 1)), ("1-2 in outside", (tk.abs_edge_in > 1) & (tk.abs_edge_in <= 2))):
                x = tk[bm & (tk.direction == k)]
                zone.append({"direction": k, "band": band, "takes": len(x), "called_strike_share": x.orig_called_strike.astype(bool).mean()})
r = pd.DataFrame(rows); r.to_csv(PROCESSED / "runs_left_post.csv", index=False)
z = pd.DataFrame(zone); z.to_csv(PROCESSED / "umpire_zone_post.csv", index=False)
print(r.round(3).to_string(index=False)); print(); print(z.round(3).to_string(index=False))
