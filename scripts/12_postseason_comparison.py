"""Postseason against regular season, batter challenges, with the brief's definitions.

A "miss" is an umpire-called strike the hitter took, that ABS would have called a
ball (edge > 0), while the batting team held a token (per-inning extras rule).
Direction is the side of the box the pitch's centre is on (judgment.miss_direction).

Usage: python scripts/12_postseason_comparison.py
Outputs (data/processed/, all with the _post suffix; regular-season tables are read only):
  postseason_comparison_post.csv      one row per quantity, regular vs postseason, with n and SE
  monthly_trend_post.csv              regular-season challenge and overturn rates by month
  team_rates_post.csv                 share of hitters' misses challenged, by team, both seasons
  direction_post.csv                  challenge rate on misses by direction and distance band, both seasons
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from abs_judgment.config import PROCESSED  # noqa: E402
from abs_judgment.judgment import miss_direction  # noqa: E402

COLS = ["game_pk", "game_date", "half", "home_team", "away_team", "bat_team", "orig_called_strike", "bat_tokens_before_ruleB",
        "abs_edge_in", "batter_challenged", "challenge_overturned", "strikes_before", "balls_before", "inning",
        "bat_score", "fld_score", "mid_x", "mid_z", "sz_top", "sz_bot"]


def se_p(p: float, n: int) -> float:
    return float(np.sqrt(p * (1 - p) / n)) if n else float("nan")


def load(label: str) -> pd.DataFrame:
    pt = pd.read_parquet(PROCESSED / f"pitch_table_{label}.parquet", columns=COLS)
    pt["team"] = np.where(pt.half == "top", pt.away_team, pt.home_team)
    pt["direction"] = miss_direction(pt.mid_x, pt.mid_z, pt.sz_top, pt.sz_bot)
    pt["close"] = (pt.bat_score - pt.fld_score).abs() <= 1
    pt["month"] = pd.to_datetime(pt.game_date).dt.month
    return pt


def misses(pt: pd.DataFrame) -> pd.DataFrame:
    m = pt[(pt.orig_called_strike & (pt.abs_edge_in > 0) & (pt.bat_tokens_before_ruleB >= 1)).fillna(False).astype(bool)].copy()
    m["chal"] = m.batter_challenged.astype(bool)
    return m


def summarize(pt: pd.DataFrame, season: str, n_games: int) -> list[dict]:
    m = misses(pt)
    ch = pt[pt.batter_challenged.astype(bool)]
    won = ch.challenge_overturned.astype(bool)
    tg = 2 * n_games
    rows = [
        {"quantity": "games", "value": n_games, "n": n_games, "se": np.nan},
        {"quantity": "batter challenges", "value": len(ch), "n": len(ch), "se": np.nan},
        {"quantity": "batter challenges per team-game", "value": len(ch) / tg, "n": tg, "se": np.nan},
        {"quantity": "umpire misses on hitters (true balls called strikes, token in hand)", "value": len(m), "n": len(m), "se": np.nan},
        {"quantity": "misses per team-game", "value": len(m) / tg, "n": tg, "se": np.nan},
        {"quantity": "share of misses challenged", "value": m.chal.mean(), "n": len(m), "se": se_p(m.chal.mean(), len(m))},
        {"quantity": "overturn rate, batter challenges", "value": won.mean(), "n": len(ch), "se": se_p(won.mean(), len(ch))},
        {"quantity": "share of batter challenges that were true balls", "value": (ch.abs_edge_in > 0).mean(), "n": len(ch), "se": se_p((ch.abs_edge_in > 0).mean(), len(ch))},
        {"quantity": "median edge of batter challenges (in; + outside)", "value": ch.abs_edge_in.median(), "n": len(ch), "se": np.nan},
        {"quantity": "median edge of LOST batter challenges (in inside)", "value": -ch.abs_edge_in[~won].median(), "n": int((~won).sum()), "se": np.nan},
        {"quantity": "share of batter challenges with two strikes", "value": (ch.strikes_before == 2).mean(), "n": len(ch), "se": se_p((ch.strikes_before == 2).mean(), len(ch))},
        {"quantity": "share of batter challenges with three balls", "value": (ch.balls_before == 3).mean(), "n": len(ch), "se": se_p((ch.balls_before == 3).mean(), len(ch))},
        {"quantity": "share of batter challenges in the 7th or later", "value": (ch.inning >= 7).mean(), "n": len(ch), "se": se_p((ch.inning >= 7).mean(), len(ch))},
        {"quantity": "share of batter challenges in a close game (within 1 run)", "value": ch.close.mean(), "n": len(ch), "se": se_p(ch.close.mean(), len(ch))},
        {"quantity": "share of misses challenged, two strikes", "value": m[m.strikes_before == 2].chal.mean(), "n": int((m.strikes_before == 2).sum()), "se": se_p(m[m.strikes_before == 2].chal.mean(), int((m.strikes_before == 2).sum()))},
        {"quantity": "share of misses challenged, 7th or later", "value": m[m.inning >= 7].chal.mean(), "n": int((m.inning >= 7).sum()), "se": se_p(m[m.inning >= 7].chal.mean(), int((m.inning >= 7).sum()))},
        {"quantity": "share of misses challenged, close and 7th or later", "value": m[m.close & (m.inning >= 7)].chal.mean(), "n": int((m.close & (m.inning >= 7)).sum()), "se": se_p(m[m.close & (m.inning >= 7)].chal.mean(), int((m.close & (m.inning >= 7)).sum()))},
        {"quantity": "strike-three misses unchallenged per team-game", "value": ((m.strikes_before == 2) & ~m.chal).sum() / tg, "n": int((m.strikes_before == 2).sum()), "se": np.nan},
    ]
    for k in ("above", "below", "side", "corner"):
        mm = m[m.direction == k]
        rows.append({"quantity": f"share of misses challenged, {k}", "value": mm.chal.mean(), "n": len(mm), "se": se_p(mm.chal.mean(), len(mm))})
        rows.append({"quantity": f"share of misses that are {k}", "value": len(mm) / len(m), "n": len(m), "se": np.nan})
    for k, mm in (("0-1 in", m[m.abs_edge_in <= 1]), ("1-2 in", m[(m.abs_edge_in > 1) & (m.abs_edge_in <= 2)]), ("2+ in", m[m.abs_edge_in > 2])):
        rows.append({"quantity": f"share of misses challenged, {k} outside", "value": mm.chal.mean(), "n": len(mm), "se": se_p(mm.chal.mean(), len(mm))})
    for r in rows:
        r["season"] = season
    return rows


def main() -> None:
    reg, post = load("mlb_2026"), load("mlb_2026_post")
    g_reg = int(pd.read_csv(PROCESSED / "game_summaries_mlb_2026.csv").game_pk.nunique())
    g_post = int(pd.read_csv(PROCESSED / "game_summaries_mlb_2026_post.csv").game_pk.nunique())

    rows = summarize(reg, "regular", g_reg) + summarize(post, "postseason", g_post)
    cmp_ = pd.DataFrame(rows).pivot(index="quantity", columns="season", values=["value", "n", "se"])
    cmp_.columns = [f"{b}_{a}" for a, b in cmp_.columns]
    order = [r["quantity"] for r in rows[: len(rows) // 2]]
    cmp_ = cmp_.loc[order, ["regular_value", "regular_n", "regular_se", "postseason_value", "postseason_n", "postseason_se"]]
    cmp_.to_csv(PROCESSED / "postseason_comparison_post.csv")
    pd.set_option("display.width", 200)
    print("=== regular vs postseason, batter challenges ===")
    print(cmp_.round(4).to_string())

    # monthly learning curve, regular season
    m = misses(reg)
    ch = reg[reg.batter_challenged.astype(bool)]
    mon = pd.DataFrame({
        "misses": m.groupby("month").size(),
        "share_challenged": m.groupby("month").chal.mean(),
        "challenges": ch.groupby("month").size(),
        "overturn_rate": ch.groupby("month").challenge_overturned.apply(lambda s: s.astype(bool).mean()),
        "share_high_misses_challenged": m[m.direction == "above"].groupby("month").chal.mean(),
        "share_low_misses_challenged": m[m.direction == "below"].groupby("month").chal.mean(),
        "team_games": reg.groupby("month").game_pk.nunique() * 2,
    })
    mon["challenges_per_team_game"] = mon.challenges / mon.team_games
    mon["se_share"] = np.sqrt(mon.share_challenged * (1 - mon.share_challenged) / mon.misses)
    mon["se_overturn"] = np.sqrt(mon.overturn_rate * (1 - mon.overturn_rate) / mon.challenges)
    mon.index.name = "month"
    mon.to_csv(PROCESSED / "monthly_trend_post.csv")
    print("\n=== regular season by month ===")
    print(mon.round(3).to_string())

    # team spread
    tr = m.groupby("team").chal.agg(misses="size", share_challenged="mean")
    mp = misses(post)
    tp = mp.groupby("team").chal.agg(misses_post="size", share_challenged_post="mean")
    chp = post[post.batter_challenged.astype(bool)].groupby("team").size().rename("batter_challenges_post")
    gp = pd.concat([post.groupby("team").game_pk.nunique().rename("games_post")], axis=1)
    teams = tr.join(tp, how="left").join(chp, how="left").join(gp, how="left").sort_values("share_challenged", ascending=False)
    teams["rank_regular"] = np.arange(1, len(teams) + 1)
    teams.to_csv(PROCESSED / "team_rates_post.csv")
    print("\n=== share of hitters' misses challenged by team (regular rank 1 = highest); postseason teams ===")
    print(f"regular: min {tr.share_challenged.min():.3f}, median {tr.share_challenged.median():.3f}, max {tr.share_challenged.max():.3f}")
    print(teams[teams.games_post.notna()].round(3).to_string())

    # direction x distance, both seasons
    out = []
    for season, mm in (("regular", m), ("postseason", mp)):
        for k in ("above", "below", "side", "corner", "all"):
            x = mm if k == "all" else mm[mm.direction == k]
            for band, bm in (("0-1", x.abs_edge_in <= 1), ("1-2", (x.abs_edge_in > 1) & (x.abs_edge_in <= 2)), ("2+", x.abs_edge_in > 2), ("all", x.abs_edge_in > 0)):
                y = x[bm]
                out.append({"season": season, "direction": k, "band_in": band, "misses": len(y), "share_challenged": y.chal.mean() if len(y) else np.nan})
    dd = pd.DataFrame(out)
    dd.to_csv(PROCESSED / "direction_post.csv", index=False)
    print("\n=== share of misses challenged by direction (all distances) ===")
    print(dd[dd.band_in == "all"].pivot(index="direction", columns="season", values=["misses", "share_challenged"]).round(3).to_string())


if __name__ == "__main__":
    main()
