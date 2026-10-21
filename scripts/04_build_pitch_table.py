"""Build the season pitch table: StatsAPI pitches + challenges + token state,
mid-plate ABS geometry, and the Statcast CSV join (run values, bat tracking).

Usage: python scripts/04_build_pitch_table.py --season 2026 --sport 1
Outputs (data/processed/):
  pitch_table_<label>.parquet   one row per pitch
  challenges_<label>.parquet    one row per ABS challenge
  game_summaries_<label>.csv    per-game ledger reconciliation
  build_report_<label>.txt      join rates and rule checks
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from abs_judgment.config import PROCESSED, RAW_STATCAST, SEASONS  # noqa: E402
from abs_judgment.savant import load_statcast_days  # noqa: E402
from abs_judgment.statsapi import extract_many, feed_path  # noqa: E402
from abs_judgment.zone import edge_distance_in_np, midplate_location_np  # noqa: E402

SWING_CODES = {"F", "S", "X", "D", "E", "T", "W", "L", "M", "R", "Q"}
TAKE_CODES = {"B", "*B", "C", "H", "I", "P", "V", "A"}
WHIFF_DESC = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}
SWING_DESC = {"foul", "swinging_strike", "hit_into_play", "foul_tip", "swinging_strike_blocked",
              "foul_bunt", "missed_bunt", "bunt_foul_tip", "foul_pitchout", "swinging_pitchout"}

SC_KEEP = [
    "game_pk", "at_bat_number", "pitch_number", "description", "events", "type", "bb_type",
    "launch_speed", "launch_angle", "launch_speed_angle", "estimated_woba_using_speedangle",
    "woba_value", "woba_denom", "delta_run_exp", "delta_pitcher_run_exp", "delta_home_win_exp",
    "home_win_exp", "bat_win_exp", "bat_score", "fld_score", "post_bat_score", "home_score_diff",
    "on_1b", "on_2b", "on_3b", "outs_when_up", "fielder_2", "plate_x", "plate_z", "sz_top", "sz_bot",
    "zone", "pitch_type", "pitch_name", "release_speed", "pfx_x", "pfx_z", "release_spin_rate",
    "release_extension", "arm_angle", "bat_speed", "swing_length", "miss_distance", "attack_angle",
    "attack_direction", "swing_path_tilt", "intercept_ball_minus_batter_pos_x_inches",
    "intercept_ball_minus_batter_pos_y_inches", "stand", "p_throws", "age_bat", "age_pit",
    "n_priorpa_thisgame_player_at_bat", "home_team", "away_team",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--sport", type=int, default=1)
    a = ap.parse_args()
    cfg = SEASONS[(a.season, a.sport)]
    label = cfg["label"]
    rep: list[str] = [f"# build report {label}"]

    games = pd.read_csv(PROCESSED / f"games_{label}.csv")
    pks = [int(pk) for pk in games.game_pk if feed_path(int(pk)).exists()]
    rep.append(f"games listed {len(games)}, feeds cached {len(pks)}")

    P, C, S = [], [], []
    for pitches, challenges, summary in extract_many(pks):
        P.extend(pitches); C.extend(challenges); S.append(summary)
    pt = pd.DataFrame(P); ch = pd.DataFrame(C); gs = pd.DataFrame(S)
    rep.append(f"pitches {len(pt):,}  challenges {len(ch):,}  games {len(gs):,}")

    # ---- ledger reconciliation and the extra-innings rule -------------------
    rep.append(f"ledger match (counts): {gs.ledger_match.mean():.4f} of games")
    for rule in ("A", "B"):
        rem_ok = ((gs[f"sim_home_remaining_rule{rule}"] == gs.ledger_home_remaining)
                  & (gs[f"sim_away_remaining_rule{rule}"] == gs.ledger_away_remaining)).mean()
        imp = gs[f"n_impossible_rule{rule}"].sum()
        rep.append(f"rule {rule}: end-of-game remaining matches ledger in {rem_ok:.4f} of games; "
                   f"challenges made with 0 reconstructed tokens: {imp}")
    xi = gs[gs.n_innings >= 10]
    rep.append(f"extra-inning games: {len(xi)}; of those, rule A remaining-match "
               f"{((xi.sim_home_remaining_ruleA == xi.ledger_home_remaining) & (xi.sim_away_remaining_ruleA == xi.ledger_away_remaining)).mean():.4f}, "
               f"rule B {((xi.sim_home_remaining_ruleB == xi.ledger_home_remaining) & (xi.sim_away_remaining_ruleB == xi.ledger_away_remaining)).mean():.4f}")
    if len(ch):
        rep.append(f"challenge side consistency: {ch.side_consistent.mean():.4f}")
        rep.append("initiator roles: " + ch.initiator_role.value_counts().to_dict().__repr__())
        rep.append(f"overturn rate: all {ch.is_overturned.mean():.3f}; " +
                   "; ".join(f"{r} {g.is_overturned.mean():.3f} (n={len(g)})" for r, g in ch.groupby("initiator_role")))

    # ---- ABS geometry for every pitch with a trajectory ----------------------
    has_traj = pt[["x0", "y0", "z0", "vx0", "vy0", "vz0", "ax", "ay", "az", "sz_top", "sz_bot"]].notna().all(axis=1)
    rep.append(f"pitches with full trajectory: {has_traj.mean():.4f}")
    t = pt[has_traj]
    mx, mz = midplate_location_np(*(t[c].to_numpy(float) for c in ["x0", "y0", "z0", "vx0", "vy0", "vz0", "ax", "ay", "az"]))
    pt.loc[has_traj, "mid_x"] = mx
    pt.loc[has_traj, "mid_z"] = mz
    pt.loc[has_traj, "abs_edge_in"] = edge_distance_in_np(mx, mz, t.sz_top.to_numpy(float), t.sz_bot.to_numpy(float))
    pt["abs_strike"] = pt.abs_edge_in <= 0
    pt.loc[pt.abs_edge_in.isna(), "abs_strike"] = np.nan

    # Validation: ABS outcome on challenged pitches should match the geometry
    chal = pt[pt.challenged & pt.abs_edge_in.notna()].copy()
    if len(chal):
        chal_truth_strike = np.where(chal.challenge_overturned.astype(bool), ~chal.is_strike.astype(bool), chal.is_strike.astype(bool))
        # final call after challenge is the ABS truth
        abs_truth_strike = chal.is_strike.astype(bool)
        agree = (abs_truth_strike.to_numpy() == (chal.abs_edge_in <= 0).to_numpy()).mean()
        rep.append(f"geometry agrees with ABS ruling on challenged pitches: {agree:.4f} (n={len(chal)})")
        near = chal[chal.abs_edge_in.abs() < 0.25]
        rep.append(f"  disagreements within 0.25 in of the edge: {(near.is_strike.astype(bool).to_numpy() != (near.abs_edge_in <= 0).to_numpy()).sum()} of {len(near)}")

    # ---- Statcast join (MLB only) --------------------------------------------
    if a.sport == 1:
        days = sorted(RAW_STATCAST.glob(f"{a.season}-*.csv"))
        sc = load_statcast_days(days)
        rep.append(f"statcast days {len(days)}, rows {len(sc):,}")
        sc = sc[[c for c in SC_KEEP if c in sc.columns]].copy()
        sc = sc.rename(columns={"zone": "sc_zone", "sz_top": "sc_sz_top", "sz_bot": "sc_sz_bot",
                                "pitch_type": "sc_pitch_type", "plate_x": "sc_plate_x", "plate_z": "sc_plate_z"})
        key = ["game_pk", "at_bat_number", "pitch_number"]
        sc = sc.drop_duplicates(key)
        before = len(pt)
        pt = pt.merge(sc, on=key, how="left", validate="one_to_one")
        joined = pt.description.notna().mean()
        rep.append(f"statcast join rate: {joined:.4f} (rows {before:,} -> {len(pt):,})")
        # plate coords should agree at the front of the plate
        ok = pt.sc_plate_x.notna() & pt.px.notna()
        rep.append(f"  front-plate x agreement |px-plate_x|<0.01: {((pt.px - pt.sc_plate_x).abs() < 0.01)[ok].mean():.4f}")
        # runs to end of half-inning (for RE288)
        grp = pt.groupby(["game_pk", "inning", "half"])
        pt["half_final_bat_score"] = grp.post_bat_score.transform("max")
        pt["runs_rest_of_inning"] = pt.half_final_bat_score - pt.bat_score
        pt["base_state"] = pt.on_1b.notna().astype(int) + 2 * pt.on_2b.notna().astype(int) + 4 * pt.on_3b.notna().astype(int)

    # ---- derived flags -------------------------------------------------------
    if "description" in pt.columns:
        pt["is_swing"] = pt.description.isin(SWING_DESC)
        pt["is_whiff"] = pt.description.isin(WHIFF_DESC)
        pt.loc[pt.description.isna(), "is_swing"] = pt.call_code.isin(SWING_CODES)
    else:
        pt["is_swing"] = pt.call_code.isin(SWING_CODES)
        pt["is_whiff"] = pt.call_code.isin({"S", "W", "M"})
    pt["is_take"] = ~pt.is_swing
    pt["is_contact"] = pt.is_swing & ~pt.is_whiff
    pt["two_strike"] = pt.strikes_before == 2
    pt["batter_challenged"] = pt.challenged & (pt.challenge_team_id == pt.bat_team)
    pt["fielder_challenged"] = pt.challenged & (pt.challenge_team_id == pt.fld_team)
    # original umpire call (pre-challenge)
    pt["orig_called_strike"] = ((pt.call_code == "C") & ~pt.challenged) | pt.batter_challenged
    pt["orig_called_ball"] = (pt.is_ball.astype("boolean").fillna(False) & ~pt.challenged) | pt.fielder_challenged
    pt["is_chase"] = pt.is_swing & (pt.abs_strike == False)  # noqa: E712
    pt["is_zone_swing"] = pt.is_swing & (pt.abs_strike == True)  # noqa: E712

    pt.to_parquet(PROCESSED / f"pitch_table_{label}.parquet", index=False)
    ch.to_parquet(PROCESSED / f"challenges_{label}.parquet", index=False)
    gs.to_csv(PROCESSED / f"game_summaries_{label}.csv", index=False)
    (PROCESSED / f"build_report_{label}.txt").write_text("\n".join(rep) + "\n")
    print("\n".join(rep))


if __name__ == "__main__":
    main()
