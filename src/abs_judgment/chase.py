"""Hitter-season chase and two-strike outcomes from the pitch table.

Chase is defined against the true ABS zone at mid-plate (any part of the
ball overlapping the zone is a strike), so the chase rate here is cleaner
than the Statcast/Gameday zone version. A rulebook-zone version is also
computed for year-over-year comparison with seasons that have no ABS
bounds in the feed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RULEBOOK_HALF_WIDTH_FT = (17 / 2 + 2.9 / 2) / 12  # plate half-width plus ball radius


def rulebook_strike(plate_x, plate_z, sz_top, sz_bot) -> pd.Series:
    """Statcast-style zone at the front of the plate, with ball-radius allowance."""
    return (plate_x.abs() <= RULEBOOK_HALF_WIDTH_FT) & (plate_z >= sz_bot - 2.9 / 24) & (plate_z <= sz_top + 2.9 / 24)


def hitter_season(pt: pd.DataFrame, min_pitches: int = 500) -> pd.DataFrame:
    p = pt.copy()
    p["pa_key"] = p.game_pk.astype(str) + "_" + p.at_bat_number.astype(str)
    p["pa_end"] = p.events.notna() if "events" in p.columns else False
    p["is_k"] = p.events.isin(["strikeout", "strikeout_double_play"]) if "events" in p.columns else False
    p["is_bb"] = p.events.isin(["walk", "intent_walk"]) if "events" in p.columns else False
    p["chase_whiff"] = p.is_chase & p.is_whiff
    p["chase_contact"] = p.is_chase & p.is_contact
    p["chase_dre"] = np.where(p.is_chase, p.delta_run_exp, np.nan) if "delta_run_exp" in p.columns else np.nan
    p["chase_xwoba_con"] = np.where(p.chase_contact, p.estimated_woba_using_speedangle, np.nan) if "estimated_woba_using_speedangle" in p.columns else np.nan
    p["ts_swing"] = p.two_strike & p.is_swing
    p["ts_whiff"] = p.two_strike & p.is_whiff
    p["ts_contact"] = p.two_strike & p.is_contact
    p["ts_chase"] = p.two_strike & p.is_chase
    p["ts_inplay"] = p.two_strike & p.is_in_play.astype("boolean").fillna(False)
    p["ts_dre"] = np.where(p.two_strike & p.is_swing, p.delta_run_exp, np.nan) if "delta_run_exp" in p.columns else np.nan
    p["abs_ball_seen"] = p.abs_edge_in > 0
    p["abs_zone_seen"] = p.abs_edge_in <= 0
    p["take_abs_ball"] = p.is_take & p.abs_ball_seen
    p["lt2_chase"] = ~p.two_strike & p.is_chase
    p["lt2_abs_ball"] = ~p.two_strike & p.abs_ball_seen
    p["ts_abs_ball"] = p.two_strike & p.abs_ball_seen
    if {"sc_plate_x", "sc_plate_z", "sc_sz_top", "sc_sz_bot"}.issubset(p.columns):
        rb = rulebook_strike(p.sc_plate_x, p.sc_plate_z, p.sc_sz_top, p.sc_sz_bot)
        p["rb_ball_seen"] = ~rb & p.sc_plate_x.notna()
        p["rb_chase"] = p.is_swing & p.rb_ball_seen
    else:
        p["rb_ball_seen"] = False
        p["rb_chase"] = False

    g = p.groupby("batter_id")
    out = pd.DataFrame({
        "batter_name": g.batter_name.first(),
        "bat_side": g.bat_side.first(),
        "pitches": g.size(),
        "pa": g.pa_key.nunique(),
        "swings": g.is_swing.sum(),
        "abs_balls_seen": g.abs_ball_seen.sum(),
        "chases": g.is_chase.sum(),
        "zone_swings": g.is_zone_swing.sum(),
        "zone_pitches": g.abs_zone_seen.sum(),
        "chase_whiffs": g.chase_whiff.sum(),
        "chase_contacts": g.chase_contact.sum(),
        "chase_dre_mean": g.chase_dre.mean(),
        "chase_xwoba_con": g.chase_xwoba_con.mean(),
        "lt2_chases": g.lt2_chase.sum(), "lt2_abs_balls": g.lt2_abs_ball.sum(),
        "ts_chases": g.ts_chase.sum(), "ts_abs_balls": g.ts_abs_ball.sum(),
        "ts_pitches": g.two_strike.sum(),
        "ts_swings": g.ts_swing.sum(), "ts_whiffs": g.ts_whiff.sum(), "ts_contacts": g.ts_contact.sum(),
        "ts_inplay": g.ts_inplay.sum(), "ts_dre_mean": g.ts_dre.mean(),
        "k": g.is_k.sum(), "bb": g.is_bb.sum(),
        "rb_balls_seen": g.rb_ball_seen.sum(), "rb_chases": g.rb_chase.sum(),
        "bat_speed": g.bat_speed.mean() if "bat_speed" in p.columns else np.nan,
        "swing_length": g.swing_length.mean() if "swing_length" in p.columns else np.nan,
        "age": g.age_bat.first() if "age_bat" in p.columns else np.nan,
    })
    out = out[out.pitches >= min_pitches].copy()
    out["chase_rate"] = out.chases / out.abs_balls_seen
    out["zone_swing_rate"] = out.zone_swings / out.zone_pitches
    out["chase_whiff_rate"] = out.chase_whiffs / out.chases
    out["chase_rate_lt2"] = out.lt2_chases / out.lt2_abs_balls
    out["chase_rate_2k"] = out.ts_chases / out.ts_abs_balls
    out["ts_contact_rate"] = out.ts_contacts / out.ts_swings
    out["ts_whiff_rate"] = out.ts_whiffs / out.ts_swings
    out["k_rate"] = out.k / out.pa
    out["bb_rate"] = out.bb / out.pa
    out["chase_runs_per_100"] = 100 * out.chase_dre_mean
    out["rb_chase_rate"] = out.rb_chases / out.rb_balls_seen
    return out.reset_index()


def rulebook_chase_from_statcast(sc: pd.DataFrame, min_pitches: int = 500) -> pd.DataFrame:
    """Rulebook-zone chase rate from a raw Statcast CSV season (no ABS needed)."""
    s = sc[sc.plate_x.notna() & sc.sz_top.notna()].copy()
    rb = rulebook_strike(s.plate_x, s.plate_z, s.sz_top, s.sz_bot)
    swing = s.description.isin({"foul", "swinging_strike", "hit_into_play", "foul_tip", "swinging_strike_blocked",
                                "foul_bunt", "missed_bunt", "bunt_foul_tip", "foul_pitchout", "swinging_pitchout"})
    s["rb_ball"] = ~rb
    s["rb_chase"] = swing & ~rb
    g = s.groupby("batter")
    out = pd.DataFrame({"pitches": g.size(), "rb_balls_seen": g.rb_ball.sum(), "rb_chases": g.rb_chase.sum()})
    out = out[out.pitches >= min_pitches]
    out["rb_chase_rate"] = out.rb_chases / out.rb_balls_seen
    return out.reset_index().rename(columns={"batter": "batter_id"})
