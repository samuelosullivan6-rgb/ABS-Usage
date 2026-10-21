"""MLB StatsAPI access: schedule, cached live feeds, and extraction of
pitch rows, ABS challenge events and per-pitch token state.

Verified structure (2026-10-21, see docs/data_findings_2026-10-21.md):
  * gameData.absChallenges            per-game token ledger
  * playEvents[i].reviewDetails       mid-PA challenge (reviewType "MJ")
  * allPlays[i].reviewDetails         PA-ending challenge (on the last pitch)
  * details.call is the POST-challenge call
"""
from __future__ import annotations

import gzip
import json
import logging
import time
from pathlib import Path
from typing import Iterable

import requests

from .config import RAW_STATSAPI, TOKENS_PER_GAME, USER_AGENT

log = logging.getLogger(__name__)

SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
FEED_URL = "https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live"
ABS_REVIEW_TYPE = "MJ"

_session = requests.Session()
_session.headers["User-Agent"] = USER_AGENT


# --------------------------------------------------------------------------- #
# Fetching
# --------------------------------------------------------------------------- #
def list_games(start: str, end: str, sport_id: int = 1, game_type: str = "R") -> list[dict]:
    """Final games between two dates (inclusive). Returns gamePk, date, teams."""
    r = _session.get(
        SCHEDULE_URL,
        params={"sportId": sport_id, "startDate": start, "endDate": end, "gameType": game_type},
        timeout=60,
    )
    r.raise_for_status()
    out = []
    for day in r.json().get("dates", []):
        for g in day["games"]:
            if g["status"].get("codedGameState") not in ("F", "O"):
                continue  # final / game over only
            out.append(
                {
                    "game_pk": g["gamePk"],
                    "game_date": day["date"],
                    "sport_id": sport_id,
                    "home_id": g["teams"]["home"]["team"]["id"],
                    "away_id": g["teams"]["away"]["team"]["id"],
                }
            )
    return out


def feed_path(pk: int, cache_dir: Path = RAW_STATSAPI) -> Path:
    return cache_dir / f"{pk}.json.gz"


def fetch_feed(pk: int, cache_dir: Path = RAW_STATSAPI, retries: int = 4) -> dict:
    """Live feed for one game, cached as gzip JSON."""
    p = feed_path(pk, cache_dir)
    if p.exists():
        with gzip.open(p, "rt", encoding="utf-8") as f:
            return json.load(f)
    for attempt in range(retries):
        try:
            r = _session.get(FEED_URL.format(pk=pk), timeout=90)
            r.raise_for_status()
            data = r.json()
            tmp = p.with_suffix(".tmp")
            with gzip.open(tmp, "wt", encoding="utf-8") as f:
                json.dump(data, f)
            tmp.replace(p)
            return data
        except Exception as e:  # noqa: BLE001
            wait = 2 ** attempt
            log.warning("feed %s attempt %d failed: %s; retry in %ss", pk, attempt + 1, e, wait)
            time.sleep(wait)
    raise RuntimeError(f"could not fetch feed {pk}")


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def _coords(pd_: dict) -> dict:
    c = pd_.get("coordinates", {}) or {}
    return {
        "x0": c.get("x0"), "y0": c.get("y0"), "z0": c.get("z0"),
        "vx0": c.get("vX0"), "vy0": c.get("vY0"), "vz0": c.get("vZ0"),
        "ax": c.get("aX"), "ay": c.get("aY"), "az": c.get("aZ"),
        "px": c.get("pX"), "pz": c.get("pZ"),
    }


def extract_game(feed: dict) -> tuple[list[dict], list[dict], dict]:
    """Return (pitch_rows, challenge_rows, game_summary).

    Token state is reconstructed chronologically under the observed rule:
    two per team per game, retained on an overturn, lost on a confirmation.
    A second column records the state under the candidate extra-innings
    rule (a team at zero is topped up to one at the start of each extra
    inning). Full-season ledger reconciliation decides between them.
    """
    gd = feed["gameData"]
    pk = feed["gamePk"]
    sport_id = gd.get("game", {}).get("sport", {}).get("id") or gd.get("teams", {}).get("home", {}).get("sport", {}).get("id")
    game_date = gd["datetime"]["officialDate"]
    home = gd["teams"]["home"]["id"]
    away = gd["teams"]["away"]["id"]
    venue_id = gd.get("venue", {}).get("id")

    tokens = {home: TOKENS_PER_GAME, away: TOKENS_PER_GAME}          # rule A: no reset
    tokens_b = {home: TOKENS_PER_GAME, away: TOKENS_PER_GAME}        # rule B: top-up in extras
    last_inning_seen_b = 0

    pitches: list[dict] = []
    challenges: list[dict] = []
    n_impossible_a = 0
    n_impossible_b = 0

    for play in feed["liveData"]["plays"]["allPlays"]:
        about = play["about"]
        inning = about["inning"]
        half = about["halfInning"]
        bat_team = away if half == "top" else home
        fld_team = home if half == "top" else away
        mu = play["matchup"]
        batter_id = mu["batter"]["id"]
        pitcher_id = mu["pitcher"]["id"]
        at_bat_number = play["atBatIndex"] + 1

        if inning >= 10 and inning != last_inning_seen_b:
            last_inning_seen_b = inning
            for t in (home, away):
                if tokens_b[t] == 0:
                    tokens_b[t] = 1

        play_rd = play.get("reviewDetails")
        play_level_abs = bool(play_rd and play_rd.get("reviewType") == ABS_REVIEW_TYPE)
        events = play["playEvents"]
        last_pitch_idx = max((i for i, ev in enumerate(events) if ev.get("isPitch")), default=None)

        balls = strikes = 0
        outs = about.get("outs", 0) if "outs" in about else None
        # outs before the play: take from the first event's count minus any out it produced is
        # unreliable; use the running count instead, seeded from the previous play's end state.
        running_outs = play["count"]["outs"] - sum(1 for ev in events if ev.get("details", {}).get("isOut"))
        running_outs = max(running_outs, 0)

        for i, ev in enumerate(events):
            det = ev.get("details", {}) or {}
            cnt = ev.get("count", {}) or {}
            if not ev.get("isPitch"):
                # pickoffs, stepoffs, automatic balls/strikes, substitutions: update running count
                balls = cnt.get("balls", balls)
                strikes = cnt.get("strikes", strikes)
                running_outs = cnt.get("outs", running_outs)
                continue

            pd_ = ev.get("pitchData", {}) or {}
            rd = ev.get("reviewDetails")
            pitch_level_abs = bool(rd and rd.get("reviewType") == ABS_REVIEW_TYPE)
            is_play_level_here = play_level_abs and i == last_pitch_idx
            challenged = pitch_level_abs or is_play_level_here
            crd = rd if pitch_level_abs else (play_rd if is_play_level_here else None)

            call = det.get("call", {}) or {}
            row = {
                "game_pk": pk, "game_date": game_date, "sport_id": sport_id, "venue_id": venue_id,
                "home_id": home, "away_id": away,
                "at_bat_number": at_bat_number, "pitch_number": ev.get("pitchNumber"),
                "event_index": ev.get("index"), "play_id": ev.get("playId"),
                "inning": inning, "half": half, "bat_team": bat_team, "fld_team": fld_team,
                "batter_id": batter_id, "batter_name": mu["batter"]["fullName"],
                "bat_side": mu["batSide"]["code"],
                "pitcher_id": pitcher_id, "pitcher_name": mu["pitcher"]["fullName"],
                "pitch_hand": mu["pitchHand"]["code"],
                "balls_before": balls, "strikes_before": strikes, "outs_before": running_outs,
                "call_code": call.get("code"), "call_desc": det.get("description"),
                "is_strike": det.get("isStrike"), "is_ball": det.get("isBall"), "is_in_play": det.get("isInPlay"),
                "pitch_type": (det.get("type") or {}).get("code"),
                "start_speed": pd_.get("startSpeed"), "zone": pd_.get("zone"),
                "sz_top": pd_.get("strikeZoneTop"), "sz_bot": pd_.get("strikeZoneBottom"),
                "bat_tokens_before": tokens[bat_team], "fld_tokens_before": tokens[fld_team],
                "bat_tokens_before_ruleB": tokens_b[bat_team], "fld_tokens_before_ruleB": tokens_b[fld_team],
                "challenged": challenged,
                "challenge_at_play_level": is_play_level_here,
                "challenge_team_id": crd.get("challengeTeamId") if crd else None,
                "challenge_player_id": (crd.get("player") or {}).get("id") if crd else None,
                "challenge_player_name": (crd.get("player") or {}).get("fullName") if crd else None,
                "challenge_overturned": crd.get("isOverturned") if crd else None,
                **_coords(pd_),
            }
            pitches.append(row)

            if challenged:
                ct = row["challenge_team_id"]
                side = "bat" if ct == bat_team else "fld"
                pid = row["challenge_player_id"]
                if pid is None:
                    role = "batter" if side == "bat" else "fielder"
                elif pid == batter_id:
                    role = "batter"
                elif pid == pitcher_id:
                    role = "pitcher"
                else:
                    role = "catcher"
                overturned = bool(row["challenge_overturned"])
                final_is_strike = bool(det.get("isStrike"))
                original_is_strike = (not final_is_strike) if overturned else final_is_strike
                ch = {
                    "game_pk": pk, "game_date": game_date, "sport_id": sport_id,
                    "at_bat_number": at_bat_number, "pitch_number": row["pitch_number"], "play_id": row["play_id"],
                    "inning": inning, "half": half,
                    "challenge_team_id": ct, "challenge_side": side,
                    "initiator_id": pid, "initiator_name": row["challenge_player_name"], "initiator_role": role,
                    "batter_id": batter_id, "pitcher_id": pitcher_id,
                    "is_overturned": overturned,
                    "original_is_strike": original_is_strike, "final_is_strike": final_is_strike,
                    "balls_before": balls, "strikes_before": strikes, "outs_before": running_outs,
                    "tokens_before": tokens[ct], "opp_tokens_before": tokens[fld_team if side == "bat" else bat_team],
                    "tokens_before_ruleB": tokens_b[ct],
                    "at_play_level": is_play_level_here,
                    "side_consistent": (side == "bat") == original_is_strike,
                }
                challenges.append(ch)
                if tokens[ct] <= 0:
                    n_impossible_a += 1
                if tokens_b[ct] <= 0:
                    n_impossible_b += 1
                if not overturned:
                    tokens[ct] = max(tokens[ct] - 1, 0)
                    tokens_b[ct] = max(tokens_b[ct] - 1, 0)

            balls = cnt.get("balls", balls)
            strikes = cnt.get("strikes", strikes)
            running_outs = cnt.get("outs", running_outs)

    ledger = gd.get("absChallenges") or {}
    box_line = next(
        (it.get("value") for it in feed["liveData"].get("boxscore", {}).get("info", []) if it.get("label") == "ABS Challenge"),
        None,
    )
    summary = {
        "game_pk": pk, "game_date": game_date, "sport_id": sport_id,
        "home_id": home, "away_id": away,
        "n_innings": max((p["about"]["inning"] for p in feed["liveData"]["plays"]["allPlays"]), default=0),
        "n_pitches": len(pitches), "n_challenges": len(challenges),
        "n_pitch_level": sum(1 for c in challenges if not c["at_play_level"]),
        "n_play_level": sum(1 for c in challenges if c["at_play_level"]),
        "ledger_home_used": (ledger.get("home") or {}).get("usedSuccessful", 0) + (ledger.get("home") or {}).get("usedFailed", 0),
        "ledger_away_used": (ledger.get("away") or {}).get("usedSuccessful", 0) + (ledger.get("away") or {}).get("usedFailed", 0),
        "ledger_home_remaining": (ledger.get("home") or {}).get("remaining"),
        "ledger_away_remaining": (ledger.get("away") or {}).get("remaining"),
        "sim_home_remaining_ruleA": tokens[home], "sim_away_remaining_ruleA": tokens[away],
        "sim_home_remaining_ruleB": tokens_b[home], "sim_away_remaining_ruleB": tokens_b[away],
        "n_impossible_ruleA": n_impossible_a, "n_impossible_ruleB": n_impossible_b,
        "extracted_home": sum(1 for c in challenges if c["challenge_team_id"] == home),
        "extracted_away": sum(1 for c in challenges if c["challenge_team_id"] == away),
        "box_abs_line": box_line,
    }
    summary["ledger_match"] = (
        summary["extracted_home"] == summary["ledger_home_used"]
        and summary["extracted_away"] == summary["ledger_away_used"]
    )
    return pitches, challenges, summary


def extract_many(pks: Iterable[int], cache_dir: Path = RAW_STATSAPI):
    """Yield (pitches, challenges, summary) for each cached game; skips missing."""
    for pk in pks:
        p = feed_path(pk, cache_dir)
        if not p.exists():
            continue
        try:
            yield extract_game(fetch_feed(pk, cache_dir))
        except Exception as e:  # noqa: BLE001
            log.error("extract failed for %s: %s", pk, e)
