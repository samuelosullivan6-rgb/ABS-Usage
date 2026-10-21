"""Extraction invariants on two cached 2026 feeds.

822724: four challenges, three pitch-level (one overturned) and one PA-ending.
823452: one PA-ending challenge, confirmed.
"""
import gzip
import json
from pathlib import Path

import pytest

from abs_judgment.statsapi import extract_game

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module", params=["822724", "823452"])
def game(request):
    with gzip.open(FIX / f"{request.param}.json.gz", "rt") as f:
        feed = json.load(f)
    return extract_game(feed)


def test_challenge_counts_match_ledger(game):
    _, ch, s = game
    assert s["ledger_match"]
    assert len(ch) == s["ledger_home_used"] + s["ledger_away_used"]


def test_token_reconstruction_matches_ledger(game):
    _, _, s = game
    assert s["sim_home_remaining_ruleA"] == s["ledger_home_remaining"]
    assert s["sim_away_remaining_ruleA"] == s["ledger_away_remaining"]
    assert s["n_impossible_ruleA"] == 0


def test_side_is_consistent_with_original_call(game):
    _, ch, _ = game
    assert all(c["side_consistent"] for c in ch)
    for c in ch:
        if c["challenge_side"] == "bat":
            assert c["original_is_strike"] and c["initiator_role"] == "batter"
        else:
            assert not c["original_is_strike"] and c["initiator_role"] in ("catcher", "pitcher")


def test_challenged_pitches_are_flagged_in_pitch_rows(game):
    p, ch, _ = game
    flagged = {(r["at_bat_number"], r["pitch_number"]) for r in p if r["challenged"]}
    assert flagged == {(c["at_bat_number"], c["pitch_number"]) for c in ch}


def test_counts_before_are_consistent(game):
    p, _, _ = game
    for r in p:
        assert 0 <= r["balls_before"] <= 3 and 0 <= r["strikes_before"] <= 2 and 0 <= r["outs_before"] <= 2


def test_overturned_batter_challenge_has_ball_as_final_call():
    with gzip.open(FIX / "822724.json.gz", "rt") as f:
        p, ch, _ = extract_game(json.load(f))
    ot = [c for c in ch if c["is_overturned"] and c["challenge_side"] == "bat"]
    assert ot, "fixture should contain an overturned batter challenge"
    for c in ot:
        row = next(r for r in p if (r["at_bat_number"], r["pitch_number"]) == (c["at_bat_number"], c["pitch_number"]))
        assert row["is_ball"] and not row["is_strike"]
        assert c["tokens_before"] >= 1
