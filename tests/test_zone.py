"""Zone geometry must reproduce Savant's own challenge coordinates and edge
distance. The fixture is Savant's embedded vizChallenges table for
2026-09-21..27 (404 rows). Trajectories come from the StatsAPI feed and
are cached by the test the first time it runs."""
from __future__ import annotations

import gzip
import json
import urllib.request
from pathlib import Path

import pytest

from abs_judgment.zone import (
    ABS_BOT_FRAC,
    ABS_TOP_FRAC,
    edge_distance_in,
    midplate_location,
    solve_at_y,
    PLATE_FRONT_Y,
)

HERE = Path(__file__).parent
FIXTURE = HERE / "savant_vizchallenges_2026-09-21_to_27.json"
FEED_CACHE = HERE / "_feed_cache"


def _feed(pk: int) -> dict:
    FEED_CACHE.mkdir(exist_ok=True)
    p = FEED_CACHE / f"{pk}.json.gz"
    if p.exists():
        with gzip.open(p, "rt") as f:
            return json.load(f)
    data = json.load(urllib.request.urlopen(f"https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live"))
    with gzip.open(p, "wt") as f:
        json.dump(data, f)
    return data


@pytest.fixture(scope="module")
def paired():
    rows = json.load(open(FIXTURE))
    by_pk: dict[int, list[dict]] = {}
    for r in rows:
        by_pk.setdefault(r["game_pk"], []).append(r)
    out = []
    for pk in sorted(by_pk)[:5]:  # five games, ~45 pitches
        feed = _feed(pk)
        idx = {ev["playId"]: ev for p in feed["liveData"]["plays"]["allPlays"] for ev in p["playEvents"] if ev.get("playId")}
        for r in by_pk[pk]:
            ev = idx.get(r["play_id"])
            if ev:
                out.append((r, ev["pitchData"]["coordinates"]))
    assert len(out) >= 20
    return out


def test_front_plate_matches_feed(paired):
    for r, c in paired:
        x, z, _ = solve_at_y(c, PLATE_FRONT_Y)
        assert abs(x - c["pX"]) < 2e-3
        assert abs(z - c["pZ"]) < 2e-3


def test_midplate_matches_savant(paired):
    for r, c in paired:
        x, z = midplate_location(c)
        assert abs(x - r["plateX"]) < 1e-4
        assert abs(z - r["plateZ"]) < 1e-4


def test_edge_distance_matches_savant(paired):
    for r, c in paired:
        x, z = midplate_location(c)
        ed = edge_distance_in(x, z, r["strikeZoneTop"], r["strikeZoneBottom"])
        assert abs(ed - r["edge_dist_calc"]) < 1e-4


def test_feed_zone_is_abs_height_zone(paired):
    for r, _ in paired:
        ratio = r["strikeZoneTop"] / r["strikeZoneBottom"]
        assert abs(ratio - ABS_TOP_FRAC / ABS_BOT_FRAC) < 0.01
