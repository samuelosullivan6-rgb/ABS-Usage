"""The judgment model.

Synthetic recovery: hitters with sharper perception (smaller sigma) must get
larger estimated slopes, and hitters with a looser criterion must get larger
intercepts, after empirical-Bayes shrinkage; direction intercepts that differ
must be recovered as fixed effects, and a hitter's criterion must be net of
his direction mix. Fixture games: the side of the box each called strike
missed on must agree with the zone geometry."""
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from abs_judgment.judgment import (
    DIR_COLS, build_decision_table, eyes_label_threshold, fit_hierarchical, miss_direction, split_half_reliability,
)
from abs_judgment.statsapi import extract_game
from abs_judgment.zone import BALL_RADIUS_IN, edge_distance_in_np, midplate_location_np

FIX = Path(__file__).parent / "fixtures"


def _simulate(n_hitters=120, n_per=300, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    sigmas = rng.uniform(0.6, 2.4, n_hitters)        # perceptual noise, inches
    taus = rng.uniform(0.8, 2.2, n_hitters)          # criterion: perceived edge needed to challenge
    for i in range(n_hitters):
        edge = rng.normal(-1.0, 2.0, n_per)          # true edge distance of called strikes
        perceived = edge + rng.normal(0, sigmas[i], n_per)
        two_strike = rng.integers(0, 2, n_per)
        # looser criterion with two strikes (strike three is costlier to accept)
        y = (perceived > taus[i] - 0.5 * two_strike).astype(int)
        rows.append(pd.DataFrame({
            "batter_id": i, "y": y, "edge": edge, "abs_ball": (edge > 0).astype(int),
            "two_strike": two_strike, "three_ball": 0, "tokens2": 1, "late": 0, "close": 0,
        }))
    return pd.concat(rows, ignore_index=True), sigmas, taus


def test_slope_tracks_perceptual_sharpness():
    d, sigmas, taus = _simulate()
    R, prior = fit_hierarchical(d, min_decisions=50, n_iter=3)
    R = R.sort_values("batter_id")
    r_slope = np.corrcoef(R.slope, 1 / sigmas)[0, 1]
    assert r_slope > 0.6, r_slope
    # criterion: hitters needing a larger perceived margin have lower intercepts
    r_int = np.corrcoef(R.edge_at_50, taus)[0, 1]
    assert r_int > 0.6, r_int
    assert prior["fixed"]["two_strike"] > 0  # two strikes loosen the criterion
    # full-covariance prior: symmetric, positive definite, correlations reported
    C = prior["cov"]
    assert np.allclose(C, C.T) and np.all(np.linalg.eigvalsh(C) > 0)
    assert -1 < prior["corr"] < 1 and R.post_corr.between(-1, 1).all()


GAMMA = {"dir_above": -0.4, "dir_below": 0.6, "dir_corner": 0.8}  # against a pitch off the side


def _simulate_direction(n_hitters=150, n_per=400, seed=3):
    """Logit challenges with direction intercepts and hitter-specific direction mixes:
    some hitters see mostly high strikes, others mostly low ones."""
    rng = np.random.default_rng(seed)
    a = rng.normal(-3.0, 0.6, n_hitters)
    b = rng.normal(0.65, 0.15, n_hitters).clip(0.2)
    w_lows = rng.uniform(0.05, 0.55, n_hitters)  # share of low strikes; the rest of the 60% are high
    rows = []
    for i in range(n_hitters):
        w_low = w_lows[i]
        mix = np.array([0.6 - w_low, w_low, 0.05, 0.35])  # above, below, corner, side
        k = rng.choice(4, n_per, p=mix / mix.sum())
        dirs = np.zeros((n_per, 3))
        dirs[np.arange(n_per)[k < 3], k[k < 3]] = 1
        edge = rng.normal(-1.0, 2.0, n_per)
        two = rng.integers(0, 2, n_per)
        eta = a[i] + b[i] * edge + 1.5 * two + dirs @ np.array(list(GAMMA.values()))
        y = (rng.random(n_per) < expit(eta)).astype(int)
        rows.append(pd.DataFrame({"batter_id": i, "y": y, "edge": edge, "abs_ball": (edge > 0).astype(int),
                                  "two_strike": two, "three_ball": 0, "tokens2": 1, "late": 0, "close": 0,
                                  **{c: dirs[:, j].astype(int) for j, c in enumerate(DIR_COLS)},
                                  **{f"{c}_x_edge": dirs[:, j] * edge for j, c in enumerate(DIR_COLS)}}))
    return pd.concat(rows, ignore_index=True), a, b, w_lows


def test_direction_intercepts_are_recovered_and_removed_from_the_criterion():
    d, a, b, w_low = _simulate_direction()
    R, prior = fit_hierarchical(d, min_decisions=50)
    R = R.sort_values("batter_id")
    for c, g in GAMMA.items():
        assert abs(prior["fixed"][c] - g) < 0.2, (c, prior["fixed"][c], g)
    # the slopes do not differ by direction, so the test must drop the interaction terms
    assert prior["direction"] and not prior["direction_slopes"]
    assert not prior["direction_test"]["keep_slopes"]
    assert prior["direction_test"]["p_intercepts"] < 1e-6
    # the direction-blind criterion absorbs the hitter's stimulus mix; the direction-aware one does not
    Rb, _ = fit_hierarchical(d, min_decisions=50, direction=False)
    Rb = Rb.sort_values("batter_id")
    mix_blind = np.corrcoef(Rb.intercept - a, w_low)[0, 1]
    mix_aware = np.corrcoef(R.intercept - a, w_low)[0, 1]
    assert mix_blind > 0.4 and abs(mix_aware) < 0.2, (mix_blind, mix_aware)
    assert np.corrcoef(R.intercept, a)[0, 1] > np.corrcoef(Rb.intercept, a)[0, 1]
    assert np.corrcoef(R.slope, b)[0, 1] > 0.5


def _fixture_pitches(pk: str) -> pd.DataFrame:
    with gzip.open(FIX / f"{pk}.json.gz", "rt") as f:
        p, _, _ = extract_game(json.load(f))
    pt = pd.DataFrame(p)
    traj = ["x0", "y0", "z0", "vx0", "vy0", "vz0", "ax", "ay", "az"]
    pt = pt[pt[traj + ["sz_top", "sz_bot"]].notna().all(axis=1)].copy()
    pt["mid_x"], pt["mid_z"] = midplate_location_np(*(pt[c].to_numpy(float) for c in traj))
    pt["abs_edge_in"] = edge_distance_in_np(pt.mid_x.to_numpy(), pt.mid_z.to_numpy(), pt.sz_top.to_numpy(float), pt.sz_bot.to_numpy(float))
    pt["batter_challenged"] = pt.challenged & (pt.challenge_team_id == pt.bat_team)
    pt["orig_called_strike"] = ((pt.call_code == "C") & ~pt.challenged) | pt.batter_challenged
    return pt


@pytest.mark.parametrize("pk", ["822724", "823452"])
def test_direction_indicators_on_fixture_games(pk):
    pt = _fixture_pitches(pk)
    d = build_decision_table(pt, edge_window=(-np.inf, np.inf))
    assert len(d) > 20 and set(d.direction) <= {"above", "below", "side", "corner"}
    # one indicator per non-reference direction, none for a pitch off the side
    ind = d[DIR_COLS].to_numpy()
    assert set(np.unique(ind)) <= {0, 1} and (ind.sum(axis=1) <= 1).all()
    for c in DIR_COLS:
        assert ((d[c] == 1) == (d.direction == c.removeprefix("dir_"))).all()
    assert ((ind.sum(axis=1) == 0) == (d.direction == "side")).all()
    for c in DIR_COLS:
        assert np.allclose(d[f"{c}_x_edge"], d[c] * d.edge)
    assert (d.direction.to_numpy() == miss_direction(d.mid_x, d.mid_z, d.sz_top, d.sz_bot)).all()
    # the default window keeps the ball side out to 6 inches
    assert build_decision_table(pt).edge.between(-4, 6).all()

    # the rule on every pitch of the game, recomputed by hand in inches, against the zone geometry
    d = pt.assign(direction=miss_direction(pt.mid_x, pt.mid_z, pt.sz_top, pt.sz_bot))
    assert set(d.direction) == {"above", "below", "side", "corner"}
    dx = d.mid_x.abs() * 12 - 8.5
    da = (d.mid_z - d.sz_top) * 12
    db = (d.sz_bot - d.mid_z) * 12
    corner = (dx > 0) & ((da > 0) | (db > 0))
    assert ((d.direction == "corner") == corner).all()
    nc = ~corner
    big = np.select([(da >= dx) & (da >= db), (db >= dx) & (db > da)], ["above", "below"], "side")
    assert (d.direction[nc] == big[nc.to_numpy()]).all()
    # agreement with the zone geometry: off a corner the centre distance is the hypotenuse,
    # elsewhere it is the largest single-axis excess (negative inside the box)
    centre = d.abs_edge_in + BALL_RADIUS_IN
    assert np.allclose(centre[nc], np.maximum(np.maximum(dx, da), db)[nc], atol=1e-6)
    assert np.allclose(centre[corner], np.hypot(dx, np.maximum(da, db))[corner], atol=1e-6)
    # a high strike's centre is in the top half of the box, a low one's in the bottom half
    mid = (d.sz_top + d.sz_bot) / 2
    assert (d.mid_z[d.direction == "above"] > mid[d.direction == "above"]).all()
    assert (d.mid_z[d.direction == "below"] < mid[d.direction == "below"]).all()


def test_miss_direction_cases():
    top, bot = 3.4, 1.6
    x = np.array([0.0, 0.0, 1.0, -1.0, 0.0, 0.95])
    z = np.array([3.6, 1.3, 2.5, 2.5, 2.5, 3.7])
    assert list(miss_direction(x, z, top, bot)) == ["above", "below", "side", "side", "side", "corner"]


def test_reliability_interval_and_label_threshold():
    rng = np.random.default_rng(0)
    t = rng.normal(size=200)
    s = split_half_reliability(t + rng.normal(size=200), t + rng.normal(size=200), n_boot=500)
    assert s["rel_lo"] < s["rel"] < s["rel_hi"] and s["boot_lo"] < s["rel"] < s["boot_hi"]
    rel = pd.DataFrame({"min_challenges_per_half": [0, 5, 10], "slope_lo": [0.1, 0.4, 0.55]})
    assert eyes_label_threshold(rel) == 20
    assert eyes_label_threshold(rel.assign(slope_lo=[0.1, 0.2, 0.3])) is None
