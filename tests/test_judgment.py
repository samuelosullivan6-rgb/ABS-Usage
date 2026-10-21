"""Synthetic recovery: hitters with sharper perception (smaller sigma) must
get larger estimated slopes, and hitters with a looser criterion must get
larger intercepts, after empirical-Bayes shrinkage."""
import numpy as np
import pandas as pd
from scipy.stats import norm

from abs_judgment.judgment import fit_hierarchical


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
