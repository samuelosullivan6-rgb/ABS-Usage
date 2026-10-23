"""Fit the challenge-judgment model and write per-hitter scores.

Usage: python scripts/05_judgment.py [--label mlb_2026] [--min-decisions 60]
Outputs: data/processed/judgment_<label>.csv, judgment_prior_<label>.json,
         judgment_curve_<label>.csv (league psychometric curve by edge bin),
         judgment_curve_direction_<label>.csv (the same, by the side of the box the pitch missed on)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from abs_judgment.config import PROCESSED  # noqa: E402
from abs_judgment.judgment import (  # noqa: E402
    DIR_COLS, DIR_SLOPE_COLS, EDGE_WINDOW_IN, build_decision_table, fit_hierarchical, sdt_metrics, split_half_reliability,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="mlb_2026")
    ap.add_argument("--min-decisions", type=int, default=60)
    a = ap.parse_args()

    pt = pd.read_parquet(PROCESSED / f"pitch_table_{a.label}.parquet")
    d = build_decision_table(pt)
    lo, hi = EDGE_WINDOW_IN
    print(f"decision set ({lo:+.0f} to {hi:+.0f} in): {len(d):,} called strikes with a token in hand, {d.y.sum():,} challenged "
          f"({d.y.mean():.3%}); true balls among them {d.abs_ball.mean():.3%}")
    print(f"  of which beyond +4 in: {(d.edge > 4).sum():,} pitches, {d.y[d.edge > 4].sum()} challenged")

    # league psychometric curve, pooled and by direction
    bins = np.arange(lo, hi + 0.5, 0.5)
    d["edge_bin"] = pd.cut(d.edge, bins, include_lowest=True)
    curve = d.groupby("edge_bin", observed=True).agg(n=("y", "size"), challenge_rate=("y", "mean"),
                                                      true_ball=("abs_ball", "mean")).reset_index()
    curve["edge_mid"] = curve.edge_bin.apply(lambda iv: iv.mid)
    curve.to_csv(PROCESSED / f"judgment_curve_{a.label}.csv", index=False)
    print("\nleague challenge rate by true edge distance (in, + = outside zone):")
    print(curve[["edge_mid", "n", "challenge_rate"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    dbins = pd.cut(d.edge, [lo, -2, 0, 1, 2, 4, hi], include_lowest=True)
    dc = d.groupby([dbins, "direction"], observed=True).agg(n=("y", "size"), challenge_rate=("y", "mean")).reset_index()
    ch = d[d.y == 1]
    ot = ch.groupby("direction").challenge_overturned.mean() if "challenge_overturned" in ch.columns else None
    dc.to_csv(PROCESSED / f"judgment_curve_direction_{a.label}.csv", index=False)
    print("\nchallenge rate by true edge distance and the side of the box the pitch missed on:")
    print(dc.pivot(index="edge", columns="direction", values="challenge_rate").to_string(float_format=lambda x: f"{x:.3f}"))
    if ot is not None:
        print("overturn rate of batter challenges by direction:", ot.round(3).to_dict())

    R, prior = fit_hierarchical(d, min_decisions=a.min_decisions)
    lr = prior["direction_test"]
    print(f"\ndirection, likelihood-ratio tests against the direction-blind pooled logit (n = {lr['n']:,}):")
    print(f"  intercept shifts: chi2 {lr['chi2_intercepts']:.1f} on {lr['df_intercepts']} df (p = {lr['p_intercepts']:.1e}); "
          + ", ".join(f"{k} {v:+.3f}" for k, v in lr["intercepts"].items()))
    print(f"  x edge slopes on top: chi2 {lr['chi2_slopes']:.1f} on {lr['df_slopes']} df (p = {lr['p_slopes']:.1e}; "
          f"BIC price {lr['bic_penalty_slopes']:.1f}); " + ", ".join(f"{k} {v:+.3f}" for k, v in lr["slopes"].items())
          + f" -> {'kept' if lr['keep_slopes'] else 'dropped'}")
    S = sdt_metrics(d, min_decisions=a.min_decisions)
    names = pt.groupby("batter_id").batter_name.first()
    J = R.merge(S, on="batter_id", how="left")
    J.insert(1, "batter_name", J.batter_id.map(names))
    J = J.sort_values("judgment_z", ascending=False)
    J.to_csv(PROCESSED / f"judgment_{a.label}.csv", index=False)

    print(f"\nhitters scored: {len(J)} (>= {a.min_decisions} decisions)")
    print(f"league intercept {prior['mean'][0]:.3f}, slope {prior['mean'][1]:.3f} per inch; "
          f"between-hitter SD intercept {np.sqrt(prior['var'][0]):.3f}, slope {np.sqrt(prior['var'][1]):.3f}; "
          f"prior intercept-slope correlation {prior['corr']:+.3f}, correlation of posterior means {prior['corr_posterior_means']:+.3f}")
    print(f"per-hitter posterior slope-intercept correlation: median {R.post_corr.median():+.3f} "
          f"(10th-90th pct {R.post_corr.quantile(0.1):+.3f} to {R.post_corr.quantile(0.9):+.3f}); "
          f"from the likelihood alone: median {R.lik_corr.median():+.3f} ({R.lik_corr.quantile(0.1):+.3f} to {R.lik_corr.quantile(0.9):+.3f})")
    print("fixed effects:", {k: round(float(v), 3) for k, v in prior["fixed"].items()})
    big = J[(J.n_decisions >= 150) & (J.n_challenges >= 8)]  # the slope is unidentified for non-challengers
    cols = ["batter_name", "n_decisions", "n_challenges", "n_opps", "slope", "slope_sd", "judgment_z", "judgment_z_sd", "edge_at_50", "d_prime"]
    print("\nsharpest perception (top 12):")
    print(big.head(12)[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nshallowest perception (bottom 12):")
    print(big.tail(12)[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # split-half reliability (first vs second half of the season), with Fisher-z intervals
    half_min = max(30, a.min_decisions // 2)
    mid = d.game_date.sort_values().iloc[len(d) // 2]
    R1, _ = fit_hierarchical(d[d.game_date < mid], min_decisions=half_min)
    R2, _ = fit_hierarchical(d[d.game_date >= mid], min_decisions=half_min)
    m12 = R1.merge(R2, on="batter_id", suffixes=("_1", "_2"))
    S1 = sdt_metrics(d[d.game_date < mid], min_decisions=half_min)
    S2 = sdt_metrics(d[d.game_date >= mid], min_decisions=half_min)
    sd = S1.merge(S2, on="batter_id", suffixes=("_1", "_2"))
    rel_full = {"judgment_z": split_half_reliability(m12.judgment_z_1, m12.judgment_z_2),
                "criterion_z": split_half_reliability(m12.criterion_z_1, m12.criterion_z_2),
                "d_prime": split_half_reliability(sd.d_prime_1, sd.d_prime_2)}
    # average fixed-effect contribution, so a marginal league curve can be drawn at a + offset + b * edge
    fixed_offset = float(sum(d[k].mean() * v for k, v in prior["fixed"].items() if k not in DIR_SLOPE_COLS))
    slope_offset = float(sum(d[k.replace("_x_edge", "")].mean() * v for k, v in prior["fixed"].items() if k in DIR_SLOPE_COLS))
    with open(PROCESSED / f"judgment_prior_{a.label}.json", "w") as f:
        json.dump({"mean": prior["mean"].tolist(), "var": prior["var"].tolist(), "cov": prior["cov"].tolist(),
                   "corr": prior["corr"], "corr_posterior_means": prior["corr_posterior_means"],
                   "post_corr_median": float(R.post_corr.median()), "lik_corr_median": float(R.lik_corr.median()),
                   "fixed": prior["fixed"], "fixed_offset_mean": fixed_offset, "fixed_slope_offset_mean": slope_offset,
                   "direction_reference": "side", "direction_cols": DIR_COLS, "direction_test": lr,
                   "edge_window": list(EDGE_WINDOW_IN), "n_decisions": int(len(d)), "n_challenged": int(d.y.sum()),
                   "n_hitters": prior["n_hitters"],
                   "reliability_full_season": {k: v["rel"] for k, v in rel_full.items()},
                   "reliability_interval": {k: {kk: v[kk] for kk in ("n", "r", "rel", "rel_lo", "rel_hi")} for k, v in rel_full.items()}},
                  f, indent=1)
    print("\nsplit-half reliability (Spearman-Brown to full season, 95% Fisher-z interval): " + "; ".join(
        f"{k} {v['rel']:.3f} [{v['rel_lo']:.3f}, {v['rel_hi']:.3f}] (n = {v['n']})" for k, v in rel_full.items()))
    print(f"corr(judgment_z, d_prime) = {J[['judgment_z', 'd_prime']].corr().iloc[0, 1]:.3f}; "
          f"corr(judgment_z, criterion_z) = {J[['judgment_z', 'criterion_z']].corr().iloc[0, 1]:.3f}")


if __name__ == "__main__":
    main()
