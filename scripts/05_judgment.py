"""Fit the challenge-judgment model and write per-hitter scores.

Usage: python scripts/05_judgment.py [--label mlb_2026] [--min-decisions 60]
Outputs: data/processed/judgment_<label>.csv, judgment_prior_<label>.json,
         judgment_curve_<label>.csv (league psychometric curve by edge bin)
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
from abs_judgment.judgment import build_decision_table, fit_hierarchical, sdt_metrics  # noqa: E402
from abs_judgment.models import judgment_stability  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="mlb_2026")
    ap.add_argument("--min-decisions", type=int, default=60)
    a = ap.parse_args()

    pt = pd.read_parquet(PROCESSED / f"pitch_table_{a.label}.parquet")
    d = build_decision_table(pt)
    print(f"decision set: {len(d):,} called strikes with a token in hand, {d.y.sum():,} challenged "
          f"({d.y.mean():.3%}); true balls among them {d.abs_ball.mean():.3%}")

    # league psychometric curve
    bins = np.arange(-4, 4.5, 0.5)
    d["edge_bin"] = pd.cut(d.edge, bins)
    curve = d.groupby("edge_bin", observed=True).agg(n=("y", "size"), challenge_rate=("y", "mean"),
                                                      true_ball=("abs_ball", "mean")).reset_index()
    curve["edge_mid"] = curve.edge_bin.apply(lambda iv: iv.mid)
    curve.to_csv(PROCESSED / f"judgment_curve_{a.label}.csv", index=False)
    print("\nleague challenge rate by true edge distance (in, + = outside zone):")
    print(curve[["edge_mid", "n", "challenge_rate"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    R, prior = fit_hierarchical(d, min_decisions=a.min_decisions)
    S = sdt_metrics(d, min_decisions=a.min_decisions)
    names = pt.groupby("batter_id").batter_name.first()
    J = R.merge(S, on="batter_id", how="left")
    J.insert(1, "batter_name", J.batter_id.map(names))
    J = J.sort_values("judgment_z", ascending=False)
    J.to_csv(PROCESSED / f"judgment_{a.label}.csv", index=False)
    with open(PROCESSED / f"judgment_prior_{a.label}.json", "w") as f:
        json.dump({"mean": prior["mean"].tolist(), "var": prior["var"].tolist(),
                   "fixed": prior["fixed"], "n_hitters": prior["n_hitters"]}, f, indent=1)

    print(f"\nhitters scored: {len(J)} (>= {a.min_decisions} decisions)")
    print(f"league intercept {prior['mean'][0]:.3f}, slope {prior['mean'][1]:.3f} per inch; "
          f"between-hitter SD intercept {np.sqrt(prior['var'][0]):.3f}, slope {np.sqrt(prior['var'][1]):.3f}")
    print("fixed effects (criterion shifts):", {k: round(v, 3) for k, v in prior["fixed"].items()})
    big = J[(J.n_decisions >= 150) & (J.n_challenges >= 8)]  # the slope is unidentified for non-challengers
    cols = ["batter_name", "n_decisions", "n_challenges", "n_opps", "slope", "slope_sd", "judgment_z", "edge_at_50", "hit_rate", "fa_rate", "d_prime"]
    print("\nsharpest perception (top 12):")
    print(big.head(12)[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nshallowest perception (bottom 12):")
    print(big.tail(12)[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # split-half reliability (first vs second half of the season)
    mid = d.game_date.sort_values().iloc[len(d) // 2]
    R1, _ = fit_hierarchical(d[d.game_date < mid], min_decisions=max(30, a.min_decisions // 2))
    R2, _ = fit_hierarchical(d[d.game_date >= mid], min_decisions=max(30, a.min_decisions // 2))
    st = judgment_stability(R1, R2, min_decisions=max(30, a.min_decisions // 2))
    S1 = sdt_metrics(d[d.game_date < mid], min_decisions=max(30, a.min_decisions // 2))
    S2 = sdt_metrics(d[d.game_date >= mid], min_decisions=max(30, a.min_decisions // 2))
    sd = S1.merge(S2, on="batter_id", suffixes=("_1", "_2"))
    r_dprime = sd[["d_prime_1", "d_prime_2"]].corr().iloc[0, 1]
    m12 = R1.merge(R2, on="batter_id", suffixes=("_1", "_2"))
    r_crit = m12[["criterion_z_1", "criterion_z_2"]].corr().iloc[0, 1]
    rel = {"judgment_z": st["spearman_brown_full"], "d_prime": 2 * r_dprime / (1 + r_dprime), "criterion_z": 2 * r_crit / (1 + r_crit)}
    # average fixed-effect contribution, so a marginal league curve can be drawn at a + offset + b * edge
    fixed_offset = float(sum(d[k].mean() * v for k, v in prior["fixed"].items()))
    with open(PROCESSED / f"judgment_prior_{a.label}.json", "w") as f:
        json.dump({"mean": prior["mean"].tolist(), "var": prior["var"].tolist(),
                   "fixed": prior["fixed"], "fixed_offset_mean": fixed_offset,
                   "n_hitters": prior["n_hitters"], "reliability_full_season": rel}, f, indent=1)
    print(f"\nsplit-half reliability (Spearman-Brown to full season): judgment_z {rel['judgment_z']:.3f} "
          f"(raw r={st['r']:.3f}, n={st['n']}), d_prime {rel['d_prime']:.3f}, criterion_z {rel['criterion_z']:.3f}")
    print(f"corr(judgment_z, d_prime) = {J[['judgment_z', 'd_prime']].corr().iloc[0, 1]:.3f}; "
          f"corr(judgment_z, criterion_z) = {J[['judgment_z', 'criterion_z']].corr().iloc[0, 1]:.3f}")


if __name__ == "__main__":
    main()
