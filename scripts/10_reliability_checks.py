"""How reliable is the per-hitter judgment slope within one season, and with what uncertainty?

Split-half reliability (first half of the season against the second,
Spearman-Brown to a full season) of the slope (eyes) and the intercept
(criterion), by model, edge window and the minimum challenges per hitter in
each half. Every figure carries a 95% interval: Fisher-z on the split-half
correlation carried through the step-up (primary) and a hitter bootstrap of
the pairs (2,000 resamples). Each refit is a vectorised empirical-Bayes fit,
so the whole table takes seconds.

Usage: python scripts/10_reliability_checks.py [--label mlb_2026]
Output: data/processed/reliability_checks_<label>.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from abs_judgment.config import PROCESSED  # noqa: E402
from abs_judgment.judgment import (  # noqa: E402
    EDGE_WINDOW_IN, build_decision_table, eyes_label_threshold, fit_hierarchical, split_half_reliability,
)

MIN_CHAL = (0, 3, 5, 8, 10, 12)
PRIMARY = "direction-aware"
# name -> (decision-table kwargs, fit kwargs)
MODELS = {
    "before (direction-blind, diagonal prior)": ({"token_col": "bat_tokens_before"}, {"direction": False, "full_cov": False}),
    "direction-blind, full covariance": ({}, {"direction": False}),
    PRIMARY: ({}, {"direction_slopes": False}),
    "direction-aware + direction x edge slopes": ({}, {"direction_slopes": True}),
}
RUNS = [  # (model, window)
    ("before (direction-blind, diagonal prior)", (-4.0, 4.0)),
    ("before (direction-blind, diagonal prior)", (-2.5, 2.5)),
    ("before (direction-blind, diagonal prior)", (-6.0, 6.0)),
    ("direction-blind, full covariance", EDGE_WINDOW_IN),
    (PRIMARY, EDGE_WINDOW_IN),
    (PRIMARY, (-4.0, 4.0)),
    (PRIMARY, (-4.0, 8.0)),
    (PRIMARY, (-2.5, 2.5)),
    ("direction-aware + direction x edge slopes", EDGE_WINDOW_IN),
]
COLS = ["orig_called_strike", "bat_tokens_before", "bat_tokens_before_ruleB", "abs_edge_in", "batter_challenged",
        "strikes_before", "balls_before", "inning", "bat_score", "fld_score", "batter_id", "game_date",
        "mid_x", "mid_z", "sz_top", "sz_bot"]


def split_half(d: pd.DataFrame, fit_kw: dict, min_dec: int = 30) -> list[dict]:
    mid = d.game_date.sort_values().iloc[len(d) // 2]
    R1, _ = fit_hierarchical(d[d.game_date < mid], min_decisions=min_dec, **fit_kw)
    R2, _ = fit_hierarchical(d[d.game_date >= mid], min_decisions=min_dec, **fit_kw)
    m = R1.merge(R2, on="batter_id", suffixes=("_1", "_2"))
    rows = []
    for k in MIN_CHAL:
        mm = m[(m.n_challenges_1 >= k) & (m.n_challenges_2 >= k)]
        row = {"min_challenges_per_half": k, "n": len(mm)}
        for c in ("slope", "intercept"):
            s = split_half_reliability(mm[f"{c}_1"], mm[f"{c}_2"], n_boot=2000, seed=k)
            row.update({c: s["rel"], f"{c}_lo": s["rel_lo"], f"{c}_hi": s["rel_hi"],
                        f"{c}_boot_lo": s["boot_lo"], f"{c}_boot_hi": s["boot_hi"], f"r_{c}": s["r"]})
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="mlb_2026")
    a = ap.parse_args()
    pt = pd.read_parquet(PROCESSED / f"pitch_table_{a.label}.parquet", columns=COLS)
    rows = []
    for model, window in RUNS:
        dt_kw, fit_kw = MODELS[model]
        d = build_decision_table(pt, edge_window=window, **dt_kw)
        for r in split_half(d, fit_kw):
            rows.append({"model": model, "window_in": f"{window[0]:g}..{window[1]:g}",
                         "primary": model == PRIMARY and window == EDGE_WINDOW_IN, **r})
            print(f"{model:44s} {rows[-1]['window_in']:>9s}  min {r['min_challenges_per_half']:2d}/half  n {r['n']:3d}  "
                  f"slope {r['slope']:.2f} [{r['slope_lo']:.2f}, {r['slope_hi']:.2f}]  "
                  f"criterion {r['intercept']:.2f} [{r['intercept_lo']:.2f}, {r['intercept_hi']:.2f}]", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(PROCESSED / f"reliability_checks_{a.label}.csv", index=False)
    for model, window in ((PRIMARY, EDGE_WINDOW_IN), ("before (direction-blind, diagonal prior)", (-4.0, 4.0))):
        sel = out[(out.model == model) & (out.window_in == f"{window[0]:g}..{window[1]:g}")]
        t = eyes_label_threshold(sel)
        print(f"\n{model}, {window[0]:g}..{window[1]:g}: eyes labels supported (slope reliability lower end >= 0.5) "
              + (f"from {t} challenges a season" if t else "at no tested volume"))


if __name__ == "__main__":
    main()
