"""How reliable can the per-hitter judgment slope be made within one season?

Varies the edge window and the minimum challenges per hitter and reports
split-half reliability (Spearman-Brown to full season) of the slope and
the criterion. Usage: python scripts/10_reliability_checks.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from abs_judgment.config import PROCESSED  # noqa: E402
from abs_judgment.judgment import build_decision_table, fit_hierarchical  # noqa: E402

LABEL = "mlb_2026"


def split_half(d: pd.DataFrame, min_dec: int, min_chal: int) -> dict:
    mid = d.game_date.sort_values().iloc[len(d) // 2]
    R1, _ = fit_hierarchical(d[d.game_date < mid], min_decisions=min_dec)
    R2, _ = fit_hierarchical(d[d.game_date >= mid], min_decisions=min_dec)
    m = R1.merge(R2, on="batter_id", suffixes=("_1", "_2"))
    m = m[(m.n_challenges_1 >= min_chal) & (m.n_challenges_2 >= min_chal)]
    out = {"n": len(m)}
    for c in ("slope", "intercept"):
        r = m[[f"{c}_1", f"{c}_2"]].corr().iloc[0, 1]
        out[c] = 2 * r / (1 + r)
    return out


if __name__ == "__main__":
    pt = pd.read_parquet(PROCESSED / f"pitch_table_{LABEL}.parquet",
                         columns=["orig_called_strike", "bat_tokens_before", "abs_edge_in", "batter_challenged",
                                  "strikes_before", "balls_before", "inning", "bat_score", "fld_score", "batter_id", "game_date"])
    rows = []
    for window in ((-4, 4), (-2.5, 2.5), (-6, 6)):
        d = build_decision_table(pt, edge_window=window)
        for min_chal in (0, 5, 10):
            r = split_half(d, min_dec=30, min_chal=min_chal)
            rows.append({"window_in": f"{window[0]}..{window[1]}", "min_challenges_per_half": min_chal, **r})
            print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(PROCESSED / f"reliability_checks_{LABEL}.csv", index=False)
    print(out.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
