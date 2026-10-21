"""Cross-check per-player challenge attribution against Savant's leaderboard CSV.

Usage: python scripts/08_validate_vs_savant.py [--label mlb_2026]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from abs_judgment.config import PROCESSED, RAW_SAVANT  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="mlb_2026")
    a = ap.parse_args()
    ch = pd.read_parquet(PROCESSED / f"challenges_{a.label}.parquet")
    season = a.label.split("_")[1]
    import unicodedata

    def key(s: str) -> str:
        s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
        return s.lower().replace(".", "").replace(" jr", "").strip()

    ch["key"] = ch.initiator_name.map(key)
    for role, ctype in (("batter", "batter"), ("catcher", "catcher"), ("pitcher", "pitcher")):
        sv = pd.read_csv(RAW_SAVANT / f"abs_leaderboard_mlb_{season}_{ctype}_R.csv", encoding="utf-8-sig")
        sv["key"] = sv.entity_name.map(key)
        id_col = "key"
        mine = (ch[ch.initiator_role == role].groupby("key")
                .agg(n_mine=("is_overturned", "size"), ot_mine=("is_overturned", "sum")).reset_index())
        m = sv[[id_col, "n_challenges", "n_overturns"]].merge(mine, on=id_col, how="outer")
        m[["n_challenges", "n_overturns", "n_mine", "ot_mine"]] = m[["n_challenges", "n_overturns", "n_mine", "ot_mine"]].fillna(0)
        exact = (m.n_challenges == m.n_mine).mean()
        within1 = ((m.n_challenges - m.n_mine).abs() <= 1).mean()
        print(f"{role:8s}: players {len(m)}; savant total {int(m.n_challenges.sum())} vs mine {int(m.n_mine.sum())}; "
              f"overturns {int(m.n_overturns.sum())} vs {int(m.ot_mine.sum())}; exact per-player match {exact:.3f}, within 1: {within1:.3f}; "
              f"corr {np.corrcoef(m.n_challenges, m.n_mine)[0, 1]:.4f}")
        off = m[(m.n_challenges - m.n_mine).abs() > 1]
        if len(off):
            print("   largest discrepancies:", off.assign(d=(off.n_challenges - off.n_mine)).nlargest(5, "d", keep="all")[[id_col, "n_challenges", "n_mine"]].to_dict("records"))


if __name__ == "__main__":
    main()
