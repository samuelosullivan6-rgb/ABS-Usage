"""The eyes-versus-approach test.

Usage: python scripts/07_eyes_vs_approach.py [--label mlb_2026] [--prev-season 2025]
Outputs: data/processed/hitter_season_<label>.csv, eyes_vs_approach_<label>.csv,
         eyes_vs_approach_strata_<label>.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from abs_judgment.chase import hitter_season, rulebook_chase_from_statcast  # noqa: E402
from abs_judgment.config import PROCESSED, RAW_STATCAST  # noqa: E402
from abs_judgment.models import OUTCOMES, assemble, run_all  # noqa: E402
from abs_judgment.savant import load_statcast_days  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="mlb_2026")
    ap.add_argument("--prev-season", type=int, default=2025)
    ap.add_argument("--min-decisions", type=int, default=60)
    ap.add_argument("--min-pa", type=int, default=150)
    ap.add_argument("--min-challenges", type=int, default=8,
                    help="slope reliability rises steeply with challenges: ~0.3 at 8+, ~0.8 at 20+ per season")
    a = ap.parse_args()

    pt = pd.read_parquet(PROCESSED / f"pitch_table_{a.label}.parquet")
    hs = hitter_season(pt, min_pitches=500)
    hs.to_csv(PROCESSED / f"hitter_season_{a.label}.csv", index=False)
    J = pd.read_csv(PROCESSED / f"judgment_{a.label}.csv")

    prev = None
    days = sorted(RAW_STATCAST.glob(f"{a.prev_season}-*.csv"))
    if days:
        sc = load_statcast_days(days)
        prev = rulebook_chase_from_statcast(sc, min_pitches=500)
        print(f"previous season {a.prev_season}: {len(days)} days, {len(sc):,} pitches, {len(prev)} hitters")

    df = assemble(J, hs, prev, min_decisions=a.min_decisions, min_pa=a.min_pa, min_challenges=a.min_challenges)
    suffix = f"{a.label}" if a.min_challenges == 8 else f"{a.label}_minchal{a.min_challenges}"
    print(f"analysis sample: {len(df)} hitter-seasons (>= {a.min_decisions} decisions, >= {a.min_challenges} challenges, >= {a.min_pa} PA)")
    print(f"corr(judgment_z, chase_rate) = {df[['judgment_z', 'chase_rate']].corr().iloc[0, 1]:+.3f}; "
          f"corr(criterion_z, chase_rate) = {df[['criterion_z', 'chase_rate']].corr().iloc[0, 1]:+.3f}")

    res, strata = run_all(df)
    res["label"] = res.outcome.map(lambda o: OUTCOMES[o][1])
    res["beta_in_sd"] = res.beta_judgment / res.sd_outcome
    import json
    rel = json.load(open(PROCESSED / f"judgment_prior_{a.label}.json")).get("reliability_full_season", {}).get("judgment_z")
    if rel and rel > 0:
        res["beta_in_sd_disattenuated"] = res.beta_in_sd / rel
        print(f"judgment_z full-season reliability {rel:.3f}: disattenuated effects = raw / {rel:.3f}")
    res.to_csv(PROCESSED / f"eyes_vs_approach_{suffix}.csv", index=False)
    pd.concat([s.assign(outcome=o) for o, s in strata.items()]).to_csv(PROCESSED / f"eyes_vs_approach_strata_{suffix}.csv", index=False)

    print("\nEffect of +1 SD judgment (sharper eyes), holding chase-rate quintile, criterion, bat speed and side fixed:")
    cols = ["outcome", "n", "beta_judgment", "se_hc1", "p_hc1", "boot_lo", "boot_hi", "beta_in_sd", "delta_r2"]
    if "beta_in_sd_disattenuated" in res.columns:
        cols.append("beta_in_sd_disattenuated")
    print(res[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nWithin chase-rate quintile, high-judgment minus low-judgment tercile:")
    for o, s in strata.items():
        print(f"  {o}: " + ", ".join(f"Q{int(r.chase_q) + 1} {r.high_minus_low:+.4f}" for r in s.itertuples()))


if __name__ == "__main__":
    main()
