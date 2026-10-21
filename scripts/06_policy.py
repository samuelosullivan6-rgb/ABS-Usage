"""Dynamic token valuation and replay of observed challenges.

Usage: python scripts/06_policy.py [--label mlb_2026] [--sigma 1.0] [--extras-topup]
Outputs: data/processed/token_value_<label>.csv, policy_replay_<label>.csv, stakes_<label>.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from abs_judgment.config import PROCESSED  # noqa: E402
from abs_judgment.policy import (  # noqa: E402
    attach_stakes, fit_overturn_curve, replay, savant_static_threshold, stake_table, token_value_dp, token_value_table,
)

NEAR_EDGE_IN = 3.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="mlb_2026")
    ap.add_argument("--sigma", type=float, default=1.0, help="perceptual noise in inches for the subjective overturn probability")
    ap.add_argument("--extras-topup", action="store_true")
    a = ap.parse_args()

    pt = pd.read_parquet(PROCESSED / f"pitch_table_{a.label}.parquet")
    pt = pt[pt.description.notna()]  # Statcast-joined rows only (need delta_run_exp)
    n_games = pt.game_pk.nunique()

    st = stake_table(pt)
    st.to_csv(PROCESSED / f"stakes_{a.label}.csv", index=False)
    print(f"stake table: {len(st)} count/out/base states; median stake {st.stake.abs().median():.3f} runs, "
          f"max {st.stake.abs().max():.3f} (3-2 with runners)")

    chal = pt[pt.challenged & pt.abs_edge_in.notna()]
    ab = fit_overturn_curve(chal)
    print(f"overturn curve: P(overturn) = expit({ab[0]:.2f} + {ab[1]:.2f} * signed edge); "
          f"at edge +0.5 in {expit(ab[0] + ab[1] * 0.5):.3f}, at -0.5 in {expit(ab[0] - ab[1] * 0.5):.3f}")

    # ---- opportunity set: near-edge taken pitches, both sides ---------------
    taken = pt[pt.is_take & pt.abs_edge_in.notna() & (pt.abs_edge_in.abs() <= NEAR_EDGE_IN)
               & (pt.orig_called_strike | pt.orig_called_ball)].copy()
    taken["side"] = np.where(taken.orig_called_strike, "bat", "fld")
    taken["e_signed"] = np.where(taken.side == "bat", taken.abs_edge_in, -taken.abs_edge_in)
    taken["p_true"] = expit(ab[0] + ab[1] * taken.e_signed)
    taken["p"] = norm.cdf(taken.e_signed / a.sigma)              # subjective, league-average eyes
    taken["stake"] = np.abs(attach_stakes(taken, st))
    taken = taken[taken.stake.notna()]
    lam = len(taken) / (n_games * 36)
    print(f"opportunities (|edge| <= {NEAR_EDGE_IN} in, taken): {len(taken):,} over {n_games} games "
          f"= {lam:.3f} per team-half-inning; true-flip share {(taken.p_true > 0.5).mean():.3f}")

    V, thr, q = token_value_dp(taken[["p", "stake"]], lam_per_half=lam, extras_topup=a.extras_topup)
    tv = token_value_table(V)
    tv.to_csv(PROCESSED / f"token_value_{a.label}.csv", index=False)
    print("\nmarginal token value (runs) at the start of each inning, top half:")
    show = tv[(tv.half == "top")].pivot(index="inning", columns="tokens", values="token_value")
    print(show.to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"Savant's static price is 0.20 runs. Dynamic price of the 2nd token in the 1st inning: "
          f"{tv[(tv.half_inning == 1) & (tv.tokens == 2)].token_value.iloc[0]:.3f}; of the last token in the 9th: "
          f"{tv[(tv.half_inning == 18) & (tv.tokens == 1)].token_value.iloc[0]:.3f}")

    # ---- replay observed challenges -----------------------------------------
    ch = chal.copy()
    ch["side"] = np.where(ch.batter_challenged, "bat", "fld")
    ch["e_signed"] = np.where(ch.side == "bat", ch.abs_edge_in, -ch.abs_edge_in)
    ch["p"] = norm.cdf(ch.e_signed / a.sigma)
    ch["stake"] = np.abs(attach_stakes(ch, st))
    ch["tokens_before"] = np.where(ch.side == "bat", ch.bat_tokens_before, ch.fld_tokens_before)
    ch = ch[ch.stake.notna()]
    rp = replay(ch, V)
    rp[["game_pk", "at_bat_number", "pitch_number", "inning", "half", "side", "challenge_player_name", "abs_edge_in",
        "p", "stake", "tokens_before", "token_value_dyn", "thr_static", "thr_dynamic", "challenge_static",
        "challenge_dynamic", "challenge_overturned", "realized_runs", "ev_static_price", "ev_dynamic_price"]
      ].to_csv(PROCESSED / f"policy_replay_{a.label}.csv", index=False)
    print(f"\nobserved challenges replayed: {len(rp):,}; overturned {rp.challenge_overturned.astype(bool).mean():.3f}")
    print(f"  endorsed by static rule: {rp.challenge_static.mean():.3f}; by dynamic rule: {rp.challenge_dynamic.mean():.3f}")
    print(f"  mean EV per challenge, static price {rp.ev_static_price.mean():+.4f}; dynamic price {rp.ev_dynamic_price.mean():+.4f} runs")
    print(f"  realized runs per challenge {rp.realized_runs.mean():.4f}; total {rp.realized_runs.sum():.1f}")
    late = rp[rp.inning >= 8]
    print(f"  8th inning or later (n={len(late)}): static endorses {late.challenge_static.mean():.3f}, dynamic {late.challenge_dynamic.mean():.3f}")

    # ---- runs not taken under dynamic pricing ----------------------------------
    nt = taken[~taken.challenged].copy()
    nt["tokens_before"] = np.where(nt.side == "bat", nt.bat_tokens_before, nt.fld_tokens_before)
    nt = nt[nt.tokens_before >= 1]
    nt = replay(nt.assign(challenge_overturned=nt.p_true > 0.5), V)
    missed_dyn = nt[nt.challenge_dynamic]
    missed_sta = nt[nt.challenge_static]
    print(f"\nuntaken opportunities with a token in hand: {len(nt):,}")
    print(f"  dynamic rule would challenge {len(missed_dyn):,} (true flips {missed_dyn.challenge_overturned.mean():.3f}, "
          f"realized {missed_dyn.realized_runs.sum():.1f} runs league-wide, {missed_dyn.realized_runs.sum() / 30:.1f} per team)")
    print(f"  static rule would challenge {len(missed_sta):,} (true flips {missed_sta.challenge_overturned.mean():.3f}, "
          f"realized {missed_sta.realized_runs.sum():.1f} runs league-wide)")
    print("  note: these use a league-average eye (sigma); a hitter-specific sigma from the judgment model changes p per player")


if __name__ == "__main__":
    main()
