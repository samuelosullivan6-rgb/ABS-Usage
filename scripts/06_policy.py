"""Dynamic token valuation, policy simulation, and per-hitter challenge thresholds.

Usage: python scripts/06_policy.py [--label mlb_2026] [--sigma <inches>] [--no-extras-topup]
Outputs (data/processed/): token_value_<label>.csv, policy_simulation_<label>.csv,
         policy_hitter_thresholds_<label>.csv, stakes_<label>.csv

Perception. The ABS ruling is a deterministic function of the true edge
distance (verified at 99.99%). What the challenger sees is the true edge
plus noise with SD sigma. The judgment model's league logit slope b maps
to sigma ~= 1.7 / b (logit-to-probit), and each hitter's slope gives a
hitter-specific sigma. The subjective probability of an overturn given a
perceived edge e' is Phi(e' / sigma). Every policy below decides on the
perceived edge and is scored on the true outcome.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from abs_judgment.config import PROCESSED  # noqa: E402
from abs_judgment.policy import (  # noqa: E402
    attach_stakes, dynamic_threshold, simulate_policy, stake_table, token_value_dp, token_value_table,
)

NEAR_EDGE_IN = 3.0
LOGIT_TO_PROBIT = 1.7


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="mlb_2026")
    ap.add_argument("--sigma", type=float, default=None, help="override perceptual noise (inches)")
    ap.add_argument("--no-extras-topup", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    topup = not a.no_extras_topup

    pt = pd.read_parquet(PROCESSED / f"pitch_table_{a.label}.parquet")
    pt = pt[pt.description.notna()]  # rows with Statcast run values
    n_games = pt.game_pk.nunique()

    # ---- perceptual noise from the judgment model ----------------------------
    prior = json.load(open(PROCESSED / f"judgment_prior_{a.label}.json"))
    slope_league = prior["mean"][1]
    sigma = a.sigma if a.sigma else LOGIT_TO_PROBIT / slope_league
    print(f"league judgment slope {slope_league:.3f} per inch -> perceptual noise sigma = {sigma:.2f} in"
          + ("" if a.sigma is None else " (override)"))

    st = stake_table(pt)
    st.to_csv(PROCESSED / f"stakes_{a.label}.csv", index=False)
    print(f"stake table: {len(st)} states; median |stake| {st.stake.abs().median():.3f} runs, max {st.stake.abs().max():.3f}")

    # ---- opportunity set: near-edge taken pitches, both sides ---------------
    opp = pt[pt.is_take & pt.abs_edge_in.notna() & (pt.abs_edge_in.abs() <= NEAR_EDGE_IN)
             & (pt.orig_called_strike | pt.orig_called_ball)].copy()
    opp["side"] = np.where(opp.orig_called_strike, "bat", "fld")
    opp["team"] = np.where(opp.side == "bat", opp.bat_team, opp.fld_team)
    opp["e_signed"] = np.where(opp.side == "bat", opp.abs_edge_in, -opp.abs_edge_in)
    opp["true_flip"] = opp.e_signed > 0
    rng = np.random.default_rng(a.seed)
    opp["e_perceived"] = opp.e_signed + rng.normal(0, sigma, len(opp))
    opp["p"] = norm.cdf(opp.e_perceived / sigma)
    opp["stake"] = np.abs(attach_stakes(opp, st))
    opp = opp[opp.stake.notna()].sort_values(["game_pk", "at_bat_number", "pitch_number"])
    lam = len(opp) / (n_games * 36)
    print(f"opportunities (taken, |edge| <= {NEAR_EDGE_IN} in): {len(opp):,} over {n_games} games = {lam:.3f} per team-half-inning; "
          f"umpire wrong on {opp.true_flip.mean():.3f}; observed challenge rate on them {opp.challenged.mean():.3f}")

    # ---- backward induction ---------------------------------------------------
    V, thr, q = token_value_dp(opp[["p", "stake"]], lam_per_half=lam, extras_topup=topup)
    tv = token_value_table(V)
    tv.to_csv(PROCESSED / f"token_value_{a.label}.csv", index=False)
    print("\nmarginal token value (runs), start of inning (top half):")
    print(tv[tv.half == "top"].pivot(index="inning", columns="tokens", values="token_value").to_string(float_format=lambda x: f"{x:.3f}"))
    print("Savant's static price: 0.200 runs for every token at every moment.")

    # ---- simulate rules on the same opportunities ----------------------------
    rows = [simulate_policy(opp, r, V, extras_topup=topup) for r in ("never", "static", "dynamic", "oracle")]
    # observed behavior: realized runs from actual challenges on these pitches
    obs = opp[opp.challenged]
    rows.append({"rule": "observed", "runs": float(obs.loc[obs.true_flip, "stake"].sum()), "challenges": int(len(obs)),
                 "overturned": int(obs.true_flip.sum()), "runs_per_team_162": obs.loc[obs.true_flip, "stake"].sum() / (2 * n_games) * 162,
                 "challenges_per_team_game": len(obs) / (2 * n_games), "overturn_rate": obs.true_flip.mean()})
    sim = pd.DataFrame(rows)
    sim.to_csv(PROCESSED / f"policy_simulation_{a.label}.csv", index=False)
    print(f"\npolicy simulation at league-average eyes (sigma {sigma:.2f} in), scored on true ABS outcomes:")
    print(sim[["rule", "challenges_per_team_game", "overturn_rate", "runs_per_team_162"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("  'oracle' challenges exactly the umpire's misses: the ceiling with perfect eyes.")
    print("  'observed' is what players actually did on the same pitches (their own eyes, their own rule).")

    # ---- per-hitter thresholds in perceived inches ---------------------------
    J = pd.read_csv(PROCESSED / f"judgment_{a.label}.csv")
    J = J[J.n_challenges >= 8]
    s_med = float(opp.stake.median())
    out = []
    for r in J.itertuples(index=False):
        sig_i = LOGIT_TO_PROBIT / max(r.slope, 0.05)
        for inning in (1, 5, 9):
            for k in (1, 2):
                t = ((inning - 1) * 2) * 8
                p_star = dynamic_threshold(V, k, t, s_med)
                out.append({"batter_id": r.batter_id, "batter_name": r.batter_name, "sigma_in": sig_i,
                            "inning": inning, "tokens": k, "p_star": p_star,
                            "perceived_margin_in": sig_i * norm.ppf(p_star),
                            "static_margin_in": sig_i * norm.ppf(0.2 / (0.2 + s_med))})
    H = pd.DataFrame(out)
    H.to_csv(PROCESSED / f"policy_hitter_thresholds_{a.label}.csv", index=False)
    show = H[(H.tokens == 1)].pivot_table(index="batter_name", columns="inning", values="perceived_margin_in")
    sharp = J.nlargest(3, "slope").batter_name.tolist() + J.nsmallest(3, "slope").batter_name.tolist()
    print(f"\nperceived margin off the plate (inches) at which to spend the last token, median stake {s_med:.2f} runs:")
    print(show.loc[[n for n in sharp if n in show.index]].to_string(float_format=lambda x: f"{x:+.2f}"))
    print("  (negative = challenge even a pitch that looks slightly inside; sharper eyes need less margin)")


if __name__ == "__main__":
    main()
