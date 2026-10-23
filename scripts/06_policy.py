"""Dynamic token valuation, policy simulation, and per-hitter challenge thresholds.

Usage: python scripts/06_policy.py [--label mlb_2026] [--sigma <inches>] [--seed 0] [--seeds 0,1,2]
                                   [--flat-prior] [--no-extras] [--skip-sensitivity]
Outputs (data/processed/):
  token_value_<label>.csv              marginal token value by half-inning (regulation)
  token_value_extras_<label>.csv       the same inside an extra inning, conditional on reaching it
  policy_simulation_<label>.csv        static / dynamic / oracle / observed, with paired SEs
  policy_hitter_thresholds_<label>.csv per-hitter perceived margins
  policy_belief_<label>.json           belief model, sigma, calibrated logit and its sigma grid, extras rates
  policy_seeds_<label>.csv             both belief models at every seed
  policy_sigma_<label>.csv             sigma sensitivity under both belief models
  stakes_<label>.csv
With --flat-prior the first five are written with a _flatprior suffix and the
main files are left alone; that reproduces the pre-2026-10-23 belief model.

Perception. The ABS ruling is a deterministic function of the true edge
distance (verified at 99.99%). What the challenger sees is the true edge
plus noise with SD sigma. The judgment model's league logit slope b maps
to sigma ~= 1.7 / b (logit-to-probit), and each hitter's slope gives a
hitter-specific sigma.

Belief. The subjective probability of an overturn given a perceived edge e'
is, by default, the calibrated posterior P(true flip | e') = expit(a + b e'),
a logit of the true flip on the perceived edge fit on the opportunity set
with the same noise draw (decision 17). A challenger holding it knows the
umpire is right on about 80% of these pitches. The flat-prior posterior
Phi(e' / sigma) used before ignores that, and a simulated "90% sure" under it
flips about half the time. Every policy decides on the perceived edge and is
scored on the true outcome.
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
from abs_judgment.policy import (  # noqa: E402
    attach_stakes, belief_grid, dynamic_threshold, extras_value, fit_logit_1d, interp_belief,
    paired_difference, perceived_margin, simulate_policy, stake_table, subjective_p, summarize_rule,
    token_value_dp, token_value_table,
)

NEAR_EDGE_IN = 3.0
LOGIT_TO_PROBIT = 1.7
SIGMA_GRID = (1.0, 1.5, 2.0, None, 3.5, 4.5)  # None = the league sigma from the judgment model
BELIEF_GRID_SIGMAS = np.round(np.arange(0.5, 12.01, 0.25), 2)
THRESHOLD_HALVES = (1, 7, 9, 17)  # start of the 1st, 4th, 5th and 9th (top halves)
BELIEFS = ("calibrated", "flat")


def run_model(opp: pd.DataFrame, belief: str, sigma: float, seed: int, lam: float, p_extras: float, rho: float,
              rules=("static", "dynamic")) -> dict:
    """One belief model at one noise draw: subjective p, DP, extras table and simulated rules."""
    rng = np.random.default_rng(seed)
    e_perc = opp.e_signed.to_numpy() + rng.normal(0, sigma, len(opp))
    ab = fit_logit_1d(e_perc, opp.true_flip.to_numpy()) if belief == "calibrated" else None
    o = opp.assign(e_perceived=e_perc, p=subjective_p(e_perc, sigma, belief, ab))
    V, _, q = token_value_dp(o[["p", "stake"]], lam_per_half=lam, p_extras=p_extras, rho=rho)
    Vx = extras_value(o[["p", "stake"]], lam, rho) if p_extras > 0 else None
    sims = {r: simulate_policy(o, r, V, extras_topup=p_extras > 0, Vx=Vx) for r in rules}
    return {"belief": belief, "sigma": sigma, "seed": seed, "ab": ab, "V": V, "Vx": Vx, "q": q, "opp": o, "sims": sims}


def calibration_table(o: pd.DataFrame) -> pd.DataFrame:
    """Simulated p against the true flip rate, in the review's bins."""
    bins = pd.cut(o.p, [0, 0.2, 0.4, 0.6, 0.8, 1.0], include_lowest=True)
    return o.groupby(bins, observed=True).agg(n=("p", "size"), mean_p=("p", "mean"), flip_rate=("true_flip", "mean"))


def calibration_quintiles(o: pd.DataFrame) -> list[dict]:
    """Mean simulated p and true flip rate in quintiles of p."""
    q = o.groupby(pd.qcut(o.p, 5), observed=True).agg(mean_p=("p", "mean"), flip_rate=("true_flip", "mean"), n=("p", "size"))
    return [{k: (int(v) if k == "n" else round(float(v), 4)) for k, v in r.items()} for r in q.to_dict("records")]


def summary_row(m: dict, obs_pg: pd.DataFrame) -> dict:
    s, d = m["sims"]["static"], m["sims"]["dynamic"]
    ds, ds_se = paired_difference(d["per_game"], s["per_game"])
    do, do_se = paired_difference(d["per_game"], obs_pg)
    V = m["V"]
    return {"belief": m["belief"], "sigma": round(m["sigma"], 3), "seed": m["seed"],
            "token1_first_pitch": V[1, 0] - V[0, 0], "token2_first_pitch": V[2, 0] - V[1, 0],
            "token1_start_9th": V[1, 16 * 8] - V[0, 16 * 8],
            "static_runs": s["runs_per_team_162"], "static_chal": s["challenges_per_team_game"], "static_ot": s["overturn_rate"],
            "dynamic_runs": d["runs_per_team_162"], "dynamic_chal": d["challenges_per_team_game"], "dynamic_ot": d["overturn_rate"],
            "dyn_minus_static": ds, "dyn_minus_static_se": ds_se, "dyn_minus_observed": do, "dyn_minus_observed_se": do_se}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="mlb_2026")
    ap.add_argument("--sigma", type=float, default=None, help="override perceptual noise (inches)")
    ap.add_argument("--seed", type=int, default=0, help="noise draw for the written tables")
    ap.add_argument("--seeds", default="0,1,2", help="noise draws for the seed spread")
    ap.add_argument("--flat-prior", action="store_true", help="write the flat-prior model, to *_flatprior files")
    ap.add_argument("--no-extras", action="store_true", help="nine-inning game: no extras continuation, no top-up")
    ap.add_argument("--skip-sensitivity", action="store_true")
    a = ap.parse_args()
    primary = "flat" if a.flat_prior else "calibrated"
    sfx = "_flatprior" if a.flat_prior else ""
    seeds = [int(x) for x in a.seeds.split(",")]

    pt = pd.read_parquet(PROCESSED / f"pitch_table_{a.label}.parquet")
    pt = pt[pt.description.notna()]  # rows with Statcast run values
    n_games = pt.game_pk.nunique()

    # ---- perceptual noise from the judgment model ----------------------------
    prior = json.load(open(PROCESSED / f"judgment_prior_{a.label}.json"))
    slope_league = prior["mean"][1]
    sigma_league = LOGIT_TO_PROBIT / slope_league
    sigma = a.sigma if a.sigma else sigma_league
    print(f"league judgment slope {slope_league:.3f} per inch -> perceptual noise sigma = {sigma:.2f} in"
          + ("" if a.sigma is None else " (override)") + f"; belief model written: {primary}")

    st = stake_table(pt)
    st.to_csv(PROCESSED / f"stakes_{a.label}.csv", index=False)
    print(f"stake table: {len(st)} states; median |stake| {st.stake.abs().median():.3f} runs, max {st.stake.abs().max():.3f}")

    # ---- extra innings: how often, how long ---------------------------------
    last = pt.groupby("game_pk").inning.max()
    n_extras = int((last >= 10).sum())
    extra_innings = int((last[last >= 10] - 9).sum())
    p_extras = 0.0 if a.no_extras else n_extras / n_games
    rho = 0.0 if a.no_extras else (extra_innings - n_extras) / extra_innings
    print(f"extras: {n_extras} of {n_games} games ({n_extras / n_games:.3f}), {extra_innings} extra innings, "
          f"P(another extra inning) = {(extra_innings - n_extras) / extra_innings:.3f}"
          + (" (ignored: --no-extras)" if a.no_extras else ""))

    # ---- opportunity set: near-edge taken pitches, both sides ---------------
    opp = pt[pt.is_take & pt.abs_edge_in.notna() & (pt.abs_edge_in.abs() <= NEAR_EDGE_IN)
             & (pt.orig_called_strike | pt.orig_called_ball)].copy()
    opp["side"] = np.where(opp.orig_called_strike, "bat", "fld")
    opp["team"] = np.where(opp.side == "bat", opp.bat_team, opp.fld_team)
    opp["e_signed"] = np.where(opp.side == "bat", opp.abs_edge_in, -opp.abs_edge_in)
    opp["true_flip"] = opp.e_signed > 0
    opp["stake"] = np.abs(attach_stakes(opp, st))
    opp = opp[opp.stake.notna()].sort_values(["game_pk", "at_bat_number", "pitch_number"])
    n_half_played = len(pt.drop_duplicates(["game_pk", "inning", "half"]))
    lam = len(opp) / (2 * n_half_played)
    print(f"opportunities (taken, |edge| <= {NEAR_EDGE_IN} in): {len(opp):,} over {n_half_played:,} half-innings played "
          f"= {lam:.3f} per team-half-inning; umpire wrong on {opp.true_flip.mean():.3f}; "
          f"observed challenge rate on them {opp.challenged.mean():.3f}")

    # observed behaviour on the same pitches: real players' eyes and rule
    obs = opp[opp.challenged]
    obs_pg = (obs.assign(r=np.where(obs.true_flip, obs.stake, 0.0), f=obs.true_flip.astype(int))
              .groupby("game_pk").agg(runs=("r", "sum"), challenges=("r", "size"), overturned=("f", "sum"))
              .reindex(opp.game_pk.unique(), fill_value=0).rename_axis("game_pk").reset_index())
    observed = summarize_rule("observed", obs_pg)

    # ---- both belief models at the primary seed --------------------------------
    models = {b: run_model(opp, b, sigma, a.seed, lam, p_extras, rho,
                           rules=("static", "dynamic", "oracle") if b == primary else ("static", "dynamic"))
              for b in BELIEFS}
    M = models[primary]
    print("\ncalibration of the simulated p against the true flip rate (seed %d):" % a.seed)
    for b in BELIEFS:
        print(f"  {b}:")
        print(calibration_table(models[b]["opp"]).to_string(float_format=lambda x: f"{x:.3f}"))
    if M["ab"] is not None:
        print(f"calibrated belief: P(flip | e') = expit({M['ab'][0]:+.3f} {M['ab'][1]:+.3f} e')")

    # ---- token value tables ------------------------------------------------------
    tv = token_value_table(M["V"])
    tv.to_csv(PROCESSED / f"token_value_{a.label}{sfx}.csv", index=False)
    if M["Vx"] is not None:
        tvx = token_value_table(M["V"], Vx=M["Vx"]).query("half_inning > 18")
        tvx.to_csv(PROCESSED / f"token_value_extras_{a.label}{sfx}.csv", index=False)
    print("\nmarginal token value (runs), start of inning (top half), both belief models:")
    both = pd.concat({b: token_value_table(models[b]["V"]).query("half == 'top'")
                      .pivot(index="inning", columns="tokens", values="token_value") for b in BELIEFS}, axis=1)
    print(both.to_string(float_format=lambda x: f"{x:.3f}"))
    if M["Vx"] is not None:
        Vx = M["Vx"]
        print(f"inside an extra inning ({primary}): first token {Vx[1, 0] - Vx[0, 0]:.3f}, second {Vx[2, 0] - Vx[1, 0]:.3f}; "
              f"a team at zero in regulation is worth V(0, t) = {M['V'][0, 0]:.4f} runs from the tenth-inning top-up")
    print("Savant's static price: 0.200 runs for every token at every moment.")

    # ---- simulated rules, with paired standard errors ------------------------
    rows = []
    for r in ("static", "dynamic", "oracle"):
        rows.append(M["sims"][r])
    rows.append(observed)
    sim = pd.DataFrame([{k: v for k, v in r.items() if k != "per_game"} for r in rows])
    for ref, pg in (("static", M["sims"]["static"]["per_game"]), ("observed", obs_pg)):
        diffs = [paired_difference(r["per_game"], pg) for r in rows]
        sim[f"minus_{ref}"] = [d[0] for d in diffs]
        sim[f"minus_{ref}_se"] = [d[1] for d in diffs]
    sim["belief"] = [primary] * 3 + ["real players"]
    sim["sigma"] = [sigma] * 3 + [np.nan]
    sim["seed"] = [a.seed] * 3 + [np.nan]
    sim.to_csv(PROCESSED / f"policy_simulation_{a.label}{sfx}.csv", index=False)

    # ---- seeds: both belief models ----------------------------------------------
    seed_rows = []
    for b in BELIEFS:
        for sd in seeds:
            m = models[b] if sd == a.seed else run_model(opp, b, sigma, sd, lam, p_extras, rho)
            seed_rows.append(summary_row(m, obs_pg))
    S = pd.DataFrame(seed_rows)
    if not a.flat_prior:
        S.to_csv(PROCESSED / f"policy_seeds_{a.label}.csv", index=False)
    print(f"\npolicy simulation (sigma {sigma:.2f} in), scored on true ABS outcomes; runs per team per 162; "
          f"SEs paired by game over {n_games:,} games:")
    for b in BELIEFS:
        print(f"  belief = {b}")
        print(S[S.belief == b][["seed", "token1_first_pitch", "static_runs", "static_chal", "static_ot", "dynamic_runs",
                                "dynamic_chal", "dynamic_ot", "dyn_minus_static", "dyn_minus_static_se",
                                "dyn_minus_observed", "dyn_minus_observed_se"]]
              .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"  oracle (perfect eyes): {M['sims']['oracle']['challenges_per_team_game']:.2f} challenges per team-game, "
          f"{M['sims']['oracle']['runs_per_team_162']:.1f} runs")
    print(f"  observed (real players' own eyes and rule): {observed['challenges_per_team_game']:.2f} challenges per team-game, "
          f"{observed['overturn_rate']:.1%} overturned, {observed['runs_per_team_162']:.1f} runs. The simulated rules share "
          "league-average Gaussian eyes; the gap to observed is not a property of the rule.")

    # ---- sigma sensitivity ------------------------------------------------------
    if not a.skip_sensitivity and not a.flat_prior:
        sens = []
        for sg in SIGMA_GRID:
            sgv = sigma_league if sg is None else sg
            for b in BELIEFS:
                m = models[b] if abs(sgv - sigma) < 1e-9 else run_model(opp, b, sgv, a.seed, lam, p_extras, rho)
                sens.append(summary_row(m, obs_pg) | {"league_sigma": sg is None})
        SS = pd.DataFrame(sens)
        SS.to_csv(PROCESSED / f"policy_sigma_{a.label}.csv", index=False)
        print(f"\nsigma sensitivity (seed {a.seed}):")
        print(SS[["belief", "sigma", "token1_first_pitch", "static_runs", "static_chal", "dynamic_runs", "dynamic_chal",
                  "dyn_minus_static", "dyn_minus_static_se"]].sort_values(["belief", "sigma"])
              .to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # ---- belief model on a sigma grid, for per-hitter margins --------------------
    grid = belief_grid(opp.e_signed.to_numpy(), BELIEF_GRID_SIGMAS, seed=a.seed)
    meta = {"belief": primary, "sigma_league": sigma_league, "sigma": sigma, "seed": a.seed,
            "a": None if M["ab"] is None else M["ab"][0], "b": None if M["ab"] is None else M["ab"][1],
            "lambda_per_team_half": lam, "p_extras": p_extras, "rho_extra_inning": rho,
            "n_games": n_games, "n_extras_games": n_extras, "n_opportunities": len(opp),
            "umpire_wrong_share": float(opp.true_flip.mean()), "n_p_above_0_6": int((M["opp"].p > 0.6).sum()),
            "calibration_quintiles": calibration_quintiles(M["opp"]),
            "grid": grid.round(6).to_dict("list")}
    with open(PROCESSED / f"policy_belief_{a.label}{sfx}.json", "w") as f:
        json.dump(meta, f, indent=1)

    # ---- per-hitter thresholds in perceived inches ---------------------------
    J = pd.read_csv(PROCESSED / f"judgment_{a.label}.csv")
    J = J[J.n_challenges >= 8].copy()
    J["sigma_in"] = LOGIT_TO_PROBIT / J.slope.clip(lower=0.05)
    ab_i = interp_belief(grid, J.sigma_in) if primary == "calibrated" else None
    s_med = float(opp.stake.median())
    p_static = 0.2 / (0.2 + s_med)
    out = []
    for h in THRESHOLD_HALVES:
        for k in (1, 2):
            p_star = dynamic_threshold(M["V"], k, (h - 1) * 8, s_med)
            out.append(pd.DataFrame({
                "batter_id": J.batter_id, "batter_name": J.batter_name, "sigma_in": J.sigma_in,
                "half_inning": h, "inning": (h - 1) // 2 + 1, "tokens": k, "belief": primary, "p_star": p_star,
                "perceived_margin_in": perceived_margin(p_star, J.sigma_in, primary, ab_i),
                "static_margin_in": perceived_margin(p_static, J.sigma_in, primary, ab_i)}))
    H = pd.concat(out, ignore_index=True)
    H.to_csv(PROCESSED / f"policy_hitter_thresholds_{a.label}{sfx}.csv", index=False)
    show = H[(H.tokens == 1)].pivot_table(index="batter_name", columns="inning", values="perceived_margin_in")
    sharp = J.nlargest(3, "slope").batter_name.tolist() + J.nsmallest(3, "slope").batter_name.tolist()
    print(f"\nperceived margin off the plate (inches) at which to spend the last token, median stake {s_med:.2f} runs "
          f"({primary} belief; Savant's rule would need p >= {p_static:.2f}):")
    print(show.loc[[n for n in sharp if n in show.index]].to_string(float_format=lambda x: f"{x:+.2f}"))
    print("  (negative = challenge even a pitch that looks slightly inside; sharper eyes need less margin)")


if __name__ == "__main__":
    main()
