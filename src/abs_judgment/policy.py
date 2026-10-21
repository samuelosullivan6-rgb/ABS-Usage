"""Dynamic ABS challenge policy: state-dependent token value by backward induction.

Savant's static breakeven rule is: challenge if p >= 0.2 / (0.2 + s), where
p is the probability the call is overturned and s is the run stake of the
flip. That prices a token at a constant 0.2 runs. The dynamic version
prices the token at V(k, t) - V(k-1, t): the expected future value of
holding k tokens rather than k-1 with t decision slots left in the game.
Tokens are worthless after the last pitch, so the threshold falls toward
"challenge anything with positive expected value" late in close games
and rises early when the token still has many chances to be spent well.

Game model. Each half-inning contains M decision slots. In each slot an
opportunity arrives with probability q = lambda / M, where lambda is the
observed rate of challengeable pitches per half-inning (both sides: a
team's tokens cover its hitters' challenges of called strikes and its
battery's challenges of called balls). An opportunity has (p, s) drawn
from the empirical joint distribution. The team may challenge: with
probability p it gains s runs and keeps the token; otherwise it loses the
token. Bellman:

  V(k, t) = (1-q) V(k, t+1) + q E_{p,s}[ max( V(k, t+1),
                                             p (s + V(k, t+1)) + (1-p) V(k-1, t+1) ) ]
  V(0, t) = 0 (plus the extra-innings top-up if that rule is confirmed)
  V(k, T) = 0

Threshold: challenge iff p >= dV / (s + dV), dV = V(k,t+1) - V(k-1,t+1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit

HALF_INNINGS = 18
SLOTS_PER_HALF = 8


# --------------------------------------------------------------------------- #
# Stakes: run value of flipping a call, from Statcast's own delta_run_exp
# --------------------------------------------------------------------------- #
def stake_table(pt: pd.DataFrame) -> pd.DataFrame:
    """E[delta_run_exp | called ball, state] - E[delta_run_exp | called strike, state],
    by (balls, strikes, outs, base_state), with coarser fallbacks."""
    d = pt[pt.is_take & pt.delta_run_exp.notna() & pt.description.isin(["ball", "called_strike", "blocked_ball"])].copy()
    d["is_ball_call"] = d.description.isin(["ball", "blocked_ball"])
    keys = ["balls_before", "strikes_before", "outs_before", "base_state"]
    g = d.groupby(keys + ["is_ball_call"]).delta_run_exp.agg(["mean", "size"]).unstack("is_ball_call")
    full = pd.DataFrame({
        "stake": g[("mean", True)] - g[("mean", False)],
        "n": g[("size", True)].fillna(0) + g[("size", False)].fillna(0),
    }).reset_index()
    g2 = d.groupby(["balls_before", "strikes_before", "outs_before", "is_ball_call"]).delta_run_exp.mean().unstack()
    coarse = (g2[True] - g2[False]).rename("stake_coarse").reset_index()
    full = full.merge(coarse, on=["balls_before", "strikes_before", "outs_before"], how="left")
    full["stake"] = np.where(full.n >= 200, full.stake, full.stake_coarse)
    return full[keys + ["stake", "n"]]


def attach_stakes(df: pd.DataFrame, st: pd.DataFrame) -> pd.Series:
    keys = ["balls_before", "strikes_before", "outs_before", "base_state"]
    m = df[keys].merge(st, on=keys, how="left")
    return m.stake.to_numpy()


# --------------------------------------------------------------------------- #
# Overturn probability as a function of true location (smoothed step)
# --------------------------------------------------------------------------- #
def fit_overturn_curve(ch_pitches: pd.DataFrame) -> tuple[float, float]:
    """Logit of overturn on signed edge distance from the challenger's view.
    For a batter challenge the call flips if the ball is outside (edge > 0);
    for a fielder challenge it flips if inside (edge <= 0). Returns (a, b)
    with P(overturn) = expit(a + b * e_signed)."""
    e = np.where(ch_pitches.batter_challenged, ch_pitches.abs_edge_in, -ch_pitches.abs_edge_in)
    y = ch_pitches.challenge_overturned.astype(float).to_numpy()
    from scipy.optimize import minimize

    def nll(ab):
        p = expit(ab[0] + ab[1] * e)
        return -(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)).sum()

    r = minimize(nll, np.array([0.0, 2.0]), method="Nelder-Mead")
    return float(r.x[0]), float(r.x[1])


def perceived_overturn_prob(e_signed: np.ndarray, sigma_in: float) -> np.ndarray:
    """A challenger with perceptual noise sigma (inches) perceives e + N(0, sigma).
    Their calibrated subjective P(overturn) given the perceived edge is the
    posterior P(true e > 0 | perceived), which for a flat prior is Phi(e_perc / sigma).
    Here we return P(overturn) as a function of the TRUE edge under that noise:
    the expected subjective probability, used for the opportunity distribution."""
    from scipy.stats import norm

    return norm.cdf(e_signed / max(sigma_in, 1e-6))


# --------------------------------------------------------------------------- #
# Backward induction
# --------------------------------------------------------------------------- #
def token_value_dp(opps: pd.DataFrame, lam_per_half: float, k_max: int = 2,
                   n_half: int = HALF_INNINGS, slots: int = SLOTS_PER_HALF,
                   extras_topup: bool = False, p_col: str = "p", s_col: str = "stake"):
    """opps: DataFrame of opportunities with columns p (overturn prob) and stake (runs).
    Returns V[k, t] and threshold[k, t] arrays over t = 0..T slots."""
    T = n_half * slots
    q = min(lam_per_half / slots, 1.0)
    P = opps[p_col].to_numpy(float)
    S = opps[s_col].to_numpy(float)
    V = np.zeros((k_max + 1, T + 1))

    for t in range(T - 1, -1, -1):
        for k in range(1, k_max + 1):
            v_keep = V[k, t + 1]
            v_lose = V[k - 1, t + 1]
            dV = v_keep - v_lose
            gain = P * (S + v_keep) + (1 - P) * v_lose
            best = np.maximum(v_keep, gain)
            V[k, t] = (1 - q) * v_keep + q * best.mean()
        if extras_topup and t % slots == 0 and t // slots >= 18:
            V[0, t] = V[1, t]  # a team at zero gets one token at the start of an extra inning
    # scalar threshold summary at the median stake
    s_med = float(np.median(S))
    thr_med = np.full((k_max + 1, T + 1), np.nan)
    for t in range(T):
        for k in range(1, k_max + 1):
            dV = V[k, t + 1] - V[k - 1, t + 1]
            thr_med[k, t] = dV / (s_med + dV)
    return V, thr_med, q


def token_value_table(V: np.ndarray, slots: int = SLOTS_PER_HALF) -> pd.DataFrame:
    """Marginal token value dV = V(k) - V(k-1) at the start of each half-inning."""
    rows = []
    for h in range(V.shape[1] // slots):
        t = h * slots
        for k in range(1, V.shape[0]):
            rows.append({"half_inning": h + 1, "inning": h // 2 + 1, "half": "top" if h % 2 == 0 else "bottom",
                         "tokens": k, "V": V[k, t], "token_value": V[k, t] - V[k - 1, t]})
    return pd.DataFrame(rows)


def savant_static_threshold(stake: np.ndarray, token_runs: float = 0.2) -> np.ndarray:
    return token_runs / (token_runs + stake)


def replay(challenge_rows: pd.DataFrame, V: np.ndarray, slots: int = SLOTS_PER_HALF) -> pd.DataFrame:
    """For each observed challenge, compare the decision under the static and
    dynamic rules and the realized run value. Requires columns: p, stake,
    tokens_before, inning, half, challenge_overturned."""
    c = challenge_rows.copy()
    h = (c.inning - 1) * 2 + (c.half == "bottom").astype(int)
    t = (h * slots).clip(upper=V.shape[1] - 2).to_numpy()
    k = c.tokens_before.clip(1, V.shape[0] - 1).to_numpy()
    dV = V[k, t + 1] - V[k - 1, t + 1]
    c["token_value_dyn"] = dV
    c["thr_dynamic"] = dV / (c.stake + dV)
    c["thr_static"] = savant_static_threshold(c.stake.to_numpy())
    c["challenge_dynamic"] = c.p >= c.thr_dynamic
    c["challenge_static"] = c.p >= c.thr_static
    c["realized_runs"] = np.where(c.challenge_overturned.astype(bool), c.stake, 0.0)
    c["ev_static_price"] = c.p * c.stake - (1 - c.p) * 0.2
    c["ev_dynamic_price"] = c.p * c.stake - (1 - c.p) * dV
    return c
