"""Dynamic ABS challenge policy: state-dependent token value by backward induction.

Savant's static breakeven rule is: challenge if p >= 0.2 / (0.2 + s), where
p is the probability the call is overturned and s is the run stake of the
flip. That prices a token at a constant 0.2 runs. The dynamic version
prices the token at V(k, t) - V(k-1, t): the expected future value of
holding k tokens rather than k-1 with t decision slots left in the game.

Game model. Each half-inning contains M decision slots. In each slot an
opportunity arrives with probability q = lambda / M, where lambda is the
observed rate of challengeable pitches per team-half-inning (both sides: a
team's tokens cover its hitters' challenges of called strikes and its
battery's challenges of called balls). An opportunity has (p, s) drawn
from the empirical joint distribution. The team may challenge: with
probability p it gains s runs and keeps the token; otherwise it loses the
token. Bellman, over the 18 regulation half-innings (T = 18 M slots):

  V(k, t) = (1-q) V(k, t+1) + q E_{p,s}[ max( V(k, t+1),
                                             p (s + V(k, t+1)) + (1-p) V(k-1, t+1) ) ]   k >= 1
  V(0, t) = V(0, t+1)                       (no token, no decision)
  V(k, T) = pi * X(max(k, 1))               (extras continuation)

Extra innings. With probability pi the game is tied after nine and goes on.
Every extra inning starts with a top-up to one token for a team at zero
(decision 5) and is followed by another with probability rho. X(k) is the
value of entering an extra inning with k tokens after the top-up; it solves
the same Bellman over one inning (2 M slots) with terminal rho * X(max(k, 1)),
by fixed-point iteration. Hence V(0, t) = pi * X(1) for every regulation
slot: a team at zero in the third still gets a token back if the game
reaches the tenth. With pi = 0 the model is the plain nine-inning game and
V(0, :) = 0.

Threshold: challenge iff p >= dV / (s + dV), dV = V(k,t+1) - V(k-1,t+1).

Beliefs. p is the challenger's subjective probability of an overturn given
the edge they perceive. Two models: the flat-prior posterior
Phi(e_perceived / sigma), which ignores that the umpire is right on most
near-edge calls, and the calibrated posterior P(true flip | e_perceived),
a logit fit on the opportunity set itself (decision 17).
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


def fit_logit_1d(x: np.ndarray, y: np.ndarray, iters: int = 50) -> tuple[float, float]:
    """Maximum-likelihood logit of y on (1, x) by Newton's method. Returns (a, b)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    X = np.column_stack([np.ones_like(x), x])
    beta = np.array([np.log((y.mean() + 1e-9) / (1 - y.mean() + 1e-9)), 0.0])
    for _ in range(iters):
        mu = expit(X @ beta)
        w = mu * (1 - mu)
        step = np.linalg.solve((X * w[:, None]).T @ X, X.T @ (y - mu))
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    return float(beta[0]), float(beta[1])


def calibrated_belief(e_signed: np.ndarray, sigma_in: float, rng: np.random.Generator,
                      true_flip: np.ndarray | None = None) -> tuple[float, float]:
    """(a, b) of the calibrated posterior P(true flip | perceived edge) = expit(a + b e').

    A challenger with noise sigma who knows how often the umpire is wrong near
    the edge holds this belief. Fit on the opportunity set with one noise draw;
    the flip is e_signed > 0 unless given."""
    e_signed = np.asarray(e_signed, float)
    y = (e_signed > 0) if true_flip is None else np.asarray(true_flip, bool)
    e_perc = e_signed + rng.normal(0, sigma_in, len(e_signed))
    return fit_logit_1d(e_perc, y)


def subjective_p(e_perceived: np.ndarray, sigma_in: float, model: str, ab: tuple[float, float] | None = None) -> np.ndarray:
    """Subjective P(overturn) from a perceived edge under the 'flat' or 'calibrated' belief model."""
    if model == "flat":
        from scipy.stats import norm

        return norm.cdf(np.asarray(e_perceived, float) / max(sigma_in, 1e-6))
    a, b = ab
    return expit(a + b * np.asarray(e_perceived, float))


def perceived_margin(p_star, sigma_in, model: str, ab: tuple[float, float] | None = None):
    """Perceived edge (inches, + = outside) at which the subjective p equals p_star.
    Inverse of subjective_p. sigma_in and ab may be arrays (per-hitter)."""
    from scipy.stats import norm

    p_star = np.clip(np.asarray(p_star, float), 1e-6, 1 - 1e-6)
    if model == "flat":
        return np.asarray(sigma_in, float) * norm.ppf(p_star)
    a, b = ab
    return (np.log(p_star / (1 - p_star)) - np.asarray(a, float)) / np.asarray(b, float)


def belief_grid(e_signed: np.ndarray, sigmas, seed: int = 0) -> pd.DataFrame:
    """Calibrated (a, b) on a grid of sigma, for per-hitter margins by interpolation."""
    rows = []
    for sg in sigmas:
        a, b = calibrated_belief(e_signed, float(sg), np.random.default_rng(seed))
        rows.append({"sigma": float(sg), "a": a, "b": b})
    return pd.DataFrame(rows)


def interp_belief(grid: pd.DataFrame, sigma_in) -> tuple[np.ndarray, np.ndarray]:
    """Calibrated (a, b) at sigma_in by linear interpolation on the grid (clamped at its ends)."""
    sg = np.asarray(sigma_in, float)
    return (np.interp(sg, grid.sigma, grid.a), np.interp(sg, grid.sigma, grid.b))


# --------------------------------------------------------------------------- #
# Backward induction
# --------------------------------------------------------------------------- #
def _bellman_step(V_next: np.ndarray, q: float, P: np.ndarray, S: np.ndarray) -> np.ndarray:
    """One slot of the Bellman recursion: values now given values at the next slot."""
    out = np.empty_like(V_next)
    out[0] = V_next[0]
    for k in range(1, len(V_next)):
        v_keep, v_lose = V_next[k], V_next[k - 1]
        best = np.maximum(v_keep, P * (S + v_keep) + (1 - P) * v_lose)
        out[k] = (1 - q) * v_keep + q * best.mean()
    return out


def extras_value(opps: pd.DataFrame, lam_per_half: float, rho: float, k_max: int = 2,
                 slots: int = SLOTS_PER_HALF, p_col: str = "p", s_col: str = "stake",
                 tol: float = 1e-10, max_iter: int = 500) -> np.ndarray:
    """Value over one extra inning, conditional on being in it: Vx[k, s] for s = 0..2*slots.

    Vx[:, 2*slots] = rho * X(max(k, 1)) with X(k) = Vx[k, 0] (the next inning, after
    the top-up), solved by fixed-point iteration (a contraction since rho < 1)."""
    q = min(lam_per_half / slots, 1.0)
    P = opps[p_col].to_numpy(float)
    S = opps[s_col].to_numpy(float)
    E = 2 * slots
    Vx = np.zeros((k_max + 1, E + 1))
    for _ in range(max_iter):
        X = Vx[:, 0].copy()
        term = rho * X[np.maximum(np.arange(k_max + 1), 1)]
        Vx[:, E] = term
        for s in range(E - 1, -1, -1):
            Vx[:, s] = _bellman_step(Vx[:, s + 1], q, P, S)
        if np.abs(Vx[:, 0] - X).max() < tol:
            break
    return Vx


def token_value_dp(opps: pd.DataFrame, lam_per_half: float, k_max: int = 2,
                   n_half: int = HALF_INNINGS, slots: int = SLOTS_PER_HALF,
                   p_extras: float = 0.0, rho: float = 0.0, p_col: str = "p", s_col: str = "stake"):
    """opps: DataFrame of opportunities with columns p (overturn prob) and stake (runs).

    Returns V[k, t] over the regulation slots t = 0..T, a threshold summary at the
    median stake, and the arrival probability q. p_extras is the share of games that
    reach the 10th and rho the chance an extra inning is followed by another; with
    p_extras = 0 (the default) V[:, T] = 0 and V[0, :] = 0. Otherwise
    V[k, T] = p_extras * X(max(k, 1)) and V[0, :] = p_extras * X(1); see the module doc."""
    T = n_half * slots
    q = min(lam_per_half / slots, 1.0)
    P = opps[p_col].to_numpy(float)
    S = opps[s_col].to_numpy(float)
    V = np.zeros((k_max + 1, T + 1))
    if p_extras > 0:
        X = extras_value(opps, lam_per_half, rho, k_max, slots, p_col, s_col)[:, 0]
        V[:, T] = p_extras * X[np.maximum(np.arange(k_max + 1), 1)]
    for t in range(T - 1, -1, -1):
        V[:, t] = _bellman_step(V[:, t + 1], q, P, S)
    # scalar threshold summary at the median stake
    s_med = float(np.median(S))
    thr_med = np.full((k_max + 1, T + 1), np.nan)
    for t in range(T):
        for k in range(1, k_max + 1):
            dV = V[k, t + 1] - V[k - 1, t + 1]
            thr_med[k, t] = dV / (s_med + dV)
    return V, thr_med, q


def token_value_table(V: np.ndarray, slots: int = SLOTS_PER_HALF, Vx: np.ndarray | None = None) -> pd.DataFrame:
    """Marginal token value dV = V(k) - V(k-1) at the start of each half-inning.
    With Vx, adds the two halves of an extra inning (half_inning 19 and 20, inning 10),
    conditional on the game being in extras."""
    rows = []
    for h in range(V.shape[1] // slots):
        t = h * slots
        for k in range(1, V.shape[0]):
            rows.append({"half_inning": h + 1, "inning": h // 2 + 1, "half": "top" if h % 2 == 0 else "bottom",
                         "tokens": k, "V": V[k, t], "token_value": V[k, t] - V[k - 1, t]})
    if Vx is not None:
        for j in range(2):
            t = j * slots
            for k in range(1, Vx.shape[0]):
                rows.append({"half_inning": V.shape[1] // slots + 1 + j, "inning": 10, "half": "top" if j == 0 else "bottom",
                             "tokens": k, "V": Vx[k, t], "token_value": Vx[k, t] - Vx[k - 1, t]})
    return pd.DataFrame(rows)


def savant_static_threshold(stake: np.ndarray, token_runs: float = 0.2) -> np.ndarray:
    return token_runs / (token_runs + stake)


def dynamic_dv(V: np.ndarray, k, t, Vx: np.ndarray | None = None, slots: int = SLOTS_PER_HALF) -> np.ndarray:
    """Marginal token value at the next slot, dV = V(k, t+1) - V(k-1, t+1), vectorized over
    k and t. k is clamped to 1..k_max. Slots past regulation (t >= T) use the extra-inning
    table Vx when given, cycling every inning; without it they use the last regulation slot."""
    k = np.clip(np.asarray(k), 1, V.shape[0] - 1)
    t = np.asarray(t)
    T = V.shape[1] - 1
    tr = np.minimum(t, T - 1)
    dv = V[k, tr + 1] - V[k - 1, tr + 1]
    if Vx is not None:
        s = (np.maximum(t, T) - T) % (2 * slots)
        dv = np.where(t >= T, Vx[k, s + 1] - Vx[k - 1, s + 1], dv)
    return dv


def dynamic_threshold(V: np.ndarray, k: int, t: int, stake: float, Vx: np.ndarray | None = None,
                      slots: int = SLOTS_PER_HALF) -> float:
    """Dynamic breakeven p at token count k and slot t (see dynamic_dv)."""
    dV = float(dynamic_dv(V, k, t, Vx, slots))
    return dV / (stake + dV) if stake + dV > 0 else 1.0


def simulate_policy(opps: pd.DataFrame, rule: str, V: np.ndarray | None = None, slots: int = SLOTS_PER_HALF,
                    token_runs: float = 0.2, extras_topup: bool = True, Vx: np.ndarray | None = None) -> dict:
    """Play every game forward with a challenge rule, tracking tokens.

    opps must be sorted by game and pitch order (each game's rows contiguous) and
    carry: game_pk, home_id, away_id, team (the side that could challenge), inning,
    half, p (subjective overturn probability from the perceived edge), true_flip
    (ABS would flip), stake (runs). rule in {"static", "dynamic", "oracle"}. The
    token rule is the DP's: two tokens, kept on a win, and a top-up to one at the
    start of every extra inning for a team at zero (extras_topup). The dynamic
    rule prices the token at the start of the half-inning, from Vx in extras.

    Returns realized runs, challenges, overturns, per-team-game rates, and
    per_game: a DataFrame of runs, challenges and overturns by game_pk (both
    teams together), for paired standard errors between rules.
    """
    gk = opps.game_pk.to_numpy()
    home = opps.home_id.to_numpy().astype(np.int64)
    team = opps.team.to_numpy().astype(np.int64)
    inning = opps.inning.to_numpy().astype(np.int64)
    flip = opps.true_flip.to_numpy().astype(bool)
    stake = opps.stake.to_numpy(float)
    p = opps.p.to_numpy(float) if rule != "oracle" else None
    if rule == "static":
        thr = np.broadcast_to(token_runs / (token_runs + stake), (3, len(opps)))
    elif rule == "dynamic":
        t = ((inning - 1) * 2 + (opps.half.to_numpy() == "bottom")) * slots
        thr = np.ones((V.shape[0], len(opps)))
        for k in range(1, V.shape[0]):
            dv = dynamic_dv(V, k, t, Vx, slots)
            den = stake + dv
            thr[k] = np.where(den > 0, dv / np.where(den > 0, den, 1.0), 1.0)
    elif rule != "oracle":
        raise ValueError(rule)
    k_cap = V.shape[0] - 1 if rule == "dynamic" else 2

    starts = np.flatnonzero(np.r_[True, gk[1:] != gk[:-1]])
    ends = np.r_[starts[1:], len(gk)]
    g_runs = np.zeros(len(starts))
    g_chal = np.zeros(len(starts), dtype=np.int64)
    g_flip = np.zeros(len(starts), dtype=np.int64)
    for gi, (a0, a1) in enumerate(zip(starts, ends)):
        tok_home, tok_away = 2, 2
        last_inning = 0
        runs, n_chal, n_flip = 0.0, 0, 0
        h_id = home[a0]
        for i in range(a0, a1):
            inn = inning[i]
            if extras_topup and inn >= 10 and inn != last_inning:
                last_inning = inn
                tok_home = tok_home or 1
                tok_away = tok_away or 1
            is_home = team[i] == h_id
            k = tok_home if is_home else tok_away
            if k == 0:
                continue
            if rule == "oracle":
                go = flip[i]
            else:
                go = p[i] >= thr[min(k, k_cap), i]
            if not go:
                continue
            n_chal += 1
            if flip[i]:
                runs += stake[i]
                n_flip += 1
            elif is_home:
                tok_home -= 1
            else:
                tok_away -= 1
        g_runs[gi], g_chal[gi], g_flip[gi] = runs, n_chal, n_flip
    per_game = pd.DataFrame({"game_pk": gk[starts], "runs": g_runs, "challenges": g_chal, "overturned": g_flip})
    return summarize_rule(rule, per_game)


def summarize_rule(rule: str, per_game: pd.DataFrame) -> dict:
    """Season totals and per-team-game rates from a per-game table (two teams per game)."""
    n_games = len(per_game)
    runs, n_chal, n_flip = float(per_game.runs.sum()), int(per_game.challenges.sum()), int(per_game.overturned.sum())
    return {"rule": rule, "runs": runs, "challenges": n_chal, "overturned": n_flip,
            "runs_per_team_162": runs / (2 * n_games) * 162,
            "challenges_per_team_game": n_chal / (2 * n_games),
            "overturn_rate": n_flip / n_chal if n_chal else np.nan,
            "per_game": per_game}


def paired_difference(a: pd.DataFrame, b: pd.DataFrame) -> tuple[float, float]:
    """Mean and standard error of (a - b) in runs per team per 162 games, paired by game.
    Each per-game row covers both teams, so per team-game is half of it."""
    m = a[["game_pk", "runs"]].merge(b[["game_pk", "runs"]], on="game_pk", suffixes=("_a", "_b"), how="outer").fillna(0.0)
    d = (m.runs_a - m.runs_b).to_numpy() / 2 * 162
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


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
