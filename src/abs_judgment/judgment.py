"""Location-adjusted challenge judgment: a hitter's perception, stripped of intent.

Decision set. Every pitch the hitter took that the umpire called a strike
while the batting team still held a token. The hitter either challenged or
did not. The true ABS location (mid-plate edge distance) is known for every
pitch from the trajectory, so this is a psychometric experiment the league
runs for free: stimulus = true edge distance, response = challenge.

Model. A hierarchical probit/logit of challenge on true edge distance with
per-hitter random intercept and random slope:

    P(challenge) = F(a + b * edge + u_i + v_i * edge + controls)

  * slope b + v_i is perceptual SHARPNESS (eyes). A hitter who sees the
    ball well transitions sharply from never-challenging inside the zone
    to challenging outside it. A hitter with poor perception challenges
    some strikes and lets some balls go, so the transition is shallow.
  * intercept a + u_i is the CRITERION (approach / conservatism): how far
    off the plate a pitch must look before the hitter spends a token.

The two are the signal-detection-theory split of sensitivity (d') from
bias (c), estimated on a continuous stimulus. Count, token state and
leverage enter as fixed effects because they shift the criterion, not the
eyes. Pitches far inside the zone carry no information about the
transition and are trimmed to keep the fit on the psychometric region.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm

log = logging.getLogger(__name__)

EDGE_WINDOW_IN = (-4.0, 4.0)


def build_decision_table(pt: pd.DataFrame, edge_window=EDGE_WINDOW_IN) -> pd.DataFrame:
    """Hitter challenge decisions on umpire-called strikes with a token in hand."""
    d = pt[pt.orig_called_strike & (pt.bat_tokens_before >= 1) & pt.abs_edge_in.notna()].copy()
    d["y"] = d.batter_challenged.astype(int)
    d["edge"] = d.abs_edge_in
    d["abs_ball"] = (d.abs_edge_in > 0).astype(int)
    d["two_strike"] = (d.strikes_before == 2).astype(int)
    d["three_ball"] = (d.balls_before == 3).astype(int)
    d["tokens2"] = (d.bat_tokens_before >= 2).astype(int)
    d["late"] = (d.inning >= 7).astype(int)
    if "bat_score" in d.columns and "fld_score" in d.columns:
        d["close"] = ((d.bat_score - d.fld_score).abs() <= 1).astype(int)
    else:
        d["close"] = 0
    lo, hi = edge_window
    d = d[(d.edge >= lo) & (d.edge <= hi)]
    return d


# --------------------------------------------------------------------------- #
# Empirical-Bayes hierarchical logit (two-step, closed form priors)
# --------------------------------------------------------------------------- #
FIXED = ["two_strike", "three_ball", "tokens2", "late", "close"]


def _design(d: pd.DataFrame) -> np.ndarray:
    X = np.column_stack([np.ones(len(d)), d.edge.to_numpy(float)] + [d[c].to_numpy(float) for c in FIXED])
    return X


def _nll(beta, X, y, prior_mean=None, prior_prec=None):
    eta = X @ beta
    p = expit(eta)
    eps = 1e-9
    nll = -(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)).sum()
    if prior_mean is not None:
        dlt = beta[:2] - prior_mean
        nll += 0.5 * dlt @ prior_prec @ dlt
    return nll


def _grad(beta, X, y, prior_mean=None, prior_prec=None):
    p = expit(X @ beta)
    g = X.T @ (p - y)
    if prior_mean is not None:
        g[:2] += prior_prec @ (beta[:2] - prior_mean)
    return g


def fit_pooled(d: pd.DataFrame) -> np.ndarray:
    X, y = _design(d), d.y.to_numpy(float)
    res = minimize(_nll, np.zeros(X.shape[1]), args=(X, y), jac=_grad, method="L-BFGS-B")
    return res.x


def fit_hierarchical(d: pd.DataFrame, min_decisions: int = 60, n_iter: int = 3) -> tuple[pd.DataFrame, dict]:
    """Empirical-Bayes random intercept + random slope on edge.

    Step 1: pooled logit gives fixed effects (count, tokens, leverage) and
            the league intercept/slope.
    Step 2: per-hitter MAP of (intercept, slope) with the league values as
            a Gaussian prior; prior variance re-estimated from the spread
            of per-hitter estimates minus their sampling variance
            (method of moments), iterated.
    Returns per-hitter posterior means and SDs plus the prior.
    """
    beta_pool = fit_pooled(d)
    fixed = beta_pool[2:]
    prior_mean = beta_pool[:2].copy()
    prior_var = np.array([1.0, 0.5])  # start wide; re-estimated below
    groups = {b: g for b, g in d.groupby("batter_id") if len(g) >= min_decisions}
    rows = None
    for it in range(n_iter):
        prior_prec = np.diag(1.0 / prior_var)
        rows = []
        for b, g in groups.items():
            X, y = _design(g), g.y.to_numpy(float)
            # offset the fixed effects, fit only intercept and slope
            off = X[:, 2:] @ fixed

            def nll(bs):
                beta = np.concatenate([bs, fixed])
                return _nll(beta, X, y, prior_mean, prior_prec)

            def grad(bs):
                beta = np.concatenate([bs, fixed])
                return _grad(beta, X, y, prior_mean, prior_prec)[:2]

            res = minimize(nll, prior_mean.copy(), jac=grad, method="L-BFGS-B")
            bs = res.x
            # observed-information posterior SD
            p = expit(X[:, :2] @ bs + off)
            W = p * (1 - p)
            H = (X[:, :2] * W[:, None]).T @ X[:, :2] + prior_prec
            try:
                cov = np.linalg.inv(H)
                sd = np.sqrt(np.diag(cov))
            except np.linalg.LinAlgError:
                sd = np.array([np.nan, np.nan])
            rows.append({
                "batter_id": b, "n_decisions": len(g), "n_challenges": int(y.sum()),
                "n_opps": int(g.abs_ball.sum()),
                "intercept": bs[0], "slope": bs[1], "intercept_sd": sd[0], "slope_sd": sd[1],
            })
        R = pd.DataFrame(rows)
        # method-of-moments update of the prior variance
        for j, c in enumerate(["intercept", "slope"]):
            tot = R[c].var()
            samp = (R[f"{c}_sd"] ** 2).mean()
            prior_var[j] = max(tot - samp, 1e-3)
        prior_mean = np.array([R.intercept.mean(), R.slope.mean()])
        log.info("EB iter %d: prior mean %s, prior var %s", it, prior_mean.round(3), prior_var.round(4))
    prior = {"mean": prior_mean, "var": prior_var, "fixed": dict(zip(FIXED, fixed)), "n_hitters": len(R)}
    # standardised judgment score: slope relative to league, in prior SDs
    R["judgment_z"] = (R.slope - prior_mean[1]) / np.sqrt(prior_var[1])
    R["criterion_z"] = (R.intercept - prior_mean[0]) / np.sqrt(prior_var[0])
    # perceived-edge threshold: edge at which P(challenge)=0.5 for this hitter, league-average controls
    R["edge_at_50"] = -R.intercept / R.slope
    return R, prior


# --------------------------------------------------------------------------- #
# Signal-detection summary (descriptive, shrunk)
# --------------------------------------------------------------------------- #
def sdt_metrics(d: pd.DataFrame, min_decisions: int = 60) -> pd.DataFrame:
    """Hit rate (challenged | true ball), false-alarm rate (challenged | true
    strike), each shrunk with a beta prior fit by moments across hitters,
    then d' = z(HR) - z(FAR)."""
    g = d.groupby("batter_id").agg(
        n_ball=("abs_ball", "sum"),
        hits=("y", lambda s: int((s * d.loc[s.index, "abs_ball"]).sum())),
        n_strike=("abs_ball", lambda s: int((1 - s).sum())),
        fas=("y", lambda s: int((s * (1 - d.loc[s.index, "abs_ball"])).sum())),
        n=("y", "size"),
    )
    g = g[g.n >= min_decisions].copy()

    def beta_shrink(k, n):
        rate = (k / n.replace(0, np.nan)).dropna()
        m, v = rate.mean(), rate.var()
        if not (0 < m < 1) or v <= 0 or v >= m * (1 - m):
            a0, b0 = 1.0, 1.0
        else:
            common = m * (1 - m) / v - 1
            a0, b0 = m * common, (1 - m) * common
        return (k + a0) / (n + a0 + b0)

    g["hit_rate"] = beta_shrink(g.hits, g.n_ball)
    g["fa_rate"] = beta_shrink(g.fas, g.n_strike)
    g["d_prime"] = norm.ppf(g.hit_rate.clip(1e-3, 1 - 1e-3)) - norm.ppf(g.fa_rate.clip(1e-3, 1 - 1e-3))
    g["criterion_c"] = -0.5 * (norm.ppf(g.hit_rate.clip(1e-3, 1 - 1e-3)) + norm.ppf(g.fa_rate.clip(1e-3, 1 - 1e-3)))
    return g.reset_index()


def judgment_table(pt: pd.DataFrame, min_decisions: int = 60) -> tuple[pd.DataFrame, dict]:
    d = build_decision_table(pt)
    R, prior = fit_hierarchical(d, min_decisions=min_decisions)
    S = sdt_metrics(d, min_decisions=min_decisions)
    names = pt.groupby("batter_id").batter_name.first()
    out = R.merge(S, on="batter_id", how="left")
    out.insert(1, "batter_name", out.batter_id.map(names))
    return out, prior
