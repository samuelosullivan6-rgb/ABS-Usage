"""Location-adjusted challenge judgment: a hitter's perception, stripped of intent.

Decision set. Every pitch the hitter took that the umpire called a strike
while the batting team still held a token. The hitter either challenged or
did not. The true ABS location (mid-plate edge distance) is known for every
pitch from the trajectory, so this is a psychometric experiment the league
runs for free: stimulus = true edge distance, response = challenge.

Model. A hierarchical logit of challenge on true edge distance with a
per-hitter random intercept and random slope:

    P(challenge) = F(a + b * edge + u_i + v_i * edge + direction + controls)

  * slope b + v_i is perceptual SHARPNESS (eyes): how steeply the hitter's
    challenge rate rises as the pitch moves from inside the zone to outside.
  * intercept a + u_i is the CRITERION (approach / conservatism): how far
    off the plate a pitch must look before the hitter spends a token.

Direction. At the same true distance hitters challenge a low called strike
about twice as often as a high one (docs/review_2026-10-23.md, B2). The side
of the box the pitch missed on (above, below, side, corner; side is the
reference) enters as fixed intercept shifts in the pooled fit and as offsets
in the per-hitter fits, so a hitter's criterion is net of his stimulus mix.
Direction x edge slopes are tested by likelihood ratio and kept only if they
earn their parameters (``direction_slopes="auto"``).

Count, token state and leverage enter as fixed effects because they shift
the criterion, not the eyes. The window is asymmetric (-4 to +6 inches):
pitches deep inside the zone carry no information about the transition,
while every challenge on the ball side lies within 6 inches and those
pitches are the only view of the upper half of the curve.

Estimator. Two-step empirical Bayes: a pooled logit gives the fixed effects
and starting values; then per-hitter MAP estimates of (intercept, slope)
under a bivariate Gaussian prior with a full 2x2 covariance, updated by EM
(prior covariance = covariance of posterior means + mean posterior
covariance). The per-hitter posterior slope-intercept correlation is
reported, because on the tail of a logistic the two are nearly collinear.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import chi2, norm

log = logging.getLogger(__name__)

EDGE_WINDOW_IN = (-4.0, 6.0)
PLATE_HALF_WIDTH_IN = 8.5
DIRECTIONS = ("above", "below", "side", "corner")
DIR_REF = "side"
DIR_COLS = [f"dir_{k}" for k in DIRECTIONS if k != DIR_REF]
DIR_SLOPE_COLS = [f"{c}_x_edge" for c in DIR_COLS]
FIXED = ["two_strike", "three_ball", "tokens2", "late", "close"]
TOKEN_COL = "bat_tokens_before_ruleB"  # the per-inning extras top-up (decision 5)


def miss_direction(mid_x, mid_z, sz_top, sz_bot) -> np.ndarray:
    """Side of the ABS box a pitch's centre is on, from mid-plate location (feet).

    In inches: dx = |mid_x| * 12 - 8.5 (beyond the plate's half-width),
    da = (mid_z - sz_top) * 12 (above), db = (sz_bot - mid_z) * 12 (below).
    Corner if dx > 0 and (da > 0 or db > 0); otherwise the largest of the
    three, which for a pitch inside the box is the nearest edge."""
    mid_x, mid_z = np.asarray(mid_x, float), np.asarray(mid_z, float)
    sz_top, sz_bot = np.asarray(sz_top, float), np.asarray(sz_bot, float)
    dx = np.abs(mid_x) * 12 - PLATE_HALF_WIDTH_IN
    da = (mid_z - sz_top) * 12
    db = (sz_bot - mid_z) * 12
    corner = (dx > 0) & ((da > 0) | (db > 0))
    largest = np.select([(da >= dx) & (da >= db), (db >= dx) & (db > da)], ["above", "below"], "side")
    return np.where(corner, "corner", largest)


def build_decision_table(pt: pd.DataFrame, edge_window=EDGE_WINDOW_IN, token_col: str | None = None) -> pd.DataFrame:
    """Hitter challenge decisions on umpire-called strikes with a token in hand.

    token_col defaults to the per-inning extras rule when the pitch table has
    it; the ledger accounting (``bat_tokens_before``) drops the 35 batter
    challenges made in extra innings by teams whose regular tokens were gone."""
    if token_col is None:
        token_col = TOKEN_COL if TOKEN_COL in pt.columns else "bat_tokens_before"
    d = pt[pt.orig_called_strike & (pt[token_col] >= 1) & pt.abs_edge_in.notna()].copy()
    d["y"] = d.batter_challenged.astype(int)
    d["edge"] = d.abs_edge_in
    d["abs_ball"] = (d.abs_edge_in > 0).astype(int)
    d["two_strike"] = (d.strikes_before == 2).astype(int)
    d["three_ball"] = (d.balls_before == 3).astype(int)
    d["tokens2"] = (d[token_col] >= 2).astype(int)
    d["late"] = (d.inning >= 7).astype(int)
    if "bat_score" in d.columns and "fld_score" in d.columns:
        d["close"] = ((d.bat_score - d.fld_score).abs() <= 1).astype(int)
    else:
        d["close"] = 0
    if {"mid_x", "mid_z", "sz_top", "sz_bot"} <= set(d.columns):
        d["direction"] = miss_direction(d.mid_x, d.mid_z, d.sz_top, d.sz_bot)
        for k, c in zip([k for k in DIRECTIONS if k != DIR_REF], DIR_COLS):
            d[c] = (d.direction == k).astype(int)
            d[f"{c}_x_edge"] = d[c] * d.edge
    lo, hi = edge_window
    d = d[(d.edge >= lo) & (d.edge <= hi)]
    return d


# --------------------------------------------------------------------------- #
# Pooled logit
# --------------------------------------------------------------------------- #
def _design(d: pd.DataFrame, fixed: list[str] = FIXED) -> np.ndarray:
    return np.column_stack([np.ones(len(d)), d.edge.to_numpy(float)] + [d[c].to_numpy(float) for c in fixed])


def _nll(beta, X, y):
    eta = X @ beta
    return float(np.sum(np.logaddexp(0.0, eta) - y * eta))


def _grad(beta, X, y):
    return X.T @ (expit(X @ beta) - y)


def fit_pooled(d: pd.DataFrame, fixed: list[str] = FIXED) -> tuple[np.ndarray, float]:
    """Pooled logit; returns coefficients [intercept, edge, *fixed] and the log-likelihood."""
    X, y = _design(d, fixed), d.y.to_numpy(float)
    res = minimize(_nll, np.zeros(X.shape[1]), args=(X, y), jac=_grad, method="L-BFGS-B",
                   options={"maxiter": 2000, "gtol": 1e-7})
    return res.x, -res.fun


def direction_lr_test(d: pd.DataFrame) -> dict:
    """Likelihood-ratio tests of direction against the direction-blind pooled model:
    intercept shifts (3 df), then direction x edge slopes on top (3 df)."""
    _, ll0 = fit_pooled(d, FIXED)
    b1, ll1 = fit_pooled(d, FIXED + DIR_COLS)
    b2, ll2 = fit_pooled(d, FIXED + DIR_COLS + DIR_SLOPE_COLS)
    n = len(d)
    k = len(DIR_COLS)
    out = {"n": n,
           "chi2_intercepts": 2 * (ll1 - ll0), "df_intercepts": k, "p_intercepts": float(chi2.sf(2 * (ll1 - ll0), k)),
           "chi2_slopes": 2 * (ll2 - ll1), "df_slopes": k, "p_slopes": float(chi2.sf(2 * (ll2 - ll1), k)),
           "chi2_both": 2 * (ll2 - ll0), "df_both": 2 * k, "p_both": float(chi2.sf(2 * (ll2 - ll0), 2 * k)),
           "bic_penalty_slopes": k * np.log(n),
           "intercepts": dict(zip(DIR_COLS, b1[2 + len(FIXED):].tolist())),
           "slopes": dict(zip(DIR_SLOPE_COLS, b2[2 + len(FIXED) + k:].tolist()))}
    # keep the slope terms only if they matter: significant at 1% AND they pay their BIC price
    out["keep_slopes"] = bool(out["p_slopes"] < 0.01 and out["chi2_slopes"] > out["bic_penalty_slopes"])
    return out


# --------------------------------------------------------------------------- #
# Empirical-Bayes hierarchical logit: per-hitter (intercept, slope), 2x2 prior
# --------------------------------------------------------------------------- #
def _fit_hitters(codes, n_groups, edge, y, off, prior_mean, prior_prec, start=None, max_iter=50, tol=1e-9):
    """Vectorised Newton-Raphson for every hitter's MAP (intercept, slope) at once.

    The penalised log-likelihood is strictly concave, so Newton from the prior
    mean converges; steps are capped at 2 logits for safety. Returns modes and
    the 2x2 posterior covariances (inverse negative Hessian at the mode), and
    the slope-intercept correlation implied by the likelihood alone."""
    th = np.tile(prior_mean, (n_groups, 1)) if start is None else start.copy()
    P = prior_prec
    for _ in range(max_iter):
        eta = th[codes, 0] + th[codes, 1] * edge + off
        p = expit(eta)
        r = p - y
        w = p * (1 - p)
        dm = th - prior_mean
        g0 = np.bincount(codes, r, n_groups) + dm @ P[0]
        g1 = np.bincount(codes, r * edge, n_groups) + dm @ P[1]
        h00 = np.bincount(codes, w, n_groups) + P[0, 0]
        h01 = np.bincount(codes, w * edge, n_groups) + P[0, 1]
        h11 = np.bincount(codes, w * edge * edge, n_groups) + P[1, 1]
        det = h00 * h11 - h01 ** 2
        s0 = (h11 * g0 - h01 * g1) / det
        s1 = (h00 * g1 - h01 * g0) / det
        step = np.column_stack([s0, s1])
        big = np.abs(step).max(axis=1, keepdims=True)
        step = step * (2.0 / np.maximum(big, 2.0))
        th = th - step
        if np.abs(step).max() < tol:
            break
    eta = th[codes, 0] + th[codes, 1] * edge + off
    p = expit(eta)
    w = p * (1 - p)
    h00 = np.bincount(codes, w, n_groups) + P[0, 0]
    h01 = np.bincount(codes, w * edge, n_groups) + P[0, 1]
    h11 = np.bincount(codes, w * edge * edge, n_groups) + P[1, 1]
    det = h00 * h11 - h01 ** 2
    cov = np.stack([np.column_stack([h11 / det, -h01 / det]), np.column_stack([-h01 / det, h00 / det])], axis=1)
    # correlation the data alone imply (likelihood information, prior removed)
    l00, l01, l11 = h00 - P[0, 0], h01 - P[0, 1], h11 - P[1, 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        lik_corr = -l01 / np.sqrt(l00 * l11)
    return th, cov, lik_corr


def fit_hierarchical(d: pd.DataFrame, min_decisions: int = 60, n_iter: int = 6, direction: bool = True,
                     direction_slopes: bool | str = "auto", full_cov: bool = True) -> tuple[pd.DataFrame, dict]:
    """Empirical-Bayes random intercept + random slope on edge.

    Step 1: pooled logit gives the fixed effects (count, tokens, leverage and,
            with ``direction``, the side of the box the pitch missed on) and
            the league intercept and slope.
    Step 2: per-hitter MAP of (intercept, slope) with the fixed effects as an
            offset and a bivariate Gaussian prior; the prior mean and
            covariance are re-estimated by EM (covariance of posterior means
            plus mean posterior covariance), iterated. ``full_cov=False``
            keeps the prior diagonal (the estimator before 2026-10-23).
    ``direction_slopes``: True / False, or "auto" to run the likelihood-ratio
    test and keep the direction x edge terms only if they matter.
    Returns per-hitter posterior means, SDs and slope-intercept correlations,
    plus the prior.
    """
    has_dir = direction and all(c in d.columns for c in DIR_COLS)
    lr = None
    fixed_cols = list(FIXED)
    if has_dir:
        fixed_cols += DIR_COLS
        if direction_slopes == "auto":
            lr = direction_lr_test(d)
            direction_slopes = lr["keep_slopes"]
        if direction_slopes:
            fixed_cols += DIR_SLOPE_COLS
    beta_pool, ll_pool = fit_pooled(d, fixed_cols)
    fixed = beta_pool[2:]
    prior_mean = beta_pool[:2].copy()
    prior_cov = np.diag([1.0, 0.5])  # start wide; re-estimated below

    counts = d.batter_id.value_counts()
    keep_ids = counts.index[counts >= min_decisions]
    g = d[d.batter_id.isin(keep_ids)]
    codes, ids = pd.factorize(g.batter_id, sort=True)
    n_groups = len(ids)
    edge = g.edge.to_numpy(float)
    y = g.y.to_numpy(float)
    off = (np.column_stack([g[c].to_numpy(float) for c in fixed_cols]) @ fixed) if fixed_cols else np.zeros(len(g))
    th = None
    for it in range(n_iter):
        prior_prec = np.linalg.inv(prior_cov)
        th, cov, lik_corr = _fit_hitters(codes, n_groups, edge, y, off, prior_mean, prior_prec, start=th)
        # EM update of the prior: the estimates are posterior means, so
        # Sigma = Cov(posterior means) + E[posterior covariance]
        new_cov = np.cov(th.T, ddof=1) + cov.mean(axis=0)
        if not full_cov:
            new_cov = np.diag(np.diag(new_cov))
        new_cov[0, 0] = max(new_cov[0, 0], 1e-4)
        new_cov[1, 1] = max(new_cov[1, 1], 1e-4)
        prior_cov = new_cov
        prior_mean = th.mean(axis=0)
        log.info("EB iter %d: prior mean %s, prior cov %s", it, prior_mean.round(3), prior_cov.round(4).tolist())

    n_dec = np.bincount(codes, minlength=n_groups)
    n_chal = np.bincount(codes, y, n_groups).astype(int)
    n_opps = np.bincount(codes, g.abs_ball.to_numpy(float), n_groups).astype(int)
    sd = np.sqrt(np.stack([cov[:, 0, 0], cov[:, 1, 1]], axis=1))
    R = pd.DataFrame({
        "batter_id": ids, "n_decisions": n_dec, "n_challenges": n_chal, "n_opps": n_opps,
        "intercept": th[:, 0], "slope": th[:, 1], "intercept_sd": sd[:, 0], "slope_sd": sd[:, 1],
        "post_corr": cov[:, 0, 1] / (sd[:, 0] * sd[:, 1]), "lik_corr": lik_corr,
    })
    prior_var = np.diag(prior_cov).copy()
    prior_corr = float(prior_cov[0, 1] / np.sqrt(prior_var[0] * prior_var[1]))
    prior = {"mean": prior_mean, "var": prior_var, "cov": prior_cov, "corr": prior_corr,
             "fixed": dict(zip(fixed_cols, fixed)), "n_hitters": n_groups, "full_cov": full_cov,
             "direction": has_dir, "direction_slopes": bool(has_dir and direction_slopes),
             "pooled_loglik": ll_pool, "direction_test": lr,
             "corr_posterior_means": float(np.corrcoef(th[:, 0], th[:, 1])[0, 1]) if n_groups > 2 else np.nan}
    # standardised scores: relative to the league, in prior SDs, with their posterior SDs
    R["judgment_z"] = (R.slope - prior_mean[1]) / np.sqrt(prior_var[1])
    R["judgment_z_sd"] = R.slope_sd / np.sqrt(prior_var[1])
    R["criterion_z"] = (R.intercept - prior_mean[0]) / np.sqrt(prior_var[0])
    R["criterion_z_sd"] = R.intercept_sd / np.sqrt(prior_var[0])
    # edge at which P(challenge) = 0.5 for this hitter on a side pitch with no count/leverage shifters
    R["edge_at_50"] = -R.intercept / R.slope
    return R, prior


# --------------------------------------------------------------------------- #
# Reliability with intervals
# --------------------------------------------------------------------------- #
def spearman_brown(r):
    r = np.asarray(r, float)
    return 2 * r / (1 + r)


def split_half_reliability(x1, x2, n_boot: int = 0, seed: int = 0, level: float = 0.95) -> dict:
    """Split-half correlation stepped up to a full season (Spearman-Brown), with a
    Fisher-z interval on r carried through the step-up, and optionally a
    hitter-bootstrap percentile interval (resampling hitter pairs)."""
    x1, x2 = np.asarray(x1, float), np.asarray(x2, float)
    n = len(x1)
    out = {"n": n, "r": np.nan, "rel": np.nan, "rel_lo": np.nan, "rel_hi": np.nan, "boot_lo": np.nan, "boot_hi": np.nan}
    if n < 5:
        return out
    r = float(np.corrcoef(x1, x2)[0, 1])
    zc = norm.ppf(0.5 + level / 2)
    z, se = np.arctanh(r), 1 / np.sqrt(n - 3)
    lo, hi = np.tanh(z - zc * se), np.tanh(z + zc * se)
    # Spearman-Brown is monotone on (-1, 1); below r = 0 the step-up is not a reliability, floor at 0
    out.update(r=r, rel=float(spearman_brown(r)), rel_lo=float(max(spearman_brown(lo), 0.0)) if lo > -1 else 0.0,
               rel_hi=float(spearman_brown(hi)))
    if n_boot:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, n, (n_boot, n))
        a, b = x1[idx], x2[idx]
        a = a - a.mean(1, keepdims=True)
        b = b - b.mean(1, keepdims=True)
        rb = (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))
        sb = spearman_brown(rb)
        out.update(boot_lo=float(np.percentile(sb, 100 * (1 - level) / 2)), boot_hi=float(np.percentile(sb, 100 * (1 + level) / 2)))
    return out


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


def eyes_label_threshold(rel: pd.DataFrame, floor: float = 0.5) -> int | None:
    """Full-season challenge count at which a per-hitter eyes label is supported.

    ``rel`` is the primary model's rows of the reliability table (one per
    minimum challenges per half-season). The threshold is the smallest
    per-half minimum whose slope reliability interval has its lower end at or
    above ``floor``, doubled to a full season (both halves must reach it).
    None when no tested minimum clears, i.e. the data support no eyes label."""
    ok = rel[rel.slope_lo >= floor].sort_values("min_challenges_per_half")
    if ok.empty:
        return None
    return int(2 * ok.min_challenges_per_half.iloc[0])
