"""The eyes-versus-approach test.

Question. Among hitters with similar chase rates, do those with poor
location-adjusted challenge judgment show (a) costlier chases, (b) worse
two-strike contact, (c) weaker year-to-year chase improvement?

Design. Judgment enters as a covariate pooled across hitters, so the
per-hitter estimate only needs to be unbiased, not precise. Chase rate is
controlled flexibly (quintile dummies) and also stratified: within each
chase-rate quintile, top-tercile judgment is compared to bottom-tercile.
Regressions are weighted by sample size (chases, two-strike swings, or
balls seen) with heteroskedasticity-robust standard errors, and the
judgment coefficient is also bootstrapped over hitters.

Measurement-error note. judgment_z is an empirical-Bayes posterior mean
and is already shrunk, so the attenuation bias runs toward zero: a
significant coefficient is conservative, a null is not proof of absence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

OUTCOMES = {
    "chase_runs_per_100": ("chases", "runs per 100 chase swings (Statcast delta_run_exp)"),
    "chase_whiff_rate": ("chases", "whiff rate on chase swings"),
    "chase_xwoba_con": ("chase_contacts", "xwOBA on chase contact"),
    "ts_contact_rate": ("ts_swings", "contact rate on two-strike swings"),
    "ts_whiff_rate": ("ts_swings", "whiff rate on two-strike swings"),
    "k_rate": ("pa", "strikeout rate"),
    "chase_rate_2k": ("ts_abs_balls", "two-strike chase rate"),
    "yoy_chase_change": ("rb_balls_seen", "rulebook chase rate 2026 minus 2025 (negative = improved)"),
}


def assemble(judg: pd.DataFrame, hs: pd.DataFrame, prev_chase: pd.DataFrame | None = None,
             min_decisions: int = 60, min_pa: int = 150, min_challenges: int = 8) -> pd.DataFrame:
    """min_challenges matters: a hitter who never challenges reveals nothing
    about their eyes, and the slope is then set by the prior and the criterion."""
    df = hs.merge(judg[["batter_id", "judgment_z", "criterion_z", "slope", "intercept", "slope_sd",
                        "d_prime", "criterion_c", "n_decisions", "n_challenges", "n_opps", "edge_at_50"]],
                  on="batter_id", how="inner")
    df = df[(df.n_decisions >= min_decisions) & (df.pa >= min_pa) & (df.n_challenges >= min_challenges)].copy()
    if prev_chase is not None:
        df = df.merge(prev_chase[["batter_id", "rb_chase_rate"]].rename(columns={"rb_chase_rate": "rb_chase_rate_prev"}),
                      on="batter_id", how="left")
        df["yoy_chase_change"] = df.rb_chase_rate - df.rb_chase_rate_prev
    df["chase_q"] = pd.qcut(df.chase_rate, 5, labels=False)
    df["judg_tercile"] = pd.qcut(df.judgment_z, 3, labels=["low", "mid", "high"])
    return df


def fit_outcome(df: pd.DataFrame, outcome: str, weight_col: str, extra: str = "") -> dict:
    d = df[df[outcome].notna() & df[weight_col].notna() & (df[weight_col] > 0)].copy()
    f = f"{outcome} ~ judgment_z + C(chase_q) + criterion_z + bat_speed + C(bat_side)" + (f" + {extra}" if extra else "")
    m = smf.wls(f, data=d, weights=d[weight_col]).fit(cov_type="HC1")
    f0 = f"{outcome} ~ C(chase_q) + criterion_z + bat_speed + C(bat_side)" + (f" + {extra}" if extra else "")
    m0 = smf.wls(f0, data=d, weights=d[weight_col]).fit()
    # bootstrap over hitters
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(500):
        idx = rng.integers(0, len(d), len(d))
        dd = d.iloc[idx]
        try:
            bs.append(smf.wls(f, data=dd, weights=dd[weight_col]).fit().params["judgment_z"])
        except Exception:  # noqa: BLE001
            pass
    bs = np.array(bs)
    return {
        "outcome": outcome, "n": len(d),
        "beta_judgment": m.params["judgment_z"], "se_hc1": m.bse["judgment_z"], "p_hc1": m.pvalues["judgment_z"],
        "boot_lo": np.percentile(bs, 2.5), "boot_hi": np.percentile(bs, 97.5),
        "sd_outcome": d[outcome].std(), "delta_r2": m.rsquared - m0.rsquared,
        "beta_criterion": m.params["criterion_z"],
    }


def stratified_contrast(df: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """Within chase-rate quintile: mean outcome for high- vs low-judgment terciles."""
    d = df[df[outcome].notna()]
    g = d.groupby(["chase_q", "judg_tercile"], observed=True)[outcome].agg(["mean", "size"]).unstack("judg_tercile")
    out = pd.DataFrame({
        "chase_rate_mean": d.groupby("chase_q").chase_rate.mean(),
        "low_judg": g[("mean", "low")], "high_judg": g[("mean", "high")],
        "n_low": g[("size", "low")], "n_high": g[("size", "high")],
    })
    out["high_minus_low"] = out.high_judg - out.low_judg
    return out.reset_index()


def run_all(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows, strata = [], {}
    for oc, (w, _) in OUTCOMES.items():
        if oc not in df.columns or df[oc].notna().sum() < 50:
            continue
        rows.append(fit_outcome(df, oc, w))
        strata[oc] = stratified_contrast(df, oc)
    return pd.DataFrame(rows), strata


def fit_yoy_baseline(df: pd.DataFrame) -> dict | None:
    """Year-over-year chase change with the PRIOR season as the control.

    The main table controls for the current season's chase rate, which is
    the endpoint of the change and biases that row toward zero. The clean
    question is: among hitters with the same prior-season chase rate, do
    sharper eyes go with a lower current chase rate? Caveat: judgment and
    the outcome are both measured in the current season, so a same-season
    common cause is not excluded. The pre-registered version uses this
    season's judgment to predict next season's change."""
    if "rb_chase_rate_prev" not in df.columns:
        return None
    d = df[df.rb_chase_rate_prev.notna() & df.rb_chase_rate.notna()].copy()
    if len(d) < 50:
        return None
    w = d.rb_balls_seen
    m = smf.wls("rb_chase_rate ~ judgment_z + criterion_z + rb_chase_rate_prev + bat_speed + C(bat_side)", data=d, weights=w).fit(cov_type="HC1")
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(500):
        dd = d.iloc[rng.integers(0, len(d), len(d))]
        try:
            bs.append(smf.wls("rb_chase_rate ~ judgment_z + criterion_z + rb_chase_rate_prev + bat_speed + C(bat_side)",
                              data=dd, weights=dd.rb_balls_seen).fit().params["judgment_z"])
        except Exception:  # noqa: BLE001
            pass
    bs = np.array(bs)
    return {"outcome": "chase_rate_given_prev", "n": len(d), "beta_judgment": m.params["judgment_z"],
            "se_hc1": m.bse["judgment_z"], "p_hc1": m.pvalues["judgment_z"],
            "boot_lo": np.percentile(bs, 2.5), "boot_hi": np.percentile(bs, 97.5),
            "sd_outcome": d.rb_chase_rate.std(), "delta_r2": np.nan, "beta_criterion": m.params["criterion_z"],
            "beta_prev": m.params["rb_chase_rate_prev"]}


def judgment_stability(j1: pd.DataFrame, j2: pd.DataFrame, min_decisions: int = 60) -> dict:
    """Split-half or year-to-year reliability of the judgment score."""
    m = j1.merge(j2, on="batter_id", suffixes=("_1", "_2"))
    m = m[(m.n_decisions_1 >= min_decisions) & (m.n_decisions_2 >= min_decisions)]
    r = np.corrcoef(m.judgment_z_1, m.judgment_z_2)[0, 1] if len(m) > 5 else np.nan
    return {"n": len(m), "r": r, "spearman_brown_full": 2 * r / (1 + r) if r == r else np.nan}
