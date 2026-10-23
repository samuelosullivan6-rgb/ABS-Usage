"""Generate the club brief: one page for coaches (front half) and the front office (back half).

Usage: python scripts/11_club_brief.py [--label mlb_2026]
Outputs (report/):
  club_brief_<season>.html             page body for web publishing (no document skeleton)
  club_brief_<season>_standalone.html  the same page wrapped for opening locally or attaching to email
Every table on the page names the processed file it came from.
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from abs_judgment.config import PROCESSED, ROOT  # noqa: E402
from abs_judgment.judgment import EDGE_WINDOW_IN, build_decision_table, eyes_label_threshold  # noqa: E402
from abs_judgment.policy import interp_belief, perceived_margin  # noqa: E402

LOGIT_TO_PROBIT = 1.7
STAKES = [("0-0 count, bases empty", 0.05), ("typical pitch", 0.19), ("3-2 count, runners on", 0.50)]
BANDS = [("1st to 3rd", 1), ("4th to 6th", 7), ("7th and 8th", 13), ("9th", 17)]
MARGIN_CAP_IN = 6.0  # margins beyond this are shown as ">6": no pitch looks that far off often enough to matter
EYES_SHOW_MIN = 8  # below this many challenges the slope is mostly the prior and the criterion; no eyes range is shown
RELIABLE_FLOOR = 0.5  # an eyes label needs the lower end of the slope's reliability interval at or above this
CRIT_LABELS = ("Chaser, free with tokens", "Chaser, sparing with tokens")
SAVANT_PRICE = 0.2
REPORT = ROOT / "report"


def pct_rank(ref: pd.Series, x: pd.Series) -> pd.Series:
    ref = np.sort(ref.dropna().to_numpy())
    out = pd.Series(np.searchsorted(ref, x.to_numpy(), side="right") / len(ref) * 100, index=x.index)
    return out.where(x.notna())  # no score, no percentile


def b64png(p: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


def build_data(label: str) -> dict:
    season = label.split("_")[1]
    pt = pd.read_parquet(PROCESSED / f"pitch_table_{label}.parquet",
                         columns=["batter_id", "half", "home_team", "away_team", "orig_called_strike",
                                  "bat_tokens_before", "bat_tokens_before_ruleB", "abs_edge_in", "batter_challenged",
                                  "strikes_before", "balls_before", "inning", "bat_score", "fld_score",
                                  "mid_x", "mid_z", "sz_top", "sz_bot"])
    pt["bat_abbr"] = np.where(pt.half == "top", pt.away_team, pt.home_team)
    team = pt.groupby("batter_id").bat_abbr.agg(lambda s: s.mode().iloc[0] if s.notna().any() else "")
    nteams = pt.groupby("batter_id").bat_abbr.nunique()
    dec = build_decision_table(pt)  # the judgment model's own decision set and window
    decision = {"n": int(len(dec)), "challenged": int(dec.y.sum()), "ump_wrong_share": float(dec.abs_ball.mean()),
                "lo": float(-EDGE_WINDOW_IN[0]), "hi": float(EDGE_WINDOW_IN[1])}

    ch = pd.read_parquet(PROCESSED / f"challenges_{label}.parquet")
    gs = pd.read_csv(PROCESSED / f"game_summaries_{label}.csv")
    bat = ch[ch.initiator_role == "batter"].groupby("initiator_id").agg(chal=("is_overturned", "size"), won=("is_overturned", "sum"))
    roles = ch.groupby("initiator_role").is_overturned.agg(["size", "mean"])

    J = pd.read_csv(PROCESSED / f"judgment_{label}.csv")
    hs = pd.read_csv(PROCESSED / f"hitter_season_{label}.csv")
    prior = json.load(open(PROCESSED / f"judgment_prior_{label}.json"))
    tv = pd.read_csv(PROCESSED / f"token_value_{label}.csv")
    sim = pd.read_csv(PROCESSED / f"policy_simulation_{label}.csv")
    seeds = pd.read_csv(PROCESSED / f"policy_seeds_{label}.csv")
    sens = pd.read_csv(PROCESSED / f"policy_sigma_{label}.csv")
    belief = json.load(open(PROCESSED / f"policy_belief_{label}.json"))
    bgrid = pd.DataFrame(belief["grid"])
    model = belief["belief"]
    ab_league = (belief["a"], belief["b"]) if model == "calibrated" else None
    res = pd.read_csv(PROCESSED / f"eyes_vs_approach_{label}.csv")
    relc = pd.read_csv(PROCESSED / f"reliability_checks_{label}.csv")
    rel_primary = relc[relc.primary.astype(bool)].sort_values("min_challenges_per_half")
    rel_before = relc[relc.model.str.startswith("before") & (relc.window_in == "-4..4")].set_index("min_challenges_per_half")
    # the challenge count from which the data support an eyes-or-approach call (None: no volume this season does)
    eyes_min = eyes_label_threshold(rel_primary, RELIABLE_FLOOR)

    sigma_league = belief["sigma_league"]

    # ---- dugout table: how sure you need to be, by inning band, tokens held and stake
    def dv(h: int, k: int) -> float:
        return float(tv[(tv.half_inning == h) & (tv.tokens == k)].token_value.iloc[0])

    dugout = []
    for band, h in BANDS:
        for k in (2, 1):
            row = {"band": band, "tokens": k}
            for name, s in STAKES:
                d = dv(h, k)
                p = d / (s + d)
                row[f"p_{name}"] = round(p * 100)
                row[f"in_{name}"] = round(float(perceived_margin(p, sigma_league, model, ab_league)), 1)
            dugout.append(row)

    # ---- hitter cards
    H = hs.merge(J, on="batter_id", how="left", suffixes=("", "_j"))
    H = H[H.pa >= 100].copy()
    H["team"] = H.batter_id.map(team).fillna("")
    H["nteams"] = H.batter_id.map(nteams).fillna(1).astype(int)
    H["chal"] = H.batter_id.map(bat.chal).fillna(0).astype(int)
    H["won"] = H.batter_id.map(bat.won).fillna(0).astype(int)
    ref = H[(H.pa >= 150) & (H.chal >= EYES_SHOW_MIN)]
    H["eyes_pct"] = pct_rank(ref.judgment_z, H.judgment_z)  # rank among rated regulars: the median split for an eyes label
    # the eyes score as a range: posterior mean -/+ one posterior SD, as percentiles of the league's distribution of eyes
    H["eyes_mid"] = 100 * norm.cdf(H.judgment_z)
    H["eyes_lo"] = 100 * norm.cdf(H.judgment_z - H.judgment_z_sd)
    H["eyes_hi"] = 100 * norm.cdf(H.judgment_z + H.judgment_z_sd)
    H["crit_pct"] = pct_rank(H[(H.pa >= 150) & H.criterion_z.notna()].criterion_z, H.criterion_z)
    H["chase_pct"] = pct_rank(H[H.pa >= 150].chase_rate, H.chase_rate)
    H["sigma"] = LOGIT_TO_PROBIT / H.slope.clip(lower=0.05)
    ab_i = interp_belief(bgrid, H.sigma) if model == "calibrated" else None
    s_typ = STAKES[1][1]
    for col, (h, k) in {"m_early2": (1, 2), "m_early1": (1, 1), "m_mid1": (7, 1), "m_late1": (17, 1)}.items():
        p = dv(h, k) / (s_typ + dv(h, k))
        H[col] = np.where(H.slope.notna(), perceived_margin(p, H.sigma, model, ab_i), np.nan)

    def label_row(r) -> str:
        if r.pa < 150:
            return "Small sample"
        if pd.isna(r.judgment_z):
            return "Not rated"
        if r.chase_pct >= 60:
            if eyes_min is not None and r.chal >= eyes_min:
                return "Eyes case" if r.eyes_pct < 50 else "Approach case"
            # below the volume the eyes score can support, sort on the criterion, which is reliable
            return CRIT_LABELS[0] if r.crit_pct >= 50 else CRIT_LABELS[1]
        if r.chase_pct <= 40:
            return "Disciplined"
        return "Average chaser"

    H["label"] = H.apply(label_row, axis=1)
    keep = ["batter_id", "batter_name", "team", "nteams", "pa", "chal", "won", "n_decisions", "judgment_z", "judgment_z_sd",
            "eyes_mid", "eyes_lo", "eyes_hi", "criterion_z", "crit_pct", "sigma", "chase_rate", "chase_pct", "k_rate", "bb_rate",
            "m_early2", "m_early1", "m_mid1", "m_late1", "label"]
    hitters = H[keep].sort_values(["team", "batter_name"])
    hitters = hitters.replace({np.nan: None}).to_dict("records")
    for r in hitters:
        for k, v in list(r.items()):
            if isinstance(v, (np.floating, float)) and v is not None:
                r[k] = round(float(v), 4)
            elif isinstance(v, (np.integer,)):
                r[k] = int(v)
            elif isinstance(v, (np.bool_, bool)):
                r[k] = bool(v)
    counts = pd.Series([r["label"] for r in hitters if r["pa"] >= 150]).value_counts().to_dict()

    # ---- study tables
    sim_rows = sim[["rule", "overturned", "challenges_per_team_game", "overturn_rate", "runs_per_team_162", "minus_static", "minus_static_se",
                    "minus_observed", "minus_observed_se"]].round(3).to_dict("records")
    seed_rows = seeds.round(4).to_dict("records")
    sens_rows = sens.sort_values(["belief", "sigma"], ascending=[True, True]).round(4).to_dict("records")

    # ---- the dugout guidance, worded from the token values (whichever way they point)
    t1 = tv[tv.tokens == 1].sort_values("half_inning")
    above = t1[t1.token_value >= SAVANT_PRICE]
    typ = STAKES[1][1]
    p_at = lambda h, k, st: dv(h, k) / (st + dv(h, k))  # noqa: E731
    sharp = sens[(sens.belief == model) & (sens.token1_first_pitch >= SAVANT_PRICE)]
    guide = {
        "never_above": bool(above.empty), "always_above": bool(len(above) == len(t1)),
        "cross_inning": int(t1[t1.token_value < SAVANT_PRICE].inning.iloc[0]) if 0 < len(above) < len(t1) else None,
        "p_early2": p_at(1, 2, typ), "p_early1": p_at(1, 1, typ), "p_static_typ": SAVANT_PRICE / (SAVANT_PRICE + typ),
        "p_big_early2": p_at(1, 2, STAKES[2][1]), "p_late1": p_at(17, 1, typ),
        "in_early2": float(perceived_margin(p_at(1, 2, typ), sigma_league, model, ab_league)),
        "in_early1": float(perceived_margin(p_at(1, 1, typ), sigma_league, model, ab_league)),
        "sharp_sigma": float(sharp.sigma.max()) if len(sharp) else None,
        "sharp_tok1": float(sharp.loc[sharp.sigma.idxmax(), "token1_first_pitch"]) if len(sharp) else None,
    }
    cal = sens[sens.belief == model].sort_values("token1_first_pitch")
    guide["cross_sigma"] = (float(np.interp(SAVANT_PRICE, cal.token1_first_pitch, cal.sigma))
                            if cal.token1_first_pitch.min() < SAVANT_PRICE < cal.token1_first_pitch.max() else None)
    tvx_path = PROCESSED / f"token_value_extras_{label}.csv"
    guide["extras_tok1"] = float(pd.read_csv(tvx_path).query("tokens == 1 and half == 'top'").token_value.iloc[0]) if tvx_path.exists() else None
    eyes_rows = res[["outcome", "n", "beta_judgment", "se_hc1", "p_hc1", "label"]].round(4).to_dict("records")
    rel_rows = [{**r, "before_slope": float(rel_before.slope.get(r["min_challenges_per_half"], np.nan)),
                 "before_lo": float(rel_before.slope_lo.get(r["min_challenges_per_half"], np.nan)),
                 "before_hi": float(rel_before.slope_hi.get(r["min_challenges_per_half"], np.nan))}
                for r in rel_primary[["min_challenges_per_half", "n", "slope", "slope_lo", "slope_hi",
                                      "intercept", "intercept_lo", "intercept_hi"]].to_dict("records")]
    yoy = res[res.outcome == "chase_rate_given_prev"]
    yoy_row = yoy.iloc[0].to_dict() if len(yoy) else None

    teams = sorted({r["team"] for r in hitters if r["team"]})

    # ---- chart series (drawn in the page; every chart also has a table view)
    token_series = [{"h": int(r.half_inning), "inning": int(r.inning), "half": r.half,
                     "t1": round(float(tv[(tv.half_inning == r.half_inning) & (tv.tokens == 1)].token_value.iloc[0]), 4),
                     "t2": round(float(tv[(tv.half_inning == r.half_inning) & (tv.tokens == 2)].token_value.iloc[0]), 4)}
                    for r in tv[tv.tokens == 1].sort_values("half_inning").itertuples()]
    curve = pd.read_csv(PROCESSED / f"judgment_curve_{label}.csv")
    off = prior.get("fixed_offset_mean", 0.0)
    rated = J[(J.n_decisions >= 150) & (J.n_challenges >= 8)]
    top, bot = rated.loc[rated.slope.idxmax()], rated.loc[rated.slope.idxmin()]
    curve_data = {
        "bins": [{"x": round(float(r.edge_mid), 2), "n": int(r.n), "rate": round(float(r.challenge_rate), 4)} for r in curve.itertuples()],
        "curves": [
            {"name": "League", "a": round(prior["mean"][0] + off, 4), "b": round(prior["mean"][1], 4)},
            {"name": top.batter_name, "a": round(float(top.intercept) + off, 4), "b": round(float(top.slope), 4)},
            {"name": bot.batter_name, "a": round(float(bot.intercept) + off, 4), "b": round(float(bot.slope), 4)},
        ],
    }
    refh = H[(H.pa >= 150) & (H.chal >= EYES_SHOW_MIN) & H.judgment_z.notna()]  # the hitters with an eyes range
    m, b0 = np.polyfit(refh.judgment_z, refh.chase_rate, 1)
    scatter = {
        "points": [{"id": int(r.batter_id), "name": r.batter_name, "x": round(float(r.judgment_z), 3),
                    "y": round(float(r.chase_rate), 4), "label": r.label, "chal": int(r.chal)} for r in refh.itertuples()],
        "fit": {"m": round(float(m), 4), "b": round(float(b0), 4), "r": round(float(np.corrcoef(refh.judgment_z, refh.chase_rate)[0, 1]), 3)},
        "x_cut": round(float(refh.judgment_z.median()), 3),
        "y_cut": round(float(H[H.pa >= 150].chase_rate.quantile(0.6)), 4),
    }
    order = ["static", "dynamic", "observed", "oracle"]
    bars = sorted([{"rule": r["rule"], "runs": round(float(r["runs_per_team_162"]), 1),
                    "chal": round(float(r["challenges_per_team_game"]), 2),
                    "ot": None if pd.isna(r["overturn_rate"]) else round(float(r["overturn_rate"]), 3)}
                   for r in sim_rows if r["rule"] in order], key=lambda b: order.index(b["rule"]))
    charts = {"token": token_series, "curve": curve_data, "scatter": scatter, "bars": bars}
    return {
        "charts": charts,
        "season": season,
        "facts": {
            "games": int(gs.game_pk.nunique()), "pitches": int(gs.n_pitches.sum()), "challenges": int(len(ch)),
            "overturn_all": float(ch.is_overturned.mean()),
            "roles": {r: {"n": int(roles.loc[r, "size"]), "ot": float(roles.loc[r, "mean"])} for r in roles.index},
            "decision": decision, "sigma_league": sigma_league, "slope": prior["mean"][1],
            "slope_sd": float(np.sqrt(prior["var"][1])), "crit_sd": float(np.sqrt(prior["var"][0])),
            "fixed": prior["fixed"], "reliability": prior.get("reliability_full_season", {}),
            "prior_corr": float(prior.get("corr", 0.0)), "post_corr_median": float(prior.get("post_corr_median", 0.0)),
            "direction_test": prior.get("direction_test") or {"chi2_slopes": 0.0, "bic_penalty_slopes": 0.0},
            "eyes_min": eyes_min, "rel_primary": rel_primary.to_dict("records"),
            "n_scored": int(len(J)), "n_ref": int(len(ref)),
            "token_first_start": dv(1, 1), "token_second_start": dv(1, 2), "token_first_9th": dv(17, 1),
            "belief": belief, "r_chase": float(np.corrcoef(refh.judgment_z, refh.chase_rate)[0, 1]), "n_r_chase": int(len(refh)),
        },
        "guide": guide, "seeds": seed_rows, "sens": sens_rows,
        "dugout": dugout, "stakes": [n for n, _ in STAKES], "teams": teams, "hitters": hitters, "label_counts": counts,
        "sim": sim_rows, "eyes": eyes_rows, "rel": rel_rows, "yoy": yoy_row,
    }


TEMPLATE = r"""<title>ABS Challenge Club Brief</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700&family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: a two-part brief. Part one is for the dugout (spend a token, hitter cards, the sort); part two is the evidence for the front office. Sticky anchor nav; wide tables scroll inside their own box. */
:root{
  --bg:#f3f5f8; --surface:#ffffff; --fg:#15202e; --muted:#5b6878; --line:#d5dce6; --soft:#e9eef5;
  --accent:#1f4fd1; --accent-ink:#ffffff; --ot:#1b7f4c; --cf:#b3261e; --warn-bg:#fff3cf; --warn-fg:#6a4a00;
  --display:"Barlow Condensed","Arial Narrow",Arial,sans-serif; --body:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif; --mono:"IBM Plex Mono",ui-monospace,Menlo,monospace;
  /* chart roles: categorical slots validated against this surface (light #ffffff, dark #161d29) */
  --series-1:#2a78d6; --series-2:#eb6834; --series-3:#1baf7a;
  --viz-grid:#e6eaf0; --viz-axis:#c9d1dc; --viz-muted:#6b7686; --viz-ink:#15202e; --viz-wash:#eef2f7;
}
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){
  --bg:#0f141d; --surface:#161d29; --fg:#e7ecf3; --muted:#9fabbc; --line:#2a3443; --soft:#1e2735;
  --accent:#7ea2ff; --accent-ink:#0f141d; --ot:#4fc98c; --cf:#f28b7d; --warn-bg:#3a2f0d; --warn-fg:#f3d98b; color-scheme:dark;
  --series-1:#3987e5; --series-2:#d95926; --series-3:#199e70;
  --viz-grid:#232d3b; --viz-axis:#3a4656; --viz-muted:#9fabbc; --viz-ink:#e7ecf3; --viz-wash:#1c2532 } }
:root[data-theme="dark"]{
  --bg:#0f141d; --surface:#161d29; --fg:#e7ecf3; --muted:#9fabbc; --line:#2a3443; --soft:#1e2735;
  --accent:#7ea2ff; --accent-ink:#0f141d; --ot:#4fc98c; --cf:#f28b7d; --warn-bg:#3a2f0d; --warn-fg:#f3d98b; color-scheme:dark;
  --series-1:#3987e5; --series-2:#d95926; --series-3:#199e70;
  --viz-grid:#232d3b; --viz-axis:#3a4656; --viz-muted:#9fabbc; --viz-ink:#e7ecf3; --viz-wash:#1c2532 }
/* chart chrome */
figure.viz{position:relative;border:1px solid var(--line);border-radius:6px;background:var(--surface);padding:12px 12px 8px;display:grid;gap:8px;min-width:0}
figure.viz .viz-head{display:flex;justify-content:space-between;gap:10px;align-items:baseline;flex-wrap:wrap}
figure.viz .viz-title{font-family:var(--body);font-weight:600;font-size:14px;color:var(--fg)}
figure.viz .viz-legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12.5px;color:var(--muted)}
figure.viz .viz-legend span{display:inline-flex;align-items:center;gap:6px}
figure.viz .viz-legend i{display:inline-block;width:16px;height:2px;border-radius:1px}
figure.viz .viz-legend i.dot{width:8px;height:8px;border-radius:50%}
figure.viz .viz-plot{width:100%;min-width:0}
figure.viz svg{display:block;width:100%;height:auto;font-family:var(--body);font-size:12px;overflow:visible}
figure.viz svg:focus-visible{outline:2px solid var(--accent);outline-offset:2px;border-radius:4px}
figure.viz .tick{fill:var(--viz-muted);font-variant-numeric:tabular-nums;font-size:11.5px}
figure.viz .axis-label{fill:var(--viz-muted);font-size:11.5px}
figure.viz .dlabel{fill:var(--viz-ink);font-size:12px;font-weight:600}
figure.viz .note{fill:var(--viz-muted);font-size:11.5px}
figure.viz details{font-size:12.5px;color:var(--muted)}
figure.viz details summary{cursor:pointer;font-family:var(--mono);font-size:11.5px;letter-spacing:.04em;text-transform:uppercase}
figure.viz details table{margin-top:6px;font-size:12.5px}
figure.viz figcaption{font-size:13px;color:var(--muted);max-width:70ch}
.tip{position:absolute;z-index:4;pointer-events:none;background:var(--surface);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:8px 10px;font-size:12.5px;box-shadow:0 4px 14px rgba(0,0,0,.12);min-width:150px;max-width:260px}
.tip .t{font-weight:600;margin-bottom:4px}
.tip .row{display:flex;gap:8px;align-items:center;justify-content:space-between}
.tip .row b{font-variant-numeric:tabular-nums}
.tip .row span{color:var(--muted);display:inline-flex;align-items:center;gap:6px}
.tip .row i{display:inline-block;width:12px;height:2px}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);font-family:var(--body);font-size:15px;line-height:1.5;padding-inline:16px;padding-block:0 48px}
.wrap{max-width:1040px;margin:0 auto}
h1,h2,h3{font-family:var(--display);line-height:1.05;text-wrap:balance;margin:0}
h1{font-size:clamp(34px,6vw,54px);font-weight:700;letter-spacing:.01em}
h2{font-size:clamp(24px,3.6vw,32px);font-weight:700;margin-top:8px}
h3{font-size:19px;font-weight:600}
p{max-width:68ch;margin:0}
.eyebrow{font-family:var(--mono);font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
header.top{padding-block:28px 18px;display:grid;gap:10px}
.lede{font-size:17px;max-width:70ch}
nav.sticky{position:sticky;top:env(safe-area-inset-top,0px);background:var(--bg);border-bottom:1px solid var(--line);padding-block:8px;z-index:5;display:flex;gap:6px;flex-wrap:wrap}
nav.sticky a{font-family:var(--mono);font-size:12px;text-decoration:none;color:var(--fg);background:var(--soft);padding:6px 10px;border-radius:999px}
nav.sticky a:hover,nav.sticky a:focus-visible{background:var(--accent);color:var(--accent-ink);outline:none}
section{padding-block:30px 8px;display:grid;gap:14px}
section+section{border-top:1px solid var(--line)}
.part{font-family:var(--mono);font-size:12px;letter-spacing:.1em;text-transform:uppercase;color:var(--accent)}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:18px;align-items:start}
.grid2>*{min-width:0}
.rules{display:grid;gap:8px;padding:0;margin:0;list-style:none;counter-reset:r}
.rules li{display:grid;grid-template-columns:28px 1fr;gap:10px;align-items:start}
.rules li::before{counter-increment:r;content:counter(r);font-family:var(--display);font-weight:700;font-size:22px;line-height:1;color:var(--accent)}
.tablebox{overflow-x:auto;border:1px solid var(--line);border-radius:6px;background:var(--surface)}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:13.5px}
th,td{padding:7px 10px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}
th{font-family:var(--mono);font-size:11.5px;letter-spacing:.04em;text-transform:uppercase;color:var(--muted);font-weight:500;background:var(--soft);position:sticky;top:0}
td.num,th.num{text-align:right}
tr:last-child td{border-bottom:0}
.src{font-family:var(--mono);font-size:11.5px;color:var(--muted)}
.pill{display:inline-block;font-family:var(--mono);font-size:11px;padding:2px 8px;border-radius:999px;border:1px solid var(--line);white-space:nowrap}
.pill.eyes{border-color:var(--cf);color:var(--cf)}
.pill.approach{border-color:var(--accent);color:var(--accent)}
.pill.disc{border-color:var(--ot);color:var(--ot)}
.pill.prov{background:var(--warn-bg);color:var(--warn-fg);border-color:transparent}
.pill.crit{border-style:dashed;border-color:var(--accent);color:var(--accent)}
.pill.crit2{border-style:dashed;border-color:var(--muted);color:var(--muted)}
.ot{color:var(--ot);font-weight:600}.cf{color:var(--cf);font-weight:600}
.callout{background:var(--warn-bg);color:var(--warn-fg);padding:12px 14px;border-radius:6px;font-size:14px;max-width:none}
.controls{display:flex;gap:10px;flex-wrap:wrap;align-items:end}
.controls label{display:grid;gap:4px;font-family:var(--mono);font-size:11.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.controls select,.controls input{font:inherit;font-size:14px;padding:7px 9px;border:1px solid var(--line);border-radius:6px;background:var(--surface);color:var(--fg);min-width:150px}
.controls input:focus-visible,.controls select:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
.check{display:flex;gap:6px;align-items:center;font-family:var(--body);text-transform:none;letter-spacing:0;font-size:14px;color:var(--fg)}
figure{margin:0;display:grid;gap:6px}
figure img{width:100%;height:auto;border:1px solid var(--line);border-radius:6px;background:#fff}
figcaption{font-size:13px;color:var(--muted);max-width:70ch}
.stat{display:grid;gap:2px}
.stat b{font-family:var(--display);font-size:30px;font-weight:700;line-height:1}
.stat span{font-size:13px;color:var(--muted)}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px;padding:14px;border:1px solid var(--line);border-radius:6px;background:var(--surface)}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:13px;color:var(--muted)}
dl{display:grid;grid-template-columns:max-content 1fr;gap:6px 14px;margin:0;max-width:72ch}
dt{font-family:var(--mono);font-size:12.5px;color:var(--muted)}dd{margin:0}
a{color:var(--accent)}
@media (max-width:520px){dl{grid-template-columns:1fr}th,td{padding:6px 7px}}
@media (prefers-reduced-motion:no-preference){nav.sticky a{transition:background .15s}}
</style>

<div class="wrap">
<header class="top">
  <div class="eyebrow">@@SEASON@@ regular season · built from public MLB data · @@GAMES@@ games, @@CHALLENGES@@ challenges</div>
  <h1>ABS challenge judgment</h1>
  <p class="lede">What the challenge system reveals about a hitter's eyes, when a token is worth spending, and how to tell an eyes problem from an approach problem. Part one is for the dugout. Part two is the evidence.</p>
  <nav class="sticky" aria-label="Sections">
    <a href="#spend">Spend a token</a><a href="#cards">Hitter cards</a><a href="#sort">Eyes or approach</a><a href="#study">The study</a><a href="#methods">Methods</a>
  </nav>
</header>

<section id="spend">
  <div class="part">Part one · for the dugout</div>
  <h2>When to spend a token</h2>
  <div class="grid2">
    <div style="display:grid;gap:12px">
      <p>A challenge token is not worth a fixed amount. It is worth what it can still buy over the rest of the game. At first pitch the first token is worth about <b>@@TOK1@@ runs</b> and the second about <b>@@TOK2@@</b>; by the 9th the last token is worth about <b>@@TOK9@@</b>. @@SPEND_VERDICT@@</p>
      <ol class="rules">@@SPEND_RULES@@</ol>
      <p class="callout">@@SPEND_REVISION@@</p>
      <p class="src">Source: token_value_@@LABEL@@.csv and token_value_extras_@@LABEL@@.csv (backward induction), stakes_@@LABEL@@.csv, policy_sigma_@@LABEL@@.csv.</p>
    </div>
    <figure class="viz" id="viz-token">
      <div class="viz-head"><div class="viz-title">What a held token is worth, by half-inning</div>
        <div class="viz-legend"><span><i style="background:var(--series-1)"></i>First token</span><span><i style="background:var(--series-2)"></i>Second token</span><span><i style="background:var(--viz-axis)"></i>Savant's flat price</span></div></div>
      <div class="viz-plot" role="img" aria-label="@@TOKEN_ARIA@@"></div>
      <figcaption>Runs. Hover or focus and use the arrow keys to read any half-inning.</figcaption>
      <details><summary>Table view</summary><div class="tablebox"><table class="viz-table"></table></div></details>
    </figure>
  </div>
  <h3>How sure you need to be</h3>
  <p>Percent chance the call flips that justifies a challenge, by inning, tokens held and what the pitch is worth. The second number in each cell is how far off the plate the pitch has to look to an average eye that knows the umpire is right on about @@UMP_RIGHT@@% of near-edge calls (negative means a pitch that looks a touch inside is still worth it).</p>
  <div class="tablebox"><table id="dugout"><thead><tr><th>Innings</th><th>Tokens</th>@@DUGOUT_HEAD@@</tr></thead><tbody>@@DUGOUT_ROWS@@</tbody></table></div>
  <p class="src">Threshold = token value / (stake + token value). Inches = the perceived edge at which the calibrated belief P(flip | perceived edge) = expit(@@BELIEF_A@@ + @@BELIEF_B@@ × edge) reaches the threshold, for league-average perceptual noise of @@SIGMA@@ in. Per-hitter versions are in the cards below. Source: policy_belief_@@LABEL@@.json.</p>
</section>

<section id="cards">
  <h2>Hitter challenge cards</h2>
  <p>Every called strike a hitter takes with a token in hand is a test of his eyes: the true location is known, and he either challenges or he doesn't. The <b>eyes</b> score is how sharply his challenge rate rises as the pitch moves from inside the zone to outside it, against the league. The <b>criterion</b> is how freely he spends tokens. The margins are how far off the plate a pitch should look <em>to him</em> before he challenges on a typical pitch, early with both tokens, early with the last token, mid-game, and in the 9th.</p>
  <div class="callout">@@CARD_CALLOUT@@</div>
  <div class="controls">
    <label>Team<select id="team"><option value="">All teams</option>@@TEAM_OPTS@@</select></label>
    <label>Search<input id="q" type="search" placeholder="hitter name" autocomplete="off"></label>
    <label>Sort<select id="sort"><option value="batter_name">Name</option><option value="eyes_mid">Eyes (best first)</option><option value="chase_pct">Chase rate (highest first)</option><option value="chal">Challenges</option><option value="crit_pct">Spends tokens (freest first)</option></select></label>
    <label class="check"><input id="rated" type="checkbox"> Only hitters with 15+ challenges</label>
  </div>
  <div class="tablebox"><table id="cards-table"><thead><tr>
    <th>Hitter</th><th>Team</th><th class="num">PA</th><th class="num">Chal</th><th class="num">Won</th><th class="num">Eyes range</th><th class="num">Spends pct</th><th class="num">Chase %</th><th class="num">Chase pct</th><th class="num">Early, 2 tok</th><th class="num">Early, last</th><th class="num">Mid, last</th><th class="num">9th, last</th><th>Sort</th>
  </tr></thead><tbody id="cards-body"></tbody></table></div>
  <div class="legend"><span><b>Won</b> = call overturned</span><span><b>Eyes range</b> = percentile band of perceptual sharpness among the league's hitters, the posterior mean plus or minus one SD; shown from @@EYES_SHOW_MIN@@ challenges</span><span><b>Spends pct</b> = percentile of willingness to challenge among @@N_CRIT_REF@@ regulars</span><span>Margins in inches off the plate on a typical pitch; &gt;6 means no pitch looks far enough off to be worth it</span></div>
  <p class="src" id="count"></p>
  <p class="src">Source: judgment_@@LABEL@@.csv, hitter_season_@@LABEL@@.csv, challenges_@@LABEL@@.parquet, token_value_@@LABEL@@.csv, policy_belief_@@LABEL@@.json. Team is the hitter's most frequent batting team. A margin rests on the hitter's eyes slope, which is mostly noise at the volumes one season gives; read it to the nearest inch.</p>
</section>

<section id="sort">
  <h2>Eyes case or approach case</h2>
  <div class="grid2">
    <div style="display:grid;gap:12px">
      <p>A hitter who chases has one of two problems. Either he does not see the pitch well enough to know it is a ball, or he sees it fine and swings anyway. The fix differs: pitch-recognition work for the first, a count plan for the second. Chase rate alone cannot separate them because every swing mixes perception with intent. The challenge decision on a take has no swing in it.</p>
      <p>@@SORT_RULE@@</p>
      <div class="stats">@@LABEL_STATS@@</div>
      <p><b>What the evidence supports.</b> Sharper eyes on takes go with chasing less (correlation @@R_CHASE@@ across the @@N_R_CHASE@@ rated regulars, stronger among frequent challengers) and with cutting the chase rate from the prior season. They do <em>not</em> go with cheaper chases, better two-strike contact or fewer strikeouts once chase rate is held fixed. So the sort tells you which lever moves the <em>rate</em>; it does not say who pays most per chase.</p>
    </div>
    <figure class="viz" id="viz-scatter">
      <div class="viz-head"><div class="viz-title">Eyes on takes against chasing on swings, @@SEASON@@ regulars</div></div>
      <div class="viz-plot" role="img" aria-label="Scatter of each regular's challenge judgment score against his chase rate, with a downward fitted line and guides marking the eyes-case and approach-case regions"></div>
      <figcaption>One dot per hitter with 150+ PA and @@EYES_SHOW_MIN@@+ challenges; each dot is a posterior mean with an SD of about @@Z_SD@@ on this scale. @@SCATTER_NOTE@@ Hover a dot, or focus the chart and use the arrow keys, to see who it is. The hitter cards above are the table view.</figcaption>
    </figure>
  </div>
</section>

<section id="study">
  <div class="part">Part two · the evidence</div>
  <h2>What the study found</h2>
  <div class="stats">
    <div class="stat"><b>@@CHALLENGES@@</b><span>challenges in @@GAMES@@ games, all recovered at the pitch level</span></div>
    <div class="stat"><b>@@OT_ALL@@%</b><span>overturned overall; batters @@OT_BAT@@%, catchers @@OT_CAT@@%</span></div>
    <div class="stat"><b>@@UMP_WRONG@@%</b><span>of called strikes from @@W_LO@@ in inside to @@W_HI@@ in outside the edge were actually balls</span></div>
    <div class="stat"><b>@@DEC_RATE@@%</b><span>of those were challenged by the hitter</span></div>
  </div>
  <div class="grid2">
    <figure class="viz" id="viz-curve">
      <div class="viz-head"><div class="viz-title">Challenge rate on called strikes, by true distance from the zone edge</div>
        <div class="viz-legend" id="curve-legend"></div></div>
      <div class="viz-plot" role="img" aria-label="Observed challenge rate by half-inch bin of true edge distance, rising from under 1 percent inside the zone to about 40 percent outside, with the fitted league curve and the sharpest and shallowest hitters' curves"></div>
      <figcaption>Dots are observed half-inch bins, sized by pitch count. Curves are the fitted logits. The shaded band is inside the ABS zone. At the edge itself hitters challenge only 11 to 14% of true balls.</figcaption>
      <details><summary>Table view</summary><div class="tablebox"><table class="viz-table"></table></div></details>
    </figure>
    <div style="display:grid;gap:12px">
      <h3>Measurement</h3>
      <p>A hierarchical logit of challenge on true edge distance gives each hitter a slope (eyes) and an intercept (criterion), net of the side of the box the pitch missed on: at the same distance a low called strike is challenged far more often than a high one (@@FX_BELOW@@ logits for a low pitch and @@FX_ABOVE@@ for a high one, against a pitch off the side; @@FX_CORNER@@ for a corner). League slope @@SLOPE@@ per inch; between-hitter spread @@SLOPE_SD@@ on the slope and @@CRIT_SD@@ on the criterion, correlated @@PRIOR_CORR@@ across hitters. Two strikes shift the criterion by @@FX_2K@@ logits; a late inning by @@FX_LATE@@.</p>
      <div class="tablebox"><table><thead><tr><th>Min challenges per half</th><th class="num">Hitters</th><th class="num">Eyes</th><th class="num">Criterion</th><th class="num">Eyes, earlier model</th></tr></thead><tbody>@@REL_ROWS@@</tbody></table></div>
      <p class="src">Eyes and criterion: split-half reliability, Spearman-Brown to a full season, with 95% intervals (Fisher z); hitters are those with the minimum in both halves. The earlier model had no direction and a diagonal prior, on a &plusmn;4 inch window. @@REL_NOTE@@ Source: reliability_checks_@@LABEL@@.csv.</p>
    </div>
  </div>
  <h3>Policy</h3>
  <p>Every game replayed with each rule. The static and dynamic rules decide on a perceived edge drawn with league-average Gaussian noise (@@SIGMA@@ in) and a calibrated belief, and are scored on the true ABS outcome. Runs per team per 162 games; differences are paired by game over @@GAMES@@ games, plus or minus one standard error.</p>
  <div class="callout"><b>The observed row is not the same eyes.</b> It is real players deciding with their own eyes: @@OBS_OT@@% overturned on @@OBS_CHAL@@ challenges per team-game. Under the league-average noise model only @@N_P06@@ of @@N_OPP@@ opportunities carry a belief above 60%, yet players won @@OBS_WON@@ challenges. Real players see something the noise model does not, so the gap between a simulated rule and observed behaviour is not a property of the rule. Compare static with dynamic; read observed as context.</div>
  <figure class="viz" id="viz-bars">
    <div class="viz-head"><div class="viz-title">Runs per team-season by challenge rule</div></div>
    <div class="viz-plot" role="img" aria-label="@@BARS_ARIA@@"></div>
    <figcaption>The gap to the oracle is the value of perception; the gap between static and dynamic is the value of pricing the token.</figcaption>
  </figure>
  <div class="tablebox"><table><thead><tr><th>Rule</th><th>Eyes</th><th class="num">Challenges per team-game</th><th class="num">Overturned</th><th class="num">Runs per team-season</th><th class="num">vs static</th><th class="num">vs observed</th></tr></thead><tbody>@@SIM_ROWS@@</tbody></table></div>
  <p class="src">Oracle challenges exactly the umpire's misses, the ceiling with perfect eyes. Observed is what players did on the same pitches. @@SEED_LINE@@ Source: policy_simulation_@@LABEL@@.csv, policy_seeds_@@LABEL@@.csv.</p>
  <h3>What the belief model does</h3>
  <p>The challenger's belief is the probability the call flips given the edge they perceive. The calibrated belief is fit on the opportunities themselves, so it knows the umpire is usually right: in every fifth of its range it is within @@CAL_GAP@@ points of the true flip rate. The flat-prior belief, P = &Phi;(perceived edge / noise), was used on this page before 2026-10-23. It ignores the umpire's base rate, and a pitch it calls 80 to 100% likely flips about half the time.</p>
  <div class="tablebox"><table><thead><tr><th>Belief</th><th class="num">First token, first pitch</th><th class="num">Static: chal / overturned / runs</th><th class="num">Dynamic: chal / overturned / runs</th><th class="num">Dynamic − static</th><th class="num">Dynamic − observed</th></tr></thead><tbody>@@BELIEF_ROWS@@</tbody></table></div>
  <p class="src">League-average noise, noise draw @@SEED@@. Source: policy_seeds_@@LABEL@@.csv.</p>
  <h3>Sensitivity to the noise level</h3>
  <p>The noise level, @@SIGMA@@ inches, comes from the league eyes slope and is an upper bound (the slope also absorbs criterion variation). Under the flat prior the token's value barely moves with it; under calibrated beliefs it does, and below about @@SHARP_SIGMA@@ inches the first token at first pitch is worth more than Savant's 0.20.</p>
  <div class="tablebox"><table><thead><tr><th class="num">Noise (in)</th><th class="num">Calibrated: first token</th><th class="num">Static runs</th><th class="num">Dynamic runs</th><th class="num">Dynamic − static</th><th class="num">Flat prior: first token</th><th class="num">Dynamic − static</th></tr></thead><tbody>@@SENS_ROWS@@</tbody></table></div>
  <p class="src">Noise draw @@SEED@@ at every level; the starred row is the league estimate. Source: policy_sigma_@@LABEL@@.csv.</p>
  <h3>The test</h3>
  <p>Among hitters with similar chase rates, does poor challenge judgment predict costlier chases, worse two-strike contact, or weaker year-over-year improvement? Effect of one SD of sharper eyes, holding chase-rate quintile, criterion, bat speed and bat side fixed. @@N_TEST@@ regulars with 8+ challenges.</p>
  <div class="tablebox"><table><thead><tr><th>Outcome</th><th class="num">n</th><th class="num">Effect</th><th class="num">SE</th><th class="num">p</th></tr></thead><tbody>@@EYES_ROWS@@</tbody></table></div>
  <p class="src">Weighted least squares with robust errors and a hitter bootstrap. The last row is the clean year-over-year test: current chase rate with the prior season's rate as the control. Source: eyes_vs_approach_@@LABEL@@.csv.</p>
  <p>Reading: judgment predicts whether a hitter chases and which way he is moving, not what happens when he swings at a ball. The one positive result, the year-over-year change, shares a season with the judgment measurement; the pre-registered version uses this season's judgment to predict next season's change.</p>
</section>

<section id="methods">
  <h2>Methods and caveats</h2>
  <dl>
    <dt>Data</dt><dd>MLB StatsAPI live feeds carry every ABS challenge at the pitch level with the initiating player and outcome (review type MJ). Per-team counts reconcile with the in-game token ledger in all but one game. Statcast CSVs supply run values and bat tracking, joined on every pitch.</dd>
    <dt>Zone</dt><dd>The feed's strike-zone bounds are the fixed ABS zone (53.5% and 27% of measured height). Ball location is solved at the plate midpoint from the nine-parameter trajectory; the signed edge distance uses a 2.9-inch ball and reproduces Savant's values exactly. It agrees with the ABS ruling on 99.99% of challenged pitches.</dd>
    <dt>Decision set</dt><dd>Umpire-called strikes the hitter took with a token in hand (counting the extra-innings top-up), from @@W_LO@@ inches inside the edge to @@W_HI@@ outside: @@DEC_N@@ pitches, @@DEC_CHAL@@ challenged. The window is wider on the ball side because every batter challenge lies within @@W_HI@@ inches outside, and those pitches are the only view of the top half of the curve.</dd>
    <dt>Judgment model</dt><dd>Logit of challenge on edge distance with a per-hitter random intercept and slope under a full-covariance prior, fit by two-step empirical Bayes; count, token and leverage shifters and the side of the box the pitch missed on (above, below, side, corner) as fixed effects. Direction-by-distance slopes were tested and left out (chi-square @@LR_SLOPES@@ on 3 df, below the BIC price of @@LR_BIC@@). A hitter's slope and intercept posteriors correlate @@POST_CORR@@ at the median. The slope is unidentified for hitters who rarely challenge.</dd>
    <dt>Token value</dt><dd>Backward induction over eight decision slots per half-inning for nine innings, with opportunities arriving at the observed rate (@@LAMBDA@@ per team-half-inning: taken pitches within 3 inches of the edge, both sides) and stakes from Statcast run expectancy. Extra innings enter as a continuation value at the last out of the 9th: @@N_EXTRAS@@ of @@GAMES@@ games went to the 10th, each extra inning is followed by another with probability @@RHO@@, and a team at zero gets one token back at the start of every extra inning, as the ledgers show. The simulated rules play the same token rule.</dd>
    <dt>Belief</dt><dd>A challenger perceives the true edge plus Gaussian noise and holds the calibrated posterior P(flip | perceived edge), a logit of the true flip on the perceived edge fit with the same noise draw. The per-hitter margins use the same fit at each hitter's own noise level.</dd>
    <dt>Caveats</dt><dd>One season of ABS. Run units, not win probability, so late close games are undervalued. The simulated rules use league-average Gaussian perception; the observed row is real players, who know more than that model. Under calibrated beliefs the token's value depends on the noise level, which is an upper bound. @@EYES_CAVEAT@@ Year-over-year judgment and outcome share a season.</dd>
    <dt>Update</dt><dd>Rerun <code>scripts/run_pipeline.sh</code> and <code>scripts/11_club_brief.py</code> on the new season. Everything on this page regenerates from the processed tables named under each section.</dd>
  </dl>
</section>
</div>

<script>
const DATA = @@DATA@@;
const fmt1 = v => v == null ? "" : v > @@CAP@@ ? "&gt;@@CAP@@" : (v > 0 ? "+" : "") + v.toFixed(1);
const pct = v => v == null ? "" : Math.round(v);
const pillClass = l => l === "Eyes case" ? "pill eyes" : l === "Approach case" ? "pill approach" : l === "Disciplined" ? "pill disc" : l === "@@CRIT_L0@@" ? "pill crit" : l === "@@CRIT_L1@@" ? "pill crit2" : "pill";
const body = document.getElementById("cards-body");
const teamSel = document.getElementById("team"), q = document.getElementById("q"), sortSel = document.getElementById("sort"), rated = document.getElementById("rated");
function render(){
  const t = teamSel.value, s = q.value.trim().toLowerCase(), k = sortSel.value, onlyRated = rated.checked;
  let rows = DATA.hitters.filter(r => (!t || r.team === t) && (!s || r.batter_name.toLowerCase().includes(s)) && (!onlyRated || r.chal >= 15));
  const desc = {eyes_mid:1, chase_pct:1, chal:1, crit_pct:1};
  rows.sort((a,b) => { const va = a[k], vb = b[k]; if (k === "batter_name") return va.localeCompare(vb); if (va == null) return 1; if (vb == null) return -1; return desc[k] ? vb - va : va - vb; });
  body.innerHTML = rows.map(r => {
    const unrated = r.chal < @@EYES_SHOW_MIN@@ || r.eyes_mid == null;
    const eyes = unrated ? "<span class='pill'>n/a</span>" : pct(r.eyes_lo) + "&ndash;" + pct(r.eyes_hi);
    const m = v => unrated ? "" : fmt1(v);
    return `<tr><td>${r.batter_name}</td><td>${r.team}${r.nteams > 1 ? " +" + (r.nteams-1) : ""}</td><td class="num">${r.pa}</td><td class="num">${r.chal}</td><td class="num"><span class="${r.won > r.chal - r.won ? "ot" : ""}">${r.won}</span></td><td class="num">${eyes}</td><td class="num">${r.crit_pct == null ? "" : pct(r.crit_pct)}</td><td class="num">${(r.chase_rate*100).toFixed(1)}</td><td class="num">${pct(r.chase_pct)}</td><td class="num">${m(r.m_early2)}</td><td class="num">${m(r.m_early1)}</td><td class="num">${m(r.m_mid1)}</td><td class="num">${m(r.m_late1)}</td><td><span class="${pillClass(r.label)}">${r.label}</span></td></tr>`;
  }).join("");
  document.getElementById("count").textContent = rows.length + " hitters shown of " + DATA.hitters.length + " with 100+ PA.";
}
[teamSel, q, sortSel, rated].forEach(el => el.addEventListener("input", render));
try { const saved = localStorage.getItem("abs-brief-team"); if (saved && [...teamSel.options].some(o => o.value === saved)) teamSel.value = saved; } catch (e) {}
teamSel.addEventListener("change", () => { try { localStorage.setItem("abs-brief-team", teamSel.value); } catch (e) {} });
render();

// ---------- charts: built from DATA.charts; every chart has a hover layer and a table view ----------
const NS = "http://www.w3.org/2000/svg";
const S1 = "var(--series-1)", S2 = "var(--series-2)", S3 = "var(--series-3)";
function el(tag, attrs, parent){ const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; }
function txt(parent, x, y, s, cls, anchor){ const t = el("text", {x, y, class: cls || "tick", "text-anchor": anchor || "start"}, parent); t.textContent = s; return t; }
function ticks(lo, hi, n){ const span = hi - lo, mag = Math.pow(10, Math.floor(Math.log10(span / n))); const step = [1, 2, 2.5, 5, 10].map(c => c * mag).find(c => span / c <= n); const out = []; for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10)); return out; }
const tip = document.createElement("div"); tip.className = "tip"; tip.hidden = true; tip.setAttribute("role", "status"); document.body.appendChild(tip);
function showTip(px, py, title, rows){
  tip.replaceChildren(); const t = document.createElement("div"); t.className = "t"; t.textContent = title; tip.appendChild(t);
  for (const r of rows){ const d = document.createElement("div"); d.className = "row"; const s = document.createElement("span"); if (r.color){ const i = document.createElement("i"); i.style.background = r.color; s.appendChild(i); } s.appendChild(document.createTextNode(r.label)); const b = document.createElement("b"); b.textContent = r.value; d.appendChild(s); d.appendChild(b); tip.appendChild(d); }
  tip.hidden = false; const w = tip.offsetWidth, vw = document.documentElement.clientWidth;
  let left = px + 14; if (left + w > vw - 8) left = px - w - 14; if (left < 8) left = 8;
  tip.style.left = (left + window.scrollX) + "px"; tip.style.top = (py + window.scrollY - 8) + "px";
}
function hideTip(){ tip.hidden = true; }
function fillTable(fig, head, rows){ const t = fig.querySelector(".viz-table"); if (!t) return; t.replaceChildren(); const th = document.createElement("thead"), tr = document.createElement("tr"); head.forEach((h, i) => { const c = document.createElement("th"); c.textContent = h; if (i) c.className = "num"; tr.appendChild(c); }); th.appendChild(tr); t.appendChild(th); const tb = document.createElement("tbody"); rows.forEach(r => { const q = document.createElement("tr"); r.forEach((v, i) => { const c = document.createElement("td"); c.textContent = v; if (i) c.className = "num"; q.appendChild(c); }); tb.appendChild(q); }); t.appendChild(tb); }
function responsive(plot, draw){ draw(); if ("ResizeObserver" in window){ let w = plot.clientWidth; new ResizeObserver(() => { if (plot.clientWidth !== w){ w = plot.clientWidth; draw(); } }).observe(plot); } }
function keyNav(svg, n, getIdx, show, hide){ svg.addEventListener("keydown", e => { if (e.key === "ArrowRight" || e.key === "ArrowLeft"){ e.preventDefault(); const i = getIdx(); show(i < 0 ? 0 : Math.min(n - 1, Math.max(0, i + (e.key === "ArrowRight" ? 1 : -1)))); } else if (e.key === "Escape") hide(); }); svg.addEventListener("blur", hide); }

// 1. token value: two-series line, reference level, crosshair tooltip
(function(){
  const fig = document.getElementById("viz-token"), plot = fig.querySelector(".viz-plot"), D = DATA.charts.token;
  const lab = d => d.inning + (d.half === "top" ? "T" : "B");
  fillTable(fig, ["Half-inning", "First token (runs)", "Second token (runs)"], D.map(d => [lab(d), d.t1.toFixed(3), d.t2.toFixed(3)]));
  responsive(plot, () => {
    const W = Math.max(280, plot.clientWidth), H = 300, m = {t: 18, r: 16, b: 34, l: 42}, n = D.length;
    plot.replaceChildren(); const svg = el("svg", {viewBox: `0 0 ${W} ${H}`, tabindex: 0}, plot);
    const x = i => m.l + i / (n - 1) * (W - m.l - m.r), yMax = Math.max(0.25, Math.ceil(Math.max(...D.map(d => d.t1)) * 1.15 * 20) / 20), y = v => m.t + (1 - v / yMax) * (H - m.t - m.b);
    ticks(0, yMax, 5).forEach(v => { el("line", {x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), stroke: "var(--viz-grid)"}, svg); txt(svg, m.l - 8, y(v) + 4, v.toFixed(2), "tick", "end"); });
    el("line", {x1: m.l, x2: W - m.r, y1: y(0), y2: y(0), stroke: "var(--viz-axis)"}, svg);
    D.forEach((d, i) => { if (i % 2 === 0) txt(svg, x(i), H - m.b + 18, lab(d), "tick", "middle"); });
    el("line", {x1: m.l, x2: W - m.r, y1: y(0.2), y2: y(0.2), stroke: "var(--viz-axis)", "stroke-width": 1.5}, svg);
    txt(svg, W - m.r, y(0.2) - 6, "Savant's flat price, 0.20", "note", "end");
    const series = [{name: "First token", key: "t1", color: S1}, {name: "Second token", key: "t2", color: S2}];
    series.forEach(s => { el("path", {d: D.map((d, i) => (i ? "L" : "M") + x(i).toFixed(1) + "," + y(d[s.key]).toFixed(1)).join(""), fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round"}, svg); txt(svg, x(0) + 8, y(D[0][s.key]) - 8, s.name, "dlabel"); });
    const cross = el("line", {y1: m.t, y2: H - m.b, stroke: "var(--viz-axis)", visibility: "hidden"}, svg);
    const marks = series.map(s => el("circle", {r: 4, fill: s.color, stroke: "var(--surface)", "stroke-width": 2, visibility: "hidden"}, svg));
    const hit = el("rect", {x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent"}, svg);
    let idx = -1;
    const show = i => { idx = i; cross.setAttribute("x1", x(i)); cross.setAttribute("x2", x(i)); cross.setAttribute("visibility", "visible"); marks.forEach((c, k) => { c.setAttribute("cx", x(i)); c.setAttribute("cy", y(D[i][series[k].key])); c.setAttribute("visibility", "visible"); }); const r = svg.getBoundingClientRect(); showTip(r.left + x(i) * r.width / W, r.top + m.t, "Start of the " + lab(D[i]).replace("T", " top").replace("B", " bottom").replace(/^(\d+)/, m0 => m0 + (["1","21","31"].includes(m0) ? "st" : ["2","22"].includes(m0) ? "nd" : ["3","23"].includes(m0) ? "rd" : "th")), series.map(s => ({label: s.name, value: D[i][s.key].toFixed(3) + " runs", color: s.color}))); };
    const hide = () => { idx = -1; cross.setAttribute("visibility", "hidden"); marks.forEach(c => c.setAttribute("visibility", "hidden")); hideTip(); };
    hit.addEventListener("pointermove", e => { const r = svg.getBoundingClientRect(), gx = (e.clientX - r.left) * W / r.width; let b = 0, bd = 1e9; for (let i = 0; i < n; i++){ const d = Math.abs(x(i) - gx); if (d < bd){ bd = d; b = i; } } show(b); });
    hit.addEventListener("pointerleave", hide); keyNav(svg, n, () => idx, show, hide);
  });
})();

// 2. psychometric curve: observed bins (sized by n) + three fitted curves, direct-labeled where they separate
(function(){
  const fig = document.getElementById("viz-curve"), plot = fig.querySelector(".viz-plot"), D = DATA.charts.curve;
  const colors = [S1, S2, S3], expit = z => 1 / (1 + Math.exp(-z)), f = (c, xv) => expit(c.a + c.b * xv);
  const leg = document.getElementById("curve-legend"); leg.replaceChildren();
  const li = (name, color, dot) => { const s = document.createElement("span"); const i = document.createElement("i"); i.style.background = color; if (dot) i.className = "dot"; s.appendChild(i); s.appendChild(document.createTextNode(name)); leg.appendChild(s); };
  li("Observed, by bin", S1, true); D.curves.forEach((c, k) => li(c.name + (k ? "" : " curve"), colors[k]));
  fillTable(fig, ["Edge (in)", "Pitches", "Observed", ...D.curves.map(c => c.name)], D.bins.map(b => [b.x.toFixed(2), b.n, (b.rate * 100).toFixed(1) + "%", ...D.curves.map(c => (f(c, b.x) * 100).toFixed(1) + "%")]));
  responsive(plot, () => {
    const W = Math.max(280, plot.clientWidth), H = 320, m = {t: 22, r: Math.min(120, Math.max(70, W * 0.2)), b: 40, l: 44}, n = D.bins.length;
    plot.replaceChildren(); const svg = el("svg", {viewBox: `0 0 ${W} ${H}`, tabindex: 0}, plot);
    const XHI = @@W_HI@@ + 0.25, x = v => m.l + (v + 4.25) / (XHI + 4.25) * (W - m.l - m.r), y = v => m.t + (1 - v) * (H - m.t - m.b);
    el("rect", {x: x(-4.25), y: m.t, width: x(0) - x(-4.25), height: H - m.t - m.b, fill: "var(--viz-wash)"}, svg);
    txt(svg, x(-4.1), m.t + 12, "inside the ABS zone", "note"); txt(svg, x(0.15), m.t + 12, "outside", "note");
    [0, 0.25, 0.5, 0.75, 1].forEach(v => { el("line", {x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), stroke: "var(--viz-grid)"}, svg); txt(svg, m.l - 8, y(v) + 4, Math.round(v * 100) + "%", "tick", "end"); });
    el("line", {x1: m.l, x2: W - m.r, y1: y(0), y2: y(0), stroke: "var(--viz-axis)"}, svg);
    [-4, -2, 0, 2, 4, 6, 8].filter(v => v <= @@W_HI@@).forEach(v => txt(svg, x(v), H - m.b + 18, (v > 0 ? "+" : "") + v + " in", "tick", "middle"));
    txt(svg, (m.l + W - m.r) / 2, H - 6, "true distance from the zone edge (ball edge to boundary)", "axis-label", "middle");
    D.bins.forEach(b => el("circle", {cx: x(b.x), cy: y(b.rate), r: (3 + Math.sqrt(b.n) / 14).toFixed(1), fill: S1, "fill-opacity": 0.35, stroke: "var(--surface)", "stroke-width": 2}, svg));
    D.curves.forEach((c, k) => { const pts = []; for (let i = 0; i <= (@@W_HI@@ + 4) * 10; i++){ const xv = -4 + i / 10; pts.push((i ? "L" : "M") + x(xv).toFixed(1) + "," + y(f(c, xv)).toFixed(1)); } el("path", {d: pts.join(""), fill: "none", stroke: colors[k], "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round"}, svg); txt(svg, x(@@W_HI@@) + 6, y(f(c, @@W_HI@@)) + 4, c.name, "dlabel"); });
    const cross = el("line", {y1: m.t, y2: H - m.b, stroke: "var(--viz-axis)", visibility: "hidden"}, svg);
    const mark = el("circle", {r: 5, fill: S1, stroke: "var(--surface)", "stroke-width": 2, visibility: "hidden"}, svg);
    const hit = el("rect", {x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent"}, svg);
    let idx = -1;
    const show = i => { idx = i; const b = D.bins[i]; cross.setAttribute("x1", x(b.x)); cross.setAttribute("x2", x(b.x)); cross.setAttribute("visibility", "visible"); mark.setAttribute("cx", x(b.x)); mark.setAttribute("cy", y(b.rate)); mark.setAttribute("visibility", "visible"); const r = svg.getBoundingClientRect(); showTip(r.left + x(b.x) * r.width / W, r.top + m.t, (b.x > 0 ? "+" : "") + b.x.toFixed(2) + " in from the edge", [{label: "Observed (" + b.n.toLocaleString() + " pitches)", value: (b.rate * 100).toFixed(1) + "%", color: S1}, ...D.curves.map((c, k) => ({label: c.name, value: (f(c, b.x) * 100).toFixed(1) + "%", color: colors[k]}))]); };
    const hide = () => { idx = -1; cross.setAttribute("visibility", "hidden"); mark.setAttribute("visibility", "hidden"); hideTip(); };
    hit.addEventListener("pointermove", e => { const r = svg.getBoundingClientRect(), gx = (e.clientX - r.left) * W / r.width; let b = 0, bd = 1e9; D.bins.forEach((q, i) => { const d = Math.abs(x(q.x) - gx); if (d < bd){ bd = d; b = i; } }); show(b); });
    hit.addEventListener("pointerleave", hide); keyNav(svg, n, () => idx, show, hide);
  });
})();

// 3. judgment vs chase: one series, fitted line as a reference, quadrant guides, nearest-point hover
(function(){
  const fig = document.getElementById("viz-scatter"), plot = fig.querySelector(".viz-plot"), D = DATA.charts.scatter;
  const P = D.points.slice().sort((a, b) => a.x - b.x), n = P.length;
  responsive(plot, () => {
    const W = Math.max(280, plot.clientWidth), H = 320, m = {t: 18, r: 18, b: 40, l: 44};
    plot.replaceChildren(); const svg = el("svg", {viewBox: `0 0 ${W} ${H}`, tabindex: 0}, plot);
    const xs = P.map(p => p.x), ys = P.map(p => p.y), xlo = Math.floor(Math.min(...xs) * 2) / 2, xhi = Math.ceil(Math.max(...xs) * 2) / 2, ylo = Math.floor(Math.min(...ys) * 20) / 20, yhi = Math.ceil(Math.max(...ys) * 20) / 20;
    const x = v => m.l + (v - xlo) / (xhi - xlo) * (W - m.l - m.r), y = v => m.t + (1 - (v - ylo) / (yhi - ylo)) * (H - m.t - m.b);
    ticks(ylo, yhi, 5).forEach(v => { el("line", {x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), stroke: "var(--viz-grid)"}, svg); txt(svg, m.l - 8, y(v) + 4, Math.round(v * 100) + "%", "tick", "end"); });
    ticks(xlo, xhi, 6).forEach(v => txt(svg, x(v), H - m.b + 18, (v > 0 ? "+" : "") + v, "tick", "middle"));
    txt(svg, (m.l + W - m.r) / 2, H - 6, "challenge judgment (z; higher = sharper eyes on takes)", "axis-label", "middle");
    el("line", {x1: x(D.x_cut), x2: x(D.x_cut), y1: m.t, y2: H - m.b, stroke: "var(--viz-axis)"}, svg);
    el("line", {x1: m.l, x2: W - m.r, y1: y(D.y_cut), y2: y(D.y_cut), stroke: "var(--viz-axis)"}, svg);
    txt(svg, x(D.x_cut) - 6, m.t + 12, "below-median eyes", "note", "end"); txt(svg, x(D.x_cut) + 6, m.t + 12, "above-median eyes", "note");
    txt(svg, W - m.r, y(D.y_cut) - 6, "top 40% chase rate above this line", "note", "end");
    el("line", {x1: x(xlo), y1: y(D.fit.m * xlo + D.fit.b), x2: x(xhi), y2: y(D.fit.m * xhi + D.fit.b), stroke: "var(--viz-muted)", "stroke-width": 2, "stroke-linecap": "round"}, svg);
    txt(svg, x(xhi) - 4, y(D.fit.m * xhi + D.fit.b) - 8, "fit, r = " + D.fit.r.toFixed(2), "note", "end");
    const dots = P.map(p => el("circle", {cx: x(p.x), cy: y(p.y), r: 4, fill: S1, "fill-opacity": 0.65, stroke: "var(--surface)", "stroke-width": 2}, svg));
    const hit = el("rect", {x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent"}, svg);
    let idx = -1;
    const show = i => { if (idx >= 0){ dots[idx].setAttribute("r", 4); dots[idx].setAttribute("fill-opacity", 0.65); } idx = i; const p = P[i]; dots[i].setAttribute("r", 6); dots[i].setAttribute("fill-opacity", 1); svg.appendChild(dots[i]); const r = svg.getBoundingClientRect(); showTip(r.left + x(p.x) * r.width / W, r.top + y(p.y) * r.height / H, p.name, [{label: "chase rate", value: (p.y * 100).toFixed(1) + "%"}, {label: "judgment z", value: (p.x > 0 ? "+" : "") + p.x.toFixed(2)}, {label: "challenges", value: String(p.chal)}, {label: "sort", value: p.label}]); };
    const hide = () => { if (idx >= 0){ dots[idx].setAttribute("r", 4); dots[idx].setAttribute("fill-opacity", 0.65); } idx = -1; hideTip(); };
    hit.addEventListener("pointermove", e => { const r = svg.getBoundingClientRect(), gx = (e.clientX - r.left) * W / r.width, gy = (e.clientY - r.top) * H / r.height; let b = -1, bd = 28 * 28; P.forEach((p, i) => { const d = (x(p.x) - gx) ** 2 + (y(p.y) - gy) ** 2; if (d < bd){ bd = d; b = i; } }); if (b < 0) hide(); else if (b !== idx) show(b); });
    hit.addEventListener("pointerleave", hide); keyNav(svg, n, () => idx, show, hide);
    svg.appendChild(hit);
  });
})();

// 4. policy simulation: one series, horizontal bars, value at the tip
(function(){
  const fig = document.getElementById("viz-bars"), plot = fig.querySelector(".viz-plot"), D = DATA.charts.bars;
  const names = {static: "Static rule (Savant breakeven)", dynamic: "Dynamic rule (this study)", observed: "Observed behavior", oracle: "Oracle (perfect eyes)"};
  responsive(plot, () => {
    const W = Math.max(280, plot.clientWidth), row = 34, m = {t: 8, r: 56, b: 28, l: Math.min(230, Math.max(120, W * 0.32))}, H = m.t + D.length * row + m.b, xMax = Math.ceil(Math.max(...D.map(d => d.runs)) * 1.08 / 20) * 20;
    plot.replaceChildren(); const svg = el("svg", {viewBox: `0 0 ${W} ${H}`, tabindex: 0}, plot);
    const x = v => m.l + v / xMax * (W - m.l - m.r);
    ticks(0, xMax, 6).forEach(v => { el("line", {x1: x(v), x2: x(v), y1: m.t, y2: H - m.b, stroke: "var(--viz-grid)"}, svg); txt(svg, x(v), H - m.b + 16, String(v), "tick", "middle"); });
    el("line", {x1: x(0), x2: x(0), y1: m.t, y2: H - m.b, stroke: "var(--viz-axis)"}, svg);
    const bars = D.map((d, i) => { const yc = m.t + i * row + row / 2, h = 22, x0 = x(0), x1 = Math.max(x(d.runs), x0 + 4), rr = 4; txt(svg, m.l - 10, yc + 4, names[d.rule] || d.rule, "tick", "end"); const b = el("path", {d: `M${x0},${yc - h / 2}H${x1 - rr}a${rr},${rr} 0 0 1 ${rr},${rr}v${h - 2 * rr}a${rr},${rr} 0 0 1 -${rr},${rr}H${x0}Z`, fill: S1}, svg); txt(svg, x1 + 8, yc + 4, d.runs.toFixed(1), "dlabel"); return b; });
    let idx = -1;
    const show = i => { if (idx >= 0) bars[idx].setAttribute("fill-opacity", 1); idx = i; bars[i].setAttribute("fill-opacity", 0.8); const r = svg.getBoundingClientRect(), d = D[i]; showTip(r.left + x(d.runs) * r.width / W, r.top + (m.t + i * row) * r.height / H, names[d.rule] || d.rule, [{label: "runs per team-season", value: d.runs.toFixed(1)}, {label: "challenges per team-game", value: d.chal.toFixed(2)}, {label: "overturned", value: d.ot == null ? "" : Math.round(d.ot * 100) + "%"}]); };
    const hide = () => { if (idx >= 0) bars[idx].setAttribute("fill-opacity", 1); idx = -1; hideTip(); };
    bars.forEach((b, i) => { b.addEventListener("pointermove", () => { if (idx !== i) show(i); }); b.addEventListener("pointerleave", hide); });
    keyNav(svg, D.length, () => idx, show, hide);
  });
})();
</script>
"""


def ordinal(n: int) -> str:
    return f"{n}{'th' if 11 <= n % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def spend_text(data: dict) -> dict:
    """The dugout's 'when to spend a token' guidance, worded from the token values."""
    f, g = data["facts"], data["guide"]
    pc = lambda v: f"{round(v * 100)}%"  # noqa: E731
    if g["never_above"]:
        verdict = ("Savant's breakeven rule prices every token at 0.20 runs. For a league-average eye that is more than a token "
                   "is worth at any point in the game, so the rule is too stingy throughout: challenge more readily than it says, "
                   "early as well as late, and more readily still as the game goes on.")
    elif g["always_above"]:
        verdict = ("Savant's breakeven rule prices every token at 0.20 runs, less than a token is worth at any point in the game, "
                   "so the rule is too free throughout: challenge less readily than it says.")
    else:
        verdict = (f"Savant's breakeven rule prices every token at 0.20 runs, which is too little before the {ordinal(g['cross_inning'])} "
                   "inning and too much after it: be stingier than the rule early and more aggressive late.")
    rules = [
        f"<b>Early, with both tokens, a {pc(g['p_early2'])} chance the call flips is enough on a typical pitch.</b> With the last token "
        f"early, about {pc(g['p_early1'])}. Savant's price asks for {pc(g['p_static_typ'])}. To an average eye that is a pitch that looks "
        f"about {g['in_early2']:.1f} inches off the plate with two tokens and {g['in_early1']:.1f} with one.",
        f"<b>Strike three and ball four are the big stakes.</b> On a 3-2 count with runners on a flip is worth half a run, and early with "
        f"both tokens a {pc(g['p_big_early2'])} chance is enough.",
        f"<b>In the 9th with your last token, about {pc(g['p_late1'])} is enough.</b> An unused token at the final out is worth nothing, "
        "and a team at zero gets one back if the game goes on.",
    ]
    if g["extras_tok1"] is not None:
        rules.append(f"<b>Extra innings give a team at zero one token back each inning.</b> A token in an extra inning is worth about "
                     f"{g['extras_tok1']:.2f} runs, the same as in the 9th, so the 9th-inning rule is the rule for the rest of the game.")
    if g["sharp_sigma"] is not None:
        rules.append(f"<b>These numbers are for league-average eyes.</b> Sharper eyes make a token worth more: with perceptual noise of "
                     f"{g['sharp_sigma']:.1f} inches instead of {f['sigma_league']:.1f}, the first token at first pitch is worth "
                     f"{g['sharp_tok1']:.2f} runs, above Savant's price, and early in the game the bar moves back toward Savant's.")
    flat = next((r for r in data["seeds"] if r["belief"] == "flat" and r["seed"] == f["belief"]["seed"]), None)
    revision = ("Revised 2026-10-23. An earlier version of this page valued the first token at "
                f"{flat['token1_first_pitch']:.2f} runs and said to be stingy early. That came from a belief model in which a challenger "
                "who felt 90% sure was right about half the time. With beliefs that know how often the umpire is right, the first "
                f"token is worth {f['token_first_start']:.2f} and the early advice reverses.") if flat else ""
    return {"verdict": verdict, "rules": "".join(f"<li><span>{r}</span></li>" for r in rules), "revision": revision}


def fmt_margin(v: float) -> str:
    return f"&gt;{MARGIN_CAP_IN:.0f}" if v > MARGIN_CAP_IN else f"{v:+.1f}"


def pm(v: float, se: float) -> str:
    return f"{v:+.1f} ± {se:.1f}"


def sgn(v: float) -> str:
    return f"{v:+.2f}".replace("-", "\u2212")


def ci(v: float, lo: float, hi: float) -> str:
    return f"{max(v, 0):.2f} [{lo:.2f}, {hi:.2f}]"


def eyes_text(data: dict) -> dict:
    """The wording that depends on whether the eyes score can support a label this season."""
    f = data["facts"]
    em = f["eyes_min"]
    rel = {int(r["min_challenges_per_half"]): r for r in f["rel_primary"]}
    top = rel[max(rel)]
    ten = rel.get(10, top)
    crit0 = rel[0]
    lab = data["label_counts"]
    n_eyes = lab.get("Eyes case", 0) + lab.get("Approach case", 0)
    crit_pills = (f'<span class="pill crit">{CRIT_LABELS[0]}</span> (spends tokens more freely than the median regular: aggressive on '
                  f'challenges as well as swings, which points to the approach) or <span class="pill crit2">{CRIT_LABELS[1]}</span> '
                  "(aggressive on swings only; whether the eyes are the reason stays open)")
    if em is None:
        rule = ("The sort uses chase rate and whichever judgment score the season can support. An eyes-or-approach call needs the eyes "
                f"score's reliability to clear {RELIABLE_FLOOR:.1f} at the low end of its interval, and this season it does not at any volume "
                f"tested: even among the {int(ten['n'])} hitters with {int(ten['min_challenges_per_half'])}+ challenges in each half it is "
                f"{ci(ten['slope'], ten['slope_lo'], ten['slope_hi'])}. So no hitter is called an eyes case or an approach case. A hitter in the "
                f"top 40% of chase rate is sorted on the criterion instead, which is reliable ({crit0['intercept']:.2f}): {crit_pills}. Hitters in "
                'the bottom 40% of chase rate are <span class="pill disc">Disciplined</span>.')
        callout = ("Read the eyes score as a range, not a rank: the band is the hitter's percentile among the league's hitters, plus or minus "
                   f"one posterior SD. Below {EYES_SHOW_MIN} challenges it is not shown. Willingness to challenge (Spends pct) is a stable trait "
                   f"within a season (reliability {crit0['intercept']:.2f}); accuracy is not yet measurable per hitter, so the sort below "
                   "makes no eyes call this season.")
        caveat = ("The eyes score is not reliable per hitter at any volume one season gives "
                  f"({ci(ten['slope'], ten['slope_lo'], ten['slope_hi'])} at {int(ten['min_challenges_per_half'])}+ challenges per half), "
                  "so it is shown as a range and drives no label.")
        note = "No hitter carries an eyes-or-approach call; the vertical guide is the median eyes score."
    else:
        rule = ("The sort uses chase rate and whichever judgment score the sample can support. A hitter in the top 40% of chase rate with "
                f"{em}+ challenges, the volume at which the eyes score's reliability clears {RELIABLE_FLOOR:.1f} at the low end of its interval, "
                'is an <span class="pill eyes">Eyes case</span> with a below-median eyes score and an <span class="pill approach">Approach case</span> '
                f"with an above-median one ({n_eyes} hitters). With fewer challenges he is sorted on the criterion, which is reliable: {crit_pills}. "
                'Hitters in the bottom 40% of chase rate are <span class="pill disc">Disciplined</span>.')
        callout = ("Read the eyes score as a range, not a rank: the band is the hitter's percentile among the league's hitters, plus or minus "
                   f"one posterior SD. Below {EYES_SHOW_MIN} challenges it is not shown, and below {em} it drives no label. Willingness to "
                   f"challenge (Spends pct) is a stable trait within a season (reliability {crit0['intercept']:.2f}).")
        caveat = f"The eyes score supports a per-hitter call only from {em} challenges; below that it is shown as a range and drives no label."
        note = f"Only hitters with {em}+ challenges carry an eyes-or-approach call."
    rel_note = (f"An eyes label needs the lower end of the interval at {RELIABLE_FLOOR:.1f} or more: "
                + ("no tested volume reaches it." if em is None else f"first reached at {em // 2} per half, so {em} a season."))
    return {"@@SORT_RULE@@": rule, "@@CARD_CALLOUT@@": callout, "@@EYES_CAVEAT@@": caveat, "@@SCATTER_NOTE@@": note, "@@REL_NOTE@@": rel_note}


def render(data: dict, label: str) -> str:
    f = data["facts"]
    B = f["belief"]
    html = TEMPLATE
    sp = spend_text(data)
    seeds_primary = [r for r in data["seeds"] if r["belief"] == B["belief"]]
    obs = next(r for r in data["sim"] if r["rule"] == "observed")
    sim_name = {"static": "Static (Savant breakeven)", "dynamic": "Dynamic (this study)", "observed": "Observed behaviour",
                "oracle": "Oracle (perfect eyes)"}
    sim_eyes = {"static": "simulated", "dynamic": "simulated", "observed": "real players", "oracle": "perfect"}
    sens = {(r["belief"], round(r["sigma"], 2)): r for r in data["sens"]}
    sigmas = sorted({round(r["sigma"], 2) for r in data["sens"]})
    league_sig = round(f["sigma_league"], 2)
    token_vals = [d["t1"] for d in data["charts"]["token"]]
    bars = data["charts"]["bars"]
    rep = {
        "@@SEASON@@": data["season"], "@@LABEL@@": label,
        "@@GAMES@@": f"{f['games']:,}", "@@CHALLENGES@@": f"{f['challenges']:,}",
        "@@TOK1@@": f"{f['token_first_start']:.2f}", "@@TOK2@@": f"{f['token_second_start']:.2f}", "@@TOK9@@": f"{f['token_first_9th']:.2f}",
        "@@SIGMA@@": f"{f['sigma_league']:.1f}", "@@NREF@@": str(f["n_ref"]),
        "@@N_CRIT_REF@@": str(sum(1 for r in data["hitters"] if r["pa"] >= 150 and r["criterion_z"] is not None)),
        "@@EYES_SHOW_MIN@@": str(EYES_SHOW_MIN), "@@CRIT_L0@@": CRIT_LABELS[0], "@@CRIT_L1@@": CRIT_LABELS[1],
        "@@W_LO@@": f"{f['decision']['lo']:g}", "@@W_HI@@": f"{f['decision']['hi']:g}",
        "@@FX_ABOVE@@": sgn(f['fixed'].get('dir_above', 0)), "@@FX_BELOW@@": sgn(f['fixed'].get('dir_below', 0)),
        "@@FX_CORNER@@": sgn(f['fixed'].get('dir_corner', 0)), "@@PRIOR_CORR@@": sgn(f['prior_corr']),
        "@@POST_CORR@@": sgn(f['post_corr_median']),
        "@@LR_SLOPES@@": f"{f['direction_test']['chi2_slopes']:.0f}", "@@LR_BIC@@": f"{f['direction_test']['bic_penalty_slopes']:.0f}",
        "@@Z_SD@@": f"{np.median([r['judgment_z_sd'] for r in data['hitters'] if r['judgment_z_sd'] is not None and r['chal'] >= EYES_SHOW_MIN]):.1f}",
        "@@OT_ALL@@": f"{f['overturn_all']*100:.1f}", "@@OT_BAT@@": f"{f['roles']['batter']['ot']*100:.0f}", "@@OT_CAT@@": f"{f['roles']['catcher']['ot']*100:.0f}",
        "@@UMP_WRONG@@": f"{f['decision']['ump_wrong_share']*100:.0f}", "@@DEC_RATE@@": f"{f['decision']['challenged']/f['decision']['n']*100:.1f}",
        "@@DEC_N@@": f"{f['decision']['n']:,}", "@@DEC_CHAL@@": f"{f['decision']['challenged']:,}",
        "@@SLOPE@@": f"{f['slope']:.2f}", "@@SLOPE_SD@@": f"{f['slope_sd']:.2f}", "@@CRIT_SD@@": f"{f['crit_sd']:.2f}",
        "@@FX_2K@@": f"{f['fixed']['two_strike']:.1f}", "@@FX_LATE@@": f"{f['fixed']['late']:.1f}",
        "@@N_TEST@@": str(next((r["n"] for r in data["eyes"] if r["outcome"] == "chase_whiff_rate"), "")),
        "@@TEAM_OPTS@@": "".join(f'<option value="{t}">{t}</option>' for t in data["teams"]),
        "@@DUGOUT_HEAD@@": "".join(f'<th class="num">{n}</th>' for n in data["stakes"]),
        "@@DUGOUT_ROWS@@": "".join(
            f"<tr><td>{r['band']}</td><td>{r['tokens']}</td>" + "".join(
                f"<td class=\"num\">{r['p_'+n]}% · {fmt_margin(r['in_'+n])} in</td>" for n in data["stakes"]) + "</tr>"
            for r in data["dugout"]),
        "@@LABEL_STATS@@": "".join(
            f'<div class="stat"><b>{data["label_counts"].get(k, 0)}</b><span>{k}</span></div>'
            for k in (("Eyes case", "Approach case") if f["eyes_min"] is not None else ()) + CRIT_LABELS
            + ("Average chaser", "Disciplined", "Not rated")),
        "@@REL_ROWS@@": "".join(
            f"<tr><td>{'none' if r['min_challenges_per_half']==0 else int(r['min_challenges_per_half'])}</td><td class=\"num\">{int(r['n'])}</td>"
            f"<td class=\"num\">{ci(r['slope'], r['slope_lo'], r['slope_hi'])}</td><td class=\"num\">{ci(r['intercept'], r['intercept_lo'], r['intercept_hi'])}</td>"
            f"<td class=\"num\">{ci(r['before_slope'], r['before_lo'], r['before_hi']) if r['before_slope'] == r['before_slope'] else ''}</td></tr>"
            for r in data["rel"]),
        **eyes_text(data),
        "@@SIM_ROWS@@": "".join(
            f"<tr><td>{sim_name.get(r['rule'], r['rule'])}</td><td>{sim_eyes.get(r['rule'], '')}</td><td class=\"num\">{r['challenges_per_team_game']:.2f}</td>"
            f"<td class=\"num\">{'' if pd.isna(r['overturn_rate']) else f'{r['overturn_rate']*100:.0f}%'}</td><td class=\"num\">{r['runs_per_team_162']:.1f}</td>"
            f"<td class=\"num\">{'' if r['rule'] == 'static' else pm(r['minus_static'], r['minus_static_se'])}</td>"
            f"<td class=\"num\">{'' if r['rule'] == 'observed' else pm(r['minus_observed'], r['minus_observed_se'])}</td></tr>"
            for r in data["sim"] if r["rule"] in sim_name),
        "@@SEED_LINE@@": ("Across noise draws " + ", ".join(str(int(r["seed"])) for r in seeds_primary) + ", dynamic minus static was "
                          + ", ".join(pm(r["dyn_minus_static"], r["dyn_minus_static_se"]) for r in seeds_primary)
                          + " and dynamic minus observed " + ", ".join(pm(r["dyn_minus_observed"], r["dyn_minus_observed_se"]) for r in seeds_primary) + "."),
        "@@BELIEF_ROWS@@": "".join(
            f"<tr><td>{'Calibrated (this page)' if r['belief'] == 'calibrated' else 'Flat prior (before 2026-10-23)'}</td>"
            f"<td class=\"num\">{r['token1_first_pitch']:.3f}</td>"
            f"<td class=\"num\">{r['static_chal']:.2f} / {r['static_ot']*100:.0f}% / {r['static_runs']:.1f}</td>"
            f"<td class=\"num\">{r['dynamic_chal']:.2f} / {r['dynamic_ot']*100:.0f}% / {r['dynamic_runs']:.1f}</td>"
            f"<td class=\"num\">{pm(r['dyn_minus_static'], r['dyn_minus_static_se'])}</td><td class=\"num\">{pm(r['dyn_minus_observed'], r['dyn_minus_observed_se'])}</td></tr>"
            for r in data["seeds"] if r["seed"] == B["seed"]),
        "@@SENS_ROWS@@": "".join(
            (lambda c, fl: f"<tr><td class=\"num\">{sg:.2f}{' *' if sg == league_sig else ''}</td><td class=\"num\">{c['token1_first_pitch']:.3f}</td>"
             f"<td class=\"num\">{c['static_runs']:.1f}</td><td class=\"num\">{c['dynamic_runs']:.1f}</td><td class=\"num\">{pm(c['dyn_minus_static'], c['dyn_minus_static_se'])}</td>"
             f"<td class=\"num\">{fl['token1_first_pitch']:.3f}</td><td class=\"num\">{pm(fl['dyn_minus_static'], fl['dyn_minus_static_se'])}</td></tr>")(
                sens[("calibrated", sg)], sens[("flat", sg)])
            for sg in sigmas),
        "@@SPEND_VERDICT@@": sp["verdict"], "@@SPEND_RULES@@": sp["rules"], "@@SPEND_REVISION@@": sp["revision"],
        "@@TOKEN_ARIA@@": (f"Line chart of the marginal value of the first and second token across 18 half-innings, the first falling from "
                           f"{token_vals[0]:.2f} to {token_vals[-1]:.2f} runs, with a flat reference at Savant's 0.20 runs"),
        "@@BARS_ARIA@@": "Horizontal bars of runs per team-season: " + "; ".join(f"{sim_name.get(b['rule'], b['rule'])} {b['runs']:.1f}" for b in bars),
        "@@UMP_RIGHT@@": f"{(1 - B['umpire_wrong_share']) * 100:.0f}",
        "@@BELIEF_A@@": f"{B['a']:.2f}" if B["a"] is not None else "", "@@BELIEF_B@@": f"{B['b']:.2f}" if B["b"] is not None else "",
        "@@OBS_OT@@": f"{obs['overturn_rate']*100:.0f}", "@@OBS_CHAL@@": f"{obs['challenges_per_team_game']:.2f}",
        "@@OBS_WON@@": f"{int(obs['overturned']):,}",
        "@@N_P06@@": f"{B['n_p_above_0_6']:,}", "@@N_OPP@@": f"{B['n_opportunities']:,}",
        "@@CAL_GAP@@": f"{max(abs(q['mean_p'] - q['flip_rate']) for q in B['calibration_quintiles']) * 100:.1f}",
        "@@SHARP_SIGMA@@": f"{data['guide']['cross_sigma']:.1f}" if data["guide"]["cross_sigma"] is not None else "the lowest level tried",
        "@@SEED@@": str(B["seed"]), "@@LAMBDA@@": f"{B['lambda_per_team_half']:.2f}", "@@N_EXTRAS@@": f"{B['n_extras_games']:,}",
        "@@RHO@@": f"{B['rho_extra_inning']:.2f}", "@@CAP@@": f"{MARGIN_CAP_IN:.0f}",
        "@@R_CHASE@@": f"{f['r_chase']:+.2f}", "@@N_R_CHASE@@": str(f["n_r_chase"]),
        "@@EYES_ROWS@@": "".join(
            f"<tr><td>{r['label']}</td><td class=\"num\">{int(r['n'])}</td><td class=\"num\">{r['beta_judgment']:+.4f}</td><td class=\"num\">{r['se_hc1']:.4f}</td><td class=\"num\">{r['p_hc1']:.2f}</td></tr>"
            for r in data["eyes"]),
        "@@DATA@@": json.dumps({"hitters": data["hitters"], "charts": data["charts"]}, separators=(",", ":"), allow_nan=False),
    }
    for k, v in rep.items():
        html = html.replace(k, v)
    assert "@@" not in html, [s for s in html.split("@@")[1::2]][:5]
    check_rendered(html)
    return html


def check_rendered(html: str) -> None:
    """No missing value may reach the page: no nan, None or undefined outside script code."""
    text = re.sub(r"<script\b.*?</script>", "", html, flags=re.S)
    bad = [m.group(0) for m in re.finditer(r"\b(?:[Nn]a[Nn]|None|undefined)\b", text)]
    assert not bad, f"rendered brief contains {sorted(set(bad))}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="mlb_2026")
    a = ap.parse_args()
    REPORT.mkdir(exist_ok=True)
    data = build_data(a.label)
    body = render(data, a.label)
    season = data["season"]
    (REPORT / f"club_brief_{season}.html").write_text(body, encoding="utf-8")
    standalone = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
                  '<meta name="viewport" content="width=device-width, initial-scale=1"></head><body>' + body + "</body></html>")
    (REPORT / f"club_brief_{season}_standalone.html").write_text(standalone, encoding="utf-8")
    print(f"wrote report/club_brief_{season}.html ({len(body)/1024:.0f} KB) and the standalone copy; "
          f"{len(data['hitters'])} hitters, {len(data['teams'])} teams; labels {data['label_counts']}")


if __name__ == "__main__":
    main()
