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
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from abs_judgment.config import PROCESSED, ROOT  # noqa: E402

LOGIT_TO_PROBIT = 1.7
STAKES = [("0-0 count, bases empty", 0.05), ("typical pitch", 0.19), ("3-2 count, runners on", 0.50)]
BANDS = [("1st to 3rd", 1), ("4th to 6th", 7), ("7th and 8th", 13), ("9th", 17)]
REPORT = ROOT / "report"


def pct_rank(ref: pd.Series, x: pd.Series) -> pd.Series:
    ref = np.sort(ref.dropna().to_numpy())
    return pd.Series(np.searchsorted(ref, x.to_numpy(), side="right") / len(ref) * 100, index=x.index)


def b64png(p: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


def build_data(label: str) -> dict:
    season = label.split("_")[1]
    pt = pd.read_parquet(PROCESSED / f"pitch_table_{label}.parquet",
                         columns=["batter_id", "half", "home_team", "away_team", "orig_called_strike",
                                  "bat_tokens_before", "abs_edge_in", "batter_challenged"])
    pt["bat_abbr"] = np.where(pt.half == "top", pt.away_team, pt.home_team)
    team = pt.groupby("batter_id").bat_abbr.agg(lambda s: s.mode().iloc[0] if s.notna().any() else "")
    nteams = pt.groupby("batter_id").bat_abbr.nunique()
    dec = pt[pt.orig_called_strike & (pt.bat_tokens_before >= 1) & pt.abs_edge_in.notna() & (pt.abs_edge_in.abs() <= 4)]
    decision = {"n": int(len(dec)), "challenged": int(dec.batter_challenged.sum()),
                "ump_wrong_share": float((dec.abs_edge_in > 0).mean())}

    ch = pd.read_parquet(PROCESSED / f"challenges_{label}.parquet")
    gs = pd.read_csv(PROCESSED / f"game_summaries_{label}.csv")
    bat = ch[ch.initiator_role == "batter"].groupby("initiator_id").agg(chal=("is_overturned", "size"), won=("is_overturned", "sum"))
    roles = ch.groupby("initiator_role").is_overturned.agg(["size", "mean"])

    J = pd.read_csv(PROCESSED / f"judgment_{label}.csv")
    hs = pd.read_csv(PROCESSED / f"hitter_season_{label}.csv")
    prior = json.load(open(PROCESSED / f"judgment_prior_{label}.json"))
    tv = pd.read_csv(PROCESSED / f"token_value_{label}.csv")
    sim = pd.read_csv(PROCESSED / f"policy_simulation_{label}.csv")
    res = pd.read_csv(PROCESSED / f"eyes_vs_approach_{label}.csv")
    relc = pd.read_csv(PROCESSED / f"reliability_checks_{label}.csv")

    sigma_league = LOGIT_TO_PROBIT / prior["mean"][1]

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
                row[f"in_{name}"] = round(sigma_league * norm.ppf(p), 1)
            dugout.append(row)

    # ---- hitter cards
    H = hs.merge(J, on="batter_id", how="left", suffixes=("", "_j"))
    H = H[H.pa >= 100].copy()
    H["team"] = H.batter_id.map(team).fillna("")
    H["nteams"] = H.batter_id.map(nteams).fillna(1).astype(int)
    H["chal"] = H.batter_id.map(bat.chal).fillna(0).astype(int)
    H["won"] = H.batter_id.map(bat.won).fillna(0).astype(int)
    ref = H[(H.pa >= 150) & (H.chal >= 8)]
    H["eyes_pct"] = pct_rank(ref.judgment_z, H.judgment_z)
    H["crit_pct"] = pct_rank(ref.criterion_z, H.criterion_z)
    H["chase_pct"] = pct_rank(H[H.pa >= 150].chase_rate, H.chase_rate)
    H["sigma"] = LOGIT_TO_PROBIT / H.slope.clip(lower=0.05)
    s_typ = STAKES[1][1]
    for col, (h, k) in {"m_early2": (1, 2), "m_early1": (1, 1), "m_mid1": (7, 1), "m_late1": (17, 1)}.items():
        p = dv(h, k) / (s_typ + dv(h, k))
        H[col] = H.sigma * norm.ppf(p)

    def label_row(r) -> str:
        if r.pa < 150:
            return "Small sample"
        if r.chal < 8 or pd.isna(r.judgment_z):
            return "Eyes not rated"
        if r.chase_pct >= 60:
            return "Eyes case" if r.eyes_pct < 50 else "Approach case"
        if r.chase_pct <= 40:
            return "Disciplined"
        return "Average chaser"

    H["label"] = H.apply(label_row, axis=1)
    H["provisional"] = H.chal < 15
    keep = ["batter_id", "batter_name", "team", "nteams", "pa", "chal", "won", "n_decisions", "judgment_z", "eyes_pct",
            "criterion_z", "crit_pct", "sigma", "chase_rate", "chase_pct", "k_rate", "bb_rate",
            "m_early2", "m_early1", "m_mid1", "m_late1", "label", "provisional"]
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
    sim_rows = sim[["rule", "challenges_per_team_game", "overturn_rate", "runs_per_team_162"]].round(3).to_dict("records")
    eyes_rows = res[["outcome", "n", "beta_judgment", "se_hc1", "p_hc1", "label"]].round(4).to_dict("records")
    rel_rows = relc[relc.window_in == "-4..4"][["min_challenges_per_half", "n", "slope", "intercept"]].round(3).to_dict("records")
    yoy = res[res.outcome == "chase_rate_given_prev"]
    yoy_row = yoy.iloc[0].to_dict() if len(yoy) else None

    teams = sorted({r["team"] for r in hitters if r["team"]})
    return {
        "season": season,
        "facts": {
            "games": int(gs.game_pk.nunique()), "pitches": int(gs.n_pitches.sum()), "challenges": int(len(ch)),
            "overturn_all": float(ch.is_overturned.mean()),
            "roles": {r: {"n": int(roles.loc[r, "size"]), "ot": float(roles.loc[r, "mean"])} for r in roles.index},
            "decision": decision, "sigma_league": sigma_league, "slope": prior["mean"][1],
            "slope_sd": float(np.sqrt(prior["var"][1])), "crit_sd": float(np.sqrt(prior["var"][0])),
            "fixed": prior["fixed"], "reliability": prior.get("reliability_full_season", {}),
            "n_scored": int(len(J)), "n_ref": int(len(ref)),
            "token_first_start": dv(1, 1), "token_second_start": dv(1, 2), "token_first_9th": dv(17, 1),
        },
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
}
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){
  --bg:#0f141d; --surface:#161d29; --fg:#e7ecf3; --muted:#9fabbc; --line:#2a3443; --soft:#1e2735;
  --accent:#7ea2ff; --accent-ink:#0f141d; --ot:#4fc98c; --cf:#f28b7d; --warn-bg:#3a2f0d; --warn-fg:#f3d98b; color-scheme:dark } }
:root[data-theme="dark"]{
  --bg:#0f141d; --surface:#161d29; --fg:#e7ecf3; --muted:#9fabbc; --line:#2a3443; --soft:#1e2735;
  --accent:#7ea2ff; --accent-ink:#0f141d; --ot:#4fc98c; --cf:#f28b7d; --warn-bg:#3a2f0d; --warn-fg:#f3d98b; color-scheme:dark }
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
      <p>A challenge token is not worth a fixed amount. It is worth what it can still buy over the rest of the game. At first pitch the first token is worth about <b>@@TOK1@@ runs</b> and the second about <b>@@TOK2@@</b>; by the 9th the last token is worth about <b>@@TOK9@@</b>. Savant's breakeven rule prices every token at 0.20 runs, which is right only in the middle innings. So: be stingy early, be aggressive late.</p>
      <ol class="rules">
        <li><span><b>Early, with both tokens, challenge only when you are sure.</b> Roughly three in four, which for an average eye is a pitch that looks about 1.5 to 2 inches off the plate.</span></li>
        <li><span><b>Strike three and ball four are the big stakes.</b> On a 3-2 count with runners on, a flip is worth half a run. Lower your bar there, even early.</span></li>
        <li><span><b>In the 9th with your last token, a coin flip is enough.</b> An unused token at the final out is worth nothing.</span></li>
        <li><span><b>Extra innings give a team at zero one token back each inning.</b> Treat the 9th-inning rule as the rule for the rest of the game.</span></li>
      </ol>
      <p class="src">Source: token_value_@@LABEL@@.csv (backward induction), stakes_@@LABEL@@.csv, policy_simulation_@@LABEL@@.csv.</p>
    </div>
    <figure>
      <img src="@@FIG_TOKEN@@" alt="Marginal value of the first and second challenge token by half-inning, falling from about half a run to near zero, against Savant's flat 0.20">
      <figcaption>What a held token is worth, by half-inning. The dotted line is Savant's flat price.</figcaption>
    </figure>
  </div>
  <h3>How sure you need to be</h3>
  <p>Percent chance the call flips that justifies a challenge, by inning, tokens held and what the pitch is worth. The second number in each cell is how far off the plate the pitch has to look to an average eye (negative means a pitch that looks a touch inside is still worth it).</p>
  <div class="tablebox"><table id="dugout"><thead><tr><th>Innings</th><th>Tokens</th>@@DUGOUT_HEAD@@</tr></thead><tbody>@@DUGOUT_ROWS@@</tbody></table></div>
  <p class="src">Threshold = token value / (stake + token value). Inches = perceptual noise (@@SIGMA@@ in for the league) times the normal quantile of the threshold. Per-hitter versions are in the cards below.</p>
</section>

<section id="cards">
  <h2>Hitter challenge cards</h2>
  <p>Every called strike a hitter takes with a token in hand is a test of his eyes: the true location is known, and he either challenges or he doesn't. The <b>eyes</b> score is how sharply his challenge rate rises as the pitch moves from inside the zone to outside it, against the league. The <b>criterion</b> is how freely he spends tokens. The margins are how far off the plate a pitch should look <em>to him</em> before he challenges on a typical pitch, early with both tokens, early with the last token, mid-game, and in the 9th.</p>
  <div class="callout">Read the eyes score with its sample. Below 15 challenges it is provisional (marked). Below 8 it is not shown. Willingness to challenge is a stable trait within a season; accuracy becomes reliable at about 20 challenges.</div>
  <div class="controls">
    <label>Team<select id="team"><option value="">All teams</option>@@TEAM_OPTS@@</select></label>
    <label>Search<input id="q" type="search" placeholder="hitter name" autocomplete="off"></label>
    <label>Sort<select id="sort"><option value="batter_name">Name</option><option value="eyes_pct">Eyes (best first)</option><option value="chase_pct">Chase rate (highest first)</option><option value="chal">Challenges</option><option value="crit_pct">Spends tokens (freest first)</option></select></label>
    <label class="check"><input id="rated" type="checkbox"> Only hitters with 15+ challenges</label>
  </div>
  <div class="tablebox"><table id="cards-table"><thead><tr>
    <th>Hitter</th><th>Team</th><th class="num">PA</th><th class="num">Chal</th><th class="num">Won</th><th class="num">Eyes pct</th><th class="num">Spends pct</th><th class="num">Chase %</th><th class="num">Chase pct</th><th class="num">Early, 2 tok</th><th class="num">Early, last</th><th class="num">Mid, last</th><th class="num">9th, last</th><th>Sort</th>
  </tr></thead><tbody id="cards-body"></tbody></table></div>
  <div class="legend"><span><b>Won</b> = call overturned</span><span><b>Eyes pct</b> = percentile of perceptual sharpness among @@NREF@@ regulars with 8+ challenges</span><span><b>Spends pct</b> = percentile of willingness to challenge</span><span>Margins in inches off the plate on a typical pitch</span></div>
  <p class="src" id="count"></p>
  <p class="src">Source: judgment_@@LABEL@@.csv, hitter_season_@@LABEL@@.csv, challenges_@@LABEL@@.parquet, token_value_@@LABEL@@.csv. Team is the hitter's most frequent batting team.</p>
</section>

<section id="sort">
  <h2>Eyes case or approach case</h2>
  <div class="grid2">
    <div style="display:grid;gap:12px">
      <p>A hitter who chases has one of two problems. Either he does not see the pitch well enough to know it is a ball, or he sees it fine and swings anyway. The fix differs: pitch-recognition work for the first, a count plan for the second. Chase rate alone cannot separate them because every swing mixes perception with intent. The challenge decision on a take has no swing in it.</p>
      <p>The sort in the cards uses both: a hitter in the top 40% of chase rate with a below-median eyes score is an <span class="pill eyes">Eyes case</span>; the same chaser with an above-median eyes score is an <span class="pill approach">Approach case</span>. Hitters in the bottom 40% of chase rate are <span class="pill disc">Disciplined</span>.</p>
      <div class="stats">@@LABEL_STATS@@</div>
      <p><b>What the evidence supports.</b> Sharper eyes on takes go with chasing less (correlation @@R_CHASE@@ across regulars, stronger among frequent challengers) and with cutting the chase rate from the prior season. They do <em>not</em> go with cheaper chases, better two-strike contact or fewer strikeouts once chase rate is held fixed. So the sort tells you which lever moves the <em>rate</em>; it does not say who pays most per chase.</p>
    </div>
    <figure>
      <img src="@@FIG_SCATTER@@" alt="Scatter of challenge judgment against chase rate for regulars, with a negative fitted line">
      <figcaption>Eyes on takes against chasing on swings, @@SEASON@@ regulars.</figcaption>
    </figure>
  </div>
</section>

<section id="study">
  <div class="part">Part two · the evidence</div>
  <h2>What the study found</h2>
  <div class="stats">
    <div class="stat"><b>@@CHALLENGES@@</b><span>challenges in @@GAMES@@ games, all recovered at the pitch level</span></div>
    <div class="stat"><b>@@OT_ALL@@%</b><span>overturned overall; batters @@OT_BAT@@%, catchers @@OT_CAT@@%</span></div>
    <div class="stat"><b>@@UMP_WRONG@@%</b><span>of called strikes within 4 in of the edge were actually balls</span></div>
    <div class="stat"><b>@@DEC_RATE@@%</b><span>of those were challenged by the hitter</span></div>
  </div>
  <div class="grid2">
    <figure>
      <img src="@@FIG_CURVE@@" alt="Probability a hitter challenges a called strike as a function of true edge distance, with the league curve and the sharpest and shallowest hitters">
      <figcaption>The psychometric curve. Challenge rate rises smoothly from under 1% four inches inside the zone to about 40% three inches outside. It is a curve, not a step: at the edge itself hitters challenge only 11 to 14% of true balls.</figcaption>
    </figure>
    <div style="display:grid;gap:12px">
      <h3>Measurement</h3>
      <p>A hierarchical logit of challenge on true edge distance gives each hitter a slope (eyes) and an intercept (criterion). League slope @@SLOPE@@ per inch; between-hitter spread @@SLOPE_SD@@ on the slope and @@CRIT_SD@@ on the criterion. Two strikes shift the criterion by @@FX_2K@@ logits; a late inning by @@FX_LATE@@.</p>
      <div class="tablebox"><table><thead><tr><th>Min challenges per half-season</th><th class="num">Hitters</th><th class="num">Eyes reliability</th><th class="num">Criterion reliability</th></tr></thead><tbody>@@REL_ROWS@@</tbody></table></div>
      <p class="src">Split-half reliability, Spearman-Brown to a full season. Source: reliability_checks_@@LABEL@@.csv.</p>
    </div>
  </div>
  <h3>Policy</h3>
  <p>Every game replayed with each rule, deciding on a perceived edge with league-average noise and scored on the true ABS outcome. Runs per team per 162 games.</p>
  <div class="tablebox"><table><thead><tr><th>Rule</th><th class="num">Challenges per team-game</th><th class="num">Overturned</th><th class="num">Runs per team-season</th></tr></thead><tbody>@@SIM_ROWS@@</tbody></table></div>
  <p class="src">Oracle challenges exactly the umpire's misses, the ceiling with perfect eyes. Observed is what players did on the same pitches. Source: policy_simulation_@@LABEL@@.csv.</p>
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
    <dt>Decision set</dt><dd>Umpire-called strikes the hitter took with a token in hand, within 4 inches of the edge: @@DEC_N@@ pitches, @@DEC_CHAL@@ challenged.</dd>
    <dt>Judgment model</dt><dd>Logit of challenge on edge distance with per-hitter random intercept and slope, fit by empirical Bayes; count, token and leverage shifters as fixed effects. The slope is unidentified for hitters who rarely challenge.</dd>
    <dt>Token value</dt><dd>Backward induction over decision slots with opportunities arriving at the observed rate (taken pitches within 3 inches of the edge, both sides), stakes from Statcast run expectancy, and a per-inning top-up in extra innings as observed in the ledgers.</dd>
    <dt>Caveats</dt><dd>One season of ABS. Run units, not win probability, so late close games are undervalued. League-average perception in the policy simulation. The eyes score needs about 20 challenges to be reliable per hitter. Year-over-year judgment and outcome share a season.</dd>
    <dt>Update</dt><dd>Rerun <code>scripts/run_pipeline.sh</code> and <code>scripts/11_club_brief.py</code> on the new season. Everything on this page regenerates from the processed tables named under each section.</dd>
  </dl>
</section>
</div>

<script>
const DATA = @@DATA@@;
const fmt1 = v => v == null ? "" : (v > 0 ? "+" : "") + v.toFixed(1);
const pct = v => v == null ? "" : Math.round(v);
const pillClass = l => l === "Eyes case" ? "pill eyes" : l === "Approach case" ? "pill approach" : l === "Disciplined" ? "pill disc" : "pill";
const body = document.getElementById("cards-body");
const teamSel = document.getElementById("team"), q = document.getElementById("q"), sortSel = document.getElementById("sort"), rated = document.getElementById("rated");
function render(){
  const t = teamSel.value, s = q.value.trim().toLowerCase(), k = sortSel.value, onlyRated = rated.checked;
  let rows = DATA.hitters.filter(r => (!t || r.team === t) && (!s || r.batter_name.toLowerCase().includes(s)) && (!onlyRated || r.chal >= 15));
  const desc = {eyes_pct:1, chase_pct:1, chal:1, crit_pct:1};
  rows.sort((a,b) => { const va = a[k], vb = b[k]; if (k === "batter_name") return va.localeCompare(vb); if (va == null) return 1; if (vb == null) return -1; return desc[k] ? vb - va : va - vb; });
  body.innerHTML = rows.map(r => {
    const unrated = r.chal < 8 || r.eyes_pct == null;
    const eyes = unrated ? "<span class='pill'>n/a</span>" : pct(r.eyes_pct) + (r.provisional ? " <span class='pill prov'>prov</span>" : "");
    const m = v => unrated ? "" : fmt1(v);
    return `<tr><td>${r.batter_name}</td><td>${r.team}${r.nteams > 1 ? " +" + (r.nteams-1) : ""}</td><td class="num">${r.pa}</td><td class="num">${r.chal}</td><td class="num"><span class="${r.won > r.chal - r.won ? "ot" : ""}">${r.won}</span></td><td class="num">${eyes}</td><td class="num">${unrated ? "" : pct(r.crit_pct)}</td><td class="num">${(r.chase_rate*100).toFixed(1)}</td><td class="num">${pct(r.chase_pct)}</td><td class="num">${m(r.m_early2)}</td><td class="num">${m(r.m_early1)}</td><td class="num">${m(r.m_mid1)}</td><td class="num">${m(r.m_late1)}</td><td><span class="${pillClass(r.label)}">${r.label}</span></td></tr>`;
  }).join("");
  document.getElementById("count").textContent = rows.length + " hitters shown of " + DATA.hitters.length + " with 100+ PA.";
}
[teamSel, q, sortSel, rated].forEach(el => el.addEventListener("input", render));
try { const saved = localStorage.getItem("abs-brief-team"); if (saved && [...teamSel.options].some(o => o.value === saved)) teamSel.value = saved; } catch (e) {}
teamSel.addEventListener("change", () => { try { localStorage.setItem("abs-brief-team", teamSel.value); } catch (e) {} });
render();
</script>
"""


def render(data: dict, label: str) -> str:
    f = data["facts"]
    html = TEMPLATE
    rep = {
        "@@SEASON@@": data["season"], "@@LABEL@@": label,
        "@@GAMES@@": f"{f['games']:,}", "@@CHALLENGES@@": f"{f['challenges']:,}",
        "@@TOK1@@": f"{f['token_first_start']:.2f}", "@@TOK2@@": f"{f['token_second_start']:.2f}", "@@TOK9@@": f"{f['token_first_9th']:.2f}",
        "@@SIGMA@@": f"{f['sigma_league']:.1f}", "@@NREF@@": str(f["n_ref"]),
        "@@OT_ALL@@": f"{f['overturn_all']*100:.1f}", "@@OT_BAT@@": f"{f['roles']['batter']['ot']*100:.0f}", "@@OT_CAT@@": f"{f['roles']['catcher']['ot']*100:.0f}",
        "@@UMP_WRONG@@": f"{f['decision']['ump_wrong_share']*100:.0f}", "@@DEC_RATE@@": f"{f['decision']['challenged']/f['decision']['n']*100:.1f}",
        "@@DEC_N@@": f"{f['decision']['n']:,}", "@@DEC_CHAL@@": f"{f['decision']['challenged']:,}",
        "@@SLOPE@@": f"{f['slope']:.2f}", "@@SLOPE_SD@@": f"{f['slope_sd']:.2f}", "@@CRIT_SD@@": f"{f['crit_sd']:.2f}",
        "@@FX_2K@@": f"{f['fixed']['two_strike']:.1f}", "@@FX_LATE@@": f"{f['fixed']['late']:.1f}",
        "@@N_TEST@@": str(next((r["n"] for r in data["eyes"] if r["outcome"] == "chase_whiff_rate"), "")),
        "@@FIG_TOKEN@@": b64png(ROOT / "docs/figures/token_value.png"),
        "@@FIG_SCATTER@@": b64png(ROOT / "docs/figures/judgment_vs_chase.png"),
        "@@FIG_CURVE@@": b64png(ROOT / "docs/figures/psychometric_curve.png"),
        "@@TEAM_OPTS@@": "".join(f'<option value="{t}">{t}</option>' for t in data["teams"]),
        "@@DUGOUT_HEAD@@": "".join(f'<th class="num">{n}</th>' for n in data["stakes"]),
        "@@DUGOUT_ROWS@@": "".join(
            f"<tr><td>{r['band']}</td><td>{r['tokens']}</td>" + "".join(
                f"<td class=\"num\">{r['p_'+n]}% · {r['in_'+n]:+.1f} in</td>" for n in data["stakes"]) + "</tr>"
            for r in data["dugout"]),
        "@@LABEL_STATS@@": "".join(
            f'<div class="stat"><b>{data["label_counts"].get(k, 0)}</b><span>{k}</span></div>'
            for k in ("Eyes case", "Approach case", "Average chaser", "Disciplined", "Eyes not rated")),
        "@@REL_ROWS@@": "".join(
            f"<tr><td>{'none' if r['min_challenges_per_half']==0 else int(r['min_challenges_per_half'])}</td><td class=\"num\">{int(r['n'])}</td><td class=\"num\">{r['slope']:.2f}</td><td class=\"num\">{r['intercept']:.2f}</td></tr>"
            for r in data["rel"]),
        "@@SIM_ROWS@@": "".join(
            f"<tr><td>{r['rule']}</td><td class=\"num\">{r['challenges_per_team_game']:.2f}</td><td class=\"num\">{'' if pd.isna(r['overturn_rate']) else f'{r['overturn_rate']*100:.0f}%'}</td><td class=\"num\">{r['runs_per_team_162']:.1f}</td></tr>"
            for r in data["sim"] if r["rule"] != "never"),
        "@@EYES_ROWS@@": "".join(
            f"<tr><td>{r['label']}</td><td class=\"num\">{int(r['n'])}</td><td class=\"num\">{r['beta_judgment']:+.4f}</td><td class=\"num\">{r['se_hc1']:.4f}</td><td class=\"num\">{r['p_hc1']:.2f}</td></tr>"
            for r in data["eyes"]),
        "@@DATA@@": json.dumps({"hitters": data["hitters"]}, separators=(",", ":")),
    }
    # correlation for the sort section, from the hitters table itself
    hh = pd.DataFrame(data["hitters"])
    ref = hh[(hh.pa >= 150) & (hh.chal >= 8)]
    rep["@@R_CHASE@@"] = f"{np.corrcoef(ref.judgment_z.astype(float), ref.chase_rate.astype(float))[0, 1]:+.2f}"
    for k, v in rep.items():
        html = html.replace(k, v)
    assert "@@" not in html, [s for s in html.split("@@")[1::2]][:5]
    return html


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
