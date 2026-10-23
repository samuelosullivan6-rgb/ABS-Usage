# ABS challenge judgment

A public measurement of a hitter's eyes, stripped of intent, built from the 2026 MLB Automated Ball-Strike challenge system. Two deliverables:

1. **A dynamic challenge policy.** The value of a held token priced by backward induction over the game, replacing Baseball Savant's constant 0.2-run price, with a per-player layer that turns each hitter's measured perception into a perceived-margin rule.
2. **The eyes-versus-approach test.** A hitter's chases are either a perception problem (pitch-recognition training) or an approach problem (a count plan). Challenge judgment on takes is the only public signal of perception with no swing in it. The test asks whether, among hitters with similar chase rates, poor location-adjusted challenge judgment predicts costlier chases, worse two-strike contact, and weaker year-over-year chase improvement. Judgment enters as a covariate pooled across hitters, so the per-hitter estimate need only be unbiased, not precise.

The decision link for the second is the hitting coordinator's development plan. The design is in [docs/research_design.md](docs/research_design.md). The data verification is in [docs/data_findings_2026-10-21.md](docs/data_findings_2026-10-21.md).

## What the data turned out to support

Everything below was verified against live endpoints on 2026-10-21.

- **Every ABS challenge is public at the pitch level.** The MLB StatsAPI live feed carries `reviewDetails` with review type `MJ`, the initiating player, and the outcome, on the pitch event for mid-plate-appearance challenges and on the play for PA-ending ones. The full 2026 regular season yields 10,557 challenges across 2,429 games, matching Savant's published total, and the per-team counts reconcile with the official in-game token ledger in all but one game.
- **The feed's strike-zone bounds are the ABS zone.** In 2026 `strikeZoneTop` and `strikeZoneBottom` are fixed per batter at 53.5% and 27% of measured height.
- **Mid-plate location is exact.** Extrapolating the feed's nine-parameter trajectory to the plate midpoint reproduces Savant's challenge coordinates to five decimals, and the resulting edge distance (ball edge to zone boundary, 2.9-inch ball) agrees with the ABS ruling on 99.99% of challenged pitches. This removes the front-of-plate caveat that public work had assumed.
- **The extra-innings rule, from the data.** 74 challenges were made by teams whose two regular tokens were gone, all in innings 10 to 13. The rule that explains every one is a top-up to one token at the start of each extra inning for a team at zero. The ledger's `remaining` counts only the regular tokens.
- **Savant's leaderboard exports CSV** (`?csv=true`) for batters, catchers, pitchers and teams, 2026 MLB and 2025 Triple-A. Its pitch-level table covers only the trailing seven days, so the season table comes from StatsAPI.

## Pipeline

```
scripts/01_pull_statsapi.py --season 2026 --sport 1   # 2,430 game feeds, cached gzip
scripts/02_pull_statcast.py --season 2026             # daily Statcast CSVs (run values, bat tracking)
scripts/02_pull_statcast.py --season 2025             # for year-over-year chase change
scripts/03_pull_savant_leaderboards.py                # Savant ABS leaderboards (reference)
scripts/04_build_pitch_table.py --season 2026         # pitch table + challenges + token state + geometry
scripts/05_judgment.py                                # per-hitter judgment scores
scripts/06_policy.py                                  # token value DP, static/dynamic/observed with SEs, seeds, sigma table
scripts/07_eyes_vs_approach.py                        # the test
```

Setup:

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]" && .venv/bin/python -m pytest -q
```

Raw data lives in `data/raw/` (gitignored, about 2 GB), processed tables in `data/processed/`.

## Modules

| Module | What it does |
|---|---|
| `statsapi.py` | Schedule, cached feeds, extraction of pitches and challenges, chronological token-state reconstruction under three candidate extra-innings rules, ledger reconciliation |
| `zone.py` | Mid-plate trajectory solve, signed edge distance, ABS strike rule; tested against Savant's embedded challenge table |
| `savant.py` | Daily Statcast CSV pulls, ABS leaderboard CSV |
| `judgment.py` | Hierarchical logit of challenge on true edge distance: random slope is perceptual sharpness, random intercept is the criterion; empirical-Bayes shrinkage; signal-detection d-prime alongside |
| `policy.py` | Stake table from Statcast run values, calibrated and flat-prior beliefs, token value by backward induction with an extra-innings continuation, game-by-game policy simulation with paired standard errors |
| `chase.py` | Hitter-season chase and two-strike outcomes against the true ABS zone, plus a rulebook-zone chase rate for cross-season comparison |
| `models.py` | The pooled-covariate regressions with chase-rate quintile controls, stratified contrasts, hitter bootstrap, split-half reliability |

## The judgment model in one paragraph

Every umpire-called strike a hitter takes while the batting team holds a token is a decision: challenge or not. The true location is known from the trajectory. Across a season a hitter faces 60 to 400 of these, mostly within a few inches of the edge. A logit of challenge on true edge distance, with per-hitter random intercept and slope, separates how sharply the hitter's challenge probability rises as the pitch moves from inside the zone to outside it (eyes) from how far off the plate a pitch must be before they spend a token (approach). Count, token state and leverage shift the criterion and enter as fixed effects.

## Results, 2026 regular season

All tables are in `data/processed/`; figures in `docs/figures/`. Sample: 2,429 games, 714,483 pitches, 10,557 challenges, Statcast joined on every pitch.

### The psychometric curve

![challenge judgment curve](docs/figures/psychometric_curve.png)

Hitters faced 65,589 called strikes with a token in hand within four inches of the zone edge and challenged 7.0% of them. The umpire was wrong on 18% of those pitches. The league curve rises smoothly from under 1% at four inches inside the zone to about 40% at three and a half inches outside. It is a curve, not a step: at the edge itself hitters challenge only 11 to 14% of true balls. Two strikes shift the criterion by 1.9 logits (strike three is worth challenging), three balls by 0.5, a second token in hand by 0.2, the seventh inning onward by 0.7.

| Quantity | Value |
|---|---|
| League slope | 0.65 per inch |
| Between-hitter SD of slope (eyes) | 0.18 |
| Between-hitter SD of intercept (criterion) | 0.66 |
| Hitters scored (60+ decisions) | 388 |
| Split-half reliability, full season: slope | 0.27 |
| Split-half reliability, full season: criterion | 0.78 |
| Split-half reliability, full season: d-prime | 0.11 |

The headline measurement result is the last three rows. A hitter's willingness to challenge is a stable trait within one season. Their accuracy looks barely measurable per hitter at first: a typical hitter makes 10 to 25 challenges, and the slope's posterior SD (about 0.13) is most of the between-hitter spread (0.18). The slope correlates 0.76 with the simpler signal-detection d-prime and 0.10 with the criterion, so the two dimensions are separated as intended.

The low reliability is a sample-size problem, not a measurement problem. Split-half reliability of the slope, by how often a hitter challenges (`scripts/10_reliability_checks.py`):

| Minimum challenges per half-season | Hitters | Slope reliability | Criterion reliability |
|---|---|---|---|
| none | 300 | 0.27 | 0.78 |
| 5 | 119 | 0.31 | 0.76 |
| 10 | 28 | 0.79 | 0.85 |

With 20 or more challenges in a season the eyes score is as reliable as the criterion. The edge window barely matters (plus or minus 6 inches gives 0.29 and 0.83 at the same thresholds; plus or minus 2.5 is worse). In 2026, 62 regulars reached 20 challenges and 102 reached 15. Judgment is unidentified for hitters who rarely challenge; rankings and the main test use a minimum of eight, and the test is repeated at 15 and 20 below.

### Token value and the dynamic policy

![token value](docs/figures/token_value.png)

Opportunities (taken pitches within three inches of the edge, both sides) arrive 1.34 times per team-half-inning played and the umpire misses 20% of them. The league judgment slope implies perceptual noise of about 2.6 inches (an upper bound, decision 9). A simulated challenger perceives the true edge plus that noise and believes the call will flip with the calibrated probability P(flip | perceived edge), a logit fit on the opportunities with the same noise draw (decision 17). That belief knows the umpire is right on four near-edge calls in five; in every quintile it is within 0.3 points of the true flip rate.

Pricing a token by backward induction, with extra innings as a continuation value (decision 18):

| State | First token | Second token |
|---|---|---|
| Start of game | 0.15 runs | 0.09 runs |
| Start of 5th | 0.11 | 0.07 |
| Start of 9th | 0.04 | 0.02 |
| Start of an extra inning (given the game is in one) | 0.04 | 0.02 |

At league-average eyes a token is worth less than Savant's flat 0.20 at every moment of the game, so Savant's breakeven rule is too stingy throughout, and more so late. The direction depends on the noise level (table below): with noise below about 1.8 inches the first token at first pitch is worth more than 0.20.

Playing every game forward with each rule, deciding on a perceived edge and scoring on the true ABS outcome (runs per team per 162 games; differences paired by game over 2,429 games, plus or minus one standard error; noise draw 0):

| Rule | Eyes | Challenges per team-game | Overturned | Runs per team per 162 | vs static | vs observed |
|---|---|---|---|---|---|---|
| Static (Savant breakeven) | simulated | 0.75 | 53.9% | 19.1 | | −13.8 ± 0.6 |
| Dynamic (this repo) | simulated | 2.83 | 40.8% | 38.3 | +19.2 ± 0.5 | +5.5 ± 0.7 |
| Observed behavior | real players | 2.04 | 54.1% | 32.9 | +13.8 ± 0.6 | |
| Oracle (perfect eyes) | perfect | 4.81 | 100% | 111.1 | +92.0 ± 0.8 | +78.2 ± 0.8 |

Across noise draws 0, 1 and 2, dynamic minus static is +19.2, +19.6 and +19.8 (SE 0.5 each) and dynamic minus observed +5.5, +5.7 and +6.3 (SE 0.7 each).

**The observed row is not the same eyes.** Static and dynamic share the simulated league-average Gaussian noise. Observed is real players with their own eyes, who overturned 54% of 2.04 challenges a game. The noise model cannot produce that: under calibrated beliefs only 2,326 of 115,389 opportunities carry p above 0.6, yet players won 5,352 challenges. Real players hold information the noise model lacks, so the gap between either simulated rule and observed is not a property of the rule. The comparison the model supports is static against dynamic.

The dynamic rule is worth about 19 runs per team-season over Savant's static rule with the same simulated eyes, because the static rule's 0.20 price makes a league-average challenger wait for a 51% chance on a typical pitch, which that challenger rarely has, so it challenges 0.75 times a game. The gap between any rule and the oracle is the value of perception, which is where the per-player layer lives: `policy_hitter_thresholds_mlb_2026.csv` states, for each hitter, how far off the plate a pitch must look before spending the last token at the median stake. Sharpest hitters need about 2.5 inches in the first inning; the median hitter 4.8; for 31% of rated hitters the margin is beyond six inches, which in practice means not to spend the last token early on a median-stakes pitch. In the ninth the median is 1.2 inches. These margins rest on a slope with split-half reliability near 0.3 at eight challenges and should be read to the nearest inch.

**What changed on 2026-10-23.** The first version used the flat-prior belief Φ(perceived edge / σ), which ignores how often the umpire is right; a simulated p of 0.8 to 1.0 flipped 49% of the time. `scripts/06_policy.py --flat-prior` reproduces it (to `*_flatprior` files), and every run prints both:

| Belief | First token, first pitch | Static: chal / overturned / runs | Dynamic: chal / overturned / runs | Dynamic − static | Dynamic − observed |
|---|---|---|---|---|---|
| Calibrated (current) | 0.150 | 0.75 / 53.9% / 19.1 | 2.83 / 40.8% / 38.3 | +19.2 ± 0.5 | +5.5 ± 0.7 |
| Flat prior (before) | 0.491 | 3.10 / 39.5% / 32.6 | 3.24 / 40.5% / 35.4 | +2.8 ± 0.3 | +2.5 ± 0.7 |

The previously published flat-prior figures (0.487, +2.65, +2.34) differ slightly from the second row because the DP now models extra innings and the arrival rate is per half-inning played rather than per scheduled half-inning. The advice reverses: the old table said to be stingy early and aggressive late; with coherent beliefs a token is worth less than Savant's price all game.

**Sensitivity to the noise level** (noise draw 0; `policy_sigma_mlb_2026.csv`):

| Noise σ (in) | Calibrated: first token | Static runs | Dynamic runs | Dynamic − static | Flat prior: first token | Dynamic − static |
|---|---|---|---|---|---|---|
| 1.0 | 0.318 | 60.9 | 69.8 | +9.0 ± 0.4 | 0.458 | +2.2 ± 0.4 |
| 1.5 | 0.232 | 41.4 | 55.0 | +13.6 ± 0.5 | 0.457 | +1.9 ± 0.4 |
| 2.0 | 0.183 | 28.5 | 45.4 | +16.9 ± 0.5 | 0.470 | +2.7 ± 0.4 |
| 2.63 (league) | 0.150 | 19.1 | 38.3 | +19.2 ± 0.5 | 0.491 | +2.8 ± 0.3 |
| 3.5 | 0.128 | 12.0 | 33.6 | +21.6 ± 0.6 | 0.514 | +3.0 ± 0.3 |
| 4.5 | 0.115 | 8.6 | 31.2 | +22.5 ± 0.6 | 0.535 | +2.4 ± 0.3 |

Under the flat prior the conclusion was insensitive to σ and entirely driven by the belief model. Under calibrated beliefs σ matters: the sharper the eyes, the more a token is worth and the smaller the gain from dynamic pricing, because Savant's 0.20 is then close to right early in the game.

### The eyes-versus-approach test

![judgment vs chase](docs/figures/judgment_vs_chase.png)

Sample: 219 hitters with 60+ decisions, 8+ challenges and 150+ PA.

**Judgment predicts chasing.** Sharper challenge judgment correlates −0.21 with chase rate (p = 0.002; −0.41 after correcting for the score's reliability). It also correlates +0.24 with walk rate and −0.18 with in-zone swing rate, and 0.00 with strikeout rate. The criterion correlates +0.18 with chase rate (hitters who spend tokens freely also swing freely) and not with walks. So the perception signal on takes is related to swing decisions in the expected direction, and the aggressiveness signal is a different thing.

**Holding chase rate fixed, judgment predicts nothing else yet.** With chase-rate quintile, criterion, bat speed and bat side controlled, a one-SD increase in judgment moves:

| Outcome | Effect per SD of judgment | Robust SE | p | Effect in outcome SDs, disattenuated |
|---|---|---|---|---|
| Runs per 100 chase swings | −0.24 | 0.18 | 0.18 | −0.48 |
| Whiff rate on chases | +0.002 | 0.010 | 0.87 | +0.06 |
| xwOBA on chase contact | −0.004 | 0.005 | 0.49 | −0.27 |
| Two-strike contact rate | +0.002 | 0.006 | 0.74 | +0.10 |
| Strikeout rate | −0.002 | 0.006 | 0.70 | −0.13 |
| Two-strike chase rate | +0.006 | 0.005 | 0.27 | +0.27 |

None is distinguishable from zero, and the within-quintile contrasts change sign across quintiles. The one with a nonzero point estimate, run value per chase, points the wrong way for the hypothesis (sharper eyes, costlier chases), and its disattenuated size is still within noise.

**Restricting to frequent challengers, where the score is reliable.** With 15 or more challenges (102 hitters), the judgment-chase correlation strengthens to −0.31 and the criterion-chase correlation to +0.26, as better measurement should produce. The conditional effects stay null:

| Outcome (15+ challenges, n = 102) | Effect per SD of judgment | Robust SE | p |
|---|---|---|---|
| Runs per 100 chase swings | −0.38 | 0.25 | 0.13 |
| Whiff rate on chases | +0.012 | 0.014 | 0.38 |
| xwOBA on chase contact | −0.007 | 0.007 | 0.31 |
| Two-strike contact rate | −0.001 | 0.008 | 0.93 |
| Strikeout rate | +0.005 | 0.009 | 0.61 |
| Two-strike chase rate | −0.000 | 0.008 | 0.97 |
| Year-over-year chase change (2025 to 2026) | −0.005 | 0.009 | 0.57 |

**Year-over-year chase change is the one positive result, with a caveat.** The row above controls for the 2026 chase rate, which is the endpoint of the change and biases that row toward zero (p = 0.10 on 175 hitters with a 2025 season). The clean question is: among hitters with the same 2025 chase rate, do sharper eyes go with a lower 2026 chase rate?

| Sample | n | 2026 chase rate per SD of judgment, holding 2025 chase rate | Robust SE | p |
|---|---|---|---|---|
| 8+ challenges | 175 | −0.0099 | 0.0040 | 0.013 |
| 15+ challenges | 88 | −0.0070 | 0.0056 | 0.22 |
| 20+ challenges | 54 | −0.0090 | 0.0072 | 0.21 |

One SD of sharper judgment goes with about one percentage point less chasing than the hitter's 2025 rate predicts, against a year-over-year change SD of 3.5 points. The point estimate is stable across thresholds; only the full sample has the power to call it. Two caveats. Judgment and the 2026 chase rate are measured in the same season, so a common cause (a hitter who simply saw the ball better in 2026) is not excluded; the pre-registered version uses 2026 judgment to predict the 2027 change. And the rulebook chase rate rose league-wide from 28.4% to 30.6% because the 2026 zone bounds in the Statcast feed are the fixed ABS bounds rather than stance-based; the shift is absorbed by the intercept but the levels are not comparable across years.

Every chase-quality point estimate leans the wrong way (sharper eyes, slightly costlier chases), so one candidate explanation was tested: that sharper hitters lay off borderline balls and their remaining chases are further off the plate, which are costlier league-wide (−0.04 runs per chase within an inch of the zone, −0.09 beyond six inches). It fails. Holding chase rate fixed, judgment does not move a hitter's mean chase edge distance or the share of far chases (p above 0.5 at both thresholds). The leanings are unexplained noise.

**Reading.** Among hitters who chase the same amount, the ones with sharper challenge judgment do not chase more cheaply, make more two-strike contact, or strike out less. What judgment does predict is the chase rate itself, both in cross-section (−0.21 overall, −0.31 among frequent challengers) and as a change from the prior season (about one point lower per SD, holding the 2025 rate). The eyes-versus-approach split as measured here says *whether* a hitter chases, and in which direction they are moving, not *what happens* when they swing at a ball. The interpretation that fits is that perception on takes near the edge governs the decision to swing, while the outcome of a swing at a ball is governed by something else (bat-to-ball skill, pitch type, how far off the plate the chase is). For the hitting coordinator this means the score sorts hitters into eyes cases and approach cases for the *rate* problem; it does not identify who will pay most per chase. The nulls on frequent challengers are no longer a reliability artifact, though 102 hitters cannot rule out small effects; 2027 doubles the sample, and the pre-registered test is 2026 judgment against 2027 chase change.

### Validation

- Per-player challenge attribution matches Savant's leaderboard: batters 99.4% exact (the only gap is two players named Max Muncy merged by name), catchers 100% within one, pitchers 100% exact.
- Geometry agrees with the ABS ruling on 99.99% of challenged pitches, with zero disagreements within a quarter inch.
- End-of-game token state matches the official ledger in 99.84% of games under the ledger's accounting; the per-inning extras top-up explains all 74 zero-token challenges.

## Club brief

`scripts/11_club_brief.py` turns the processed tables into a single self-contained page for coaches and front offices, in `report/`:

- `club_brief_2026.html` is the web version (publish it as a shareable page).
- `club_brief_2026_standalone.html` opens locally and can be attached to an email.

Part one is for the dugout: when to spend a token (thresholds by inning, tokens held and stake, in percent and in inches for an average eye), per-hitter challenge cards with a team selector (challenges, overturns, eyes and criterion percentiles, and the perceived margin each hitter should want before challenging early, mid-game and in the ninth), and the eyes-versus-approach sort. Part two is the evidence: the psychometric curve, reliability by challenge volume, the policy simulation with paired standard errors and noise draws, the belief-model comparison and the σ table, the test results, and methods. The brief generator refuses to write a page that shows nan, None or undefined outside script code. Every table names the processed file it came from, so the page regenerates on a new season with `scripts/run_pipeline.sh` followed by this script.

## Limits

- One season of ABS. Year-over-year stability of the judgment score cannot be measured until 2027; split-half reliability within 2026 is reported instead.
- Judgment is measured on takes near the edge and may not transfer to pitches hitters chase, which are often further off the plate. That is the test, not an assumption.
- The 2025 Triple-A feed has the same challenge structure but omits the initiating player.
- The Statcast join covers Statcast-equipped parks only; the feed geometry covers every pitch.
- The policy simulation gives every challenger league-average Gaussian eyes. Real players do better than that model on the challenges they make (54% overturned), so the observed row is context, not a comparison of rules.
- Under calibrated beliefs the token's value depends on the noise level σ, and σ = 1.7 / slope is an upper bound. If the true league noise is below about 1.8 inches, the first token early is worth more than Savant's 0.20 and the early-game advice moves back toward Savant's rule.
- The DP prices the token at the start of each half-inning, plays all 18 half-innings in every game, and treats run stakes as independent of the score. Run units, not win probability.
- Per-hitter margins use each hitter's own σ from a slope with reliability near 0.3 at eight challenges; they are mostly the prior for infrequent challengers.
