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
scripts/06_policy.py                                  # token value DP, replay vs Savant's static rule
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
| `policy.py` | Stake table from Statcast run values, overturn curve, token value by backward induction, replay of observed challenges under static and dynamic pricing |
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

Opportunities (taken pitches within three inches of the edge, both sides) arrive 1.32 times per team-half-inning and the umpire misses 20% of them. The league judgment slope implies perceptual noise of about 2.6 inches. Pricing a token by backward induction gives:

| State | First token | Second token |
|---|---|---|
| Start of game | 0.49 runs | 0.23 runs |
| Start of 5th | 0.33 | 0.15 |
| Start of 9th | 0.10 | 0.03 |

Savant's flat 0.20 is right for the first token around the sixth inning and for the second around the second. Early in the game the static rule challenges too readily; late it hoards.

Playing every game forward with each rule, deciding on a perceived edge with league-average noise and scoring on the true ABS outcome:

| Rule | Challenges per team-game | Overturned | Runs per team per 162 |
|---|---|---|---|
| Static (Savant breakeven) | 3.10 | 39.5% | 32.6 |
| Dynamic (this repo) | 3.29 | 40.0% | 35.2 |
| Observed behavior | 2.04 | 54.1% | 32.9 |
| Oracle (perfect eyes) | 4.81 | 100% | 111.1 |

The dynamic rule is worth about 2.5 runs per team-season over either the static rule or what teams actually did, with the same eyes. That is the low-single-digit figure the research brief anticipated, and it confirms that most of Savant's "runs not taken" is rational token conservation rather than error. The gap between any rule and the oracle (about 75 runs) is the value of perception, which is where the per-player layer lives: the policy table in `policy_hitter_thresholds_mlb_2026.csv` states, for each hitter, how far off the plate a pitch must look before spending the last token (about 1.9 inches in the first inning for the sharpest hitters, 4.5 to 5.7 for the shallowest, near zero for everyone in the ninth).

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

Every chase-quality point estimate leans the wrong way (sharper eyes, slightly costlier chases), so one candidate explanation was tested: that sharper hitters lay off borderline balls and their remaining chases are further off the plate, which are costlier league-wide (−0.04 runs per chase within an inch of the zone, −0.09 beyond six inches). It fails. Holding chase rate fixed, judgment does not move a hitter's mean chase edge distance or the share of far chases (p above 0.5 at both thresholds). The leanings are unexplained noise.

**Reading.** Among hitters who chase the same amount, the ones with sharper challenge judgment do not chase more cheaply, make more two-strike contact, strike out less, or improve their chase rate more from the prior season. The eyes-versus-approach split as measured here predicts *whether* a hitter chases, not *what happens* when they do. The strongest remaining interpretation is that perception on takes near the edge is a different skill from what governs the outcome of a swing at a ball, and that chase rate itself already carries the perception signal (the −0.31 correlation). The null on frequent challengers is no longer a reliability artifact, though 102 hitters cannot rule out small effects; 2027 doubles the sample.

### Validation

- Per-player challenge attribution matches Savant's leaderboard: batters 99.4% exact (the only gap is two players named Max Muncy merged by name), catchers 100% within one, pitchers 100% exact.
- Geometry agrees with the ABS ruling on 99.99% of challenged pitches, with zero disagreements within a quarter inch.
- End-of-game token state matches the official ledger in 99.84% of games under the ledger's accounting; the per-inning extras top-up explains all 74 zero-token challenges.

## Limits

- One season of ABS. Year-over-year stability of the judgment score cannot be measured until 2027; split-half reliability within 2026 is reported instead.
- Judgment is measured on takes near the edge and may not transfer to pitches hitters chase, which are often further off the plate. That is the test, not an assumption.
- The 2025 Triple-A feed has the same challenge structure but omits the initiating player.
- The Statcast join covers Statcast-equipped parks only; the feed geometry covers every pitch.
