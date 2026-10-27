# ABS challenge judgment

Public data from the 2026 MLB Automated Ball-Strike challenge system, used two ways: to price a challenge token by game state, and to measure a hitter's eyes from his challenge decisions on takes, where no swing is involved. Full tables: [docs/results_2026.md](docs/results_2026.md). Method: [docs/research_design.md](docs/research_design.md). Modeling choices: [docs/decisions.md](docs/decisions.md). The review that corrected the first version: [docs/review_2026-10-23.md](docs/review_2026-10-23.md).

## Data

Every ABS challenge is public at the pitch level in the MLB StatsAPI live feed (`reviewDetails`, review type `MJ`), with initiator and outcome. The 2026 regular season gives 10,557 challenges in 2,429 games, matching Savant's total, and per-team counts reconcile with the in-game token ledger in all but one game. The feed's zone bounds are the fixed ABS zone (53.5% and 27% of measured height). The mid-plate geometry reproduces Savant's to five decimals and agrees with the ABS ruling on 99.99% of challenged pitches. Statcast run values and bat tracking are joined on every pitch.

## What a club can act on now

Counted from the feeds with every pitch's true location, no model (`first_actions` in `scripts/11_club_brief.py`):

- Hitters challenged 19% of the 11,992 true balls called strikes while holding a token, leaving about 40 runs per team-season if every miss were caught. Catchers challenged 32% of true strikes called balls, leaving 32.
- The high strike is the one to challenge. Umpires call strikes on 49% of takes in the first inch above the box against 40% off the side and below, and hitters challenge only 15% of high misses against 26% of low ones.
- Strike three on a true ball went unchallenged 0.14 times per team-game, worth 7.3 runs per team-season.
- Of the 2,438 losing batter challenges, the median was 1.2 inches inside the zone and 36% were low pitches. Hitters over-challenge low and under-challenge high.

## The judgment model

Every umpire-called strike a hitter takes with a token in hand is a decision with a known true location. A logit of challenge on true edge distance (4 inches inside to 6 outside), with a per-hitter random intercept and slope under a full-covariance empirical-Bayes prior, separates how sharply the challenge rate rises as the pitch leaves the zone (eyes) from how far off a pitch must look before a token is spent (criterion). Count, token state, leverage and the side of the box the pitch missed on are fixed effects; at the same distance hitters challenge a low strike about twice as often as a high one.

| Measurement, 2026 | Value |
|---|---|
| Hitters scored (60+ decisions) | 390 |
| Criterion split-half reliability | 0.78 [0.72, 0.82] |
| Eyes (slope) reliability, all hitters | 0.27 [0.08, 0.42] |
| Eyes reliability, 10+ challenges per half-season (29 hitters) | 0.63 [0.21, 0.83] |

Willingness to challenge is a stable trait within a season. Accuracy is not reliably measured for any individual hitter in one season, so the brief shows the eyes score as a range and gives no hitter an "eyes case" or "approach case" label.

## Token value and policy

A simulated challenger perceives the true edge plus Gaussian noise (2.7 inches, an upper bound from the league slope) and holds the calibrated belief P(flip | perceived edge), which knows the umpire is right on four near-edge calls in five. Backward induction prices the first token at 0.15 runs at first pitch, 0.11 at the fifth inning, 0.04 in the ninth, with the second token at about 60% of that. Savant's flat 0.20 is above the token's value all game, so its breakeven rule is too stingy throughout. Played forward over 2,429 games with the same simulated eyes, the dynamic rule beats the static rule by 19.6 ± 0.5 runs per team-season. Real players overturn 54% of their challenges, which the noise model cannot reproduce, so observed behavior is context rather than a third rule. With noise below about 1.8 inches the first token early is worth more than 0.20 and the early-game advice moves back toward Savant's.

The first version's flat-prior belief ignored the umpire's accuracy, priced the token at 0.49 and gave the opposite advice; `scripts/06_policy.py --flat-prior` reproduces it.

## The eyes-versus-approach test

Among hitters with similar chase rates, does poor challenge judgment predict costlier chases, worse two-strike contact, or weaker year-over-year improvement? With judgment as a pooled covariate and chase-rate quintile, criterion, bat speed and side controlled, on 221 regulars with 8+ challenges: judgment predicts the chase rate itself (correlation −0.18; −0.30 among 104 hitters with 15+ challenges) and the 2025-to-2026 change holding the 2025 rate (about one percentage point less chasing per SD, p = 0.007; not significant at 15+ and 20+). It predicts nothing else: run value per chase, whiff rate, xwOBA on chase contact, two-strike contact and strikeout rate are null at every threshold. The positive result is one of 27 tests and shares a season with the judgment measurement; the pre-registered version is 2026 judgment against 2027 chase change.

## Club brief

`scripts/11_club_brief.py` writes `report/club_brief_2026.html`, one page for coaches and front offices: the facts above, when to spend a token, per-hitter challenge cards, and the evidence with its uncertainty. Every chart is drawn from the processed tables and has a table view.

## Pipeline

```
scripts/01_pull_statsapi.py --season 2026 --sport 1
scripts/02_pull_statcast.py --season 2026   # and 2025, for year-over-year
scripts/03_pull_savant_leaderboards.py
scripts/run_pipeline.sh 2026                # build, judgment, policy, test, reliability
scripts/11_club_brief.py
```

Setup: `python3 -m venv .venv && .venv/bin/pip install -e ".[dev]" && .venv/bin/python -m pytest -q`. Raw data is gitignored; processed tables land in `data/processed/`.

## Limits

- One season. Year-over-year stability of the judgment score waits for 2027.
- The token's value depends on the noise level, which is an upper bound, and is in run units rather than win probability.
- Hitters let a third of obvious misses go for reasons other than perception, and the invisible box sits inside the umpire's zone at the top, so a high-strike decision is partly a decision about where the box is. The slope absorbs some of both.
