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

## Results

See the Results section below once the full-season fits are recorded here, and `data/processed/` for the tables.

## Limits

- One season of ABS. Year-over-year stability of the judgment score cannot be measured until 2027; split-half reliability within 2026 is reported instead.
- Judgment is measured on takes near the edge and may not transfer to pitches hitters chase, which are often further off the plate. That is the test, not an assumption.
- The 2025 Triple-A feed has the same challenge structure but omits the initiating player.
- The Statcast join covers Statcast-equipped parks only; the feed geometry covers every pitch.
