# Data findings, 2026-10-21

These are verified against live endpoints on 2026-10-21. They replace the "not verified" caveats in the research-ideas doc (MLB RESEARCH/research/mlb_rd_research_ideas_2026-10-21.md, Idea 2).

## 1. Pitch-level ABS challenge events are public (MLB StatsAPI)

Endpoint: `https://statsapi.mlb.com/api/v1.1/game/{gamePk}/feed/live`

- `gameData.absChallenges` is the per-game token ledger: `{away|home: {usedSuccessful, usedFailed, remaining}}`.
- Mid-plate-appearance challenges: the challenged pitch's `playEvents[i]` carries `reviewDetails` with `reviewType: "MJ"`, `challengeTeamId`, `player` (the initiator), `isOverturned`, and `details.hasReview: true`.
- Plate-appearance-ending challenges: `reviewDetails` is lifted to the play level (`allPlays[i].reviewDetails`) and the pitch event is NOT flagged. The challenged pitch is the last pitch of the PA.
- Six games on 2026-06-15 had 18 challenges per the ledgers; 14 were found at pitch level and 4 at play level. Full recovery.
- `details.call` on the pitch is the POST-challenge call. Original call = opposite if `isOverturned`, else the same. Cross-check: batting-team challenges are always of called strikes, fielding-team challenges always of balls.
- Initiator side: `challengeTeamId == batting team` means batter; otherwise compare `player.id` to the matchup pitcher to split catcher from pitcher.
- Other `reviewType` codes seen: `MA` (tag play), `MF` (play at first), `MO`. Filter on `MJ`.
- `liveData.boxscore.info` has an "ABS Challenge" line listing every challenge as `Name (Strike-Confirmed | Strike-Overturned to Ball | Ball-Confirmed | Ball-Overturned to Strike)`. Use it as a per-game count check.
- Join key to Statcast CSV: `(game_pk, at_bat_number == atBatIndex + 1, pitch_number)`. The Statcast CSV has no challenge column.

## 2. The feed's strike-zone bounds are the ABS zone

In 2026 `pitchData.strikeZoneTop/Bottom` are fixed per batter with a top/bottom ratio of 1.981 on every batter checked, i.e. 53.5% and 27% of measured height. Measured height is recoverable as `top / 0.535`. The "re-measured heights" question is closed.

## 3. Mid-plate location is exactly reproducible

Extrapolating the feed's nine-parameter trajectory to y = 17/12 - 8.5/12 ft reproduces Savant's `plateX/plateZ` to 1e-5 on 17 of 17 test pitches. `edge_dist_calc` is reproduced exactly by: signed distance from ball CENTER to nearest zone boundary (negative inside), times 12, minus the ball radius of 1.45 in. Negative means any part of the ball overlaps the zone (ABS strike). The research doc's front-of-plate caveat no longer applies. Code: `src/abs_judgment/zone.py`.

## 4. Baseball Savant endpoints

- Leaderboard CSV works: `https://baseballsavant.mlb.com/leaderboard/abs-challenges?season=2026&challengeType=batter&csv=true`. `challengeType` accepts batter, catcher, pitcher, batting-team, catching-team. `season=2025&level=aaa` works. Columns include runs_above_average, runs_gained, runs_cost, xruns_gained, xruns_cost, poss_runs, runs_taken_percent, runs_not_taken, n_challenges, n_overturns, n_confirms, rate_overturns, exp_chal and expected-rate baselines.
- `https://baseballsavant.mlb.com/abs?year=2026&gameType=regular&level=mlb` embeds `totalData` (2026 regular season: 10,557 challenges on 345,826 pitches; batter/fielder/catcher/pitcher splits) and `vizChallenges`, a pitch-level table for the LAST 7 DAYS ONLY (404 rows) with Savant's `sz_challenge_prob`, `edge_dist_calc`, `original_isStrike_ump`, `N_prior_challengeABS_lost`, count and score state. Useful as a validation fixture, not as the season table. `gameType` accepts regular, playoff, spring; `level` accepts mlb, aaa; `year` 2025 or 2026.
- The research doc's refuted claim of "about 10,557 challenges" is in fact what Savant's page reports. Recompute from StatsAPI regardless.

## 5. Statcast search CSV (2026)

Standard pitch-level columns plus `delta_run_exp`, `delta_pitcher_run_exp`, `bat_speed`, `swing_length`, `miss_distance`, `attack_angle`, `swing_path_tilt`, intercept positions, `home_win_exp`. About 2,800 rows per 10-game day. This supplies chase cost and two-strike contact outcomes.

## Still unverified

- Extra-innings token reset rule in the ledger (find a 2026 game past nine innings and inspect `absChallenges`).
- Whether the 2025 Triple-A feed (sportId=11) carries the same `reviewDetails` structure.
- Full-season pull volume: about 2,430 feeds at roughly 800 KB each.
