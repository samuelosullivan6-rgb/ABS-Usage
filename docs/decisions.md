# Modeling decisions

Each entry: the choice, the alternative, why.

1. **Season table from StatsAPI feeds, Statcast CSV joined for run values.** Alternative: Savant's embedded pitch-level challenge table. Savant only exposes the trailing seven days; the feed has every pitch with the full trajectory and the challenge events.

2. **ABS zone from the feed's `strikeZoneTop/Bottom`.** Alternative: rebuild from listed heights. The feed values are the fixed ABS bounds (ratio 1.981 on every batter), verified against Savant's edge distances.

3. **Mid-plate location by solving the nine-parameter trajectory at y = 17/12 - 8.5/12 ft.** Alternative: Statcast `plate_x/plate_z`. The public CSV coordinates sit at a different reference plane; the solve reproduces Savant to five decimals.

4. **Edge distance = ball-edge to zone boundary, 2.9-inch ball, negative inside.** Matches `edge_dist_calc` exactly and the ABS ruling on 99.99% of challenges.

5. **Extra-innings rule: top-up to one at the start of each extra inning for a team at zero.** Alternatives: none, or once at the 10th. Only the per-inning rule explains all 74 zero-token challenges, including three in one inning by one team.

6. **Judgment decision set: umpire-called strikes taken with a token in hand, edge within plus or minus 4 inches.** Alternative: all called strikes. Pitches deep in the zone are never challenged and add nothing to the slope; pitches with no token carry no decision.

7. **Hierarchical logit with random intercept and slope, fit by two-step empirical Bayes.** Alternative: statsmodels' variational mixed GLM. The two-step is transparent, fast, and the prior update is the standard EM one: tau-squared equals the variance of posterior means plus the mean posterior variance. (The first implementation subtracted, which collapsed the slope prior; fixed.)

8. **Criterion shifters as fixed effects: two strikes, three balls, a second token, late innings, close score.** These move how far off the plate a pitch must look before a token is spent; they do not change the eyes.

9. **Perceptual noise sigma = 1.7 / league logit slope.** The logit-to-probit constant. The slope includes within-hitter criterion variability, so sigma is an upper bound on pure perceptual noise.

10. **Minimum 8 challenges for per-hitter use.** A hitter who never challenges reveals nothing about perception; the posterior slope is then the prior plus the criterion.

11. **Policy opportunity set: taken pitches within 3 inches of the edge, both sides.** A team's tokens serve its hitters and its battery. Beyond 3 inches the umpire is essentially never wrong.

12. **Decision slots: 8 per half-inning, 18 half-innings, arrival probability from the observed opportunity rate.** Coarse enough to be fast, fine enough that the slot value changes smoothly.

13. **Stakes from Statcast `delta_run_exp` on called balls minus called strikes in the same count, outs and base state**, with a count-and-outs fallback below 200 pitches per cell. Alternative: a hand-built RE288. Savant's own run expectancy keeps the policy comparable to Savant's pricing.

14. **Policies are decided on a simulated perceived edge and scored on the true outcome.** Alternative: decide on the true edge. That would credit the policy with perfect eyes; the oracle row shows that ceiling separately.

15. **The test weights hitters by the sample behind each outcome and controls chase rate with quintile dummies.** Alternative: linear chase-rate control. The relationship is not assumed linear and the within-quintile contrast is reported alongside.

16. **Disattenuation by split-half reliability** is reported next to raw coefficients, never in place of them.

17. **Belief model: the calibrated posterior P(true flip | perceived edge), not the flat-prior Φ(perceived edge / σ).** (2026-10-23, after the review in `docs/review_2026-10-23.md`, A1.) The simulated challenger's subjective probability is a logit of the true flip on the perceived edge, fit on the policy opportunity set with the same noise draw the policies decide on; per-hitter margins use the same fit at each hitter's σ (a grid of σ, interpolated). Alternatives: the flat prior, or the exact Bayes posterior under the empirical distribution of the true edge. The flat prior ignores that the umpire is right on 80% of near-edge calls, so a simulated p of 0.8 to 1.0 flipped 49% of the time; a challenger who is that overconfident is not a model of anyone, and the DP then valued two tokens at 117 runs a season while the same policy realized 35. The logit is within 0.3 points of the true flip rate in every quintile of p, which is as close as the exact posterior would be, and it is a two-number summary the brief can print. It is pooled over batter and fielder opportunities; on each side alone the pooled p is off by about 1.5 points on average (low for batters, high for fielders). The change cuts the first token's value at first pitch from 0.49 to 0.15 runs and reverses the dugout advice. `--flat-prior` reproduces the old model.

18. **Extra innings modeled in the DP as a continuation value, the same rule as the simulation.** (2026-10-23, review A4.) The first DP had a top-up branch that tested `t // slots >= 18` with T = 18 × 8 slots, so it never ran and V(0, t) was zero everywhere, while the simulation did top up; the two disagreed. Now: at the last slot of the 9th, V(k, T) = π X(max(k, 1)), with π = 209 / 2,429 games reaching the 10th and X(k) the value of entering an extra inning with k tokens after the top-up. X solves the same Bellman over one inning (16 slots) with terminal ρ X(max(k, 1)), ρ = 0.27 the share of extra innings followed by another (285 extra innings in 209 games), by fixed-point iteration. A team at zero in regulation is then worth π X(1) = 0.005 runs, the tenth-inning top-up. In extras the simulation prices the token from the extra-inning table. Alternative: remove the branch and the claims. The rule is confirmed by the ledgers (decision 5), so modeling it costs little. Its effect is small and has a clean shape: the first token's marginal value is unchanged, because losing the last token costs the same with or without a top-up that comes either way, and the second token gains at most 0.002 runs, late in the 9th. The arrival rate is now per half-inning actually played (43,203) rather than 36 per game, since extras pitches are in the numerator. Not modeled: a game ending without the bottom of the 9th, and the dependence of π on the score.

19. **Standard errors paired by game, three noise draws, and a σ table.** (2026-10-23, review A3 and A7.) `simulate_policy` returns runs by game; differences between rules are paired over the 2,429 games. Noise draws 0, 1 and 2 are run for both belief models, and σ is varied from 1.0 to 4.5 inches. Under calibrated beliefs σ moves the token's value across Savant's 0.20 (at about 1.8 inches), so the σ table is part of the result, not a robustness footnote. The "never" rule is dropped.
