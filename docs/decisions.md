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
