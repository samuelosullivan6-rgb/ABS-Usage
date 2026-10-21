# Research design: challenge judgment as the public signal of a hitter's eyes

## The problem this solves

A hitter who chases has one of two problems. Either they do not see the pitch well enough to know it is a ball (an eyes problem), or they see it fine and swing anyway because of their count plan or aggression (an approach problem). The coaching fix differs: pitch-recognition training for the first, a count plan for the second. Chase rate cannot tell them apart because every swing decision mixes perception with intent.

The ABS challenge system creates, for the first time, a public measurement of perception with intent removed. When a hitter takes a pitch and the umpire calls a strike, the hitter decides whether to challenge. That decision is a report of where the hitter believed the ball was, and the true location is known from the trajectory. No swing is involved. It is a psychometric experiment the league runs on every called strike.

## Starting point 1: dynamic challenge policy with per-player calibration

Savant already prices every challenge in runs and publishes a static breakeven rule that values a token at a constant 0.2 runs. The gap is that a token's value is not constant. It depends on how many are held, how many decision opportunities remain, and (if the rule is confirmed from the ledgers) whether extra innings restore a token. `policy.py` prices the token by backward induction over decision slots and produces a state-dependent threshold. The replay in `06_policy.py` compares observed challenges, the static rule and the dynamic rule on the same pitches.

The per-player layer: the probability a challenge succeeds is the challenger's subjective probability, which depends on how well they see the ball. The judgment model below yields each hitter's perceptual noise, so the policy threshold can be stated per hitter as a perceived margin off the plate.

## The judgment model

Decision set: every umpire-called strike the hitter took while the batting team held at least one token, within four inches of the zone edge either way (pitches deep in the zone carry no information about the transition).

    P(challenge) = logit^-1( a + b * edge + u_i + v_i * edge + count and leverage controls )

* `edge` is the true signed distance, in inches, from the ball's edge to the ABS zone boundary at the plate midpoint. Positive means the pitch was a ball.
* `b + v_i` is hitter i's perceptual sharpness. A hitter who sees the ball well goes from never challenging inside the zone to challenging outside it over a short distance. A hitter with poor perception challenges some true strikes and lets some true balls pass, which flattens the curve.
* `a + u_i` is the criterion: how far off the plate a pitch must appear before the hitter spends a token. Two strikes, three balls, a second token in hand and late-close situations shift the criterion and are fixed effects.

This is the signal-detection split of sensitivity from bias, estimated on a continuous stimulus. Empirical-Bayes shrinkage pools every hitter toward the league curve with a prior variance estimated from the data, so the score is usable with the 60 to 400 decisions a hitter accumulates in a season. A descriptive d-prime from hit rate (challenged given true ball) and false-alarm rate (challenged given true strike) is reported alongside.

Why it is stripped of intent: the hitter did not swing. The only intent in the decision is whether to spend a token, and that is the criterion, which is estimated separately from the slope.

## Starting point 2: the test

Among hitters with similar chase rates, do those with poor location-adjusted challenge judgment show:

1. costlier chases (run value and whiff rate on swings at ABS balls, xwOBA on chase contact),
2. worse two-strike contact (contact and whiff rate on two-strike swings, strikeout rate),
3. weaker year-over-year chase improvement (rulebook-zone chase rate in 2026 minus 2025)?

Judgment enters as a covariate pooled across hitters. The per-hitter estimate need only be unbiased, not precise, because the question is about the slope across hitters. Chase rate is held fixed with quintile dummies and the contrast is also shown within quintile (top versus bottom judgment tercile). Regressions are weighted by the sample behind each outcome, with robust standard errors and a hitter bootstrap.

Decision link: the hitting coordinator's development plan. A hitter with a high chase rate and sharp judgment is an approach case. A hitter with a high chase rate and shallow judgment is an eyes case. The test asks whether that distinction predicts anything a coach cares about.

## Identification concerns, stated plainly

* Selective challenging. Hitters challenge when confident, so the overturn rate alone conflates eyes with conservatism. The model handles this by using every called strike, challenged or not, and by estimating the criterion separately.
* Shrinkage attenuates. judgment_z is a posterior mean, so a coefficient on it is biased toward zero. Significant effects are conservative; nulls are not proof of absence.
* The catcher's framing and the umpire's tendencies affect which pitches become called strikes, not the hitter's perception of them once called. They change the stimulus distribution, not the response function, and are absorbed by the location control.
* Perception is measured on takes near the edge. It may not transfer to the pitches hitters chase, which are often further off the plate. The test is whether it does.
* One season of ABS. Year-over-year stability of judgment cannot be measured until 2027. Split-half reliability within 2026 is reported instead.

## What would change the conclusion

If split-half reliability of the judgment slope is below about 0.3, the score is too noisy to use per hitter and only the pooled test stands. If the judgment coefficient is null on every outcome after controlling for chase rate, challenge judgment and chase perception are different skills and the eyes-versus-approach split must come from somewhere else.
