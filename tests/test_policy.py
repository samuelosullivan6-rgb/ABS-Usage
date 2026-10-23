import numpy as np
import pandas as pd

from abs_judgment.policy import savant_static_threshold, token_value_dp, token_value_table


def _opps(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"p": rng.beta(2, 2, n), "stake": rng.gamma(2.0, 0.08, n)})


def test_token_value_falls_to_zero_at_game_end():
    V, thr, q = token_value_dp(_opps(), lam_per_half=0.6)
    assert V[1, -1] == 0 and V[2, -1] == 0
    assert V[1, 0] > 0 and V[2, 0] > V[1, 0]
    tv = token_value_table(V)
    first = tv[(tv.half_inning == 1) & (tv.tokens == 1)].token_value.iloc[0]
    last = tv[(tv.half_inning == 18) & (tv.tokens == 1)].token_value.iloc[0]
    assert first > last >= 0


def test_second_token_worth_less_than_first():
    V, _, _ = token_value_dp(_opps(), lam_per_half=0.6)
    tv = token_value_table(V)
    early = tv[tv.half_inning == 1].set_index("tokens").token_value
    assert early[2] <= early[1]


def test_dynamic_threshold_below_static_late_and_bounded():
    V, thr, _ = token_value_dp(_opps(), lam_per_half=0.6)
    assert np.nanmin(thr[1:, :-1]) >= 0 and np.nanmax(thr[1:, :-1]) <= 1
    s_med = float(np.median(_opps().stake))
    static = savant_static_threshold(np.array([s_med]))[0]
    # in the final half-inning a token is nearly worthless, so the threshold is far below Savant's
    assert thr[1, -2] < static


def test_more_opportunities_raise_token_value():
    V_lo, _, _ = token_value_dp(_opps(), lam_per_half=0.2)
    V_hi, _, _ = token_value_dp(_opps(), lam_per_half=1.2)
    assert V_hi[1, 0] > V_lo[1, 0]


def test_v0_is_zero_without_extras_and_the_topup_value_with_them():
    from abs_judgment.policy import extras_value

    o = _opps()
    V, _, _ = token_value_dp(o, lam_per_half=0.6)
    assert np.all(V[0, :] == 0)
    pi, rho = 209 / 2429, 0.27
    V, _, _ = token_value_dp(o, lam_per_half=0.6, p_extras=pi, rho=rho)
    X = extras_value(o, 0.6, rho)[:, 0]
    # a team at zero holds only the tenth-inning top-up, the same at every regulation slot
    assert np.allclose(V[0, :], pi * X[1])
    assert V[0, 0] > 0
    # at the end of the ninth: k tokens are worth pi * X(max(k, 1))
    assert np.isclose(V[1, -1], pi * X[1]) and np.isclose(V[2, -1], pi * X[2])
    # the top-up comes whether or not the last token is lost, so the first token's
    # marginal value is unchanged; the second token gains a little
    V0, _, _ = token_value_dp(o, lam_per_half=0.6)
    assert np.allclose(V[1] - V[0], V0[1] - V0[0])
    assert np.all(V[2] - V[1] >= V0[2] - V0[1] - 1e-12) and (V[2, -2] - V[1, -2]) > (V0[2, -2] - V0[1, -2])


def test_extras_value_is_a_fixed_point():
    from abs_judgment.policy import extras_value

    o = _opps()
    rho = 0.27
    Vx = extras_value(o, 0.6, rho)
    X = Vx[:, 0]
    assert np.allclose(Vx[:, -1], rho * X[np.maximum(np.arange(3), 1)])
    assert Vx[2, 0] > Vx[1, 0] > Vx[0, 0] > 0


def test_calibrated_belief_matches_true_flip_rate_in_bins():
    from abs_judgment.policy import fit_logit_1d, subjective_p

    rng = np.random.default_rng(1)
    # a near-edge-like population: the umpire is right on most pitches
    e = np.concatenate([rng.uniform(-3, 0, 80_000), rng.uniform(0, 3, 20_000)])
    e_perc = e + rng.normal(0, 2.6, len(e))
    y = e > 0
    ab = fit_logit_1d(e_perc, y)
    p = subjective_p(e_perc, 2.6, "calibrated", ab)
    bins = pd.qcut(p, 5)
    t = pd.DataFrame({"p": p, "y": y}).groupby(bins, observed=True).mean()
    assert np.all(np.abs(t.p - t.y) < 0.03)
    # the flat prior is the miscalibration the review found: overconfident at the top
    pf = subjective_p(e_perc, 2.6, "flat")
    top = pf >= 0.8
    assert pf[top].mean() - y[top].mean() > 0.2


def test_perceived_margin_inverts_subjective_p():
    from abs_judgment.policy import perceived_margin, subjective_p

    for model, ab in (("flat", None), ("calibrated", (-1.2, 0.33))):
        m = perceived_margin(np.array([0.1, 0.4, 0.7]), 2.6, model, ab)
        assert np.allclose(subjective_p(m, 2.6, model, ab), [0.1, 0.4, 0.7])


def test_simulation_uses_the_dp_token_rule_and_pairs_by_game():
    from abs_judgment.policy import paired_difference, simulate_policy

    rng = np.random.default_rng(2)
    rows = []
    for g in range(40):
        for inning in range(1, 12 if g % 5 == 0 else 10):
            for half in ("top", "bottom"):
                for _ in range(3):
                    rows.append({"game_pk": g, "home_id": 1, "away_id": 2, "team": 2 if half == "top" else 1,
                                 "inning": inning, "half": half, "p": rng.uniform(), "true_flip": rng.uniform() < 0.2,
                                 "stake": rng.gamma(2.0, 0.08)})
    o = pd.DataFrame(rows)
    V, _, _ = token_value_dp(o, lam_per_half=3, p_extras=0.2, rho=0.27)
    d = simulate_policy(o, "dynamic", V)
    s = simulate_policy(o, "static", V)
    assert len(d["per_game"]) == 40 and np.isclose(d["per_game"].runs.sum(), d["runs"])
    # oracle never loses a token, so it challenges every flip
    orc = simulate_policy(o, "oracle")
    assert orc["challenges"] == int(o.true_flip.sum())
    m, se = paired_difference(d["per_game"], s["per_game"])
    assert np.isclose(m, d["runs_per_team_162"] - s["runs_per_team_162"]) and se > 0
