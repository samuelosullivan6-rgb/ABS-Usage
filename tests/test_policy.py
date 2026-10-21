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
