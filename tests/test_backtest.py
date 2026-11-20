"""Backtest engine sanity checks on synthetic data with a known answer."""
import numpy as np
import pandas as pd

from stockpred.backtest import perf_stats, quintile_backtest
from stockpred.factors import composite_score


def synthetic(n_months=24, n_names=100, signal=0.01, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for m in pd.date_range("2020-01-31", periods=n_months, freq="ME"):
        score = rng.normal(size=n_names)
        ret = signal * score + rng.normal(scale=0.01, size=n_names)
        rows += [{"month_end": m, "ticker": f"T{i}", "score": s, "fwd_ret": r}
                 for i, (s, r) in enumerate(zip(score, ret))]
    return pd.DataFrame(rows)


def test_perfect_signal_is_monotonic_and_significant():
    s = quintile_backtest(synthetic(), "score").summary()
    assert s["monotonic"]
    assert s["LS_tstat"] > 5 and s["IC_mean"] > 0.5


def test_pure_noise_has_no_edge():
    s = quintile_backtest(synthetic(n_months=60, signal=0.0), "score").summary()
    assert abs(s["IC_tstat"]) < 3


def test_min_names_filter():
    r = quintile_backtest(synthetic(n_names=30), "score", min_names=50)
    assert r.quintiles.empty


def test_perf_stats_constant_return():
    st = perf_stats(pd.Series([0.01] * 12))
    assert np.isclose(st["ann_return"], 1.01 ** 12 - 1)
    assert st["max_drawdown"] == 0


def test_composite_flips_negative_direction():
    t = pd.DataFrame({"month_end": pd.Timestamp("2020-01-31"), "fscore": [1, 2, 3],
                      "text_change": [0.9, 0.5, 0.1]})
    c = composite_score(t, ["fscore", "text_change"])
    assert list(c.rank()) == [1, 2, 3]          # high F-score + low text change = best
