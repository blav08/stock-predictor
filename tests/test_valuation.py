"""Sanity checks of the DCF maths against closed-form results."""
import numpy as np
import pandas as pd

from stockpred.valuation_mc import CompanyInputs, DCFAssumptions, dcf_per_share, simulate


def company(**kw):
    base = dict(ticker="TEST", fiscal_year=2025, revenue=100.0, net_cash=0.0, shares=1.0, price=100.0)
    return CompanyInputs(**{**base, **kw})


def test_dcf_matches_gordon_growth_when_growth_is_constant():
    # constant growth g everywhere -> value = FCF_1 / (wacc - g)
    g, m, w = 0.03, 0.10, 0.08
    v = dcf_per_share(company(), g, m, w, g, high_years=5, horizon=10)[0]
    expected = 100 * (1 + g) * m / (w - g)
    assert np.isclose(v, expected)


def test_net_cash_and_shares_scale_value():
    base = dcf_per_share(company(), 0.03, 0.1, 0.08, 0.03, 5, 10)[0]
    assert np.isclose(dcf_per_share(company(net_cash=50), 0.03, 0.1, 0.08, 0.03, 5, 10)[0], base + 50)
    assert np.isclose(dcf_per_share(company(shares=2), 0.03, 0.1, 0.08, 0.03, 5, 10)[0], base / 2)


def test_simulation_is_reproducible_and_zero_sd_is_deterministic():
    a = DCFAssumptions(growth_mean=0.05, growth_sd=0.0, margin_mean=0.1, margin_sd=0.0,
                       wacc_sd=0.0, terminal_growth_sd=0.0, n_sims=500)
    r1, r2 = simulate(company(), a), simulate(company(), a)
    assert np.allclose(r1.values, r2.values)
    assert np.allclose(r1.values, r1.values[0])        # no randomness left


def test_higher_wacc_lowers_value():
    lo = dcf_per_share(company(), 0.05, 0.1, 0.07, 0.02, 5, 10)[0]
    hi = dcf_per_share(company(), 0.05, 0.1, 0.11, 0.02, 5, 10)[0]
    assert hi < lo
