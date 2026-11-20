"""Monte Carlo DCF valuation.

Instead of one "fair value", we draw thousands of plausible futures and look at the
DISTRIBUTION of intrinsic value per share.

Model (per simulated path):
  revenue_t  = revenue_{t-1} * (1 + g_t)
      g_t    = g_high                  for the first `high_growth_years`
               linear fade to g_term   until `horizon` (default year 10)
  FCF_t      = revenue_t * fcf_margin
  value      = sum FCF_t / (1+wacc)^t  +  TV / (1+wacc)^horizon
  TV         = FCF_horizon * (1+g_term) / (wacc - g_term)      (Gordon growth)
  equity     = enterprise value + net cash ;  per share = equity / diluted shares

Random inputs: g_high, fcf_margin, wacc, g_term (one draw per path, kept for all years -
a company is either a "high-growth" or a "low-growth" world, not both in alternating years).

⚠️ The result is only as good as the input distributions. Defaults are estimated from
the company's own history, which extrapolates the past - adjust them with judgement.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

FINANCIAL_SUB_INDUSTRIES = ("bank", "insurance", "capital markets", "multi-sector holdings", "consumer finance",
                            "asset management", "investment banking", "reinsurance")


def is_financial(sub_industry: str | None) -> bool:
    return bool(sub_industry) and any(k in sub_industry.lower() for k in FINANCIAL_SUB_INDUSTRIES)


@dataclass(frozen=True)
class DCFAssumptions:
    growth_mean: float            # revenue growth in the high-growth phase
    growth_sd: float
    margin_mean: float            # free cash flow / revenue
    margin_sd: float
    wacc_mean: float = 0.09       # discount rate
    wacc_sd: float = 0.01
    terminal_growth: float = 0.025
    terminal_growth_sd: float = 0.005
    high_growth_years: int = 5
    horizon: int = 10
    n_sims: int = 10_000
    seed: int = 42                # fixed seed -> moving a slider changes only that input

    def with_(self, **changes) -> "DCFAssumptions":
        return replace(self, **changes)


@dataclass
class CompanyInputs:
    ticker: str
    fiscal_year: int
    revenue: float
    net_cash: float               # cash - long-term debt (approximation)
    shares: float                 # diluted, split-adjusted
    price: float
    warnings: list[str] = field(default_factory=list)
    dcf_applicable: bool = True   # False for banks / insurers


@dataclass
class MCResult:
    values: np.ndarray            # intrinsic value per share, one per path
    price: float
    warnings: list[str] = field(default_factory=list)   # input warnings + simulation checks

    @property
    def prob_undervalued(self) -> float:
        return float(np.mean(self.values > self.price))

    def percentiles(self, qs=(5, 25, 50, 75, 95)) -> dict[int, float]:
        return {q: float(np.percentile(self.values, q)) for q in qs}

    @property
    def margin_of_safety(self) -> float:
        """Median value vs. price: +30 % means the median path is worth 30 % more than the price."""
        return float(np.median(self.values) / self.price - 1)


# ---------------------------------------------------------------- estimation
def estimate_inputs(panel: pd.DataFrame, ticker: str, price: float,
                    sub_industry: str | None = None, fallback_shares: float | None = None,
                    margin_basis: str = "fcf", lookback: int = 5,
                    growth_cap: tuple[float, float] = (-0.05, 0.30)
                    ) -> tuple[CompanyInputs, DCFAssumptions]:
    """Default inputs from the company's last `lookback` fiscal years.

    margin_basis:
      "fcf" - free cash flow margin (cash reality, but depressed during big capex cycles:
              Amazon, Walmart, Oracle building data centres look almost worthless)
      "net" - net income margin as a proxy for 'normalised' FCF once investment slows."""
    hist = panel[panel["ticker"] == ticker].sort_values("fiscal_year")
    hist = hist.dropna(subset=["revenue"])
    if hist.empty:
        raise ValueError(f"No revenue history for {ticker}")
    last = hist.iloc[-1]
    recent = hist.tail(lookback)
    warns: list[str] = []

    # growth: median is robust to one-off years (acquisitions, COVID)
    g = recent["revenue_growth"].dropna()
    g_mean = float(g.median()) if len(g) else 0.05
    if not growth_cap[0] <= g_mean <= growth_cap[1]:
        warns.append(f"Historical growth {g_mean:.0%} capped to {np.clip(g_mean, *growth_cap):.0%} "
                     "- hyper-growth rarely lasts 5 more years.")
        g_mean = float(np.clip(g_mean, *growth_cap))
    g_sd = max(float(g.std()) if len(g) > 1 else 0.05, 0.03)
    g_sd = min(g_sd, 0.15)

    # margin: FCF or net margin; FCF falls back to net margin when capex is not tagged
    m = pd.Series(dtype=float)
    if margin_basis == "fcf" and "fcf_margin" in recent:
        m = recent["fcf_margin"].dropna()
        if m.empty:
            warns.append("FCF margin unavailable - using net margin instead.")
        elif "net_margin" in recent and recent["net_margin"].mean() > 2 * m.mean() > 0:
            warns.append(f"FCF margin ({m.mean():.0%}) is far below net margin "
                         f"({recent['net_margin'].mean():.0%}) - heavy investment phase? "
                         "Try margin basis 'net'.")
    if m.empty:
        m = recent["net_margin"].dropna()
    m_mean = float(m.mean()) if len(m) else 0.10
    m_sd = max(float(m.std()) if len(m) > 1 else 0.03, 0.02)
    if m_mean <= 0:
        warns.append("Average FCF margin is negative - a DCF on current margins implies value < 0.")

    cash = last.get("cash", np.nan)
    debt = last.get("long_term_debt", np.nan)
    if pd.isna(cash) or pd.isna(debt):
        warns.append("Cash or long-term debt missing - treated as 0.")
    net_cash = float(np.nan_to_num(cash) - np.nan_to_num(debt))

    shares = float(last.get("diluted_shares", np.nan))
    if pd.isna(shares) or shares <= 0:
        # multi-class companies (Visa) report EPS per class only -> use market cap / price
        if fallback_shares and fallback_shares > 0:
            shares = float(fallback_shares)
            warns.append("No diluted share count in XBRL - using market cap / price.")
        else:
            raise ValueError(f"No diluted share count for {ticker}")
    if bool(last.get("diluted_shares_implied", False)):
        warns.append("Diluted shares implied from net income / EPS (not tagged by the company).")

    if is_financial(sub_industry):
        warns.append(f"{sub_industry}: free cash flow is not meaningful for financial firms - "
                     "a DCF is the wrong tool (use P/B or a dividend model).")

    inputs = CompanyInputs(ticker, int(last["fiscal_year"]), float(last["revenue"]),
                           net_cash, shares, float(price), warns,
                           dcf_applicable=not is_financial(sub_industry))
    assumptions = DCFAssumptions(growth_mean=round(g_mean, 3), growth_sd=round(g_sd, 3),
                                 margin_mean=round(m_mean, 3), margin_sd=round(m_sd, 3))
    return inputs, assumptions


# ---------------------------------------------------------------- core maths
def dcf_per_share(inputs: CompanyInputs, g_high, margin, wacc, g_term,
                  high_years: int, horizon: int) -> np.ndarray:
    """Vectorised DCF: every argument may be a scalar or an array of n paths."""
    g_high, margin, wacc, g_term = np.broadcast_arrays(*map(np.atleast_1d, (g_high, margin, wacc, g_term)))
    t = np.arange(1, horizon + 1)                                    # years 1..horizon
    fade = np.clip((t - high_years) / max(horizon - high_years, 1), 0, 1)   # 0 during high growth -> 1
    growth = g_high[:, None] + (g_term - g_high)[:, None] * fade[None, :]   # (n, horizon)
    revenue = inputs.revenue * np.cumprod(1 + growth, axis=1)
    fcf = revenue * margin[:, None]
    discount = (1 + wacc[:, None]) ** t[None, :]
    pv_fcf = (fcf / discount).sum(axis=1)
    terminal = fcf[:, -1] * (1 + g_term) / (wacc - g_term)
    pv_terminal = terminal / discount[:, -1]
    return (pv_fcf + pv_terminal + inputs.net_cash) / inputs.shares


def simulate(inputs: CompanyInputs, a: DCFAssumptions) -> MCResult:
    rng = np.random.default_rng(a.seed)
    n = a.n_sims
    # draw standard normals once, then scale -> changing one slider does not reshuffle the others
    z = rng.standard_normal((4, n))
    g_high = np.clip(a.growth_mean + a.growth_sd * z[0], -0.30, 0.80)
    margin = np.clip(a.margin_mean + a.margin_sd * z[1], -0.30, 0.70)
    wacc = np.clip(a.wacc_mean + a.wacc_sd * z[2], 0.04, 0.20)
    g_term = a.terminal_growth + a.terminal_growth_sd * z[3]
    g_term = np.minimum(g_term, wacc - 0.01)       # Gordon formula explodes when g -> wacc
    values = dcf_per_share(inputs, g_high, margin, wacc, g_term, a.high_growth_years, a.horizon)

    warns = list(inputs.warnings)           # copy - never mutate the inputs
    ratio = np.median(values) / inputs.price
    if ratio > 10 or ratio < 0.1:
        warns.append(f"Median value is {ratio:.1f}x the price - check the share basis "
                     "(multi-class shares, e.g. Berkshire reports Class A equivalents).")
    return MCResult(values, inputs.price, warns)


def sensitivity_grid(inputs: CompanyInputs, a: DCFAssumptions,
                     waccs=None, growths=None) -> pd.DataFrame:
    """Deterministic value per share (all other inputs at their mean) over WACC x growth."""
    waccs = np.round(np.arange(a.wacc_mean - 0.02, a.wacc_mean + 0.0201, 0.005), 4) if waccs is None else waccs
    growths = np.round(np.arange(a.growth_mean - 0.06, a.growth_mean + 0.0601, 0.02), 4) if growths is None else growths
    W, G = np.meshgrid(waccs, growths)
    tg = np.minimum(a.terminal_growth, W - 0.01)
    v = dcf_per_share(inputs, G.ravel(), a.margin_mean, W.ravel(), tg.ravel(),
                      a.high_growth_years, a.horizon).reshape(G.shape)
    return pd.DataFrame(v, index=pd.Index(growths, name="growth"), columns=pd.Index(waccs, name="wacc"))


def screen_universe(panel: pd.DataFrame, universe: pd.DataFrame, prices: pd.Series,
                    margin_basis: str = "fcf", **overrides) -> pd.DataFrame:
    """Run the default Monte Carlo for every company. `overrides` apply to all (e.g. wacc_mean=0.1).
    Financial companies are kept in the table but ranked last (DCF not applicable)."""
    rows = []
    for r in universe.itertuples(index=False):
        price = prices.get(r.ticker, np.nan)
        try:
            mcap = getattr(r, "market_cap", np.nan)
            fallback = mcap / price if price and not pd.isna(mcap) else None
            inp, a = estimate_inputs(panel, r.ticker, price, getattr(r, "sub_industry", None),
                                     fallback, margin_basis=margin_basis)
            res = simulate(inp, a.with_(**overrides))
        except (ValueError, KeyError) as exc:
            rows.append({"ticker": r.ticker, "error": str(exc)})
            continue
        p = res.percentiles()
        rows.append({
            "ticker": r.ticker, "name": r.name, "sector": r.sector, "price": price,
            "dcf_applicable": inp.dcf_applicable,
            "value_p5": p[5], "value_median": p[50], "value_p95": p[95],
            "margin_of_safety": res.margin_of_safety, "prob_undervalued": res.prob_undervalued,
            "growth_assumed": a.growth_mean, "fcf_margin_assumed": a.margin_mean,
            "warnings": " | ".join(res.warnings),
        })
    out = pd.DataFrame(rows)
    return out.sort_values(["dcf_applicable", "prob_undervalued"], ascending=[False, False]) \
              .reset_index(drop=True)
