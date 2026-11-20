"""Point-in-time factor table for the backtest.

Every factor value at a month end may only use information PUBLIC at that date:
fundamentals become usable on the 10-K filing date (`available_date`), prices are
taken at the month-end close. pd.merge_asof does the "latest known value" lookup.

Factors (direction: higher = better unless stated):
  fscore            Piotroski F-score 0-9 (Piotroski 2000) - financial health / improvement
  gross_prof        gross profit / total assets (Novy-Marx 2013) - quality
  earnings_yield    net income / market cap - value
  fcf_yield         free cash flow / market cap - value
  revenue_growth    YoY revenue growth - growth
  mom_12_1          price return t-12 -> t-1 (Jegadeesh & Titman 1993) - momentum, sanity check
  dcf_mos           Monte Carlo DCF median value / price - 1 - our valuation model
  text_change       share of new sentences in Risk Factors (LOWER = better, Lazy Prices)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .valuation_mc import estimate_inputs, is_financial, simulate

FACTOR_DIRECTION = {
    "fscore": 1, "gross_prof": 1, "earnings_yield": 1, "fcf_yield": 1,
    "revenue_growth": 1, "mom_12_1": 1, "dcf_mos": 1, "text_change": -1,
}


def _safe_div(a, b):
    return (a / b).replace([np.inf, -np.inf], np.nan)


def add_piotroski(panel: pd.DataFrame) -> pd.DataFrame:
    """Piotroski F-score: 9 binary signals comparing this fiscal year with the previous one.
    Requires a continuous fiscal-year index per ticker (company_fundamentals guarantees it)."""
    df = panel.sort_values(["ticker", "fiscal_year"]).copy()
    prev = df.groupby("ticker").shift(1)
    assets_begin = prev["total_assets"].fillna(df["total_assets"])        # beginning-of-year assets

    roa = _safe_div(df["net_income"], assets_begin)
    roa_prev = _safe_div(prev["net_income"], prev.groupby(df["ticker"])["total_assets"].shift(1)
                         .fillna(prev["total_assets"]))
    cfo = _safe_div(df["operating_cash_flow"], assets_begin)
    lev = _safe_div(df["long_term_debt"], df["total_assets"])
    lev_prev = _safe_div(prev["long_term_debt"], prev["total_assets"])
    cr = _safe_div(df["current_assets"], df["current_liabilities"])
    cr_prev = _safe_div(prev["current_assets"], prev["current_liabilities"])
    turnover = _safe_div(df["revenue"], assets_begin)
    turnover_prev = _safe_div(prev["revenue"], prev.groupby(df["ticker"])["total_assets"].shift(1)
                              .fillna(prev["total_assets"]))

    def signal(cond: pd.Series, *inputs: pd.Series) -> pd.Series:
        # 1/0 where all inputs are known, NaN otherwise (missing data != failed test)
        known = pd.concat(inputs, axis=1).notna().all(axis=1)
        return cond.astype(float).where(known)

    signals = pd.DataFrame({
        "f_roa": signal(roa > 0, roa),
        "f_cfo": signal(cfo > 0, cfo),
        "f_droa": signal(roa > roa_prev, roa, roa_prev),
        "f_accrual": signal(cfo > roa, cfo, roa),
        "f_dlever": signal(lev <= lev_prev, lev, lev_prev),
        "f_dliquid": signal(cr > cr_prev, cr, cr_prev),
        "f_noissue": signal(df["diluted_shares"] <= prev["diluted_shares"] * 1.005,
                            df["diluted_shares"], prev["diluted_shares"]),
        "f_dmargin": signal(df["gross_margin"] > prev["gross_margin"], df["gross_margin"], prev["gross_margin"]),
        "f_dturn": signal(turnover > turnover_prev, turnover, turnover_prev),
    }, index=df.index)
    available = signals.notna().sum(axis=1)
    # rescale to 0-9 when some signals are missing (banks have no gross margin); need >= 6
    df["fscore"] = (signals.sum(axis=1) * 9 / available).where(available >= 6)
    df["gross_prof"] = _safe_div(df["gross_profit"], df["total_assets"])
    return pd.concat([df, signals], axis=1)


def dcf_values(panel: pd.DataFrame, sub_industry: dict[str, str] | None = None,
               n_sims: int = 2_000, min_years: int = 4) -> pd.DataFrame:
    """Median Monte Carlo DCF value per share for every (ticker, fiscal_year), using ONLY
    data up to that fiscal year. The value does not depend on the price, so it is computed
    once per 10-K and compared with every later month-end price."""
    rows = []
    sub_industry = sub_industry or {}
    for ticker, hist in panel.groupby("ticker"):
        if is_financial(sub_industry.get(ticker)):
            continue
        hist = hist.sort_values("fiscal_year")
        for fy in hist["fiscal_year"][min_years - 1:]:
            past = hist[hist["fiscal_year"] <= fy]
            if pd.isna(past.iloc[-1]["revenue"]):
                continue
            try:
                inp, a = estimate_inputs(past, ticker, price=1.0)
                res = simulate(inp, a.with_(n_sims=n_sims))
            except (ValueError, KeyError):
                continue
            rows.append({"ticker": ticker, "fiscal_year": int(fy),
                         "dcf_value": float(np.median(res.values))})
    return pd.DataFrame(rows)


def build_factor_table(membership: pd.DataFrame, panel: pd.DataFrame, tri: pd.DataFrame,
                       close: pd.DataFrame, dcf: pd.DataFrame | None = None,
                       text: pd.DataFrame | None = None, max_age_days: int = 550) -> pd.DataFrame:
    """One row per (month_end, ticker) member with all factors + next month's return.

    membership: [month_end, ticker] (point-in-time index members, current tickers)
    tri / close: month-end total-return index and split-adjusted close (wide)."""
    base = membership.sort_values("month_end").copy()

    # --- prices: forward return (target) and 12-1 momentum
    fwd = (tri.shift(-1) / tri - 1).stack().rename("fwd_ret")
    mom = (tri.shift(1) / tri.shift(12) - 1).stack().rename("mom_12_1")
    px = close.stack().rename("close")
    prices = pd.concat([fwd, mom, px], axis=1)
    prices.index.names = ["month_end", "ticker"]
    base = base.merge(prices.reset_index(), on=["month_end", "ticker"], how="left")

    # --- fundamentals: latest 10-K filed on or before the month end
    f = panel.dropna(subset=["available_date"]).copy()
    if dcf is not None and len(dcf):
        f = f.merge(dcf, on=["ticker", "fiscal_year"], how="left")
    keep = ["ticker", "fiscal_year", "available_date", "fscore", "gross_prof", "net_income",
            "free_cash_flow", "revenue_growth", "diluted_shares"] + \
           (["dcf_value"] if "dcf_value" in f else [])
    f = f[keep].sort_values("available_date")
    base = pd.merge_asof(base.sort_values("month_end"), f, left_on="month_end",
                         right_on="available_date", by="ticker", direction="backward")
    stale = (base["month_end"] - base["available_date"]).dt.days > max_age_days
    base.loc[stale, keep[1:]] = np.nan              # older than ~18 months = no current 10-K

    mcap = base["close"] * base["diluted_shares"]
    base["market_cap"] = mcap
    base["earnings_yield"] = _safe_div(base["net_income"], mcap)
    base["fcf_yield"] = _safe_div(base["free_cash_flow"], mcap)
    if "dcf_value" in base:
        base["dcf_mos"] = _safe_div(base["dcf_value"], base["close"]) - 1

    # --- text: latest Risk Factors change known at the month end
    if text is not None and len(text):
        t = (text[(text["section"] == "risk_factors") & text["share_new_sentences"].notna()]
             [["ticker", "filing_date", "share_new_sentences"]]
             .rename(columns={"share_new_sentences": "text_change"}).sort_values("filing_date"))
        base = pd.merge_asof(base.sort_values("month_end"), t, left_on="month_end",
                             right_on="filing_date", by="ticker", direction="backward")
        base.loc[(base["month_end"] - base["filing_date"]).dt.days > max_age_days, "text_change"] = np.nan
    return base.sort_values(["month_end", "ticker"]).reset_index(drop=True)


def composite_score(table: pd.DataFrame, factors: list[str], min_share: float = 0.5,
                    group_col: str | None = None) -> pd.Series:
    """Average cross-sectional percentile rank (0-1) of the chosen factors, per month.
    Ranking makes factors with different units comparable and robust to outliers.

    group_col="sector" ranks companies only against their own sector (sector-neutral):
    banks are always 'cheap' on P/E and software always has high gross margins - without
    neutralising, the score mostly picks sectors instead of companies."""
    keys = ["month_end"] + ([group_col] if group_col else [])
    ranks = pd.DataFrame({
        f: table.groupby(keys)[f].rank(pct=True) * (1 if FACTOR_DIRECTION[f] > 0 else -1)
           + (0 if FACTOR_DIRECTION[f] > 0 else 1)
        for f in factors
    })
    enough = ranks.notna().mean(axis=1) >= min_share
    return ranks.mean(axis=1).where(enough)
