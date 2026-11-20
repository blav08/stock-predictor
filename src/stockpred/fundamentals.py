"""Turn raw SEC XBRL 'companyfacts' JSON into a clean annual fundamentals table.

Key ideas:
* Companies use different XBRL tags for the same concept -> each metric has a
  priority-ordered list of tags; the first one available for a period wins.
* Only annual data from 10-K filings is used. Duration items (revenue, cash flow)
  must cover ~1 year; instant items (assets, equity) are point-in-time.
* Each 10-K also repeats prior years. We keep the value from the EARLIEST filing
  ("as first reported") and store its filing date -> no look-ahead bias later.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

ANNUAL_FORMS = {"10-K", "10-K/A", "10-KT"}
MIN_DAYS, MAX_DAYS = 350, 380  # accepts 52/53-week fiscal years


@dataclass(frozen=True)
class Concept:
    tags: tuple[str, ...]
    kind: Literal["duration", "instant"] = "duration"
    unit: str = "USD"
    taxonomy: str = "us-gaap"


CONCEPTS: dict[str, Concept] = {
    # --- income statement (duration)
    "revenue": Concept((
        "Revenues",
        "RevenuesNetOfInterestExpense",          # banks / brokers (GS, MS, WFC)
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
    )),
    "gross_profit": Concept(("GrossProfit",)),
    "operating_income": Concept(("OperatingIncomeLoss",)),
    "net_income": Concept(("NetIncomeLoss", "ProfitLoss")),
    "rnd": Concept(("ResearchAndDevelopmentExpense",)),
    "eps_diluted": Concept(("EarningsPerShareDiluted",), unit="USD/shares"),
    "diluted_shares": Concept(("WeightedAverageNumberOfDilutedSharesOutstanding",), unit="shares"),
    # --- cash flow (duration)
    "operating_cash_flow": Concept((
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    )),
    "capex": Concept((
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
    )),
    # --- balance sheet (instant)
    "total_assets": Concept(("Assets",), kind="instant"),
    "total_liabilities": Concept(("Liabilities",), kind="instant"),
    "equity": Concept((
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ), kind="instant"),
    "cash": Concept((
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ), kind="instant"),
    "long_term_debt": Concept(("LongTermDebtNoncurrent", "LongTermDebt"), kind="instant"),
    "current_assets": Concept(("AssetsCurrent",), kind="instant"),
    "current_liabilities": Concept(("LiabilitiesCurrent",), kind="instant"),
}


def fiscal_year_of(period_end: pd.Series) -> pd.Series:
    """Compustat convention: FY ending Jan-May belongs to the previous calendar year.
    (Walmart's year ending 31 Jan 2026 is FY2025.)  Note: SEC's own 'fy' field is the
    fiscal year of the *filing*, not of the period, so it must not be used here."""
    return np.where(period_end.dt.month >= 6, period_end.dt.year, period_end.dt.year - 1)


def extract_metric(facts: dict, concept: Concept) -> pd.DataFrame:
    """Annual values for one concept -> columns [period_end, value, filed, tag]."""
    rows = []
    taxonomy = facts.get("facts", {}).get(concept.taxonomy, {})
    for priority, tag in enumerate(concept.tags):
        for entry in taxonomy.get(tag, {}).get("units", {}).get(concept.unit, []):
            if entry.get("form") not in ANNUAL_FORMS:
                continue
            end = pd.Timestamp(entry["end"])
            if concept.kind == "duration":
                if "start" not in entry:
                    continue
                days = (end - pd.Timestamp(entry["start"])).days
                if not MIN_DAYS <= days <= MAX_DAYS:
                    continue  # quarterly or multi-year value inside a 10-K
            rows.append({
                "period_end": end,
                "value": float(entry["val"]),
                "filed": pd.Timestamp(entry["filed"]),
                "tag": tag,
                "priority": priority,
            })

    if not rows:
        return pd.DataFrame(columns=["period_end", "value", "filed", "tag"])

    df = pd.DataFrame(rows)
    # 1) as first reported: earliest filing per (period, tag)
    df = df.sort_values("filed").drop_duplicates(["period_end", "tag"], keep="first")
    # 2) best tag per period: lowest priority number wins
    df = df.sort_values("priority").drop_duplicates("period_end", keep="first")
    return df.drop(columns="priority").sort_values("period_end").reset_index(drop=True)


def company_fundamentals(facts: dict, start_year: int = 2010) -> pd.DataFrame:
    """Wide annual table for one company: one row per fiscal year, one column per metric."""
    values, filed = {}, {}
    for name, concept in CONCEPTS.items():
        m = extract_metric(facts, concept)
        if m.empty:
            continue
        m["fiscal_year"] = fiscal_year_of(m["period_end"])
        # if a fiscal year appears twice (fiscal-year change), keep the latest period
        m = m.sort_values("period_end").drop_duplicates("fiscal_year", keep="last")
        values[name] = m.set_index("fiscal_year")["value"]
        filed[name] = m.set_index("fiscal_year")["filed"]

    if not values:
        return pd.DataFrame()

    df = pd.DataFrame(values)
    # date of the original 10-K for that year = earliest filing among all metrics
    # (a metric first seen in a later 10-K only means the company started tagging it later)
    df["available_date"] = pd.DataFrame(filed).min(axis=1)
    df = df[df.index >= start_year].sort_index()
    # reindex to a continuous year range so growth rates never skip a missing year
    df = df.reindex(range(df.index.min(), df.index.max() + 1)) if not df.empty else df
    df.index.name = "fiscal_year"
    return add_derived_metrics(df)


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    """Division that returns NaN instead of inf for zero or negative-equity edge cases."""
    return (a / b).replace([np.inf, -np.inf], np.nan)


def repair_share_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-check diluted shares against net income / EPS (the definition of EPS).

    * missing share count (ExxonMobil, Visa, Berkshire do not tag it) -> use the implied one
    * reported count off by > 50 % (McDonald's tags FY2023+ in MILLIONS: 732.3 instead of
      732,300,000) -> use the implied one
    Skipped when EPS is tiny (|EPS| < 0.10), where the implied count is unstable."""
    if not {"net_income", "eps_diluted"} <= set(df.columns):
        return df
    df = df.copy()
    usable = df["eps_diluted"].abs() >= 0.10
    implied = _safe_div(df["net_income"], df["eps_diluted"]).where(lambda x: (x > 0) & usable)
    if "diluted_shares" not in df:
        df["diluted_shares"] = np.nan
    ratio = df["diluted_shares"] / implied
    wrong = implied.notna() & (df["diluted_shares"].isna() | (ratio < 0.67) | (ratio > 1.5))
    df["diluted_shares_implied"] = wrong
    df.loc[wrong, "diluted_shares"] = implied[wrong]
    return df


def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = repair_share_counts(df)
    cols = df.columns
    if {"operating_cash_flow", "capex"} <= set(cols):
        df["free_cash_flow"] = df["operating_cash_flow"] - df["capex"]
    if "revenue" in cols:
        for num, name in [
            ("gross_profit", "gross_margin"),
            ("operating_income", "operating_margin"),
            ("net_income", "net_margin"),
            ("free_cash_flow", "fcf_margin"),
        ]:
            if num in df.columns:
                df[name] = _safe_div(df[num], df["revenue"])
        df["revenue_growth"] = df["revenue"].pct_change(fill_method=None)
    if {"net_income", "equity"} <= set(cols):
        # equity <= 0 makes ROE meaningless (e.g. heavy buybacks) -> NaN
        df["roe"] = _safe_div(df["net_income"], df["equity"].where(df["equity"] > 0))
    if {"total_liabilities", "equity"} <= set(cols):
        df["liabilities_to_equity"] = _safe_div(
            df["total_liabilities"], df["equity"].where(df["equity"] > 0)
        )
    if "eps_diluted" in cols:
        df["eps_growth"] = df["eps_diluted"].pct_change(fill_method=None)
    return df


def adjust_for_splits(df: pd.DataFrame, splits: pd.Series) -> pd.DataFrame:
    """Restate per-share values to today's share basis.

    As-first-reported EPS of Apple FY2012 is $44.15 - before the 7:1 (2014) and
    4:1 (2020) splits. For every fiscal year we multiply all split ratios that
    happened AFTER that year's filing date: EPS is divided, share counts multiplied.
    `splits`: index = split date, value = ratio (4.0 means 4-for-1)."""
    if splits is None or splits.empty or "available_date" not in df:
        return df
    df = df.copy()
    split_dates = pd.to_datetime(splits.index).tz_localize(None)
    ratios = splits.to_numpy(dtype=float)

    def factor(available: pd.Timestamp) -> float:
        if pd.isna(available):
            return 1.0
        return float(np.prod(ratios[split_dates > available])) or 1.0

    f = df["available_date"].map(factor)
    if "eps_diluted" in df:
        df["eps_diluted"] = df["eps_diluted"] / f
    if "diluted_shares" in df:
        df["diluted_shares"] = df["diluted_shares"] * f
    if "eps_diluted" in df:
        df["eps_growth"] = df["eps_diluted"].pct_change(fill_method=None)
    return df


def merge_facts(fact_payloads: list[dict]) -> dict:
    """Combine companyfacts of several CIKs (e.g. a new holding company + its
    predecessor). Fact lists are simply concatenated - extract_metric() already
    de-duplicates by period and keeps the earliest filing."""
    merged: dict = {"facts": {}}
    for payload in fact_payloads:
        for taxonomy, tags in payload.get("facts", {}).items():
            for tag, node in tags.items():
                target = merged["facts"].setdefault(taxonomy, {}).setdefault(tag, {"units": {}})
                for unit, entries in node.get("units", {}).items():
                    target["units"].setdefault(unit, []).extend(entries)
    return merged


def build_panel(
    client, universe: pd.DataFrame, start_year: int = 2010, splits: dict | None = None,
    predecessors: dict[int, list[int]] | None = None,
) -> pd.DataFrame:
    """Long table for the whole universe: rows = (ticker, fiscal_year).
    `splits`: optional {ticker: split Series} used to restate per-share data.
    `predecessors`: {cik: [older ciks]} for companies that changed legal entity."""
    frames, empty = [], []
    splits, predecessors = splits or {}, predecessors or {}
    for row in universe.itertuples(index=False):
        ciks = [int(row.cik), *predecessors.get(int(row.cik), [])]
        try:
            facts = merge_facts([client.company_facts(c) for c in ciks])
        except Exception as exc:          # e.g. 404: company never filed XBRL
            print(f"  ! {row.ticker} (CIK {row.cik}): {exc}")
            empty.append(row.ticker)
            continue
        df = company_fundamentals(facts, start_year)
        if df.empty:
            empty.append(row.ticker)
            continue
        df = adjust_for_splits(df, splits.get(row.ticker))
        df.insert(0, "ticker", row.ticker)
        frames.append(df.reset_index())
    if empty:
        warnings.warn(f"No annual fundamentals found for: {', '.join(empty)} - "
                      "check the CIK (new legal entity? add it to cik_predecessors)")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
