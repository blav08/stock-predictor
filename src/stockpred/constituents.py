"""Historical S&P 500 membership (point in time) and mapping of old tickers to CIK.

Source: github.com/fja05680/sp500 (MIT licence) - daily list of index members since 1996,
built from Andreas Clenow's data (Trading Evolved) + Wikipedia changes.

Why it matters: backtesting only today's members = survivorship bias. Companies that
dropped out (often after poor performance) silently disappear and the strategy looks better.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

HISTORY_URL = ("https://raw.githubusercontent.com/fja05680/sp500/master/"
               "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv")


def fetch_history(cache_path: Path, refresh: bool = False) -> pd.DataFrame:
    """One row per change date: [date, tickers (list)]."""
    cache_path = Path(cache_path)
    if not cache_path.exists() or refresh:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(requests.get(HISTORY_URL, timeout=60).content)
    hist = pd.read_csv(cache_path, parse_dates=["date"])
    hist["tickers"] = hist["tickers"].str.split(",")
    return hist.sort_values("date").reset_index(drop=True)


def monthly_membership(hist: pd.DataFrame, start: str, end: str | None = None) -> pd.DataFrame:
    """Long table [month_end, ticker]: members as of each month end (last change on/before it)."""
    months = pd.date_range(start, end or pd.Timestamp.today(), freq="ME")
    idx = hist["date"].searchsorted(months, side="right") - 1      # last row <= month end
    rows = [(m, t) for m, i in zip(months, idx) if i >= 0 for t in hist.at[i, "tickers"]]
    return pd.DataFrame(rows, columns=["month_end", "hist_ticker"])


def resolve_tickers(hist_tickers: list[str], sec_map: dict[str, int],
                    wiki: pd.DataFrame, renames: dict[str, str]) -> pd.DataFrame:
    """Map historical tickers -> today's ticker + CIK.

    1. rename table (FB -> META ...): Yahoo keeps the whole price history under the NEW ticker
    2. current S&P 500 list from Wikipedia (has CIK)
    3. SEC ticker list (all current registrants, incl. companies that left the index)
    Unresolved = mostly acquired / bankrupt companies -> no free price data either."""
    wiki_map = {t.replace(".", "-"): int(c) for t, c in zip(wiki["ticker"], wiki["cik"])}
    rows = []
    for old in sorted(set(hist_tickers)):
        cur = renames.get(old, old).replace(".", "-")
        cik, how = wiki_map.get(cur), "wikipedia"
        if cik is None:
            cik, how = sec_map.get(cur), "sec"
        if cik is not None and old in renames:
            how = "rename+" + how
        rows.append({"hist_ticker": old, "ticker": cur, "cik": cik,
                     "resolved_by": how if cik is not None else None})
    return pd.DataFrame(rows)


def build_membership(client, hist: pd.DataFrame, wiki: pd.DataFrame, renames: dict[str, str],
                     start: str, end: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Point-in-time membership in TODAY's tickers + CIKs.

    Returns
      membership: [month_end, hist_ticker, ticker, cik] - one row per resolved member per month
                  (two share classes of one company, e.g. GOOG/GOOGL, are kept once)
      coverage:   share of members resolved per year - the survivorship gap"""
    monthly = monthly_membership(hist, start, end)
    resolved = resolve_tickers(monthly["hist_ticker"].unique().tolist(), client.ticker_map(), wiki, renames)
    m = monthly.merge(resolved, on="hist_ticker", how="left")
    coverage = (m.assign(resolved=m["cik"].notna())
                 .groupby(m["month_end"].dt.year)["resolved"].mean().rename("share_resolved"))
    m = m.dropna(subset=["cik"]).astype({"cik": int}).drop_duplicates(["month_end", "cik"])
    return m[["month_end", "hist_ticker", "ticker", "cik"]].reset_index(drop=True), coverage.to_frame()
