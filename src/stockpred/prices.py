"""Market prices via yfinance (unofficial Yahoo Finance API) with a parquet cache."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf


def to_yahoo_symbol(ticker: str) -> str:
    """Wikipedia/SEC use 'BRK.B', Yahoo uses 'BRK-B'."""
    return ticker.replace(".", "-")


def last_close(tickers: list[str]) -> pd.Series:
    """Most recent close for many tickers in a single batch request."""
    symbols = [to_yahoo_symbol(t) for t in tickers]
    close = yf.download(symbols, period="5d", auto_adjust=False, progress=False)["Close"]
    latest = close.ffill().iloc[-1]                     # last non-missing value per column
    return latest.rename(index={to_yahoo_symbol(t): t for t in tickers})


def market_caps(tickers: list[str]) -> pd.Series:
    """Market cap straight from Yahoo, one request per ticker (slower, used as fallback).
    Yahoo sums all share classes, which SEC data cannot do for GOOGL, META, BRK.B, V..."""
    caps = {}
    for t in tickers:
        try:
            caps[t] = yf.Ticker(to_yahoo_symbol(t)).fast_info["marketCap"]
        except Exception:
            caps[t] = float("nan")
    return pd.Series(caps, dtype="float64")


def price_history(
    tickers: list[str], start: str, cache_path: Path, refresh: bool = False
) -> pd.DataFrame:
    """Daily adjusted close (dividends/splits included), one column per ticker."""
    cache_path = Path(cache_path)
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)

    symbols = [to_yahoo_symbol(t) for t in tickers]
    close = yf.download(symbols, start=start, auto_adjust=True, progress=False)["Close"]
    close = close.rename(columns={to_yahoo_symbol(t): t for t in tickers})
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    close.to_parquet(cache_path)
    return close


def split_history(tickers: list[str], cache_path: Path, refresh: bool = False,
                  time_budget_s: float | None = None) -> dict[str, pd.Series]:
    """Stock split history per ticker (date -> ratio), cached incrementally.

    One Yahoo request per ticker, so for hundreds of tickers the cache is filled in
    steps: tickers already checked (with or without splits) are never requested again.
    With `time_budget_s` the call may return before all tickers are checked."""
    import time

    cache_path = Path(cache_path)
    checked_path = cache_path.with_suffix(".checked.txt")
    long = pd.read_parquet(cache_path) if cache_path.exists() and not refresh else \
        pd.DataFrame(columns=["ticker", "date", "ratio"])
    checked = set(checked_path.read_text().split()) if checked_path.exists() and not refresh else set()
    if not checked and not long.empty:
        checked = set(tickers)          # cache written by an older version = complete

    todo = [t for t in tickers if t not in checked]
    t0, parts = time.time(), [long]
    for t in todo:
        if time_budget_s and time.time() - t0 > time_budget_s:
            break
        s = yf.Ticker(to_yahoo_symbol(t)).splits
        if len(s):
            parts.append(pd.DataFrame({"ticker": t, "date": s.index.tz_localize(None), "ratio": s.values}))
        checked.add(t)
    if todo:
        long = pd.concat([p for p in parts if len(p)], ignore_index=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        long.to_parquet(cache_path)
        checked_path.write_text("\n".join(sorted(checked)))
    return {t: g.set_index("date")["ratio"] for t, g in long.groupby("ticker")}


def monthly_prices(tickers: list[str], start: str, cache_dir: Path, chunk_size: int = 80,
                   time_budget_s: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Month-end prices for many tickers, cached INCREMENTALLY in one parquet file.

    Returns (total_return_index, close):
      * total return = 'Adj Close' (dividends reinvested, splits adjusted) -> for returns
      * close        = split-adjusted but NOT dividend-adjusted -> for market cap
        (our share counts are split-adjusted to today's basis, so they match)
    Only tickers missing from the cache are downloaded (in chunks, saved after each one).
    With `time_budget_s`, returns None if not finished - call again to continue.
    Note: the cache is not refreshed automatically; delete it to get newer months."""
    import time

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"monthly_{start}.parquet"
    data = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    have = {c.split("|", 1)[1] for c in data.columns}
    symbols = sorted({to_yahoo_symbol(t) for t in tickers})
    missing = [s for s in symbols if s not in have]

    t0 = time.time()
    for i in range(0, len(missing), chunk_size):
        if time_budget_s and time.time() - t0 > time_budget_s:
            return None
        chunk = missing[i:i + chunk_size]
        raw = yf.download(chunk, start=start, auto_adjust=False, progress=False, threads=True)
        monthly = raw[["Adj Close", "Close"]].resample("ME").last()
        monthly.columns = [f"{field}|{sym}" for field, sym in monthly.columns]
        for sym in chunk:                         # failed downloads -> empty column, never retried
            for field in ("Adj Close", "Close"):
                if f"{field}|{sym}" not in monthly:
                    monthly[f"{field}|{sym}"] = float("nan")
        data = pd.concat([data, monthly], axis=1) if len(data) else monthly
        data.to_parquet(path)

    cols = lambda field: [f"{field}|{s}" for s in symbols]
    tri, close = data[cols("Adj Close")].copy(), data[cols("Close")].copy()
    tri.columns = close.columns = symbols
    return tri, close
