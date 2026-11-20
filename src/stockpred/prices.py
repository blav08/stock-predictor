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


def split_history(tickers: list[str], cache_path: Path, refresh: bool = False) -> dict[str, pd.Series]:
    """Stock split history per ticker (date -> ratio), cached as one parquet file."""
    cache_path = Path(cache_path)
    if cache_path.exists() and not refresh:
        long = pd.read_parquet(cache_path)
    else:
        parts = []
        for t in tickers:
            s = yf.Ticker(to_yahoo_symbol(t)).splits
            if len(s):
                parts.append(pd.DataFrame({"ticker": t, "date": s.index.tz_localize(None), "ratio": s.values}))
        long = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["ticker", "date", "ratio"])
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        long.to_parquet(cache_path)
    return {t: g.set_index("date")["ratio"] for t, g in long.groupby("ticker")}
