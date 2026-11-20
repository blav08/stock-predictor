"""Investment universe: current S&P 500 constituents, CIK mapping and top-N by market cap."""
from __future__ import annotations

import io
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from .edgar import EdgarClient
from .prices import last_close, market_caps

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def fetch_sp500(user_agent: str, cache_path: Path, refresh: bool = False) -> pd.DataFrame:
    """Current S&P 500 members scraped from Wikipedia (table includes the SEC CIK)."""
    cache_path = Path(cache_path)
    if cache_path.exists() and not refresh:
        return pd.read_csv(cache_path)

    html = requests.get(WIKI_URL, headers={"User-Agent": user_agent}, timeout=30).text
    table = pd.read_html(io.StringIO(html))[0]
    df = table.rename(
        columns={
            "Symbol": "ticker",
            "Security": "name",
            "GICS Sector": "sector",
            "GICS Sub-Industry": "sub_industry",
            "Date added": "date_added",
            "CIK": "cik",
        }
    )[["ticker", "name", "sector", "sub_industry", "date_added", "cik"]]
    df["cik"] = df["cik"].astype(int)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df


def _recent_quarter_frames(today: date, n: int = 4) -> list[str]:
    """Frame period codes for the last n quarter ends, newest first, e.g. 'CY2026Q3I'.
    The trailing 'I' means 'instant' (a point-in-time value such as shares outstanding)."""
    year, quarter = today.year, (today.month - 1) // 3 + 1
    periods = []
    for _ in range(n):
        quarter -= 1
        if quarter == 0:
            year, quarter = year - 1, 4
        periods.append(f"CY{year}Q{quarter}I")
    return periods


def shares_outstanding(client: EdgarClient, today: date | None = None) -> pd.Series:
    """Latest reported shares outstanding for every SEC filer (index = CIK).

    One request per quarter instead of one per company. Newer quarters win;
    older quarters fill companies that have not filed yet."""
    result = pd.Series(dtype="float64")
    for period in _recent_quarter_frames(today or date.today()):
        try:
            data = client.frame("dei", "EntityCommonStockSharesOutstanding", "shares", period)
        except Exception:
            continue  # frame not published yet
        frame = pd.DataFrame(data["data"]).groupby("cik")["val"].max()
        result = result.combine_first(frame) if not result.empty else frame
    return result


def add_market_cap(universe: pd.DataFrame, client: EdgarClient) -> pd.DataFrame:
    """Market cap = latest SEC shares outstanding x last close price.

    Multi-class companies (GOOGL, META, BRK.B, V, MA...) report shares per class with
    XBRL dimensions, which the SEC frames API omits -> they come back empty. For those
    we fall back to Yahoo's market cap (all classes summed). Good enough for ranking."""
    df = universe.copy()
    df["shares"] = df["cik"].map(shares_outstanding(client))
    df["price"] = df["ticker"].map(last_close(df["ticker"].tolist()))
    df["market_cap"] = df["shares"] * df["price"]
    df["market_cap_source"] = "sec_shares_x_price"

    missing = df["market_cap"].isna()
    if missing.any():
        fallback = market_caps(df.loc[missing, "ticker"].tolist())
        df.loc[missing, "market_cap"] = df.loc[missing, "ticker"].map(fallback)
        df.loc[missing, "market_cap_source"] = "yahoo"
    return df


def top_n(universe: pd.DataFrame, n: int) -> pd.DataFrame:
    """Largest n companies; one row per CIK (drops the second share class, e.g. GOOG)."""
    return (
        universe.dropna(subset=["market_cap"])
        .sort_values("market_cap", ascending=False)
        .drop_duplicates(subset="cik", keep="first")
        .head(n)
        .reset_index(drop=True)
    )
