"""Minimal SEC EDGAR client with rate limiting, retries and an on-disk JSON cache.

SEC fair-access rules: identify yourself via User-Agent and stay under 10 requests/s.
Every response is cached, so re-running notebooks does not hit the SEC again.
"""
from __future__ import annotations

import gzip
import json
import time
from pathlib import Path

import requests

DATA_BASE = "https://data.sec.gov"


class EdgarClient:
    def __init__(
        self,
        user_agent: str,
        cache_dir: Path,
        max_requests_per_second: float = 8.0,
        max_age_days: int = 7,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_age_seconds = max_age_days * 86_400
        self._min_interval = 1.0 / max_requests_per_second
        self._last_request = 0.0

        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}
        )

    # ------------------------------------------------------------------ internals
    def _throttle(self) -> None:
        """Sleep just enough to respect the requests-per-second limit."""
        wait = self._min_interval - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def _is_fresh(self, path: Path) -> bool:
        return path.exists() and (time.time() - path.stat().st_mtime) < self.max_age_seconds

    def get_json(self, url: str, cache_name: str, retries: int = 3) -> dict:
        """GET a JSON document, using the cache when it is fresh enough.

        Cached as gzip (XBRL JSON compresses ~10x: S&P 500 history is ~3 GB raw,
        ~300 MB gzipped). Plain .json files from older versions are still read."""
        path = self.cache_dir / (cache_name + ".gz")
        legacy = self.cache_dir / cache_name
        if self._is_fresh(path):
            return json.loads(gzip.decompress(path.read_bytes()))
        if self._is_fresh(legacy):
            return json.loads(legacy.read_text(encoding="utf-8"))

        for attempt in range(retries):
            self._throttle()
            resp = self.session.get(url, timeout=30)
            # 429 = too many requests, 5xx = SEC hiccup -> back off and retry
            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            data = resp.json()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(gzip.compress(json.dumps(data).encode("utf-8"), compresslevel=6))
            return data
        raise RuntimeError(f"SEC request failed after {retries} attempts: {url}")

    # --------------------------------------------------------------- public API
    def company_facts(self, cik: int) -> dict:
        """All XBRL facts a company ever reported (income statement, balance sheet...)."""
        return self.get_json(
            f"{DATA_BASE}/api/xbrl/companyfacts/CIK{cik:010d}.json",
            f"companyfacts/CIK{cik:010d}.json",
        )

    def submissions(self, cik: int) -> dict:
        """Filing history (forms, dates, accession numbers) - needed for 10-K texts later."""
        return self.get_json(
            f"{DATA_BASE}/submissions/CIK{cik:010d}.json",
            f"submissions/CIK{cik:010d}.json",
        )

    def all_filings(self, cik: int, since_year: int | None = None) -> list[dict]:
        """Complete filing history as a list of row dicts.

        The submissions endpoint only contains the most recent ~1000 filings in
        'recent' (for Apple that is just 2015+, because of many Form 4 insider
        filings). Older filings live in extra JSON pages listed under 'files'."""
        sub = self.submissions(cik)
        pages = [sub["filings"]["recent"]]
        for extra in sub["filings"].get("files", []):
            # skip pages that end before the period we care about
            if since_year and int(extra["filingTo"][:4]) < since_year:
                continue
            pages.append(self.get_json(f"{DATA_BASE}/submissions/{extra['name']}",
                                       f"submissions/{extra['name']}"))
        rows = []
        for page in pages:
            keys = list(page.keys())
            rows.extend(dict(zip(keys, values)) for values in zip(*page.values()))
        return rows

    def get_text(self, url: str, retries: int = 3) -> str:
        """Download a document (e.g. a 10-K HTML) WITHOUT caching - callers cache
        the extracted result instead, because raw 10-K HTML is large (1-10 MB each)."""
        for attempt in range(retries):
            self._throttle()
            resp = self.session.get(url, timeout=60)
            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            return resp.text
        raise RuntimeError(f"SEC request failed after {retries} attempts: {url}")

    def ticker_map(self) -> dict[str, int]:
        """Current ticker -> CIK for every SEC registrant (Yahoo-style tickers, e.g. BRK-B)."""
        data = self.get_json("https://www.sec.gov/files/company_tickers.json", "company_tickers.json")
        return {v["ticker"].upper(): int(v["cik_str"]) for v in data.values()}

    def frame(self, taxonomy: str, tag: str, unit: str, period: str) -> dict:
        """One concept for *all* companies in one period, e.g. shares outstanding CY2026Q3I."""
        return self.get_json(
            f"{DATA_BASE}/api/xbrl/frames/{taxonomy}/{tag}/{unit}/{period}.json",
            f"frames/{taxonomy}_{tag}_{unit}_{period}.json",
        )
