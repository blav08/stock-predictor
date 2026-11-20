"""Minimal SEC EDGAR client with rate limiting, retries and an on-disk JSON cache.

SEC fair-access rules: identify yourself via User-Agent and stay under 10 requests/s.
Every response is cached, so re-running notebooks does not hit the SEC again.
"""
from __future__ import annotations

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
        """GET a JSON document, using the cache when it is fresh enough."""
        path = self.cache_dir / cache_name
        if self._is_fresh(path):
            return json.loads(path.read_text(encoding="utf-8"))

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
            path.write_text(json.dumps(data), encoding="utf-8")
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

    def frame(self, taxonomy: str, tag: str, unit: str, period: str) -> dict:
        """One concept for *all* companies in one period, e.g. shares outstanding CY2026Q3I."""
        return self.get_json(
            f"{DATA_BASE}/api/xbrl/frames/{taxonomy}/{tag}/{unit}/{period}.json",
            f"frames/{taxonomy}_{tag}_{unit}_{period}.json",
        )
