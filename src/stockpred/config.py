"""Project configuration: paths, settings from config.yaml and secrets from .env."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

# src/stockpred/config.py -> project root is two levels above the package
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    """All runtime settings in one immutable object."""

    sec_user_agent: str
    top_n: int
    start_year: int
    max_age_days: int
    max_requests_per_second: float
    data_dir: Path
    cik_predecessors: dict[int, list[int]]
    backtest_start: str
    ticker_renames: dict[str, str]

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"


def load_settings(config_path: Path | None = None) -> Settings:
    """Read config.yaml + .env and validate the SEC User-Agent."""
    load_dotenv(PROJECT_ROOT / ".env")
    path = config_path or PROJECT_ROOT / "config.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))

    user_agent = os.getenv("SEC_USER_AGENT", "").strip().strip('"')
    if not user_agent or "example.com" in user_agent:
        raise RuntimeError(
            "SEC_USER_AGENT is missing. Copy .env.example to .env and fill in your e-mail."
        )

    settings = Settings(
        sec_user_agent=user_agent,
        top_n=int(cfg["universe"]["top_n"]),
        start_year=int(cfg["fundamentals"]["start_year"]),
        max_age_days=int(cfg["cache"]["max_age_days"]),
        max_requests_per_second=float(cfg["sec"]["max_requests_per_second"]),
        data_dir=PROJECT_ROOT / "data",
        cik_predecessors={int(k): [int(x) for x in v] for k, v in
                          (cfg["universe"].get("cik_predecessors") or {}).items()},
        backtest_start=str(cfg.get("backtest", {}).get("start", "2011-01-31")),
        ticker_renames={str(k): str(v) for k, v in
                        (cfg.get("backtest", {}).get("ticker_renames") or {}).items()},
    )
    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    return settings
