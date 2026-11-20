# Stock Predictor

An interactive Jupyter project that supports stock picking using annual reports (10-K) of S&P 500 companies:
fundamental trends, text analysis (sentiment + year-over-year text changes), Monte Carlo DCF valuation and backtesting.

> ⚠️ Educational / personal project. Not investment advice.

**Status:** design phase – see [SPEC.md](SPEC.md).

## Getting started (Windows, PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env   # fill in your e-mail for the SEC User-Agent
jupyter lab
```

## Structure

- `src/stockpred/` – core logic (data ingestion, fundamentals, text, valuation, backtest)
- `notebooks/` – thin notebooks calling `src/`
- `data/` – local cache (not tracked by git)

## Data sources

- [SEC EDGAR](https://www.sec.gov/edgar) – fundamentals (XBRL) and 10-K filings
- [yfinance](https://github.com/ranaroussi/yfinance) – market prices (unofficial Yahoo Finance API)
