# Stock Predictor

An interactive Jupyter project that supports stock picking using annual reports (10-K) of S&P 500 companies:
fundamental trends, text analysis (sentiment + year-over-year text changes), Monte Carlo DCF valuation and backtesting.

> ⚠️ Educational / personal project. Not investment advice.

**Status:** phase 1 (fundamentals + interactive explorer) – see [SPEC.md](SPEC.md) for the roadmap.

## Getting started (Windows, PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .              # makes `import stockpred` work in notebooks
copy .env.example .env        # fill in your e-mail for the SEC User-Agent
jupyter lab
```

Then run the notebooks in order:

| Notebook | What it does | Runtime (first run) |
|---|---|---|
| `notebooks/01_build_dataset.ipynb` | universe → splits → fundamentals → data quality check | ~3–5 min (downloads ~250 MB from SEC, cached afterwards) |
| `notebooks/10_explorer.ipynb` | interactive charts: company trends, comparison, snapshot table | seconds |

Run tests with `pytest`.

## How it works

- **Universe** – current S&P 500 from Wikipedia (includes SEC CIK) → top N by market cap.
- **Fundamentals** – SEC XBRL `companyfacts`; annual 10-K values only, *as first reported* (no later restatements → no look-ahead bias), each year tagged with the original filing date.
- **Tag mapping** – one metric = a priority list of XBRL tags (companies use different tags for the same concept).
- **Split adjustment** – per-share values restated to today's share basis using split history.

## Known limitations

- **Survivorship bias** – today's index members only (historical constituents planned for phase 4).
- **Banks / insurers** – no gross profit / operating income in XBRL; revenue concept differs.
- **Berkshire Hathaway** – EPS is reported per Class A share.
- Some companies do not tag some concepts (e.g. operating income) – see the coverage table in notebook 01.
- `yfinance` is an unofficial API and may break or rate-limit.

## Structure

- `src/stockpred/` – core logic: `edgar.py` (SEC client + cache), `universe.py`, `fundamentals.py`, `prices.py`, `plots.py`
- `notebooks/` – thin notebooks calling `src/`
- `tests/` – unit tests on synthetic XBRL data
- `data/` – local cache (not tracked by git)

## Data sources

- [SEC EDGAR](https://www.sec.gov/edgar) – fundamentals (XBRL) and 10-K filings
- [Wikipedia – List of S&P 500 companies](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies)
- [yfinance](https://github.com/ranaroussi/yfinance) – prices, splits, market cap fallback
