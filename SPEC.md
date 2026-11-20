# Stock Predictor – Project Specification

> Version 0.4 · 2026-11-20 · status: **phase 1 implemented** (fundamentals + explorer)

## 1. Goal

A local Jupyter project that supports stock picking:
- download annual reports (10-K) and fundamentals of **S&P 500** companies for ~15 years,
- visualise fundamental trends,
- analyse annual report text (sentiment, year-over-year changes),
- value companies with a **Monte Carlo DCF** (a distribution of intrinsic value instead of a single number),
- everything interactive (ipywidgets + Plotly).

Purpose: **personal investing + CV portfolio** → clean repo, README, reproducibility.

## 2. Decisions

| Area | Decision |
|---|---|
| Universe | **S&P 500 including historical constituents** (single source = SEC EDGAR, ~15 years of data, reduced survivorship bias). Global / ESEF extension later |
| LLM layer | **Postponed.** If enabled: top 50 only, only text changed vs. previous year, Batch API |
| Environment | Local Jupyter (Windows), Python `venv` |
| Version control | Git + public GitHub repo |
| Language | English (code, comments, docs) |
| Budget | **100 % free for now** (decided 2026-11-20). LLM extraction via Claude API postponed – revisit after phase 4 |
| Interactivity | `ipywidgets` + `plotly` (possibly Voilà / Streamlit later) |
| Modelling | Monte Carlo DCF (main) + portfolio Monte Carlo (evaluation); GBM fan chart as a visual only |
| Prediction target | **Not** absolute price. Relative ranking / probability of undervaluation / fundamentals |

## 3. Known risks (address from day one)

1. **XBRL tags are inconsistent even within EDGAR** – companies use different tags for the same concept (e.g. revenue: `Revenues` vs. `RevenueFromContractWithCustomerExcludingAssessedTax`); a mapping table with fallbacks is required.
2. **Survivorship bias** – mitigate with historical S&P 500 constituents (incl. removed companies). Delisted companies may lack prices in `yfinance` → document in README.
3. **Look-ahead bias** – a fundamental is known only from the filing date, not from the fiscal year end.
4. **Units and scaling** – XBRL values come in different units (USD, shares, USD/share); fiscal years differ from calendar years.
5. **Text is a weak signal** – year-over-year text change is more informative than tone (Cohen, Malloy, Nguyen: *Lazy Prices*, Journal of Finance 2020).
6. **Monte Carlo is not a prediction** – the output is only as good as the chosen input distributions.
7. **`yfinance` is unofficial** – it scrapes Yahoo Finance and may break or rate-limit; cache everything locally.

## 4. Data strategy

| Data | Source | Note |
|---|---|---|
| Fundamentals | SEC EDGAR XBRL `companyfacts` API | free, `User-Agent` with e-mail (in `.env`), max ~10 requests/s |
| 10-K text | EDGAR filing archive (HTML) | extract Item 7 (MD&A) and Item 1A (Risk Factors) |
| Filing dates | EDGAR `submissions` API | to avoid look-ahead bias |
| S&P 500 constituents | current list + historical changes (public datasets) | to reduce survivorship bias |
| Prices | `yfinance` | cached as parquet |
| Ticker ↔ CIK mapping | SEC `company_tickers.json` | |

**Future extension:** EU/UK via ESEF (filings.xbrl.org, FY2021+), foreign companies filing 20-F on EDGAR.

## 5. Repository structure

```
stock-predictor/
├── README.md               # project overview for CV, screenshots, how to run
├── SPEC.md                 # this document
├── requirements.txt
├── requirements-nlp.txt    # optional heavy NLP dependencies
├── config.yaml             # universe, years, paths
├── .env.example            # SEC_USER_AGENT, ANTHROPIC_API_KEY (optional)
├── data/                   # gitignored
│   ├── raw/                # downloaded JSON / HTML
│   ├── interim/
│   └── processed/          # parquet tables
├── src/stockpred/          # logic in modules (testable)
│   ├── universe.py
│   ├── ingest_edgar.py
│   ├── prices.py
│   ├── fundamentals.py     # XBRL tag mapping → standard metrics
│   ├── text.py             # sections, sentiment, similarity
│   ├── llm_extract.py      # optional, postponed
│   ├── valuation_mc.py     # Monte Carlo DCF
│   ├── portfolio_mc.py
│   └── backtest.py
├── notebooks/              # thin notebooks calling src/
│   ├── 01_universe.ipynb
│   ├── 02_fundamentals_trends.ipynb
│   ├── 03_text_analysis.ipynb
│   ├── 04_mc_valuation.ipynb
│   ├── 05_scoring_backtest.ipynb
│   └── 10_explorer.ipynb   # interactive single-ticker dashboard
└── tests/
```

Why modules + thin notebooks: better impression in a CV, reusability, testability, less notebook chaos.

## 6. Modules

1. **Universe** – S&P 500 (current + historical constituents), ticker ↔ CIK mapping, top 50 by market cap.
2. **Fundamentals** – revenue, EBIT, net income, EPS, FCF, margins, ROIC, debt, share count; normalisation of XBRL tags; YoY growth, CAGR.
3. **Text** – section extraction (MD&A, Risk Factors), sentiment (Loughran-McDonald dictionary, FinBERT), YoY similarity (TF-IDF cosine, embeddings); optionally LLM extraction (guidance, new risks, strategy changes → JSON).
4. **Monte Carlo DCF** – input distributions (revenue growth, margin, reinvestment, WACC, terminal growth) estimated from company history + sliders for manual adjustment; output: intrinsic value histogram, P(value > price), percentiles.
5. **Scoring + backtest** – composite score (Piotroski F-score, quality, growth, text signals, MC margin of safety), quintile portfolios vs. benchmark, respecting filing dates.
6. **Portfolio Monte Carlo** – bootstrapped returns of a selected portfolio: VaR, max drawdown, P(loss).
7. **Explorer** – interactive notebook: pick a ticker → trends, text, MC valuation in one place.

## 7. Phases (MVP first)

| Phase | Scope | Done when |
|---|---|---|
| 1 ✅ | Top 50 of S&P 500, fundamentals + trends + explorer | trend chart for any of the 50 tickers |
| 2 | Text analysis (free methods) | sentiment + similarity over years |
| 3 | Monte Carlo DCF with sliders | value histogram + P(undervalued) |
| 4 | Scale to full S&P 500 (incl. historical constituents) + scoring + backtest | quintile backtest with chart |
| 5 | *(optional, paid – postponed)* LLM extraction for top 50; optional ESEF extension | JSON with risks / guidance per year |
| 6 | README, screenshots, tests, cleanup → CV | polished public GitHub repo |

## 8. LLM budget (POSTPONED – project currently runs for free)

Model: Claude Haiku 5.5 – $0.10 / $0.50 per 1M input / output tokens for prompts up to 100k tokens, 5× more above 100k (prices as of 10/2026).

| Measure | Saving | Note |
|---|---|---|
| **Send only paragraphs changed vs. previous year** (diff) | typically large – 10-Ks change little year to year | same idea as *Lazy Prices*; also a more interesting signal |
| MD&A + Risk Factors sections only | large | never the full 10-K |
| **Batch API** | −50 % | asynchronous, results within 24 h – acceptable |
| Prompt caching of instructions | small | instructions are short |
| Keep prompts under 100k tokens | avoids 5× tier | chunk the text |
| Short JSON output with a schema | fewer output tokens | |
| Free pre-filter (FinBERT / LM dictionary) | optional | LLM only for paragraphs with significant change |
| Local LLM (Ollama) | 100 % | free, but slow on a laptop without GPU and lower quality |

Rough estimate (top 50 × 10 years, ~15k diff tokens/year, batch): ~**$1**. Full S&P 500 × 10 years with diff + batch: ~$5 → borderline, not planned for now.

## 9. Open questions

- [x] Universe: S&P 500 (LLM top 50 only)
- [x] Environment: `venv`
- [x] Git + GitHub repo
- [x] SEC User-Agent e-mail (local `.env`, never committed)
- [x] Language: English
- [ ] Include 10-Q (quarterly) filings for more data points?
- [ ] Python version (3.11 / 3.12 recommended for `torch` / `transformers` compatibility)
