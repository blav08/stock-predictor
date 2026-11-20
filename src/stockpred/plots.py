"""Plotly figures for fundamentals (used by the explorer notebook)."""
from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

LABELS = {
    "revenue": "Revenue",
    "gross_profit": "Gross profit",
    "operating_income": "Operating income",
    "net_income": "Net income",
    "free_cash_flow": "Free cash flow",
    "operating_cash_flow": "Operating cash flow",
    "capex": "Capex",
    "rnd": "R&D expense",
    "eps_diluted": "EPS (diluted, split-adj.)",
    "diluted_shares": "Diluted shares",
    "total_assets": "Total assets",
    "total_liabilities": "Total liabilities",
    "equity": "Shareholders' equity",
    "cash": "Cash",
    "long_term_debt": "Long-term debt",
    "gross_margin": "Gross margin",
    "operating_margin": "Operating margin",
    "net_margin": "Net margin",
    "fcf_margin": "FCF margin",
    "roe": "ROE",
    "revenue_growth": "Revenue growth (YoY)",
    "eps_growth": "EPS growth (YoY)",
    "liabilities_to_equity": "Liabilities / equity",
}
PERCENT = {"gross_margin", "operating_margin", "net_margin", "fcf_margin", "roe",
           "revenue_growth", "eps_growth"}
USD_BILLIONS = {"revenue", "gross_profit", "operating_income", "net_income", "free_cash_flow",
                "operating_cash_flow", "capex", "rnd", "total_assets", "total_liabilities",
                "equity", "cash", "long_term_debt"}


def _scaled(series: pd.Series, metric: str) -> tuple[pd.Series, str]:
    """Convert raw values to a readable unit and return (values, axis suffix)."""
    if metric in USD_BILLIONS:
        return series / 1e9, "USD bn"
    if metric == "diluted_shares":
        return series / 1e9, "bn shares"
    if metric in PERCENT:
        return series * 100, "%"
    return series, ""


def trend_figure(panel: pd.DataFrame, ticker: str, metrics: list[str]) -> go.Figure:
    """Grid of small charts, one per metric: bars for amounts, lines for ratios."""
    df = panel[panel["ticker"] == ticker].sort_values("fiscal_year")
    cols = 2 if len(metrics) > 1 else 1
    rows = math.ceil(len(metrics) / cols)
    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=[LABELS.get(m, m) for m in metrics],
                        vertical_spacing=0.12 if rows > 1 else 0.2)

    for i, metric in enumerate(metrics):
        r, c = i // cols + 1, i % cols + 1
        if metric not in df:
            continue
        y, unit = _scaled(df[metric], metric)
        trace_cls = go.Scatter if metric in PERCENT or metric.endswith("_to_equity") else go.Bar
        kwargs = {"mode": "lines+markers"} if trace_cls is go.Scatter else {}
        fig.add_trace(trace_cls(x=df["fiscal_year"], y=y, name=LABELS.get(metric, metric),
                                hovertemplate="FY%{x}: %{y:,.2f} " + unit + "<extra></extra>",
                                **kwargs), row=r, col=c)
        fig.update_yaxes(title_text=unit, row=r, col=c)

    fig.update_layout(height=280 * rows, showlegend=False, template="plotly_white",
                      title=f"{ticker} – annual fundamentals (10-K, as first reported)",
                      margin=dict(t=80, l=40, r=20, b=40))
    return fig


def compare_figure(panel: pd.DataFrame, tickers: list[str], metric: str) -> go.Figure:
    """One metric, several companies, one line each."""
    fig = go.Figure()
    unit = ""
    for t in tickers:
        df = panel[panel["ticker"] == t].sort_values("fiscal_year")
        if metric not in df:
            continue
        y, unit = _scaled(df[metric], metric)
        fig.add_trace(go.Scatter(x=df["fiscal_year"], y=y, mode="lines+markers", name=t))
    fig.update_layout(template="plotly_white", height=450,
                      title=f"{LABELS.get(metric, metric)} – comparison",
                      yaxis_title=unit, xaxis_title="Fiscal year")
    return fig


def snapshot_table(panel: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    """Latest fiscal year per company + 5-year revenue CAGR, for a sortable overview."""
    latest = panel.dropna(subset=["revenue"]).sort_values("fiscal_year").groupby("ticker").tail(1)

    def cagr(t: str, fy: int, years: int = 5) -> float:
        rev = panel[(panel["ticker"] == t)].set_index("fiscal_year")["revenue"]
        start, end = rev.get(fy - years), rev.get(fy)
        if start is None or end is None or pd.isna(start) or start <= 0 or pd.isna(end):
            return float("nan")
        return (end / start) ** (1 / years) - 1

    latest = latest.assign(revenue_cagr_5y=[cagr(t, fy) for t, fy in zip(latest["ticker"], latest["fiscal_year"])])
    cols = ["ticker", "fiscal_year", "revenue", "revenue_cagr_5y", "gross_margin",
            "operating_margin", "net_margin", "fcf_margin", "roe"]
    out = latest[cols].merge(universe[["ticker", "name", "sector", "market_cap"]], on="ticker")
    out["revenue"] = out["revenue"] / 1e9
    out["market_cap"] = out["market_cap"] / 1e9
    return out.rename(columns={"revenue": "revenue_usd_bn", "market_cap": "market_cap_usd_bn"}) \
              .sort_values("market_cap_usd_bn", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------ text features
SECTION_LABELS = {"risk_factors": "Risk Factors (Item 1A)", "mdna": "MD&A (Item 7)"}
TEXT_PANELS = [
    ("net_tone", "Net tone (LM, -1…1)", False),
    ("lm_uncertainty", "Uncertainty words (%)", True),
    ("share_new_sentences", "New sentences vs. last year (%)", True),
    ("n_words", "Length (words)", False),
]


def text_trend_figure(text: pd.DataFrame, ticker: str) -> go.Figure:
    """Four small charts, one line per 10-K section."""
    df = text[(text["ticker"] == ticker) & text["extracted"]].sort_values("fiscal_year")
    fig = make_subplots(rows=2, cols=2, subplot_titles=[t for _, t, _ in TEXT_PANELS],
                        vertical_spacing=0.18)
    colors = {"risk_factors": "#c0392b", "mdna": "#2c7fb8"}
    for i, (col, _, pct) in enumerate(TEXT_PANELS):
        r, c = i // 2 + 1, i % 2 + 1
        for section, g in df.groupby("section"):
            if col not in g:
                continue
            y = g[col] * 100 if pct else g[col]
            fig.add_trace(go.Scatter(x=g["fiscal_year"], y=y, mode="lines+markers",
                                     name=SECTION_LABELS.get(section, section),
                                     line=dict(color=colors.get(section)),
                                     legendgroup=section, showlegend=(i == 0)), row=r, col=c)
    fig.update_layout(height=600, template="plotly_white",
                      title=f"{ticker} – 10-K text signals",
                      legend=dict(orientation="h", y=-0.12))
    return fig
