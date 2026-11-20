"""Cross-sectional factor backtest: monthly quintile portfolios.

Each month end t:
  1. take the index members known at t that have a score and a price
  2. sort them into 5 equal-sized groups by score (Q1 worst ... Q5 best)
  3. hold each group equally weighted until t+1 and record its return
The factor "works" if returns rise from Q1 to Q5 (monotonic) and the long-short
portfolio Q5 - Q1 is positive with a t-stat clearly above ~2.

We also report the Information Coefficient (IC): the Spearman rank correlation between
the score at t and the return t -> t+1, averaged over months. IC ~0.02-0.05 is already
considered useful in practice; noise is large, so look at the t-stat.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    name: str
    quintiles: pd.DataFrame      # month_end x [Q1..Q5] monthly returns
    long_short: pd.Series        # Q5 - Q1
    benchmark: pd.Series         # equal-weight of all scored names
    ic: pd.Series                # monthly Spearman IC
    turnover: float              # avg share of Q5 names replaced each month
    n_names: pd.Series           # scored names per month

    def summary(self) -> dict:
        ls = perf_stats(self.long_short)
        q = {f"{c}_ann": perf_stats(self.quintiles[c])["ann_return"] for c in self.quintiles}
        ic_t = self.ic.mean() / self.ic.std() * np.sqrt(self.ic.count()) if self.ic.count() > 2 else np.nan
        excess = self.quintiles["Q5"] - self.benchmark
        return {
            "factor": self.name, **q,
            "LS_ann": ls["ann_return"], "LS_sharpe": ls["sharpe"], "LS_tstat": ls["t_stat"],
            "LS_maxdd": ls["max_drawdown"], "Q5_vs_EW_ann": perf_stats(excess)["ann_return"],
            "IC_mean": self.ic.mean(), "IC_tstat": ic_t,
            "monotonic": bool(np.all(np.diff([q[f"Q{i}_ann"] for i in range(1, 6)]) > 0)),
            "turnover_Q5": self.turnover, "avg_names": self.n_names.mean(), "months": len(self.long_short),
        }


def perf_stats(r: pd.Series, periods: int = 12) -> dict:
    """Annualised return (geometric), volatility, Sharpe (rf = 0), t-stat of the mean, max drawdown."""
    r = r.dropna()
    if len(r) < 3:
        return dict(ann_return=np.nan, ann_vol=np.nan, sharpe=np.nan, t_stat=np.nan, max_drawdown=np.nan)
    wealth = (1 + r).cumprod()
    ann = wealth.iloc[-1] ** (periods / len(r)) - 1
    vol = r.std() * np.sqrt(periods)
    return dict(
        ann_return=ann, ann_vol=vol,
        sharpe=r.mean() / r.std() * np.sqrt(periods) if r.std() > 0 else np.nan,
        t_stat=r.mean() / r.std() * np.sqrt(len(r)) if r.std() > 0 else np.nan,
        max_drawdown=(wealth / wealth.cummax() - 1).min(),
    )


def quintile_backtest(table: pd.DataFrame, score_col: str, name: str | None = None,
                      n_groups: int = 5, min_names: int = 50, cost_bps: float = 0.0) -> BacktestResult:
    """`table` needs columns month_end, ticker, fwd_ret and `score_col`.
    cost_bps: one-way trading cost applied to the long-short turnover (0 = gross)."""
    df = table.dropna(subset=[score_col, "fwd_ret"])
    df = df[df.groupby("month_end")[score_col].transform("count") >= min_names].copy()
    if df.empty:                                   # not enough coverage in any month
        empty = pd.Series(dtype=float)
        return BacktestResult(name or score_col, pd.DataFrame(), empty, empty, empty, np.nan, empty)
    # rank first so ties do not break qcut; Q1 = lowest score
    df["q"] = df.groupby("month_end")[score_col].transform(
        lambda s: pd.qcut(s.rank(method="first"), n_groups, labels=False) + 1)

    q = df.pivot_table(index="month_end", columns="q", values="fwd_ret", aggfunc="mean")
    q.columns = [f"Q{int(c)}" for c in q.columns]
    bench = df.groupby("month_end")["fwd_ret"].mean()
    # Spearman = Pearson correlation of ranks (no scipy needed)
    ranked = df[["month_end"]].assign(
        s=df.groupby("month_end")[score_col].rank(), r=df.groupby("month_end")["fwd_ret"].rank())
    ic = ranked.groupby("month_end")[["s", "r"]].corr().xs("s", level=1)["r"]

    # turnover of the top group: share of names that were not in Q5 last month
    top = df[df["q"] == n_groups].groupby("month_end")["ticker"].apply(set)
    changes = [1 - len(a & b) / len(b) for a, b in zip(top.iloc[:-1], top.iloc[1:]) if b]
    turnover = float(np.mean(changes)) if changes else np.nan

    ls = q[f"Q{n_groups}"] - q["Q1"]
    if cost_bps:
        ls = ls - 2 * 2 * turnover * cost_bps / 1e4     # both legs, buy + sell
    return BacktestResult(name or score_col, q, ls, bench, ic, turnover,
                          df.groupby("month_end").size())


def cumulative(returns: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Growth of 1 USD."""
    return (1 + returns.fillna(0)).cumprod()
