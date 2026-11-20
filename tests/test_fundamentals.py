"""Unit tests on small synthetic XBRL payloads - no network needed."""
import numpy as np
import pandas as pd

from stockpred.fundamentals import (
    CONCEPTS, adjust_for_splits, company_fundamentals, extract_metric, fiscal_year_of,
)


def fact(start, end, val, filed, form="10-K"):
    d = {"end": end, "val": val, "filed": filed, "form": form}
    if start:
        d["start"] = start
    return d


def payload(tag_entries: dict, unit="USD") -> dict:
    return {"facts": {"us-gaap": {tag: {"units": {unit: e}} for tag, e in tag_entries.items()}}}


def test_quarterly_values_and_10q_are_ignored():
    facts = payload({"Revenues": [
        fact("2023-01-01", "2023-12-31", 100, "2024-02-01"),           # annual -> keep
        fact("2023-10-01", "2023-12-31", 30, "2024-02-01"),            # quarter inside 10-K
        fact("2024-01-01", "2024-03-31", 25, "2024-05-01", "10-Q"),    # 10-Q
    ]})
    m = extract_metric(facts, CONCEPTS["revenue"])
    assert m["value"].tolist() == [100]


def test_as_first_reported_wins_over_restatement():
    facts = payload({"Revenues": [
        fact("2023-01-01", "2023-12-31", 100, "2024-02-01"),
        fact("2023-01-01", "2023-12-31", 90, "2025-02-01"),            # restated later
    ]})
    m = extract_metric(facts, CONCEPTS["revenue"])
    assert m.loc[0, "value"] == 100
    assert m.loc[0, "filed"] == pd.Timestamp("2024-02-01")


def test_tag_fallback_when_primary_tag_missing():
    facts = payload({
        "Revenues": [fact("2023-01-01", "2023-12-31", 100, "2024-02-01")],
        "SalesRevenueNet": [fact("2019-01-01", "2019-12-31", 70, "2020-02-01")],
    })
    m = extract_metric(facts, CONCEPTS["revenue"])
    assert set(m["tag"]) == {"Revenues", "SalesRevenueNet"}


def test_fiscal_year_convention():
    ends = pd.Series(pd.to_datetime(["2026-01-31", "2026-06-30", "2026-09-26"]))
    assert list(fiscal_year_of(ends)) == [2025, 2026, 2026]


def test_split_adjustment_only_after_filing():
    df = pd.DataFrame({
        "eps_diluted": [8.0, 2.0],
        "available_date": pd.to_datetime(["2019-11-01", "2021-11-01"]),
    }, index=[2019, 2020])
    splits = pd.Series([4.0], index=pd.to_datetime(["2020-08-31"]))
    out = adjust_for_splits(df, splits)
    assert np.allclose(out["eps_diluted"], [2.0, 2.0])


def test_growth_does_not_skip_missing_years():
    facts = payload({"Revenues": [
        fact("2020-01-01", "2020-12-31", 100, "2021-02-01"),
        fact("2022-01-01", "2022-12-31", 150, "2023-02-01"),           # 2021 missing
    ]})
    df = company_fundamentals(facts, start_year=2015)
    assert 2021 in df.index
    assert df["revenue_growth"].isna().all()
