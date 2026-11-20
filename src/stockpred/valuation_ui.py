"""Interactive Monte Carlo DCF widget (ipywidgets) - used by notebook 04 and the explorer."""
from __future__ import annotations

import html

import ipywidgets as widgets
import numpy as np
import pandas as pd
from IPython.display import display

from .plots import mc_histogram_figure, sensitivity_heatmap
from .valuation_mc import estimate_inputs, sensitivity_grid, simulate


def _slider(desc: str, lo: float, hi: float, step: float, fmt: str = ".1%") -> widgets.FloatSlider:
    # continuous_update=False -> recompute only when the mouse is released
    return widgets.FloatSlider(description=desc, min=lo, max=hi, step=step, readout_format=fmt,
                               continuous_update=False, style={"description_width": "150px"},
                               layout={"width": "460px"})


def build_valuation_widget(panel: pd.DataFrame, universe: pd.DataFrame,
                           prices: pd.Series) -> widgets.Widget:
    uni = universe.set_index("ticker")
    options = [(f"{t} – {uni.loc[t, 'name']}", t) for t in universe["ticker"]]

    ticker = widgets.Dropdown(options=options, description="Company", layout={"width": "420px"})
    basis = widgets.ToggleButtons(options=[("FCF margin", "fcf"), ("Net margin", "net")],
                                  description="Margin basis")
    s = {
        "growth_mean": _slider("Growth (yrs 1–N)", -0.10, 0.50, 0.005),
        "growth_sd": _slider("Growth uncertainty ±", 0.0, 0.30, 0.005),
        "margin_mean": _slider("Margin", -0.20, 0.70, 0.005),
        "margin_sd": _slider("Margin uncertainty ±", 0.0, 0.20, 0.005),
        "wacc_mean": _slider("WACC", 0.05, 0.15, 0.0025, ".2%"),
        "wacc_sd": _slider("WACC uncertainty ±", 0.0, 0.03, 0.0025, ".2%"),
        "terminal_growth": _slider("Terminal growth", 0.0, 0.04, 0.0025, ".2%"),
    }
    years = widgets.IntSlider(description="High-growth years", min=1, max=10, value=5,
                              continuous_update=False, style={"description_width": "150px"},
                              layout={"width": "460px"})
    reset = widgets.Button(description="Reset to history", icon="undo")
    info, hist_out, sens_out = widgets.HTML(), widgets.Output(), widgets.Output()
    state = {"inputs": None, "defaults": None, "busy": False}

    def load_defaults(*_):
        t = ticker.value
        price = float(prices.get(t, np.nan))
        mcap = uni.loc[t].get("market_cap", np.nan)
        fallback = mcap / price if price and not pd.isna(mcap) else None
        inputs, a = estimate_inputs(panel, t, price, uni.loc[t].get("sub_industry"),
                                    fallback, margin_basis=basis.value)
        state.update(inputs=inputs, defaults=a, busy=True)
        for k, w in s.items():                # set sliders without triggering 7 recomputes
            w.value = float(getattr(a, k))
        years.value = a.high_growth_years
        state["busy"] = False
        redraw()

    def redraw(*_):
        if state["busy"] or state["inputs"] is None:
            return
        inp = state["inputs"]
        a = state["defaults"].with_(**{k: w.value for k, w in s.items()},
                                    high_growth_years=years.value)
        res = simulate(inp, a)
        warn_html = "".join(f"<li>⚠️ {html.escape(w)}</li>" for w in res.warnings)
        info.value = (
            f"<b>{inp.ticker}</b> · base FY{inp.fiscal_year} revenue {inp.revenue / 1e9:,.1f} bn USD · "
            f"net cash {inp.net_cash / 1e9:,.1f} bn · diluted shares {inp.shares / 1e9:,.2f} bn · "
            f"price {inp.price:,.2f} USD"
            + (f"<ul style='margin:4px 0'>{warn_html}</ul>" if warn_html else "")
        )
        with hist_out:
            hist_out.clear_output(wait=True)
            mc_histogram_figure(res, inp.ticker).show()
        with sens_out:
            sens_out.clear_output(wait=True)
            sensitivity_heatmap(sensitivity_grid(inp, a), inp.price).show()

    ticker.observe(load_defaults, names="value")
    basis.observe(load_defaults, names="value")
    for w in [*s.values(), years]:
        w.observe(redraw, names="value")
    reset.on_click(load_defaults)

    load_defaults()
    controls = widgets.VBox([widgets.HBox([ticker, basis]),
                             widgets.HBox([widgets.VBox(list(s.values())[:4]),
                                           widgets.VBox([*list(s.values())[4:], years, reset])])])
    return widgets.VBox([controls, info, hist_out, sens_out])
