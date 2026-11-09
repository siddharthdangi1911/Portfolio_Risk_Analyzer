"""
src/portfolio.py
=============================================================================
Portfolio risk tools used by the export script and the website.

  * realised metrics for fixed weights (daily rebalanced): annualised return,
    volatility, Sharpe ratio (risk-free rate assumed 0 - stated as a
    simplification), maximum drawdown
  * ex-ante portfolio volatility from per-asset vol forecasts and a trailing
    60-day correlation matrix:  sigma_p = sqrt( w' D C D w ),  D = diag(vol forecasts)
    This is how the per-asset ML forecasts become a portfolio risk forecast.
=============================================================================
"""

from typing import Dict, List

import numpy as np
import pandas as pd

from src.config import CONFIG

TD = CONFIG.TRADING_DAYS


def realised_metrics(returns: pd.DataFrame, weights: Dict[str, float]) -> Dict[str, float]:
    w = pd.Series(weights).reindex(returns.columns).fillna(0.0)
    w = w / w.sum()
    simple = np.expm1(returns)                 # log -> simple returns for portfolio aggregation
    port = simple.mul(w, axis=1).sum(axis=1)
    wealth = (1 + port).cumprod()
    ann_ret = float(wealth.iloc[-1] ** (TD / len(port)) - 1)
    ann_vol = float(port.std() * np.sqrt(TD))
    mdd = float((wealth / wealth.cummax() - 1).min())
    return {"ann_return_pct": ann_ret * 100, "ann_vol_pct": ann_vol * 100,
            "sharpe_rf0": ann_ret / ann_vol if ann_vol > 0 else float("nan"), "max_drawdown_pct": mdd * 100}


def trailing_correlations(returns: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Pairwise rolling correlations, known at the close of each date (no look-ahead)."""
    cols = list(returns.columns)
    out = {}
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            out[f"{cols[i]}|{cols[j]}"] = returns[cols[i]].rolling(window).corr(returns[cols[j]])
    return pd.DataFrame(out, index=returns.index)


def presets(assets: List[str]) -> Dict[str, Dict[str, float]]:
    p = {"Equal weight": {a: 1 / len(assets) for a in assets}}
    if "SPY" in assets:
        p["100% SPY"] = {"SPY": 1.0}
    if "SPY" in assets and "TLT" in assets:
        p["60/40 stocks/bonds"] = {"SPY": 0.6, "TLT": 0.4}
    if {"SPY", "TLT", "GLD"} <= set(assets):
        p["40/40/20 stocks/bonds/gold"] = {"SPY": 0.4, "TLT": 0.4, "GLD": 0.2}
    if "QQQ" in assets:
        p["100% QQQ"] = {"QQQ": 1.0}
    return p
