"""
src/feature_engineering.py
=============================================================================
Builds the modelling table: one row = one asset on one trading day t.

Every feature uses prices up to and including the close of day t (rolling
windows look backwards only). The target uses days t+1 ... t+20 only.

Target
  target_vol_20d : annualised realised volatility (%) of daily log returns
                   over the next 20 trading days
                   = std(r_{t+1..t+20}) * sqrt(252) * 100
=============================================================================
"""

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.config import CONFIG

logger = logging.getLogger(__name__)

ANN = np.sqrt(CONFIG.TRADING_DAYS) * 100.0

FEATURE_INFO = {
    "ret_1d":        ("Return, today (%)", "Momentum", "Log return from close t-1 to close t."),
    "abs_ret_1d":    ("|Return|, today (%)", "Momentum", "Size of today's move; large moves tend to be followed by large moves."),
    "ret_5d":        ("Return, past 5 days (%)", "Momentum", "Cumulative log return over the last week."),
    "ret_21d":       ("Return, past 21 days (%)", "Momentum", "Cumulative log return over the last month."),
    "drawdown_60d":  ("Drawdown from 60-day high (%)", "Momentum", "How far the price is below its 60-day peak; captures the leverage effect."),
    "vol_5d":        ("Realised vol, 5 days (%)", "Volatility", "Annualised std of the last 5 daily returns."),
    "vol_10d":       ("Realised vol, 10 days (%)", "Volatility", "Annualised std of the last 10 daily returns."),
    "vol_20d":       ("Realised vol, 20 days (%)", "Volatility", "Annualised std of the last 20 daily returns (= the persistence baseline)."),
    "vol_60d":       ("Realised vol, 60 days (%)", "Volatility", "Annualised std of the last 60 daily returns."),
    "vol_120d":      ("Realised vol, 120 days (%)", "Volatility", "Slow-moving vol level; helps with mean reversion."),
    "ewma_vol":      ("EWMA vol, lambda=0.94 (%)", "Volatility", "RiskMetrics exponentially weighted volatility."),
    "downside_vol_20d": ("Downside vol, 20 days (%)", "Volatility", "Annualised root-mean-square of negative returns only."),
    "vol_ratio_5_60": ("Vol ratio 5d / 60d", "Volatility", ">1 means a recent volatility shock."),
    "vol_change_20d": ("Vol change vs 20 days ago", "Volatility", "vol_20d today divided by vol_20d 20 days ago."),
    "vix_level":     ("VIX level", "Market", "Implied 30-day S&P 500 volatility (market's own forecast)."),
    "vix_change_5d": ("VIX change, 5 days (%)", "Market", "Percent change in VIX over a week."),
    "vix_vs_60d":    ("VIX / 60-day mean", "Market", "Whether fear is high relative to its recent norm."),
    "mkt_vol_20d":   ("SPY realised vol, 20 days (%)", "Market", "Equity-market volatility, shared by all assets."),
}


LOG_FEATURES = ["vol_5d", "vol_10d", "vol_20d", "vol_60d", "vol_120d", "ewma_vol",
                "downside_vol_20d", "vix_level", "mkt_vol_20d"]


def _asset_features(close: pd.Series, vix: pd.Series, mkt_vol20: pd.Series) -> pd.DataFrame:
    r = np.log(close / close.shift(1))
    f = pd.DataFrame(index=close.index)
    f["ret_1d"] = r * 100
    f["abs_ret_1d"] = r.abs() * 100
    f["ret_5d"] = r.rolling(5).sum() * 100
    f["ret_21d"] = r.rolling(21).sum() * 100
    f["drawdown_60d"] = (close / close.rolling(60).max() - 1) * 100
    for w in (5, 10, 20, 60, 120):
        f[f"vol_{w}d"] = r.rolling(w).std() * ANN
    # RiskMetrics: sigma^2_t = 0.94 sigma^2_{t-1} + 0.06 r_t^2 (zero-mean), alpha = 1 - lambda
    f["ewma_vol"] = np.sqrt((r ** 2).ewm(alpha=0.06, adjust=False).mean()) * ANN
    f["downside_vol_20d"] = np.sqrt((np.minimum(r, 0) ** 2).rolling(20).mean()) * ANN
    f["vol_ratio_5_60"] = f["vol_5d"] / f["vol_60d"].clip(lower=1e-6)
    f["vol_change_20d"] = f["vol_20d"] / f["vol_20d"].shift(20).clip(lower=1e-6)
    f["vix_level"] = vix
    f["vix_change_5d"] = (vix / vix.shift(5) - 1) * 100
    f["vix_vs_60d"] = vix / vix.rolling(60).mean()
    f["mkt_vol_20d"] = mkt_vol20
    # Target: realised vol over the NEXT 20 days (t+1..t+20)
    f["target_vol_20d"] = r.rolling(CONFIG.HORIZON).std().shift(-CONFIG.HORIZON) * ANN
    # Daily return of the next day is kept for portfolio analysis only (never a feature)
    f["next_ret"] = r.shift(-1)
    return f


def build_model_table(prices: pd.DataFrame, save: bool = True) -> Tuple[pd.DataFrame, List[str], List[str]]:
    assets = [c for c in prices.columns if c != "VIX"]
    vix = prices["VIX"]
    mkt = "SPY" if "SPY" in assets else assets[0]
    mkt_vol20 = np.log(prices[mkt]).diff().rolling(20).std() * ANN

    frames = []
    for a in assets:
        f = _asset_features(prices[a], vix, mkt_vol20)
        f["asset"] = a
        frames.append(f)
    table = pd.concat(frames)
    table.index.name = "Date"

    # Volatility levels and VIX enter the models in logs: they are strictly
    # positive and right-skewed, and with a log target this gives a log-log
    # (HAR-style) relation, so a VIX spike cannot blow up exp(linear) forecasts.
    # Raw columns are kept for the baselines and for display.
    for c in LOG_FEATURES:
        table[f"log_{c}"] = np.log(table[c].clip(lower=0.1))
    numeric_features = [f"log_{c}" if c in LOG_FEATURES else c for c in FEATURE_INFO]
    # Asset identity lets models learn that e.g. TLT and SPY have different vol levels.
    asset_dummies = []
    if len(assets) > 1:
        for a in assets:
            col = f"is_{a}"
            table[col] = (table["asset"] == a).astype(float)
            asset_dummies.append(col)
    feature_cols = numeric_features + asset_dummies

    # Drop warm-up rows (incomplete windows). Rows with a missing target
    # (the last 20 days) are kept separately for inference.
    table = table.dropna(subset=feature_cols)
    table = table.reset_index().sort_values(["Date", "asset"]).set_index("Date")

    if save:
        table.to_csv(CONFIG.FEATURES_FILE, index_label="Date")
    logger.info(f"Model table: {len(table)} asset-days, {len(feature_cols)} features, assets={assets}")
    return table, feature_cols, assets
