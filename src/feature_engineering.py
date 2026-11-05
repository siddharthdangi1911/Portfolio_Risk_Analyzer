"""
src/feature_engineering.py
=============================================================================
Machine Learning Feature Engineering Pipeline: 1-Day Stock Risk Forecasting
Course: ES 335 - Machine Learning
=============================================================================
Constructs the 22 engineered features and 1-day prediction targets directly
from the 7 raw market input streams matching the user's specification:
  - Input Columns (7): Open, High, Low, Close, Volume, Benchmark_Close, VIX
  - Engineered Features (22):
      ret, ret_lag_1 .. ret_lag_5, ret_21d, drawdown_60d,
      vol_5d, vol_20d, vol_60d, vol_ratio, ewma_vol,
      parkinson_vol, parkinson_vol_10d,
      skew_30d, kurt_30d, hist_var_95_60d,
      beta_60d, vix_level, vix_change_5d, rel_volume_20d
  - Target Columns (2):
      target_next_ret: 1-Day Forward Log Return in %
      target_high_risk: Binary indicator (1 if next-day return <= -1.60%, bottom 5% tail loss)
"""

import logging
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from src.config import CONFIG

logger = logging.getLogger(__name__)

FEATURE_METADATA = {
    "ret": "Current day's logarithmic return: ln(Close_t / Close_{t-1})",
    "ret_lag_1": "1-day lagged logarithmic return",
    "ret_lag_2": "2-day lagged logarithmic return",
    "ret_lag_3": "3-day lagged logarithmic return",
    "ret_lag_4": "4-day lagged logarithmic return",
    "ret_lag_5": "5-day lagged logarithmic return",
    "ret_21d": "Cumulative percentage return over the past 21 trading days (1 month)",
    "drawdown_60d": "Percentage drawdown from the highest closing price in past 60 trading days",
    "vol_5d": "5-day rolling realized volatility (standard deviation of daily returns)",
    "vol_20d": "20-day rolling realized volatility",
    "vol_60d": "60-day rolling realized volatility",
    "vol_ratio": "Ratio of short-term volatility (vol_5d) to medium-term volatility (vol_60d)",
    "ewma_vol": "Exponentially Weighted Moving Average volatility (alpha = 0.06)",
    "parkinson_vol": "Single-day intraday range volatility from High and Low prices",
    "parkinson_vol_10d": "10-day rolling average of Parkinson intraday volatility",
    "skew_30d": "30-day rolling skewness of daily returns",
    "kurt_30d": "30-day rolling kurtosis of daily returns",
    "hist_var_95_60d": "60-day rolling historical 5th percentile return (empirical tail floor)",
    "beta_60d": "60-day rolling market beta relative to benchmark index",
    "vix_level": "Current closing level of the VIX volatility index",
    "vix_change_5d": "5-day percentage change in the VIX index",
    "rel_volume_20d": "Ratio of current trading volume to 20-day average volume",
}


def build_ml_feature_dataset(
    raw_data: pd.DataFrame,
    primary_ticker: str = CONFIG.PRIMARY_ASSET,
    bench_ticker: str = CONFIG.BENCHMARK_ASSET,
    save_csv: bool = True
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Constructs the complete 22-feature matrix and next-day target variables from raw 7-column market data.
    """
    logger.info("Extracting columns for feature engineering...")

    if isinstance(raw_data.columns, pd.MultiIndex):
        close = raw_data[("Close", primary_ticker)]
        open_px = raw_data[("Open", primary_ticker)]
        high = raw_data[("High", primary_ticker)]
        low = raw_data[("Low", primary_ticker)]
        volume = raw_data[("Volume", primary_ticker)]
        bench_close = raw_data[("Close", bench_ticker)]
        vix = raw_data[("Close", "^VIX")]
    else:
        close = raw_data["Close"]
        open_px = raw_data["Open"]
        high = raw_data["High"]
        low = raw_data["Low"]
        volume = raw_data["Volume"]
        bench_close = raw_data["Benchmark_Close"]
        vix = raw_data["VIX"]

    # Daily log returns
    ret = np.log(close / close.shift(1))
    bench_ret = np.log(bench_close / bench_close.shift(1))

    df = pd.DataFrame(index=close.index)

    # 1. Log returns and lags 1 through 5
    df["ret"] = ret
    for k in range(1, 6):
        df[f"ret_lag_{k}"] = ret.shift(k)

    # 2. Cumulative return and drawdown
    df["ret_21d"] = (close / close.shift(21) - 1.0) * 100.0
    df["drawdown_60d"] = (close / close.rolling(60).max() - 1.0) * 100.0

    # 3. Rolling volatilities
    df["vol_5d"] = ret.rolling(5).std() * np.sqrt(252.0) * 100.0
    df["vol_20d"] = ret.rolling(20).std() * np.sqrt(252.0) * 100.0
    df["vol_60d"] = ret.rolling(60).std() * np.sqrt(252.0) * 100.0
    df["vol_ratio"] = df["vol_5d"] / np.maximum(1e-4, df["vol_60d"])

    # 4. EWMA Volatility (alpha = 0.06)
    df["ewma_vol"] = ret.ewm(alpha=0.06).std() * np.sqrt(252.0) * 100.0

    # 5. Parkinson Range Volatility
    ratio = np.maximum(high / np.maximum(low, 1e-8), 1.0)
    park_var = (1.0 / (4.0 * np.log(2.0))) * (np.log(ratio) ** 2)
    df["parkinson_vol"] = np.sqrt(park_var * 252.0) * 100.0
    df["parkinson_vol_10d"] = df["parkinson_vol"].rolling(10).mean()

    # 6. Higher Moments & Empirical Quantile
    df["skew_30d"] = ret.rolling(30).skew()
    df["kurt_30d"] = ret.rolling(30).kurt()
    df["hist_var_95_60d"] = ret.rolling(60).quantile(0.05) * 100.0

    # 7. Rolling Beta relative to Benchmark
    cov_60 = ret.rolling(60).cov(bench_ret)
    var_60 = bench_ret.rolling(60).var()
    df["beta_60d"] = cov_60 / np.maximum(1e-6, var_60)

    # 8. Macro Indicators & Relative Volume
    df["vix_level"] = vix
    df["vix_change_5d"] = (vix / vix.shift(5) - 1.0) * 100.0
    df["rel_volume_20d"] = volume / np.maximum(1.0, volume.rolling(20).mean())

    # 9. Targets (1-Day Horizon)
    # Next day return in percent (%)
    df["target_next_ret"] = ret.shift(-1) * 100.0
    
    # High-Risk threshold (bottom 5% tail drop across training distribution)
    # Target 1 = High Risk day, 0 = Normal day
    q05_threshold = CONFIG.TAIL_RISK_THRESHOLD  # -1.60%
    df["target_high_risk"] = (df["target_next_ret"] <= q05_threshold).astype(int)

    # Clean out NaN warmup rows
    clean_df = df.dropna().copy()
    feature_cols = [c for c in clean_df.columns if not c.startswith("target_")]

    if save_csv:
        clean_df.to_csv(CONFIG.ENGINEERED_DATA_FILE, index_label="Date")
        logger.info(f"Saved engineered dataset to: {CONFIG.ENGINEERED_DATA_FILE}")

    logger.info(f"Feature matrix engineered: {clean_df.shape[0]} samples x {len(feature_cols)} features.")
    return clean_df, feature_cols
