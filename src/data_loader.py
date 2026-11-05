"""
src/data_loader.py
=============================================================================
Market Data Ingestion & Caching Module (Exact 7-Column Input Schema)
Course: ES 335 - Machine Learning (Undergraduate Track)
Team: Siddharth Dangi (25110300) & Aryan Vatsa Chauhan (25110059)
=============================================================================
Loads the 7 canonical market data streams matching the user's specification:
  - SPY Open, High, Low, Close, Volume
  - QQQ Benchmark Close
  - CBOE VIX Close
"""

import logging
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np

from src.config import CONFIG, DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)


def load_raw_market_data(
    file_path: Optional[Path] = None,
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Loads raw 7-column market data table:
      ['Open', 'High', 'Low', 'Close', 'Volume', 'Benchmark_Close', 'VIX']
    
    Indexed by DatetimeIndex (Daily trading days).
    """
    path = file_path or CONFIG.RAW_DATA_FILE

    if use_cache and path.exists():
        logger.info(f"Loading cached 7-column market data from: {path}")
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df

    logger.info("Cache not found; downloading clean market data via yfinance...")
    try:
        import yfinance as yf
        tickers = [CONFIG.PRIMARY_ASSET, CONFIG.BENCHMARK_ASSET, CONFIG.VIX_INDEX]
        raw = yf.download(
            tickers=tickers,
            start=CONFIG.START_DATE,
            end=CONFIG.END_DATE,
            interval="1d",
            auto_adjust=False,
            progress=False
        )
        
        df = pd.DataFrame(index=raw.index)
        df["Open"] = raw[("Open", CONFIG.PRIMARY_ASSET)]
        df["High"] = raw[("High", CONFIG.PRIMARY_ASSET)]
        df["Low"] = raw[("Low", CONFIG.PRIMARY_ASSET)]
        df["Close"] = raw[("Close", CONFIG.PRIMARY_ASSET)]
        df["Volume"] = raw[("Volume", CONFIG.PRIMARY_ASSET)]
        df["Benchmark_Close"] = raw[("Close", CONFIG.BENCHMARK_ASSET)]
        df["VIX"] = raw[("Close", CONFIG.VIX_INDEX)]
        
        df = df.dropna().copy()
        df.to_csv(path, index_label="Date")
        logger.info(f"Downloaded and saved clean 7-column market data to: {path} ({len(df)} days)")
        return df
    except Exception as e:
        logger.error(f"Failed to fetch market data via yfinance: {e}")
        raise RuntimeError(f"Market data file not found at {path} and live download failed.") from e


if __name__ == "__main__":
    data = load_raw_market_data(use_cache=True)
    print("Market Data Shape:", data.shape)
    print("Columns:", data.columns.tolist())
    print(data.head(3))
