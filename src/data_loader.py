"""
src/data_loader.py
=============================================================================
Loads daily closing prices for the asset universe plus the VIX index.

Source: Yahoo Finance via the `yfinance` package (public, no credentials).
Cache:  data/prices.csv  (wide table: Date index, one column per ticker, VIX).

Order of preference
  1. data/prices.csv if present (unless refresh=True)
  2. download with yfinance (adjusted closes, so dividends/splits do not
     appear as fake returns)
  3. fall back to the older data/market_data_7_features.csv (SPY, QQQ, VIX
     only; unadjusted closes) so the pipeline still runs offline.
=============================================================================
"""

import json
import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.config import ARTIFACTS_DIR, CONFIG

logger = logging.getLogger(__name__)


def _download_yfinance() -> pd.DataFrame:
    import yfinance as yf

    tickers = list(CONFIG.ASSETS) + [CONFIG.VIX_TICKER]
    raw = yf.download(
        tickers=tickers,
        start=CONFIG.START_DATE,
        end=CONFIG.END_DATE,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned no data")
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    close = close.rename(columns={CONFIG.VIX_TICKER: "VIX"})
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close.index.name = "Date"
    keep = [t for t in CONFIG.ASSETS if t in close.columns and close[t].notna().sum() > 0]
    if "VIX" not in close.columns:
        raise RuntimeError("VIX missing from download")
    prices = close[keep + ["VIX"]].copy()
    prices.attrs["source"] = "yfinance (auto_adjust=True)"
    return prices


def _load_legacy() -> pd.DataFrame:
    df = pd.read_csv(CONFIG.LEGACY_FILE, index_col=0, parse_dates=True)
    prices = pd.DataFrame(
        {"SPY": df["Close"], "QQQ": df["Benchmark_Close"], "VIX": df["VIX"]},
        index=df.index,
    )
    prices.index.name = "Date"
    prices.attrs["source"] = "legacy market_data_7_features.csv (yfinance, unadjusted closes)"
    return prices


def load_prices(refresh: bool = False) -> pd.DataFrame:
    path = CONFIG.PRICES_FILE
    source_file = path.with_suffix(".source.txt")

    if path.exists() and not refresh:
        prices = pd.read_csv(path, index_col=0, parse_dates=True)
        prices.attrs["source"] = source_file.read_text().strip() if source_file.exists() else "data/prices.csv"
        logger.info(f"Loaded cached prices {path.name}: {prices.shape}, assets={[c for c in prices if c != 'VIX']}")
        return prices

    try:
        logger.info("Downloading prices from Yahoo Finance ...")
        prices = _download_yfinance()
    except Exception as exc:  # network blocked, package missing, API change ...
        if refresh:
            logger.warning(f"Download failed ({exc}).")
        if not CONFIG.LEGACY_FILE.exists():
            raise RuntimeError("No cached prices and download failed.") from exc
        logger.warning("Falling back to legacy SPY/QQQ/VIX file.")
        prices = _load_legacy()

    prices.to_csv(path, index_label="Date")
    source_file.write_text(prices.attrs.get("source", "unknown"))
    logger.info(f"Saved {path.name}: {prices.shape}")
    return prices


def clean_and_check(prices: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Quality checks + light cleaning. Returns cleaned prices and a report."""
    report: Dict = {"source": prices.attrs.get("source", "unknown")}
    p = prices.sort_index().copy()

    dup = int(p.index.duplicated().sum())
    p = p[~p.index.duplicated(keep="first")]
    report["duplicate_dates_removed"] = dup
    report["missing_values_before"] = {c: int(p[c].isna().sum()) for c in p.columns}

    nonpos = {c: int((p[c] <= 0).sum()) for c in p.columns}
    report["non_positive_prices"] = nonpos
    for c in p.columns:
        p.loc[p[c] <= 0, c] = np.nan

    # Align: keep only days where every series exists. A short forward fill
    # (max 2 days) covers isolated holiday mismatches (e.g. VIX vs ETF calendars).
    p = p.ffill(limit=2).dropna()
    report["rows_after_alignment"] = int(len(p))
    report["date_range"] = [p.index.min().strftime("%Y-%m-%d"), p.index.max().strftime("%Y-%m-%d")]

    rets = np.log(p.drop(columns="VIX")).diff()
    extreme = {}
    for c in rets.columns:
        big = rets[c][rets[c].abs() > 0.10]
        extreme[c] = [{"date": d.strftime("%Y-%m-%d"), "log_return_pct": round(float(v) * 100, 2)} for d, v in big.items()]
    report["daily_moves_over_10pct"] = extreme

    gaps = p.index.to_series().diff().dt.days
    report["calendar_gaps_over_4_days"] = int((gaps > 4).sum())
    report["assets"] = [c for c in p.columns if c != "VIX"]

    (ARTIFACTS_DIR / "data_quality.json").write_text(json.dumps(report, indent=2))
    return p, report
