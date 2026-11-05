"""
src/config.py
=============================================================================
Configuration & Global Constants for 1-Day Risk & Return Machine Learning Platform
Course: ES 335 - Machine Learning (Undergraduate Track)
Team: Siddharth Dangi (25110300) & Aryan Vatsa Chauhan (25110059)
=============================================================================
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple

# Project Base Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DOCS_DIR = BASE_DIR / "docs"
DOCS_DATA_DIR = DOCS_DIR / "data"

for directory in [DATA_DIR, ARTIFACTS_DIR, DOCS_DIR, DOCS_DATA_DIR]:
    os.makedirs(directory, exist_ok=True)


@dataclass(frozen=True)
class ProjectConfig:
    """Project-wide configuration constants for 1-day ML models."""

    # Deterministic Seed
    SEED: int = 42

    # Target Market Assets (Exact 7-column schema)
    PRIMARY_ASSET: str = "SPY"          # SPDR S&P 500 ETF (OHLCV)
    BENCHMARK_ASSET: str = "QQQ"        # Invesco QQQ Trust (Benchmark Close)
    VIX_INDEX: str = "^VIX"             # CBOE Volatility Index (Market Fear Gauge)

    # 1-Day Prediction Horizon
    HORIZON_DAYS: int = 1

    # Date Range (2012-01-01 to 2024-10-31: 3,167 Trading Days)
    START_DATE: str = "2012-01-01"
    END_DATE: str = "2024-10-31"

    # Chronological Walk-Forward Splits (Zero Lookahead Bias)
    TRAIN_START: str = "2012-01-01"
    TRAIN_END: str = "2021-12-31"       # Train split: 2,457 days (~10 years)
    VAL_START: str = "2022-01-01"
    VAL_END: str = "2022-12-31"         # Validation split: 251 days (2022 Bear Market)
    TEST_START: str = "2023-01-01"
    TEST_END: str = "2024-10-31"        # Out-of-sample Test split: 459 days (2023-2024)

    # Risk Targets
    # Next day return in percent: target_next_ret
    # Extreme tail crash threshold (historical 5th percentile): -1.60%
    TAIL_RISK_THRESHOLD: float = -1.60

    # Data file paths
    RAW_DATA_FILE: Path = DATA_DIR / "market_data_7_features.csv"
    ENGINEERED_DATA_FILE: Path = DATA_DIR / "dataset_22_features.csv"


# Global Config Instance
CONFIG = ProjectConfig()
