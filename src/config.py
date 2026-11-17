"""
src/config.py
=============================================================================
Single source of truth for every choice that affects the results:
asset universe, dates, split boundaries, forecast horizon, seeds.
Course: ES 335 - Machine Learning (Undergraduate Track)
Team: Siddharth Dangi (25110300) & Aryan Vatsa Chauhan (25110059)
=============================================================================
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
DOCS_DIR = BASE_DIR / "docs"            # static build of the website (GitHub Pages serves this folder)
WEB_DIR = BASE_DIR / "web"               # Next.js + TypeScript source of the website
DOCS_DATA_DIR = WEB_DIR / "src" / "data"  # site_data.json, imported by the Next.js pages at build time
DOCS_FIG_DIR = WEB_DIR / "public" / "figures"

for _d in (DATA_DIR, ARTIFACTS_DIR, FIGURES_DIR, DOCS_DATA_DIR, DOCS_FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ProjectConfig:
    SEED: int = 42

    # ---- Asset universe -------------------------------------------------
    # One row of the modelling table = one asset on one trading day.
    # The pipeline uses every asset below that is present in data/prices.csv.
    # Run `python scripts/run_all.py --refresh-data` to download all of them.
    ASSETS: Dict[str, str] = field(default_factory=lambda: {
        "SPY": "US large-cap equities (S&P 500 ETF)",
        "QQQ": "US tech-heavy equities (Nasdaq-100 ETF)",
        "TLT": "Long-term US Treasury bonds (20+ yr ETF)",
        "GLD": "Gold (SPDR Gold Shares ETF)",
    })
    VIX_TICKER: str = "^VIX"          # market-wide implied volatility, used as a feature only

    START_DATE: str = "2012-01-01"
    END_DATE: str = "2024-11-01"      # exclusive upper bound for the download

    # ---- Prediction task ------------------------------------------------
    # Target: annualised realised volatility (%) of the asset over the NEXT
    # 20 trading days (t+1 ... t+20), using only information known at the
    # close of day t.
    HORIZON: int = 20
    TRADING_DAYS: int = 252

    # ---- Chronological split (dates refer to the forecast origin t) -----
    TRAIN_END: str = "2020-12-31"     # 2012-2020: includes 2015-16, 2018 and the 2020 COVID shock
    VAL_START: str = "2021-01-01"     # 2021 (calm) + 2022 (bear market): tuning sees both regimes
    VAL_END: str = "2022-12-31"
    TEST_START: str = "2023-01-01"    # 2023-2024: untouched until the single final evaluation
    # Rows whose 20-day target window crosses a split boundary are dropped
    # (an "embargo" of HORIZON days) so no target overlaps the next split.

    # ---- Model selection ------------------------------------------------
    CV_FOLDS: int = 5                 # expanding-window folds inside the training period
    PRIMARY_METRIC: str = "RMSE"      # used for every selection decision
    BOOTSTRAP_REPS: int = 2000        # moving-block bootstrap for test-set uncertainty
    BOOTSTRAP_BLOCK: int = 20         # block length (days) = horizon, because targets overlap

    PRICES_FILE: Path = DATA_DIR / "prices.csv"
    LEGACY_FILE: Path = DATA_DIR / "market_data_7_features.csv"
    FEATURES_FILE: Path = DATA_DIR / "model_table.csv"


CONFIG = ProjectConfig()
