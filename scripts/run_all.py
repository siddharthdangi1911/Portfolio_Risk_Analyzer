"""
scripts/run_all.py
=============================================================================
One command to reproduce everything:
    python scripts/run_all.py                 # use cached data/prices.csv
    python scripts/run_all.py --refresh-data  # re-download SPY, QQQ, TLT, GLD, ^VIX
Steps: train/select/evaluate -> export website data -> figures.
Team: Siddharth Dangi (25110300) & Aryan Vatsa Chauhan (25110059)
=============================================================================
"""

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.export_web_data import export_for_github_pages
from scripts.generate_real_figures import generate_all_figures
from src.train_pipeline import run_full_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
for noisy in ("yfinance", "peewee", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.CRITICAL)
logger = logging.getLogger("run_all")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh-data", action="store_true", help="re-download prices from Yahoo Finance")
    args = ap.parse_args()
    t0 = time.time()
    logger.info("STEP 1/3  model screening, tuning and final test evaluation")
    run_full_pipeline(refresh_data=args.refresh_data)
    logger.info("STEP 2/3  exporting website data")
    export_for_github_pages()
    logger.info("STEP 3/3  figures")
    generate_all_figures()
    logger.info(f"Done in {time.time() - t0:.1f}s. Next: cd web && npm run export:docs to rebuild the website.")


if __name__ == "__main__":
    main()
