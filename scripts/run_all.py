"""
scripts/run_all.py
=============================================================================
Master Execution Script - End-to-End Pipeline & Web Data Generator
Course: ES 335 - Machine Learning (Undergraduate Track)
Team: Siddharth Dangi (25110300) & Aryan Vatsa Chauhan (25110059)
=============================================================================
Usage:
  python scripts/run_all.py
"""

import sys
import time
import logging
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.train_pipeline import run_full_pipeline
from scripts.export_web_data import export_for_github_pages
from scripts.generate_real_figures import generate_all_figures

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("MASTER_RUNNER")


def main():
    start_time = time.time()
    logger.info("=" * 70)
    logger.info("STARTING ES 335 PORTFOLIO RISK ANALYZER COMPLETE PIPELINE")
    logger.info("Project Team: Siddharth Dangi (25110300) & Aryan Vatsa Chauhan (25110059)")
    logger.info("=" * 70)
    
    # 1. Execute ML training, walk-forward evaluation, and figure generation
    logger.info("\n>>> STEP 1: Running Model Training & Benchmarking Pipeline...")
    evaluation_results, adf_summary = run_full_pipeline()
    
    # 2. Export interactive datasets for GitHub Pages
    logger.info("\n>>> STEP 2: Exporting Interactive Datasets to docs/data/...")
    export_for_github_pages()

    # 3. Generate Real Publication Figures
    logger.info("\n>>> STEP 3: Generating Real Publication Figures from Out-of-Sample Test Set...")
    generate_all_figures()
    
    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info(f"PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS!")
    logger.info("Web deliverables ready in docs/ for GitHub Pages hosting.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
