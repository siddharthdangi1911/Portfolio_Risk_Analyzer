"""
scripts/export_web_data.py
=============================================================================
Packs the pipeline outputs into one data file for the static website:
  docs/data/site_data.js     (window.SITE_DATA = {...}; works on file:// too)
  docs/data/site_data.json   (same content, for anyone who prefers fetch())
Everything shown on the site comes from artifacts produced by
src/train_pipeline.py - nothing is typed in by hand.
=============================================================================
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.config import ARTIFACTS_DIR, CONFIG, DOCS_DATA_DIR
from src.data_loader import clean_and_check, load_prices
from src.models.ml_models import PRIMARY_BASELINE
from src.portfolio import presets, realised_metrics, trailing_correlations

logger = logging.getLogger(__name__)


def _rl(a, k=3):
    return [None if (v is None or (isinstance(v, float) and np.isnan(v))) else round(float(v), k) for v in a]


def export_for_github_pages() -> Path:
    res = json.loads((ARTIFACTS_DIR / "evaluation_metrics.json").read_text())
    preds = pd.read_csv(ARTIFACTS_DIR / "test_predictions.csv", index_col=0, parse_dates=True)
    prices, _ = clean_and_check(load_prices())
    assets = res["data"]["assets"]
    fam = res["final_model"]["family"]
    model_col = f"pred::{fam} (final)"

    # ---- forecast timeline per asset (test period) ----
    timeline = {}
    for a in assets:
        d = preds[preds["asset"] == a].sort_index()
        timeline[a] = {
            "dates": d.index.strftime("%Y-%m-%d").tolist(),
            "actual": _rl(d["target_vol_20d"]),
            "model": _rl(d[model_col]),
            "recent": _rl(d[f"pred::{PRIMARY_BASELINE}"]),
            "hist": _rl(d["pred::Historical average"]),
            "ewma": _rl(d["pred::EWMA vol (RiskMetrics)"]),
            "vix": _rl(d["vix_level"], 2),
        }

    # ---- portfolio data ----
    rets = np.log(prices[assets]).diff().dropna()
    test_start = pd.Timestamp(CONFIG.TEST_START)
    rets_test = rets[rets.index >= test_start]
    corr60 = trailing_correlations(rets)
    fdates = pd.DatetimeIndex(sorted(preds.index.unique()))
    pm = preds.pivot_table(index=preds.index, columns="asset", values=model_col).reindex(fdates)
    pb = preds.pivot_table(index=preds.index, columns="asset", values=f"pred::{PRIMARY_BASELINE}").reindex(fdates)
    train_rets = rets[rets.index <= pd.Timestamp(CONFIG.TRAIN_END)]

    pre = presets(assets)
    portfolio = {
        "return_dates": rets_test.index.strftime("%Y-%m-%d").tolist(),
        "returns_pct": {a: _rl(rets_test[a] * 100, 4) for a in assets},
        "forecast_dates": fdates.strftime("%Y-%m-%d").tolist(),
        "pred_model": {a: _rl(pm[a]) for a in assets},
        "pred_baseline": {a: _rl(pb[a]) for a in assets},
        "corr60": {k: _rl(v.reindex(fdates), 3) for k, v in corr60.items()},
        "corr_train": train_rets.corr().round(3).values.tolist(),
        "corr_test": rets_test.corr().round(3).values.tolist(),
        "presets": pre,
        "preset_metrics_test": {k: {m: round(x, 3) for m, x in realised_metrics(rets_test, w).items()} for k, w in pre.items()},
        "single_asset_metrics_test": {a: {m: round(x, 3) for m, x in realised_metrics(rets_test, {a: 1.0}).items()} for a in assets},
    }

    site = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "assets": assets,
        "asset_descriptions": res["data"]["asset_descriptions"],
        "results": res,
        "primary_baseline": PRIMARY_BASELINE,
        "timeline": timeline,
        "portfolio": portfolio,
    }
    js = "window.SITE_DATA = " + json.dumps(site, separators=(",", ":")) + ";\n"
    out_js = DOCS_DATA_DIR / "site_data.js"
    out_js.write_text(js)
    (DOCS_DATA_DIR / "site_data.json").write_text(json.dumps(site, separators=(",", ":")))
    logger.info(f"Wrote {out_js} ({out_js.stat().st_size / 1024:.0f} KB)")
    return out_js


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    export_for_github_pages()
