"""
src/inference.py
=============================================================================
Inference with the single selected model (artifacts/final_model.joblib).

For the most recent date in the price data it builds exactly the same
features used in training, then reports for every asset:
  * the model's forecast of annualised volatility over the next 20 trading days
  * the baselines (recent 20-day vol, EWMA) for comparison
and, for a chosen set of weights, an ex-ante portfolio volatility forecast
  sigma_p = sqrt(w' D C D w)   (D = forecast vols, C = trailing 60-day correlation)

Usage
  python -m src.inference
  python -m src.inference --weights SPY=0.6,QQQ=0.4
  python -m src.inference --date 2024-08-02          # forecast as of a past date
  python -m src.inference --refresh-data             # pull latest prices first
=============================================================================
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd

from src.config import ARTIFACTS_DIR, CONFIG
from src.data_loader import clean_and_check, load_prices
from src.feature_engineering import build_model_table
from src.portfolio import trailing_correlations


class VolatilityForecaster:
    def __init__(self, model_path: Path = ARTIFACTS_DIR / "final_model.joblib"):
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} not found - run `python scripts/run_all.py` first.")
        self.bundle = joblib.load(model_path)
        self.model = self.bundle["model"]
        self.features = self.bundle["features"]

    def forecast(self, prices: pd.DataFrame, as_of: Optional[str] = None,
                 weights: Optional[Dict[str, float]] = None) -> Dict:
        table, feats, assets = build_model_table(prices, save=False)
        if feats != self.features:
            raise ValueError("Feature set differs from the trained model (asset universe changed?). Re-run training.")
        if as_of is not None:
            table = table[table.index <= pd.Timestamp(as_of)]
        last_date = table.index.max()
        rows = table.loc[[last_date]]

        pred = self.model.predict(rows[feats].to_numpy())
        out = {"as_of": last_date.strftime("%Y-%m-%d"), "model": f"{self.bundle['family']} {self.bundle['params']}",
               "trained_through": self.bundle["trained_through"], "assets": {}}
        for (_, r), p in zip(rows.iterrows(), pred):
            realised = r["target_vol_20d"]
            out["assets"][r["asset"]] = {
                "forecast_vol_20d": round(float(p), 2),
                "recent_20d_vol": round(float(r["vol_20d"]), 2),
                "ewma_vol": round(float(r["ewma_vol"]), 2),
                "vix": round(float(r["vix_level"]), 2),
                "realised_next_20d": None if pd.isna(realised) else round(float(realised), 2),
            }

        if weights:
            rets = np.log(prices[assets]).diff()
            corr = trailing_correlations(rets).loc[:last_date].iloc[-1]
            C = np.eye(len(assets))
            for k, v in corr.items():
                a, b = k.split("|")
                i, j = assets.index(a), assets.index(b)
                C[i, j] = C[j, i] = v
            w = np.array([weights.get(a, 0.0) for a in assets], float)
            w = w / w.sum()
            for key, label in (("forecast_vol_20d", "model"), ("recent_20d_vol", "baseline")):
                d = np.array([out["assets"][a][key] for a in assets])
                out.setdefault("portfolio", {})[f"{label}_vol_20d"] = round(float(np.sqrt(w @ (np.outer(d, d) * C) @ w)), 2)
            out["portfolio"]["weights"] = {a: round(float(x), 3) for a, x in zip(assets, w)}
        return out


def _parse_weights(s: Optional[str]) -> Optional[Dict[str, float]]:
    if not s:
        return None
    return {k.strip().upper(): float(v) for k, v in (p.split("=") for p in s.split(","))}


def main():
    logging.basicConfig(level=logging.WARNING)
    ap = argparse.ArgumentParser(description="20-day volatility forecast with the selected model")
    ap.add_argument("--date", help="forecast origin (YYYY-MM-DD); default = latest available")
    ap.add_argument("--weights", help="portfolio weights, e.g. SPY=0.6,TLT=0.4")
    ap.add_argument("--refresh-data", action="store_true")
    a = ap.parse_args()

    prices, _ = clean_and_check(load_prices(refresh=a.refresh_data))
    f = VolatilityForecaster()
    weights = _parse_weights(a.weights) or {x: 1.0 for x in f.bundle["assets"]}
    r = f.forecast(prices, as_of=a.date, weights=weights)

    print("\n" + "=" * 72)
    print(f" 20-day volatility forecast as of {r['as_of']}   (model: {r['model']})")
    print(f" model trained on data through {r['trained_through']}")
    print("=" * 72)
    print(f" {'asset':<6}{'forecast':>10}{'recent 20d':>12}{'EWMA':>8}{'VIX':>7}{'realised*':>11}")
    for k, v in r["assets"].items():
        real = "-" if v["realised_next_20d"] is None else f"{v['realised_next_20d']:.2f}"
        print(f" {k:<6}{v['forecast_vol_20d']:>10.2f}{v['recent_20d_vol']:>12.2f}{v['ewma_vol']:>8.2f}{v['vix']:>7.2f}{real:>11}")
    if "portfolio" in r:
        p = r["portfolio"]
        print("-" * 72)
        print(f" portfolio {p['weights']}")
        print(f"   forecast vol (model) {p['model_vol_20d']:.2f}%   |   baseline (recent vol) {p['baseline_vol_20d']:.2f}%")
    print(" * realised = actual vol over the next 20 days, only known for past dates")
    print(" Volatility is annualised, in %. Forecasts are statistical estimates, not investment advice.\n")


if __name__ == "__main__":
    main()
