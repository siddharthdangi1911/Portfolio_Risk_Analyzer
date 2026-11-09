"""
src/evaluation.py
=============================================================================
Metrics and error analysis for volatility forecasts (all in vol % units).

  RMSE  - primary metric (penalises large misses, e.g. missing a vol spike)
  MAE   - typical miss in vol points, easy to explain
  R2    - share of variance explained (relative to predicting the test mean)
  QLIKE - standard loss for volatility forecasts; robust to noise in the
          realised-vol proxy and penalises under-prediction of risk more
          than over-prediction:  QLIKE = mean( s2/h2 - log(s2/h2) - 1 )
  Bias  - mean(prediction - actual); negative = risk under-estimated
=============================================================================
"""

from typing import Dict, List

import numpy as np
import pandas as pd

from src.config import CONFIG


def regression_metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 1e-3, None)
    e = p - y
    ratio = (y ** 2) / (p ** 2)
    return {
        "RMSE": float(np.sqrt(np.mean(e ** 2))),
        "MAE": float(np.mean(np.abs(e))),
        "R2": float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)),
        "QLIKE": float(np.mean(ratio - np.log(ratio) - 1)),
        "Bias": float(np.mean(e)),
        "n": int(len(y)),
    }


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2)))


def block_bootstrap_compare(dates: pd.Index, y: np.ndarray, p_model: np.ndarray, p_base: np.ndarray,
                            reps: int = CONFIG.BOOTSTRAP_REPS, block: int = CONFIG.BOOTSTRAP_BLOCK,
                            seed: int = CONFIG.SEED) -> Dict[str, float]:
    """Moving-block bootstrap over trading DATES (all assets of a date move
    together). Blocks keep the autocorrelation created by overlapping
    20-day targets, which an ordinary bootstrap would ignore."""
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(pd.unique(dates)))
    pos = {d: i for i, d in enumerate(uniq)}
    di = np.array([pos[d] for d in dates])
    se_m = (p_model - y) ** 2
    se_b = (p_base - y) ** 2
    # per-date sums so resampling is cheap
    n_d = len(uniq)
    sum_m = np.bincount(di, se_m, n_d)
    sum_b = np.bincount(di, se_b, n_d)
    cnt = np.bincount(di, minlength=n_d)
    n_blocks = int(np.ceil(n_d / block))
    rm, rb, diff = [], [], []
    for _ in range(reps):
        starts = rng.integers(0, n_d - block + 1, n_blocks)
        idx = (starts[:, None] + np.arange(block)).ravel()[:n_d]
        c = cnt[idx].sum()
        m = np.sqrt(sum_m[idx].sum() / c)
        b = np.sqrt(sum_b[idx].sum() / c)
        rm.append(m); rb.append(b); diff.append(m - b)
    diff = np.array(diff)
    return {
        "model_RMSE_CI95": [float(np.percentile(rm, 2.5)), float(np.percentile(rm, 97.5))],
        "baseline_RMSE_CI95": [float(np.percentile(rb, 2.5)), float(np.percentile(rb, 97.5))],
        "RMSE_diff_mean": float(diff.mean()),
        "RMSE_diff_CI95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
        "share_of_resamples_model_better": float(np.mean(diff < 0)),
    }


def regime_breakdown(df: pd.DataFrame, preds: Dict[str, np.ndarray], thresholds: List[float]) -> List[Dict]:
    """Errors in calm / normal / stressed markets. Regimes are defined by the
    VIX level on the forecast date, using tercile cut-offs from the TRAINING
    data so the definition does not peek at the test set."""
    lo, hi = thresholds
    vix = df["vix_level"].to_numpy()
    y = df["target_vol_20d"].to_numpy()
    masks = {
        f"Calm (VIX < {lo:.1f})": vix < lo,
        f"Normal ({lo:.1f}-{hi:.1f})": (vix >= lo) & (vix <= hi),
        f"Stressed (VIX > {hi:.1f})": vix > hi,
    }
    out = []
    for name, m in masks.items():
        row = {"regime": name, "n": int(m.sum())}
        for k, p in preds.items():
            row[k] = {"RMSE": rmse(y[m], p[m]) if m.any() else None,
                      "MAE": float(np.mean(np.abs(p[m] - y[m]))) if m.any() else None}
        out.append(row)
    return out
