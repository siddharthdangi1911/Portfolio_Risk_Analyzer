"""
src/model_selection.py
=============================================================================
Chronological splits and the three-stage selection protocol.

  Stage 1  SCREEN  every candidate family with expanding-window CV that
                   uses ONLY the training period  -> pick the best family
  Stage 2  TUNE    the selected family's hyperparameters: fit on train,
                   score on the validation period -> pick best settings
  Stage 3  TEST    refit once on train+validation, evaluate once on the
                   untouched test period (done in train_pipeline.py)

Embargo: a row dated t has a target built from t+1..t+20. Any row whose
target window would reach into the next split is dropped, and every CV fold
leaves a gap of HORIZON trading days between its training and scoring
dates. Without this, the last 20 training targets would contain returns
from the validation/test period (leakage).
=============================================================================
"""

import logging
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.evaluation import rmse
from src.models.ml_models import (FAMILIES, SCREEN_GRIDS, TUNE_GRIDS, make_baselines, make_model)

logger = logging.getLogger(__name__)
H = CONFIG.HORIZON


def _dates_before_embargo(dates: pd.DatetimeIndex, end: str) -> pd.DatetimeIndex:
    """Trading dates <= end, minus the last H of them (their targets cross `end`)."""
    d = dates[dates <= pd.Timestamp(end)]
    return d[:-H] if len(d) > H else d[:0]


def make_splits(table: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    labelled = table[table["target_vol_20d"].notna()]
    all_dates = pd.DatetimeIndex(sorted(table.index.unique()))   # full calendar, incl. unlabelled tail

    train_dates = _dates_before_embargo(all_dates, CONFIG.TRAIN_END)
    val_all = all_dates[(all_dates >= pd.Timestamp(CONFIG.VAL_START))]
    val_dates = _dates_before_embargo(val_all, CONFIG.VAL_END)
    trval_dates = _dates_before_embargo(all_dates, CONFIG.VAL_END)

    splits = {
        "train": labelled[labelled.index.isin(train_dates)],
        "val": labelled[labelled.index.isin(val_dates)],
        "train_val": labelled[labelled.index.isin(trval_dates)],
        "test": labelled[labelled.index >= pd.Timestamp(CONFIG.TEST_START)],
        "inference": table[table["target_vol_20d"].isna()],
    }
    return splits


def split_summary(splits: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
    out = {}
    for k in ("train", "val", "test"):
        s = splits[k]
        out[k] = {"start": s.index.min().strftime("%Y-%m-%d"), "end": s.index.max().strftime("%Y-%m-%d"),
                  "dates": int(s.index.nunique()), "rows": int(len(s))}
    return out


def expanding_folds(df: pd.DataFrame, n_folds: int = CONFIG.CV_FOLDS) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Date-based expanding-window folds with an H-day gap. Splitting on dates
    (not rows) keeps all assets of one day on the same side of a split."""
    dates = np.array(sorted(df.index.unique()))
    chunks = np.array_split(dates, n_folds + 1)
    folds = []
    for k in range(1, n_folds + 1):
        tr_dates = np.concatenate(chunks[:k])[:-H]
        va_dates = chunks[k]
        tr = np.flatnonzero(df.index.isin(tr_dates))
        va = np.flatnonzero(df.index.isin(va_dates))
        folds.append((tr, va))
    return folds


def _fit_predict(family: str, params: Dict, Xtr, ytr, Xva) -> np.ndarray:
    m = make_model(family, params)
    m.fit(Xtr, ytr)
    return m.predict(Xva)


def screen_families(train: pd.DataFrame, feats: List[str]) -> Dict[str, Any]:
    """Stage 1: CV score of every family (best of its small screening grid) and of the baselines."""
    folds = expanding_folds(train)
    X = train[feats].to_numpy()
    y = train["target_vol_20d"].to_numpy()
    fold_info = [{"fold": i + 1,
                  "train_end": str(train.index[tr].max().date()), "score_start": str(train.index[va].min().date()),
                  "score_end": str(train.index[va].max().date()), "train_rows": int(len(tr)), "score_rows": int(len(va))}
                 for i, (tr, va) in enumerate(folds)]

    results = []
    for b in make_baselines():
        scores = []
        for tr, va in folds:
            b.fit(train.iloc[tr], y[tr])
            scores.append(rmse(y[va], b.predict(train.iloc[va])))
        logger.info(f"  [screen] {b.name:<26} CV RMSE {np.mean(scores):.3f} +/- {np.std(scores):.3f}  (baseline)")
        results.append({"name": b.name, "kind": "baseline", "cv_rmse_mean": float(np.mean(scores)),
                        "cv_rmse_std": float(np.std(scores)), "fold_rmse": scores, "best_params": None,
                        "fits": 0, "seconds": 0.0})

    for fam in FAMILIES:
        t0 = time.perf_counter()
        best = None
        n_fits = 0
        for params in SCREEN_GRIDS[fam]:
            scores = []
            for tr, va in folds:
                scores.append(rmse(y[va], _fit_predict(fam, params, X[tr], y[tr], X[va])))
                n_fits += 1
            cand = (float(np.mean(scores)), float(np.std(scores)), scores, params)
            if best is None or cand[0] < best[0]:
                best = cand
        secs = time.perf_counter() - t0
        logger.info(f"  [screen] {fam:<26} CV RMSE {best[0]:.3f} +/- {best[1]:.3f}  ({n_fits} fits, {secs:.1f}s)")
        results.append({"name": fam, "kind": "model", "cv_rmse_mean": best[0], "cv_rmse_std": best[1],
                        "fold_rmse": best[2], "best_params": best[3], "fits": n_fits, "seconds": round(secs, 2)})

    models = [r for r in results if r["kind"] == "model"]
    winner = min(models, key=lambda r: r["cv_rmse_mean"])
    return {"folds": fold_info, "results": results, "selected_family": winner["name"]}


def tune_on_validation(family: str, train: pd.DataFrame, val: pd.DataFrame, feats: List[str]) -> Dict[str, Any]:
    """Stage 2: every config of the selected family is fit on train and scored on validation."""
    Xtr, ytr = train[feats].to_numpy(), train["target_vol_20d"].to_numpy()
    Xva, yva = val[feats].to_numpy(), val["target_vol_20d"].to_numpy()
    t0 = time.perf_counter()
    trials = []
    for params in TUNE_GRIDS[family]:
        p = _fit_predict(family, params, Xtr, ytr, Xva)
        trials.append({"params": params, "val_rmse": rmse(yva, p),
                       "val_mae": float(np.mean(np.abs(p - yva)))})
    trials.sort(key=lambda t: t["val_rmse"])
    secs = time.perf_counter() - t0
    logger.info(f"  [tune]   {family}: {len(trials)} configs in {secs:.1f}s; best val RMSE {trials[0]['val_rmse']:.3f} "
                f"with {trials[0]['params']}")
    return {"family": family, "n_configs": len(trials), "seconds": round(secs, 2),
            "best_params": trials[0]["params"], "trials": trials}
