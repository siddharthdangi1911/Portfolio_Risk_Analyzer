"""
src/models/ml_models.py
=============================================================================
Baselines and candidate model families for 20-day volatility forecasting.

Baselines (no learning, or a single mean):
  * HistoricalAverage  - each asset's average realised vol in the training data
                         (the "simple historical average" in our research question)
  * RecentVol20        - today's trailing 20-day vol (vol persistence / random walk)
  * EWMAVol            - RiskMetrics EWMA vol (classic industry benchmark)

Candidate ML families (all wrapped in a scikit-learn Pipeline so that the
StandardScaler is fitted only on whatever data the model is trained on -
no leakage across CV folds, validation or test):
  * Ridge regression            - linear, regularised; vol is highly persistent so
                                   a linear combination of past vols is a strong model
  * Decision tree               - simple non-linear rules (regimes)
  * Random forest               - averaged trees, captures interactions, low variance
  * Support vector regression   - RBF kernel, smooth non-linear fit
  * Gradient boosting           - sequential trees (sklearn HistGradientBoosting)
  * LightGBM                    - leaf-wise gradient-boosted trees (lightgbm library)

`log_target=True` trains on log(volatility): vol is right-skewed and its
errors scale with its level, so the log scale is closer to the
homoscedastic-error assumption of squared-error training. Predictions are
always converted back to volatility % before scoring. Because exp(E[log y])
under-estimates E[y] (Jensen's inequality), back-transformed predictions are
multiplied by Duan's smearing factor mean(y / exp(fitted log y)), estimated
on the training data only.
=============================================================================
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from src.config import CONFIG

SEED = CONFIG.SEED


# ----------------------------------------------------------------------------
# Baselines
# ----------------------------------------------------------------------------
class HistoricalAverage:
    name = "Historical average"

    def fit(self, df: pd.DataFrame, y: np.ndarray):
        s = pd.Series(np.asarray(y), index=df.index)
        self.means_ = s.groupby(df["asset"].values).mean().to_dict()
        self.global_ = float(np.mean(y))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return df["asset"].map(self.means_).fillna(self.global_).to_numpy(dtype=float)


class ColumnBaseline:
    def __init__(self, name: str, column: str):
        self.name, self.column = name, column

    def fit(self, df, y):
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return df[self.column].to_numpy(dtype=float)


def make_baselines() -> List[Any]:
    return [
        HistoricalAverage(),
        ColumnBaseline("Recent 20-day vol", "vol_20d"),
        ColumnBaseline("EWMA vol (RiskMetrics)", "ewma_vol"),
    ]


PRIMARY_BASELINE = "Recent 20-day vol"   # the baseline named in our proposal


# ----------------------------------------------------------------------------
# Candidate families
# ----------------------------------------------------------------------------
def _estimator(family: str, p: Dict[str, Any]):
    if family == "Ridge":
        return Ridge(alpha=p["alpha"])
    if family == "Decision tree":
        return DecisionTreeRegressor(max_depth=p["max_depth"], min_samples_leaf=p["min_samples_leaf"], random_state=SEED)
    if family == "Random forest":
        return RandomForestRegressor(
            n_estimators=p.get("n_estimators", 300), max_depth=p["max_depth"],
            min_samples_leaf=p["min_samples_leaf"], max_features=p["max_features"],
            n_jobs=-1, random_state=SEED,
        )
    if family == "SVR (RBF)":
        return SVR(kernel="rbf", C=p["C"], epsilon=p["epsilon"], gamma=p.get("gamma", "scale"))
    if family == "Gradient boosting":
        return HistGradientBoostingRegressor(
            learning_rate=p["learning_rate"], max_iter=p["max_iter"], max_depth=p["max_depth"],
            min_samples_leaf=p["min_samples_leaf"], l2_regularization=p.get("l2", 0.0),
            early_stopping=False, random_state=SEED,
        )
    if family == "LightGBM":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(
            n_estimators=p["n_estimators"], learning_rate=p["learning_rate"], num_leaves=p["num_leaves"],
            min_child_samples=p["min_child_samples"], subsample=p.get("subsample", 0.8), subsample_freq=1,
            colsample_bytree=p.get("colsample_bytree", 0.8), reg_lambda=p.get("reg_lambda", 1.0),
            random_state=SEED, n_jobs=-1, verbose=-1,
        )
    raise ValueError(family)


class LogTargetRegressor(RegressorMixin, BaseEstimator):
    """Fit on log(y); predict exp(f(x)) * smearing factor."""

    def __init__(self, regressor=None):
        self.regressor = regressor

    def fit(self, X, y):
        y = np.asarray(y, float)
        self.regressor_ = clone(self.regressor).fit(X, np.log(y))
        self.smearing_ = float(np.mean(y / np.exp(self.regressor_.predict(X))))
        return self

    def predict(self, X):
        return np.exp(self.regressor_.predict(X)) * self.smearing_


def make_model(family: str, params: Dict[str, Any]):
    pipe = Pipeline([("scale", StandardScaler()), ("model", _estimator(family, params))])
    if params.get("log_target", True):
        return LogTargetRegressor(pipe)
    return pipe


# Stage 1 (screening): a small, sensible grid per family, scored by
# expanding-window cross-validation inside the training period.
SCREEN_GRIDS: Dict[str, List[Dict[str, Any]]] = {
    "Ridge": [{"alpha": a, "log_target": lt} for a in (1.0, 10.0) for lt in (True, False)],
    "Decision tree": [{"max_depth": d, "min_samples_leaf": 50, "log_target": True} for d in (4, 6)],
    "Random forest": [{"max_depth": d, "min_samples_leaf": 20, "max_features": 0.5, "n_estimators": 200, "log_target": True} for d in (6, None)],
    "SVR (RBF)": [{"C": c, "epsilon": 0.05, "log_target": True} for c in (1.0, 10.0)],
    "Gradient boosting": [{"learning_rate": 0.05, "max_iter": 200, "max_depth": d, "min_samples_leaf": 40, "log_target": True} for d in (3, None)],
    "LightGBM": [{"n_estimators": 300, "learning_rate": 0.03, "num_leaves": nl, "min_child_samples": 50, "log_target": True} for nl in (7, 31)],
}

# Stage 2 (tuning of the selected family only): finer grid, each config is
# trained on the training split and scored on the validation split.
TUNE_GRIDS: Dict[str, List[Dict[str, Any]]] = {
    "Ridge": [{"alpha": a, "log_target": lt} for a in (0.001, 0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0) for lt in (True, False)],
    "Decision tree": [{"max_depth": d, "min_samples_leaf": m, "log_target": lt}
                      for d in (3, 4, 5, 6, 8, 10) for m in (20, 50, 100, 200) for lt in (True, False)],
    "Random forest": [{"max_depth": d, "min_samples_leaf": m, "max_features": mf, "n_estimators": 300, "log_target": True}
                      for d in (4, 6, 8, 12, None) for m in (5, 20, 50) for mf in (0.33, 0.5, 1.0)],
    "SVR (RBF)": [{"C": c, "epsilon": e, "gamma": g, "log_target": True}
                  for c in (0.3, 1.0, 3.0, 10.0, 30.0) for e in (0.02, 0.05, 0.1) for g in ("scale", 0.01, 0.05)],
    "Gradient boosting": [{"learning_rate": lr, "max_iter": it, "max_depth": d, "min_samples_leaf": m, "l2": 1.0, "log_target": True}
                          for lr in (0.02, 0.05, 0.1) for it in (100, 300) for d in (2, 3, 5) for m in (20, 80)],
    "LightGBM": [{"n_estimators": n, "learning_rate": lr, "num_leaves": nl, "min_child_samples": m, "reg_lambda": 1.0, "log_target": True}
                 for n in (200, 500) for lr in (0.01, 0.03, 0.1) for nl in (7, 15, 31) for m in (20, 100)],
}

FAMILIES = list(SCREEN_GRIDS)
