"""
src/train_pipeline.py
=============================================================================
Machine Learning Training, Validation & Benchmarking Pipeline
Course: ES 335 - Machine Learning
=============================================================================
Orchestrates:
  1. Data ingestion & 22-feature matrix engineering.
  2. Chronological Walk-Forward train/val/test splits (2012-21, 2022, 2023-24).
  3. Feature scaling via StandardScaler.
  4. Model fitting for:
     - Linear Regression (Ridge)
     - Decision Tree (CART)
     - Support Vector Machine (SVR / SVC RBF)
     - Logistic Regression Classifier
     - LightGBM Gradient Boosted Trees
  5. Multi-metric evaluation (RMSE, MAE, R2, Accuracy, F1, ROC-AUC, Confusion Matrix).
  6. Residual error distributions and volatility regime error analysis.
  7. Artifact serialization.
"""

import logging
import json
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

from src.config import CONFIG, ARTIFACTS_DIR
from src.data_loader import load_raw_market_data
from src.feature_engineering import build_ml_feature_dataset, FEATURE_METADATA
from src.models.ml_models import (
    LinearRegressionRiskModel,
    DecisionTreeRiskModel,
    SVMRiskModel,
    LogisticRiskClassifier,
    LightGBMRiskModel
)
from src.evaluation import (
    compute_regression_metrics,
    compute_classification_metrics,
    compute_regime_error_breakdown
)

logger = logging.getLogger(__name__)


def run_full_pipeline() -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Executes the complete machine learning pipeline and returns performance results."""
    logger.info("=" * 70)
    logger.info("EXECUTING 1-DAY RISK PREDICTION MACHINE LEARNING PIPELINE")
    logger.info("Models: Linear Regression, Decision Tree, SVM, Logistic Regression, LightGBM")
    logger.info("=" * 70)

    # 1. Ingest raw data & build 22 features
    raw_data = load_raw_market_data(use_cache=True)
    df, feature_cols = build_ml_feature_dataset(raw_data)

    # 2. Chronological Splits
    train_mask = df.index < "2022-01-01"
    val_mask = (df.index >= "2022-01-01") & (df.index < "2023-01-01")
    test_mask = df.index >= "2023-01-01"

    X_train_raw = df.loc[train_mask, feature_cols].values
    X_val_raw = df.loc[val_mask, feature_cols].values
    X_test_raw = df.loc[test_mask, feature_cols].values

    # Target values (Returns in %, High Risk in 0/1)
    y_reg_train = df.loc[train_mask, "target_next_ret"].values
    y_reg_val = df.loc[val_mask, "target_next_ret"].values
    y_reg_test = df.loc[test_mask, "target_next_ret"].values

    y_cls_train = df.loc[train_mask, "target_high_risk"].values
    y_cls_val = df.loc[val_mask, "target_high_risk"].values
    y_cls_test = df.loc[test_mask, "target_high_risk"].values

    vix_test = df.loc[test_mask, "vix_level"].values
    test_dates = [d.strftime("%Y-%m-%d") for d in df.loc[test_mask].index]

    logger.info(f"Splits -> Train: {len(X_train_raw)}, Val: {len(X_val_raw)}, Test: {len(X_test_raw)}")

    # 3. Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    scaler_path = ARTIFACTS_DIR / "feature_scaler.joblib"
    joblib.dump(scaler, scaler_path)
    logger.info(f"Saved feature scaler to: {scaler_path}")

    # 4. Model 1: Linear Regression (Ridge)
    logger.info("Training Model 1: Linear Regression (Ridge)...")
    lr_model = LinearRegressionRiskModel(alpha=1.0)
    lr_model.fit(X_train, y_reg_train)
    p_lr_reg = lr_model.predict(X_test)
    joblib.dump(lr_model, ARTIFACTS_DIR / "best_linear_regression.joblib")

    # 5. Model 2: Decision Tree (CART)
    logger.info("Training Model 2: Decision Tree Regressor & Classifier...")
    dt_model = DecisionTreeRiskModel(max_depth=4, min_samples_leaf=20)
    dt_model.fit(X_train, y_reg_train, y_cls_train)
    p_dt_reg, p_dt_cls, p_dt_prob = dt_model.predict(X_test)
    joblib.dump(dt_model, ARTIFACTS_DIR / "best_decision_tree.joblib")

    # 6. Model 3: Support Vector Machine (SVR & SVC with RBF)
    logger.info("Training Model 3: Support Vector Machine (RBF kernel)...")
    svm_model = SVMRiskModel(C=1.0, epsilon=0.1)
    svm_model.fit(X_train, y_reg_train, y_cls_train)
    p_svm_reg, p_svm_cls, p_svm_prob = svm_model.predict(X_test)
    joblib.dump(svm_model, ARTIFACTS_DIR / "best_svm_model.joblib")

    # 7. Model 4: Logistic Regression Classifier
    logger.info("Training Model 4: Logistic Regression Classifier...")
    log_model = LogisticRiskClassifier(C=1.0)
    log_model.fit(X_train, y_cls_train)
    p_log_cls, p_log_prob = log_model.predict(X_test)
    joblib.dump(log_model, ARTIFACTS_DIR / "best_logistic_regression.joblib")

    # 8. Model 5: LightGBM Gradient Boosted Trees
    logger.info("Training Model 5: LightGBM Gradient Boosted Trees...")
    lgbm_model = LightGBMRiskModel(n_estimators=100, learning_rate=0.03, max_depth=4)
    lgbm_model.fit(X_train, y_reg_train, y_cls_train)
    p_lgb_reg, p_lgb_cls, p_lgb_prob = lgbm_model.predict(X_test)
    joblib.dump(lgbm_model, ARTIFACTS_DIR / "best_lightgbm_model.joblib")

    # 9. Evaluate All Models
    logger.info("Computing evaluation metrics and error distributions...")

    results = {
        "Linear Regression": {
            "type": "Regression",
            "category": "Linear Parametric Model",
            "regression_metrics": compute_regression_metrics(y_reg_test, p_lr_reg),
            "regime_errors": compute_regime_error_breakdown(y_reg_test, p_lr_reg, vix_test),
            "coefficients": lr_model.get_coefficients(feature_cols),
            "predictions_sample": [round(float(p), 2) for p in p_lr_reg[:50]],
        },
        "Decision Tree": {
            "type": "Dual (Regression + Classification)",
            "category": "Non-Parametric Partitioning",
            "regression_metrics": compute_regression_metrics(y_reg_test, p_dt_reg),
            "classification_metrics": compute_classification_metrics(y_cls_test, p_dt_cls, p_dt_prob),
            "regime_errors": compute_regime_error_breakdown(y_reg_test, p_dt_reg, vix_test),
            "feature_importances": dt_model.get_feature_importances(feature_cols),
            "predictions_sample": [round(float(p), 2) for p in p_dt_reg[:50]],
        },
        "Support Vector Machine": {
            "type": "Dual (SVR + SVC RBF)",
            "category": "Kernel Margin Optimization",
            "regression_metrics": compute_regression_metrics(y_reg_test, p_svm_reg),
            "classification_metrics": compute_classification_metrics(y_cls_test, p_svm_cls, p_svm_prob),
            "regime_errors": compute_regime_error_breakdown(y_reg_test, p_svm_reg, vix_test),
            "predictions_sample": [round(float(p), 2) for p in p_svm_reg[:50]],
        },
        "Logistic Regression": {
            "type": "Classification",
            "category": "Probabilistic Classifier",
            "classification_metrics": compute_classification_metrics(y_cls_test, p_log_cls, p_log_prob),
            "coefficients": log_model.get_coefficients(feature_cols),
            "predicted_probabilities_sample": [round(float(p), 3) for p in p_log_prob[:50]],
        },
        "LightGBM Ensembles": {
            "type": "Dual (Regression + Classification)",
            "category": "Gradient Boosted Decision Trees",
            "regression_metrics": compute_regression_metrics(y_reg_test, p_lgb_reg),
            "classification_metrics": compute_classification_metrics(y_cls_test, p_lgb_cls, p_lgb_prob),
            "regime_errors": compute_regime_error_breakdown(y_reg_test, p_lgb_reg, vix_test),
            "feature_importances": lgbm_model.get_feature_importances(feature_cols),
            "predictions_sample": [round(float(p), 2) for p in p_lgb_reg[:50]],
        },
    }

    # Save metrics JSON to artifacts
    metrics_path = ARTIFACTS_DIR / "evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Evaluation metrics written to: {metrics_path}")

    return results, df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
    run_full_pipeline()
