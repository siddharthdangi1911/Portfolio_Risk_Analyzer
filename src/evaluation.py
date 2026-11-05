"""
src/evaluation.py
=============================================================================
Machine Learning Evaluation & Error Analysis Framework
Course: ES 335 - Machine Learning
=============================================================================
Standardized evaluation metrics for 1-day risk prediction:
  1. Regression Metrics: MSE, RMSE, MAE, R^2, Max Error, Residual Statistics
  2. Classification Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
  3. Diagnostic Error Analysis: Confusion Matrix, Residual Error Distribution, Regime Error Breakdown
"""

import logging
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    max_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
    confusion_matrix,
)

logger = logging.getLogger(__name__)


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Computes standard regression performance metrics and residual statistics.
    All outputs in percentages (%).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    residuals = y_true - y_pred
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    max_err = float(max_error(y_true, y_pred))
    mean_res = float(np.mean(residuals))
    std_res = float(np.std(residuals))

    # Compute 15-bin residual histogram for web visualization
    counts, bin_edges = np.histogram(residuals, bins=15)
    hist_data = [
        {"bin": round(float((bin_edges[i] + bin_edges[i + 1]) / 2), 2), "count": int(counts[i])}
        for i in range(len(counts))
    ]

    return {
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "R2": round(r2, 4),
        "Max_Error": round(max_err, 4),
        "Mean_Residual": round(mean_res, 4),
        "Std_Residual": round(std_res, 4),
        "Residual_Histogram": hist_data,
    }


def compute_classification_metrics(y_true: np.ndarray, y_pred_cls: np.ndarray, y_pred_prob: np.ndarray) -> Dict[str, Any]:
    """
    Computes binary risk classification metrics and confusion matrix.
    Positive Class (1) = High-Risk Tail Event (Next Day Drop <= -1.60%)
    Negative Class (0) = Normal / Safe Trading Day
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred_cls = np.asarray(y_pred_cls, dtype=int)
    y_pred_prob = np.asarray(y_pred_prob, dtype=np.float64)

    acc = float(accuracy_score(y_true, y_pred_cls))
    prec = float(precision_score(y_true, y_pred_cls, zero_division=0))
    rec = float(recall_score(y_true, y_pred_cls, zero_division=0))
    f1 = float(f1_score(y_true, y_pred_cls, zero_division=0))

    try:
        auc = float(roc_auc_score(y_true, y_pred_prob))
    except Exception:
        auc = 0.50

    cm = confusion_matrix(y_true, y_pred_cls)
    # Format: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])) if cm.shape == (2, 2) else (len(y_true), 0, 0, 0)

    # ROC curve points
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
        # Sample down to 20 points for smooth web rendering
        step = max(1, len(fpr) // 20)
        roc_pts = [{"x": round(float(fpr[i]), 4), "y": round(float(tpr[i]), 4)} for i in range(0, len(fpr), step)]
        if roc_pts[-1]["x"] != 1.0:
            roc_pts.append({"x": 1.0, "y": 1.0})
    except Exception:
        roc_pts = [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}]

    return {
        "Accuracy": round(acc * 100.0, 2),
        "Precision": round(prec * 100.0, 2),
        "Recall": round(rec * 100.0, 2),
        "F1_Score": round(f1, 4),
        "ROC_AUC": round(auc, 4),
        "Confusion_Matrix": {
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
            "Total": len(y_true)
        },
        "ROC_Curve": roc_pts
    }


def compute_regime_error_breakdown(y_true: np.ndarray, y_pred: np.ndarray, vix_levels: np.ndarray) -> Dict[str, Any]:
    """
    Evaluates MAE and RMSE across market volatility regimes:
      - Low Volatility (VIX < 16)
      - Medium Volatility (16 <= VIX <= 22)
      - High Volatility / Crisis (VIX > 22)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    vix = np.asarray(vix_levels)

    regimes = {
        "Low_Vol_VIX_under_16": vix < 16.0,
        "Med_Vol_VIX_16_to_22": (vix >= 16.0) & (vix <= 22.0),
        "High_Vol_VIX_over_22": vix > 22.0,
    }

    breakdown = {}
    for name, mask in regimes.items():
        if np.sum(mask) > 0:
            sub_true = y_true[mask]
            sub_pred = y_pred[mask]
            breakdown[name] = {
                "count": int(np.sum(mask)),
                "MAE": round(float(mean_absolute_error(sub_true, sub_pred)), 4),
                "RMSE": round(float(np.sqrt(mean_squared_error(sub_true, sub_pred))), 4),
            }
        else:
            breakdown[name] = {"count": 0, "MAE": 0.0, "RMSE": 0.0}

    return breakdown
