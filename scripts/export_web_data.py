"""
scripts/export_web_data.py
=============================================================================
Export Machine Learning Model Evaluations, 22-Feature Catalog, & Test Timeline
Course: ES 335 - Machine Learning (Undergraduate Track)
Team: Siddharth Dangi (25110300) & Aryan Vatsa Chauhan (25110059)
=============================================================================
Serializes:
  1. Complete metadata for 22 engineered features matching user specification
  2. Model scorecards: Linear Regression, Decision Tree, SVM, Logistic Regression, LightGBM
  3. Comprehensive error analysis (Residual histograms, Regime error breakdown, Confusion matrices, ROC curves)
  4. Feature importances across all parametric and non-parametric models
  5. 459-day out-of-sample test timeline with actuals, predictions, and errors
  6. Interactive presets for live in-browser risk prediction lab
Saves directly into `docs/data/portfolio_data.json` for zero-latency web execution.
"""

import sys
import json
import logging
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import joblib

from src.config import CONFIG, DOCS_DATA_DIR, ARTIFACTS_DIR
from src.data_loader import load_raw_market_data
from src.feature_engineering import build_ml_feature_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)


FEATURE_DEFINITIONS = {
    "ret": {
        "name": "Daily Log Return %",
        "group": "Price Action & Momentum",
        "formula": r"\ln(P_t / P_{t-1}) \times 100",
        "description": "Logarithmic price return of the primary asset over the most recent 1-day period.",
    },
    "ret_lag_1": {
        "name": "Return Lag 1",
        "group": "Price Action & Momentum",
        "formula": r"r_{t-1}",
        "description": "Log return from 1 day prior, capturing short-term mean-reversion or momentum.",
    },
    "ret_lag_2": {
        "name": "Return Lag 2",
        "group": "Price Action & Momentum",
        "formula": r"r_{t-2}",
        "description": "Log return from 2 days prior.",
    },
    "ret_lag_3": {
        "name": "Return Lag 3",
        "group": "Price Action & Momentum",
        "formula": r"r_{t-3}",
        "description": "Log return from 3 days prior.",
    },
    "ret_lag_4": {
        "name": "Return Lag 4",
        "group": "Price Action & Momentum",
        "formula": r"r_{t-4}",
        "description": "Log return from 4 days prior.",
    },
    "ret_lag_5": {
        "name": "Return Lag 5",
        "group": "Price Action & Momentum",
        "formula": r"r_{t-5}",
        "description": "Log return from 5 days prior (1 full trading week lag).",
    },
    "ret_21d": {
        "name": "21-Day Cumulative Return %",
        "group": "Price Action & Momentum",
        "formula": r"(P_t / P_{t-21} - 1) \times 100",
        "description": "1-month cumulative percentage price performance.",
    },
    "drawdown_60d": {
        "name": "60-Day Peak Drawdown %",
        "group": "Price Action & Momentum",
        "formula": r"(P_t / \max_{60}(P) - 1) \times 100",
        "description": "Percentage decline from the rolling 60-day price peak.",
    },
    "vol_5d": {
        "name": "5-Day Realized Volatility %",
        "group": "Volatility Dynamics",
        "formula": r"\sigma_{5d} \times \sqrt{252} \times 100",
        "description": "Annualized standard deviation of daily returns over a 1-week rolling window.",
    },
    "vol_20d": {
        "name": "20-Day Realized Volatility %",
        "group": "Volatility Dynamics",
        "formula": r"\sigma_{20d} \times \sqrt{252} \times 100",
        "description": "Annualized standard deviation of daily returns over a 1-month rolling window.",
    },
    "vol_60d": {
        "name": "60-Day Realized Volatility %",
        "group": "Volatility Dynamics",
        "formula": r"\sigma_{60d} \times \sqrt{252} \times 100",
        "description": "Annualized standard deviation of daily returns over a 1-quarter rolling window.",
    },
    "vol_ratio": {
        "name": "Short/Long Volatility Ratio",
        "group": "Volatility Dynamics",
        "formula": r"\sigma_{5d} / \sigma_{60d}",
        "description": "Term structure ratio of short-term (5d) to quarterly (60d) volatility. Values > 1 indicate volatility shocks.",
    },
    "ewma_vol": {
        "name": "EWMA Volatility % (RiskMetrics)",
        "group": "Volatility Dynamics",
        "formula": r"\sqrt{252 \times \text{EWMA}(r_t^2, \alpha=0.06)} \times 100",
        "description": "Exponentially weighted moving average volatility placing heavier weight on recent trading shocks.",
    },
    "parkinson_vol": {
        "name": "Parkinson High-Low Volatility %",
        "group": "High-Frequency Volatility",
        "formula": r"\sqrt{ \frac{\ln(H_t/L_t)^2}{4 \ln 2} \times 252 } \times 100",
        "description": "Extreme-value volatility estimator utilizing intraday High and Low prices (5x more efficient than Close-to-Close).",
    },
    "parkinson_vol_10d": {
        "name": "10-Day Parkinson Volatility %",
        "group": "High-Frequency Volatility",
        "formula": r"\text{SMA}_{10d}(\text{Parkinson Vol})",
        "description": "10-day moving average of intraday Parkinson range volatility.",
    },
    "skew_30d": {
        "name": "30-Day Rolling Skewness",
        "group": "Distribution Geometry",
        "formula": r"\mathbb{E}[(r - \mu)^3] / \sigma^3",
        "description": "Third standardized moment; measures asymmetric tail risk. Negative values signal fat left downside tails.",
    },
    "kurt_30d": {
        "name": "30-Day Rolling Kurtosis",
        "group": "Distribution Geometry",
        "formula": r"\mathbb{E}[(r - \mu)^4] / \sigma^4 - 3",
        "description": "Excess kurtosis measuring outlier tail probability relative to a Gaussian bell curve.",
    },
    "hist_var_95_60d": {
        "name": "60-Day 95% Parametric VaR %",
        "group": "Tail Risk Measures",
        "formula": r"-1.645 \times \frac{\sigma_{60d}}{\sqrt{252}} \times 100",
        "description": "Baseline Gaussian 1-day 95% Value-at-Risk benchmark.",
    },
    "beta_60d": {
        "name": "60-Day Benchmark Beta",
        "group": "Market Regime & Macro",
        "formula": r"\text{Cov}(r_t, r_{\text{bm}, t}) / \text{Var}(r_{\text{bm}, t})",
        "description": "Rolling 60-day systematic market sensitivity relative to benchmark (QQQ).",
    },
    "vix_level": {
        "name": "CBOE VIX Index Level",
        "group": "Market Regime & Macro",
        "formula": r"\text{VIX Close}",
        "description": "Market-implied volatility fear gauge derived from S&P 500 index options.",
    },
    "vix_change_5d": {
        "name": "5-Day VIX Velocity %",
        "group": "Market Regime & Macro",
        "formula": r"(VIX_t / VIX_{t-5} - 1) \times 100",
        "description": "5-day percentage change in the VIX index, measuring speed of market panic onset.",
    },
    "rel_volume_20d": {
        "name": "Relative Volume vs 20d Average",
        "group": "Market Regime & Macro",
        "formula": r"\text{Vol}_t / \text{SMA}_{20d}(\text{Vol})",
        "description": "Ratio of current trading volume to its 20-day moving average. Surges > 1.5 indicate institutional participation.",
    },
}


def export_for_github_pages():
    """Generates docs/data/portfolio_data.json containing all ML results and interactive datasets."""
    logger.info("Generating Machine Learning web data payload...")

    # 1. Load engineered dataset
    raw_data = load_raw_market_data(use_cache=True)
    df, feature_cols = build_ml_feature_dataset(raw_data)

    # 2. Load trained models & scaler
    scaler = joblib.load(ARTIFACTS_DIR / "feature_scaler.joblib")
    lr_model = joblib.load(ARTIFACTS_DIR / "best_linear_regression.joblib")
    dt_model = joblib.load(ARTIFACTS_DIR / "best_decision_tree.joblib")
    svm_model = joblib.load(ARTIFACTS_DIR / "best_svm_model.joblib")
    log_model = joblib.load(ARTIFACTS_DIR / "best_logistic_regression.joblib")
    lgb_model = joblib.load(ARTIFACTS_DIR / "best_lightgbm_model.joblib")

    # Load precomputed evaluation metrics
    with open(ARTIFACTS_DIR / "evaluation_metrics.json", "r") as f:
        eval_metrics = json.load(f)

    # 3. Test set slice (2023-01 to 2024-10)
    test_mask = df.index >= CONFIG.TEST_START
    test_df = df.loc[test_mask].copy()

    X_test_raw = test_df[feature_cols].values
    X_test_scaled = scaler.transform(X_test_raw)
    y_reg_test = test_df["target_next_ret"].values
    y_cls_test = test_df["target_high_risk"].values
    test_dates = [d.strftime("%Y-%m-%d") for d in test_df.index]

    # Predict test set
    p_lr_reg = lr_model.predict(X_test_scaled)
    p_dt_reg, p_dt_cls, p_dt_prob = dt_model.predict(X_test_scaled)
    p_svm_reg, p_svm_cls, p_svm_prob = svm_model.predict(X_test_scaled)
    p_log_cls, p_log_prob = log_model.predict(X_test_scaled)
    p_lgb_reg, p_lgb_cls, p_lgb_prob = lgb_model.predict(X_test_scaled)

    # 4. Feature Metadata with Descriptive Statistics
    feature_catalog = {}
    train_mask = df.index < CONFIG.VAL_START
    train_df = df.loc[train_mask]

    for col in feature_cols:
        meta = FEATURE_DEFINITIONS.get(col, {"name": col, "group": "Engineered", "formula": col, "description": col})
        vals = train_df[col].dropna()
        feature_catalog[col] = {
            "name": meta["name"],
            "group": meta["group"],
            "formula": meta["formula"],
            "description": meta["description"],
            "mean": round(float(vals.mean()), 4),
            "std": round(float(vals.std()), 4),
            "min": round(float(vals.min()), 4),
            "p25": round(float(vals.quantile(0.25)), 4),
            "p50": round(float(vals.median()), 4),
            "p75": round(float(vals.quantile(0.75)), 4),
            "max": round(float(vals.max()), 4),
        }

    # 5. Build Test Set Timeline (Day-by-Day for Replay & Error Analysis)
    test_timeline = []
    for i in range(len(test_df)):
        date_str = test_dates[i]
        actual_ret = round(float(y_reg_test[i]), 2)
        actual_risk = int(y_cls_test[i])
        vix = round(float(test_df["vix_level"].iloc[i]), 2)

        feat_dict = {c: round(float(test_df[c].iloc[i]), 3) for c in feature_cols}

        test_timeline.append({
            "date": date_str,
            "actual_ret": actual_ret,
            "actual_high_risk": actual_risk,
            "vix": vix,
            "features": feat_dict,
            "preds": {
                "linear_regression": {
                    "pred_ret": round(float(p_lr_reg[i]), 2),
                    "residual": round(float(actual_ret - p_lr_reg[i]), 2),
                },
                "decision_tree": {
                    "pred_ret": round(float(p_dt_reg[i]), 2),
                    "residual": round(float(actual_ret - p_dt_reg[i]), 2),
                    "risk_prob": round(float(p_dt_prob[i]), 3),
                    "pred_risk": int(p_dt_cls[i]),
                },
                "svm": {
                    "pred_ret": round(float(p_svm_reg[i]), 2),
                    "residual": round(float(actual_ret - p_svm_reg[i]), 2),
                    "risk_prob": round(float(p_svm_prob[i]), 3),
                    "pred_risk": int(p_svm_cls[i]),
                },
                "logistic_regression": {
                    "risk_prob": round(float(p_log_prob[i]), 3),
                    "pred_risk": int(p_log_cls[i]),
                },
                "lightgbm": {
                    "pred_ret": round(float(p_lgb_reg[i]), 2),
                    "residual": round(float(actual_ret - p_lgb_reg[i]), 2),
                    "risk_prob": round(float(p_lgb_prob[i]), 3),
                    "pred_risk": int(p_lgb_cls[i]),
                },
            }
        })

    # 6. Interactive Presets for the Live Prediction Lab
    # Select distinct market states from test set
    vix_sorted_indices = np.argsort(test_df["vix_level"].values)
    calm_idx = int(vix_sorted_indices[5])          # Low VIX
    panic_idx = int(vix_sorted_indices[-5])        # High VIX spike
    rebound_idx = int(np.argmin(y_reg_test))       # Worst drop day
    rally_idx = int(np.argmax(y_reg_test))         # Best rally day
    median_idx = int(vix_sorted_indices[len(vix_sorted_indices) // 2])

    presets = [
        {
            "id": "calm_bull",
            "name": f"Calm Bull Day ({test_dates[calm_idx]})",
            "description": "Low VIX fear (< 13.5), muted volatility ratio, steady upward price momentum.",
            "date": test_dates[calm_idx],
            "features": {c: round(float(test_df[c].iloc[calm_idx]), 3) for c in feature_cols},
        },
        {
            "id": "vix_panic",
            "name": f"Volatility Panic Spike ({test_dates[panic_idx]})",
            "description": "Sharp VIX escalation (> 25), high short-to-long volatility ratio (> 1.4), surging volume.",
            "date": test_dates[panic_idx],
            "features": {c: round(float(test_df[c].iloc[panic_idx]), 3) for c in feature_cols},
        },
        {
            "id": "tail_drop",
            "name": f"Tail Crash Reversal ({test_dates[rebound_idx]})",
            "description": "Steep single-day selloff triggering 5% tail loss threshold, heavy negative skewness.",
            "date": test_dates[rebound_idx],
            "features": {c: round(float(test_df[c].iloc[rebound_idx]), 3) for c in feature_cols},
        },
        {
            "id": "momentum_rally",
            "name": f"Strong Momentum Rally ({test_dates[rally_idx]})",
            "description": "High return momentum across multiple lags, positive skew, expanding trading volume.",
            "date": test_dates[rally_idx],
            "features": {c: round(float(test_df[c].iloc[rally_idx]), 3) for c in feature_cols},
        },
        {
            "id": "median_day",
            "name": f"Typical / Median Trading Day ({test_dates[median_idx]})",
            "description": "Median VIX level (~16), moderate Parkinson high-low spread, neutral return momentum.",
            "date": test_dates[median_idx],
            "features": {c: round(float(test_df[c].iloc[median_idx]), 3) for c in feature_cols},
        },
    ]

    # Default median values for feature sliders
    default_slider_values = {c: round(float(train_df[c].median()), 3) for c in feature_cols}

    # Scaler parameters for in-browser standardized prediction calculations
    scaler_params = {
        "mean": {c: round(float(scaler.mean_[i]), 5) for i, c in enumerate(feature_cols)},
        "scale": {c: round(float(scaler.scale_[i]), 5) for i, c in enumerate(feature_cols)},
    }

    # Model parameters for client-side direct inference (Linear & Logistic Regression)
    linear_reg_weights = lr_model.get_coefficients(feature_cols)
    linear_reg_intercept = round(float(lr_model.regressor.intercept_), 4)

    logistic_weights = log_model.get_coefficients(feature_cols)
    logistic_intercept = round(float(log_model.classifier.intercept_[0]), 4)

    # 7. Assemble Master JSON Payload
    web_payload = {
        "metadata": {
            "title": "Next-Day Risk & Return Machine Learning Platform",
            "course": "ES 335: Machine Learning (Undergraduate Track)",
            "team": "Siddharth Dangi (25110300) & Aryan Vatsa Chauhan (25110059)",
            "horizon": "1 Day (Next-Day Log Return & Tail Risk)",
            "date_range": "2012-01-01 to 2024-10-31 (3,167 Clean Trading Days)",
            "split_info": {
                "train": {"samples": int((df.index < CONFIG.VAL_START).sum()), "period": "2012-01-01 to 2021-12-31"},
                "val": {"samples": int(((df.index >= CONFIG.VAL_START) & (df.index < CONFIG.TEST_START)).sum()), "period": "2022-01-01 to 2022-12-31"},
                "test": {"samples": int((df.index >= CONFIG.TEST_START).sum()), "period": "2023-01-01 to 2024-10-31"},
            },
            "num_features": len(feature_cols),
            "target_threshold_q05": -1.60,
        },
        "feature_cols": feature_cols,
        "feature_catalog": feature_catalog,
        "eval_metrics": eval_metrics,
        "scaler_params": scaler_params,
        "client_models": {
            "linear_regression": {
                "weights": linear_reg_weights,
                "intercept": linear_reg_intercept,
            },
            "logistic_regression": {
                "weights": logistic_weights,
                "intercept": logistic_intercept,
            },
        },
        "presets": presets,
        "default_slider_values": default_slider_values,
        "test_timeline": test_timeline,
    }

    # Save to docs/data/portfolio_data.json
    DOCS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DOCS_DATA_DIR / "portfolio_data.json"
    with open(out_path, "w") as f:
        json.dump(web_payload, f, indent=2)
    logger.info(f"Successfully serialized web data payload to {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

    # Also save to artifacts/
    with open(ARTIFACTS_DIR / "web_portfolio_data.json", "w") as f:
        json.dump(web_payload, f, indent=2)


if __name__ == "__main__":
    export_for_github_pages()
