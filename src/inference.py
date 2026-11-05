"""
src/inference.py
=============================================================================
Easy, Human-Centric Machine Learning Risk & Return Inference Engine
Course: ES 335 - Machine Learning (Undergraduate Track)
Team: Siddharth Dangi (25110300) & Aryan Vatsa Chauhan (25110059)
=============================================================================
Designed so that ANY HUMAN without a finance background can input simple market
observations (e.g., today's return %, market fear/VIX, volume) and receive:
  1. Predicted Next-Day Return (%) & Trend Direction
  2. Crash Risk Level (🟢 Low / 🟡 Moderate / 🔴 High Tail Risk)
  3. Probability of a Catastrophic Market Drop (≤ -1.60%)
  4. Plain-English AI Explanation of WHY the models made this prediction
  5. Consensus Breakdown across all 5 Core ML Models
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import joblib

from src.config import CONFIG, ARTIFACTS_DIR

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)


# Standard baseline mean feature vector computed from 10-year training data
DEFAULT_FEATURE_VECTOR = {
    "ret": 0.0005,
    "ret_lag_1": 0.0005,
    "ret_lag_2": 0.0005,
    "ret_lag_3": 0.0005,
    "ret_lag_4": 0.0005,
    "ret_lag_5": 0.0005,
    "ret_21d": 1.1279,
    "drawdown_60d": -2.3974,
    "vol_5d": 12.5651,
    "vol_20d": 13.4587,
    "vol_60d": 14.0661,
    "vol_ratio": 0.9108,
    "ewma_vol": 13.8722,
    "parkinson_vol": 9.7332,
    "parkinson_vol_10d": 9.7338,
    "skew_30d": -0.2743,
    "kurt_30d": 0.9736,
    "hist_var_95_60d": -1.3652,
    "beta_60d": 0.7273,
    "vix_level": 17.0500,
    "vix_change_5d": 1.4162,
    "rel_volume_20d": 1.0075,
}

FEATURE_ORDER = [
    "ret", "ret_lag_1", "ret_lag_2", "ret_lag_3", "ret_lag_4", "ret_lag_5",
    "ret_21d", "drawdown_60d", "vol_5d", "vol_20d", "vol_60d", "vol_ratio",
    "ewma_vol", "parkinson_vol", "parkinson_vol_10d", "skew_30d", "kurt_30d",
    "hist_var_95_60d", "beta_60d", "vix_level", "vix_change_5d", "rel_volume_20d"
]


class RiskInferenceEngine:
    """Production inference wrapper that translates simple human inputs into 5 ML predictions."""

    def __init__(self):
        self.scaler = joblib.load(ARTIFACTS_DIR / "feature_scaler.joblib")
        self.lr_model = joblib.load(ARTIFACTS_DIR / "best_linear_regression.joblib")
        self.dt_model = joblib.load(ARTIFACTS_DIR / "best_decision_tree.joblib")
        self.svm_model = joblib.load(ARTIFACTS_DIR / "best_svm_model.joblib")
        self.log_model = joblib.load(ARTIFACTS_DIR / "best_logistic_regression.joblib")
        self.lgb_model = joblib.load(ARTIFACTS_DIR / "best_lightgbm_model.joblib")

    def run_human_inference(
        self,
        today_return_pct: float = 0.0,
        vix_level: float = 16.0,
        volatility_level: str = "normal",   # 'low', 'normal', 'elevated', 'extreme'
        volume_activity: str = "normal",    # 'quiet', 'normal', 'heavy', 'surge'
        recent_market_trend: str = "flat",  # 'strong_up', 'mild_up', 'flat', 'down', 'crash'
    ) -> Dict[str, Any]:
        """
        Takes 5 human-understandable questions and generates an easy-to-read AI report.
        """
        # Start from calibrated median feature vector
        feats = DEFAULT_FEATURE_VECTOR.copy()

        # 1. Map today's return & short lags (ret is decimal log return, e.g. -0.015 for -1.5%)
        feats["ret"] = float(today_return_pct) / 100.0
        feats["ret_lag_1"] = (float(today_return_pct) / 100.0) * 0.7

        # 2. Map VIX fear gauge
        feats["vix_level"] = float(vix_level)

        # 3. Map volatility condition
        vol_map = {
            "low": (8.5, 0.70),
            "normal": (12.5, 0.90),
            "elevated": (22.0, 1.35),
            "extreme": (38.0, 2.10),
        }
        vol_5d, vol_ratio = vol_map.get(volatility_level.lower(), (12.5, 0.90))
        feats["vol_5d"] = vol_5d
        feats["vol_ratio"] = vol_ratio
        feats["parkinson_vol"] = vol_5d * 0.92

        # 4. Map volume activity
        vol_act_map = {
            "quiet": 0.70,
            "normal": 1.00,
            "heavy": 1.45,
            "surge": 2.20,
        }
        feats["rel_volume_20d"] = vol_act_map.get(volume_activity.lower(), 1.00)

        # 5. Map recent market trend
        trend_map = {
            "strong_up": (4.5, -0.5, 0.4, 0.5),
            "mild_up": (1.8, -1.2, 0.1, 0.8),
            "flat": (0.2, -2.0, -0.1, 1.1),
            "down": (-3.5, -6.5, -0.6, 2.4),
            "crash": (-9.0, -14.0, -1.5, 4.8),
        }
        ret_21d, dd60, skew, kurt = trend_map.get(recent_market_trend.lower(), (0.2, -2.0, -0.1, 1.1))
        feats["ret_21d"] = ret_21d
        feats["drawdown_60d"] = dd60
        feats["skew_30d"] = skew
        feats["kurt_30d"] = kurt

        # Vectorize and scale
        vec = np.array([[feats[k] for k in FEATURE_ORDER]], dtype=np.float64)
        vec_scaled = self.scaler.transform(vec)

        # Execute 5 Models
        p_lr_ret = float(self.lr_model.predict(vec_scaled)[0])
        p_dt_ret, p_dt_cls, p_dt_prob = self.dt_model.predict(vec_scaled)
        p_svm_ret, p_svm_cls, p_svm_prob = self.svm_model.predict(vec_scaled)
        p_log_cls, p_log_prob = self.log_model.predict(vec_scaled)
        p_lgb_ret, p_lgb_cls, p_lgb_prob = self.lgb_model.predict(vec_scaled)

        # Consensus metrics
        consensus_return = float(np.median([p_lr_ret, float(p_dt_ret[0]), float(p_svm_ret[0]), float(p_lgb_ret[0])]))
        crash_prob = float(np.median([float(p_log_prob[0]), float(p_lgb_prob[0]), float(p_svm_prob[0])]))

        # Plain-English Risk Rating
        if crash_prob >= 0.50 or vix_level >= 25.0:
            risk_rating = "🔴 HIGH CRASH RISK"
            action_advice = "Market stress detected. High probability of downside volatility tomorrow."
        elif crash_prob >= 0.25 or vix_level >= 19.0:
            risk_rating = "🟡 MODERATE RISK (Caution)"
            action_advice = "Market volatility is elevated. Expect mild to moderate swings."
        else:
            risk_rating = "🟢 LOW RISK (Safe)"
            action_advice = "Market indicators are calm and stable. Normal trading conditions expected."

        # Plain-English Explanation
        explanation = self._build_plain_english_explanation(
            today_return_pct, vix_level, volatility_level, volume_activity, recent_market_trend, crash_prob, consensus_return
        )

        return {
            "consensus_predicted_return": round(consensus_return, 2),
            "predicted_direction": "📈 Expected Upward Move" if consensus_return >= 0.10 else ("📉 Expected Downward Drop" if consensus_return <= -0.10 else "➡️ Flat / Muted Drift"),
            "crash_risk_probability": round(crash_prob * 100.0, 1),
            "risk_rating": risk_rating,
            "action_advice": action_advice,
            "plain_english_explanation": explanation,
            "model_breakdown": {
                "Linear Regression (Ridge)": {
                    "predicted_return": round(p_lr_ret, 2),
                    "role": "Baseline Linear Trend"
                },
                "Decision Tree (CART)": {
                    "predicted_return": round(float(p_dt_ret[0]), 2),
                    "crash_prob": round(float(p_dt_prob[0]) * 100.0, 1),
                    "role": "Rule-Based Regime Splitting"
                },
                "Support Vector Machine (RBF)": {
                    "predicted_return": round(float(p_svm_ret[0]), 2),
                    "crash_prob": round(float(p_svm_prob[0]) * 100.0, 1),
                    "role": "Non-Linear Margin Optimization"
                },
                "Logistic Regression": {
                    "crash_prob": round(float(p_log_prob[0]) * 100.0, 1),
                    "is_crash_alert": bool(p_log_cls[0] == 1),
                    "role": "Calibrated Log-Odds Probability"
                },
                "LightGBM Benchmark": {
                    "predicted_return": round(float(p_lgb_ret[0]), 2),
                    "crash_prob": round(float(p_lgb_prob[0]) * 100.0, 1),
                    "role": "Gradient Boosted Tree Ensemble"
                }
            }
        }

    def _build_plain_english_explanation(self, ret, vix, vol, vol_act, trend, crash_prob, pred_ret):
        reasons = []
        if vix >= 24:
            reasons.append(f"CBOE VIX fear index is elevated at {vix:.1f}, signaling defensive institutional hedging")
        elif vix <= 14:
            reasons.append(f"CBOE VIX fear index is low ({vix:.1f}), showing calm market complacency")

        if ret <= -1.5:
            reasons.append(f"today's steep {ret:.1f}% selloff triggers negative short-term momentum")
        elif ret >= 1.5:
            reasons.append(f"today's {ret:+.1f}% rally provides positive momentum buffer")

        if vol == "extreme" or vol == "elevated":
            reasons.append("realized volatility is running well above historical norms")

        if vol_act == "surge" or vol_act == "heavy":
            reasons.append("trading volume surged, confirming strong institutional conviction")

        if not reasons:
            reasons.append("market metrics are near historical median baseline values")

        why_text = "; ".join(reasons) + "."
        return f"AI Diagnosis: {why_text} Overall, 5 Machine Learning models estimate a {crash_prob * 100.0:.0f}% chance of a catastrophic drop (≤ -1.60%) and expect next-day return around {pred_ret:+.2f}%."


def print_formatted_report(res: Dict[str, Any]):
    """Prints a clean ASCII human-understandable report card."""
    print("\n" + "=" * 65)
    print("      🤖 MACHINE LEARNING NEXT-DAY RISK & RETURN REPORT      ")
    print("=" * 65)
    print(f"  RISK STATUS:           {res['risk_rating']}")
    print(f"  PREDICTED DIRECTION:   {res['predicted_direction']} ({res['consensus_predicted_return']:+.2f}%)")
    print(f"  CRASH PROBABILITY:     {res['crash_risk_probability']}% chance of drop ≤ -1.60%")
    print(f"  RECOMMENDED ACTION:    {res['action_advice']}")
    print("-" * 65)
    print("  EXPLANATION FOR HUMANS:")
    print(f"  {res['plain_english_explanation']}")
    print("-" * 65)
    print("  5-MODEL CONSENSUS BREAKDOWN:")
    for model, details in res["model_breakdown"].items():
        ret_str = f"Pred Return: {details['predicted_return']:+.2f}%" if "predicted_return" in details else ""
        prob_str = f"Crash Prob: {details['crash_prob']}%" if "crash_prob" in details else ""
        vals = ", ".join([v for v in [ret_str, prob_str] if v])
        print(f"    • {model:<28} -> {vals}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Human-Centric Next-Day ML Risk Inference")
    parser.add_argument("--preset", choices=["calm", "normal", "panic", "rally"], default=None, help="Quick scenario preset")
    parser.add_argument("--return-pct", type=float, default=0.0, help="Today's stock return in percent (e.g. -1.5 for drop, +1.2 for gain)")
    parser.add_argument("--vix", type=float, default=16.0, help="CBOE VIX level (e.g. 13 for calm, 28 for high fear)")
    parser.add_argument("--volatility", choices=["low", "normal", "elevated", "extreme"], default="normal", help="Volatility environment")
    parser.add_argument("--volume", choices=["quiet", "normal", "heavy", "surge"], default="normal", help="Trading volume activity")
    parser.add_argument("--trend", choices=["strong_up", "mild_up", "flat", "down", "crash"], default="flat", help="Recent market trend")

    args = parser.parse_args()

    # Presets override individual arguments if specified
    if args.preset == "calm":
        ret, vix, vol, vol_act, trend = 0.4, 12.8, "low", "normal", "mild_up"
    elif args.preset == "panic":
        ret, vix, vol, vol_act, trend = -2.8, 29.5, "extreme", "surge", "crash"
    elif args.preset == "rally":
        ret, vix, vol, vol_act, trend = 2.4, 14.5, "normal", "heavy", "strong_up"
    elif args.preset == "normal":
        ret, vix, vol, vol_act, trend = 0.05, 16.0, "normal", "normal", "flat"
    else:
        ret, vix, vol, vol_act, trend = args.return_pct, args.vix, args.volatility, args.volume, args.trend

    engine = RiskInferenceEngine()
    result = engine.run_human_inference(
        today_return_pct=ret,
        vix_level=vix,
        volatility_level=vol,
        volume_activity=vol_act,
        recent_market_trend=trend
    )
    print_formatted_report(result)
