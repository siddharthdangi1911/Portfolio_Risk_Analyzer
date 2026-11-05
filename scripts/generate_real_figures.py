"""
scripts/generate_real_figures.py
=============================================================================
Generate Real Model Evaluation Figures from Out-of-Sample Test Data
Course: ES 335 - Machine Learning (Undergraduate Track)
Team: Siddharth Dangi (25110300) & Aryan Vatsa Chauhan (25110059)
=============================================================================
Generates 6 publication-grade figures saved to docs/figures/ and artifacts/figures/:
  1. model_predictions_vs_actual.png (Out-of-sample return predictions vs ground truth)
  2. residual_error_distribution.png (Residual error histograms with normal fit)
  3. roc_curves_classification.png (ROC curves with exact AUC scores)
  4. feature_importance_ranking.png (Normalized feature contributions across models)
  5. confusion_matrix_heatmaps.png (2x2 confusion matrices for all classifiers)
  6. regime_error_analysis.png (Error degradation across volatility regimes)
"""

import sys
import logging
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import joblib

from src.config import CONFIG, ARTIFACTS_DIR, DOCS_DIR
from src.data_loader import load_raw_market_data
from src.feature_engineering import build_ml_feature_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)

DOCS_FIGS_DIR = DOCS_DIR / "figures"
ARTIFACTS_FIGS_DIR = ARTIFACTS_DIR / "figures"
DOCS_FIGS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_FIGS_DIR.mkdir(parents=True, exist_ok=True)

# Styling palette
DARK_BG = "#0b0f19"
CARD_BG = "#131b2e"
TEXT_COLOR = "#f1f5f9"
MUTED_TEXT = "#94a3b8"
GRID_COLOR = "#23304e"

COLORS = {
    "actual": "#38bdf8",
    "lightgbm": "#ec4899",
    "svm": "#10b981",
    "decision_tree": "#a855f7",
    "linear_regression": "#f59e0b",
    "chance": "#64748b",
}


def apply_dark_theme(fig, ax):
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)
    ax.tick_params(colors=MUTED_TEXT, labelsize=9)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.grid(True, linestyle="--", alpha=0.35, color=GRID_COLOR)


def generate_all_figures():
    logger.info("Loading real data and models for figure generation...")
    raw = load_raw_market_data(use_cache=True)
    df, feature_cols = build_ml_feature_dataset(raw, save_csv=False)

    scaler = joblib.load(ARTIFACTS_DIR / "feature_scaler.joblib")
    lr = joblib.load(ARTIFACTS_DIR / "best_linear_regression.joblib")
    dt = joblib.load(ARTIFACTS_DIR / "best_decision_tree.joblib")
    svm = joblib.load(ARTIFACTS_DIR / "best_svm_model.joblib")
    log = joblib.load(ARTIFACTS_DIR / "best_logistic_regression.joblib")
    lgb = joblib.load(ARTIFACTS_DIR / "best_lightgbm_model.joblib")

    test_mask = df.index >= CONFIG.TEST_START
    test_df = df.loc[test_mask].copy()

    X_test_raw = test_df[feature_cols].values
    X_test_scaled = scaler.transform(X_test_raw)
    y_reg_test = test_df["target_next_ret"].values
    y_cls_test = test_df["target_high_risk"].values
    test_dates = test_df.index

    # Inference
    p_lr_reg = lr.predict(X_test_scaled)
    p_dt_reg, p_dt_cls, p_dt_prob = dt.predict(X_test_scaled)
    p_svm_reg, p_svm_cls, p_svm_prob = svm.predict(X_test_scaled)
    p_log_cls, p_log_prob = log.predict(X_test_scaled)
    p_lgb_reg, p_lgb_cls, p_lgb_prob = lgb.predict(X_test_scaled)

    # -------------------------------------------------------------------------
    # 1. Model Predictions vs Ground Truth Return
    # -------------------------------------------------------------------------
    logger.info("Generating Figure 1: Model Predictions vs Ground Truth...")
    fig, ax = plt.subplots(figsize=(12, 5), dpi=200)
    apply_dark_theme(fig, ax)

    ax.plot(test_dates, y_reg_test, label="Ground Truth Return (%)", color=COLORS["actual"], alpha=0.9, lw=1.5)
    ax.plot(test_dates, p_lgb_reg, label="LightGBM Regressor", color=COLORS["lightgbm"], alpha=0.8, lw=1.3)
    ax.plot(test_dates, p_svm_reg, label="Support Vector Machine (SVR RBF)", color=COLORS["svm"], alpha=0.75, lw=1.2)
    ax.plot(test_dates, p_lr_reg, label="Linear Regression (Ridge)", color=COLORS["linear_regression"], alpha=0.75, lw=1.2, ls="--")

    # Highlight crash days
    crash_mask = y_cls_test == 1
    ax.scatter(test_dates[crash_mask], y_reg_test[crash_mask], color="#ef4444", s=35, zorder=5, label="Tail Crash (r ≤ -1.60%)")

    ax.set_title("Out-of-Sample Next-Day Return Predictions vs Ground Truth (2023–2024, 459 Days)", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Next-Day Log Return (%)", fontsize=10)
    ax.legend(facecolor=DARK_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=8.5, loc="upper right")
    fig.tight_layout()
    save_fig(fig, "1_model_predictions_vs_actual.png")

    # -------------------------------------------------------------------------
    # 2. Residual Error Distributions
    # -------------------------------------------------------------------------
    logger.info("Generating Figure 2: Residual Error Distributions...")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=200)
    models_res = [
        ("Linear Regression (Ridge)", y_reg_test - p_lr_reg, COLORS["linear_regression"]),
        ("Decision Tree (CART Depth 4)", y_reg_test - p_dt_reg, COLORS["decision_tree"]),
        ("Support Vector Machine (SVR)", y_reg_test - p_svm_reg, COLORS["svm"]),
        ("LightGBM Ensembles", y_reg_test - p_lgb_reg, COLORS["lightgbm"]),
    ]

    for ax_idx, (m_name, res, col) in enumerate(models_res):
        r_ax = axes[ax_idx // 2, ax_idx % 2]
        apply_dark_theme(fig, r_ax)

        r_ax.hist(res, bins=25, density=True, color=col, alpha=0.65, edgecolor=DARK_BG)
        
        # Fit Gaussian curve
        mu, std = np.mean(res), np.std(res)
        x_pts = np.linspace(-3.5, 3.5, 200)
        p_pts = (1.0 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_pts - mu) / std) ** 2)
        r_ax.plot(x_pts, p_pts, color=TEXT_COLOR, lw=1.5, ls="--", label=f"Normal Fit (μ={mu:+.2f}%, σ={std:.2f}%)")

        r_ax.axvline(0, color="#cbd5e1", ls=":", lw=1.0)
        r_ax.set_title(m_name, fontsize=10, fontweight="bold")
        r_ax.set_xlabel("Residual Error (Actual - Pred) %", fontsize=8.5)
        r_ax.set_ylabel("Density", fontsize=8.5)
        r_ax.legend(facecolor=DARK_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=7.5)

    fig.suptitle("Residual Error Distributions Across All Regression Architectures", fontsize=12, fontweight="bold", color=TEXT_COLOR, y=0.99)
    fig.tight_layout()
    save_fig(fig, "2_residual_error_distribution.png")

    # -------------------------------------------------------------------------
    # 3. ROC Curves (Tail Crash Classification)
    # -------------------------------------------------------------------------
    logger.info("Generating Figure 3: ROC Curves...")
    from sklearn.metrics import roc_curve, auc

    fig, ax = plt.subplots(figsize=(7, 6), dpi=200)
    apply_dark_theme(fig, ax)

    ax.plot([0, 1], [0, 1], color=COLORS["chance"], ls="--", lw=1.5, label="Random Chance (AUC = 0.500)")

    cls_preds = [
        ("LightGBM Classifier", p_lgb_prob, COLORS["lightgbm"]),
        ("Logistic Regression", p_log_prob, COLORS["actual"]),
        ("SVM Classifier (SVC RBF)", p_svm_prob, COLORS["svm"]),
        ("Decision Tree Classifier", p_dt_prob, COLORS["decision_tree"]),
    ]

    for name, probs, col in cls_preds:
        fpr, tpr, _ = roc_curve(y_cls_test, probs)
        score = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=col, lw=2.0, label=f"{name} (AUC = {score:.3f})")

    ax.set_title("Out-of-Sample ROC Curves (Tail Crash Detection: r ≤ -1.60%)", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=9.5)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=9.5)
    ax.legend(facecolor=DARK_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=8.5, loc="lower right")
    fig.tight_layout()
    save_fig(fig, "3_roc_curves_classification.png")

    # -------------------------------------------------------------------------
    # 4. Feature Importance Rankings
    # -------------------------------------------------------------------------
    logger.info("Generating Figure 4: Feature Importance...")
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=200)
    apply_dark_theme(fig, ax)

    lgb_imp = lgb.get_feature_importances(feature_cols)
    sorted_feats = sorted(lgb_imp.items(), key=lambda x: x[1], reverse=True)[:10]
    f_names = [x[0] for x in sorted_feats][::-1]
    f_vals = [x[1] for x in sorted_feats][::-1]

    bars = ax.barh(f_names, f_vals, color="#38bdf8", alpha=0.75, edgecolor="#38bdf8", height=0.6)
    ax.set_title("Top 10 Most Predictive Features (LightGBM Normalized Split Gain)", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Normalized Feature Importance Score", fontsize=9.5)

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.005, bar.get_y() + bar.get_height() / 2, f"{w:.3f}", va="center", ha="left", color=TEXT_COLOR, fontsize=8, family="monospace")

    fig.tight_layout()
    save_fig(fig, "4_feature_importance_ranking.png")

    # -------------------------------------------------------------------------
    # 5. Confusion Matrix Heatmaps
    # -------------------------------------------------------------------------
    logger.info("Generating Figure 5: Confusion Matrix Heatmaps...")
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), dpi=200)

    models_cm = [
        ("Logistic Regression", p_log_cls),
        ("LightGBM", p_lgb_cls),
        ("SVM (SVC RBF)", p_svm_cls),
        ("Decision Tree", p_dt_cls),
    ]

    for idx, (m_name, p_cls) in enumerate(models_cm):
        c_ax = axes[idx]
        apply_dark_theme(fig, c_ax)

        tp = np.sum((y_cls_test == 1) & (p_cls == 1))
        fp = np.sum((y_cls_test == 0) & (p_cls == 1))
        tn = np.sum((y_cls_test == 0) & (p_cls == 0))
        fn = np.sum((y_cls_test == 1) & (p_cls == 0))

        cm = np.array([[tn, fp], [fn, tp]])
        c_ax.imshow(cm, cmap="Blues", alpha=0.75)

        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                tag = "TN" if (i, j) == (0, 0) else ("FP" if (i, j) == (0, 1) else ("FN" if (i, j) == (1, 0) else "TP"))
                c_ax.text(j, i, f"{val}\n({tag})", ha="center", va="center", color="#fff", fontweight="bold", fontsize=9)

        c_ax.set_xticks([0, 1])
        c_ax.set_yticks([0, 1])
        c_ax.set_xticklabels(["Pred 0", "Pred 1"], fontsize=8)
        c_ax.set_yticklabels(["True 0", "True 1"], fontsize=8)
        c_ax.set_title(m_name, fontsize=9.5, fontweight="bold")

    fig.suptitle("Out-of-Sample Confusion Matrices (Threshold = 0.50)", fontsize=11, fontweight="bold", color=TEXT_COLOR, y=0.98)
    fig.tight_layout()
    save_fig(fig, "5_confusion_matrix_heatmaps.png")

    # -------------------------------------------------------------------------
    # 6. Regime Error Analysis
    # -------------------------------------------------------------------------
    logger.info("Generating Figure 6: Volatility Regime Errors...")
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=200)
    apply_dark_theme(fig, ax)

    vix_vals = test_df["vix_level"].values
    regimes = [
        ("Low Vol (VIX < 16)", vix_vals < 16),
        ("Med Vol (16 ≤ VIX ≤ 22)", (vix_vals >= 16) & (vix_vals <= 22)),
        ("High Vol (VIX > 22)", vix_vals > 22),
    ]

    labels = [r[0] for r in regimes]
    x = np.arange(len(labels))
    width = 0.2

    mae_lr = [np.mean(np.abs(y_reg_test[r[1]] - p_lr_reg[r[1]])) for r in regimes]
    mae_dt = [np.mean(np.abs(y_reg_test[r[1]] - p_dt_reg[r[1]])) for r in regimes]
    mae_svm = [np.mean(np.abs(y_reg_test[r[1]] - p_svm_reg[r[1]])) for r in regimes]
    mae_lgb = [np.mean(np.abs(y_reg_test[r[1]] - p_lgb_reg[r[1]])) for r in regimes]

    ax.bar(x - 1.5 * width, mae_lr, width, label="Linear Regression", color=COLORS["linear_regression"], alpha=0.85)
    ax.bar(x - 0.5 * width, mae_dt, width, label="Decision Tree", color=COLORS["decision_tree"], alpha=0.85)
    ax.bar(x + 0.5 * width, mae_svm, width, label="SVM (RBF)", color=COLORS["svm"], alpha=0.85)
    ax.bar(x + 1.5 * width, mae_lgb, width, label="LightGBM", color=COLORS["lightgbm"], alpha=0.85)

    ax.set_ylabel("Mean Absolute Error (MAE) %", fontsize=9.5)
    ax.set_title("Prediction Error Progression Across Market Volatility Regimes", fontsize=11, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(facecolor=DARK_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=8.5)

    fig.tight_layout()
    save_fig(fig, "6_regime_error_analysis.png")
    logger.info("All 6 publication figures generated successfully!")


def save_fig(fig, filename):
    out_docs = DOCS_FIGS_DIR / filename
    out_artifacts = ARTIFACTS_FIGS_DIR / filename
    fig.savefig(out_docs, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    fig.savefig(out_artifacts, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    logger.info(f"Saved figure to: {out_docs}")


if __name__ == "__main__":
    generate_all_figures()
