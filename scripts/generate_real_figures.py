"""
scripts/generate_real_figures.py
=============================================================================
Static figures for the report/README, drawn only from pipeline artifacts:
  1_cv_screening.png          Stage 1: CV RMSE of every family vs baselines
  2_validation_tuning.png     Stage 2: validation RMSE across the tuning grid
  3_test_forecast_<ASSET>.png Stage 3: forecast vs realised volatility (test)
  4_test_regimes.png          errors in calm / normal / stressed markets
  5_pred_vs_actual.png        calibration scatter on the test set
  6_permutation_importance.png
=============================================================================
"""

import json
import logging
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import ARTIFACTS_DIR, DOCS_FIG_DIR, FIGURES_DIR
from src.models.ml_models import PRIMARY_BASELINE

logger = logging.getLogger(__name__)

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
C_MODEL, C_BASE, C_HIST, C_EWMA = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 150, "font.size": 10, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 11, "legend.frameon": False,
})


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate_all_figures():
    for old in FIGURES_DIR.glob("*.png"):
        try:
            old.unlink()
        except OSError:
            pass
    res = json.loads((ARTIFACTS_DIR / "evaluation_metrics.json").read_text())
    preds = pd.read_csv(ARTIFACTS_DIR / "test_predictions.csv", index_col=0, parse_dates=True)
    fam = res["final_model"]["family"]
    mkey = f"{fam} (final)"

    # 1. screening
    rows = sorted(res["stage1_screening"]["results"], key=lambda r: r["cv_rmse_mean"])
    fig, ax = plt.subplots(figsize=(7, 3.8))
    y = np.arange(len(rows))
    cols = [C_MODEL if r["name"] == fam else (C_BASE if r["kind"] == "baseline" else "#9ec5f4") for r in rows]
    ax.barh(y, [r["cv_rmse_mean"] for r in rows], color=cols, height=0.6)
    for i, r in enumerate(rows):
        ax.scatter(r["fold_rmse"], [i] * len(r["fold_rmse"]), s=10, color=INK2, zorder=3)
        ax.text(r["cv_rmse_mean"] + 0.2, i + 0.28, f"{r['cv_rmse_mean']:.2f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, [f"{r['name']}{' (baseline)' if r['kind'] == 'baseline' else ''}" for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("RMSE (vol points) - bar = mean of 5 folds, dots = individual folds")
    ax.set_title(f"Stage 1: expanding-window CV on 2012-2020 - selected: {fam}")
    _save(fig, "1_cv_screening.png")

    # 2. tuning
    trials = res["stage2_tuning"]["top_trials"]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    labels = [", ".join(f"{k}={v}" for k, v in t["params"].items()) for t in trials[:12]]
    ax.barh(range(len(labels)), [t["val_rmse"] for t in trials[:12]], color=[C_MODEL] + ["#9ec5f4"] * 11, height=0.6)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.invert_yaxis()
    vc = res["validation_comparison"]
    for name, c in ((PRIMARY_BASELINE, C_BASE), ("EWMA vol (RiskMetrics)", C_EWMA)):
        ax.axvline(vc[name]["RMSE"], color=c, ls="--", lw=1.5, label=f"{name}: {vc[name]['RMSE']:.2f}")
    lo = min(t["val_rmse"] for t in trials[:12])
    ax.set_xlim(lo * 0.85, max(max(t["val_rmse"] for t in trials[:12]), vc[PRIMARY_BASELINE]["RMSE"]) * 1.05)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlabel("Validation RMSE (2021-2022)")
    ax.set_title(f"Stage 2: tuning {fam} on validation (best 12 of {res['stage2_tuning']['n_configs']} configs)")
    _save(fig, "2_validation_tuning.png")

    # 3. timelines
    for a in res["data"]["assets"]:
        d = preds[preds["asset"] == a].sort_index()
        fig, ax = plt.subplots(figsize=(9, 3.6))
        ax.plot(d.index, d["target_vol_20d"], color=INK, lw=1.6, label="Realised (next 20 days)")
        ax.plot(d.index, d[f"pred::{mkey}"], color=C_MODEL, lw=1.6, label=f"{fam} forecast")
        ax.plot(d.index, d[f"pred::{PRIMARY_BASELINE}"], color=C_BASE, lw=1.2, alpha=0.9, label=f"Baseline: {PRIMARY_BASELINE}")
        ax.plot(d.index, d["pred::Historical average"], color=C_HIST, lw=1.2, ls="--", label="Baseline: historical average")
        ax.set_ylabel("Annualised volatility (%)")
        ax.set_title(f"{a}: 20-day volatility forecasts on the untouched test period")
        ax.legend(fontsize=8, ncol=2, loc="upper left")
        _save(fig, f"3_test_forecast_{a}.png")

    # 4. regimes
    reg = res["test_regimes"]
    keys = [mkey, PRIMARY_BASELINE, "Historical average"]
    colors = [C_MODEL, C_BASE, C_HIST]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    x = np.arange(len(reg))
    w = 0.26
    for i, (k, c) in enumerate(zip(keys, colors)):
        vals = [r[k]["RMSE"] for r in reg]
        ax.bar(x + (i - 1) * w, vals, w - 0.02, color=c, label=k)
    ax.set_xticks(x, [f"{r['regime']}\n(n={r['n']})" for r in reg], fontsize=8)
    ax.set_ylabel("Test RMSE (vol points)")
    ax.set_title("Test errors by market regime (VIX terciles from training data)")
    ax.legend(fontsize=8)
    _save(fig, "4_test_regimes.png")

    # 5. scatter
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    for a, mk in zip(res["data"]["assets"], ["o", "s", "^", "D"]):
        d = preds[preds["asset"] == a]
        ax.scatter(d[f"pred::{mkey}"], d["target_vol_20d"], s=9, alpha=0.5, marker=mk, label=a)
    lim = [0, float(max(preds["target_vol_20d"].max(), preds[f"pred::{mkey}"].max())) * 1.05]
    ax.plot(lim, lim, color=INK2, lw=1, ls="--")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Forecast vol (%)"); ax.set_ylabel("Realised vol (%)")
    ax.set_title("Test set: forecast vs realised")
    ax.legend(fontsize=8)
    _save(fig, "5_pred_vs_actual.png")

    # 6. importance
    imp = res["permutation_importance_test"][:12]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.barh(range(len(imp)), [d["importance"] for d in imp], xerr=[d["std"] for d in imp], color=C_MODEL, height=0.6,
            error_kw={"ecolor": INK2, "lw": 1})
    ax.set_yticks(range(len(imp)), [d["feature"] for d in imp], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Increase in test RMSE when the feature is shuffled")
    ax.set_title("Permutation importance (final model, test set)")
    _save(fig, "6_permutation_importance.png")

    for old in DOCS_FIG_DIR.glob("*.png"):
        try:
            old.unlink()
        except OSError:
            pass
    for f in FIGURES_DIR.glob("*.png"):
        shutil.copy(f, DOCS_FIG_DIR / f.name)
    logger.info(f"Figures written to {FIGURES_DIR} and copied to {DOCS_FIG_DIR}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    generate_all_figures()
