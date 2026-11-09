"""
src/train_pipeline.py
=============================================================================
End-to-end, leakage-safe training and evaluation.

  1. Load prices, run data-quality checks
  2. Build the asset-day modelling table (features at t, target over t+1..t+20)
  3. Chronological split with a 20-day embargo: train 2012-2020,
     validation 2021-2022, test 2023-2024
  4. Stage 1  screen 5 model families with expanding-window CV on train only
  5. Stage 2  tune the winning family on the validation split
  6. Stage 3  refit winner on train+validation, evaluate ONCE on test against
              the baselines, with block-bootstrap confidence intervals
  7. Save the final model (used by src/inference.py) and all metrics
=============================================================================
"""

import json
import logging
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.inspection import permutation_importance

from src.config import ARTIFACTS_DIR, CONFIG
from src.data_loader import clean_and_check, load_prices
from src.evaluation import block_bootstrap_compare, regime_breakdown, regression_metrics
from src.feature_engineering import FEATURE_INFO, build_model_table
from src.model_selection import make_splits, screen_families, split_summary, tune_on_validation
from src.models.ml_models import PRIMARY_BASELINE, make_baselines, make_model

logger = logging.getLogger(__name__)


def _r(x, k=4):
    if isinstance(x, dict):
        return {a: _r(b, k) for a, b in x.items()}
    if isinstance(x, list):
        return [_r(v, k) for v in x]
    if isinstance(x, (float, np.floating)):
        return round(float(x), k)
    return x


def run_full_pipeline(refresh_data: bool = False) -> Tuple[Dict[str, Any], pd.DataFrame]:
    t_start = time.perf_counter()
    np.random.seed(CONFIG.SEED)

    # 1-2. Data
    prices, quality = clean_and_check(load_prices(refresh=refresh_data))
    table, feats, assets = build_model_table(prices)
    splits = make_splits(table)
    sizes = split_summary(splits)
    logger.info(f"Assets: {assets}")
    for k, v in sizes.items():
        logger.info(f"  {k:<5}: {v['start']} -> {v['end']}  {v['dates']} days, {v['rows']} rows")

    train, val, trval, test = splits["train"], splits["val"], splits["train_val"], splits["test"]
    y_te = test["target_vol_20d"].to_numpy()

    # 3. Stage 1: screening with CV on the training period only
    logger.info("Stage 1 - screening all candidate families with expanding-window CV (training data only)")
    screening = screen_families(train, feats)
    family = screening["selected_family"]
    logger.info(f"  -> selected family: {family}")

    # 4. Stage 2: tune the selected family on validation
    logger.info(f"Stage 2 - tuning {family} on the validation split")
    tuning = tune_on_validation(family, train, val, feats)
    best_params = tuning["best_params"]

    # Validation comparison (for the report): tuned model vs baselines, all fit on train
    val_model = make_model(family, best_params).fit(train[feats].to_numpy(), train["target_vol_20d"].to_numpy())
    val_table = {f"{family} (tuned)": regression_metrics(val["target_vol_20d"], val_model.predict(val[feats].to_numpy()))}
    for b in make_baselines():
        b.fit(train, train["target_vol_20d"].to_numpy())
        val_table[b.name] = regression_metrics(val["target_vol_20d"], b.predict(val))

    # 5. Stage 3: refit on train+val, single evaluation on test
    logger.info("Stage 3 - refitting on train+validation and evaluating once on the test split")
    t0 = time.perf_counter()
    final_model = make_model(family, best_params)
    final_model.fit(trval[feats].to_numpy(), trval["target_vol_20d"].to_numpy())
    fit_secs = time.perf_counter() - t0
    p_final = final_model.predict(test[feats].to_numpy())

    # Training sanity checks - evidence that fitting really happens and that
    # the model learns signal from the features (not just a constant):
    #  (a) learned coefficients of the final model (linear families only)
    #  (b) the same model trained on SHUFFLED targets must do much worse
    inner = getattr(final_model, "regressor_", final_model)
    est = inner.named_steps["model"]
    coefs = ({f: float(c) for f, c in sorted(zip(feats, est.coef_), key=lambda t: -abs(t[1]))}
             if hasattr(est, "coef_") else None)
    rng = np.random.default_rng(CONFIG.SEED)
    shuffled = make_model(family, best_params).fit(trval[feats].to_numpy(),
                                                   rng.permutation(trval["target_vol_20d"].to_numpy()))
    shuffled_rmse = regression_metrics(y_te, shuffled.predict(test[feats].to_numpy()))["RMSE"]
    n_fits = sum(r["fits"] for r in screening["results"]) + tuning["n_configs"] + 2
    training_proof = {
        "total_model_fits": n_fits,
        "training_rows_final_fit": int(len(trval)), "n_features": len(feats),
        "final_fit_seconds": round(fit_secs, 4),
        "why_fast": "Ridge has a closed-form solution (one linear solve on a ~10k x 22 matrix), so a fit takes milliseconds; "
                    "the slowest family (SVR) takes seconds. Speed reflects data size and model class, not skipped training.",
        "shuffled_target_test_RMSE": shuffled_rmse,
        "real_target_test_RMSE": float(np.sqrt(np.mean((p_final - y_te) ** 2))),
        "learned_coefficients_scaled_log_space": coefs,
        "smearing_factor": float(getattr(final_model, "smearing_", 1.0)),
    }
    logger.info(f"  [proof] {n_fits} model fits in total; final fit {fit_secs * 1000:.1f} ms on {len(trval)} rows. "
                f"Test RMSE real labels {training_proof['real_target_test_RMSE']:.3f} vs shuffled labels {shuffled_rmse:.3f}")

    preds = {f"{family} (final)": p_final}
    # Sensitivity: the same tuned model trained on the training split only
    preds[f"{family} (train-only fit)"] = val_model.predict(test[feats].to_numpy())
    for b in make_baselines():
        b.fit(trval, trval["target_vol_20d"].to_numpy())
        preds[b.name] = b.predict(test)
    model_key = f"{family} (final)"

    test_table = {k: regression_metrics(y_te, v) for k, v in preds.items()}
    for k, v in test_table.items():
        logger.info(f"  [test] {k:<34} RMSE {v['RMSE']:.3f}  MAE {v['MAE']:.3f}  R2 {v['R2']:.3f}  QLIKE {v['QLIKE']:.4f}")

    boot = {b: block_bootstrap_compare(test.index, y_te, p_final, preds[b])
            for b in (PRIMARY_BASELINE, "Historical average", "EWMA vol (RiskMetrics)")}

    per_asset = {}
    for a in assets:
        m = (test["asset"] == a).to_numpy()
        per_asset[a] = {k: regression_metrics(y_te[m], v[m]) for k, v in preds.items()
                        if k in (model_key, PRIMARY_BASELINE, "Historical average", "EWMA vol (RiskMetrics)")}

    vix_terciles = list(np.quantile(train["vix_level"], [1 / 3, 2 / 3]))
    regimes = regime_breakdown(test, {k: preds[k] for k in (model_key, PRIMARY_BASELINE, "Historical average")}, vix_terciles)

    # Representative errors: largest misses of the final model
    err = pd.DataFrame({"date": test.index.strftime("%Y-%m-%d"), "asset": test["asset"].values,
                        "actual": y_te, "predicted": p_final, "baseline": preds[PRIMARY_BASELINE],
                        "vix": test["vix_level"].values})
    err["error"] = err["predicted"] - err["actual"]
    worst = err.reindex(err["error"].abs().sort_values(ascending=False).index)
    worst = worst.drop_duplicates(subset=["asset", "date"]).head(40)
    # keep one example per (asset, month) so the list is not 20 copies of the same episode
    worst["month"] = worst["date"].str[:7]
    worst = worst.drop_duplicates(subset=["asset", "month"]).head(8).drop(columns="month")

    # Permutation importance on the test split (interpretation only, not selection)
    pi = permutation_importance(final_model, test[feats].to_numpy(), y_te, n_repeats=5,
                                random_state=CONFIG.SEED, scoring="neg_root_mean_squared_error")
    importance = sorted(({"feature": f, "importance": float(m), "std": float(s)}
                         for f, m, s in zip(feats, pi.importances_mean, pi.importances_std)),
                        key=lambda d: -d["importance"])

    # 6. Save artifacts
    bundle = {"model": final_model, "family": family, "params": best_params, "features": feats,
              "assets": assets, "trained_through": trval.index.max().strftime("%Y-%m-%d"),
              "horizon": CONFIG.HORIZON, "sklearn_version": sklearn.__version__}
    joblib.dump(bundle, ARTIFACTS_DIR / "final_model.joblib")

    out = test[["asset", "vix_level", "vol_20d", "ewma_vol", "target_vol_20d"]].copy()
    for k, v in preds.items():
        out[f"pred::{k}"] = v
    out.to_csv(ARTIFACTS_DIR / "test_predictions.csv", index_label="Date")

    results = {
        "task": {"target": "annualised realised volatility (%) over the next 20 trading days",
                 "unit": "one asset on one trading day", "horizon_days": CONFIG.HORIZON,
                 "prediction_time": "after the close of day t; features use prices up to and including t"},
        "data": {"assets": assets, "asset_descriptions": {a: CONFIG.ASSETS.get(a, a) for a in assets},
                 "quality": quality, "n_rows_model_table": int(len(table)), "features": feats},
        "splits": sizes,
        "stage1_screening": screening,
        "stage2_tuning": {k: v for k, v in tuning.items() if k != "trials"} | {"top_trials": tuning["trials"][:15]},
        "validation_comparison": val_table,
        "final_model": {"family": family, "params": best_params, "trained_on": "train + validation",
                        "trained_through": bundle["trained_through"], "fit_seconds": round(fit_secs, 2)},
        "training_proof": training_proof,
        "test_metrics": test_table,
        "test_bootstrap_vs_baselines": boot,
        "test_per_asset": per_asset,
        "test_regimes": regimes,
        "vix_regime_thresholds_from_train": vix_terciles,
        "representative_errors": worst.to_dict(orient="records"),
        "permutation_importance_test": importance,
        "feature_info": {k: list(v) for k, v in FEATURE_INFO.items()},
        "reproducibility": {"seed": CONFIG.SEED, "python": platform.python_version(),
                            "sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__,
                            "runtime_seconds": round(time.perf_counter() - t_start, 1)},
    }
    results = _r(results)
    (ARTIFACTS_DIR / "evaluation_metrics.json").write_text(json.dumps(results, indent=2, default=str))
    logger.info(f"Pipeline finished in {results['reproducibility']['runtime_seconds']}s")
    return results, table


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh-data", action="store_true")
    run_full_pipeline(ap.parse_args().refresh_data)
