# Portfolio Risk Analyzer: Can ML forecast market volatility better than history?

**Course:** ES 335 Machine Learning (Undergraduate track), IIT Gandhinagar
**Team:** Siddharth Dangi (25110300) & Aryan Vatsa Chauhan (25110059)
**Website:** [`docs/`](docs/), published with GitHub Pages at https://siddharthdangi1911.github.io/Portfolio_Risk_Analyzer/

---

## 1. Question and claim

> Can machine learning predict an asset's volatility over the next 20 trading days better than simple historical estimates? And how can these forecasts help compare the risk of different portfolios?

| | |
|---|---|
| **Target** | Annualised realised volatility (%) over the **next 20 trading days**: `std(r[t+1..t+20]) * sqrt(252) * 100`, with daily log returns `r` |
| **Unit (row)** | One asset on one trading day |
| **Prediction time** | After the close of day *t*. All features use prices up to and including day *t* |
| **Baselines** | (1) the asset's historical average volatility in the training data, (2) recent 20-day volatility (the proposal's baseline), (3) RiskMetrics EWMA (λ = 0.94) |
| **Candidates** | Ridge regression, decision tree, random forest, SVR (RBF), gradient boosting (scikit-learn), LightGBM |
| **Metrics** | RMSE (primary), MAE, R², QLIKE, bias |
| **Not claimed** | Return or direction prediction, trading profits, or investment advice. Results hold only for the assets and period studied |

## 2. Results (from `artifacts/evaluation_metrics.json`)

Data: dividend-adjusted daily closes for **SPY, QQQ, TLT, GLD** plus **^VIX** (Yahoo Finance, 2012-01-03 to 2024-10-31), 12,436 asset-days.
Splits (after a 20-day embargo): train 2012-06 to 2020-12 (8,500 rows), validation 2021-01 to 2022-12 (1,932 rows), test 2023-01 to 2024-10 (1,764 rows).

**Stage 1, screening (5-fold expanding-window CV on train only, RMSE):** Ridge 6.66, LightGBM 7.16, random forest 7.22, gradient boosting 7.37, decision tree 7.65, SVR 8.11 (baselines: EWMA 7.23, recent vol 7.77, historical average 7.85). Ridge won every one of the 5 folds (LightGBM, the best tree model, was second), so it is the only family carried forward.

**Stage 2, tuning Ridge on validation (24 configs):** best `alpha = 30`, log target. On validation, EWMA (RMSE 5.08) still beat the tuned Ridge (5.46): 2022 was a persistent high-volatility year and the model pulled forecasts back toward normal too fast (bias −1.7 vol points).

**Stage 3, test (2023-01-03 to 2024-10-03, 1,764 asset-days), evaluated once:**

| Forecast | RMSE | MAE | R² | QLIKE |
|---|---:|---:|---:|---:|
| **Ridge (selected, refit on train+val)** | **3.64** | **2.78** | **0.27** | **0.121** |
| Ridge (train-only fit, sensitivity) | 3.64 | 2.76 | 0.27 | 0.121 |
| Historical average | 4.13 | 3.33 | 0.06 | 0.158 |
| Recent 20-day vol (proposal baseline) | 4.48 | 3.50 | −0.11 | 0.219 |
| EWMA (RiskMetrics) | 4.10 | 3.20 | 0.07 | 0.164 |

Moving-block bootstrap (20-day blocks, 2000 resamples) of the RMSE difference, model minus baseline:
* vs recent 20-day vol: −0.87, 95% CI [−1.15, −0.54]. **Better, beyond resampling noise.**
* vs EWMA: −0.46, 95% CI [−0.71, −0.18]. **Better, beyond resampling noise.**
* vs historical average: −0.48, 95% CI [−0.98, 0.04]. Probably better, but the interval just includes 0.

So the answer to our question is **"yes in 2023–2024, by about 0.5–0.9 vol points, but not in every regime"**: the same model lost to EWMA on the 2022 validation period.

## 3. Pipeline and why each step is there

```
data/prices.csv ──► quality checks ──► asset-day table (features at t, target t+1..t+20)
                                         │
              chronological split + 20-day embargo
              train 2012–2020 │ validation 2021–2022 │ test 2023–2024
                                         │
 Stage 1  SCREEN  6 families, expanding-window CV on TRAIN only ──► pick best family
 Stage 2  TUNE    that family's hyperparameters: fit on train, score on VALIDATION
 Stage 3  TEST    refit on train+val, evaluate ONCE on TEST vs 3 baselines (+ bootstrap CIs)
                                         │
            artifacts/final_model.joblib ──► src/inference.py (the only model used for inference)
```

* **Leakage control.** The scaler sits inside a scikit-learn `Pipeline`, so it is refit on each fold's training rows only. Rows whose 20-day target window crosses a split boundary are dropped (embargo), and every CV fold leaves a 20-day gap. Folds split on dates, so all assets of one day stay on the same side.
* **Log transforms.** Volatility is positive and right-skewed, and its errors grow with its level. The models are trained on log-volatility with log volatility/VIX features (a HAR-style log-log relation). The back-transform is corrected with Duan's smearing factor, estimated on training data. This correction was added after the validation set showed a systematic negative bias; the test set was not used for that decision.
* **Regimes.** Calm, normal and stressed periods are defined by VIX terciles from the *training* data.
* **Uncertainty.** Overlapping 20-day targets are autocorrelated, so the confidence intervals come from a moving-block bootstrap over dates.

### Is training really happening? Why is it so fast?
Yes. Every run logs a `[proof]` line and saves a `training_proof` block in `evaluation_metrics.json`:
* **96 model fits** in total: 70 CV fits to screen the 6 families, 24 validation fits to tune Ridge, the final refit, and a control fit.
* The final Ridge fit on 10,512 rows × 22 features takes about **6 ms** because Ridge is one closed-form linear solve. The slowest family (SVR) takes about 20 s for its 10 CV fits. A full run takes about 30 s. Speed reflects the data size and model class, not skipped training.
* **Shuffled-label control:** the same model trained on randomly shuffled targets scores test RMSE 4.27, against 3.64 with the real targets, so the model learns real signal from the features.
* The old pipeline's real problem was not speed: it fit all five models once with fixed settings, never used the validation set, and averaged all five at inference. Now one model is selected by CV, tuned on validation, tested once, and that same saved model (`artifacts/final_model.joblib`) is the only one used by `src/inference.py`.

## 4. Repository layout

```
src/config.py              assets, dates, splits, horizon, seed (single source of truth)
src/data_loader.py         Yahoo Finance download + cache + data-quality report
src/feature_engineering.py asset-day features and 20-day forward volatility target
src/models/ml_models.py    baselines, 6 candidate families, screening & tuning grids
src/model_selection.py     embargoed splits, expanding-window CV, screening, validation tuning
src/evaluation.py          RMSE/MAE/R²/QLIKE, block bootstrap, regime breakdown
src/portfolio.py           portfolio metrics, trailing correlations, presets
src/train_pipeline.py      end-to-end run, saves artifacts/
src/inference.py           forecasts with the selected model (CLI)
scripts/run_all.py         train → export website data → figures
scripts/export_web_data.py writes docs/data/site_data.js
scripts/generate_real_figures.py  static figures in artifacts/figures and docs/figures
docs/                      website (index.html, app.js, style.css, data/, figures/)
```

## 5. Reproduce

```bash
python -m venv .venv && .venv\Scripts\activate        # Windows  (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
python scripts/run_all.py --refresh-data              # about 30 s
python -m src.inference --weights SPY=0.4,TLT=0.4,GLD=0.2
python -m src.inference --date 2024-08-05             # forecast as of a past date, with the realised value
python -m http.server 8000 -d docs                    # view the site at http://localhost:8000
```

Data: Yahoo Finance through `yfinance`, public, no credentials or API keys. The fixed seed is 42. The library versions used are recorded in `evaluation_metrics.json → reproducibility`.

## 6. Limitations
* The advantage depends on the regime: the model beat the baselines in calm 2023–2024 but lost to EWMA in 2022.
* One-day VIX spikes that fade quickly, such as 5 August 2024, cause large over-predictions.
* 20-day realised volatility is a noisy measure of true volatility, and overlapping targets reduce the effective sample size.
* The test period's largest misses: the 5 August 2024 VIX spike (over-prediction, e.g. QQQ forecast 42 vs realised 23) and the July 2024 tech rotation (under-prediction, QQQ forecast 15 vs realised 31).
* The universe is a few US ETFs and the horizon is one, so the results may not carry over to other assets or horizons.
* The 2020 COVID fold dominates the CV mean (RMSE ≈ 14 vs ≈ 4–6 in other folds), although Ridge also wins the other four folds.
* Prices come from a single vendor (Yahoo Finance) and were not cross-checked. The data ends 2024-10-31 by design (fixed `END_DATE`) so results are reproducible.
