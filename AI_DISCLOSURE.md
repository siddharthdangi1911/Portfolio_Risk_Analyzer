# AI Use Disclosure

**Project:** Portfolio Risk Analyzer: Can ML forecast market volatility better than history?
**Course:** ES 335 Machine Learning (Undergraduate track), IIT Gandhinagar
**Team:** Siddharth Dangi (25110300) & Aryan Vatsa Chauhan (25110059)
**Repository:** https://github.com/siddharthdangi1911/Portfolio_Risk_Analyzer

---

## 1. Course policy we followed

> AI tools are permitted for brainstorming, coding, debugging, editing, and visualization. Disclose the tools and kinds of assistance used, check important outputs independently, and cite external code or material. You are responsible for every result and may be asked to explain or defend any part of the work in a viva.

This file is our disclosure under that policy. We have used AI only for the permitted purposes listed above.

---

## 2. AI tools used

| Tool | Provider | How we accessed it | Used for |
|---|---|---|---|
| **Claude Code** (model: Claude Opus 5.5) | Anthropic | Claude desktop app, Code tab, working directly in the project folder | Auditing, coding, debugging, refactoring, website build, visualization, editing documentation |
| _Add any other tool you used (e.g. ChatGPT, Gemini, GitHub Copilot) here, or delete this row_ | | | |

---

## 3. Kinds of assistance, by part of the project

| Part of the project | Who did it | What the AI did |
|---|---|---|
| Research question, proposal, choice of assets, baseline and metrics | **Team** | _Fill in: state whether any AI tool helped brainstorm or edit the proposal, and which one._ |
| Code written before the Claude Code sessions (earlier commits) | _Fill in_ | _Fill in: say who wrote it and whether any AI tool was used._ |
| Data download and cleaning (`src/data_loader.py`) | AI-assisted coding, team-reviewed | Switched the data source to dividend-adjusted Yahoo Finance prices for SPY, QQQ, TLT, GLD and VIX; wrote the data-quality report. |
| Feature engineering (`src/feature_engineering.py`) | AI-assisted coding, team-reviewed | Implemented the 22 backward-looking features and the 20-day forward volatility target. |
| Model selection protocol (`src/model_selection.py`, `src/models/ml_models.py`) | AI-assisted coding, team-reviewed | Audited the old pipeline (all five models fit once, validation unused, ensemble at inference). Rebuilt it as: CV screening on train only → tune the winner on validation → single test. Added LightGBM as a sixth candidate at the team's request. |
| Training and evaluation (`src/train_pipeline.py`, `src/evaluation.py`) | AI-assisted coding, team-reviewed | Bootstrap confidence intervals, regime breakdown, permutation importance, and a `training_proof` block (fit counts, timings, shuffled-label control) that answers "is training really happening?". |
| Inference (`src/inference.py`) | AI-assisted coding, team-reviewed | Ensured inference uses only the single saved selected model. |
| Debugging | AI | Diagnosed the "training is too fast" concern (Ridge is a closed-form solve on ~10k rows, so milliseconds is expected), a missing `yfinance` install, mobile layout overflow, and Next.js build errors. |
| Website (`web/`, published to `docs/`) | AI-assisted coding, team-reviewed and edited | Built the interactive dashboard, first in plain HTML/JS and then in Next.js + TypeScript + Recharts with static export to GitHub Pages. The team later edited page content themselves. |
| Visualization | AI-assisted | Chart design, colour palette (colour-blind-checked roles), matplotlib figures in `artifacts/figures/`. |
| README and on-site text | AI-drafted, team-edited | Drafted explanations of results, limitations, and the "why not ROC AUC" note. |
| Interpretation of results and conclusions | **Team**, with AI drafting | AI drafted wording; the team is responsible for every claim. |

_Edit this table so it matches exactly what each team member did. Add rows for anything AI was used for that is not listed._

---

## 4. Independent checks of important outputs

The AI ran some checks while building the project. Because we are responsible for every result, **we re-checked the important outputs ourselves**. Tick each box only after you have done it.

### Checks built into the pipeline (re-run with `python scripts/run_all.py`)
- [ ] **No test leakage:** the scaler sits inside a scikit-learn `Pipeline`; a 20-day embargo drops rows whose target window crosses a split; CV folds split on dates. Verified by reading `src/model_selection.py` and `src/models/ml_models.py`.
- [ ] **The test set is used once:** only `train_pipeline.py` Stage 3 touches `splits["test"]`; screening uses train only and tuning uses validation only.
- [ ] **Training really happens:** the log shows 96 model fits; a model trained on shuffled targets scores test RMSE ≈ 4.27 against ≈ 3.64 with real targets.
- [ ] **Baselines are computed correctly:** recent 20-day vol equals the `vol_20d` feature; EWMA uses λ = 0.94; the historical average uses training data only.
- [ ] **Reproducibility:** a fresh run with seed 42 gives the same numbers as `artifacts/evaluation_metrics.json`.

### Checks we did by hand
- [ ] Recomputed the 20-day realised volatility for one asset and date in a spreadsheet or notebook and matched the `target_vol_20d` value.
- [ ] Spot-checked a few prices in `data/prices.csv` against Yahoo Finance.
- [ ] Recomputed test RMSE and MAE for the selected model from `artifacts/test_predictions.csv`.
- [ ] Confirmed that the CV leaderboard (Ridge 6.66 < LightGBM 7.16 < …) matches the run log.
- [ ] Opened every page of the live website and confirmed the numbers match `evaluation_metrics.json`.
- [ ] Read every line of code we submit and can explain what it does.

---

## 5. External code, data and material

No code was copied from external repositories, tutorials or Q&A sites. All third-party code is used through the open-source libraries below, imported as dependencies (see `requirements.txt` and `web/package.json`).

### Data
- **Yahoo Finance** daily prices, accessed through the open-source `yfinance` package. Public data, no credentials. Used for academic purposes only.

### Libraries
- Python: NumPy, pandas, scikit-learn, LightGBM, matplotlib, joblib, yfinance
- Website: Next.js, React, TypeScript, Recharts
- An earlier version of the site used Chart.js and chartjs-plugin-zoom.

### Methods and ideas we rely on
- **EWMA volatility (λ = 0.94):** J.P. Morgan/Reuters (1996). *RiskMetrics — Technical Document*, 4th ed.
- **HAR-style log volatility features:** Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics*, 7(2), 174–196.
- **Smearing back-transform for a log target:** Duan, N. (1983). Smearing estimate: a nonparametric retransformation method. *Journal of the American Statistical Association*, 78(383), 605–610.
- **QLIKE loss for volatility forecasts:** Patton, A. J. (2011). Volatility forecast comparison using imperfect volatility proxies. *Journal of Econometrics*, 160(1), 246–256.
- **Moving-block bootstrap:** Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations. *The Annals of Statistics*, 17(3), 1217–1241.
- **Permutation feature importance:** Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32. We used the scikit-learn implementation.

---

## 6. Responsibility and viva readiness

We understand that **we are responsible for every result, line of code and claim** in this project, whether or not an AI tool helped produce it, and that we may be asked to explain or defend any part in the viva. Before submission, each of us made sure we can explain:

- the research question, the target (20-day forward realised volatility), the unit (one ETF on one day), and what we do **not** claim;
- why the split is chronological with a 20-day embargo, and why the test set is used only once;
- why Ridge was selected over LightGBM and the other models, and how alpha was tuned on validation;
- why training takes milliseconds and how the shuffled-label control shows the model learns;
- what RMSE, MAE, R², QLIKE and the bootstrap confidence intervals mean, and why ROC AUC does not apply to a regression task;
- why EWMA beat the model on the 2022 validation period, and the main limitations;
- how one small change would affect the results (for example a different horizon, adding an asset, or removing VIX features).

| Team member | Main contribution | Signature / date |
|---|---|---|
| Siddharth Dangi (25110300) | _fill in_ | |
| Aryan Vatsa Chauhan (25110059) | _fill in_ | |
