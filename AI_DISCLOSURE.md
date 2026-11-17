# AI Use Disclosure

**Project:** Portfolio Risk Analyzer: Can ML forecast market volatility better than history?
**Course:** ES 335 Machine Learning (Undergraduate track), IIT Gandhinagar
**Team:** Siddharth Dangi (25110300) & Aryan Vatsa Chauhan (25110059)
**Repository:** https://github.com/siddharthdangi1911/Portfolio_Risk_Analyzer

---

## 1. AI tools used

| Tool | Provider | How we accessed it | Used for |
|---|---|---|---|
| **Claude Code** (model: Claude Opus 5.5) | Anthropic | Claude desktop app | Auditing, coding, website build, visualization |

---

## 2. Kinds of assistance, by part of the project

| Part of the project | Who did it | What the AI did |
|---|---|---|
| Research question, proposal, choice of assets, baseline and metrics | **Team** | ChatGPT help us to get more insight into finance terms |
| Data download and cleaning (`src/data_loader.py`) | team | Yahoo Finance prices for SPY, QQQ, TLT, GLD and VIX |
| Feature engineering (`src/feature_engineering.py`) | AI-assisted coding, team-reviewed | Implemented the 22 backward-looking features and the 20-day forward volatility target. |
| Model selection protocol (`src/model_selection.py`, `src/models/ml_models.py`) | AI-assisted coding, team-reviewed | Audited the old pipeline (all five models fit once, validation unused, ensemble at inference). Rebuilt it as: CV screening on train only → tune the winner on validation → single test. Added LightGBM as a sixth candidate at the team's request. |
| Training and evaluation (`src/train_pipeline.py`, `src/evaluation.py`) | AI-assisted coding, team-reviewed | Bootstrap confidence intervals, regime breakdown, permutation importance, and a `training_proof` block (fit counts, timings, shuffled-label control) that answers "is training really happening?". |
| Inference (`src/inference.py`) |  team | Ensured inference uses only the single saved selected model. |
| Debugging | AI | Diagnosed the "training is too fast" concern (Ridge is a closed-form solve on ~10k rows, so milliseconds is expected), a missing `yfinance` install, mobile layout overflow, and Next.js build errors. |
| Website (`web/`, published to `docs/`) | AI-assisted coding, team-reviewed and edited | Built the interactive dashboard, first in plain HTML/JS and then in Next.js + TypeScript + Recharts with static export to GitHub Pages. The team later edited page content themselves. |
| Visualization | AI-assisted | Chart design, colour palette (colour-blind-checked roles), matplotlib figures in `artifacts/figures/`. |

---

## 3. External code, data and material

All third-party code is used through the open-source libraries below, imported as dependencies (see `requirements.txt` and `web/package.json`).

### Data
- **Yahoo Finance** daily prices, accessed through the open-source `yfinance` package. Public data, no credentials. Used for academic purposes only.

### Libraries
- Python: NumPy, pandas, scikit-learn, LightGBM, matplotlib, joblib, yfinance
- Website: Next.js, React, TypeScript, Recharts.

### Methods and ideas we rely on
- **EWMA volatility (λ = 0.94):** J.P. Morgan/Reuters (1996). *RiskMetrics — Technical Document*, 4th ed.
- **HAR-style log volatility features:** Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics*, 7(2), 174–196.
- **Smearing back-transform for a log target:** Duan, N. (1983). Smearing estimate: a nonparametric retransformation method. *Journal of the American Statistical Association*, 78(383), 605–610.
- **QLIKE loss for volatility forecasts:** Patton, A. J. (2011). Volatility forecast comparison using imperfect volatility proxies. *Journal of Econometrics*, 160(1), 246–256.
- **Moving-block bootstrap:** Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations. *The Annals of Statistics*, 17(3), 1217–1241.
- **Permutation feature importance:** Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32. We used the scikit-learn implementation.

---
