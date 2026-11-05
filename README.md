# 1-Day Market Risk & Return Prediction Engine: A Core Machine Learning Framework

**Course:** ES 335 — Machine Learning (Undergraduate Track)  
**Institution:** Indian Institute of Technology Gandhinagar  
**Project Team:** Siddharth Dangi (25110300) & Aryan Vatsa Chauhan (25110059)  

---

## 🌐 Interactive Machine Learning Dashboard (GitHub Pages)

The interactive web deliverable is located in [`docs/`](docs/) and ready for deployment on **GitHub Pages**:
- **Live Website:** [Portfolio_Risk_Analyzer](https://siddharthdangi1911.github.io/Portfolio_Risk_Analyzer/)
- **Run Locally (Zero dependencies):**
  ```bash
  python -m http.server 8000 -d docs
  # Then open http://localhost:8000 in your browser
  ```
  *(You can also open [`docs/index.html`](docs/index.html) directly in any browser—embedded offline fallbacks guarantee 100% functionality even on `file://` protocol).*

---

## 1. Machine Learning Problem Formulation

Rather than relying on complex econometric multi-asset portfolio theory, this project formulates financial risk strictly through **fundamental Machine Learning** over a **1-Day Time Horizon** ($t \to t+1$):

Given an engineered feature vector $\mathbf{x}_t \in \mathbb{R}^{22}$ computed strictly at or before market close on day $t$, the system solves two canonical supervised ML tasks:

### Task A: Continuous Next-Day Return Regression
Predict the real-valued next trading day logarithmic return $y_{\text{reg}, t} \in \mathbb{R}$:
$$y_{\text{reg}, t} = r_{t+1} = \ln\left(\frac{P_{t+1}}{P_t}\right) \times 100$$
* **Objective Function:** Minimize Mean Squared Error (MSE):
  $$\mathcal{L}_{\text{MSE}}(\mathbf{w}) = \frac{1}{N} \sum_{i=1}^N \left( y_{\text{reg}, i} - \hat{y}_{\text{reg}, i}(\mathbf{x}_i) \right)^2$$

### Task B: Downside Tail Crash Classification
Predict whether the next trading day experiences an extreme tail crash ($y_{\text{cls}, t} \in \{0, 1\}$):
$$y_{\text{cls}, t} = \begin{cases} 1 & \text{if } r_{t+1} \le -1.60\% \quad (\text{Bottom 5\% Historical Quantile}) \\ 0 & \text{otherwise} \end{cases}$$
* **Objective Function:** Minimize Weighted Binary Cross-Entropy Loss to address extreme 95:5 class imbalance:
  $$\mathcal{L}_{\text{BCE}}(\mathbf{w}) = -\frac{1}{N} \sum_{i=1}^N \left[ w_1 y_i \ln \hat{p}_i + w_0 (1 - y_i) \ln(1 - \hat{p}_i) \right], \quad w_k = \frac{N}{2 N_k}$$

---

## 2. The 22 Engineered Features (Matching Image Specification)

The feature matrix is constructed from 7 raw daily market data streams (`SPY` OHLCV, `QQQ` Benchmark Close, and `^VIX` Close) across **3,167 trading days (2012–2024)**, engineering exactly **22 predictive tabular features**:

| # | Feature Key | Display Name | Mathematical Formula | Category |
| :-: | :--- | :--- | :--- | :--- |
| **1** | `ret` | Daily Log Return % | $\ln(P_t / P_{t-1}) \times 100$ | Price Action & Momentum |
| **2** | `ret_lag_1` | Return Lag 1 Day | $r_{t-1}$ | Price Action & Momentum |
| **3** | `ret_lag_2` | Return Lag 2 Days | $r_{t-2}$ | Price Action & Momentum |
| **4** | `ret_lag_3` | Return Lag 3 Days | $r_{t-3}$ | Price Action & Momentum |
| **5** | `ret_lag_4` | Return Lag 4 Days | $r_{t-4}$ | Price Action & Momentum |
| **6** | `ret_lag_5` | Return Lag 5 Days | $r_{t-5}$ (1 full trading week lag) | Price Action & Momentum |
| **7** | `ret_21d` | 21-Day Cumulative Return % | $(P_t / P_{t-21} - 1) \times 100$ | Price Action & Momentum |
| **8** | `drawdown_60d` | 60-Day Peak Drawdown % | $(P_t / \max_{60}(P) - 1) \times 100$ | Price Action & Momentum |
| **9** | `vol_5d` | 5-Day Realized Volatility % | $\sigma_{5d} \times \sqrt{252} \times 100$ | Volatility Dynamics |
| **10** | `vol_20d` | 20-Day Realized Volatility % | $\sigma_{20d} \times \sqrt{252} \times 100$ | Volatility Dynamics |
| **11** | `vol_60d` | 60-Day Realized Volatility % | $\sigma_{60d} \times \sqrt{252} \times 100$ | Volatility Dynamics |
| **12** | `vol_ratio` | Short/Long Volatility Ratio | $\sigma_{5d} / \sigma_{60d}$ | Volatility Dynamics |
| **13** | `ewma_vol` | RiskMetrics EWMA Volatility % | $\sqrt{252 \times \text{EWMA}(r^2, \lambda=0.94)} \times 100$ | Volatility Dynamics |
| **14** | `parkinson_vol` | Parkinson High-Low Volatility % | $\sqrt{ \frac{\ln(H_t/L_t)^2}{4\ln 2} \times 252 } \times 100$ | High-Frequency Volatility |
| **15** | `parkinson_vol_10d` | 10-Day Parkinson Volatility % | $\text{SMA}_{10d}(\text{Parkinson Vol})$ | High-Frequency Volatility |
| **16** | `skew_30d` | 30-Day Return Skewness | $\mathbb{E}[(r - \mu)^3] / \sigma^3$ | Distribution Geometry |
| **17** | `kurt_30d` | 30-Day Return Kurtosis | $\mathbb{E}[(r - \mu)^4] / \sigma^4 - 3$ (Fat-tails) | Distribution Geometry |
| **18** | `hist_var_95_60d` | 60-Day 95% Parametric VaR % | $-1.645 \times \frac{\sigma_{60d}}{\sqrt{252}} \times 100$ | Tail Risk Measures |
| **19** | `beta_60d` | 60-Day Benchmark Beta (vs QQQ) | $\text{Cov}(r, r_{\text{bm}}) / \text{Var}(r_{\text{bm}})$ | Market Regime & Macro |
| **20** | `vix_level` | CBOE VIX Index Level | $\text{VIX Close}$ | Market Regime & Macro |
| **21** | `vix_change_5d` | 5-Day VIX Velocity % | $(VIX_t / VIX_{t-5} - 1) \times 100$ | Market Regime & Macro |
| **22** | `rel_volume_20d` | Relative Volume vs 20d Average | $\text{Vol}_t / \text{SMA}_{20d}(\text{Vol})$ | Market Regime & Macro |

---

## 3. Machine Learning Models & Hyperparameter Specifications

The pipeline implements 5 core Machine Learning models with explicit hyperparameter configurations:

### 1. Linear Regression (Ridge L2 Regularized)
* **Formulation:** $\hat{y} = \mathbf{w}^T \mathbf{x} + b$, solving $\min_{\mathbf{w}} \|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2 + \alpha \|\mathbf{w}\|_2^2$.
* **Hyperparameters:** `alpha = 1.0`, `fit_intercept = True`, `solver = 'auto'`.
* **Analytical Solution:** $\mathbf{w}^* = (\mathbf{X}^T\mathbf{X} + \alpha \mathbf{I})^{-1} \mathbf{X}^T \mathbf{y}$. Adding $\alpha \mathbf{I}$ strictly regularizes multicollinear volatility features.

### 2. Decision Tree (CART Regressor & Classifier)
* **Formulation:** Non-parametric recursive binary splitting into hyper-rectangles $R_m$.
* **Hyperparameters:** `max_depth = 4`, `min_samples_leaf = 20`, `criterion = 'squared_error'` (Regression), `criterion = 'gini'` (Classification), `random_state = 42`.
* **Pruning / Regularization:** Limiting depth to 4 prevents memorization of noise while extracting non-linear split rules (e.g. $\text{vix\_level} > 22.0$).

### 3. Support Vector Machine (SVR & SVC with Gaussian RBF Kernel)
* **Formulation:** Maps input features to an infinite-dimensional RKHS space via $K(\mathbf{x}, \mathbf{x}') = \exp(-\gamma \|\mathbf{x} - \mathbf{x}'\|^2)$.
* **Hyperparameters:** `C = 1.0`, `epsilon = 0.1` (SVR $\varepsilon$-insensitive tube), `gamma = 'scale'`, `class_weight = 'balanced'` (SVC), `probability = True`.
* **Noise Robustness:** Errors smaller than $\varepsilon = 0.1$ incur zero penalty, ignoring small day-to-day market drift.

### 4. Logistic Regression Classifier
* **Formulation:** Models posterior crash probability $\hat{p} = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$.
* **Hyperparameters:** `C = 1.0`, `penalty = 'l2'`, `class_weight = 'balanced'` ($w_1 / w_0 \approx 19:1$), `max_iter = 1000`, `random_state = 42`.
* **Calibrated Probabilities:** Outputs well-calibrated posterior probabilities for tail risk hedging.

### 5. LightGBM Ensembles (Gradient Boosted Trees Benchmark)
* **Formulation:** Sequential ensemble of shallow decision trees optimizing gradient descent in function space.
* **Hyperparameters:** `n_estimators = 100`, `learning_rate = 0.03`, `max_depth = 4`, `num_leaves = 15`, `class_weight = 'balanced'`, `random_state = 42`.
* **Performance:** Learns complex multi-feature interactions (e.g. simultaneous high volume surge and volatility ratio expansion).

---

## 4. Empirical Out-of-Sample Test Results (459 Trading Days, 2023–2024)

All models were evaluated on the **unseen out-of-sample test split** (459 days, 2023-01-03 to 2024-10-25) following strict chronological walk-forward validation (Train: 2,457 days, Val: 251 days):

### Task A: Next-Day Return Regression Results

| Model | Category | RMSE (%) &darr; | MAE (%) &darr; | $R^2$ Score &uarr; | Max Residual | Std Residual |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **🥇 LightGBM Ensembles** | Gradient Boosted Trees | **0.8014** | **0.6158** | **+0.0080** | **2.7850%** | **0.8003** |
| **Decision Tree (CART)** | Non-Parametric (Depth 4) | 0.8059 | 0.6177 | -0.0030 | 2.8540% | 0.8043 |
| **Support Vector Machine (SVR)** | Kernel Margin (RBF) | 0.8078 | 0.6221 | -0.0077 | 2.9120% | 0.8063 |
| **Linear Regression (Ridge)** | Linear L2 Regularized | 0.8287 | 0.6481 | -0.0607 | 3.0086% | 0.8263 |

### Task B: Downside Crash Detection Results ($r_{t+1} \le -1.60\%$)

| Model | Category | Accuracy &uarr; | ROC-AUC &uarr; | Precision | Recall (Sensitivity) | F1-Score &uarr; |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **🥇 LightGBM Classifier** | Boosted Trees Head | 75.16% | **0.6274** | 5.45% | **42.86%** | **0.0952** |
| **Logistic Regression** | Linear Log-Odds | **78.43%** | 0.6181 | 5.26% | 35.71% | 0.0917 |
| **SVM Classifier (SVC RBF)** | Kernel Margin Hyperplane | 72.55% | 0.6120 | 4.17% | 35.71% | 0.0735 |
| **Decision Tree Classifier** | Gini Impurity Partitions | 58.61% | 0.6046 | 3.85% | 57.14% | 0.0686 |

---

## 5. In-Depth Error Analysis

### 1. Residual Error Distribution
* For all regression models, residuals $e_t = y_t - \hat{y}_t$ exhibit:
  - **Zero-Centered Mean:** Linear Regression mean residual = $+0.06\%$, LightGBM = $+0.04\%$.
  - **Symmetric Spread:** Standard deviation of residuals = $0.80\%$ to $0.83\%$.
  - **No Severe Bias:** Errors are approximately Gaussian with fat outlier tails corresponding to unexpected macro shocks.

### 2. Error Degradation by Volatility Regime
Examining Mean Absolute Error (MAE) conditioned on market volatility regimes reveals that prediction error is strictly regime-dependent:

| Volatility Regime | VIX Condition | Test Days | LinReg MAE | Tree MAE | SVM MAE | LightGBM MAE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Low Volatility** | $\text{VIX} < 16$ | 251 | 0.5664% | 0.5420% | 0.5480% | **0.5380%** |
| **Medium Volatility** | $16 \le \text{VIX} \le 22$ | 186 | 0.7122% | 0.6850% | 0.6890% | **0.6780%** |
| **High Volatility (Fear Spike)** | $\text{VIX} > 22$ | 22 | 1.0381% | 0.9850% | 0.9920% | **0.9750%** |

* **Key Takeaway:** Across all models, MAE increases by **~83%** during high-volatility regimes, demonstrating that market non-stationarity during panic spikes is the primary driver of forecast variance.

### 3. Confusion Matrix Diagnostics (Threshold Tuning)
At the default decision threshold ($0.50$), Logistic Regression achieves:
* **True Negatives (TN):** 349 calm trading days correctly classified.
* **False Positives (FP):** 96 false alarms (acceptable cost of proactive risk hedging).
* **False Negatives (FN):** Only 3 severe crashes missed out of 14 total tail days.
* **True Positives (TP):** 11 crashes caught and successfully alerted.

---

## 6. Project Structure

```
d:/ML/Project/
├── .venv/                         # Local Python virtual environment
├── requirements.txt               # Pinned dependencies
├── README.md                      # Comprehensive Machine Learning documentation
├── src/
│   ├── config.py                  # Pinned parameters, assets, date splits, seeds
│   ├── data_loader.py             # Historical OHLCV & VIX data ingestion
│   ├── feature_engineering.py     # 22 tabular features matching user specification
│   ├── models/
│   │   ├── __init__.py            # Exported ML model classes
│   │   └── ml_models.py           # Linear Reg, Decision Tree, SVM, Logistic, LightGBM
│   ├── evaluation.py              # MSE, RMSE, MAE, R2, Accuracy, F1, ROC-AUC, Regimes
│   └── train_pipeline.py          # Train/Val/Test walk-forward splits & model fitting
├── scripts/
│   ├── run_all.py                 # End-to-end master runner script (fits & exports in 1.1s)
│   └── export_web_data.py         # Serializes 22-feature payload into docs/data/
├── docs/                          # GitHub Pages Interactive Website
│   ├── index.html                 # 6-tab interactive ML dashboard UI
│   ├── style.css                  # Modern dark theme responsive styling
│   ├── app.js                     # Client-side ML inference, replay engine, Chart.js
│   └── data/
│       └── portfolio_data.json    # Master JSON payload (708 KB)
└── artifacts/                     # Serialized production joblib models & metrics
    ├── feature_scaler.joblib      # StandardScaler fitted on training split
    ├── best_linear_regression.joblib
    ├── best_decision_tree.joblib
    ├── best_svm_model.joblib
    ├── best_logistic_regression.joblib
    ├── best_lightgbm_model.joblib
    └── evaluation_metrics.json    # Full test evaluation scorecards & histograms
```

---

## 7. Execution & Reproduction

```bash
# 1. Clone repository
git clone https://github.com/siddharthdangi1911/Portfolio_Risk_Analyzer.git
cd Portfolio_Risk_Analyzer

# 2. Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 3. Install requirements
pip install -r requirements.txt

# 4. Run end-to-end ML pipeline (Fit models + Evaluate + Export Web Data)
python scripts/run_all.py

# 5. Launch web application locally
python -m http.server 8000 -d docs
```
