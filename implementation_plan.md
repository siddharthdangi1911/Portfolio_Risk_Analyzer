# Implementation Plan: Human-Centric Machine Learning Inference Engine

**Course:** ES 335 — Machine Learning (Undergraduate Track)  
**Project:** 1-Day Market Risk & Return Prediction Engine  
**Team:** Siddharth Dangi (25110300) & Aryan Vatsa Chauhan (25110059)  
**Focus:** Strictly Inference — From Intuitive Human Inputs to Desired AI Results  

---

## 1. Executive Summary & Design Philosophy

### The Problem with Traditional Financial ML Inference
Traditional financial risk systems fail ordinary humans because they demand esoteric inputs:
* *"Enter the 60-day Garman-Klass intraday range estimator"*
* *"Enter the 21-day rolling cross-asset beta vs benchmark"*
* *"Enter the 4th standardized central moment (excess kurtosis)"*

No human outside quantitative finance knows these numbers off the top of their head. Furthermore, traditional outputs (`VaR_0.05 = -0.0248`, `Kupiec LR = 1.42`) mean nothing to everyday investors or students.

### Our Solution: Human-Centric Inference Architecture
Our inference engine acts as an **intelligent abstraction layer**:
1. It asks **4 simple, qualitative questions** that any person can answer using everyday observations (e.g. *"Did stocks jump or drop today?"*, *"How nervous is the market?"*).
2. It mathematically maps these intuitive responses into a complete, standardized **22-feature vector** ($\mathbf{x} \in \mathbb{R}^{22}$) using 10-year empirical baseline distributions.
3. It passes this vector through **5 fitted Machine Learning models** (Linear Regression, Decision Tree, Support Vector Machine, Logistic Regression, LightGBM).
4. It synthesizes the model outputs into **plain-English, human-understandable verdicts**:
   - 🟢 / 🟡 / 🔴 **Visual Risk Status** (Low Risk / Moderate Caution / High Crash Risk)
   - **Expected Price Direction & Percentage Move**
   - **Probability of an Extreme Crash (Drop $\le -1.60\%$)**
   - **A Plain-English Explanation of *Why***
   - **5-Model Consensus Breakdown**

```
┌────────────────────────────────────────────────────────┐
│               1. HUMAN-FRIENDLY INPUTS                 │
│  • Today's Move: -2.0% Selloff                         │
│  • Market Fear: High Fear (VIX 28)                     │
│  • Trading Volume: Surge Volume                        │
│  • Recent 1-Month Trend: Bear Market Drop              │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│            2. FEATURE RECONSTRUCTION ENGINE            │
│  • Injects baseline empirical medians for unobserved   │
│    technical features (EWMA vol, beta, 5d lags)        │
│  • Constructs standardized feature vector z in R^22    │
│    using fitted StandardScaler: z = (x - mu) / sigma   │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│             3. MULTI-MODEL PARALLEL INFERENCE          │
│  • Linear Regression (Ridge): Baseline drift           │
│  • Decision Tree (CART): Non-linear regime split       │
│  • Support Vector Machine: Non-linear RBF margin       │
│  • Logistic Regression: Calibrated crash probability   │
│  • LightGBM Ensembles: Boosted non-linear consensus    │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│             4. PLAIN-ENGLISH SYNTHESIS ENGINE          │
│  • Risk Rating: 🔴 HIGH CRASH RISK                     │
│  • Expected Direction: 📉 Downward Drop (-0.40%)       │
│  • Crash Probability: 80% chance of drop <= -1.60%     │
│  • Action Advice: Proactive downside hedging suggested │
│  • Human Explanation: "Because VIX fear is high (28)   │
│    and volume surged during a selloff, models detect   │
│    high tail risk for tomorrow."                       │
└────────────────────────────────────────────────────────┘
```

---

## 2. Input Specification: 4 Intuitive Questions

Instead of forcing users to compute 22 statistical formulas, the user provides 4 intuitive inputs:

| # | Question for Human | Input Mechanism | Accepted Range / Options | Default Value |
| :-: | :--- | :--- | :--- | :--- |
| **Q1** | **How did the stock market move today?** | Continuous Slider or Buttons | `-4.0%` (Steep Selloff) to `+4.0%` (Sharp Rally) | `0.00%` (Flat) |
| **Q2** | **How nervous / scared is the market? (Fear Index)** | Continuous Slider or Buttons | `10.0` (Very Calm) to `45.0+` (Severe Panic) | `16.0` (Normal) |
| **Q3** | **What was today's trading volume activity?** | 4 Clickable Pills | `Quiet` (0.75x), `Normal` (1.00x), `Heavy` (1.50x), `Surge` (2.20x) | `Normal` (1.00x) |
| **Q4** | **What has the market's trend been over the past month?** | 4 Clickable Pills | `Bear Drop`, `Sideways / Flat`, `Mild Uptrend`, `Strong Bull` | `Sideways / Flat` |

### Clickable Scenario Presets
To make inference instantaneous, 4 pre-packaged scenarios are provided:
1. ☕ **Calm Market:** Today $+0.5\%$, VIX $12.5$, Volume Quiet, Trend Mild Uptrend.
2. 📊 **Typical / Median Day:** Today $0.0\%$, VIX $16.0$, Volume Normal, Trend Sideways.
3. 🚀 **Strong Bull Rally:** Today $+2.4\%$, VIX $14.0$, Volume Heavy, Trend Strong Bull.
4. 🚨 **Panic Crisis (e.g. SVB shock / COVID):** Today $-2.8\%$, VIX $32.0$, Volume Surge, Trend Bear Drop.

---

## 3. Mathematical Mapping: From Human Inputs to $\mathbf{x} \in \mathbb{R}^{22}$

The 22 features expected by the trained ML models are constructed as follows:

### Step 1: Base Anchor
Start with the empirical mean vector $\boldsymbol{\mu}_{\text{train}} \in \mathbb{R}^{22}$ computed over 2,457 historical training trading days (2012–2021):
$$\mathbf{x}_{\text{base}} = \left[ \mu_{\text{ret}}, \mu_{\text{lag1}}, \dots, \mu_{\text{vix}}, \mu_{\text{vol}} \right]$$

### Step 2: Dynamic Feature Overwrites
Translate human inputs into the corresponding mathematical features:
1. **Daily Return & Autoregressive Lags:**
   $$x_{\text{ret}} = \frac{\text{Today Return \%}}{100.0}$$
   $$x_{\text{ret\_lag\_1}} = x_{\text{ret}} \times 0.70, \quad x_{\text{ret\_lag\_2}} = x_{\text{ret}} \times 0.40$$
2. **CBOE VIX Level:**
   $$x_{\text{vix\_level}} = \text{VIX Input}$$
3. **Volatility Proxy & Short/Long Volatility Ratio:**
   $$\sigma_{5d} \approx \text{VIX} \times 0.80 \times \text{Volume Factor}$$
   $$x_{\text{vol\_5d}} = \sigma_{5d}, \quad x_{\text{parkinson\_vol}} = \sigma_{5d} \times 0.92$$
   $$x_{\text{vol\_ratio}} = \frac{\sigma_{5d}}{\sigma_{60d, \text{base}}}$$
4. **Volume Factor:**
   $$x_{\text{rel\_volume\_20d}} = \text{Volume Factor} \in \{0.75, 1.00, 1.50, 2.20\}$$
5. **1-Month Trend Conditioning (Drawdown & Moments):**
   - If `Strong Bull`: $x_{\text{ret\_21d}} = +4.5\%$, $x_{\text{drawdown\_60d}} = -0.5\%$, $x_{\text{skew\_30d}} = +0.40$, $x_{\text{kurt\_30d}} = 0.50$.
   - If `Sideways`: $x_{\text{ret\_21d}} = +0.2\%$, $x_{\text{drawdown\_60d}} = -2.0\%$, $x_{\text{skew\_30d}} = -0.10$, $x_{\text{kurt\_30d}} = 1.10$.
   - If `Bear Drop`: $x_{\text{ret\_21d}} = -6.5\%$, $x_{\text{drawdown\_60d}} = -12.0\%$, $x_{\text{skew\_30d}} = -1.20$, $x_{\text{kurt\_30d}} = 4.50$.

### Step 3: Standardization
Standardize the vector using the production `StandardScaler` fitted on the training split:
$$z_j = \frac{x_j - \mu_j}{\sigma_j}, \quad \forall j \in \{1, \dots, 22\}$$

---

## 4. Multi-Model Inference Engine

Once $\mathbf{z} \in \mathbb{R}^{22}$ is formed, inference executes across all 5 models:

### Model 1: Linear Regression (Ridge L2)
* **Mathematical Operation:** Matrix inner product with learned regularized weights:
  $$\hat{y}_{\text{lin}} = \mathbf{w}_{\text{Ridge}}^T \mathbf{z} + b_{\text{Ridge}}$$
* **Role:** Estimates baseline linear drift of expected returns.

### Model 2: Decision Tree (CART)
* **Mathematical Operation:** Traverse binary decision nodes based on threshold checks ($z_j \le s_{j, k}$):
  $$\hat{y}_{\text{dt}} = \bar{y}_{R_m}, \quad \hat{p}_{\text{dt}} = \frac{N_{\text{crash}, R_m}}{N_{R_m}}$$
* **Role:** Captures sharp, non-linear regime splits (e.g. VIX $> 22.0$ with high volume).

### Model 3: Support Vector Machine (SVR & SVC with RBF Kernel)
* **Mathematical Operation:** Dual evaluation across support vectors $\mathbf{s}_k$:
  $$\hat{y}_{\text{svm}} = \sum_{k=1}^{N_{\text{SV}}} \alpha_k \exp\left(-\gamma \|\mathbf{z} - \mathbf{s}_k\|^2\right) + b$$
* **Role:** Evaluates robust margins that ignore small daily noise ($\varepsilon = 0.1$).

### Model 4: Logistic Regression Classifier
* **Mathematical Operation:** Sigmoid activation over log-odds logit:
  $$\text{Logit} = \mathbf{w}_{\text{Log}}^T \mathbf{z} + b_{\text{Log}}$$
  $$\hat{p}_{\text{crash}} = \sigma(\text{Logit}) = \frac{1}{1 + e^{-\text{Logit}}}$$
* **Role:** Produces a calibrated posterior probability of a bottom 5% tail crash.

### Model 5: LightGBM Ensembles (Gradient Boosted Trees Benchmark)
* **Mathematical Operation:** Sum of leaf scores across 100 boosted regression and classification trees:
  $$\hat{y}_{\text{lgb}} = \sum_{m=1}^M f_m(\mathbf{z}), \quad \hat{p}_{\text{lgb}} = \sigma\left(\sum_{m=1}^M g_m(\mathbf{z})\right)$$
* **Role:** Highest out-of-sample accuracy (AUC: 0.627), synthesizing multi-feature interactions.

---

## 5. Plain-English Output Synthesis

To make the AI report immediately understandable to any human, the raw model numbers are synthesized into 4 clear outputs:

### 1. Risk Status Badge
* **Criteria:**
  - If $\hat{p}_{\text{crash}} \ge 50\%$ OR $\text{VIX} \ge 25.0$ &rarr; 🔴 **HIGH CRASH RISK**
  - If $\hat{p}_{\text{crash}} \ge 25\%$ OR $\text{VIX} \ge 19.0$ &rarr; 🟡 **MODERATE RISK (CAUTION)**
  - Otherwise &rarr; 🟢 **LOW RISK (SAFE)**

### 2. Expected Market Direction
* **Criteria:**
  - If $\text{Consensus Return} \ge +0.10\%$ &rarr; 📈 **Expected Upward Move**
  - If $\text{Consensus Return} \le -0.10\%$ &rarr; 📉 **Expected Downward Drop**
  - Otherwise &rarr; ➡️ **Flat / Muted Drift**

### 3. Crash Probability
* **Display:** e.g. `28.5% chance of severe drop (≤ -1.60%)`.

### 4. Automated Plain-English Rationale ("Why?")
A natural language synthesis rule generates sentences based on active triggers:
* If VIX is elevated: *"Because market fear is high (VIX 28), signaling aggressive defensive hedging..."*
* If return dropped sharply: *"...and today's -2.0% selloff creates negative short-term momentum..."*
* If volume surged: *"...reinforced by surge trading volume indicating institutional participation..."*
* Verdict: *"4 out of 5 Machine Learning models detect high downside crash risk for tomorrow."*

---

## 6. Execution Implementations

The inference engine is implemented in two zero-friction formats:

### Implementation A: Python CLI / Module ([`src/inference.py`](file:///d:/ML/Project/src/inference.py))
Any user or developer can run inference directly from the command line:

```bash
# Preset scenarios:
python src/inference.py --preset calm
python src/inference.py --preset panic

# Custom human inputs:
python src/inference.py --return-pct -2.5 --vix 31.0 --volume surge --trend crash
```

#### Example Terminal Output:
```
=================================================================
      🤖 MACHINE LEARNING NEXT-DAY RISK & RETURN REPORT      
=================================================================
  RISK STATUS:           🔴 HIGH CRASH RISK
  PREDICTED DIRECTION:   📉 Expected Downward Drop (-0.40%)
  CRASH PROBABILITY:     80.2% chance of drop ≤ -1.60%
  RECOMMENDED ACTION:    Market stress detected. High probability of downside volatility tomorrow.
-----------------------------------------------------------------
  EXPLANATION FOR HUMANS:
  AI Diagnosis: CBOE VIX fear index is elevated at 31.0, signaling
  defensive institutional hedging; today's steep -2.5% selloff
  triggers negative short-term momentum; realized volatility is
  running well above historical norms; trading volume surged,
  confirming strong institutional conviction. Overall, 5 Machine
  Learning models estimate an 80% chance of a catastrophic drop
  (≤ -1.60%) and expect next-day return around -0.40%.
-----------------------------------------------------------------
  5-MODEL CONSENSUS BREAKDOWN:
    • Linear Regression (Ridge)    -> Pred Return: -2.19%
    • Decision Tree (CART)         -> Pred Return: +0.05%, Crash Prob: 81.7%
    • Support Vector Machine (RBF) -> Pred Return: +0.20%, Crash Prob: 5.1%
    • Logistic Regression          -> Crash Prob: 99.4%
    • LightGBM Benchmark           -> Pred Return: -0.86%, Crash Prob: 80.2%
=================================================================
```

### Implementation B: Interactive Web Application ([`docs/index.html`](file:///d:/ML/Project/docs/index.html) & [`docs/app.js`](file:///d:/ML/Project/docs/app.js))
In the browser, the inference logic is embedded in pure client-side JavaScript (`initHumanInference()`):
* Users move intuitive sliders or click scenario buttons.
* Models re-evaluate **in real-time (< 1 millisecond)** without needing a backend server.
* Works 100% offline and on GitHub Pages with embedded model weights and scaler parameters.

---

## 7. Verification & Test Plan

| Scenario | Inputs | Expected Output | Status |
| :--- | :--- | :--- | :---: |
| **Calm Bull Market** | Today: $+0.5\%$, VIX: $12.5$, Vol: Quiet, Trend: Uptrend | 🟢 **LOW RISK (SAFE)**, Return: $\approx +0.10\%$, Crash Prob: $< 25\%$ | **PASSED** |
| **Typical Trading Day** | Today: $0.0\%$, VIX: $16.0$, Vol: Normal, Trend: Sideways | 🟢/🟡 **LOW/MODERATE**, Return: $\approx +0.04\%$, Crash Prob: $\approx 30\%$ | **PASSED** |
| **Market Panic (SVB Crisis)** | Today: $-2.8\%$, VIX: $32.0$, Vol: Surge, Trend: Bear Drop | 🔴 **HIGH CRASH RISK**, Return: $\le -0.40\%$, Crash Prob: $> 75\%$ | **PASSED** |
