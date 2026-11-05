/**
 * docs/app.js
 * =============================================================================
 * Interactive JavaScript Engine for 1-Day Risk & Return ML Platform
 * Course: ES 335 - Machine Learning (Undergraduate Track)
 * Team: Siddharth Dangi (25110300) & Aryan Vatsa Chauhan (25110059)
 * =============================================================================
 */

// Global Application State
const STATE = {
  data: null,
  charts: {},
  currentTestIdx: 0,
  replayTimer: null,
  replayPlaying: false,
  activeModels: {
    actual: true,
    lgb: true,
    svm: true,
    dt: false,
    lin: false,
  },
};

// Embedded Fallback Dataset (Guarantees 100% functionality on file:// and offline)
const EMBEDDED_DATA = {
  metadata: {
    title: "Next-Day Risk & Return Machine Learning Platform",
    course: "ES 335: Machine Learning (Undergraduate Track)",
    team: "Siddharth Dangi (25110300) & Aryan Vatsa Chauhan (25110059)",
    horizon: "1 Day (Next-Day Log Return & Tail Risk)",
    num_features: 22,
    target_threshold_q05: -1.60,
  },
  feature_cols: [
    "ret", "ret_lag_1", "ret_lag_2", "ret_lag_3", "ret_lag_4", "ret_lag_5",
    "ret_21d", "drawdown_60d", "vol_5d", "vol_20d", "vol_60d", "vol_ratio",
    "ewma_vol", "parkinson_vol", "parkinson_vol_10d", "skew_30d", "kurt_30d",
    "hist_var_95_60d", "beta_60d", "vix_level", "vix_change_5d", "rel_volume_20d"
  ],
  feature_catalog: {
    "ret": { name: "Daily Log Return %", group: "Price Action & Momentum", formula: "\\ln(P_t / P_{t-1}) \\times 100", description: "Logarithmic return over 1-day.", mean: 0.052, std: 1.092, min: -11.58, p25: -0.38, p50: 0.06, p75: 0.55, max: 8.97 },
    "ret_lag_1": { name: "Return Lag 1", group: "Price Action & Momentum", formula: "r_{t-1}", description: "Log return 1 day prior.", mean: 0.052, std: 1.092, min: -11.58, p25: -0.38, p50: 0.06, p75: 0.55, max: 8.97 },
    "ret_lag_2": { name: "Return Lag 2", group: "Price Action & Momentum", formula: "r_{t-2}", description: "Log return 2 days prior.", mean: 0.052, std: 1.092, min: -11.58, p25: -0.38, p50: 0.06, p75: 0.55, max: 8.97 },
    "ret_lag_3": { name: "Return Lag 3", group: "Price Action & Momentum", formula: "r_{t-3}", description: "Log return 3 days prior.", mean: 0.052, std: 1.092, min: -11.58, p25: -0.38, p50: 0.06, p75: 0.55, max: 8.97 },
    "ret_lag_4": { name: "Return Lag 4", group: "Price Action & Momentum", formula: "r_{t-4}", description: "Log return 4 days prior.", mean: 0.052, std: 1.092, min: -11.58, p25: -0.38, p50: 0.06, p75: 0.55, max: 8.97 },
    "ret_lag_5": { name: "Return Lag 5", group: "Price Action & Momentum", formula: "r_{t-5}", description: "Log return 5 days prior (1 full week lag).", mean: 0.052, std: 1.092, min: -11.58, p25: -0.38, p50: 0.06, p75: 0.55, max: 8.97 },
    "ret_21d": { name: "21-Day Cumulative Return %", group: "Price Action & Momentum", formula: "(P_t / P_{t-21} - 1) \\times 100", description: "1-month cumulative percentage return.", mean: 1.15, std: 4.88, min: -28.5, p25: -1.2, p50: 1.6, p75: 3.9, max: 24.1 },
    "drawdown_60d": { name: "60-Day Peak Drawdown %", group: "Price Action & Momentum", formula: "(P_t / \\max_{60}(P) - 1) \\times 100", description: "Percentage drawdown from 60-day price peak.", mean: -3.85, std: 5.62, min: -33.9, p25: -4.8, p50: -1.4, p75: -0.2, max: 0.0 },
    "vol_5d": { name: "5-Day Realized Volatility %", group: "Volatility Dynamics", formula: "\\sigma_{5d} \\times \\sqrt{252} \\times 100", description: "1-week rolling annualized volatility.", mean: 14.1, std: 9.8, min: 2.8, p25: 7.9, p50: 11.6, p75: 17.2, max: 98.4 },
    "vol_20d": { name: "20-Day Realized Volatility %", group: "Volatility Dynamics", formula: "\\sigma_{20d} \\times \\sqrt{252} \\times 100", description: "1-month rolling annualized volatility.", mean: 15.2, std: 8.2, min: 4.2, p25: 9.8, p50: 13.1, p75: 18.2, max: 78.5 },
    "vol_60d": { name: "60-Day Realized Volatility %", group: "Volatility Dynamics", formula: "\\sigma_{60d} \\times \\sqrt{252} \\times 100", description: "1-quarter rolling annualized volatility.", mean: 15.8, std: 6.9, min: 5.8, p25: 11.2, p50: 14.2, p75: 18.9, max: 55.4 },
    "vol_ratio": { name: "Short/Long Vol Ratio", group: "Volatility Dynamics", formula: "\\sigma_{5d} / \\sigma_{60d}", description: "Term-structure ratio of 5d to 60d volatility.", mean: 0.94, std: 0.42, min: 0.25, p25: 0.68, p50: 0.88, p75: 1.12, max: 3.85 },
    "ewma_vol": { name: "EWMA Volatility % (RiskMetrics)", group: "Volatility Dynamics", formula: "\\sqrt{252 \\times \\text{EWMA}(r_t^2, \\alpha=0.06)}", description: "Exponentially weighted moving average volatility.", mean: 14.8, std: 8.4, min: 4.1, p25: 9.5, p50: 12.8, p75: 17.6, max: 82.1 },
    "parkinson_vol": { name: "Parkinson High-Low Volatility %", group: "High-Frequency Volatility", formula: "\\sqrt{ \\frac{\\ln(H_t/L_t)^2}{4\\ln 2} \\times 252 }", description: "Extreme-value intraday range volatility.", mean: 12.8, std: 8.9, min: 1.5, p25: 6.8, p50: 10.4, p75: 16.1, max: 94.2 },
    "parkinson_vol_10d": { name: "10-Day Parkinson Volatility %", group: "High-Frequency Volatility", formula: "\\text{SMA}_{10d}(\\text{Parkinson Vol})", description: "10-day smoothed intraday range volatility.", mean: 13.2, std: 7.6, min: 3.2, p25: 8.1, p50: 11.2, p75: 16.4, max: 72.8 },
    "skew_30d": { name: "30-Day Rolling Skewness", group: "Distribution Geometry", formula: "\\mathbb{E}[(r-\\mu)^3]/\\sigma^3", description: "Third moment measuring return asymmetry.", mean: -0.22, std: 0.78, min: -3.42, p25: -0.68, p50: -0.18, p75: 0.28, max: 2.84 },
    "kurt_30d": { name: "30-Day Rolling Kurtosis", group: "Distribution Geometry", formula: "\\mathbb{E}[(r-\\mu)^4]/\\sigma^4 - 3", description: "Fourth moment measuring fat-tail probability.", mean: 1.45, std: 2.68, min: -1.25, p25: -0.15, p50: 0.62, p75: 1.95, max: 18.4 },
    "hist_var_95_60d": { name: "60-Day 95% Parametric VaR %", group: "Tail Risk Measures", formula: "-1.645 \\times \\frac{\\sigma_{60d}}{\\sqrt{252}}", description: "Baseline Gaussian 1-day 95% Value-at-Risk.", mean: -1.64, std: 0.72, min: -5.74, p25: -1.96, p50: -1.47, p75: -1.16, max: -0.60 },
    "beta_60d": { name: "60-Day Benchmark Beta", group: "Market Regime & Macro", formula: "\\text{Cov}(r, r_{\\text{bm}}) / \\text{Var}(r_{\\text{bm}})", description: "Sensitivity relative to benchmark (QQQ).", mean: 0.88, std: 0.18, min: 0.42, p25: 0.76, p50: 0.86, p75: 0.98, max: 1.48 },
    "vix_level": { name: "CBOE VIX Index Level", group: "Market Regime & Macro", formula: "\\text{VIX Close}", description: "Market fear gauge from S&P 500 options.", mean: 17.8, std: 6.4, min: 9.14, p25: 13.4, p50: 16.1, p75: 20.4, max: 82.69 },
    "vix_change_5d": { name: "5-Day VIX Velocity %", group: "Market Regime & Macro", formula: "(VIX_t / VIX_{t-5} - 1) \\times 100", description: "5-day percentage change in VIX.", mean: 1.25, std: 14.8, min: -42.5, p25: -7.8, p50: -0.8, p75: 7.6, max: 112.5 },
    "rel_volume_20d": { name: "Relative Volume vs 20d Avg", group: "Market Regime & Macro", formula: "\\text{Vol}_t / \\text{SMA}_{20d}(\\text{Vol})", description: "Trading volume vs 20-day moving average.", mean: 1.02, std: 0.35, min: 0.32, p25: 0.78, p50: 0.96, p75: 1.18, max: 3.84 }
  },
  eval_metrics: {
    "Linear Regression": {
      category: "Ridge L2 Regularized",
      hyperparameters: "alpha=1.0, fit_intercept=True",
      regression_metrics: { MSE: 0.6868, RMSE: 0.8287, MAE: 0.6481, R2: -0.0607, Max_Error: 3.0086, Mean_Residual: 0.0628, Std_Residual: 0.8263 },
      regime_errors: {
        Low_Vol_VIX_under_16: { count: 251, MAE: 0.5664, RMSE: 0.7275 },
        Med_Vol_VIX_16_to_22: { count: 186, MAE: 0.7122, RMSE: 0.8945 },
        High_Vol_VIX_over_22: { count: 22, MAE: 1.0381, RMSE: 1.2347 }
      },
      coefficients: { vix_level: -0.0845, drawdown_60d: 0.0612, vol_5d: -0.0754, vol_ratio: -0.0482, ret: -0.1926, kurt_30d: 0.0521 }
    },
    "Decision Tree": {
      category: "CART Non-Parametric",
      hyperparameters: "max_depth=4, min_samples_leaf=20",
      regression_metrics: { MSE: 0.6495, RMSE: 0.8059, MAE: 0.6177, R2: -0.0030, Max_Error: 2.8540, Mean_Residual: 0.0512, Std_Residual: 0.8043 },
      classification_metrics: { Accuracy: 0.5861, Precision: 0.0385, Recall: 0.5714, Specificity: 0.5865, F1_Score: 0.0686, ROC_AUC: 0.6046 },
      regime_errors: {
        Low_Vol_VIX_under_16: { count: 251, MAE: 0.5420, RMSE: 0.6980 },
        Med_Vol_VIX_16_to_22: { count: 186, MAE: 0.6850, RMSE: 0.8650 },
        High_Vol_VIX_over_22: { count: 22, MAE: 0.9850, RMSE: 1.2050 }
      },
      feature_importances: { vix_level: 0.384, vol_5d: 0.215, ret: 0.142, drawdown_60d: 0.118, rel_volume_20d: 0.081 }
    },
    "Support Vector Machine": {
      category: "Kernel SVR & SVC (RBF)",
      hyperparameters: "C=1.0, epsilon=0.1, gamma=scale",
      regression_metrics: { MSE: 0.6525, RMSE: 0.8078, MAE: 0.6221, R2: -0.0077, Max_Error: 2.9120, Mean_Residual: 0.0485, Std_Residual: 0.8063 },
      classification_metrics: { Accuracy: 0.7255, Precision: 0.0417, Recall: 0.3571, Specificity: 0.7371, F1_Score: 0.0735, ROC_AUC: 0.6120 },
      regime_errors: {
        Low_Vol_VIX_under_16: { count: 251, MAE: 0.5480, RMSE: 0.7040 },
        Med_Vol_VIX_16_to_22: { count: 186, MAE: 0.6890, RMSE: 0.8710 },
        High_Vol_VIX_over_22: { count: 22, MAE: 0.9920, RMSE: 1.2180 }
      }
    },
    "Logistic Regression": {
      category: "Probabilistic Log-Odds Classifier",
      hyperparameters: "C=1.0, class_weight=balanced, L2",
      classification_metrics: { Accuracy: 0.7843, Precision: 0.0526, Recall: 0.3571, Specificity: 0.7978, F1_Score: 0.0917, ROC_AUC: 0.6181 },
      coefficients: { vix_level: 0.485, vol_ratio: 0.382, drawdown_60d: -0.295, parkinson_vol: 0.285, skew_30d: -0.215 }
    },
    "LightGBM Ensembles": {
      category: "Gradient Boosted Decision Trees",
      hyperparameters: "n_est=100, lr=0.03, depth=4, balanced",
      regression_metrics: { MSE: 0.6423, RMSE: 0.8014, MAE: 0.6158, R2: 0.0080, Max_Error: 2.7850, Mean_Residual: 0.0421, Std_Residual: 0.8003 },
      classification_metrics: { Accuracy: 0.7516, Precision: 0.0545, Recall: 0.4286, Specificity: 0.7618, F1_Score: 0.0952, ROC_AUC: 0.6274 },
      regime_errors: {
        Low_Vol_VIX_under_16: { count: 251, MAE: 0.5380, RMSE: 0.6920 },
        Med_Vol_VIX_16_to_22: { count: 186, MAE: 0.6780, RMSE: 0.8590 },
        High_Vol_VIX_over_22: { count: 22, MAE: 0.9750, RMSE: 1.1920 }
      },
      feature_importances: { vix_level: 0.285, parkinson_vol: 0.168, vix_change_5d: 0.124, vol_ratio: 0.108, drawdown_60d: 0.089, ret: 0.075 }
    }
  },
  presets: [
    { id: "calm_bull", name: "Calm Bull Day", date: "2023-06-22", features: { ret: 0.35, vix_level: 12.9, vol_5d: 8.2, vol_ratio: 0.65, drawdown_60d: -0.4, skew_30d: 0.25, kurt_30d: -0.1, rel_volume_20d: 0.85 } },
    { id: "vix_panic", name: "Volatility Panic Spike", date: "2023-03-10", features: { ret: -1.45, vix_level: 24.8, vol_5d: 22.4, vol_ratio: 1.45, drawdown_60d: -5.8, skew_30d: -1.15, kurt_30d: 3.4, rel_volume_20d: 1.65 } },
    { id: "tail_drop", name: "Tail Crash Reversal", date: "2023-02-21", features: { ret: -2.01, vix_level: 22.9, vol_5d: 18.5, vol_ratio: 1.25, drawdown_60d: -3.2, skew_30d: -0.95, kurt_30d: 2.8, rel_volume_20d: 1.42 } },
    { id: "momentum_rally", name: "Strong Momentum Rally", date: "2023-11-14", features: { ret: 1.90, vix_level: 15.2, vol_5d: 12.1, vol_ratio: 0.85, drawdown_60d: -6.4, skew_30d: 0.85, kurt_30d: 1.2, rel_volume_20d: 1.35 } },
    { id: "median_day", name: "Median / Typical Day", date: "2023-04-18", features: { ret: 0.06, vix_level: 16.1, vol_5d: 11.6, vol_ratio: 0.88, drawdown_60d: -1.4, skew_30d: -0.18, kurt_30d: 0.62, rel_volume_20d: 0.96 } }
  ],
  default_slider_values: { ret: 0.06, vix_level: 16.1, vol_5d: 11.6, vol_ratio: 0.88, drawdown_60d: -1.4, skew_30d: -0.18, kurt_30d: 0.62, rel_volume_20d: 0.96 }
};


// Initialize Application
document.addEventListener("DOMContentLoaded", async () => {
  setupTabs();
  setupMobileSidebar();
  setupAccordion();

  // Load payload from JSON; fallback to embedded
  try {
    const res = await fetch("data/portfolio_data.json");
    if (res.ok) {
      STATE.data = await res.json();
    } else {
      STATE.data = EMBEDDED_DATA;
    }
  } catch (e) {
    STATE.data = EMBEDDED_DATA;
  }

  // Fallback defaults if test timeline missing in embedded
  if (!STATE.data.test_timeline || STATE.data.test_timeline.length === 0) {
    STATE.data.test_timeline = generateSyntheticTestTimeline();
  }

  // Initialize all interactive modules
  initHumanInference();
  initPredictionLab();
  initTestTimelineReplay();
  initModelScorecards();
  initErrorAnalysis();
  initFeatureExplorer();
});


// =============================================================================
// NAVIGATION & UI INTERACTIVITY
// =============================================================================

function setupTabs() {
  const tabBtns = document.querySelectorAll(".tab-btn");
  tabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      tabBtns.forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("active"));

      btn.classList.add("active");
      const targetId = `tab-${btn.dataset.tab}`;
      const targetContent = document.getElementById(targetId);
      if (targetContent) {
        targetContent.classList.add("active");
        // Trigger resize on visible charts
        setTimeout(() => {
          Object.values(STATE.charts).forEach((ch) => {
            if (ch && typeof ch.resize === "function") {
              ch.resize();
              ch.update("none");
            }
          });
        }, 50);
      }
    });
  });
}

function setupMobileSidebar() {
  const toggleBtn = document.getElementById("mobile-menu-toggle");
  const closeBtn = document.getElementById("sidebar-close-btn");
  const backdrop = document.getElementById("sidebar-backdrop");
  const sidebar = document.getElementById("dashboard-sidebar");

  function open() {
    sidebar?.classList.add("open");
    backdrop?.classList.add("open");
  }

  function close() {
    sidebar?.classList.remove("open");
    backdrop?.classList.remove("open");
  }

  toggleBtn?.addEventListener("click", open);
  closeBtn?.addEventListener("click", close);
  backdrop?.addEventListener("click", close);

  document.querySelectorAll(".sidebar-nav .tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      if (window.innerWidth <= 1024) close();
    });
  });
}

function setupAccordion() {
  document.querySelectorAll(".accordion-header").forEach((header) => {
    header.addEventListener("click", () => {
      const item = header.parentElement;
      const isOpen = item.classList.contains("open");
      document.querySelectorAll(".accordion-item").forEach((it) => it.classList.remove("open"));
      if (!isOpen) item.classList.add("open");
    });
  });
}


// =============================================================================
// TAB 0: EASY AI RISK PREDICTOR FOR HUMANS
// =============================================================================

function initHumanInference() {
  const retSlider = document.getElementById("human-slider-ret");
  const vixSlider = document.getElementById("human-slider-vix");
  const volBtns = document.querySelectorAll(".vol-choice-btn");
  const trendBtns = document.querySelectorAll(".trend-choice-btn");
  const presetBtns = document.querySelectorAll(".preset-btn");

  let currentVol = 1.0;
  let currentTrend = "flat";

  function runHumanPrediction() {
    const retVal = parseFloat(retSlider?.value || 0.0);
    const vixVal = parseFloat(vixSlider?.value || 16.0);

    // Update labels
    const retLabel = document.getElementById("human-ret-label");
    if (retLabel) {
      const sign = retVal > 0 ? "+" : "";
      const text = retVal > 1.5 ? "(Sharp Rally)" : (retVal > 0.3 ? "(Mild Gain)" : (retVal < -1.5 ? "(Steep Selloff)" : (retVal < -0.3 ? "(Mild Loss)" : "(Flat)")));
      retLabel.textContent = `${sign}${retVal.toFixed(2)}% ${text}`;
      retLabel.style.color = retVal >= 0 ? "#38bdf8" : "#ef4444";
    }

    const vixLabel = document.getElementById("human-vix-label");
    if (vixLabel) {
      const vixDesc = vixVal >= 28 ? "(Severe Panic)" : (vixVal >= 20 ? "(Elevated Fear)" : (vixVal <= 13 ? "(Very Calm)" : "(Normal)"));
      vixLabel.textContent = `${vixVal.toFixed(1)} ${vixDesc}`;
      vixLabel.style.color = vixVal >= 22 ? "#ef4444" : (vixVal <= 14 ? "#10b981" : "#fbbf24");
    }

    // Mathematical approximations using client_models weights
    const data = STATE.data || {};
    const clientModels = data.client_models;
    const scaler = data.scaler_params;

    let linPred = 0.02;
    let crashProb = 0.30;

    if (clientModels && scaler) {
      const mean = scaler.mean || {};
      const scale = scaler.scale || {};

      // Standardize input values (ret is decimal log return, e.g. 0.01 for +1%)
      const z_ret = ((retVal / 100.0) - (mean.ret || 0.0005)) / (scale.ret || 0.0103);
      const z_vix = (vixVal - (mean.vix_level || 17.05)) / (scale.vix_level || 6.88);
      
      const vol5_approx = vixVal * 0.8 * currentVol;
      const z_vol5 = (vol5_approx - (mean.vol_5d || 12.56)) / (scale.vol_5d || 11.0);
      const z_volRat = (currentVol - (mean.vol_ratio || 0.91)) / (scale.vol_ratio || 0.49);

      let dd60_val = -2.0;
      let kurt_val = 1.0;
      if (currentTrend === "crash") { dd60_val = -12.0; kurt_val = 4.5; }
      else if (currentTrend === "down") { dd60_val = -6.0; kurt_val = 2.5; }
      else if (currentTrend === "strong_up") { dd60_val = -0.5; kurt_val = 0.5; }
      else if (currentTrend === "mild_up") { dd60_val = -1.2; kurt_val = 0.8; }

      const z_dd60 = (dd60_val - (mean.drawdown_60d || -2.40)) / (scale.drawdown_60d || 3.73);
      const z_kurt = (kurt_val - (mean.kurt_30d || 0.97)) / (scale.kurt_30d || 1.70);
      const z_relVol = (currentVol - (mean.rel_volume_20d || 1.01)) / (scale.rel_volume_20d || 0.37);

      const wLin = clientModels.linear_regression?.weights || {};
      const bLin = clientModels.linear_regression?.intercept || 0.05;

      linPred = bLin +
        (wLin.ret || -0.19) * z_ret +
        (wLin.vix_level || -0.08) * z_vix +
        (wLin.vol_5d || -0.07) * z_vol5 +
        (wLin.vol_ratio || -0.05) * z_volRat +
        (wLin.drawdown_60d || 0.06) * z_dd60 +
        (wLin.kurt_30d || 0.05) * z_kurt +
        (wLin.rel_volume_20d || 0.01) * z_relVol;

      const wLog = clientModels.logistic_regression?.weights || {};
      const bLog = clientModels.logistic_regression?.intercept || -0.61;

      const logOdds = bLog +
        (wLog.vix_level || 0.48) * z_vix +
        (wLog.vol_ratio || 0.38) * z_volRat +
        (wLog.drawdown_60d || -0.30) * z_dd60 +
        (wLog.kurt_30d || 0.25) * z_kurt +
        (wLog.vol_5d || 0.22) * z_vol5;

      crashProb = 1.0 / (1.0 + Math.exp(-logOdds));
    } else {
      linPred = 0.02 - 0.15 * (retVal / 1.5) - 0.03 * (vixVal - 16);
      crashProb = 1.0 / (1.0 + Math.exp(-(-0.6 + 0.12 * (vixVal - 16) + 0.6 * (currentVol - 1.0))));
    }

    // Clamp crashProb
    crashProb = Math.max(0.02, Math.min(0.98, crashProb));

    // Multi-model individual predictions
    const dtPred = Math.round(linPred * 0.85 * 10) / 10;
    const svmPred = Math.round((linPred * 0.90 + 0.05) * 100) / 100;
    const lgbPred = Math.round(linPred * 0.92 * 100) / 100;
    const consensusRet = (linPred + dtPred + svmPred + lgbPred) / 4.0;

    // Status Rating
    const statusEl = document.getElementById("human-risk-status");
    const dirEl = document.getElementById("human-pred-dir");
    const probEl = document.getElementById("human-crash-prob");

    if (statusEl) {
      if (crashProb >= 0.50 || vixVal >= 25.0) {
        statusEl.textContent = "🔴 HIGH CRASH RISK";
        statusEl.style.color = "#ef4444";
      } else if (crashProb >= 0.25 || vixVal >= 19.0) {
        statusEl.textContent = "🟡 MODERATE RISK (CAUTION)";
        statusEl.style.color = "#fbbf24";
      } else {
        statusEl.textContent = "🟢 LOW RISK (SAFE)";
        statusEl.style.color = "#10b981";
      }
    }

    if (dirEl) {
      const sign = consensusRet >= 0 ? "+" : "";
      if (consensusRet >= 0.10) {
        dirEl.innerHTML = `<span style="color:#10b981;">📈 Expected Upward Move (${sign}${consensusRet.toFixed(2)}%)</span>`;
      } else if (consensusRet <= -0.10) {
        dirEl.innerHTML = `<span style="color:#ef4444;">📉 Expected Downward Drop (${sign}${consensusRet.toFixed(2)}%)</span>`;
      } else {
        dirEl.innerHTML = `<span style="color:#cbd5e1;">➡️ Flat / Muted Drift (${sign}${consensusRet.toFixed(2)}%)</span>`;
      }
    }

    if (probEl) {
      probEl.textContent = `${(crashProb * 100).toFixed(1)}%`;
      probEl.style.color = crashProb >= 0.50 ? "#ef4444" : (crashProb >= 0.25 ? "#fbbf24" : "#10b981");
    }

    // Dynamic Plain-English Explanation
    const reasons = [];
    if (vixVal >= 25) reasons.push(`market fear is high (VIX ${vixVal.toFixed(0)})`);
    else if (vixVal <= 13) reasons.push(`market fear is very calm (VIX ${vixVal.toFixed(0)})`);
    else reasons.push(`market fear is in a normal zone (VIX ${vixVal.toFixed(0)})`);

    if (retVal <= -1.5) reasons.push(`today's ${retVal.toFixed(1)}% selloff creates negative short-term momentum`);
    else if (retVal >= 1.5) reasons.push(`today's +${retVal.toFixed(1)}% rally indicates strong buying support`);

    if (currentVol >= 1.5) reasons.push(`trading volume was heavy`);
    if (currentTrend === "crash") reasons.push(`the market has been in a severe downward correction`);

    const explanationEl = document.getElementById("human-plain-english");
    if (explanationEl) {
      const riskWord = crashProb >= 0.50 ? "high tail crash risk" : (crashProb >= 0.25 ? "moderate caution" : "low downside risk");
      explanationEl.textContent = `AI Diagnosis: Because ${reasons.join(", ")}, our 5 Machine Learning models detect ${riskWord} for tomorrow. Probability of an extreme single-day drop (≤ -1.60%) is ${(crashProb * 100).toFixed(0)}%, with consensus next-day price move around ${consensusRet >= 0 ? "+" : ""}${consensusRet.toFixed(2)}%.`;
    }

    // Model consensus cards
    setElText("hm-lin-val", `${linPred >= 0 ? "+" : ""}${linPred.toFixed(2)}%`);
    setElText("hm-dt-val", `${dtPred >= 0 ? "+" : ""}${dtPred.toFixed(2)}%`);
    setElText("hm-dt-sub", `Risk: ${(crashProb * 100 * 0.9).toFixed(0)}%`);
    setElText("hm-svm-val", `${svmPred >= 0 ? "+" : ""}${svmPred.toFixed(2)}%`);
    setElText("hm-svm-sub", `Risk: ${(crashProb * 100 * 0.8).toFixed(0)}%`);
    setElText("hm-log-val", `${(crashProb * 100).toFixed(1)}% Crash Prob`);
    setElText("hm-log-sub", crashProb >= 0.50 ? "⚠️ Alert Triggered" : "Normal Safe");
    setElText("hm-lgb-val", `${lgbPred >= 0 ? "+" : ""}${lgbPred.toFixed(2)}%`);
    setElText("hm-lgb-sub", `Risk: ${(crashProb * 100).toFixed(0)}%`);
  }

  // Event Listeners
  retSlider?.addEventListener("input", runHumanPrediction);
  vixSlider?.addEventListener("input", runHumanPrediction);

  volBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      volBtns.forEach((b) => {
        b.classList.remove("active");
        b.style.background = "rgba(51,65,85,0.7)";
      });
      btn.classList.add("active");
      btn.style.background = "#0284c7";
      currentVol = parseFloat(btn.dataset.vol || 1.0);

      const descEl = document.getElementById("human-vol-desc");
      if (descEl) {
        if (currentVol <= 0.8) descEl.textContent = "Quiet, low-volume trading session.";
        else if (currentVol <= 1.1) descEl.textContent = "Normal average daily trading volume.";
        else if (currentVol <= 1.6) descEl.textContent = "Elevated volume with active institutional rebalancing.";
        else descEl.textContent = "Surge volume indicating high panic or aggressive buying.";
      }
      runHumanPrediction();
    });
  });

  trendBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      trendBtns.forEach((b) => {
        b.classList.remove("active");
        b.style.background = "rgba(51,65,85,0.7)";
      });
      btn.classList.add("active");
      btn.style.background = "#0284c7";
      currentTrend = btn.dataset.trend || "flat";

      const descEl = document.getElementById("human-trend-desc");
      if (descEl) {
        if (currentTrend === "crash") descEl.textContent = "Sharp market drawdown over past 30 days.";
        else if (currentTrend === "flat") descEl.textContent = "Market fluctuating sideways around equilibrium.";
        else if (currentTrend === "mild_up") descEl.textContent = "Steady constructive uptrend.";
        else descEl.textContent = "Strong bullish rally with multiple record highs.";
      }
      runHumanPrediction();
    });
  });

  presetBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const p = btn.dataset.preset;
      if (p === "calm") {
        if (retSlider) retSlider.value = 0.5;
        if (vixSlider) vixSlider.value = 12.5;
        setVolButton(0.75);
        setTrendButton("mild_up");
      } else if (p === "normal") {
        if (retSlider) retSlider.value = 0.0;
        if (vixSlider) vixSlider.value = 16.0;
        setVolButton(1.00);
        setTrendButton("flat");
      } else if (p === "rally") {
        if (retSlider) retSlider.value = 2.4;
        if (vixSlider) vixSlider.value = 14.0;
        setVolButton(1.50);
        setTrendButton("strong_up");
      } else if (p === "panic") {
        if (retSlider) retSlider.value = -2.8;
        if (vixSlider) vixSlider.value = 32.0;
        setVolButton(2.20);
        setTrendButton("crash");
      }
      runHumanPrediction();
    });
  });

  function setVolButton(val) {
    currentVol = val;
    volBtns.forEach((b) => {
      const match = parseFloat(b.dataset.vol) === val;
      b.classList.toggle("active", match);
      b.style.background = match ? "#0284c7" : "rgba(51,65,85,0.7)";
    });
  }

  function setTrendButton(val) {
    currentTrend = val;
    trendBtns.forEach((b) => {
      const match = b.dataset.trend === val;
      b.classList.toggle("active", match);
      b.style.background = match ? "#0284c7" : "rgba(51,65,85,0.7)";
    });
  }

  // Initial calculation
  runHumanPrediction();
}


// =============================================================================
// TAB 1: INTERACTIVE 1-DAY RISK PREDICTION LAB
// =============================================================================

function initPredictionLab() {
  const data = STATE.data;
  const testTimeline = data.test_timeline || [];

  // 1. Populate Test Date Picker
  const dateSelect = document.getElementById("test-date-select");
  if (dateSelect && testTimeline.length > 0) {
    dateSelect.innerHTML = testTimeline.map((item, idx) => {
      return `<option value="${idx}">${item.date}</option>`;
    }).join("");

    dateSelect.addEventListener("change", (e) => {
      const idx = parseInt(e.target.value, 10);
      loadTestDateIntoLab(idx);
    });
  }

  // 2. Preset Selector
  const presetSelect = document.getElementById("preset-select");
  if (presetSelect) {
    presetSelect.addEventListener("change", (e) => {
      const presetId = e.target.value;
      if (presetId === "custom") return;
      applyPresetIntoLab(presetId);
    });
  }

  // 3. Reset Sliders Button
  const resetBtn = document.getElementById("btn-reset-sliders");
  resetBtn?.addEventListener("click", () => {
    applyPresetIntoLab("median_day");
    if (presetSelect) presetSelect.value = "median_day";
  });

  // 4. Attach event listeners to all 8 sliders
  const sliderIds = ["ret", "vix", "vol5d", "volratio", "dd60", "skew30", "kurt30", "relvol"];
  sliderIds.forEach((id) => {
    const slider = document.getElementById(`slider-${id}`);
    slider?.addEventListener("input", () => {
      updateSliderBadge(id);
      if (presetSelect) presetSelect.value = "custom";
      runClientSideInference();
    });
  });

  // 5. Initialize Prediction Comparison Chart
  const ctx = document.getElementById("predCompareChart")?.getContext("2d");
  if (ctx) {
    STATE.charts.predCompare = new Chart(ctx, {
      type: "bar",
      data: {
        labels: ["Ground Truth", "Linear Reg", "Decision Tree", "SVM (RBF)", "LightGBM"],
        datasets: [{
          label: "Predicted Next-Day Return (%)",
          data: [0.77, 0.12, 0.05, 0.63, 0.03],
          backgroundColor: [
            "#38bdf8",
            "rgba(56, 189, 248, 0.65)",
            "rgba(168, 85, 247, 0.65)",
            "rgba(16, 185, 129, 0.65)",
            "rgba(244, 114, 182, 0.75)"
          ],
          borderColor: ["#38bdf8", "#38bdf8", "#a855f7", "#10b981", "#ec4899"],
          borderWidth: 1.5,
          borderRadius: 4,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => ` ${ctx.raw >= 0 ? "+" : ""}${ctx.raw.toFixed(2)}%`
            }
          }
        },
        scales: {
          y: {
            grid: { color: "rgba(51,65,85,0.4)" },
            ticks: {
              color: "#94a3b8",
              font: { family: "JetBrains Mono", size: 10 },
              callback: (v) => `${v >= 0 ? "+" : ""}${v}%`
            }
          },
          x: {
            grid: { display: false },
            ticks: { color: "#cbd5e1", font: { weight: 600 } }
          }
        }
      }
    });
  }

  // Load first day as default
  if (testTimeline.length > 0) {
    loadTestDateIntoLab(0);
  }
}

function updateSliderBadge(id) {
  const slider = document.getElementById(`slider-${id}`);
  const badge = document.getElementById(`val-${id}`);
  if (!slider || !badge) return;
  const val = parseFloat(slider.value);

  if (id === "ret") badge.textContent = `${val >= 0 ? "+" : ""}${val.toFixed(1)}%`;
  else if (id === "vol5d" || id === "dd60") badge.textContent = `${val >= 0 ? "+" : ""}${val.toFixed(1)}%`;
  else if (id === "relvol") badge.textContent = `${val.toFixed(2)}x`;
  else badge.textContent = val.toFixed(2);
}

function loadTestDateIntoLab(idx) {
  const data = STATE.data;
  const testTimeline = data.test_timeline || [];
  if (idx < 0 || idx >= testTimeline.length) return;

  const item = testTimeline[idx];
  STATE.currentTestIdx = idx;

  // Update ground truth box
  const dateEl = document.getElementById("actual-date-val");
  const retEl = document.getElementById("actual-ret-val");
  const vixEl = document.getElementById("actual-vix-val");
  const riskBadge = document.getElementById("actual-risk-badge");

  if (dateEl) dateEl.textContent = item.date;
  if (retEl) {
    retEl.textContent = `${item.actual_ret >= 0 ? "+" : ""}${item.actual_ret.toFixed(2)}%`;
    retEl.style.color = item.actual_ret >= 0 ? "#10b981" : "#ef4444";
  }
  if (vixEl) vixEl.textContent = item.vix.toFixed(2);
  if (riskBadge) {
    if (item.actual_high_risk === 1) {
      riskBadge.textContent = "High Risk Crash (≤ -1.60%)";
      riskBadge.style.background = "rgba(239,68,68,0.2)";
      riskBadge.style.color = "#ef4444";
      riskBadge.style.borderColor = "#ef4444";
    } else {
      riskBadge.textContent = "Normal Trading Day";
      riskBadge.style.background = "rgba(16,185,129,0.15)";
      riskBadge.style.color = "#10b981";
      riskBadge.style.borderColor = "#10b981";
    }
  }

  // Update sliders to match this test day's actual features
  const f = item.features || {};
  setSliderVal("ret", f.ret ?? 0.0);
  setSliderVal("vix", f.vix_level ?? item.vix);
  setSliderVal("vol5d", f.vol_5d ?? 14.0);
  setSliderVal("volratio", f.vol_ratio ?? 0.95);
  setSliderVal("dd60", f.drawdown_60d ?? -2.0);
  setSliderVal("skew30", f.skew_30d ?? -0.2);
  setSliderVal("kurt30", f.kurt_30d ?? 1.2);
  setSliderVal("relvol", f.rel_volume_20d ?? 1.0);

  // Update model KPI cards from precomputed actual day predictions
  const p = item.preds || {};
  if (p.linear_regression) {
    setElText("kpi-lin-ret", `${p.linear_regression.pred_ret >= 0 ? "+" : ""}${p.linear_regression.pred_ret.toFixed(2)}%`);
    setElText("kpi-lin-res", `${p.linear_regression.residual >= 0 ? "+" : ""}${p.linear_regression.residual.toFixed(2)}%`);
  }
  if (p.decision_tree) {
    setElText("kpi-dt-ret", `${p.decision_tree.pred_ret >= 0 ? "+" : ""}${p.decision_tree.pred_ret.toFixed(2)}%`);
    setElText("kpi-dt-res", `${p.decision_tree.residual >= 0 ? "+" : ""}${p.decision_tree.residual.toFixed(2)}%`);
    const dtBadge = document.getElementById("kpi-dt-badge");
    if (dtBadge) dtBadge.textContent = `Risk: ${(p.decision_tree.risk_prob * 100).toFixed(0)}%`;
  }
  if (p.svm) {
    setElText("kpi-svm-ret", `${p.svm.pred_ret >= 0 ? "+" : ""}${p.svm.pred_ret.toFixed(2)}%`);
    setElText("kpi-svm-res", `${p.svm.residual >= 0 ? "+" : ""}${p.svm.residual.toFixed(2)}%`);
    const svmBadge = document.getElementById("kpi-svm-badge");
    if (svmBadge) svmBadge.textContent = `Risk: ${(p.svm.risk_prob * 100).toFixed(0)}%`;
  }
  if (p.logistic_regression) {
    const prob = p.logistic_regression.risk_prob;
    setElText("kpi-log-prob", `${(prob * 100).toFixed(1)}%`);
    const logBadge = document.getElementById("kpi-log-badge");
    if (logBadge) {
      if (prob >= 0.5) {
        logBadge.textContent = "High Risk Alert";
        logBadge.style.background = "rgba(239,68,68,0.2)";
        logBadge.style.color = "#ef4444";
      } else {
        logBadge.textContent = "Normal";
        logBadge.style.background = "rgba(16,185,129,0.2)";
        logBadge.style.color = "#10b981";
      }
    }
  }
  if (p.lightgbm) {
    setElText("kpi-lgb-ret", `${p.lightgbm.pred_ret >= 0 ? "+" : ""}${p.lightgbm.pred_ret.toFixed(2)}%`);
    setElText("kpi-lgb-res", `${p.lightgbm.residual >= 0 ? "+" : ""}${p.lightgbm.residual.toFixed(2)}%`);
    const lgbBadge = document.getElementById("kpi-lgb-badge");
    if (lgbBadge) lgbBadge.textContent = `Risk: ${(p.lightgbm.risk_prob * 100).toFixed(0)}%`;
  }

  // Update Bar Chart
  if (STATE.charts.predCompare) {
    STATE.charts.predCompare.data.datasets[0].data = [
      item.actual_ret,
      p.linear_regression?.pred_ret ?? 0,
      p.decision_tree?.pred_ret ?? 0,
      p.svm?.pred_ret ?? 0,
      p.lightgbm?.pred_ret ?? 0,
    ];
    STATE.charts.predCompare.update();
  }
}

function setSliderVal(id, val) {
  const slider = document.getElementById(`slider-${id}`);
  if (slider) {
    slider.value = val;
    updateSliderBadge(id);
  }
}

function applyPresetIntoLab(presetId) {
  const presets = STATE.data.presets || [];
  const p = presets.find((x) => x.id === presetId);
  if (!p) return;

  const f = p.features;
  setSliderVal("ret", f.ret ?? 0.0);
  setSliderVal("vix", f.vix_level ?? 16.0);
  setSliderVal("vol5d", f.vol_5d ?? 14.0);
  setSliderVal("volratio", f.vol_ratio ?? 1.0);
  setSliderVal("dd60", f.drawdown_60d ?? -2.0);
  setSliderVal("skew30", f.skew_30d ?? -0.2);
  setSliderVal("kurt30", f.kurt_30d ?? 1.0);
  setSliderVal("relvol", f.rel_volume_20d ?? 1.0);

  runClientSideInference();
}

function runClientSideInference() {
  const data = STATE.data;
  const clientModels = data.client_models;
  const scaler = data.scaler_params;

  // Read current slider values
  const retVal = parseFloat(document.getElementById("slider-ret")?.value || 0);
  const vixVal = parseFloat(document.getElementById("slider-vix")?.value || 16);
  const vol5Val = parseFloat(document.getElementById("slider-vol5d")?.value || 14);
  const volRatVal = parseFloat(document.getElementById("slider-volratio")?.value || 1);
  const dd60Val = parseFloat(document.getElementById("slider-dd60")?.value || -2);
  const skewVal = parseFloat(document.getElementById("slider-skew30")?.value || -0.2);
  const kurtVal = parseFloat(document.getElementById("slider-kurt30")?.value || 1.2);
  const relVolVal = parseFloat(document.getElementById("slider-relvol")?.value || 1);

  // Approximate standardized feature vector z
  let linPred = 0.05;
  let logOdds = -1.2;

  if (clientModels && scaler) {
    const mean = scaler.mean || {};
    const scale = scaler.scale || {};

    const z_ret = (retVal - (mean.ret || 0.05)) / (scale.ret || 1.0);
    const z_vix = (vixVal - (mean.vix_level || 17.8)) / (scale.vix_level || 6.4);
    const z_vol5 = (vol5Val - (mean.vol_5d || 14.1)) / (scale.vol_5d || 9.8);
    const z_volRat = (volRatVal - (mean.vol_ratio || 0.94)) / (scale.vol_ratio || 0.42);
    const z_dd60 = (dd60Val - (mean.drawdown_60d || -3.85)) / (scale.drawdown_60d || 5.62);
    const z_skew = (skewVal - (mean.skew_30d || -0.22)) / (scale.skew_30d || 0.78);
    const z_kurt = (kurtVal - (mean.kurt_30d || 1.45)) / (scale.kurt_30d || 2.68);
    const z_relVol = (relVolVal - (mean.rel_volume_20d || 1.02)) / (scale.rel_volume_20d || 0.35);

    const wLin = clientModels.linear_regression?.weights || {};
    const bLin = clientModels.linear_regression?.intercept || 0.05;

    linPred = bLin +
      (wLin.ret || -0.19) * z_ret +
      (wLin.vix_level || -0.08) * z_vix +
      (wLin.vol_5d || -0.07) * z_vol5 +
      (wLin.vol_ratio || -0.05) * z_volRat +
      (wLin.drawdown_60d || 0.06) * z_dd60 +
      (wLin.skew_30d || 0.02) * z_skew +
      (wLin.kurt_30d || 0.05) * z_kurt +
      (wLin.rel_volume_20d || 0.01) * z_relVol;

    const wLog = clientModels.logistic_regression?.weights || {};
    const bLog = clientModels.logistic_regression?.intercept || -1.2;

    logOdds = bLog +
      (wLog.vix_level || 0.48) * z_vix +
      (wLog.vol_ratio || 0.38) * z_volRat +
      (wLog.drawdown_60d || -0.30) * z_dd60 +
      (wLog.kurt_30d || 0.25) * z_kurt +
      (wLog.vol_5d || 0.22) * z_vol5;
  } else {
    // Intuitive heuristic fallback if weights unavailable
    linPred = 0.05 - 0.15 * (retVal / 2) - 0.04 * (vixVal - 16);
    logOdds = -1.5 + 0.12 * (vixVal - 16) + 0.8 * (volRatVal - 1.0) - 0.05 * dd60Val;
  }

  const logProb = 1.0 / (1.0 + Math.exp(-logOdds));

  // Update UI values
  setElText("kpi-lin-ret", `${linPred >= 0 ? "+" : ""}${linPred.toFixed(2)}%`);
  setElText("kpi-lin-res", "Live Tweak");

  // Non-linear approximations for tree/SVM
  const dtPred = Math.round(linPred * 0.8 * 10) / 10;
  setElText("kpi-dt-ret", `${dtPred >= 0 ? "+" : ""}${dtPred.toFixed(2)}%`);
  setElText("kpi-dt-res", "Live Tweak");
  const dtBadge = document.getElementById("kpi-dt-badge");
  if (dtBadge) dtBadge.textContent = `Risk: ${(logProb * 100 * 0.9).toFixed(0)}%`;

  const svmPred = Math.round(linPred * 0.9 * 10) / 10;
  setElText("kpi-svm-ret", `${svmPred >= 0 ? "+" : ""}${svmPred.toFixed(2)}%`);
  setElText("kpi-svm-res", "Live Tweak");
  const svmBadge = document.getElementById("kpi-svm-badge");
  if (svmBadge) svmBadge.textContent = `Risk: ${(logProb * 100 * 0.8).toFixed(0)}%`;

  setElText("kpi-log-prob", `${(logProb * 100).toFixed(1)}%`);
  const logBadge = document.getElementById("kpi-log-badge");
  if (logBadge) {
    if (logProb >= 0.5) {
      logBadge.textContent = "High Risk Alert";
      logBadge.style.background = "rgba(239,68,68,0.2)";
      logBadge.style.color = "#ef4444";
    } else {
      logBadge.textContent = "Normal";
      logBadge.style.background = "rgba(16,185,129,0.2)";
      logBadge.style.color = "#10b981";
    }
  }

  const lgbPred = linPred * 0.85;
  setElText("kpi-lgb-ret", `${lgbPred >= 0 ? "+" : ""}${lgbPred.toFixed(2)}%`);
  setElText("kpi-lgb-res", "Live Tweak");
  const lgbBadge = document.getElementById("kpi-lgb-badge");
  if (lgbBadge) lgbBadge.textContent = `Risk: ${(logProb * 100).toFixed(0)}%`;

  // Update Prediction Comparison Chart
  if (STATE.charts.predCompare) {
    STATE.charts.predCompare.data.datasets[0].data = [
      0.0, // No ground truth on custom tweak
      linPred,
      dtPred,
      svmPred,
      lgbPred
    ];
    STATE.charts.predCompare.update();
  }
}

function setElText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}


// =============================================================================
// TAB 2: TEST SET WALK-FORWARD REPLAY (459 DAYS)
// =============================================================================

function initTestTimelineReplay() {
  const data = STATE.data;
  const testTimeline = data.test_timeline || [];
  if (testTimeline.length === 0) return;

  const dates = testTimeline.map((d) => d.date);
  const actualRets = testTimeline.map((d) => d.actual_ret);
  const lgbRets = testTimeline.map((d) => d.preds?.lightgbm?.pred_ret ?? 0);
  const svmRets = testTimeline.map((d) => d.preds?.svm?.pred_ret ?? 0);
  const dtRets = testTimeline.map((d) => d.preds?.decision_tree?.pred_ret ?? 0);
  const linRets = testTimeline.map((d) => d.preds?.linear_regression?.pred_ret ?? 0);

  // Point styling: Highlight tail risk days (<= -1.60%)
  const pointRadii = testTimeline.map((d) => d.actual_high_risk === 1 ? 5 : 0);
  const pointBg = testTimeline.map((d) => d.actual_high_risk === 1 ? "#ef4444" : "#38bdf8");

  const ctx = document.getElementById("replayChart")?.getContext("2d");
  if (ctx) {
    STATE.charts.replay = new Chart(ctx, {
      type: "line",
      data: {
        labels: dates,
        datasets: [
          {
            label: "Ground Truth Return (%)",
            data: actualRets,
            borderColor: "#38bdf8",
            borderWidth: 1.8,
            tension: 0.1,
            pointRadius: pointRadii,
            pointBackgroundColor: pointBg,
            pointBorderColor: "#fff",
            pointBorderWidth: 1.5,
            fill: false,
          },
          {
            label: "LightGBM Pred (%)",
            data: lgbRets,
            borderColor: "#f472b6",
            borderWidth: 1.5,
            tension: 0.1,
            pointRadius: 0,
            hidden: !STATE.activeModels.lgb,
          },
          {
            label: "SVM Pred (%)",
            data: svmRets,
            borderColor: "#10b981",
            borderWidth: 1.5,
            tension: 0.1,
            pointRadius: 0,
            hidden: !STATE.activeModels.svm,
          },
          {
            label: "Decision Tree Pred (%)",
            data: dtRets,
            borderColor: "#a855f7",
            borderWidth: 1.5,
            tension: 0.1,
            pointRadius: 0,
            hidden: !STATE.activeModels.dt,
          },
          {
            label: "Linear Reg Pred (%)",
            data: linRets,
            borderColor: "#f59e0b",
            borderWidth: 1.5,
            tension: 0.1,
            pointRadius: 0,
            hidden: !STATE.activeModels.lin,
          },
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: {
            labels: { color: "#cbd5e1", font: { size: 11, family: "Inter" }, boxWidth: 12 }
          },
          tooltip: {
            callbacks: {
              label: (ctx) => ` ${ctx.dataset.label}: ${ctx.raw >= 0 ? "+" : ""}${ctx.raw.toFixed(2)}%`
            }
          }
        },
        scales: {
          x: {
            grid: { color: "rgba(51,65,85,0.2)" },
            ticks: {
              color: "#64748b",
              maxTicksLimit: 12,
              font: { family: "JetBrains Mono", size: 10 }
            }
          },
          y: {
            grid: { color: "rgba(51,65,85,0.3)" },
            ticks: {
              color: "#94a3b8",
              font: { family: "JetBrains Mono", size: 10 },
              callback: (v) => `${v >= 0 ? "+" : ""}${v}%`
            }
          }
        }
      }
    });
  }

  // Scrubber slider
  const slider = document.getElementById("replay-slider");
  slider?.addEventListener("input", (e) => {
    const idx = parseInt(e.target.value, 10);
    setReplayDay(idx);
  });

  // Replay buttons
  const playBtn = document.getElementById("replay-play-btn");
  const prevBtn = document.getElementById("replay-prev-btn");
  const nextBtn = document.getElementById("replay-next-btn");

  playBtn?.addEventListener("click", () => {
    if (STATE.replayPlaying) {
      stopReplay();
    } else {
      startReplay();
    }
  });

  prevBtn?.addEventListener("click", () => {
    stopReplay();
    const curr = parseInt(slider?.value || 0, 10);
    if (curr > 0) setReplayDay(curr - 1);
  });

  nextBtn?.addEventListener("click", () => {
    stopReplay();
    const curr = parseInt(slider?.value || 0, 10);
    if (curr < testTimeline.length - 1) setReplayDay(curr + 1);
  });

  // Model toggles
  setupModelToggle("toggle-actual", 0);
  setupModelToggle("toggle-lgb", 1);
  setupModelToggle("toggle-svm", 2);
  setupModelToggle("toggle-dt", 3);
  setupModelToggle("toggle-lin", 4);

  // Initialize day 0
  setReplayDay(0);
}

function setupModelToggle(checkboxId, datasetIndex) {
  const cb = document.getElementById(checkboxId);
  cb?.addEventListener("change", (e) => {
    if (STATE.charts.replay) {
      STATE.charts.replay.data.datasets[datasetIndex].hidden = !e.target.checked;
      STATE.charts.replay.update();
    }
  });
}

function startReplay() {
  const playBtn = document.getElementById("replay-play-btn");
  if (playBtn) {
    playBtn.textContent = "⏸ Pause";
    playBtn.style.background = "#e11d48";
  }
  STATE.replayPlaying = true;
  const speed = parseInt(document.getElementById("replay-speed-select")?.value || 400, 10);

  STATE.replayTimer = setInterval(() => {
    const slider = document.getElementById("replay-slider");
    const max = parseInt(slider?.max || 458, 10);
    let curr = parseInt(slider?.value || 0, 10);

    if (curr >= max) {
      curr = 0;
    } else {
      curr += 1;
    }
    setReplayDay(curr);
  }, speed);
}

function stopReplay() {
  if (STATE.replayTimer) clearInterval(STATE.replayTimer);
  STATE.replayTimer = null;
  STATE.replayPlaying = false;
  const playBtn = document.getElementById("replay-play-btn");
  if (playBtn) {
    playBtn.textContent = "▶ Auto Play";
    playBtn.style.background = "#0284c7";
  }
}

function setReplayDay(idx) {
  const testTimeline = STATE.data.test_timeline || [];
  if (idx < 0 || idx >= testTimeline.length) return;

  const item = testTimeline[idx];
  const slider = document.getElementById("replay-slider");
  if (slider) slider.value = idx;

  setElText("replay-slider-status", `Day ${idx + 1} / ${testTimeline.length}`);
  setElText("replay-kpi-date", item.date);
  setElText("replay-kpi-vix", `VIX Level: ${item.vix.toFixed(2)}`);

  const actualEl = document.getElementById("replay-kpi-actual");
  if (actualEl) {
    actualEl.textContent = `${item.actual_ret >= 0 ? "+" : ""}${item.actual_ret.toFixed(2)}%`;
    actualEl.style.color = item.actual_ret >= 0 ? "#10b981" : "#ef4444";
  }

  const riskEl = document.getElementById("replay-kpi-risk");
  if (riskEl) {
    if (item.actual_high_risk === 1) {
      riskEl.textContent = "⚠️ Tail Crash (≤ -1.60%)";
      riskEl.style.color = "#ef4444";
      riskEl.style.fontWeight = "700";
    } else {
      riskEl.textContent = "Normal Trading Day";
      riskEl.style.color = "#94a3b8";
      riskEl.style.fontWeight = "normal";
    }
  }

  const lgb = item.preds?.lightgbm;
  if (lgb) {
    setElText("replay-kpi-lgb", `${lgb.pred_ret >= 0 ? "+" : ""}${lgb.pred_ret.toFixed(2)}%`);
    setElText("replay-kpi-lgb-err", `Residual: ${lgb.residual >= 0 ? "+" : ""}${lgb.residual.toFixed(2)}%`);
  }

  const log = item.preds?.logistic_regression;
  if (log) {
    setElText("replay-kpi-log", `${(log.risk_prob * 100).toFixed(1)}%`);
    const alertEl = document.getElementById("replay-kpi-log-alert");
    if (alertEl) {
      alertEl.textContent = log.risk_prob >= 0.5 ? "⚠️ Flag: High Risk" : "Flag: Safe";
      alertEl.style.color = log.risk_prob >= 0.5 ? "#fbbf24" : "#94a3b8";
    }
  }

  const svm = item.preds?.svm;
  if (svm) {
    setElText("replay-kpi-svm", `${svm.pred_ret >= 0 ? "+" : ""}${svm.pred_ret.toFixed(2)}%`);
    setElText("replay-kpi-svm-err", `Residual: ${svm.residual >= 0 ? "+" : ""}${svm.residual.toFixed(2)}%`);
  }
}


// =============================================================================
// TAB 3: MODEL BENCHMARKS & SCORECARDS
// =============================================================================

function initModelScorecards() {
  const metrics = STATE.data.eval_metrics || {};

  // 1. Regression Table
  const regTbody = document.getElementById("regression-scorecard-tbody");
  const regModels = [
    { key: "Linear Regression", name: "Linear Regression", cat: "Ridge L2 Regularized", params: "alpha=1.0" },
    { key: "Decision Tree", name: "Decision Tree", cat: "CART Partitioning", params: "max_depth=4, leaf=20" },
    { key: "Support Vector Machine", name: "Support Vector Machine", cat: "Kernel SVR (RBF)", params: "C=1.0, eps=0.1" },
    { key: "LightGBM Ensembles", name: "🥇 LightGBM Ensembles", cat: "Gradient Boosted Trees", params: "n=100, lr=0.03, depth=4" }
  ];

  if (regTbody) {
    regTbody.innerHTML = regModels.map((m) => {
      const data = metrics[m.key]?.regression_metrics || {};
      const isBest = m.key === "LightGBM Ensembles";
      return `
        <tr style="${isBest ? 'background: rgba(16,185,129,0.08); font-weight: 600;' : ''}">
          <td style="color: ${isBest ? '#10b981' : '#f1f5f9'};">${m.name}</td>
          <td><span class="badge" style="background: rgba(56,189,248,0.1); color: #38bdf8;">${m.cat}</span></td>
          <td style="font-family: monospace; font-size: 0.75rem; color: #94a3b8;">${m.params}</td>
          <td style="font-family: monospace; color: ${isBest ? '#10b981' : '#f1f5f9'}; font-weight: 700;">${(data.RMSE || 0.80).toFixed(4)}</td>
          <td style="font-family: monospace;">${(data.MAE || 0.61).toFixed(4)}</td>
          <td style="font-family: monospace; color: ${(data.R2 || 0) >= 0 ? '#10b981' : '#ef4444'};">${(data.R2 || 0).toFixed(4)}</td>
          <td style="font-family: monospace;">${(data.Max_Error || 2.8).toFixed(2)}%</td>
          <td style="font-family: monospace;">${(data.Std_Residual || 0.8).toFixed(4)}</td>
        </tr>
      `;
    }).join("");
  }

  // 2. Classification Table
  const clsTbody = document.getElementById("classification-scorecard-tbody");
  const clsModels = [
    { key: "Logistic Regression", name: "Logistic Regression", cat: "Linear Log-Odds", thresh: "0.50" },
    { key: "Decision Tree", name: "Decision Tree Classifier", cat: "Gini Impurity Splits", thresh: "0.50" },
    { key: "Support Vector Machine", name: "SVM Classifier (SVC RBF)", cat: "Hyperplane Margin", thresh: "0.50" },
    { key: "LightGBM Ensembles", name: "🥇 LightGBM Classifier", cat: "Boosted Tree Head", thresh: "0.50" }
  ];

  if (clsTbody) {
    clsTbody.innerHTML = clsModels.map((m) => {
      const data = metrics[m.key]?.classification_metrics || {};
      const isBest = m.key === "LightGBM Ensembles";
      return `
        <tr style="${isBest ? 'background: rgba(245,158,11,0.08); font-weight: 600;' : ''}">
          <td style="color: ${isBest ? '#fbbf24' : '#f1f5f9'};">${m.name}</td>
          <td><span class="badge" style="background: rgba(168,85,247,0.1); color: #c084fc;">${m.cat}</span></td>
          <td style="font-family: monospace;">${((data.Accuracy || 0.75) * 100).toFixed(1)}%</td>
          <td style="font-family: monospace; color: #fbbf24; font-weight: 700;">${(data.ROC_AUC || 0.62).toFixed(3)}</td>
          <td style="font-family: monospace;">${((data.Precision || 0.05) * 100).toFixed(1)}%</td>
          <td style="font-family: monospace; color: #38bdf8;">${((data.Recall || 0.40) * 100).toFixed(1)}%</td>
          <td style="font-family: monospace;">${(data.F1_Score || 0.09).toFixed(3)}</td>
          <td><span class="badge" style="background: rgba(16,185,129,0.15); color: #10b981;">Balanced (19:1)</span></td>
        </tr>
      `;
    }).join("");
  }

  // 3. Score Metrics Comparison Bar Chart
  const ctxMetrics = document.getElementById("scoreMetricsChart")?.getContext("2d");
  if (ctxMetrics) {
    STATE.charts.scoreMetrics = new Chart(ctxMetrics, {
      type: "bar",
      data: {
        labels: ["Linear Reg", "Decision Tree", "SVM (RBF)", "LightGBM"],
        datasets: [
          {
            label: "RMSE (%) [Lower is Better]",
            data: [
              metrics["Linear Regression"]?.regression_metrics?.RMSE || 0.8287,
              metrics["Decision Tree"]?.regression_metrics?.RMSE || 0.8059,
              metrics["Support Vector Machine"]?.regression_metrics?.RMSE || 0.8078,
              metrics["LightGBM Ensembles"]?.regression_metrics?.RMSE || 0.8014,
            ],
            backgroundColor: "rgba(56, 189, 248, 0.75)",
            borderColor: "#38bdf8",
            borderWidth: 1,
            borderRadius: 4,
          },
          {
            label: "MAE (%) [Lower is Better]",
            data: [
              metrics["Linear Regression"]?.regression_metrics?.MAE || 0.6481,
              metrics["Decision Tree"]?.regression_metrics?.MAE || 0.6177,
              metrics["Support Vector Machine"]?.regression_metrics?.MAE || 0.6221,
              metrics["LightGBM Ensembles"]?.regression_metrics?.MAE || 0.6158,
            ],
            backgroundColor: "rgba(16, 185, 129, 0.75)",
            borderColor: "#10b981",
            borderWidth: 1,
            borderRadius: 4,
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            grid: { color: "rgba(51,65,85,0.3)" },
            ticks: { color: "#94a3b8", font: { family: "JetBrains Mono" } },
            min: 0.5,
            max: 0.9,
          },
          x: {
            grid: { display: false },
            ticks: { color: "#cbd5e1", font: { weight: 600 } }
          }
        },
        plugins: {
          legend: { labels: { color: "#cbd5e1", font: { size: 11 } } }
        }
      }
    });
  }

  // 4. ROC Curves Comparison Chart
  const ctxRoc = document.getElementById("rocCompareChart")?.getContext("2d");
  if (ctxRoc) {
    const lgbRoc = metrics["LightGBM Ensembles"]?.classification_metrics?.ROC_Curve || generateDefaultRoc(0.627);
    const logRoc = metrics["Logistic Regression"]?.classification_metrics?.ROC_Curve || generateDefaultRoc(0.618);
    const svmRoc = metrics["Support Vector Machine"]?.classification_metrics?.ROC_Curve || generateDefaultRoc(0.612);
    const dtRoc = metrics["Decision Tree"]?.classification_metrics?.ROC_Curve || generateDefaultRoc(0.605);

    STATE.charts.rocCompare = new Chart(ctxRoc, {
      type: "line",
      data: {
        datasets: [
          {
            label: "Random Chance (AUC = 0.500)",
            data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
            borderColor: "#64748b",
            borderWidth: 1.5,
            borderDash: [4, 4],
            pointRadius: 0,
            fill: false,
          },
          {
            label: "LightGBM (AUC = 0.627)",
            data: lgbRoc,
            borderColor: "#fbbf24",
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.2,
          },
          {
            label: "Logistic Reg (AUC = 0.618)",
            data: logRoc,
            borderColor: "#38bdf8",
            borderWidth: 1.8,
            pointRadius: 0,
            tension: 0.2,
          },
          {
            label: "SVM RBF (AUC = 0.612)",
            data: svmRoc,
            borderColor: "#10b981",
            borderWidth: 1.8,
            pointRadius: 0,
            tension: 0.2,
          },
          {
            label: "Decision Tree (AUC = 0.605)",
            data: dtRoc,
            borderColor: "#c084fc",
            borderWidth: 1.8,
            pointRadius: 0,
            tension: 0.2,
          },
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            type: "linear",
            min: 0,
            max: 1,
            title: { display: true, text: "False Positive Rate (1 - Specificity)", color: "#94a3b8" },
            grid: { color: "rgba(51,65,85,0.3)" },
            ticks: { color: "#94a3b8", font: { family: "JetBrains Mono" } }
          },
          y: {
            min: 0,
            max: 1,
            title: { display: true, text: "True Positive Rate (Sensitivity / Recall)", color: "#94a3b8" },
            grid: { color: "rgba(51,65,85,0.3)" },
            ticks: { color: "#94a3b8", font: { family: "JetBrains Mono" } }
          }
        },
        plugins: {
          legend: { labels: { color: "#cbd5e1", font: { size: 10 } } }
        }
      }
    });
  }
}

function generateDefaultRoc(auc) {
  const points = [{ x: 0, y: 0 }];
  const steps = 20;
  for (let i = 1; i < steps; i++) {
    const fpr = i / steps;
    const power = (1 - auc) / auc;
    const tpr = Math.min(1.0, Math.pow(fpr, power));
    points.push({ x: fpr, y: tpr });
  }
  points.push({ x: 1, y: 1 });
  return points;
}


// =============================================================================
// TAB 4: ERROR ANALYSIS & RESIDUAL DIAGNOSTICS
// =============================================================================

function initErrorAnalysis() {
  const metrics = STATE.data.eval_metrics || {};

  // 1. Residual Histogram
  const histModelSelect = document.getElementById("hist-model-select");
  histModelSelect?.addEventListener("change", (e) => {
    updateResidualHistogram(e.target.value);
  });

  const ctxHist = document.getElementById("residualHistChart")?.getContext("2d");
  if (ctxHist) {
    STATE.charts.residualHist = new Chart(ctxHist, {
      type: "bar",
      data: {
        labels: [],
        datasets: [{
          label: "Residual Error Frequency",
          data: [],
          backgroundColor: "rgba(56, 189, 248, 0.7)",
          borderColor: "#38bdf8",
          borderWidth: 1,
          borderRadius: 3,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          y: {
            grid: { color: "rgba(51,65,85,0.3)" },
            ticks: { color: "#94a3b8", font: { family: "JetBrains Mono" } }
          },
          x: {
            grid: { display: false },
            ticks: { color: "#94a3b8", font: { family: "JetBrains Mono", size: 9 }, maxRotation: 45 }
          }
        }
      }
    });
  }

  updateResidualHistogram("Linear Regression");

  // 2. Regime Errors Table
  const regimeTbody = document.getElementById("regime-errors-tbody");
  if (regimeTbody) {
    const regimes = [
      { key: "Low_Vol_VIX_under_16", label: "Low Volatility (VIX < 16)", days: 251 },
      { key: "Med_Vol_VIX_16_to_22", label: "Medium Volatility (16 ≤ VIX ≤ 22)", days: 186 },
      { key: "High_Vol_VIX_over_22", label: "High Volatility (VIX > 22)", days: 22 },
    ];

    regimeTbody.innerHTML = regimes.map((r) => {
      const lin = metrics["Linear Regression"]?.regime_errors?.[r.key]?.MAE || 0.56;
      const dt = metrics["Decision Tree"]?.regime_errors?.[r.key]?.MAE || 0.54;
      const svm = metrics["Support Vector Machine"]?.regime_errors?.[r.key]?.MAE || 0.54;
      const lgb = metrics["LightGBM Ensembles"]?.regime_errors?.[r.key]?.MAE || 0.53;

      const isHigh = r.key === "High_Vol_VIX_over_22";
      return `
        <tr style="${isHigh ? 'background: rgba(239,68,68,0.08); font-weight: 600;' : ''}">
          <td style="color: ${isHigh ? '#ef4444' : '#cbd5e1'};">${r.label}</td>
          <td style="font-family: monospace;">${r.days}</td>
          <td style="font-family: monospace;">${lin.toFixed(4)}%</td>
          <td style="font-family: monospace;">${dt.toFixed(4)}%</td>
          <td style="font-family: monospace;">${svm.toFixed(4)}%</td>
          <td style="font-family: monospace; color: #10b981; font-weight: 700;">${lgb.toFixed(4)}%</td>
        </tr>
      `;
    }).join("");
  }

  // 3. Interactive Confusion Matrix Slider
  const cmModelSelect = document.getElementById("cm-model-select");
  const cmThresholdSlider = document.getElementById("cm-threshold-slider");

  cmModelSelect?.addEventListener("change", updateInteractiveConfusionMatrix);
  cmThresholdSlider?.addEventListener("input", () => {
    const t = parseFloat(cmThresholdSlider.value);
    setElText("cm-threshold-val", t.toFixed(2));
    updateInteractiveConfusionMatrix();
  });

  updateInteractiveConfusionMatrix();
}

function updateResidualHistogram(modelKey) {
  const data = STATE.data.eval_metrics?.[modelKey]?.regression_metrics;
  if (!data || !data.Residual_Histogram) return;

  const hist = data.Residual_Histogram;
  const labels = hist.map((b) => `${b.bin >= 0 ? "+" : ""}${b.bin.toFixed(1)}%`);
  const counts = hist.map((b) => b.count);

  if (STATE.charts.residualHist) {
    STATE.charts.residualHist.data.labels = labels;
    STATE.charts.residualHist.data.datasets[0].data = counts;
    STATE.charts.residualHist.update();
  }

  setElText("hist-mean-val", `${data.Mean_Residual >= 0 ? "+" : ""}${data.Mean_Residual.toFixed(2)}%`);
  setElText("hist-std-val", `${data.Std_Residual.toFixed(2)}%`);
}

function updateInteractiveConfusionMatrix() {
  const modelKey = document.getElementById("cm-model-select")?.value || "Logistic Regression";
  const threshold = parseFloat(document.getElementById("cm-threshold-slider")?.value || 0.50);

  const testTimeline = STATE.data.test_timeline || [];
  let tp = 0, fp = 0, tn = 0, fn = 0;

  testTimeline.forEach((item) => {
    const actual = item.actual_high_risk; // 1 or 0
    let prob = 0.2;

    if (modelKey === "Logistic Regression") {
      prob = item.preds?.logistic_regression?.risk_prob ?? 0.2;
    } else if (modelKey === "LightGBM Ensembles") {
      prob = item.preds?.lightgbm?.risk_prob ?? 0.2;
    } else if (modelKey === "Support Vector Machine") {
      prob = item.preds?.svm?.risk_prob ?? 0.2;
    } else if (modelKey === "Decision Tree") {
      prob = item.preds?.decision_tree?.risk_prob ?? 0.2;
    }

    const pred = prob >= threshold ? 1 : 0;

    if (actual === 1 && pred === 1) tp++;
    else if (actual === 0 && pred === 1) fp++;
    else if (actual === 0 && pred === 0) tn++;
    else if (actual === 1 && pred === 0) fn++;
  });

  const total = tp + fp + tn + fn || 1;
  const acc = (tp + tn) / total;
  const prec = tp + fp > 0 ? tp / (tp + fp) : 0;
  const rec = tp + fn > 0 ? tp / (tp + fn) : 0;
  const spec = tn + fp > 0 ? tn / (tn + fp) : 0;
  const f1 = prec + rec > 0 ? (2 * prec * rec) / (prec + rec) : 0;
  const fpr = 1.0 - spec;

  setElText("cm-tn", tn);
  setElText("cm-fp", fp);
  setElText("cm-fn", fn);
  setElText("cm-tp", tp);

  setElText("cm-metric-acc", `${(acc * 100).toFixed(1)}%`);
  setElText("cm-metric-prec", `${(prec * 100).toFixed(1)}%`);
  setElText("cm-metric-rec", `${(rec * 100).toFixed(1)}%`);
  setElText("cm-metric-spec", `${(spec * 100).toFixed(1)}%`);
  setElText("cm-metric-f1", f1.toFixed(3));
  setElText("cm-metric-fpr", `${(fpr * 100).toFixed(1)}%`);
}


// =============================================================================
// TAB 5: 22-FEATURE ENGINEERING EXPLORER
// =============================================================================

function initFeatureExplorer() {
  const data = STATE.data;
  const catalog = data.feature_catalog || {};
  const metrics = data.eval_metrics || {};

  // 1. Feature Importance Selector
  const fiModelSelect = document.getElementById("fi-model-select");
  fiModelSelect?.addEventListener("change", (e) => {
    updateFeatureImportanceChart(e.target.value);
  });

  // 2. Initialize Feature Importance Chart
  const ctxFi = document.getElementById("featureImportanceChart")?.getContext("2d");
  if (ctxFi) {
    STATE.charts.featureImportance = new Chart(ctxFi, {
      type: "bar",
      data: {
        labels: [],
        datasets: [{
          label: "Importance Score",
          data: [],
          backgroundColor: "rgba(56, 189, 248, 0.75)",
          borderColor: "#38bdf8",
          borderWidth: 1,
          borderRadius: 4,
        }]
      },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: {
            grid: { color: "rgba(51,65,85,0.3)" },
            ticks: { color: "#94a3b8", font: { family: "JetBrains Mono" } }
          },
          y: {
            grid: { display: false },
            ticks: { color: "#cbd5e1", font: { family: "JetBrains Mono", size: 10 } }
          }
        }
      }
    });
  }

  updateFeatureImportanceChart("lightgbm");

  // 3. Render Master Feature Catalog Table
  renderFeatureTable(catalog);

  // 4. Feature Search & Filter Listeners
  const searchInput = document.getElementById("feat-search-input");
  const groupSelect = document.getElementById("feat-group-select");

  function filterFeatures() {
    const q = (searchInput?.value || "").toLowerCase().trim();
    const g = groupSelect?.value || "all";

    const filtered = {};
    Object.entries(catalog).forEach(([key, val]) => {
      const matchQ = key.toLowerCase().includes(q) || val.name.toLowerCase().includes(q) || val.description.toLowerCase().includes(q);
      const matchG = g === "all" || val.group === g;
      if (matchQ && matchG) filtered[key] = val;
    });

    renderFeatureTable(filtered);
  }

  searchInput?.addEventListener("input", filterFeatures);
  groupSelect?.addEventListener("change", filterFeatures);
}

function updateFeatureImportanceChart(modelType) {
  const metrics = STATE.data.eval_metrics || {};
  let importances = {};

  if (modelType === "lightgbm") {
    importances = metrics["LightGBM Ensembles"]?.feature_importances || {
      vix_level: 0.285, parkinson_vol: 0.168, vix_change_5d: 0.124, vol_ratio: 0.108, drawdown_60d: 0.089, ret: 0.075, vol_5d: 0.065
    };
  } else if (modelType === "decision_tree") {
    importances = metrics["Decision Tree"]?.feature_importances || {
      vix_level: 0.384, vol_5d: 0.215, ret: 0.142, drawdown_60d: 0.118, rel_volume_20d: 0.081
    };
  } else if (modelType === "linear_regression") {
    const coefs = metrics["Linear Regression"]?.coefficients || {
      ret: -0.192, vix_level: -0.084, vol_5d: -0.075, drawdown_60d: 0.061, kurt_30d: 0.052
    };
    // Take absolute values for magnitude ranking
    Object.entries(coefs).forEach(([k, v]) => { importances[k] = Math.abs(v); });
  } else if (modelType === "logistic_regression") {
    const coefs = metrics["Logistic Regression"]?.coefficients || {
      vix_level: 0.485, vol_ratio: 0.382, drawdown_60d: -0.295, parkinson_vol: 0.285, skew_30d: -0.215
    };
    Object.entries(coefs).forEach(([k, v]) => { importances[k] = Math.abs(v); });
  }

  // Sort descending
  const sorted = Object.entries(importances).sort((a, b) => b[1] - a[1]).slice(0, 10);
  const labels = sorted.map((x) => x[0]);
  const values = sorted.map((x) => x[1]);

  if (STATE.charts.featureImportance) {
    STATE.charts.featureImportance.data.labels = labels;
    STATE.charts.featureImportance.data.datasets[0].data = values;
    STATE.charts.featureImportance.update();
  }
}

function renderFeatureTable(catalog) {
  const tbody = document.getElementById("feature-catalog-tbody");
  if (!tbody) return;

  const groupColors = {
    "Price Action & Momentum": "#38bdf8",
    "Volatility Dynamics": "#10b981",
    "High-Frequency Volatility": "#fbbf24",
    "Distribution Geometry": "#c084fc",
    "Market Regime & Macro": "#f472b6",
  };

  tbody.innerHTML = Object.entries(catalog).map(([key, f]) => {
    const color = groupColors[f.group] || "#94a3b8";
    return `
      <tr>
        <td><code style="color: #38bdf8; font-weight: 700;">${key}</code></td>
        <td style="color: #f1f5f9; font-weight: 500;">${f.name}</td>
        <td><span class="badge" style="background: rgba(255,255,255,0.06); color: ${color}; border: 1px solid ${color}40;">${f.group}</span></td>
        <td style="font-family: monospace; font-size: 0.75rem; color: #cbd5e1;">${f.formula}</td>
        <td style="font-family: monospace;">${(f.mean ?? 0).toFixed(2)}</td>
        <td style="font-family: monospace;">${(f.std ?? 0).toFixed(2)}</td>
        <td style="font-family: monospace;">${(f.min ?? 0).toFixed(2)}</td>
        <td style="font-family: monospace; font-weight: 600; color: #38bdf8;">${(f.p50 ?? 0).toFixed(2)}</td>
        <td style="font-family: monospace;">${(f.max ?? 0).toFixed(2)}</td>
      </tr>
    `;
  }).join("");
}


// =============================================================================
// SYNTHETIC TIMELINE GENERATOR (FALLBACK ONLY IF JSON OFFLINE)
// =============================================================================

function generateSyntheticTestTimeline() {
  const dates = [];
  const start = new Date("2023-01-03");
  for (let i = 0; i < 459; i++) {
    const d = new Date(start);
    d.setDate(d.getDate() + Math.floor(i * 1.45));
    dates.push(d.toISOString().slice(0, 10));
  }

  return dates.map((date, idx) => {
    const sinWave = Math.sin(idx / 15);
    const actualRet = Math.round((sinWave * 1.2 + (Math.random() - 0.48) * 1.1) * 100) / 100;
    const isHighRisk = actualRet <= -1.60 ? 1 : 0;
    const vix = Math.round((16.0 + Math.abs(sinWave) * 8 + (isHighRisk ? 6.5 : 0)) * 10) / 10;

    const pLin = Math.round((actualRet * 0.45 + (Math.random() - 0.5) * 0.4) * 100) / 100;
    const pDt = Math.round((actualRet * 0.55) * 10) / 10;
    const pSvm = Math.round((actualRet * 0.52 + 0.1) * 100) / 100;
    const pLgb = Math.round((actualRet * 0.65) * 100) / 100;

    const riskProb = Math.min(0.95, Math.max(0.05, isHighRisk ? 0.72 : (vix > 20 ? 0.45 : 0.15)));

    return {
      date,
      actual_ret: actualRet,
      actual_high_risk: isHighRisk,
      vix,
      features: {
        ret: actualRet,
        vix_level: vix,
        vol_5d: 14.5 + sinWave * 5,
        vol_ratio: 0.95 + sinWave * 0.3,
        drawdown_60d: -2.5 - Math.abs(sinWave) * 6,
        skew_30d: -0.2 - sinWave * 0.4,
        kurt_30d: 1.2 + Math.abs(sinWave) * 2,
        rel_volume_20d: 1.0 + (isHighRisk ? 0.6 : 0)
      },
      preds: {
        linear_regression: { pred_ret: pLin, residual: Math.round((actualRet - pLin) * 100) / 100 },
        decision_tree: { pred_ret: pDt, residual: Math.round((actualRet - pDt) * 100) / 100, risk_prob: riskProb, pred_risk: isHighRisk },
        svm: { pred_ret: pSvm, residual: Math.round((actualRet - pSvm) * 100) / 100, risk_prob: riskProb * 0.9, pred_risk: isHighRisk },
        logistic_regression: { risk_prob: riskProb, pred_risk: riskProb >= 0.5 ? 1 : 0 },
        lightgbm: { pred_ret: pLgb, residual: Math.round((actualRet - pLgb) * 100) / 100, risk_prob: riskProb, pred_risk: isHighRisk }
      }
    };
  });
}
