// Shape of web/src/data/site_data.json (written by scripts/export_web_data.py).
export type Metrics = { RMSE: number; MAE: number; R2: number; QLIKE: number; Bias: number; n: number };
export type MetricName = "RMSE" | "MAE" | "R2" | "QLIKE" | "Bias";
export type Num = number | null;
export type Series = {
  dates: string[];
  actual: Num[];
  model: Num[];
  recent: Num[];
  hist: Num[];
  ewma: Num[];
  vix: Num[];
};
export type Params = Record<string, number | string | boolean | null>;
export type ScreenResult = {
  name: string;
  kind: "baseline" | "model";
  cv_rmse_mean: number;
  cv_rmse_std: number;
  fold_rmse: number[];
  best_params: Params | null;
  fits: number;
  seconds: number;
};
export type Trial = { params: Params; val_rmse: number; val_mae: number };
export type Boot = {
  model_RMSE_CI95: [number, number];
  baseline_RMSE_CI95: [number, number];
  RMSE_diff_mean: number;
  RMSE_diff_CI95: [number, number];
  share_of_resamples_model_better: number;
};
export type RegimeCell = { RMSE: number | null; MAE: number | null };
export type Regime = { regime: string; n: number; [k: string]: RegimeCell | string | number };
export type ErrorRow = { date: string; asset: string; actual: number; predicted: number; baseline: number; vix: number; error: number };
export type PortfolioMetrics = { ann_return_pct: number; ann_vol_pct: number; sharpe_rf0: number; max_drawdown_pct: number };

export type Results = {
  data: {
    assets: string[];
    quality: {
      source: string;
      duplicate_dates_removed: number;
      missing_values_before: Record<string, number>;
      non_positive_prices: Record<string, number>;
      rows_after_alignment: number;
      date_range: [string, string];
      daily_moves_over_10pct: Record<string, { date: string; log_return_pct: number }[]>;
      calendar_gaps_over_4_days: number;
    };
    n_rows_model_table: number;
    features: string[];
  };
  splits: Record<"train" | "val" | "test", { start: string; end: string; dates: number; rows: number }>;
  stage1_screening: {
    folds: { fold: number; train_end: string; score_start: string; score_end: string; train_rows: number; score_rows: number }[];
    results: ScreenResult[];
    selected_family: string;
  };
  stage2_tuning: { family: string; n_configs: number; seconds: number; best_params: Params; top_trials: Trial[]; all_trials?: Trial[] };
  validation_comparison: Record<string, Metrics>;
  final_model: { family: string; params: Params; trained_on: string; trained_through: string; fit_seconds: number };
  training_proof?: {
    total_model_fits: number;
    training_rows_final_fit: number;
    n_features: number;
    final_fit_seconds: number;
    why_fast: string;
    shuffled_target_test_RMSE: number;
    real_target_test_RMSE: number;
    learned_coefficients_scaled_log_space: Record<string, number> | null;
    smearing_factor: number;
  };
  test_metrics: Record<string, Metrics>;
  test_bootstrap_vs_baselines: Record<string, Boot>;
  test_per_asset: Record<string, Record<string, Metrics>>;
  test_regimes: Regime[];
  vix_regime_thresholds_from_train: [number, number];
  representative_errors: ErrorRow[];
  permutation_importance_test: { feature: string; importance: number; std: number }[];
  feature_info: Record<string, [string, string, string]>;
  reproducibility: { seed: number; python: string; sklearn: string; pandas: string; numpy: string; runtime_seconds: number };
};

export type Portfolio = {
  return_dates: string[];
  returns_pct: Record<string, number[]>;
  forecast_dates: string[];
  pred_model: Record<string, Num[]>;
  pred_baseline: Record<string, Num[]>;
  corr60: Record<string, Num[]>;
  corr_train: number[][];
  corr_test: number[][];
  presets: Record<string, Record<string, number>>;
  preset_metrics_test: Record<string, PortfolioMetrics>;
  single_asset_metrics_test: Record<string, PortfolioMetrics>;
};

export type SiteData = {
  generated: string;
  assets: string[];
  asset_descriptions: Record<string, string>;
  results: Results;
  primary_baseline: string;
  timeline: Record<string, Series>;
  val_timeline: Record<string, Series>;
  portfolio: Portfolio;
  history: { dates: string[]; close: Record<string, Num[]> };
};
