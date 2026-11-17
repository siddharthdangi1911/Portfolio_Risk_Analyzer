// Server-side access to the pipeline outputs. Pages import from here at build
// time and pass only the slices each chart needs to client components.
import raw from "@/data/site_data.json";
import type { SiteData } from "./types";

export const site = raw as unknown as SiteData;
export const R = site.results;
export const FAM = R.final_model.family;
export const MKEY = `${FAM} (final)`;
export const BASE = site.primary_baseline;
export const HIST = "Historical average";
export const EWMA = "EWMA vol (RiskMetrics)";
export const ASSETS = site.assets;
export const FAMILIES = R.stage1_screening.results.filter((r) => r.kind === "model").map((r) => r.name);

/** Forecasts compared on the test set, in a fixed order (colour follows the forecast, not its rank). */
export const FORECASTS = [MKEY, EWMA, HIST, BASE];

export const shortName = (k: string) =>
  k.replace(" vol (RiskMetrics)", "").replace(" (final)", "").replace(" (tuned)", "");

export const featureName = (f: string) => {
  const info = R.feature_info[f.replace(/^log_/, "")];
  if (!info) return f.startsWith("is_") ? `Asset is ${f.slice(3)}` : f;
  return info[0];
};

export const improvement = (base: string) => 1 - R.test_metrics[MKEY].RMSE / R.test_metrics[base].RMSE;
export const significant = (base: string) => R.test_bootstrap_vs_baselines[base].RMSE_diff_CI95[1] < 0;

export const paramText = (p: Record<string, unknown> | null) =>
  p
    ? Object.entries(p)
        .filter(([k]) => k !== "log_target")
        .map(([k, v]) => `${k} = ${v}`)
        .join(", ") + (p.log_target ? " · log target" : "")
    : "–";
