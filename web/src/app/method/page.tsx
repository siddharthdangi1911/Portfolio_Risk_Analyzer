import type { Metadata } from "next";
import { HBars } from "@/components/charts/common";
import { Card, Kpi, PageHead } from "@/components/ui";
import { C } from "@/lib/colors";
import { FAM, featureName, R } from "@/lib/data";
import { f2, f3, fmtInt } from "@/lib/stats";

export const metadata: Metadata = { title: "Method" };

const PREP = [
  ["Log returns", "They add over time; volatility = std of daily returns × √252."],
  ["Log target and log volatility features", "Volatility is positive and skewed; in logs the relation is near-linear. Back-transform uses Duan’s smearing factor from training data."],
  ["Scaler inside the pipeline", "Refit on each fold’s training rows only, so no validation or test statistics leak in."],
  ["20-day embargo", "Rows whose target window crosses a split boundary are dropped; CV folds keep a 20-day gap."],
  ["Date-based folds", "All ETFs of one day stay on the same side of every split."],
];

export default function MethodPage() {
  const tp = R.training_proof;
  const rp = R.reproducibility;
  const imp = R.permutation_importance_test.slice(0, 10);
  const coefs = tp?.learned_coefficients_scaled_log_space ? Object.entries(tp.learned_coefficients_scaled_log_space).slice(0, 10) : [];
  return (
    <>
      <PageHead eyebrow="Preprocessing, integrity, reproducibility" title="Method">
        What the model learned, proof that training really happens, and how to rerun everything.
      </PageHead>

      {tp && (
        <div className="kpis">
          <Kpi label="Model fits run" value={tp.total_model_fits} sub="CV + tuning + final + control" />
          <Kpi label="Final fit time" value={`${(tp.final_fit_seconds * 1000).toFixed(1)} s`} sub={`${fmtInt(tp.training_rows_final_fit)} rows × ${tp.n_features} features`} />
          <Kpi label="Test RMSE, real labels" value={f2(tp.real_target_test_RMSE)} />
          <Kpi label="Test RMSE, shuffled labels" value={f2(tp.shuffled_target_test_RMSE)} badge="real labels clearly better" tone="good" />
        </div>
      )}

      <div className="grid g2">
        <Card title="What drives the forecast" sub="Permutation importance: RMSE increase when a feature is shuffled (test set)">
          <HBars height="lg" rows={imp.map((x) => ({ label: featureName(x.feature), value: x.importance, color: C.model }))} xLabel="RMSE increase" digits={3} />
        </Card>
        <Card title={`${FAM} coefficients`} sub="Effect on log-volatility per 1 sd of the feature · red raises, blue lowers">
          <HBars height="lg" rows={coefs.map(([f, v]) => ({ label: featureName(f), value: v, color: v >= 0 ? C.pos : C.neg }))} xLabel="Coefficient" digits={3} zeroLine domain={["auto", "auto"]} />
        </Card>
      </div>

      <div className="flex w-full">
        <Card title="Preprocessing choices">
          <ul className="clean">{PREP.map(([t, d]) => <li key={t}><span><b>{t}.</b> {d}</span></li>)}</ul>
        </Card>
      </div>
    </>
  );
}
