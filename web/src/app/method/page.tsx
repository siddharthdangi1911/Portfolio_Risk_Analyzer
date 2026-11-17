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
          <Kpi label="Final fit time" value={`${(tp.final_fit_seconds * 1000).toFixed(1)} ms`} sub={`${fmtInt(tp.training_rows_final_fit)} rows × ${tp.n_features} features`} />
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

      <div className="grid g2">
        <Card title="Preprocessing choices">
          <ul className="clean">{PREP.map(([t, d]) => <li key={t}><span><b>{t}.</b> {d}</span></li>)}</ul>
        </Card>
        <Card title="Why not ROC AUC?">
          <ul className="clean">
            <li><span><b>ROC AUC is for classification</b> (yes/no outcomes ranked by a probability). Our target is a continuous number, so there is no ROC curve to draw.</span></li>
            <li><span><b>We use regression metrics:</b> RMSE to select models, plus MAE, R², bias and QLIKE, the standard volatility loss.</span></li>
            <li><span>AUC would only fit a reframed task, e.g. “will volatility exceed 25%?”, which throws away information and is not our question.</span></li>
          </ul>
        </Card>
      </div>

      <div className="grid g2">
        <Card title="Reproduce">
          <pre>{`pip install -r requirements.txt
python scripts/run_all.py --refresh-data
python -m src.inference --weights SPY=0.6,TLT=0.4
cd web && npm install && npm run export:docs`}</pre>
          <p className="hint">Public Yahoo Finance data, no API keys. Every number on this site is read from the pipeline output at build time.</p>
        </Card>
        <Card title="Run details">
          <dl className="facts">
            <dt>Final model</dt><dd>{FAM} · α = {String(R.final_model.params.alpha ?? "–")} · trained through {R.final_model.trained_through}</dd>
            <dt>Seed</dt><dd>{rp.seed}</dd>
            <dt>Versions</dt><dd>Python {rp.python} · scikit-learn {rp.sklearn} · pandas {rp.pandas}</dd>
            <dt>Runtime</dt><dd>{rp.runtime_seconds} s for the full pipeline</dd>
            {tp && <><dt>Smearing factor</dt><dd>{f3(tp.smearing_factor)}</dd></>}
            <dt>Saved model</dt><dd><code>artifacts/final_model.joblib</code>, the only model used for inference</dd>
          </dl>
        </Card>
      </div>
    </>
  );
}
