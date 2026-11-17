import type { Metadata } from "next";
import { ErrorHist, ForestPlot, MetricCompare, PredScatter } from "@/components/charts/ResultCharts";
import { Card, Kpi, PageHead } from "@/components/ui";
import { C, forecastColor } from "@/lib/colors";
import { ASSETS, BASE, EWMA, FAM, FORECASTS, HIST, improvement, MKEY, R, shortName, significant, site } from "@/lib/data";
import { f2, f3, fmtInt, pct0 } from "@/lib/stats";

export const metadata: Metadata = { title: "Test results" };

export default function ResultsPage() {
  const tm = R.test_metrics;
  const m = tm[MKEY];
  const boot = R.test_bootstrap_vs_baselines;
  const forecasts = FORECASTS.map((k) => ({ key: k, label: shortName(k), color: forecastColor(k, FAM) }));
  const vs = (k: string) => (
    <Kpi label={`vs ${shortName(k)}`} value={`−${pct0(improvement(k))}`} sub={`ΔRMSE 95% CI ${f2(boot[k].RMSE_diff_CI95[0])} to ${f2(boot[k].RMSE_diff_CI95[1])}`}
      badge={significant(k) ? "significant" : "borderline"} tone={significant(k) ? "good" : "neutral"} />
  );
  return (
    <>
      <PageHead eyebrow="Untouched test set · 2023–2024" title="Test results">
        {FAM} was refit on train + validation and scored once on {fmtInt(m.n)} asset-days it had never seen.
      </PageHead>

      <div className="kpis">
        <Kpi label={`${FAM} test RMSE`} value={f2(m.RMSE)} sub={`MAE ${f2(m.MAE)} · R² ${f2(m.R2)}`} />
        {vs(BASE)}
        {vs(EWMA)}
        {vs(HIST)}
      </div>

      <Card title="Compare forecasts" className="mb">
        <MetricCompare forecasts={forecasts} overall={tm} perAsset={R.test_per_asset} assets={ASSETS} />
      </Card>

      <div className="grid g2">
        <Card title="Is the gain real?" sub="Moving-block bootstrap, 2000 resamples · left of 0 = model better">
          <ForestPlot rows={[BASE, EWMA, HIST].map((k) => ({ label: `vs ${shortName(k)}`, mean: boot[k].RMSE_diff_mean, lo: boot[k].RMSE_diff_CI95[0], hi: boot[k].RMSE_diff_CI95[1], share: boot[k].share_of_resamples_model_better }))} />
          <p className="hint">20-day blocks keep the overlap between neighbouring targets, so intervals are honest about the small effective sample.</p>
        </Card>
        <Card title="All metrics">
          <div className="table-wrap">
            <table>
              <thead><tr><th>Forecast</th><th className="num">RMSE</th><th className="num">MAE</th><th className="num">R²</th><th className="num">QLIKE</th><th className="num">Bias</th></tr></thead>
              <tbody>
                {FORECASTS.map((k, i) => (
                  <tr key={k} className={i === 0 ? "hl" : ""}>
                    <td><i className="swatch" style={{ background: forecastColor(k, FAM), marginRight: 8 }} />{shortName(k)}</td>
                    <td className="num">{f2(tm[k].RMSE)}</td><td className="num">{f2(tm[k].MAE)}</td><td className="num">{f2(tm[k].R2)}</td>
                    <td className="num">{f3(tm[k].QLIKE)}</td><td className="num">{f2(tm[k].Bias)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="hint">QLIKE is the standard volatility loss and penalises under-predicting risk more than over-predicting it.</p>
        </Card>
      </div>

      <div className="grid g2">
        <Card title="Predicted vs realised" sub="Each dot is one asset-day · dashed line = perfect forecast">
          <PredScatter timeline={site.timeline} fam={FAM} />
        </Card>
        <Card title="Error distribution" sub="Forecast − actual · left of 0 = risk under-estimated">
          <ErrorHist timeline={site.timeline} fam={FAM}
            options={[{ k: "recent", label: "Recent vol", color: C.base }, { k: "ewma", label: "EWMA", color: C.ewma }, { k: "hist", label: "Hist. avg", color: C.hist }]} />
        </Card>
      </div>
    </>
  );
}
