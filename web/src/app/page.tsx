import Link from "next/link";
import { HBars } from "@/components/charts/common";
import { ForecastByAsset } from "@/components/charts/OverviewCharts";
import { Callout, Card, Kpi, PageHead } from "@/components/ui";
import { forecastColor } from "@/lib/colors";
import { ASSETS, BASE, EWMA, FAM, FAMILIES, FORECASTS, HIST, improvement, MKEY, R, shortName, significant, site } from "@/lib/data";
import { f2, fmtInt, pct0 } from "@/lib/stats";

export default function Overview() {
  const tm = R.test_metrics;
  const m = tm[MKEY];
  const lgb = R.stage1_screening.results.find((r) => r.name === "LightGBM");
  return (
    <>
      <PageHead eyebrow="" title="Can ML forecast next-month volatility better than history?">
        At each day’s close we forecast how much an ETF will move over the next 20 trading days, then check the forecast on years the model never saw.
      </PageHead>

      <div className="kpis">
        <Kpi label="Selected model" value={FAM} sub={`best of ${FAMILIES.length} by cross-validation`} />
        <Kpi label="Test RMSE" value={f2(m.RMSE)} sub="volatility points" badge={`${pct0(improvement(BASE))} below recent vol`} tone="good" />
        <Kpi label="vs EWMA" value={`−${pct0(improvement(EWMA))}`} badge={significant(EWMA) ? "significant" : "not conclusive"} tone={significant(EWMA) ? "good" : "neutral"} />
        <Kpi label="Test R²" value={f2(m.R2)} sub={`recent vol: ${f2(tm[BASE].R2)}`} />
        <Kpi label="Asset-days" value={fmtInt(R.data.n_rows_model_table)} sub={ASSETS.join(" · ")} />
      </div>

      <div className="grid g-21">
        <Card title="Forecast vs reality" sub="Test period 2023–24">
          <ForecastByAsset timeline={site.timeline} fam={FAM} />
        </Card>
        <Card title="Test error by forecast" sub="RMSE, lower is better">
          <HBars rows={FORECASTS.map((k) => ({ label: shortName(k), value: tm[k].RMSE, color: forecastColor(k, FAM) }))} xLabel="RMSE" height="sm" />
          <ul className="clean" style={{ marginTop: 14 }}>
            <li><span><b>{FAM}</b> won every CV fold{lgb ? `; LightGBM was runner-up` : ""}.</span></li>
            <li><span>Gain over recent vol and EWMA is beyond bootstrap noise; over the {HIST.toLowerCase()} it is borderline.</span></li>
            <li><span>Biggest misses: one-day VIX spikes and sudden, unsignalled stress.</span></li>
          </ul>
        </Card>
      </div>

      <Card title="How the model was chosen" className="mb">
        <div className="steps">
          <Link href="/models/" className="step"><span className="step-n">STAGE 1</span><b>Screen {FAMILIES.length} models</b><span>Cross-validation on 2012–2020 only</span></Link>
          <Link href="/models/" className="step"><span className="step-n">PICK</span><b>{FAM} wins</b><span>Lowest CV error; only it continues</span></Link>
          <Link href="/models/" className="step"><span className="step-n">STAGE 2</span><b>Tune on validation</b><span>{R.stage2_tuning.n_configs} settings scored on 2021–22</span></Link>
          <Link href="/results/" className="step"><span className="step-n">STAGE 3</span><b>Test once</b><span>Refit, then score 2023–24 a single time</span></Link>
        </div>
      </Card>
    </>
  );
}
