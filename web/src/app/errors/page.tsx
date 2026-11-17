import type { Metadata } from "next";
import { ErrorTable, ErrorVsVix, RegimeChart } from "@/components/charts/ErrorCharts";
import { Card, PageHead } from "@/components/ui";
import { forecastColor } from "@/lib/colors";
import { BASE, FAM, HIST, MKEY, R, shortName, site } from "@/lib/data";
import { f1 } from "@/lib/stats";
import type { RegimeCell } from "@/lib/types";

export const metadata: Metadata = { title: "Errors & limits" };

const CONTEXT: Record<string, string> = {
  "2024-08": "Yen carry-trade unwind: VIX hit 38 for a day, then calmed.",
  "2024-07": "Rotation out of big tech after a very calm half-year.",
  "2024-04": "Rate-cut hopes pushed back; brief pullback.",
  "2023-03": "US regional bank failures (SVB).",
  "2023-10": "Rate-driven sell-off.",
  "2023-05": "Debt-ceiling standoff.",
};

const LIMITS = [
  ["Regime dependence", "The model expects volatility to fall back to normal: right in calm 2023–24, wrong in 2022 when stress persisted."],
  ["One-day shocks", "A VIX spike that fades in days (5 Aug 2024) is read as the start of a crisis."],
  ["Overlapping targets", "Neighbouring days share 19 of 20 target days, so the effective sample is far smaller than the row count."],
  ["Noisy target", "Realised volatility from 20 returns is itself an estimate; even a perfect model shows error."],
  ["Narrow scope", "Four liquid US ETFs, one horizon, one data vendor."],
];

export default function ErrorsPage() {
  const keys = [MKEY, BASE, HIST];
  const th = R.vix_regime_thresholds_from_train;
  const rows = R.test_regimes.map((r) => ({ regime: r.regime, n: r.n, values: Object.fromEntries(keys.map((k) => [k, r[k] as RegimeCell])) }));
  return (
    <>
      <PageHead eyebrow="Errors and limitations" title="Where the model goes wrong">
        How errors change with market stress, the largest misses, and what limits the claim.
      </PageHead>

      <div className="grid g2">
        <Card title="Error by market regime" sub={`VIX terciles from the training years: calm < ${f1(th[0])} < normal < ${f1(th[1])} < stressed`}>
          <RegimeChart rows={rows} forecasts={keys.map((k) => ({ key: k, label: shortName(k), color: forecastColor(k, FAM) }))} />
        </Card>
        <Card title="Error vs VIX" sub="Big over-predictions sit at short-lived VIX spikes">
          <ErrorVsVix timeline={site.timeline} fam={FAM} />
        </Card>
      </div>

      <Card title="Largest test-set misses" sub="Open any row in the Forecast explorer" className="mb">
        <ErrorTable rows={R.representative_errors} context={CONTEXT} />
      </Card>

      <Card title="Limitations">
        <ul className="clean">
          {LIMITS.map(([t, d]) => <li key={t}><span><b>{t}.</b> {d}</span></li>)}
        </ul>
      </Card>
    </>
  );
}
