import type { Metadata } from "next";
import { Fragment } from "react";
import { FeatureTable, Heatmap, PriceHistory, ReturnDist, VolHistory } from "@/components/charts/DataCharts";
import { Card, Kpi, PageHead } from "@/components/ui";
import { ASSETS, R, site } from "@/lib/data";
import { fmtInt } from "@/lib/stats";

export const metadata: Metadata = { title: "Data" };

export default function DataPage() {
  const q = R.data.quality;
  const sp = R.splits;
  const total = sp.train.dates + sp.val.dates + sp.test.dates;
  const bands = [
    { from: sp.train.start, to: sp.train.end, color: "var(--band-train)", label: "TRAIN" },
    { from: sp.val.start, to: sp.val.end, color: "var(--band-val)", label: "VALIDATION" },
    { from: sp.test.start, to: sp.test.end, color: "var(--band-test)", label: "TEST" },
  ];
  const missing = Object.values(q.missing_values_before).reduce((s, x) => s + x, 0) + Object.values(q.non_positive_prices).reduce((s, x) => s + x, 0);
  const moves = Object.entries(q.daily_moves_over_10pct).flatMap(([a, l]) => l.map((x) => `${a} ${x.date} (${x.log_return_pct}%)`));
  const features = R.data.features.map((f) => {
    const info = R.feature_info[f.replace(/^log_/, "")];
    return info
      ? { code: f, name: info[0], group: info[1], meaning: info[2], log: f.startsWith("log_") }
      : { code: f, name: `Asset is ${f.slice(3)}`, group: "Identity", meaning: "1 for this ETF, else 0", log: false };
  });

  return (
    <>
      <PageHead eyebrow="Data understanding" title="Data">
        Dividend-adjusted daily closes from Yahoo Finance (public, via <code>yfinance</code>, no credentials), {q.date_range[0].slice(0, 4)}–{q.date_range[1].slice(0, 4)}.
      </PageHead>

      <div className="kpis">
        <Kpi label="Trading days" value={fmtInt(q.rows_after_alignment)} sub={`${q.date_range[0]} → ${q.date_range[1]}`} />
        <Kpi label="Asset-day rows" value={fmtInt(R.data.n_rows_model_table)} sub="after 120-day warm-up" />
        <Kpi label="ETFs" value={ASSETS.length} sub="+ VIX as a feature" />
        <Kpi label="Features" value={R.data.features.length} sub="all known at day t" />
        <Kpi label="Missing or bad prices" value={missing} sub="before alignment" />
      </div>

      <Card title="Volatility history and the chronological split" sub="Trailing 20-day volatility · drag the handles below the chart to zoom" className="mb">
        <VolHistory history={site.history} assets={ASSETS} bands={bands} />
        <div className="splitbar" role="img" aria-label="Train, validation and test periods">
          <div style={{ flex: sp.train.dates, background: "var(--s-model)" }}>Train {sp.train.start.slice(0, 4)}–{sp.train.end.slice(0, 4)} · {fmtInt(sp.train.rows)} rows</div>
          <div className="gap" style={{ flex: total * 0.015 }} />
          <div style={{ flex: sp.val.dates, background: "var(--s-ewma)" }}>Validation · {fmtInt(sp.val.rows)}</div>
          <div className="gap" style={{ flex: total * 0.015 }} />
          <div style={{ flex: sp.test.dates, background: "var(--s-base)" }}>Test · {fmtInt(sp.test.rows)}</div>
        </div>
        <p className="hint">Hatched gaps are a 20-day embargo: rows whose target window would cross into the next period are dropped.</p>
      </Card>

      <Card title="Price history" sub="Rebased to 100 · drag the handles to zoom" className="mb">
        <PriceHistory history={site.history} assets={ASSETS} />
      </Card>

      <div className="grid g2">
        <Card title="Daily returns have fat tails" sub="Histogram vs a normal curve with the same mean and spread">
          <ReturnDist history={site.history} assets={ASSETS} />
        </Card>
        <Card title="Correlation of daily returns" sub="Low or negative values mean diversification">
          <Heatmap assets={ASSETS} train={site.portfolio.corr_train} test={site.portfolio.corr_test} />
          <dl className="facts" style={{ marginTop: 18 }}>
            {ASSETS.map((a) => (<Fragment key={a}><dt>{a}</dt><dd>{site.asset_descriptions[a]}</dd></Fragment>))}
            <dt>VIX</dt><dd>Market’s 30-day implied volatility (feature only)</dd>
          </dl>
        </Card>
      </div>

      <div className="grid g-12">
        <Card title="Quality checks">
          <dl className="facts">
            <dt>Duplicate dates</dt><dd>{q.duplicate_dates_removed}</dd>
            <dt>Missing / non-positive</dt><dd>{missing}</dd>
            <dt>Gaps over 4 days</dt><dd>{q.calendar_gaps_over_4_days} (market closure)</dd>
            <dt>Moves over 10%</dt><dd>{moves.length ? moves.join(", ") : "none"} — real March 2020 crash days, kept</dd>
            <dt>Adjustment</dt><dd>Dividend and split adjusted</dd>
          </dl>
        </Card>
        <Card title="Features" sub="Returns, volatility windows, VIX and asset identity">
          <FeatureTable rows={features} />
        </Card>
      </div>
    </>
  );
}
