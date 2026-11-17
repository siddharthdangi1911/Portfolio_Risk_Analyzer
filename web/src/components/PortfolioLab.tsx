"use client";

import { useMemo, useState } from "react";
import { CartesianGrid, LabelList, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from "recharts";
import { CAT, C } from "@/lib/colors";
import { bias, f1, f2, mae, mean, rmse, std } from "@/lib/stats";
import type { Portfolio } from "@/lib/types";
import { axisProps, TimeSeries } from "./charts/common";
import { Heatmap } from "./charts/DataCharts";
import { Kpi } from "./ui";

const H = 20;

function simulate(P: Portfolio, assets: string[], w: Record<string, number>) {
  const n = P.return_dates.length;
  const port = new Array<number>(n).fill(0);
  assets.forEach((a) => { const r = P.returns_pct[a]; for (let i = 0; i < n; i++) port[i] += (w[a] ?? 0) * (Math.exp(r[i] / 100) - 1); });
  let wealth = 1, peak = 1, mdd = 0;
  const path: number[] = [];
  for (const x of port) { wealth *= 1 + x; peak = Math.max(peak, wealth); mdd = Math.min(mdd, wealth / peak - 1); path.push(wealth * 100); }
  const mu = mean(port), sd = Math.sqrt(port.reduce((s, x) => s + (x - mu) ** 2, 0) / (n - 1));
  const annRet = Math.pow(wealth, 252 / n) - 1, annVol = sd * Math.sqrt(252);
  return { port, path, annRet, annVol, sharpe: annRet / annVol, mdd };
}

export function PortfolioLab({ P, assets, fam }: { P: Portfolio; assets: string[]; fam: string }) {
  const [raw, setRaw] = useState<Record<string, number>>(() => Object.fromEntries(assets.map((a) => [a, 1 / assets.length])));
  const total = assets.reduce((s, a) => s + raw[a], 0);
  const w = useMemo(() => Object.fromEntries(assets.map((a) => [a, total > 0 ? raw[a] / total : 1 / assets.length])), [raw, total, assets]);
  const r = useMemo(() => simulate(P, assets, w), [P, assets, w]);
  const singles = useMemo(() => Object.fromEntries(assets.map((a) => [a, simulate(P, assets, { [a]: 1 }).path])), [P, assets]);
  const avgVol = assets.reduce((s, a) => s + w[a] * P.single_asset_metrics_test[a].ann_vol_pct, 0);

  const pv = useMemo(() => {
    const pos: Record<string, number> = {};
    P.return_dates.forEach((d, i) => (pos[d] = i));
    const corr = (k: number, a: string, b: string) => {
      if (a === b) return 1;
      const key = P.corr60[`${a}|${b}`] ? `${a}|${b}` : `${b}|${a}`;
      return P.corr60[key][k];
    };
    const fc = (src: "pred_model" | "pred_baseline", k: number) => {
      let s = 0;
      for (const a of assets) for (const b of assets) {
        const va = P[src][a][k], vb = P[src][b][k], c = corr(k, a, b);
        if (va == null || vb == null || c == null) return null;
        s += w[a] * w[b] * va * vb * c;
      }
      return Math.sqrt(s);
    };
    const dates: string[] = [], act: number[] = [], pm: (number | null)[] = [], pb: (number | null)[] = [];
    P.forecast_dates.forEach((d, k) => {
      const i = pos[d];
      if (i == null || i + H >= P.return_dates.length) return;
      const win = r.port.slice(i + 1, i + 1 + H).map((x) => Math.log(1 + x));
      dates.push(d); act.push(std(win) * Math.sqrt(252) * 100); pm.push(fc("pred_model", k)); pb.push(fc("pred_baseline", k));
    });
    return { dates, act, pm, pb };
  }, [P, assets, w, r.port]);

  const presets = P.preset_metrics_test;
  const rr = Object.entries(presets).map(([k, m]) => ({ x: m.ann_vol_pct, y: m.ann_return_pct, n: k }));

  return (
    <>
      <div className="grid g-12">
        <section className="card">
          <div className="card-head"><div><h2 className="card-title">Weights</h2><div className="card-sub">Drag a slider or pick a preset · weights are normalised to 100%</div></div></div>
          {assets.map((a, i) => (
            <div className="weight-row" key={a}>
              <b><i className="dot" style={{ background: CAT[i], marginRight: 6 }} />{a}</b>
              <input type="range" min={0} max={100} step={5} value={Math.round(raw[a] * 100)} aria-label={`Weight of ${a}`}
                onChange={(e) => setRaw({ ...raw, [a]: +e.target.value / 100 })} />
              <span>{Math.round(w[a] * 100)}%</span>
            </div>
          ))}
          <div className="controls" style={{ margin: "14px 0 16px" }}>
            {Object.entries(P.presets).map(([k, p]) => (
              <button key={k} type="button" className="chip" onClick={() => setRaw(Object.fromEntries(assets.map((a) => [a, p[a] ?? 0])))}>{k}</button>
            ))}
          </div>
          <div className="kpis" style={{ gridTemplateColumns: "repeat(2, minmax(0,1fr))", margin: 0 }}>
            <Kpi label="Annual return" value={`${f1(r.annRet * 100)}%`} />
            <Kpi label="Annual volatility" value={`${f1(r.annVol * 100)}%`} sub={`Assets alone, weighted: ${f1(avgVol)}%`} />
            <Kpi label="Sharpe (rf = 0)" value={f2(r.sharpe)} />
            <Kpi label="Max drawdown" value={`${f1(r.mdd * 100)}%`} />
          </div>
          <p className="hint">Test period, daily rebalancing. Diversification shows when portfolio volatility is below the weighted average of the assets.</p>
        </section>
        <section className="card">
          <div className="card-head"><div><h2 className="card-title">Growth of 100</h2><div className="card-sub">Your portfolio against each asset alone, 2023–2024</div></div></div>
          <TimeSeries dates={P.return_dates} unit="" yLabel="Value" height="lg" zero={false}
            lines={[{ key: "p", label: "Your portfolio", color: C.text, data: r.path, width: 2.8 }, ...assets.map((a, i) => ({ key: a, label: a, color: CAT[i], data: singles[a], width: 1.3 }))]} />
        </section>
      </div>
      <section className="card mb">
        <div className="card-head"><div><h2 className="card-title">Portfolio risk forecast vs what happened</h2>
          <div className="card-sub">σₚ = √(w′DCDw): per-asset vol forecasts D with the trailing 60-day correlation C · updates with your weights</div></div></div>
        <TimeSeries dates={pv.dates} yLabel="Volatility (%)"
          lines={[
            { key: "a", label: "Realised (next 20 days)", color: C.actual, data: pv.act, width: 2.2 },
            { key: "m", label: `From ${fam} forecasts`, color: C.model, data: pv.pm },
            { key: "b", label: "From recent 20-day vols", color: C.base, data: pv.pb, width: 1.4 },
          ]} />
        <div className="table-wrap" style={{ marginTop: 10 }}>
          <table>
            <thead><tr><th>Portfolio forecast built from</th><th className="num">RMSE</th><th className="num">MAE</th><th className="num">Bias</th></tr></thead>
            <tbody>
              <tr className={rmse(pv.act, pv.pm) <= rmse(pv.act, pv.pb) ? "hl" : ""}><td>{fam} asset forecasts</td><td className="num">{f2(rmse(pv.act, pv.pm))}</td><td className="num">{f2(mae(pv.act, pv.pm))}</td><td className="num">{f2(bias(pv.act, pv.pm))}</td></tr>
              <tr className={rmse(pv.act, pv.pm) > rmse(pv.act, pv.pb) ? "hl" : ""}><td>Recent 20-day vols</td><td className="num">{f2(rmse(pv.act, pv.pb))}</td><td className="num">{f2(mae(pv.act, pv.pb))}</td><td className="num">{f2(bias(pv.act, pv.pb))}</td></tr>
            </tbody>
          </table>
        </div>
      </section>
      <div className="grid g2">
        <section className="card">
          <div className="card-head"><div><h2 className="card-title">Risk vs return</h2><div className="card-sub">Presets over the test period · hollow dot = your weights</div></div></div>
          <div className="chart md">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 12, right: 24, bottom: 18, left: 4 }}>
                <CartesianGrid stroke={C.grid} />
                <XAxis type="number" dataKey="x" domain={["auto", "auto"]} {...axisProps} unit="%" label={{ value: "Annual volatility", position: "insideBottom", offset: -12 }} />
                <YAxis type="number" dataKey="y" domain={["auto", "auto"]} {...axisProps} unit="%" width={44} label={{ value: "Annual return", angle: -90, position: "insideLeft", offset: 12 }} />
                <Tooltip content={({ active, payload }) => {
                  const p = payload?.[0]?.payload as { x: number; y: number; n: string } | undefined;
                  if (!active || !p) return null;
                  return <div className="tip"><div className="tip-title">{p.n}</div><div className="tip-row">Return<span>{f1(p.y)}%</span></div><div className="tip-row">Volatility<span>{f1(p.x)}%</span></div></div>;
                }} />
                <Scatter data={rr} fill={C.model} isAnimationActive={false}>
                  <LabelList dataKey="n" position="right" style={{ fill: "var(--text-2)", fontSize: 11 }} />
                </Scatter>
                <Scatter data={[{ x: r.annVol * 100, y: r.annRet * 100, n: "Your weights" }]} fill="var(--surface)" stroke={C.text} strokeWidth={2.5} isAnimationActive={false} />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </section>
        <section className="card">
          <div className="card-head"><div><h2 className="card-title">Daily return correlation</h2><div className="card-sub">Stocks and bonds moved together more in 2023–24, weakening the 60/40 hedge</div></div></div>
          <Heatmap assets={assets} train={P.corr_train} test={P.corr_test} />
        </section>
      </div>
    </>
  );
}
