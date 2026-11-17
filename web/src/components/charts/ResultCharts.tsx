"use client";

import { useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, ComposedChart, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from "recharts";
import { C } from "@/lib/colors";
import { f2, histogram } from "@/lib/stats";
import type { MetricName, Metrics, Series } from "@/lib/types";
import { axisProps, HBars, Seg, Tip } from "./common";
import { Legend } from "../ui";

const METRIC_INFO: Record<MetricName, string> = {
  RMSE: "Root mean squared error in vol points · lower is better",
  MAE: "Typical miss in vol points · lower is better",
  R2: "Share of variance explained · higher is better",
  QLIKE: "Volatility loss that punishes under-predicting risk · lower is better",
  Bias: "Average forecast − actual · near 0 is best; negative = risk under-estimated",
};

type F = { key: string; label: string; color: string };

export function MetricCompare({ forecasts, overall, perAsset, assets }: {
  forecasts: F[]; overall: Record<string, Metrics>; perAsset: Record<string, Record<string, Metrics>>; assets: string[];
}) {
  const [m, setM] = useState<MetricName>("RMSE");
  const signed = m === "R2" || m === "Bias";
  const rows = assets.map((a) => {
    const r: Record<string, string | number> = { a };
    forecasts.forEach((f) => (r[f.label] = perAsset[a][f.key]?.[m] ?? 0));
    return r;
  });
  return (
    <div className="grid g2" style={{ marginBottom: 0 }}>
      <div>
        <div className="controls" style={{ marginBottom: 6 }}>
          <Seg items={(Object.keys(METRIC_INFO) as MetricName[]).map((k) => ({ v: k, t: k === "R2" ? "R²" : k }))} value={m} onChange={setM} label="Metric" />
        </div>
        <div className="card-sub" style={{ marginBottom: 6 }}>{METRIC_INFO[m]}</div>
        <HBars rows={forecasts.map((f) => ({ label: f.label, value: overall[f.key][m], color: f.color }))} xLabel={m === "R2" ? "R²" : m}
          digits={3} zeroLine={signed} domain={signed ? ["auto", "auto"] : [0, "auto"]} />
      </div>
      <div>
        <div className="card-title" style={{ marginBottom: 2 }}>By asset</div>
        <div className="card-sub" style={{ marginBottom: 6 }}>Same metric, split by ETF</div>
        <div className="chart md">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={rows} margin={{ top: 8, right: 8, bottom: 4, left: 0 }} barGap={2} barCategoryGap="22%">
              <CartesianGrid stroke={C.grid} vertical={false} />
              <XAxis dataKey="a" {...axisProps} />
              <YAxis {...axisProps} width={40} />
              <Tooltip cursor={{ fill: "var(--surface-2)" }} content={<Tip fmt={(v) => v.toFixed(3)} />} />
              {forecasts.map((f) => <Bar key={f.key} dataKey={f.label} name={f.label} fill={f.color} radius={[3, 3, 0, 0]} isAnimationActive={false} />)}
            </BarChart>
          </ResponsiveContainer>
        </div>
        <Legend items={forecasts.map((f) => ({ label: f.label, color: f.color }))} />
      </div>
    </div>
  );
}

/** Forest plot of bootstrap confidence intervals for RMSE(model) - RMSE(baseline). */
export function ForestPlot({ rows }: { rows: { label: string; mean: number; lo: number; hi: number; share: number }[] }) {
  const lo = Math.min(-0.2, ...rows.map((r) => r.lo)) * 1.15;
  const hi = Math.max(0.2, ...rows.map((r) => r.hi)) * 1.15;
  const W = 440, L = 140, R = 12, rowH = 46, top = 8, H = top + rows.length * rowH + 34;
  const x = (v: number) => L + ((v - lo) / (hi - lo)) * (W - L - R);
  const ticks = [-1, -0.75, -0.5, -0.25, 0, 0.25].filter((t) => t >= lo && t <= hi);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Bootstrap confidence intervals">
      {ticks.map((t) => (
        <g key={t}>
          <line x1={x(t)} x2={x(t)} y1={top} y2={H - 30} stroke={t === 0 ? "var(--text-3)" : "var(--grid)"} strokeWidth={t === 0 ? 1.5 : 1} />
          <text x={x(t)} y={H - 16} textAnchor="middle" fontSize={11} fill="var(--text-3)">{t}</text>
        </g>
      ))}
      <text x={(x(lo) + x(hi)) / 2 + L / 2} y={H - 2} textAnchor="middle" fontSize={11} fill="var(--text-3)">RMSE(model) − RMSE(baseline)</text>
      {rows.map((r, i) => {
        const y = top + i * rowH + rowH / 2;
        const sig = r.hi < 0;
        const col = sig ? "var(--good)" : "var(--text-3)";
        return (
          <g key={r.label}>
            <title>{`${r.label}: mean ${f2(r.mean)}, 95% CI ${f2(r.lo)} to ${f2(r.hi)}; model better in ${Math.round(r.share * 100)}% of resamples`}</title>
            <text x={0} y={y - 2} fontSize={12.5} fill="var(--text)" fontWeight={600}>{r.label}</text>
            <text x={0} y={y + 14} fontSize={11} fill={col}>{sig ? "significant" : "not conclusive"}</text>
            <rect x={x(r.lo)} y={y - 5} width={x(r.hi) - x(r.lo)} height={10} rx={5} fill={col} opacity={0.35} />
            <circle cx={x(r.mean)} cy={y} r={6} fill="var(--text)" stroke="var(--surface)" strokeWidth={2} />
          </g>
        );
      })}
    </svg>
  );
}

export function PredScatter({ timeline, fam }: { timeline: Record<string, Series>; fam: string }) {
  const assets = Object.keys(timeline);
  const [a, setA] = useState("All");
  const [showBase, setShowBase] = useState(false);
  const pts = (k: "model" | "recent") =>
    (a === "All" ? assets : [a]).flatMap((s) => {
      const T = timeline[s];
      return T.dates.map((d, i) => (T.actual[i] != null && T[k][i] != null ? { x: T[k][i] as number, y: T.actual[i] as number, d, s } : null)).filter((p): p is { x: number; y: number; d: string; s: string } => !!p);
    });
  const mp = pts("model");
  const mx = Math.ceil(Math.max(...mp.map((p) => Math.max(p.x, p.y))) / 5) * 5;
  return (
    <>
      <div className="controls" style={{ marginBottom: 10, justifyContent: "space-between" }}>
        <Seg items={[{ v: "All", t: "All" }, ...assets.map((x) => ({ v: x, t: x }))]} value={a} onChange={setA} label="Asset" />
        <label className="toggle"><input type="checkbox" checked={showBase} onChange={(e) => setShowBase(e.target.checked)} /> Recent vol too</label>
      </div>
      <div className="chart">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 8, right: 12, bottom: 18, left: 4 }}>
            <CartesianGrid stroke={C.grid} />
            <XAxis type="number" dataKey="x" domain={[0, mx]} {...axisProps} label={{ value: "Forecast (%)", position: "insideBottom", offset: -12 }} />
            <YAxis type="number" dataKey="y" domain={[0, mx]} {...axisProps} width={40} label={{ value: "Realised (%)", angle: -90, position: "insideLeft", offset: 12 }} />
            <Tooltip content={({ active, payload }) => {
              const p = payload?.[0]?.payload as { x: number; y: number; d: string; s: string } | undefined;
              if (!active || !p || !p.d) return null;
              return <div className="tip"><div className="tip-title">{p.s} · {p.d}</div><div className="tip-row">Forecast<span>{p.x.toFixed(1)}%</span></div><div className="tip-row">Realised<span>{p.y.toFixed(1)}%</span></div></div>;
            }} />
            <Scatter name="Perfect forecast" data={[{ x: 0, y: 0 }, { x: mx, y: mx }]} line={{ stroke: C.text3, strokeDasharray: "5 4" }} shape={() => <g />} isAnimationActive={false} />
            <Scatter name={`${fam} forecast`} data={mp} fill={C.model} fillOpacity={0.45} isAnimationActive={false} shape="circle" />
            {showBase && <Scatter name="Recent 20-day vol" data={pts("recent")} fill={C.base} fillOpacity={0.4} isAnimationActive={false} shape="circle" />}
          </ScatterChart>
        </ResponsiveContainer>
      </div>
      <Legend items={[{ label: `${fam} forecast`, color: C.model }, ...(showBase ? [{ label: "Recent 20-day vol", color: C.base }] : []), { label: "Perfect forecast", color: C.text3, dashed: true }]} />
    </>
  );
}

export function ErrorHist({ timeline, fam, options }: { timeline: Record<string, Series>; fam: string; options: { k: "recent" | "ewma" | "hist"; label: string; color: string }[] }) {
  const [k, setK] = useState(options[0].k);
  const opt = options.find((o) => o.k === k)!;
  const rows = useMemo(() => {
    const errs = (key: "model" | "recent" | "ewma" | "hist") =>
      Object.values(timeline).flatMap((T) => T.dates.map((_, i) => (T.actual[i] != null && T[key][i] != null ? (T[key][i] as number) - (T.actual[i] as number) : null)).filter((x): x is number => x != null));
    const a = histogram(errs("model"), -20, 20, 1), b = histogram(errs(k), -20, 20, 1);
    return a.map((r, i) => ({ x: r.x, model: r.count, other: b[i].count }));
  }, [timeline, k]);
  return (
    <>
      <div className="controls" style={{ marginBottom: 10 }}>
        <Seg items={options.map((o) => ({ v: o.k, t: `vs ${o.label}` }))} value={k} onChange={setK} label="Compare with" />
      </div>
      <div className="chart">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={rows} margin={{ top: 8, right: 8, bottom: 18, left: 0 }} barGap={0} barCategoryGap={2}>
            <CartesianGrid stroke={C.grid} vertical={false} />
            <XAxis dataKey="x" {...axisProps} interval={4} tickFormatter={(v) => String(Math.round(+v))} label={{ value: "Forecast − actual (vol points)", position: "insideBottom", offset: -12 }} />
            <YAxis {...axisProps} width={40} />
            <Tooltip cursor={{ fill: "var(--surface-2)" }} content={<Tip fmt={(v) => v.toFixed(0)} title={(l) => `Error ≈ ${Math.round(+(l ?? 0))}`} />} />
            <Bar dataKey="model" name={fam} fill={C.model} radius={[2, 2, 0, 0]} isAnimationActive={false} />
            <Bar dataKey="other" name={opt.label} fill={opt.color} radius={[2, 2, 0, 0]} isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <Legend items={[{ label: fam, color: C.model }, { label: opt.label, color: opt.color }]} />
    </>
  );
}
