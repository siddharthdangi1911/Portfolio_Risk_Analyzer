"use client";

import Link from "next/link";
import { useState } from "react";
import { Bar, BarChart, CartesianGrid, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from "recharts";
import { C } from "@/lib/colors";
import { f1 } from "@/lib/stats";
import type { ErrorRow, Series } from "@/lib/types";
import { axisProps, fmtDate, Seg, Tip } from "./common";
import { Legend } from "../ui";

export function RegimeChart({ rows, forecasts }: { rows: { regime: string; n: number; values: Record<string, { RMSE: number | null; MAE: number | null }> }[]; forecasts: { key: string; label: string; color: string }[] }) {
  const [m, setM] = useState<"RMSE" | "MAE">("RMSE");
  const data = rows.map((r) => {
    const o: Record<string, string | number | null> = { regime: `${r.regime.split(" (")[0]} · ${r.n}` };
    forecasts.forEach((f) => (o[f.label] = r.values[f.key]?.[m] ?? null));
    return o;
  });
  return (
    <>
      <div className="controls" style={{ marginBottom: 10 }}>
        <Seg items={[{ v: "RMSE", t: "RMSE" }, { v: "MAE", t: "MAE" }]} value={m} onChange={(v) => setM(v as "RMSE" | "MAE")} label="Metric" />
      </div>
      <div className="chart md">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 8, right: 8, bottom: 4, left: 0 }} barGap={2} barCategoryGap="24%">
            <CartesianGrid stroke={C.grid} vertical={false} />
            <XAxis dataKey="regime" {...axisProps} />
            <YAxis {...axisProps} width={36} />
            <Tooltip cursor={{ fill: "var(--surface-2)" }} content={<Tip />} />
            {forecasts.map((f) => <Bar key={f.key} dataKey={f.label} name={f.label} fill={f.color} radius={[3, 3, 0, 0]} isAnimationActive={false} />)}
          </BarChart>
        </ResponsiveContainer>
      </div>
      <Legend items={forecasts.map((f) => ({ label: f.label, color: f.color }))} />
    </>
  );
}

export function ErrorVsVix({ timeline, fam }: { timeline: Record<string, Series>; fam: string }) {
  const assets = Object.keys(timeline);
  const [a, setA] = useState(assets[0]);
  const T = timeline[a];
  const pts = (k: "model" | "recent") =>
    T.dates.map((d, i) => (T.actual[i] != null && T[k][i] != null && T.vix[i] != null ? { x: T.vix[i] as number, y: (T[k][i] as number) - (T.actual[i] as number), d } : null)).filter(Boolean) as { x: number; y: number; d: string }[];
  return (
    <>
      <div className="controls" style={{ marginBottom: 10 }}>
        <Seg items={assets.map((x) => ({ v: x, t: x }))} value={a} onChange={setA} label="Asset" />
      </div>
      <div className="chart md">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 8, right: 12, bottom: 18, left: 4 }}>
            <CartesianGrid stroke={C.grid} />
            <XAxis type="number" dataKey="x" domain={["auto", "auto"]} {...axisProps} label={{ value: "VIX on forecast date", position: "insideBottom", offset: -12 }} />
            <YAxis type="number" dataKey="y" {...axisProps} width={40} label={{ value: "Forecast − actual", angle: -90, position: "insideLeft", offset: 12 }} />
            <ReferenceLine y={0} stroke={C.text3} />
            <Tooltip content={({ active, payload }) => {
              const p = payload?.[0]?.payload as { x: number; y: number; d: string } | undefined;
              if (!active || !p) return null;
              return <div className="tip"><div className="tip-title">{fmtDate(p.d)}</div><div className="tip-row">VIX<span>{f1(p.x)}</span></div><div className="tip-row">Error<span>{f1(p.y)}</span></div></div>;
            }} />
            <Scatter name={`${fam} error`} data={pts("model")} fill={C.model} fillOpacity={0.5} isAnimationActive={false} />
            <Scatter name="Recent-vol error" data={pts("recent")} fill={C.base} fillOpacity={0.4} isAnimationActive={false} />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
      <Legend items={[{ label: `${fam} error`, color: C.model }, { label: "Recent-vol error", color: C.base }]} />
    </>
  );
}

export function ErrorTable({ rows, context }: { rows: ErrorRow[]; context: Record<string, string> }) {
  return (
    <div className="table-wrap">
      <table>
        <thead><tr><th>Date</th><th>Asset</th><th className="num">Realised</th><th className="num">Forecast</th><th className="num">VIX</th><th>What happened</th><th /></tr></thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.date + r.asset}>
              <td style={{ whiteSpace: "nowrap" }}>{fmtDate(r.date)}</td>
              <td><b style={{ fontWeight: 600 }}>{r.asset}</b></td>
              <td className="num">{f1(r.actual)}</td>
              <td className="num"><span className={r.error > 0 ? "lose" : "win"}>{f1(r.predicted)}</span></td>
              <td className="num">{f1(r.vix)}</td>
              <td className="small" style={{ color: "var(--text-2)" }}>{r.error > 0 ? "Over-predicted. " : "Under-predicted. "}{context[r.date.slice(0, 7)] ?? ""}</td>
              <td><Link href={`/explorer/?asset=${r.asset}&date=${r.date}`} className="small" style={{ whiteSpace: "nowrap" }}>Inspect →</Link></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
