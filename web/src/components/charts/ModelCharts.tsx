"use client";

import { useState } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from "recharts";
import { C } from "@/lib/colors";
import type { Trial } from "@/lib/types";
import { axisProps, Tip } from "./common";
import { Legend } from "../ui";

export function FoldChart({ folds, series }: { folds: string[]; series: { name: string; color: string; data: number[]; selected?: boolean; dashed?: boolean }[] }) {
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const rows = folds.map((f, i) => {
    const r: Record<string, string | number> = { f };
    series.forEach((s) => (r[s.name] = s.data[i]));
    return r;
  });
  const flip = (n: string) => setHidden((h) => { const x = new Set(h); if (x.has(n)) x.delete(n); else x.add(n); return x; });
  return (
    <>
      <div className="chart">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={rows} margin={{ top: 8, right: 16, bottom: 4, left: 4 }}>
            <CartesianGrid stroke={C.grid} vertical={false} />
            <XAxis dataKey="f" {...axisProps} interval={0} tick={{ fontSize: 11 }} />
            <YAxis {...axisProps} width={40} label={{ value: "RMSE", angle: -90, position: "insideLeft", offset: 12 }} />
            <Tooltip content={<Tip />} cursor={{ stroke: C.text3, strokeDasharray: "3 3" }} />
            {series.filter((s) => !hidden.has(s.name)).map((s) => (
              <Line key={s.name} dataKey={s.name} name={s.name} stroke={s.color} strokeWidth={s.selected ? 3 : 1.6} strokeDasharray={s.dashed ? "5 4" : undefined}
                dot={{ r: s.selected ? 4 : 3, strokeWidth: 0, fill: s.color }} isAnimationActive={false} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="controls" style={{ marginTop: 10 }}>
        {series.map((s) => (
          <button key={s.name} type="button" className={`chip ${hidden.has(s.name) ? "off" : "on"}`} aria-pressed={!hidden.has(s.name)} onClick={() => flip(s.name)}>
            <i className="swatch" style={{ background: s.dashed ? `repeating-linear-gradient(90deg, ${s.color} 0 4px, transparent 4px 7px)` : s.color }} /> {s.name}
          </button>
        ))}
      </div>
    </>
  );
}

export function TuningCurve({ trials }: { trials: Trial[] }) {
  const pts = (lt: boolean) =>
    trials.filter((t) => !!t.params.log_target === lt).map((t) => ({ x: Number(t.params.alpha), y: t.val_rmse })).sort((a, b) => a.x - b.x);
  const best = [...trials].sort((a, b) => a.val_rmse - b.val_rmse)[0];
  return (
    <>
    <div className="chart md">
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 8, right: 16, bottom: 18, left: 4 }}>
          <CartesianGrid stroke={C.grid} />
          <XAxis type="number" dataKey="x" scale="log" domain={["auto", "auto"]} {...axisProps} tickFormatter={(v) => (v >= 1 ? String(v) : v.toExponential(0))}
            label={{ value: "Regularisation α (log scale)", position: "insideBottom", offset: -12 }} />
          <YAxis type="number" dataKey="y" {...axisProps} width={44} domain={["auto", "auto"]} label={{ value: "Validation RMSE", angle: -90, position: "insideLeft", offset: 10 }} />
          <Tooltip content={<Tip fmt={(v, n) => (n === "α" ? String(v) : v.toFixed(3))} title={() => "Ridge setting"} />} />
          <Scatter name="Log-volatility target" data={pts(true)} fill={C.model} line={{ stroke: C.model, strokeWidth: 2 }} isAnimationActive={false} />
          <Scatter name="Raw volatility target" data={pts(false)} fill={C.base} line={{ stroke: C.base, strokeWidth: 2 }} isAnimationActive={false} />
          <Scatter name="Selected" data={[{ x: Number(best.params.alpha), y: best.val_rmse }]} fill="var(--surface)" stroke={C.text} strokeWidth={2.5} shape="circle" isAnimationActive={false} />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
    <Legend items={[{ label: "Log-volatility target", color: C.model }, { label: "Raw volatility target", color: C.base }, { label: "Selected", color: C.text }]} />
    </>
  );
}
