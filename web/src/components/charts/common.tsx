"use client";

import { useMemo } from "react";
import {
  Bar, BarChart, Brush, CartesianGrid, Cell, ErrorBar, Line, LineChart, ReferenceArea, ReferenceLine,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { C } from "@/lib/colors";
import type { Num } from "@/lib/types";
import { Legend } from "../ui";

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
export const fmtMonth = (d: string) => `${MONTHS[+d.slice(5, 7) - 1]} ${d.slice(2, 4)}`;
export const fmtDate = (d: string) => `${+d.slice(8, 10)} ${MONTHS[+d.slice(5, 7) - 1]} ${d.slice(0, 4)}`;

type TipItem = { name?: string | number; value?: unknown; color?: string; stroke?: string; fill?: string; payload?: Record<string, unknown> };
export function Tip({ active, payload, label, fmt = (v: number) => v.toFixed(2), title }: {
  active?: boolean; payload?: TipItem[]; label?: string | number; fmt?: (v: number, name: string) => string; title?: (label: string | number | undefined, p?: TipItem[]) => string;
}) {
  if (!active || !payload || !payload.length) return null;
  return (
    <div className="tip">
      <div className="tip-title">{title ? title(label, payload) : typeof label === "string" && /^\d{4}-\d\d-\d\d$/.test(label) ? fmtDate(label) : label}</div>
      {payload.filter((p) => p.value != null && !Array.isArray(p.value)).map((p) => (
        <div className="tip-row" key={String(p.name)}>
          <i className="dot" style={{ background: p.color || p.stroke || p.fill }} />
          {p.name}
          <span>{fmt(Number(p.value), String(p.name))}</span>
        </div>
      ))}
    </div>
  );
}

export const axisProps = { tickLine: false, axisLine: false, tick: { fontSize: 11.5 } } as const;

export type LineSpec = { key: string; label: string; color: string; data: Num[]; width?: number; dashed?: boolean };
export type Band = { from: string; to: string; color: string; label: string };

/** Time series on a date category axis, with optional shaded bands and a brush to zoom. */
export function TimeSeries({ dates, lines, yLabel, bands, brush = false, height = "", unit = "%", log = false, legend = true, zero = true }: {
  dates: string[]; lines: LineSpec[]; yLabel: string; bands?: Band[]; brush?: boolean; height?: string; unit?: string; log?: boolean; legend?: boolean; zero?: boolean;
}) {
  const rows = useMemo(
    () => dates.map((d, i) => { const r: Record<string, string | number | null> = { d }; lines.forEach((l) => (r[l.key] = l.data[i] ?? null)); return r; }),
    [dates, lines],
  );
  return (
    <>
    <div className={`chart ${height}`}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={rows} margin={{ top: 8, right: 12, bottom: 4, left: 4 }}>
          <CartesianGrid stroke={C.grid} vertical={false} />
          {bands?.map((b) => (
            <ReferenceArea key={b.label} x1={b.from} x2={b.to} fill={b.color} fillOpacity={1} ifOverflow="hidden"
              label={{ value: b.label, position: "insideTopLeft", fill: C.text3, fontSize: 11, fontWeight: 600 }} />
          ))}
          <XAxis dataKey="d" {...axisProps} tickFormatter={fmtMonth} minTickGap={48} />
          <YAxis {...axisProps} width={44} scale={log ? "log" : "auto"} domain={log || !zero ? ["auto", "auto"] : [0, "auto"]} allowDataOverflow={log}
            label={{ value: yLabel, angle: -90, position: "insideLeft", offset: 10, style: { textAnchor: "middle" } }} />
          <Tooltip content={<Tip fmt={(v) => `${v.toFixed(1)}${unit}`} />} cursor={{ stroke: C.text3, strokeDasharray: "3 3" }} />
          {lines.map((l) => (
            <Line key={l.key} dataKey={l.key} name={l.label} stroke={l.color} strokeWidth={l.width ?? 1.8} dot={false}
              strokeDasharray={l.dashed ? "5 4" : undefined} isAnimationActive={false} connectNulls={false} activeDot={{ r: 3.5, strokeWidth: 0 }} />
          ))}
          {brush && <Brush dataKey="d" height={26} stroke={C.text3} fill="transparent" travellerWidth={8} tickFormatter={fmtMonth} />}
        </LineChart>
      </ResponsiveContainer>
    </div>
    {legend && lines.length > 1 && <Legend items={lines.map((l) => ({ label: l.label, color: l.color, dashed: l.dashed }))} />}
    </>
  );
}

export type BarRow = { label: string; value: number; color: string; lo?: number; hi?: number; note?: string };

/** Horizontal bars, optional asymmetric error whiskers. */
export function HBars({ rows, xLabel, height = "md", digits = 2, zeroLine = false, domain }: {
  rows: BarRow[]; xLabel: string; height?: string; digits?: number; zeroLine?: boolean; domain?: [number | string, number | string];
}) {
  const data = rows.map((r) => ({ ...r, err: r.lo != null && r.hi != null ? [r.value - r.lo, r.hi - r.value] : undefined }));
  const hasErr = data.some((d) => d.err);
  const longest = Math.max(...rows.map((r) => r.label.length));
  return (
    <div className={`chart ${height}`}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ top: 4, right: 20, bottom: 18, left: 4 }} barCategoryGap="28%">
          <CartesianGrid stroke={C.grid} horizontal={false} />
          <XAxis type="number" {...axisProps} domain={domain ?? [0, "auto"]} label={{ value: xLabel, position: "insideBottom", offset: -12 }} />
          <YAxis type="category" dataKey="label" {...axisProps} width={Math.min(190, 12 + longest * 6.6)} tick={{ fontSize: 12, fill: "var(--text-2)" }} />
          {zeroLine && <ReferenceLine x={0} stroke={C.text3} />}
          <Tooltip cursor={{ fill: "var(--surface-2)" }} content={<Tip fmt={(v) => v.toFixed(digits)} title={(l, p) => String(l) + (p?.[0]?.payload?.note ? ` · ${p[0].payload.note}` : "")} />} />
          <Bar dataKey="value" name={xLabel} radius={[0, 5, 5, 0]} isAnimationActive={false}>
            {data.map((d) => <Cell key={d.label} fill={d.color} />)}
            {hasErr && <ErrorBar dataKey="err" direction="x" width={6} stroke="var(--text-2)" strokeWidth={1.2} />}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function Seg<T extends string>({ items, value, onChange, label }: { items: { v: T; t: string }[]; value: T; onChange: (v: T) => void; label?: string }) {
  return (
    <div className="seg" role="group" aria-label={label}>
      {items.map((i) => (
        <button key={i.v} type="button" className={i.v === value ? "on" : ""} aria-pressed={i.v === value} onClick={() => onChange(i.v)}>
          {i.t}
        </button>
      ))}
    </div>
  );
}
