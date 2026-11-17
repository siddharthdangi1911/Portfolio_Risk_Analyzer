"use client";

import { useMemo, useState } from "react";
import { Bar, CartesianGrid, ComposedChart, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { CAT, C } from "@/lib/colors";
import { f1, f2, histogram, mean, std } from "@/lib/stats";
import type { Num } from "@/lib/types";
import { axisProps, type Band, Seg, TimeSeries, Tip } from "./common";

type History = { dates: string[]; close: Record<string, Num[]> };

const logReturns = (p: Num[]) => p.map((v, i) => (i === 0 || v == null || p[i - 1] == null ? null : Math.log(v / (p[i - 1] as number)) * 100));

function AssetChips({ assets, on, toggle }: { assets: string[]; on: Set<string>; toggle: (a: string) => void }) {
  return (
    <div className="controls">
      {assets.map((a, i) => (
        <button key={a} type="button" className={`chip ${on.has(a) ? "on" : "off"}`} aria-pressed={on.has(a)} onClick={() => toggle(a)}>
          <i className="dot" style={{ background: CAT[i] }} /> {a}
        </button>
      ))}
    </div>
  );
}

function useToggleSet(all: string[]) {
  const [on, setOn] = useState(() => new Set(all));
  const toggle = (a: string) =>
    setOn((s) => {
      const n = new Set(s);
      if (n.has(a) && n.size > 1) n.delete(a);
      else n.add(a);
      return n;
    });
  return [on, toggle] as const;
}

export function PriceHistory({ history, assets }: { history: History; assets: string[] }) {
  const [on, toggle] = useToggleSet(assets);
  const [log, setLog] = useState(false);
  const lines = useMemo(
    () =>
      assets.filter((a) => on.has(a)).map((a) => {
        const p = history.close[a];
        const first = p.find((v) => v != null) as number;
        return { key: a, label: a, color: CAT[assets.indexOf(a)], data: p.map((v) => (v == null ? null : (v / first) * 100)), width: 1.6 };
      }),
    [assets, history, on],
  );
  return (
    <>
      <div className="controls" style={{ marginBottom: 10, justifyContent: "space-between" }}>
        <AssetChips assets={assets} on={on} toggle={toggle} />
        <label className="toggle"><input type="checkbox" checked={log} onChange={(e) => setLog(e.target.checked)} /> Log scale</label>
      </div>
      <TimeSeries dates={history.dates} lines={lines} yLabel="Value of 100 invested" brush height="lg" unit="" log={log} legend={false} zero={false} />
    </>
  );
}

export function VolHistory({ history, assets, bands }: { history: History; assets: string[]; bands: Band[] }) {
  const [on, toggle] = useToggleSet(assets);
  const vols = useMemo(() => {
    const out: Record<string, Num[]> = {};
    assets.forEach((a) => {
      const r = logReturns(history.close[a]);
      out[a] = r.map((_, i) => {
        if (i < 20) return null;
        const w = r.slice(i - 19, i + 1);
        return w.some((x) => x == null) ? null : std(w as number[]) * Math.sqrt(252);
      });
    });
    return out;
  }, [assets, history]);
  const lines = assets.filter((a) => on.has(a)).map((a) => ({ key: a, label: a, color: CAT[assets.indexOf(a)], data: vols[a], width: 1.3 }));
  return (
    <>
      <div style={{ marginBottom: 10 }}><AssetChips assets={assets} on={on} toggle={toggle} /></div>
      <TimeSeries dates={history.dates} lines={lines} yLabel="20-day volatility (%)" bands={bands} brush height="lg" legend={false} />
    </>
  );
}

export function ReturnDist({ history, assets }: { history: History; assets: string[] }) {
  const [a, setA] = useState(assets[0]);
  const { rows, stats } = useMemo(() => {
    const r = logReturns(history.close[a]).filter((x): x is number => x != null);
    const mu = mean(r), sd = std(r);
    const w = 0.25;
    const rows = histogram(r, -6, 6, w).map((b) => ({ ...b, normal: (r.length * w * Math.exp(-0.5 * ((b.x - mu) / sd) ** 2)) / (sd * Math.sqrt(2 * Math.PI)) }));
    const kurt = r.reduce((s, x) => s + ((x - mu) / sd) ** 4, 0) / r.length - 3;
    const tails = r.filter((x) => Math.abs(x - mu) > 3 * sd).length;
    return { rows, stats: { sd, ann: sd * Math.sqrt(252), kurt, tails, expected: Math.round(r.length * 0.0027) } };
  }, [a, history]);
  const color = CAT[assets.indexOf(a)];
  return (
    <>
      <div className="controls" style={{ marginBottom: 10 }}>
        <Seg items={assets.map((x) => ({ v: x, t: x }))} value={a} onChange={setA} label="Asset" />
      </div>
      <div className="chart md">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={rows} margin={{ top: 6, right: 8, bottom: 16, left: 0 }} barCategoryGap={1}>
            <CartesianGrid stroke={C.grid} vertical={false} />
            <XAxis dataKey="x" {...axisProps} type="category" interval={7} tickFormatter={(v) => `${Math.round(+v)}%`} label={{ value: "Daily return (tails clipped at ±6%)", position: "insideBottom", offset: -10 }} />
            <YAxis {...axisProps} width={40} />
            <Tooltip cursor={{ fill: "var(--surface-2)" }} content={<Tip fmt={(v) => v.toFixed(0)} title={(l) => `Return ≈ ${(+(l ?? 0)).toFixed(2)}%`} />} />
            <Bar dataKey="count" name="Days" fill={color} radius={[2, 2, 0, 0]} isAnimationActive={false} />
            <Line dataKey="normal" name="Normal curve" stroke={C.text2} strokeDasharray="5 4" dot={false} strokeWidth={1.5} isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <div className="kpis" style={{ gridTemplateColumns: "repeat(3, 1fr)", margin: "12px 0 0" }}>
        <div><div className="kpi-label">Annual volatility</div><div style={{ fontWeight: 650, fontSize: 18 }}>{f1(stats.ann)}%</div></div>
        <div><div className="kpi-label">Excess kurtosis</div><div style={{ fontWeight: 650, fontSize: 18 }}>{f1(stats.kurt)}</div></div>
        <div><div className="kpi-label">Days beyond 3σ</div><div style={{ fontWeight: 650, fontSize: 18 }}>{stats.tails} <span className="small muted">vs {stats.expected} if normal</span></div></div>
      </div>
    </>
  );
}

export function Heatmap({ assets, train, test }: { assets: string[]; train: number[][]; test: number[][] }) {
  const [k, setK] = useState<"train" | "test">("train");
  const M = k === "train" ? train : test;
  const mix = (v: number) => `color-mix(in srgb, ${v >= 0 ? "var(--div-pos)" : "var(--div-neg)"} ${Math.round(Math.abs(v) * 85)}%, var(--div-mid))`;
  return (
    <>
      <div className="controls" style={{ marginBottom: 12 }}>
        <Seg items={[{ v: "train", t: "Train years" }, { v: "test", t: "Test years" }]} value={k} onChange={setK} label="Period" />
      </div>
      <div className="heat" style={{ gridTemplateColumns: `44px repeat(${assets.length}, 1fr)` }} role="table" aria-label="Correlation matrix">
        <div />
        {assets.map((a) => <div key={a} className="hdr">{a}</div>)}
        {assets.map((a, i) => (
          <div key={a} style={{ display: "contents" }}>
            <div className="hdr">{a}</div>
            {assets.map((b, j) => (
              <div key={b} className="cell" title={`${a} vs ${b}: ${f2(M[i][j])}`} style={{ background: mix(M[i][j]), color: Math.abs(M[i][j]) > 0.6 ? "#fff" : "var(--text)" }}>
                {M[i][j].toFixed(2)}
              </div>
            ))}
          </div>
        ))}
      </div>
    </>
  );
}

export function FeatureTable({ rows }: { rows: { code: string; name: string; group: string; meaning: string; log: boolean }[] }) {
  const [q, setQ] = useState("");
  const shown = rows.filter((r) => `${r.code} ${r.name} ${r.group} ${r.meaning}`.toLowerCase().includes(q.toLowerCase()));
  return (
    <>
      <input type="search" placeholder="Filter features…" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Filter features" style={{ width: "100%", marginBottom: 10 }} />
      <div className="table-wrap" style={{ maxHeight: 360, overflowY: "auto" }}>
        <table>
          <thead><tr><th>Feature</th><th>Group</th><th>Meaning</th></tr></thead>
          <tbody>
            {shown.map((r) => (
              <tr key={r.code}>
                <td><b style={{ fontWeight: 600 }}>{r.name}</b>{r.log && <span className="tag">log</span>}<div className="small muted"><code>{r.code}</code></div></td>
                <td>{r.group}</td>
                <td className="small" style={{ color: "var(--text-2)" }}>{r.meaning}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
