"use client";

import { useEffect, useMemo, useState } from "react";
import { C } from "@/lib/colors";
import { bias, f2, idxOfDate, mae, rmse } from "@/lib/stats";
import type { Series } from "@/lib/types";
import { fmtDate, Seg, TimeSeries } from "./charts/common";

type Period = "test" | "val";
type Key = "model" | "recent" | "hist" | "ewma";

const PRESETS: Record<Period, [string, string | null, string | null][]> = {
  test: [["Whole period", null, null], ["2023", "2023-01-01", "2023-12-31"], ["2024", "2024-01-01", "2024-12-31"], ["SVB stress, Mar 2023", "2023-02-15", "2023-05-15"], ["Aug 2024 VIX spike", "2024-06-15", "2024-09-30"]],
  val: [["Whole period", null, null], ["2021 (calm)", "2021-01-01", "2021-12-31"], ["2022 (bear market)", "2022-01-01", "2022-12-31"], ["Ukraine invasion", "2022-02-01", "2022-04-30"]],
};

export function Explorer({ test, val, fam }: { test: Record<string, Series>; val: Record<string, Series>; fam: string }) {
  const assets = Object.keys(test);
  const [period, setPeriod] = useState<Period>("test");
  const [asset, setAsset] = useState(assets[0]);
  const [on, setOn] = useState<Record<Key, boolean>>({ model: true, recent: true, hist: false, ewma: false });
  const T = (period === "test" ? test : val)[asset];
  const n = T.dates.length;
  const [win, setWin] = useState<[number, number]>([0, n - 1]);

  // Deep link from the Errors page: ?asset=QQQ&date=2024-08-05
  useEffect(() => {
    const q = new URLSearchParams(window.location.search);
    const a = q.get("asset"), d = q.get("date");
    if (a && test[a]) {
      setAsset(a);
      if (d) {
        const dates = test[a].dates, i = idxOfDate(dates, d);
        setWin([Math.max(0, i - 40), Math.min(dates.length - 1, i + 40)]);
      }
    }
  }, [test]);

  const setPreset = (from: string | null, to: string | null) =>
    setWin([from ? idxOfDate(T.dates, from) : 0, to ? Math.min(n - 1, idxOfDate(T.dates, to)) : n - 1]);
  const changePeriod = (p: Period) => { setPeriod(p); const m = (p === "test" ? test : val)[asset].dates.length; setWin([0, m - 1]); };

  const [a0, a1] = [Math.min(win[0], n - 2), Math.min(win[1], n - 1)];
  const sl = <X,>(arr: X[]) => arr.slice(a0, a1 + 1);
  const modelLabel = period === "val" ? `${fam} (fit on train only)` : `${fam} forecast`;
  const SPEC: { k: Key; label: string; color: string; dashed?: boolean }[] = [
    { k: "model", label: modelLabel, color: C.model },
    { k: "recent", label: "Recent 20-day vol", color: C.base },
    { k: "hist", label: "Historical average", color: C.hist, dashed: true },
    { k: "ewma", label: "EWMA", color: C.ewma },
  ];
  const lines = useMemo(
    () => [
      { key: "actual", label: "Realised (next 20 days)", color: C.actual, data: sl(T.actual), width: 2.4 },
      ...SPEC.filter((s) => on[s.k]).map((s) => ({ key: s.k, label: s.label, color: s.color, data: sl(T[s.k]), dashed: s.dashed, width: s.k === "model" ? 2 : 1.5 })),
    ],
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [T, a0, a1, on, period],
  );
  const y = sl(T.actual);
  const stats = SPEC.map((s) => ({ ...s, r: rmse(y, sl(T[s.k])), a: mae(y, sl(T[s.k])), b: bias(y, sl(T[s.k])) }));
  const best = Math.min(...stats.map((s) => s.r));
  const pct = (i: number) => (i / (n - 1)) * 100;

  return (
    <>
      <div className="card mb">
        <div className="controls" style={{ marginBottom: 12, justifyContent: "space-between" }}>
          <div className="controls">
            <Seg items={[{ v: "test", t: "Test 2023–24" }, { v: "val", t: "Validation 2021–22" }]} value={period} onChange={changePeriod} label="Period" />
            <Seg items={assets.map((x) => ({ v: x, t: x }))} value={asset} onChange={setAsset} label="Asset" />
          </div>
          <div className="controls">
            {SPEC.map((s) => (
              <button key={s.k} type="button" className={`chip ${on[s.k] ? "on" : "off"}`} aria-pressed={on[s.k]} onClick={() => setOn({ ...on, [s.k]: !on[s.k] })}>
                <i className="swatch" style={{ background: s.dashed ? `repeating-linear-gradient(90deg, ${s.color} 0 4px, transparent 4px 7px)` : s.color }} />
                {s.k === "model" ? fam : s.label}
              </button>
            ))}
          </div>
        </div>
        <div className="controls" style={{ marginBottom: 14 }}>
          <span className="small muted">Jump to</span>
          {PRESETS[period].map(([label, from, to]) => (
            <button key={label} type="button" className="chip" onClick={() => setPreset(from, to)}>{label}</button>
          ))}
        </div>
        <TimeSeries dates={sl(T.dates)} lines={lines} yLabel="Volatility (%)" height="lg" />
        <div style={{ padding: "8px 4px 0 48px" }}>
          <div className="dual-range">
            <div className="track" />
            <div className="fill" style={{ left: `${pct(a0)}%`, width: `${pct(a1) - pct(a0)}%` }} />
            <input type="range" min={0} max={n - 1} value={a0} aria-label="Window start" onChange={(e) => setWin([Math.min(+e.target.value, a1 - 5), a1])} />
            <input type="range" min={0} max={n - 1} value={a1} aria-label="Window end" onChange={(e) => setWin([a0, Math.max(+e.target.value, a0 + 5)])} />
          </div>
          <div className="range-labels"><span>{fmtDate(T.dates[a0])}</span><span>{a1 - a0 + 1} trading days</span><span>{fmtDate(T.dates[a1])}</span></div>
        </div>
      </div>
      <div className="card">
        <div className="card-head">
          <div>
            <h2 className="card-title">Accuracy in this window</h2>
            <div className="card-sub">{asset} · {fmtDate(T.dates[a0])} → {fmtDate(T.dates[a1])}</div>
          </div>
        </div>
        <div className="table-wrap">
          <table>
            <thead><tr><th>Forecast</th><th className="num">RMSE</th><th className="num">MAE</th><th className="num">Bias</th><th className="num">{fam} vs this</th></tr></thead>
            <tbody>
              {stats.map((s, i) => (
                <tr key={s.k} className={i === 0 ? "hl" : ""}>
                  <td><i className="swatch" style={{ background: s.color, marginRight: 8 }} />{s.label}{s.r === best && <span className="tag accent">best here</span>}</td>
                  <td className="num">{f2(s.r)}</td>
                  <td className="num">{f2(s.a)}</td>
                  <td className="num">{f2(s.b)}</td>
                  <td className="num">{i === 0 ? "–" : <span className={stats[0].r < s.r ? "win" : "lose"}>{stats[0].r < s.r ? "better" : "worse"} by {f2(Math.abs(stats[0].r - s.r))}</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </>
  );
}
