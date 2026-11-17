"use client";

import { useState } from "react";
import { C } from "@/lib/colors";
import type { Series } from "@/lib/types";
import { Seg, TimeSeries } from "./common";

export function ForecastByAsset({ timeline, fam }: { timeline: Record<string, Series>; fam: string }) {
  const assets = Object.keys(timeline);
  const [a, setA] = useState(assets[0]);
  const T = timeline[a];
  return (
    <>
      <div className="controls" style={{ marginBottom: 10 }}>
        <Seg items={assets.map((x) => ({ v: x, t: x }))} value={a} onChange={setA} label="Asset" />
      </div>
      <TimeSeries
        dates={T.dates}
        yLabel="Volatility (%)"
        lines={[
          { key: "actual", label: "Realised (next 20 days)", color: C.actual, data: T.actual, width: 2.2 },
          { key: "model", label: `${fam} forecast`, color: C.model, data: T.model },
          { key: "recent", label: "Recent 20-day vol", color: C.base, data: T.recent, width: 1.4 },
        ]}
      />
    </>
  );
}
