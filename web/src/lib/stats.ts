type N = number | null | undefined;

const pairs = (y: N[], p: N[]) => {
  const out: [number, number][] = [];
  for (let i = 0; i < y.length; i++) {
    const a = y[i];
    const b = p[i];
    if (a != null && b != null) out.push([a, b]);
  }
  return out;
};
export const rmse = (y: N[], p: N[]) => {
  const q = pairs(y, p);
  return q.length ? Math.sqrt(q.reduce((s, [a, b]) => s + (b - a) ** 2, 0) / q.length) : NaN;
};
export const mae = (y: N[], p: N[]) => {
  const q = pairs(y, p);
  return q.length ? q.reduce((s, [a, b]) => s + Math.abs(b - a), 0) / q.length : NaN;
};
export const bias = (y: N[], p: N[]) => {
  const q = pairs(y, p);
  return q.length ? q.reduce((s, [a, b]) => s + (b - a), 0) / q.length : NaN;
};
export const mean = (a: number[]) => a.reduce((s, x) => s + x, 0) / a.length;
export const std = (a: number[]) => {
  const m = mean(a);
  return Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / (a.length - 1));
};

export function histogram(values: number[], lo: number, hi: number, width: number) {
  const nb = Math.round((hi - lo) / width);
  const counts = new Array<number>(nb).fill(0);
  values.forEach((x) => {
    counts[Math.min(nb - 1, Math.max(0, Math.floor((x - lo) / width)))]++;
  });
  return counts.map((c, i) => ({ x: +(lo + (i + 0.5) * width).toFixed(3), count: c }));
}

export const f1 = (x: N) => (x == null || Number.isNaN(x) ? "–" : x.toFixed(1));
export const f2 = (x: N) => (x == null || Number.isNaN(x) ? "–" : x.toFixed(2));
export const f3 = (x: N) => (x == null || Number.isNaN(x) ? "–" : x.toFixed(3));
export const pct0 = (x: number) => `${Math.round(x * 100)}%`;
export const fmtInt = (x: number) => x.toLocaleString("en-US");

/** Index of the first date >= d (dates sorted ascending, ISO strings). */
export function idxOfDate(dates: string[], d: string) {
  let lo = 0;
  let hi = dates.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (dates[mid] < d) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}
