// Colour roles. Values live in globals.css as CSS variables (separate light and
// dark sets), so charts follow the theme without re-rendering.
export const C = {
  actual: "var(--s-actual)",
  model: "var(--s-model)",
  base: "var(--s-base)",
  hist: "var(--s-hist)",
  ewma: "var(--s-ewma)",
  text: "var(--text)",
  text2: "var(--text-2)",
  text3: "var(--text-3)",
  grid: "var(--grid)",
  surface: "var(--surface)",
  pos: "var(--div-pos)",
  neg: "var(--div-neg)",
  good: "var(--good)",
  muted: "var(--muted-fill)",
};
export const CAT = ["var(--c1)", "var(--c2)", "var(--c3)", "var(--c4)", "var(--c5)", "var(--c6)"];

export function forecastColor(key: string, fam: string): string {
  if (key.startsWith(fam)) return C.model;
  if (key.startsWith("Recent")) return C.base;
  if (key.startsWith("Historical")) return C.hist;
  if (key.startsWith("EWMA")) return C.ewma;
  return C.text3;
}
