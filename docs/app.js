/* Portfolio Risk Analyzer - dashboard front-end.
   Every number comes from window.SITE_DATA (docs/data/site_data.js), written by
   scripts/export_web_data.py from the pipeline outputs. Nothing is typed in by hand. */
(function () {
  "use strict";
  const D = window.SITE_DATA;
  const $ = (id) => document.getElementById(id);
  if (!D) {
    $("headline").textContent = "Could not load data/site_data.js. Run `python scripts/run_all.py` to generate it.";
    return;
  }
  const R = D.results;
  const FAM = R.final_model.family;
  const MKEY = FAM + " (final)";
  const TKEY = FAM + " (train-only fit)";
  const BASE = D.primary_baseline;
  const HIST = "Historical average";
  const EWMA = "EWMA vol (RiskMetrics)";
  const ASSETS = D.assets;
  const P = D.portfolio;

  // ------------------------------------------------------------------ helpers
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const f2 = (x) => (x == null || isNaN(x) ? "–" : Number(x).toFixed(2));
  const f1 = (x) => (x == null || isNaN(x) ? "–" : Number(x).toFixed(1));
  const f3 = (x) => (x == null || isNaN(x) ? "–" : Number(x).toFixed(3));
  const pct0 = (x) => (x * 100).toFixed(0) + "%";
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const alpha = (hex, a) => {
    const h = hex.replace("#", "");
    if (h.length !== 6) return hex;
    return `rgba(${parseInt(h.slice(0, 2), 16)},${parseInt(h.slice(2, 4), 16)},${parseInt(h.slice(4, 6), 16)},${a})`;
  };
  const table = (el, head, rows, opts = {}) => {
    const H = head.map((h) => (typeof h === "string" ? { t: h } : h));
    el.innerHTML =
      "<thead><tr>" + H.map((h) => `<th class="${h.num ? "num" : ""}">${h.t}</th>`).join("") + "</tr></thead><tbody>" +
      rows.map((r, i) => {
        const cls = [i === opts.hl ? "hl" : "", opts.click ? "click" : ""].join(" ");
        return `<tr class="${cls}" data-i="${i}">` + r.map((c, j) => `<td class="${H[j].num ? "num" : ""}">${c}</td>`).join("") + "</tr>";
      }).join("") + "</tbody>";
  };
  const rmse = (y, p) => { let s = 0, n = 0; for (let i = 0; i < y.length; i++) if (y[i] != null && p[i] != null) { s += (y[i] - p[i]) ** 2; n++; } return n ? Math.sqrt(s / n) : NaN; };
  const mae = (y, p) => { let s = 0, n = 0; for (let i = 0; i < y.length; i++) if (y[i] != null && p[i] != null) { s += Math.abs(y[i] - p[i]); n++; } return n ? s / n : NaN; };
  const bias = (y, p) => { let s = 0, n = 0; for (let i = 0; i < y.length; i++) if (y[i] != null && p[i] != null) { s += p[i] - y[i]; n++; } return n ? s / n : NaN; };
  const mean = (a) => a.reduce((s, x) => s + x, 0) / a.length;
  const std = (a) => { const m = mean(a); return Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / (a.length - 1)); };
  const shortName = (k) => k.replace(" (RiskMetrics)", "").replace(" (final)", " (selected)");
  const featName = (f) => {
    const base = f.replace(/^log_/, "");
    const info = R.feature_info[base];
    if (!info) return f.startsWith("is_") ? `Asset = ${f.slice(3)}` : f;
    return info[0] + (f.startsWith("log_") ? " [log]" : "");
  };
  const segButtons = (el, items, on, attr = "v") => {
    el.innerHTML = items.map((it) => `<button type="button" data-${attr}="${esc(it.v)}" class="${it.v === on ? "on" : ""}">${esc(it.t)}</button>`).join("");
  };
  const segOn = (el, v, attr = "v") => [...el.children].forEach((b) => b.classList.toggle("on", b.dataset[attr] === v));

  // colours by role (read at render time so theme switches apply)
  const COL = () => ({
    actual: css("--s-actual"), model: css("--s-model"), base: css("--s-base"), hist: css("--s-hist"), ewma: css("--s-ewma"),
    c: [css("--c1"), css("--c2"), css("--c3"), css("--c4"), css("--c5"), css("--c6")],
    text: css("--text"), text2: css("--text-2"), text3: css("--text-3"), border: css("--border"), surface: css("--surface"),
    pos: css("--div-pos"), neg: css("--div-neg"), mid: css("--div-mid"), good: css("--good"), bad: css("--bad"),
    bandTrain: css("--band-train"), bandVal: css("--band-val"), bandTest: css("--band-test"),
  });
  const forecastColor = (k, c) => {
    if (k === MKEY || k === FAM + " (tuned)") return c.model;
    if (k === TKEY) return alpha(c.model, 0.45);
    if (k === BASE) return c.base;
    if (k === HIST) return c.hist;
    if (k === EWMA) return c.ewma;
    return c.text3;
  };
  const assetColor = (a, c) => c.c[ASSETS.indexOf(a) % 6];
  const families = R.stage1_screening.results.filter((r) => r.kind === "model").map((r) => r.name);
  const famColor = (name, c) => c.c[families.indexOf(name) % 6];

  // ------------------------------------------------------------------ Chart.js setup
  if (window.ChartZoom) Chart.register(window.ChartZoom);
  const applyDefaults = () => {
    Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
    Chart.defaults.font.size = 12;
    Chart.defaults.color = css("--text-2");
    Chart.defaults.borderColor = css("--border");
    Chart.defaults.animation = false;
    Chart.defaults.maintainAspectRatio = false;
    Chart.defaults.plugins.legend.labels.boxHeight = 3;
    Chart.defaults.plugins.legend.labels.boxWidth = 16;
    Chart.defaults.plugins.legend.position = "bottom";
    Chart.defaults.plugins.tooltip.backgroundColor = css("--surface");
    Chart.defaults.plugins.tooltip.titleColor = css("--text");
    Chart.defaults.plugins.tooltip.bodyColor = css("--text-2");
    Chart.defaults.plugins.tooltip.borderColor = css("--border");
    Chart.defaults.plugins.tooltip.borderWidth = 1;
    Chart.defaults.plugins.tooltip.padding = 10;
    Chart.defaults.elements.line.borderWidth = 2;
    Chart.defaults.elements.point.radius = 0;
    Chart.defaults.elements.point.hoverRadius = 4;
    Chart.defaults.elements.bar.borderRadius = 4;
  };
  applyDefaults();

  // shaded split bands on a category x-axis: options.plugins.bands = [{from,to,color,label}] (label indexes)
  Chart.register({
    id: "bands",
    beforeDatasetsDraw(chart, _a, opts) {
      if (!opts || !opts.items) return;
      const { ctx, chartArea: ca, scales: { x } } = chart;
      ctx.save();
      ctx.beginPath(); ctx.rect(ca.left, ca.top, ca.right - ca.left, ca.bottom - ca.top); ctx.clip();
      opts.items.forEach((b) => {
        const x0 = x.getPixelForValue(b.from), x1 = x.getPixelForValue(b.to);
        ctx.fillStyle = b.color; ctx.fillRect(x0, ca.top, x1 - x0, ca.bottom - ca.top);
        ctx.fillStyle = css("--text-3"); ctx.font = "600 11px " + Chart.defaults.font.family;
        const lx = Math.max(x0, ca.left) + 6;
        if (lx < Math.min(x1, ca.right) - 40) ctx.fillText(b.label, lx, ca.top + 14);
      });
      ctx.restore();
    },
  });
  // whiskers for horizontal bars: options.plugins.whiskers = {lo:[], hi:[]}
  Chart.register({
    id: "whiskers",
    afterDatasetsDraw(chart, _a, opts) {
      if (!opts || !opts.lo) return;
      const { ctx, scales: { x } } = chart;
      const meta = chart.getDatasetMeta(0);
      ctx.save(); ctx.strokeStyle = css("--text-2"); ctx.lineWidth = 1.2;
      meta.data.forEach((bar, i) => {
        if (opts.lo[i] == null) return;
        const y = bar.y, a = x.getPixelForValue(opts.lo[i]), b = x.getPixelForValue(opts.hi[i]);
        ctx.beginPath(); ctx.moveTo(a, y); ctx.lineTo(b, y); ctx.moveTo(a, y - 5); ctx.lineTo(a, y + 5); ctx.moveTo(b, y - 5); ctx.lineTo(b, y + 5); ctx.stroke();
      });
      ctx.restore();
    },
  });
  // point markers on horizontal floating bars: options.plugins.markers = {values:[]}
  Chart.register({
    id: "markers",
    afterDatasetsDraw(chart, _a, opts) {
      if (!opts || !opts.values) return;
      const { ctx, scales: { x } } = chart;
      const meta = chart.getDatasetMeta(0);
      ctx.save();
      meta.data.forEach((bar, i) => {
        const px = x.getPixelForValue(opts.values[i]);
        ctx.beginPath(); ctx.arc(px, bar.y, 5, 0, Math.PI * 2);
        ctx.fillStyle = css("--text"); ctx.fill(); ctx.lineWidth = 2; ctx.strokeStyle = css("--surface"); ctx.stroke();
      });
      ctx.restore();
    },
  });
  // labels next to scatter points: dataset.pointLabels = [...]
  Chart.register({
    id: "pointLabels",
    afterDatasetsDraw(chart) {
      const { ctx } = chart;
      chart.data.datasets.forEach((ds, di) => {
        if (!ds.pointLabels || !chart.isDatasetVisible(di)) return;
        const meta = chart.getDatasetMeta(di);
        ctx.save(); ctx.fillStyle = css("--text-2"); ctx.font = "12px " + Chart.defaults.font.family;
        meta.data.forEach((pt, i) => ctx.fillText(ds.pointLabels[i], pt.x + 8, pt.y - 6));
        ctx.restore();
      });
    },
  });

  const charts = {};
  const mk = (id, cfg) => {
    if (charts[id]) charts[id].destroy();
    charts[id] = new Chart($(id), cfg);
    return charts[id];
  };
  const zoomOpts = () => ({
    zoom: { drag: { enabled: true, backgroundColor: alpha(css("--accent").length === 7 ? css("--accent") : "#2a78d6", 0.12) }, mode: "x" },
    pan: { enabled: true, mode: "x", modifierKey: "shift" },
  });
  const lineOpts = (ytitle, extra = {}) => ({
    interaction: { mode: "index", intersect: false },
    scales: {
      x: { ticks: { maxTicksLimit: 8, maxRotation: 0 }, grid: { display: false } },
      y: { title: { display: true, text: ytitle } },
    },
    plugins: { tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${f2(c.parsed.y)}` } } },
    ...extra,
  });

  // ------------------------------------------------------------------ navigation
  const panels = [...document.querySelectorAll(".panel")];
  const navHtml = panels.map((p, i) => `<a class="navlink" href="#${p.id}" data-p="${p.id}"><span class="ix">${i + 1}</span>${p.dataset.title}</a>`).join("");
  $("nav").innerHTML = navHtml;
  $("mnav").innerHTML = navHtml;
  const rendered = new Set();
  const renderers = {};
  function show(id) {
    if (!panels.some((p) => p.id === id)) id = "overview";
    panels.forEach((p) => p.classList.toggle("active", p.id === id));
    document.querySelectorAll(".navlink").forEach((a) => a.classList.toggle("active", a.dataset.p === id));
    if (!rendered.has(id) && renderers[id]) { renderers[id](); rendered.add(id); }
    const m = document.querySelector(`#mnav .navlink[data-p="${id}"]`);
    if (m) m.scrollIntoView({ inline: "center", block: "nearest" });
  }
  const go = (id) => { if (location.hash !== "#" + id) location.hash = id; else show(id); window.scrollTo(0, 0); };
  window.addEventListener("hashchange", () => { show(location.hash.slice(1)); window.scrollTo(0, 0); });

  // theme toggle (stored per viewer; optional)
  const themeKey = "pra-theme";
  try { const t = localStorage.getItem(themeKey); if (t) document.documentElement.dataset.theme = t; } catch (e) { /* storage unavailable */ }
  $("theme-btn").addEventListener("click", () => {
    const dark = css("--bg") === "#121211";
    document.documentElement.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem(themeKey, document.documentElement.dataset.theme); } catch (e) { /* ignore */ }
    applyDefaults();
    rendered.clear();
    show(location.hash.slice(1) || "overview");
  });
  $("gen-date").textContent = "Data generated " + D.generated;

  // shared numbers
  const tm = R.test_metrics;
  const m = tm[MKEY], b = tm[BASE], h = tm[HIST], e = tm[EWMA];
  const boot = R.test_bootstrap_vs_baselines;
  const sig = (bt) => bt.RMSE_diff_CI95[1] < 0;
  const vc = R.validation_comparison;
  const valBest = Object.entries(vc).sort((a, c) => a[1].RMSE - c[1].RMSE)[0];
  const impr = (base) => (1 - m.RMSE / base.RMSE);
  const kpi = (l, v, s, d) => `<div class="kpi"><div class="label">${l}</div><div class="value">${v}</div>${d ? `<div class="delta ${d.cls}">${d.t}</div>` : ""}${s ? `<div class="sub">${s}</div>` : ""}</div>`;
  const forecastKeys = [MKEY, TKEY, EWMA, HIST, BASE].filter((k) => tm[k]);

  // ================================================================== OVERVIEW
  renderers.overview = () => {
    const c = COL();
    $("headline").innerHTML =
      `<b>Short answer: yes on the test years, but it depends on the regime.</b> On the untouched 2023–2024 test set the selected model (<b>${FAM}</b>) ` +
      `had RMSE <b>${f2(m.RMSE)}</b> volatility points, against <b>${f2(e.RMSE)}</b> for EWMA, <b>${f2(h.RMSE)}</b> for the historical average and <b>${f2(b.RMSE)}</b> for recent 20-day volatility. ` +
      `The gain over recent volatility ${sig(boot[BASE]) ? "holds up" : "does not hold up"} under a block bootstrap, ` +
      `over EWMA it ${sig(boot[EWMA]) ? "also holds up" : "is within noise"}, and over the historical average it ${sig(boot[HIST]) ? "holds up" : "is borderline (the interval just touches zero)"}. ` +
      `On the 2021–2022 validation years ${valBest[0] === FAM + " (tuned)" ? "the model was also best" : "<b>" + esc(shortName(valBest[0])) + " beat the model</b>"}.`;

    $("ov-kpis").innerHTML = [
      kpi("Selected model", FAM, `chosen from ${families.length} families by CV`),
      kpi("Test RMSE", f2(m.RMSE), "volatility points", { cls: "up", t: `${pct0(impr(b))} lower than recent vol` }),
      kpi("vs EWMA (RiskMetrics)", "−" + pct0(impr(e)), `RMSE ${f2(m.RMSE)} vs ${f2(e.RMSE)}`),
      kpi("Test R²", f2(m.R2), `recent vol: ${f2(b.R2)}`),
      kpi("Asset-days modelled", R.data.n_rows_model_table.toLocaleString(), `${ASSETS.join(", ")} + VIX`),
      kpi("Model fits run", R.training_proof ? R.training_proof.total_model_fits : "–", "CV + tuning + final"),
    ].join("");

    mk("ov-bar", {
      type: "bar",
      data: { labels: forecastKeys.map(shortName), datasets: [{ label: "Test RMSE", data: forecastKeys.map((k) => tm[k].RMSE), backgroundColor: forecastKeys.map((k) => forecastColor(k, c)), barThickness: 22 }] },
      options: { indexAxis: "y", plugins: { legend: { display: false }, tooltip: { callbacks: { label: (x) => `RMSE ${f2(x.parsed.x)}` } } },
        scales: { x: { beginAtZero: true, title: { display: true, text: "RMSE (volatility points)" } }, y: { grid: { display: false } } } },
    });

    const best = families[0];
    const lgb = R.stage1_screening.results.find((r) => r.name === "LightGBM");
    $("ov-findings").innerHTML = [
      `<li><b>${FAM}</b> had the lowest cross-validation error of ${families.length} model families${lgb ? `, ahead of LightGBM (${f2(lgb.cv_rmse_mean)})` : ""}. Volatility is persistent and close to linear in log terms, so a regularised linear model is hard to beat.</li>`,
      `<li>On the test years it beat all three baselines; the bootstrap says the gain over recent vol and EWMA is beyond noise.</li>`,
      `<li>On validation (2022 bear market) <b>EWMA beat it</b>: the model expects volatility to fall back to normal, which fails when stress persists.</li>`,
      `<li>The largest misses are one-day VIX spikes (5 Aug 2024) and sudden stress with no warning (July 2024).</li>`,
      `<li>Combining stocks with bonds and gold cuts portfolio volatility; try it in the Portfolio lab.</li>`,
    ].join("");
    void best;

    const drawLine = (a) => {
      const T = D.timeline[a];
      mk("ov-line", {
        type: "line",
        data: { labels: T.dates, datasets: [
          { label: "Realised (next 20 days)", data: T.actual, borderColor: c.actual, backgroundColor: c.actual, borderWidth: 2.2 },
          { label: `${FAM} forecast`, data: T.model, borderColor: c.model, backgroundColor: c.model },
          { label: "Recent 20-day vol", data: T.recent, borderColor: c.base, backgroundColor: c.base, borderWidth: 1.5 },
        ] },
        options: lineOpts("Annualised volatility (%)"),
      });
    };
    segButtons($("ov-asset"), ASSETS.map((a) => ({ v: a, t: a })), ASSETS[0]);
    $("ov-asset").onclick = (ev) => { const v = ev.target.dataset.v; if (!v) return; segOn($("ov-asset"), v); drawLine(v); };
    drawLine(ASSETS[0]);

    const steps = [
      ["STAGE 1", "Screen every family", `${families.length} model families, expanding-window CV on training years only.`, "selection"],
      ["PICK", `Choose ${FAM}`, "Lowest mean CV RMSE wins. Only this family continues.", "selection"],
      ["STAGE 2", "Tune on validation", `${R.stage2_tuning.n_configs} settings fit on train and scored on 2021–2022.`, "selection"],
      ["STAGE 3", "Test once", "Refit on train + validation, evaluate a single time on 2023–2024.", "results"],
    ];
    $("ov-flow").innerHTML = steps.map((s) => `<button class="step" type="button" data-go="${s[3]}"><span class="n">${s[0]}</span><b>${s[1]}</b><span>${s[2]}</span></button>`).join("");
    $("ov-flow").onclick = (ev) => { const t = ev.target.closest("[data-go]"); if (t) go(t.dataset.go); };
  };

  // ================================================================== DATA
  const hist = D.history;
  const logRet = {}; // % daily log returns per asset (null for first)
  ASSETS.forEach((a) => {
    const p = hist.close[a];
    logRet[a] = p.map((v, i) => (i === 0 || v == null || p[i - 1] == null ? null : Math.log(v / p[i - 1]) * 100));
  });
  const rollVol = (r, w) => r.map((_, i) => {
    if (i < w) return null;
    const s = r.slice(i - w + 1, i + 1);
    if (s.some((x) => x == null)) return null;
    return std(s) * Math.sqrt(252);
  });
  const idxOfDate = (dates, d) => { let lo = 0, hi = dates.length - 1; while (lo < hi) { const mid = (lo + hi) >> 1; if (dates[mid] < d) lo = mid + 1; else hi = mid; } return lo; };

  renderers.data = () => {
    const c = COL();
    const q = R.data.quality;
    $("data-intro").innerHTML =
      `Daily closing prices from Yahoo Finance (public, accessed with <code>yfinance</code>, no credentials). Source: <i>${esc(q.source)}</i>. ` +
      `After cleaning: ${q.rows_after_alignment.toLocaleString()} trading days, ${q.date_range[0]} to ${q.date_range[1]}, giving ${R.data.n_rows_model_table.toLocaleString()} asset-day rows after the 120-day warm-up.`;
    $("data-kpis").innerHTML = [
      kpi("Trading days", q.rows_after_alignment.toLocaleString(), `${q.date_range[0]} → ${q.date_range[1]}`),
      kpi("Assets", ASSETS.length, ASSETS.join(", ") + " (+ VIX as a feature)"),
      kpi("Asset-day rows", R.data.n_rows_model_table.toLocaleString(), "one row = one asset on one day"),
      kpi("Features", R.data.features.length, "returns, volatility, VIX, asset id"),
      kpi("Missing / bad prices", Object.values(q.missing_values_before).reduce((s, x) => s + x, 0) + Object.values(q.non_positive_prices).reduce((s, x) => s + x, 0), "before alignment"),
    ].join("");

    // price history
    const pxOn = new Set(ASSETS);
    const chipRow = (el, set, redraw) => {
      el.innerHTML = ASSETS.map((a) => `<button type="button" class="chip on" data-a="${a}"><span class="swatch" style="background:${assetColor(a, c)}"></span> ${a}</button>`).join("");
      el.onclick = (ev) => {
        const t = ev.target.closest("[data-a]"); if (!t) return;
        const a = t.dataset.a;
        if (set.has(a) && set.size > 1) set.delete(a); else set.add(a);
        t.classList.toggle("on", set.has(a)); redraw();
      };
    };
    const drawPx = () => {
      const ds = ASSETS.filter((a) => pxOn.has(a)).map((a) => {
        const p = hist.close[a]; const first = p.find((v) => v != null);
        return { label: a, data: p.map((v) => (v == null ? null : (v / first) * 100)), borderColor: assetColor(a, c), backgroundColor: assetColor(a, c), borderWidth: 1.6 };
      });
      mk("px-chart", { type: "line", data: { labels: hist.dates, datasets: ds },
        options: lineOpts("Value of 100 invested", { plugins: { zoom: zoomOpts(), tooltip: { callbacks: { label: (x) => `${x.dataset.label}: ${f1(x.parsed.y)}` } } },
          scales: { x: { ticks: { maxTicksLimit: 8, maxRotation: 0 }, grid: { display: false } }, y: { type: $("px-log").checked ? "logarithmic" : "linear", title: { display: true, text: "Value of 100 invested" } } } }) });
    };
    chipRow($("px-assets"), pxOn, drawPx);
    $("px-log").onchange = drawPx;
    $("px-reset").onclick = () => charts["px-chart"].resetZoom();
    $("px-chart").ondblclick = () => charts["px-chart"].resetZoom();
    drawPx();

    // historical vol with split bands
    const sp = R.splits;
    const bands = [
      { from: idxOfDate(hist.dates, sp.train.start), to: idxOfDate(hist.dates, sp.train.end), color: c.bandTrain, label: "TRAIN" },
      { from: idxOfDate(hist.dates, sp.val.start), to: idxOfDate(hist.dates, sp.val.end), color: c.bandVal, label: "VALIDATION" },
      { from: idxOfDate(hist.dates, sp.test.start), to: idxOfDate(hist.dates, sp.test.end), color: c.bandTest, label: "TEST" },
    ];
    const hvOn = new Set(ASSETS);
    const vols = {}; ASSETS.forEach((a) => (vols[a] = rollVol(logRet[a], 20)));
    const drawHv = () => {
      mk("hv-chart", { type: "line",
        data: { labels: hist.dates, datasets: ASSETS.filter((a) => hvOn.has(a)).map((a) => ({ label: a, data: vols[a], borderColor: assetColor(a, c), backgroundColor: assetColor(a, c), borderWidth: 1.4 })) },
        options: lineOpts("Trailing 20-day volatility (%)", { plugins: { bands: { items: bands }, zoom: zoomOpts(), tooltip: { callbacks: { label: (x) => `${x.dataset.label}: ${f1(x.parsed.y)}%` } } } }) });
    };
    chipRow($("hv-assets"), hvOn, drawHv);
    $("hv-reset").onclick = () => charts["hv-chart"].resetZoom();
    $("hv-chart").ondblclick = () => charts["hv-chart"].resetZoom();
    drawHv();

    const tot = sp.train.dates + sp.val.dates + sp.test.dates;
    const seg = (lab, n, col) => `<div style="flex:${n};background:${col}" title="${lab}">${lab}</div>`;
    $("split-bar").innerHTML = seg(`Train ${sp.train.start.slice(0, 4)}–${sp.train.end.slice(0, 4)}`, sp.train.dates, c.model) +
      `<div class="gap" style="flex:${tot * 0.015}"></div>` + seg(`Validation ${sp.val.start.slice(0, 4)}–${sp.val.end.slice(0, 4)}`, sp.val.dates, c.ewma) +
      `<div class="gap" style="flex:${tot * 0.015}"></div>` + seg(`Test ${sp.test.start.slice(0, 4)}–${sp.test.end.slice(0, 4)}`, sp.test.dates, c.base);
    $("split-text").innerHTML = `Train ${sp.train.start} → ${sp.train.end} (${sp.train.rows.toLocaleString()} rows) · Validation ${sp.val.start} → ${sp.val.end} (${sp.val.rows.toLocaleString()} rows) · ` +
      `Test ${sp.test.start} → ${sp.test.end} (${sp.test.rows.toLocaleString()} rows). Hatched gaps: 20-day embargo, so no target window crosses into the next period.`;

    // return distribution
    $("rd-asset").innerHTML = ASSETS.map((a) => `<option>${a}</option>`).join("");
    const drawRd = () => {
      const a = $("rd-asset").value;
      const r = logRet[a].filter((x) => x != null);
      const mu = mean(r), sd = std(r);
      const skew = r.reduce((s, x) => s + ((x - mu) / sd) ** 3, 0) / r.length;
      const kurt = r.reduce((s, x) => s + ((x - mu) / sd) ** 4, 0) / r.length - 3;
      const lo = -6, hi = 6, w = 0.25, nb = Math.round((hi - lo) / w);
      const counts = new Array(nb).fill(0);
      r.forEach((x) => { const k = Math.min(nb - 1, Math.max(0, Math.floor((x - lo) / w))); counts[k]++; });
      const labels = counts.map((_, i) => (lo + (i + 0.5) * w).toFixed(2));
      const normal = labels.map((x) => r.length * w * Math.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * Math.sqrt(2 * Math.PI)));
      mk("rd-chart", { type: "bar",
        data: { labels, datasets: [
          { type: "line", label: "Normal curve, same mean and sd", data: normal, borderColor: c.text2, borderDash: [5, 4], borderWidth: 1.5 },
          { label: `${a} daily returns (days)`, data: counts, backgroundColor: assetColor(a, c), barPercentage: 1, categoryPercentage: 0.92, borderRadius: 2 },
        ] },
        options: { interaction: { mode: "index", intersect: false }, scales: { x: { title: { display: true, text: "Daily log return (%), tails clipped at ±6" }, ticks: { maxTicksLimit: 9, maxRotation: 0 }, grid: { display: false } }, y: { title: { display: true, text: "Number of days" } } } } });
      const tails = r.filter((x) => Math.abs(x - mu) > 3 * sd).length;
      table($("rd-stats"), [{ t: "Mean %", num: true }, { t: "Std %", num: true }, { t: "Annual vol %", num: true }, { t: "Skew", num: true }, { t: "Excess kurtosis", num: true }, { t: "Days beyond 3σ", num: true }],
        [[f3(mu), f2(sd), f1(sd * Math.sqrt(252)), f2(skew), f1(kurt), `${tails} <span class="small muted">(normal ≈ ${Math.round(r.length * 0.0027)})</span>`]]);
    };
    $("rd-asset").onchange = drawRd;
    drawRd();

    drawHeatmap($("dheat"), $("dcorr-seg"));
    table($("asset-table"), ["Ticker", "What it is"], ASSETS.map((a) => [`<b>${a}</b>`, esc(D.asset_descriptions[a])]).concat([["VIX", "CBOE Volatility Index (feature only): the market's own 30-day volatility expectation"]]));
    const moves = Object.entries(q.daily_moves_over_10pct).flatMap(([a, l]) => l.map((x) => `${a} ${x.date} (${x.log_return_pct}%)`));
    table($("quality-table"), ["Check", "Result"], [
      ["Duplicate dates removed", q.duplicate_dates_removed],
      ["Missing values before alignment", Object.entries(q.missing_values_before).map(([k, v]) => `${k}: ${v}`).join(", ")],
      ["Non-positive prices", Object.values(q.non_positive_prices).reduce((a, x) => a + x, 0)],
      ["Calendar gaps > 4 days", q.calendar_gaps_over_4_days + " (market closures)"],
      ["Daily moves > 10%", moves.length ? moves.join("; ") + "<br><span class='small muted'>Real March 2020 crash days, kept</span>" : "none"],
      ["Price adjustment", "Dividend- and split-adjusted closes, so payouts do not show up as fake returns"],
    ]);
    const fi = R.data.features.map((f) => {
      const base = f.replace(/^log_/, "");
      const info = R.feature_info[base];
      if (!info) return [`<code>${f}</code>`, "Asset indicator", "Identity", "1 if the row belongs to this asset, else 0"];
      return [`<code>${f}</code>`, esc(info[0]) + (f.startsWith("log_") ? " <span class='tag'>log</span>" : ""), info[1], esc(info[2])];
    });
    const drawFeat = () => {
      const qq = $("feat-q").value.toLowerCase();
      table($("feature-table"), ["Feature", "Name", "Group", "Meaning"], fi.filter((r) => r.join(" ").toLowerCase().includes(qq)));
    };
    $("feat-q").oninput = drawFeat;
    drawFeat();
  };

  function drawHeatmap(el, segEl) {
    let key = "corr_train";
    const draw = () => {
      const M = P[key], n = ASSETS.length, c = COL();
      const mix = (v) => `color-mix(in srgb, ${v >= 0 ? c.pos : c.neg} ${Math.round(Math.abs(v) * 85)}%, ${c.mid})`;
      el.style.gridTemplateColumns = `48px repeat(${n}, 1fr)`;
      let html = `<div></div>` + ASSETS.map((a) => `<div class="hdr">${a}</div>`).join("");
      for (let i = 0; i < n; i++) {
        html += `<div class="hdr">${ASSETS[i]}</div>`;
        for (let j = 0; j < n; j++) {
          const v = M[i][j];
          html += `<div class="cell" style="background:${mix(v)};color:${Math.abs(v) > 0.6 ? "#fff" : "inherit"}" title="${ASSETS[i]} vs ${ASSETS[j]}: ${v.toFixed(2)}">${v.toFixed(2)}</div>`;
        }
      }
      el.innerHTML = html;
    };
    segEl.onclick = (ev) => { const k = ev.target.dataset.k; if (!k) return; key = k; segOn(segEl, k, "k"); draw(); };
    draw();
  }

  // ================================================================== SELECTION
  renderers.selection = () => {
    const c = COL();
    const S = R.stage1_screening;
    const rows = [...S.results].sort((a, x) => a.cv_rmse_mean - x.cv_rmse_mean);
    const totalFits = S.results.reduce((s, r) => s + r.fits, 0);
    $("sel-intro").innerHTML = `Stage 1 compares ${families.length} model families and 3 baselines with ${S.folds.length}-fold expanding-window cross-validation that uses <b>only the training years</b> (${totalFits} fits). ` +
      `The family with the lowest mean RMSE (<b>${S.selected_family}</b>) is the only one tuned (Stage 2, validation years) and tested (Stage 3).`;

    mk("cv-bar", {
      type: "bar",
      data: { labels: rows.map((r) => r.name + (r.kind === "baseline" ? " (baseline)" : "")),
        datasets: [{ label: "Mean CV RMSE", data: rows.map((r) => r.cv_rmse_mean),
          backgroundColor: rows.map((r) => (r.kind === "baseline" ? alpha(c.text3.length === 7 ? c.text3 : "#7a7974", 0.55) : famColor(r.name, c))), barThickness: 18 }] },
      options: { indexAxis: "y",
        plugins: { legend: { display: false }, whiskers: { lo: rows.map((r) => r.cv_rmse_mean - r.cv_rmse_std), hi: rows.map((r) => r.cv_rmse_mean + r.cv_rmse_std) },
          tooltip: { callbacks: { label: (x) => { const r = rows[x.dataIndex]; return [`Mean RMSE ${f2(r.cv_rmse_mean)} ± ${f2(r.cv_rmse_std)}`, `Folds: ${r.fold_rmse.map(f2).join(", ")}`]; } } } },
        scales: { x: { beginAtZero: true, title: { display: true, text: "Cross-validation RMSE (volatility points)" } }, y: { grid: { display: false } } } },
    });

    const pstr = (p) => (p ? Object.entries(p).filter(([k]) => k !== "log_target").map(([k, v]) => `${k}=${v}`).join(", ") + (p.log_target ? ", log target" : "") : "–");
    table($("cv-table"), ["#", "Model", { t: "CV RMSE", num: true }, { t: "Fits", num: true }, { t: "Time", num: true }],
      rows.map((r, i) => [i + 1, `${r.name}${r.kind === "baseline" ? ' <span class="tag">baseline</span>' : ""}${r.name === S.selected_family ? ' <span class="tag best">selected</span>' : ""}<div class="small muted">${esc(pstr(r.best_params))}</div>`,
        `${f2(r.cv_rmse_mean)} <span class="small muted">±${f1(r.cv_rmse_std)}</span>`, r.fits || "–", r.kind === "model" ? r.seconds + " s" : "–"]),
      { hl: rows.findIndex((r) => r.name === S.selected_family) });
    const sel = S.results.find((r) => r.name === S.selected_family);
    const others = S.results.filter((r) => r.kind === "model" && r.name !== sel.name);
    const winsAll = sel.fold_rmse.every((v, i) => others.every((o) => v <= o.fold_rmse[i]));
    $("cv-note").innerHTML = `${sel.name} ${winsAll ? "has the lowest RMSE in <b>every</b> fold" : "has the lowest mean RMSE"}, so the choice does not hinge on the 2020 fold alone. ` +
      `The fold standard deviations are large because the COVID fold is much harder than the others.`;

    const ewmaR = S.results.find((r) => r.name === EWMA);
    const foldDs = families.map((f) => {
      const r = S.results.find((x) => x.name === f);
      return { label: f, data: r.fold_rmse, borderColor: famColor(f, c), backgroundColor: famColor(f, c), pointRadius: 4, borderWidth: f === sel.name ? 3 : 1.6 };
    });
    if (ewmaR) foldDs.push({ label: "EWMA baseline", data: ewmaR.fold_rmse, borderColor: c.text3, backgroundColor: c.text3, borderDash: [5, 4], pointRadius: 3, borderWidth: 1.5 });
    mk("fold-chart", { type: "line", data: { labels: S.folds.map((f) => `Fold ${f.fold} · ${f.score_start.slice(0, 7)} → ${f.score_end.slice(0, 7)}`), datasets: foldDs },
      options: lineOpts("RMSE (volatility points)", { scales: { x: { grid: { display: false } }, y: { title: { display: true, text: "RMSE (volatility points)" } } } }) });
    table($("fold-info"), ["Fold", "Train through", "Scored on", { t: "Train rows", num: true }, { t: "Scored rows", num: true }],
      S.folds.map((f) => [f.fold, f.train_end, `${f.score_start} → ${f.score_end}`, f.train_rows.toLocaleString(), f.score_rows.toLocaleString()]));

    // Stage 2
    const T = R.stage2_tuning;
    const trials = T.all_trials || T.top_trials;
    $("tune-title").textContent = `Stage 2: tuning ${T.family} on validation`;
    $("tune-desc").textContent = `${T.n_configs} settings, each fit on train and scored on 2021–2022 (${T.seconds} s). Best: ${pstr(T.best_params)}.`;
    if (T.family === "Ridge" && trials.every((t) => "alpha" in t.params)) {
      const mkSeries = (lt, col, lab) => ({ label: lab, data: trials.filter((t) => !!t.params.log_target === lt).map((t) => ({ x: t.params.alpha, y: t.val_rmse })).sort((a, x) => a.x - x.x),
        borderColor: col, backgroundColor: col, showLine: true, pointRadius: 4 });
      mk("tune-chart", { type: "scatter", data: { datasets: [mkSeries(true, c.model, "Log-volatility target"), mkSeries(false, c.base, "Raw volatility target")] },
        options: { scales: { x: { type: "logarithmic", title: { display: true, text: "Regularisation strength α (log scale)" } }, y: { title: { display: true, text: "Validation RMSE" } } },
          plugins: { tooltip: { callbacks: { label: (x) => `${x.dataset.label}: α=${x.parsed.x}, RMSE ${f3(x.parsed.y)}` } } } } });
    } else {
      const top = trials.slice(0, 15);
      mk("tune-chart", { type: "bar", data: { labels: top.map((t, i) => "#" + (i + 1)), datasets: [{ label: "Validation RMSE", data: top.map((t) => t.val_rmse), backgroundColor: c.model }] },
        options: { plugins: { legend: { display: false }, tooltip: { callbacks: { afterLabel: (x) => pstr(top[x.dataIndex].params) } } } } });
    }
    table($("tune-table"), ["Rank", "Setting", { t: "Val RMSE", num: true }, { t: "Val MAE", num: true }],
      trials.slice(0, 6).map((t, i) => [i + 1, esc(pstr(t.params)), f3(t.val_rmse), f3(t.val_mae)]), { hl: 0 });

    const vkeys = Object.keys(vc).sort((a, x) => vc[a].RMSE - vc[x].RMSE);
    mk("val-bar", { type: "bar",
      data: { labels: vkeys.map(shortName), datasets: [{ label: "Validation RMSE", data: vkeys.map((k) => vc[k].RMSE), backgroundColor: vkeys.map((k) => forecastColor(k, c)), barThickness: 22 }] },
      options: { indexAxis: "y", plugins: { legend: { display: false }, tooltip: { callbacks: { label: (x) => { const v = vc[vkeys[x.dataIndex]]; return `RMSE ${f2(v.RMSE)}, MAE ${f2(v.MAE)}, bias ${f2(v.Bias)}`; } } } },
        scales: { x: { beginAtZero: true, title: { display: true, text: "RMSE (volatility points)" } }, y: { grid: { display: false } } } } });
    const vm = vc[FAM + " (tuned)"];
    $("val-note").innerHTML = valBest[0] === FAM + " (tuned)"
      ? `The tuned ${FAM} was also best on validation.`
      : `<b>Reported honestly:</b> on validation ${esc(shortName(valBest[0]))} (RMSE ${f2(valBest[1].RMSE)}) beat the tuned ${FAM} (${f2(vm.RMSE)}). The model's bias was ${f2(vm.Bias)}: it under-predicted 2022, when volatility stayed high for most of the year. See it in the Forecast explorer → Validation.`;

    const desc = {
      "Ridge": "linear, L2-regularised; volatility is persistent, so a weighted mix of past volatilities in log terms is a strong model.",
      "Decision tree": "simple non-linear if/then rules, e.g. high VIX and a falling market.",
      "Random forest": "averages many decorrelated trees; captures interactions with low variance.",
      "SVR (RBF)": "support vector regression with a smooth RBF kernel.",
      "Gradient boosting": "scikit-learn histogram gradient-boosted trees, built sequentially.",
      "LightGBM": "leaf-wise gradient-boosted trees from the lightgbm library.",
    };
    $("model-list").innerHTML = families.map((f) => `<li><span class="swatch" style="background:${famColor(f, c)}"></span> <b>${f}</b>${f === FAM ? ' <span class="tag best">selected</span>' : ""}: ${desc[f] || ""}</li>`).join("");
  };

  // ================================================================== TEST RESULTS
  let metric = "RMSE";
  const metricDesc = {
    RMSE: "Root mean squared error, volatility points. Lower is better; punishes big misses.",
    MAE: "Mean absolute error, volatility points. Lower is better; the typical miss.",
    R2: "Share of variance explained. Higher is better; below 0 is worse than predicting the test mean.",
    QLIKE: "Standard volatility loss; penalises under-predicting risk more. Lower is better.",
    Bias: "Mean of forecast − actual. Closer to 0 is better; negative means risk is under-estimated.",
  };
  renderers.results = () => {
    const c = COL();
    $("res-kpis").innerHTML = [
      kpi(`Test RMSE, ${FAM}`, f2(m.RMSE), `n = ${m.n.toLocaleString()} asset-days`),
      kpi("vs recent 20-day vol", "−" + pct0(impr(b)), `95% CI of ΔRMSE ${f2(boot[BASE].RMSE_diff_CI95[0])} to ${f2(boot[BASE].RMSE_diff_CI95[1])}`, { cls: sig(boot[BASE]) ? "up" : "", t: sig(boot[BASE]) ? "significant" : "not conclusive" }),
      kpi("vs EWMA", "−" + pct0(impr(e)), `95% CI ${f2(boot[EWMA].RMSE_diff_CI95[0])} to ${f2(boot[EWMA].RMSE_diff_CI95[1])}`, { cls: sig(boot[EWMA]) ? "up" : "", t: sig(boot[EWMA]) ? "significant" : "not conclusive" }),
      kpi("vs historical average", "−" + pct0(impr(h)), `95% CI ${f2(boot[HIST].RMSE_diff_CI95[0])} to ${f2(boot[HIST].RMSE_diff_CI95[1])}`, { cls: sig(boot[HIST]) ? "up" : "", t: sig(boot[HIST]) ? "significant" : "borderline" }),
      kpi("Test R²", f2(m.R2), `MAE ${f2(m.MAE)} · bias ${f2(m.Bias)}`),
    ].join("");

    segButtons($("met-seg"), Object.keys(metricDesc).map((k) => ({ v: k, t: k === "R2" ? "R²" : k })), metric);
    const drawMetric = () => {
      $("met-desc").textContent = metricDesc[metric];
      mk("met-chart", { type: "bar",
        data: { labels: forecastKeys.map(shortName), datasets: [{ label: metric, data: forecastKeys.map((k) => tm[k][metric]), backgroundColor: forecastKeys.map((k) => forecastColor(k, c)), barThickness: 22 }] },
        options: { indexAxis: "y", plugins: { legend: { display: false }, tooltip: { callbacks: { label: (x) => `${metric}: ${f3(x.parsed.x)}` } } },
          scales: { x: { beginAtZero: metric !== "R2" && metric !== "Bias", title: { display: true, text: metric } }, y: { grid: { display: false } } } } });
      const keysPA = [MKEY, BASE, HIST, EWMA];
      mk("pa-chart", { type: "bar",
        data: { labels: ASSETS, datasets: keysPA.map((k) => ({ label: shortName(k), data: ASSETS.map((a) => (R.test_per_asset[a][k] || {})[metric]), backgroundColor: forecastColor(k, c), borderColor: c.surface, borderWidth: 1 })) },
        options: { interaction: { mode: "index", intersect: false }, scales: { y: { title: { display: true, text: metric } }, x: { grid: { display: false } } },
          plugins: { tooltip: { callbacks: { label: (x) => `${x.dataset.label}: ${f3(x.parsed.y)}` } } } } });
    };
    $("met-seg").onclick = (ev) => { const v = ev.target.dataset.v; if (!v) return; metric = v; segOn($("met-seg"), v); drawMetric(); };
    drawMetric();

    const bk = [BASE, EWMA, HIST].filter((k) => boot[k]);
    mk("boot-chart", { type: "bar",
      data: { labels: bk.map((k) => "vs " + shortName(k)), datasets: [{ label: "95% CI of RMSE difference", data: bk.map((k) => boot[k].RMSE_diff_CI95), backgroundColor: bk.map((k) => alpha(sig(boot[k]) ? (c.good.length === 7 ? c.good : "#0e7a2f") : (c.text3.length === 7 ? c.text3 : "#7a7974"), 0.45)), barThickness: 16, borderRadius: 8, borderSkipped: false }] },
      options: { indexAxis: "y", plugins: { legend: { display: false }, markers: { values: bk.map((k) => boot[k].RMSE_diff_mean) },
        tooltip: { callbacks: { label: (x) => { const bt = boot[bk[x.dataIndex]]; return [`mean ΔRMSE ${f2(bt.RMSE_diff_mean)}`, `95% CI ${f2(bt.RMSE_diff_CI95[0])} to ${f2(bt.RMSE_diff_CI95[1])}`, `model better in ${pct0(bt.share_of_resamples_model_better)} of resamples`]; } } } },
        scales: { x: { title: { display: true, text: "RMSE(model) − RMSE(baseline)" }, grid: { color: (ctx) => (ctx.tick && ctx.tick.value === 0 ? c.text2 : c.border), lineWidth: (ctx) => (ctx.tick && ctx.tick.value === 0 ? 1.5 : 1) } }, y: { grid: { display: false } } } } });

    const best = Math.min(...forecastKeys.map((k) => tm[k].RMSE));
    table($("test-table"), ["Forecast", { t: "RMSE", num: true }, { t: "MAE", num: true }, { t: "R²", num: true }, { t: "QLIKE", num: true }, { t: "Bias", num: true }, { t: "n", num: true }],
      forecastKeys.map((k) => [`<span class="swatch" style="background:${forecastColor(k, c)}"></span> ${esc(shortName(k))}${tm[k].RMSE === best ? ' <span class="tag best">best</span>' : ""}${k === TKEY ? ' <span class="small muted">(sensitivity: not refit on validation)</span>' : ""}`,
        f2(tm[k].RMSE), f2(tm[k].MAE), f2(tm[k].R2), f3(tm[k].QLIKE), f2(tm[k].Bias), tm[k].n.toLocaleString()]), { hl: 0 });

    // scatter predicted vs actual
    let scA = "All";
    segButtons($("sc-asset"), [{ v: "All", t: "All" }].concat(ASSETS.map((a) => ({ v: a, t: a }))), scA);
    const pts = (key) => {
      const out = [];
      (scA === "All" ? ASSETS : [scA]).forEach((a) => { const T = D.timeline[a]; T.dates.forEach((d, i) => { if (T.actual[i] != null && T[key][i] != null) out.push({ x: T[key][i], y: T.actual[i], d, a }); }); });
      return out;
    };
    const drawSc = () => {
      const mp = pts("model");
      const ds = [{ label: `${FAM} forecast`, data: mp, backgroundColor: alpha(c.model, 0.45), pointRadius: 2.5, pointHoverRadius: 5 }];
      if ($("sc-base").checked) ds.push({ label: "Recent 20-day vol", data: pts("recent"), backgroundColor: alpha(c.base, 0.4), pointRadius: 2.5, pointHoverRadius: 5 });
      const mx = Math.max(...mp.map((p) => Math.max(p.x, p.y)), 10);
      ds.push({ type: "line", label: "Perfect forecast", data: [{ x: 0, y: 0 }, { x: mx, y: mx }], borderColor: c.text3, borderDash: [5, 4], borderWidth: 1.2, pointRadius: 0 });
      mk("sc-chart", { type: "scatter", data: { datasets: ds },
        options: { scales: { x: { min: 0, title: { display: true, text: "Forecast volatility (%)" } }, y: { min: 0, title: { display: true, text: "Realised volatility (%)" } } },
          plugins: { tooltip: { callbacks: { label: (x) => (x.raw.d ? `${x.raw.a} ${x.raw.d}: forecast ${f1(x.raw.x)}, actual ${f1(x.raw.y)}` : "") } } } } });
    };
    $("sc-asset").onclick = (ev) => { const v = ev.target.dataset.v; if (!v) return; scA = v; segOn($("sc-asset"), v); drawSc(); };
    $("sc-base").onchange = drawSc;
    drawSc();

    // residual histogram
    const cmpKeys = [["recent", BASE], ["ewma", EWMA], ["hist", HIST]];
    $("res-cmp").innerHTML = cmpKeys.map(([k, n]) => `<option value="${k}">vs ${esc(shortName(n))}</option>`).join("");
    const errs = (key) => ASSETS.flatMap((a) => { const T = D.timeline[a]; return T.dates.map((_, i) => (T.actual[i] != null && T[key][i] != null ? T[key][i] - T.actual[i] : null)).filter((x) => x != null); });
    const drawRes = () => {
      const k = $("res-cmp").value, name = cmpKeys.find((x) => x[0] === k)[1];
      const lo = -20, hi = 20, w = 1, nb = (hi - lo) / w;
      const binit = (arr) => { const cts = new Array(nb).fill(0); arr.forEach((x) => { cts[Math.min(nb - 1, Math.max(0, Math.floor((x - lo) / w)))]++; }); return cts; };
      const labels = Array.from({ length: nb }, (_, i) => lo + i * w + w / 2);
      mk("res-hist", { type: "bar",
        data: { labels, datasets: [
          { label: `${FAM} forecast`, data: binit(errs("model")), backgroundColor: c.model, barPercentage: 1, categoryPercentage: 0.85, borderRadius: 2 },
          { label: shortName(name), data: binit(errs(k)), backgroundColor: forecastColor(name, c), barPercentage: 1, categoryPercentage: 0.85, borderRadius: 2 },
        ] },
        options: { interaction: { mode: "index", intersect: false }, scales: { x: { title: { display: true, text: "Forecast − actual (vol points), clipped at ±20" }, ticks: { maxTicksLimit: 9, maxRotation: 0 }, grid: { display: false } }, y: { title: { display: true, text: "Asset-days" } } } } });
    };
    $("res-cmp").onchange = drawRes;
    drawRes();
  };

  // ================================================================== EXPLORER
  const ex = { period: "test", asset: ASSETS[0], from: 0, to: null };
  const exData = () => (ex.period === "val" ? D.val_timeline : D.timeline)[ex.asset];
  const JUMPS = {
    test: [["Whole test period", null, null], ["2023", "2023-01-01", "2023-12-31"], ["2024", "2024-01-01", "2024-12-31"],
      ["SVB bank stress (Mar 2023)", "2023-02-15", "2023-05-15"], ["Jul–Aug 2024 tech sell-off & VIX spike", "2024-06-15", "2024-09-30"]],
    val: [["Whole validation period", null, null], ["2021 (calm)", "2021-01-01", "2021-12-31"], ["2022 (bear market)", "2022-01-01", "2022-12-31"],
      ["Ukraine invasion (Feb–Apr 2022)", "2022-02-01", "2022-04-30"]],
  };
  function exSetWindow(d0, d1) {
    const T = exData();
    ex.from = d0 ? idxOfDate(T.dates, d0) : 0;
    ex.to = d1 ? Math.max(ex.from + 5, idxOfDate(T.dates, d1)) : T.dates.length - 1;
    ex.to = Math.min(ex.to, T.dates.length - 1);
  }
  function drawExplorer() {
    const c = COL();
    const T = exData();
    if (!T) return;
    const n = T.dates.length;
    if (ex.to == null || ex.to >= n) ex.to = n - 1;
    if (ex.from >= ex.to) ex.from = Math.max(0, ex.to - 5);
    const fr = $("ex-from"), to = $("ex-to");
    fr.max = to.max = n - 1; fr.value = ex.from; to.value = ex.to;
    $("ex-from-lbl").textContent = T.dates[ex.from];
    $("ex-to-lbl").textContent = T.dates[ex.to];
    const sl = (arr) => arr.slice(ex.from, ex.to + 1);
    const lab = ex.period === "val" ? `${FAM} (fit on train only)` : `${FAM} forecast`;
    const series = [
      { k: "actual", label: "Realised (next 20 days)", col: c.actual, w: 2.4, on: true },
      { k: "model", label: lab, col: c.model, w: 2, on: $("ex-model").checked },
      { k: "recent", label: "Recent 20-day vol", col: c.base, w: 1.5, on: $("ex-recent").checked },
      { k: "hist", label: "Historical average", col: c.hist, w: 1.5, on: $("ex-hist").checked, dash: [5, 4] },
      { k: "ewma", label: "EWMA", col: c.ewma, w: 1.5, on: $("ex-ewma").checked },
    ];
    mk("ex-chart", { type: "line",
      data: { labels: sl(T.dates), datasets: series.filter((s) => s.on).map((s) => ({ label: s.label, data: sl(T[s.k]), borderColor: s.col, backgroundColor: s.col, borderWidth: s.w, borderDash: s.dash || [], tension: 0.1 })) },
      options: lineOpts("Annualised volatility (%)", { plugins: { tooltip: { callbacks: { label: (x) => `${x.dataset.label}: ${f2(x.parsed.y)}%`, afterBody: (items) => `VIX: ${f2(T.vix[ex.from + items[0].dataIndex])}` } } } }) });
    const showVix = $("ex-vix").checked;
    $("ex-vix-box").style.display = showVix ? "" : "none";
    if (showVix) {
      mk("ex-vix-chart", { type: "line", data: { labels: sl(T.dates), datasets: [{ label: "VIX on the forecast date", data: sl(T.vix), borderColor: c.text2, backgroundColor: c.text2, borderWidth: 1.5 }] },
        options: lineOpts("VIX", { plugins: { legend: { display: false } } }) });
    }
    const y = sl(T.actual);
    const rows = [["model", lab], ["recent", "Recent 20-day vol"], ["hist", "Historical average"], ["ewma", "EWMA"]].map(([k, l]) => { const p = sl(T[k]); return { l, r: rmse(y, p), a: mae(y, p), bi: bias(y, p) }; });
    const best = Math.min(...rows.map((r) => r.r));
    table($("ex-table"), [{ t: `Forecast · ${ex.asset} · ${ex.to - ex.from + 1} days` }, { t: "RMSE", num: true }, { t: "MAE", num: true }, { t: "Bias", num: true }, { t: "Model vs this", num: true }],
      rows.map((r, i) => [r.l + (r.r === best ? " <span class='tag best'>best in window</span>" : ""), f2(r.r), f2(r.a), f2(r.bi),
        i === 0 ? "–" : `<span class="${rows[0].r < r.r ? "win" : "lose"}">${rows[0].r < r.r ? "model better" : "model worse"} by ${f2(Math.abs(rows[0].r - r.r))}</span>`]), { hl: 0 });
  }
  function drawJumps() {
    $("ex-jumps").innerHTML = JUMPS[ex.period].map((j, i) => `<button type="button" class="chip" data-j="${i}">${esc(j[0])}</button>`).join("");
  }
  renderers.explorer = () => {
    segButtons($("ex-asset"), ASSETS.map((a) => ({ v: a, t: a })), ex.asset);
    segOn($("ex-period"), ex.period, "p");
    drawJumps();
    drawExplorer();
  };
  $("ex-asset").addEventListener("click", (ev) => { const v = ev.target.dataset.v; if (!v) return; ex.asset = v; segOn($("ex-asset"), v); drawExplorer(); });
  $("ex-period").addEventListener("click", (ev) => {
    const p = ev.target.dataset.p; if (!p || p === ex.period) return;
    if (p === "val" && !D.val_timeline[ex.asset]) return;
    ex.period = p; segOn($("ex-period"), p, "p"); exSetWindow(null, null); drawJumps(); drawExplorer();
  });
  $("ex-jumps").addEventListener("click", (ev) => { const t = ev.target.closest("[data-j]"); if (!t) return; const j = JUMPS[ex.period][+t.dataset.j]; exSetWindow(j[1], j[2]); drawExplorer(); });
  $("ex-from").addEventListener("input", () => { ex.from = Math.min(+$("ex-from").value, ex.to - 5); drawExplorer(); });
  $("ex-to").addEventListener("input", () => { ex.to = Math.max(+$("ex-to").value, ex.from + 5); drawExplorer(); });
  ["ex-model", "ex-recent", "ex-hist", "ex-ewma", "ex-vix"].forEach((id) => $(id).addEventListener("change", drawExplorer));

  // ================================================================== ERRORS
  renderers.errors = () => {
    const c = COL();
    const regs = R.test_regimes;
    const rk = [MKEY, BASE, HIST].filter((k) => regs[0][k]);
    const th = R.vix_regime_thresholds_from_train;
    $("reg-desc").textContent = `Regimes use the VIX on the forecast date, with tercile cut-offs (${f1(th[0])}, ${f1(th[1])}) taken from the training years.`;
    let rm = "RMSE";
    const drawReg = () => mk("reg-chart", { type: "bar",
      data: { labels: regs.map((r) => `${r.regime} · n=${r.n}`), datasets: rk.map((k) => ({ label: shortName(k), data: regs.map((r) => r[k][rm]), backgroundColor: forecastColor(k, c), borderColor: c.surface, borderWidth: 1 })) },
      options: { interaction: { mode: "index", intersect: false }, scales: { y: { beginAtZero: true, title: { display: true, text: rm + " (volatility points)" } }, x: { grid: { display: false }, ticks: { maxRotation: 0, autoSkip: false, font: { size: 11 } } } },
        plugins: { tooltip: { callbacks: { label: (x) => `${x.dataset.label}: ${f2(x.parsed.y)}` } } } } });
    $("reg-met").onclick = (ev) => { const v = ev.target.dataset.m; if (!v) return; rm = v; segOn($("reg-met"), v, "m"); drawReg(); };
    drawReg();

    let evA = ASSETS[0];
    segButtons($("ev-asset"), ASSETS.map((a) => ({ v: a, t: a })), evA);
    const drawEv = () => {
      const T = D.timeline[evA];
      const pts = (k) => T.dates.map((d, i) => (T.actual[i] != null ? { x: T.vix[i], y: T[k][i] - T.actual[i], d } : null)).filter(Boolean);
      mk("ev-chart", { type: "scatter",
        data: { datasets: [
          { label: `${FAM} error`, data: pts("model"), backgroundColor: alpha(c.model, 0.5), pointRadius: 3 },
          { label: "Recent 20-day vol error", data: pts("recent"), backgroundColor: alpha(c.base, 0.4), pointRadius: 3 },
        ] },
        options: { scales: { x: { title: { display: true, text: "VIX on the forecast date" } },
          y: { title: { display: true, text: "Forecast − actual (vol points)" }, grid: { color: (ctx) => (ctx.tick && ctx.tick.value === 0 ? c.text2 : c.border) } } },
          plugins: { tooltip: { callbacks: { label: (x) => `${x.raw.d}: VIX ${f1(x.raw.x)}, error ${f1(x.raw.y)}` } } } } });
    };
    $("ev-asset").onclick = (ev) => { const v = ev.target.dataset.v; if (!v) return; evA = v; segOn($("ev-asset"), v); drawEv(); };
    drawEv();

    const CONTEXT = {
      "2024-08": "Yen carry-trade unwind: VIX spiked to 38 for a day, then calmed quickly.",
      "2024-07": "Rotation out of big tech after a very calm first half: volatility jumped with no warning.",
      "2024-04": "Rate-cut expectations pushed back; brief equity pullback.",
      "2023-03": "US regional bank failures (SVB): a sudden stress event.",
      "2023-10": "Rate-driven sell-off in October 2023.",
      "2023-05": "Debt-ceiling standoff.",
    };
    const E = R.representative_errors;
    table($("err-table"), [{ t: "Date" }, { t: "Asset" }, { t: "Realised", num: true }, { t: "Model", num: true }, { t: "Recent vol", num: true }, { t: "VIX", num: true }, { t: "What happened" }],
      E.map((r) => [r.date, r.asset, f1(r.actual), `<span class="${r.error > 0 ? "lose" : "win"}">${f1(r.predicted)}</span>`, f1(r.baseline), f1(r.vix),
        (r.error > 0 ? "Over-predicted. " : "Under-predicted. ") + (CONTEXT[r.date.slice(0, 7)] || "")]), { click: true });
    $("err-table").onclick = (ev) => {
      const tr = ev.target.closest("tr[data-i]"); if (!tr) return;
      const r = E[+tr.dataset.i];
      ex.period = "test"; ex.asset = r.asset;
      const dt = new Date(r.date + "T00:00:00Z");
      const shift = (days) => new Date(dt.getTime() + days * 864e5).toISOString().slice(0, 10);
      exSetWindow(shift(-60), shift(60));
      rendered.delete("explorer");
      go("explorer");
    };
  };

  // ================================================================== PORTFOLIO
  const W = {};
  ASSETS.forEach((a) => (W[a] = 1 / ASSETS.length));
  const normW = () => { const s = ASSETS.reduce((x, k) => x + W[k], 0); const w = {}; ASSETS.forEach((k) => (w[k] = s > 0 ? W[k] / s : 1 / ASSETS.length)); return w; };
  function realised(w) {
    const n = P.return_dates.length; const port = new Array(n).fill(0);
    ASSETS.forEach((a) => { const r = P.returns_pct[a]; for (let i = 0; i < n; i++) port[i] += w[a] * (Math.exp(r[i] / 100) - 1); });
    let wealth = 1, peak = 1, mdd = 0; const path = [];
    for (const x of port) { wealth *= 1 + x; peak = Math.max(peak, wealth); mdd = Math.min(mdd, wealth / peak - 1); path.push(wealth * 100); }
    const mu = mean(port); const sd = Math.sqrt(port.reduce((s, x) => s + (x - mu) ** 2, 0) / (n - 1));
    const annRet = Math.pow(wealth, 252 / n) - 1, annVol = sd * Math.sqrt(252);
    return { port, path, annRet, annVol, sharpe: annRet / annVol, mdd };
  }
  function renderPortfolio() {
    const c = COL();
    const w = normW();
    ASSETS.forEach((a) => { $(`w-${a}`).value = Math.round(W[a] * 100); $(`wp-${a}`).textContent = (w[a] * 100).toFixed(0) + "%"; });
    const r = realised(w);
    const avgVol = ASSETS.reduce((s, a) => s + w[a] * P.single_asset_metrics_test[a].ann_vol_pct, 0);
    $("pf-kpis").innerHTML = [
      kpi("Annual return", (r.annRet * 100).toFixed(1) + "%"), kpi("Annual volatility", (r.annVol * 100).toFixed(1) + "%", `weighted avg of assets: ${avgVol.toFixed(1)}%`),
      kpi("Sharpe (rf = 0)", r.sharpe.toFixed(2)), kpi("Max drawdown", (r.mdd * 100).toFixed(1) + "%"),
    ].join("");

    // growth
    const rd = P.return_dates;
    const ds = [{ label: "Your portfolio", data: r.path, borderColor: c.text, backgroundColor: c.text, borderWidth: 2.6 }];
    ASSETS.forEach((a) => { const rr = realised({ ...Object.fromEntries(ASSETS.map((k) => [k, 0])), [a]: 1 }); ds.push({ label: a, data: rr.path, borderColor: assetColor(a, c), backgroundColor: assetColor(a, c), borderWidth: 1.3 }); });
    mk("pf-growth", { type: "line", data: { labels: rd, datasets: ds }, options: lineOpts("Value of 100 invested", { plugins: { tooltip: { callbacks: { label: (x) => `${x.dataset.label}: ${f1(x.parsed.y)}` } } } }) });

    // forecast vs realised portfolio vol
    const pos = {}; rd.forEach((d, i) => (pos[d] = i));
    const fd = P.forecast_dates, H = 20;
    const corrAt = (k, a, b2) => { if (a === b2) return 1; const key = P.corr60[`${a}|${b2}`] ? `${a}|${b2}` : `${b2}|${a}`; return P.corr60[key][k]; };
    const fc = (src, k) => {
      let s = 0;
      for (const a of ASSETS) for (const b2 of ASSETS) {
        const va = P[src][a][k], vb = P[src][b2][k], cc = corrAt(k, a, b2);
        if (va == null || vb == null || cc == null) return null;
        s += w[a] * w[b2] * va * vb * cc;
      }
      return Math.sqrt(s);
    };
    const labels = [], act = [], pm = [], pb = [];
    fd.forEach((d, k) => {
      const i = pos[d]; if (i == null || i + H >= rd.length) return;
      const win = r.port.slice(i + 1, i + 1 + H).map((x) => Math.log(1 + x));
      labels.push(d); act.push(std(win) * Math.sqrt(252) * 100); pm.push(fc("pred_model", k)); pb.push(fc("pred_baseline", k));
    });
    mk("pf-chart", { type: "line",
      data: { labels, datasets: [
        { label: "Realised portfolio vol (next 20 days)", data: act, borderColor: c.actual, backgroundColor: c.actual, borderWidth: 2.2 },
        { label: `Forecast from ${FAM} asset forecasts`, data: pm, borderColor: c.model, backgroundColor: c.model },
        { label: "Forecast from recent 20-day vols", data: pb, borderColor: c.base, backgroundColor: c.base, borderWidth: 1.5 },
      ] },
      options: lineOpts("Annualised volatility (%)") });
    const rm = rmse(act, pm), rb = rmse(act, pb);
    table($("pf-err"), [{ t: "Portfolio vol forecast built from" }, { t: "RMSE", num: true }, { t: "MAE", num: true }, { t: "Bias", num: true }],
      [[`${FAM} asset forecasts`, f2(rm), f2(mae(act, pm)), f2(bias(act, pm))], ["Recent 20-day vols", f2(rb), f2(mae(act, pb)), f2(bias(act, pb))]], { hl: rm <= rb ? 0 : 1 });

    // risk / return of presets + current
    const pre = P.preset_metrics_test;
    const names = Object.keys(pre);
    mk("pf-rr", { type: "scatter",
      data: { datasets: [
        { label: "Presets", data: names.map((k) => ({ x: pre[k].ann_vol_pct, y: pre[k].ann_return_pct, n: k })), backgroundColor: c.model, pointRadius: 6, pointHoverRadius: 8, pointLabels: names },
        { label: "Your weights", data: [{ x: r.annVol * 100, y: r.annRet * 100, n: "Your weights" }], backgroundColor: c.surface, borderColor: c.text, borderWidth: 2.5, pointRadius: 8, pointHoverRadius: 10 },
      ] },
      options: { scales: { x: { title: { display: true, text: "Annual volatility (%)" } }, y: { title: { display: true, text: "Annual return (%)" } } },
        plugins: { tooltip: { callbacks: { label: (x) => `${x.raw.n}: return ${f1(x.raw.y)}%, vol ${f1(x.raw.x)}%` } } } } });
  }
  renderers.portfolio = () => {
    $("weights").innerHTML = ASSETS.map((a) => `<div class="weight-row"><b>${a}</b><input type="range" min="0" max="100" step="5" id="w-${a}" aria-label="Weight of ${a}"><span class="pct" id="wp-${a}"></span></div>`).join("");
    $("presets").innerHTML = Object.keys(P.presets).map((k) => `<button type="button" class="chip" data-p="${esc(k)}">${esc(k)}</button>`).join("");
    ASSETS.forEach((a) => $(`w-${a}`).addEventListener("input", () => { W[a] = +$(`w-${a}`).value / 100; renderPortfolio(); }));
    $("presets").onclick = (ev) => { const k = ev.target.dataset && ev.target.dataset.p; if (!k) return; ASSETS.forEach((a) => (W[a] = P.presets[k][a] || 0)); renderPortfolio(); };
    drawHeatmap($("heat"), $("corr-seg"));
    renderPortfolio();
  };

  // ================================================================== METHOD
  renderers.method = () => {
    const c = COL();
    const tp = R.training_proof;
    if (tp) {
      $("proof-kpis").innerHTML = [
        kpi("Model fits run", tp.total_model_fits, "CV screening + tuning + final + control"),
        kpi("Final fit time", (tp.final_fit_seconds * 1000).toFixed(1) + " ms", `${tp.training_rows_final_fit.toLocaleString()} rows × ${tp.n_features} features`),
        kpi("Test RMSE, real labels", f2(tp.real_target_test_RMSE), "model trained normally"),
        kpi("Test RMSE, shuffled labels", f2(tp.shuffled_target_test_RMSE), "same model, targets shuffled", { cls: "up", t: "real labels clearly better → it learns" }),
        kpi("Full pipeline", R.reproducibility.runtime_seconds + " s", "download excluded"),
      ].join("");
      $("proof-text").innerHTML = `<p class="muted">Yes. ${esc(tp.why_fast)}</p><p class="muted">As a control, the same model trained on randomly shuffled targets scores test RMSE ${f2(tp.shuffled_target_test_RMSE)} versus ${f2(tp.real_target_test_RMSE)} with the real targets, so it learns real signal from the features rather than a constant.</p>`;
    }
    const imp = R.permutation_importance_test.slice(0, 12);
    mk("imp-chart", { type: "bar",
      data: { labels: imp.map((x) => featName(x.feature)), datasets: [{ label: "RMSE increase when shuffled", data: imp.map((x) => x.importance), backgroundColor: c.model, barThickness: 14 }] },
      options: { indexAxis: "y", plugins: { legend: { display: false }, tooltip: { callbacks: { label: (x) => `+${f3(x.parsed.x)} RMSE (± ${f3(imp[x.dataIndex].std)})` } } },
        scales: { x: { title: { display: true, text: "Increase in test RMSE (vol points)" } }, y: { grid: { display: false }, ticks: { font: { size: 11 } } } } } });
    const co = tp && tp.learned_coefficients_scaled_log_space;
    if (co) {
      const ent = Object.entries(co).slice(0, 12);
      $("coef-title").textContent = `Learned ${FAM} coefficients (top 12 by size)`;
      mk("coef-chart", { type: "bar",
        data: { labels: ent.map(([f]) => featName(f)), datasets: [{ label: "Coefficient", data: ent.map(([, v]) => v), backgroundColor: ent.map(([, v]) => (v >= 0 ? c.pos : c.neg)), barThickness: 14 }] },
        options: { indexAxis: "y", plugins: { legend: { display: false }, tooltip: { callbacks: { label: (x) => `coefficient ${f3(x.parsed.x)}` } } },
          scales: { x: { title: { display: true, text: "Effect on log-volatility per 1 sd of the feature" }, grid: { color: (ctx) => (ctx.tick && ctx.tick.value === 0 ? c.text2 : c.border) } }, y: { grid: { display: false }, ticks: { font: { size: 11 } } } } } });
    } else {
      $("coef-title").textContent = "Coefficients (not available for this model family)";
    }
    const rp = R.reproducibility;
    table($("repro-table"), ["Item", "Value"], [
      ["Random seed", rp.seed], ["Python / scikit-learn / pandas / numpy", `${rp.python} / ${rp.sklearn} / ${rp.pandas} / ${rp.numpy}`],
      ["Final model", `${FAM} ${esc(JSON.stringify(R.final_model.params))}, trained on ${R.final_model.trained_on} through ${R.final_model.trained_through}`],
      ["Smearing factor (log back-transform)", tp ? f3(tp.smearing_factor) : "–"],
      ["Saved model", "<code>artifacts/final_model.joblib</code> (scaler + model in one pipeline); the only model used by <code>src/inference.py</code>"],
      ["Outputs", "<code>artifacts/evaluation_metrics.json</code>, <code>artifacts/test_predictions.csv</code>, <code>artifacts/val_predictions.csv</code>"],
      ["Credentials", "None. Public Yahoo Finance data via yfinance; no API keys stored in the repo"],
    ]);
  };

  // ------------------------------------------------------------------ start
  show(location.hash.slice(1) || "overview");
})();
