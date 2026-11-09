/* Portfolio Risk Analyzer - front-end. All numbers come from window.SITE_DATA
   (docs/data/site_data.js, written by scripts/export_web_data.py). */
(function () {
  "use strict";
  const D = window.SITE_DATA;
  if (!D) {
    document.getElementById("headline-answer").textContent =
      "Could not load data/site_data.js. Run `python scripts/run_all.py` to generate it.";
    return;
  }
  const R = D.results;
  const FAM = R.final_model.family;
  const MKEY = FAM + " (final)";
  const BASE = D.primary_baseline;
  const ASSETS = D.assets;

  const $ = (id) => document.getElementById(id);
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const f2 = (x) => (x == null || isNaN(x) ? "–" : Number(x).toFixed(2));
  const f1 = (x) => (x == null || isNaN(x) ? "–" : Number(x).toFixed(1));
  const pct = (x) => (x * 100).toFixed(0) + "%";
  const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  const table = (el, head, rows, hlIndex) => {
    el.innerHTML =
      "<thead><tr>" + head.map((h) => `<th class="${h.num ? "num" : ""}">${h.t || h}</th>`).join("") + "</tr></thead><tbody>" +
      rows.map((r, i) => `<tr class="${i === hlIndex ? "hl" : ""}">` + r.map((c, j) => `<td class="${head[j].num ? "num" : ""}">${c}</td>`).join("") + "</tr>").join("") +
      "</tbody>";
  };
  const rmse = (y, p) => {
    let s = 0, n = 0;
    for (let i = 0; i < y.length; i++) if (y[i] != null && p[i] != null) { s += (y[i] - p[i]) ** 2; n++; }
    return n ? Math.sqrt(s / n) : NaN;
  };
  const mae = (y, p) => {
    let s = 0, n = 0;
    for (let i = 0; i < y.length; i++) if (y[i] != null && p[i] != null) { s += Math.abs(y[i] - p[i]); n++; }
    return n ? s / n : NaN;
  };

  Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
  Chart.defaults.color = css("--text-2");
  Chart.defaults.borderColor = css("--border");
  Chart.defaults.plugins.legend.labels.boxHeight = 2;
  Chart.defaults.plugins.legend.labels.boxWidth = 16;
  const C = () => ({
    actual: css("--s-actual"), model: css("--s-model"), base: css("--s-base"),
    hist: css("--s-hist"), ewma: css("--s-ewma"), border: css("--border"), text2: css("--text-2"),
    surface: css("--surface"),
  });

  // ---------------------------------------------------------------- headline
  const tm = R.test_metrics;
  const m = tm[MKEY], b = tm[BASE], h = tm["Historical average"], e = tm["EWMA vol (RiskMetrics)"];
  const bootB = R.test_bootstrap_vs_baselines[BASE];
  const bootH = R.test_bootstrap_vs_baselines["Historical average"];
  const bootE = R.test_bootstrap_vs_baselines["EWMA vol (RiskMetrics)"];
  const vc = R.validation_comparison;
  const valModel = vc[FAM + " (tuned)"];
  const valBest = Object.entries(vc).sort((a, c) => a[1].RMSE - c[1].RMSE)[0];
  const ciTxt = (bt) => `95% CI of the difference ${f2(bt.RMSE_diff_CI95[0])} to ${f2(bt.RMSE_diff_CI95[1])}`;
  const sig = (bt) => bt.RMSE_diff_CI95[1] < 0;

  $("headline-answer").innerHTML =
    `<b>Short answer: partly.</b> On the untouched 2023–2024 test set, the selected model (<b>${FAM}</b>) had an RMSE of ` +
    `<b>${f2(m.RMSE)}</b> volatility points, against <b>${f2(h.RMSE)}</b> for the historical average, <b>${f2(b.RMSE)}</b> for recent 20-day volatility ` +
    `and <b>${f2(e.RMSE)}</b> for EWMA. ` +
    `The improvement over recent volatility ${sig(bootB) ? "holds up" : "does not hold up"} under a block bootstrap (${ciTxt(bootB)}), ` +
    `${sig(bootH) ? "and so does" : "but"} the improvement over the historical average ${sig(bootH) ? "" : "is within noise "}(${ciTxt(bootH)}). ` +
    `On the 2021–2022 validation years the ${valBest[0] === FAM + " (tuned)" ? "model was best" : "<b>" + esc(valBest[0]) + " beat the model</b>"}, so the advantage depends on the market regime.`;

  const tiles = [
    { l: `Test RMSE, ${FAM}`, v: f2(m.RMSE), s: "volatility points (lower is better)" },
    { l: "Test RMSE, historical average", v: f2(h.RMSE), s: `model better in ${pct(bootH.share_of_resamples_model_better)} of bootstrap resamples` },
    { l: "Test RMSE, recent 20-day vol", v: f2(b.RMSE), s: `model better in ${pct(bootB.share_of_resamples_model_better)} of resamples` },
    { l: "Test MAE, model vs EWMA", v: `${f2(m.MAE)} / ${f2(e.MAE)}`, s: "typical miss in volatility points" },
  ];
  $("headline-tiles").innerHTML = tiles.map((t) => `<div class="tile"><div class="label">${t.l}</div><div class="value">${t.v}</div><div class="sub">${t.s}</div></div>`).join("");

  // ---------------------------------------------------------------- data
  const q = R.data.quality;
  $("data-intro").innerHTML =
    `Daily closing prices from Yahoo Finance (public, accessed with <code>yfinance</code>, no credentials needed). Source of this build: <i>${esc(q.source)}</i>. ` +
    `After cleaning: ${q.rows_after_alignment.toLocaleString()} trading days from ${q.date_range[0]} to ${q.date_range[1]}, giving ${R.data.n_rows_model_table.toLocaleString()} asset-day rows ` +
    `after the warm-up period (the longest feature window is 120 days).`;
  table($("asset-table"), ["Ticker", "What it is"], ASSETS.map((a) => [`<b>${a}</b>`, esc(D.asset_descriptions[a])]).concat([["VIX", "CBOE Volatility Index (feature only): the market's own 30-day volatility expectation"]]));
  const moves = Object.entries(q.daily_moves_over_10pct).flatMap(([a, l]) => l.map((x) => `${a} ${x.date} (${x.log_return_pct}%)`));
  table($("quality-table"), ["Check", "Result"], [
    ["Duplicate dates removed", q.duplicate_dates_removed],
    ["Missing values (before alignment)", Object.entries(q.missing_values_before).map(([k, v]) => `${k}: ${v}`).join(", ")],
    ["Non-positive prices", Object.values(q.non_positive_prices).reduce((a, c) => a + c, 0)],
    ["Calendar gaps > 4 days", q.calendar_gaps_over_4_days + " (market closures)"],
    ["Daily moves > 10%", moves.length ? moves.join("; ") + "<br><span class='small muted'>Real March 2020 crash days, kept as they are</span>" : "none"],
  ]);

  const sp = R.splits;
  const tot = sp.train.dates + sp.val.dates + sp.test.dates;
  const bar = $("split-bar");
  const seg = (lab, n, col) => `<div style="flex:${n};background:${col}" title="${lab}">${lab}</div>`;
  bar.innerHTML = seg(`Train ${sp.train.start.slice(0, 4)}–${sp.train.end.slice(0, 4)}`, sp.train.dates, css("--s-model")) +
    `<div class="gap" style="flex:${tot * 0.012}"></div>` +
    seg(`Validation ${sp.val.start.slice(0, 4)}–${sp.val.end.slice(0, 4)}`, sp.val.dates, css("--s-ewma")) +
    `<div class="gap" style="flex:${tot * 0.012}"></div>` +
    seg(`Test ${sp.test.start.slice(0, 4)}–${sp.test.end.slice(0, 4)}`, sp.test.dates, css("--s-base"));
  $("split-text").innerHTML =
    `Train ${sp.train.start} → ${sp.train.end} (${sp.train.rows.toLocaleString()} rows) · Validation ${sp.val.start} → ${sp.val.end} (${sp.val.rows} rows) · ` +
    `Test ${sp.test.start} → ${sp.test.end} (${sp.test.rows} rows). Hatched gaps: 20-day embargo, where rows whose target window crosses into the next period are dropped.`;
  const fi = R.data.features.map((f) => {
    const base = f.replace(/^log_/, "");
    const info = R.feature_info[base];
    if (!info) return [`<code>${f}</code>`, "Asset indicator", "Identity", "1 if the row belongs to this asset, else 0"];
    return [`<code>${f}</code>`, esc(info[0]) + (f.startsWith("log_") ? " <span class='tag'>log</span>" : ""), info[1], esc(info[2])];
  });
  table($("feature-table"), ["Feature", "Name", "Group", "Meaning"], fi);

  // ---------------------------------------------------------------- selection
  const scr = R.stage1_screening;
  const scrRows = scr.results.slice().sort((a, c) => a.cv_rmse_mean - c.cv_rmse_mean);
  $("selection-intro").innerHTML =
    `Stage 1 uses ${scr.folds.length} expanding-window folds inside the training years. Each family competes with the best of a small hyperparameter grid, and the baselines are scored on the same folds. ` +
    `<b>${FAM}</b> had the lowest mean CV RMSE and was the only family carried forward. The other families were never tuned on validation and never touched the test set.`;
  const cc = C();
  const cvLabels = scrRows.map((r) => r.name + (r.kind === "baseline" ? " (baseline)" : ""));
  new Chart($("cvChart"), {
    type: "bar",
    data: {
      labels: cvLabels,
      datasets: [
        {
          label: "Mean CV RMSE", order: 2, data: scrRows.map((r) => r.cv_rmse_mean), borderRadius: 4, borderSkipped: "start", barPercentage: 0.7,
          backgroundColor: scrRows.map((r) => (r.name === FAM ? cc.model : r.kind === "baseline" ? cc.base : cc.model + "55")),
        },
        {
          type: "scatter", label: "Individual folds", data: scrRows.flatMap((r, i) => r.fold_rmse.map((v) => ({ x: v, y: cvLabels[i] }))),
          backgroundColor: cc.text2, pointRadius: 4, pointBorderColor: cc.surface, pointBorderWidth: 1.5, order: 1,
        },
      ],
    },
    options: {
      indexAxis: "y", maintainAspectRatio: false,
      scales: {
        x: { title: { display: true, text: "RMSE (volatility points)" }, beginAtZero: true },
        y: { grid: { display: false } },
      },
      plugins: {
        legend: { position: "bottom" },
        tooltip: { callbacks: { label: (ctx) => ctx.dataset.type === "scatter" ? `fold RMSE ${f2(ctx.raw.x)}` : `mean ${f2(ctx.raw)} ± ${f2(scrRows[ctx.dataIndex].cv_rmse_std)}` } },
      },
    },
  });
  table($("fold-table"),
    [{ t: "Model" }].concat(scr.folds.map((f) => ({ t: `Fold ${f.fold}<br><span class="small">${f.score_start.slice(0, 7)} → ${f.score_end.slice(0, 7)}</span>`, num: true }))).concat([{ t: "Mean", num: true }, { t: "Fits", num: true }]),
    scrRows.map((r) => [r.name + (r.kind === "baseline" ? " <span class='tag'>baseline</span>" : "")].concat(r.fold_rmse.map(f2)).concat([`<b>${f2(r.cv_rmse_mean)}</b>`, r.fits || "–"])),
    scrRows.findIndex((r) => r.name === FAM));
  const foldWinners = scr.folds.map((_, k) => scrRows.filter((r) => r.kind === "model").sort((a, c) => a.fold_rmse[k] - c.fold_rmse[k])[0].name);
  const totalFits = scrRows.reduce((a, r) => a + (r.fits || 0), 0);
  $("cv-note").innerHTML =
    `Best ML family in each fold: ${foldWinners.map((w, i) => `fold ${i + 1}: ${w}`).join(", ")}. The last fold contains the 2020 COVID shock and dominates the mean. ` +
    `Training is fast (${totalFits} fits in a few seconds) because the table is small (about ${(R.splits.train.rows / 1000).toFixed(1)}k rows × ${R.data.features.length} features) and the models are classical. ` +
    `A linear fit takes milliseconds. Runtime is not a sign of quality either way.`;

  const tune = R.stage2_tuning;
  table($("tune-table"), [{ t: "#" }, { t: "Settings" }, { t: "Val RMSE", num: true }, { t: "Val MAE", num: true }],
    tune.top_trials.slice(0, 8).map((t, i) => [i + 1, Object.entries(t.params).map(([k, v]) => `${k}=${v}`).join(", "), f2(t.val_rmse), f2(t.val_mae)]), 0);
  const vrows = Object.entries(vc).sort((a, c) => a[1].RMSE - c[1].RMSE);
  table($("val-table"), [{ t: "Forecast" }, { t: "RMSE", num: true }, { t: "MAE", num: true }, { t: "Bias", num: true }],
    vrows.map(([k, v]) => [esc(k), f2(v.RMSE), f2(v.MAE), f2(v.Bias)]), vrows.findIndex(([k]) => k === FAM + " (tuned)"));
  $("val-note").innerHTML =
    `We tested ${tune.n_configs} settings and picked <code>${Object.entries(tune.best_params).map(([k, v]) => `${k}=${v}`).join(", ")}</code>. ` +
    (valBest[0] !== FAM + " (tuned)"
      ? `<b>Honest note:</b> on validation, ${esc(valBest[0])} (RMSE ${f2(valBest[1].RMSE)}) beat the tuned model (${f2(valModel.RMSE)}). The model under-predicted volatility in 2022 (bias ${f2(valModel.Bias)}), when volatility stayed high for most of the year. We kept the pre-registered protocol and did not switch models after seeing this.`
      : `The tuned model was also the best forecast on validation.`);

  // ---------------------------------------------------------------- test results
  const order = [MKEY, FAM + " (train-only fit)", "Historical average", BASE, "EWMA vol (RiskMetrics)"];
  table($("test-table"),
    [{ t: "Forecast" }, { t: "RMSE", num: true }, { t: "MAE", num: true }, { t: "R²", num: true }, { t: "QLIKE", num: true }, { t: "Bias", num: true }],
    order.map((k) => [esc(k) + (k === MKEY ? " <span class='tag'>selected, refit on train+val</span>" : k.includes("train-only") ? " <span class='tag'>sensitivity</span>" : " <span class='tag'>baseline</span>"),
      f2(tm[k].RMSE), f2(tm[k].MAE), f2(tm[k].R2), tm[k].QLIKE.toFixed(3), f2(tm[k].Bias)]), 0);
  $("boot-note").innerHTML =
    `<b>Uncertainty.</b> Moving-block bootstrap over test dates (20-day blocks, ${R.reproducibility.seed ? "seeded" : ""}): model RMSE 95% CI ${f2(bootB.model_RMSE_CI95[0])}–${f2(bootB.model_RMSE_CI95[1])}. ` +
    `RMSE difference vs recent vol: ${f2(bootB.RMSE_diff_mean)} [${f2(bootB.RMSE_diff_CI95[0])}, ${f2(bootB.RMSE_diff_CI95[1])}]; ` +
    `vs EWMA: ${f2(bootE.RMSE_diff_mean)} [${f2(bootE.RMSE_diff_CI95[0])}, ${f2(bootE.RMSE_diff_CI95[1])}]; ` +
    `vs historical average: ${f2(bootH.RMSE_diff_mean)} [${f2(bootH.RMSE_diff_CI95[0])}, ${f2(bootH.RMSE_diff_CI95[1])}]. ` +
    `An interval entirely below 0 means the model is better beyond resampling noise. QLIKE is a standard loss for volatility forecasts that penalises under-predicting risk more heavily.`;

  const reg = R.test_regimes;
  const regKeys = [[MKEY, cc.model], [BASE, cc.base], ["Historical average", cc.hist]];
  new Chart($("regimeChart"), {
    type: "bar",
    data: {
      labels: reg.map((r) => [r.regime, `n = ${r.n}`]),
      datasets: regKeys.map(([k, col]) => ({ label: k, data: reg.map((r) => r[k].RMSE), backgroundColor: col, borderRadius: 4, borderSkipped: "start" })),
    },
    options: {
      maintainAspectRatio: false, scales: { y: { beginAtZero: true, title: { display: true, text: "RMSE" } }, x: { grid: { display: false } } },
      plugins: { legend: { position: "bottom" }, tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${f2(c.raw)}` } } },
    },
  });
  const th = R.vix_regime_thresholds_from_train;
  $("regime-note").textContent = `Regimes use the VIX on the forecast date, with cut-offs (${f1(th[0])}, ${f1(th[1])}) set at the training-data terciles so that the test set does not define its own buckets.`;

  const pa = R.test_per_asset;
  table($("asset-metrics"), [{ t: "Asset" }, { t: "Model", num: true }, { t: "Recent vol", num: true }, { t: "Hist. avg", num: true }, { t: "EWMA", num: true }],
    ASSETS.map((a) => {
      const best = Math.min(pa[a][MKEY].RMSE, pa[a][BASE].RMSE, pa[a]["Historical average"].RMSE, pa[a]["EWMA vol (RiskMetrics)"].RMSE);
      const cell = (v) => (v === best ? `<b>${f2(v)}</b>` : f2(v));
      return [a, cell(pa[a][MKEY].RMSE), cell(pa[a][BASE].RMSE), cell(pa[a]["Historical average"].RMSE), cell(pa[a]["EWMA vol (RiskMetrics)"].RMSE)];
    }));
  const imp = R.permutation_importance_test.slice(0, 8);
  new Chart($("impChart"), {
    type: "bar",
    data: { labels: imp.map((d) => d.feature), datasets: [{ label: "RMSE increase when shuffled", data: imp.map((d) => d.importance), backgroundColor: cc.model, borderRadius: 4, borderSkipped: "start" }] },
    options: { indexAxis: "y", maintainAspectRatio: false, plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => `+${c.raw.toFixed(3)} RMSE (± ${imp[c.dataIndex].std.toFixed(3)})` } } },
      scales: { x: { title: { display: true, text: "Increase in test RMSE when the feature is shuffled" } }, y: { grid: { display: false } } } },
  });

  // ---------------------------------------------------------------- explorer
  const exAsset = $("ex-asset");
  exAsset.innerHTML = ASSETS.map((a) => `<option>${a}</option>`).join("");
  const exWin = $("ex-window");
  const dates0 = D.timeline[ASSETS[0]].dates;
  const quarters = [...new Set(dates0.map((d) => d.slice(0, 4) + "-Q" + (Math.floor((+d.slice(5, 7) - 1) / 3) + 1)))];
  const years = [...new Set(dates0.map((d) => d.slice(0, 4)))];
  exWin.innerHTML += years.map((y) => `<option value="Y${y}">${y}</option>`).join("") + quarters.map((qq) => `<option value="${qq}">${qq}</option>`).join("");
  const inWin = (d, w) => w === "all" ? true : w[0] === "Y" ? d.startsWith(w.slice(1)) : (d.slice(0, 4) + "-Q" + (Math.floor((+d.slice(5, 7) - 1) / 3) + 1)) === w;
  let exChart;
  function drawExplorer() {
    const c = C();
    const T = D.timeline[exAsset.value];
    const idx = T.dates.map((d, i) => (inWin(d, exWin.value) ? i : -1)).filter((i) => i >= 0);
    const pick = (arr) => idx.map((i) => arr[i]);
    const series = [
      { k: "actual", label: "Realised (next 20 days)", col: c.actual, w: 2.2, on: true },
      { k: "model", label: `${FAM} forecast`, col: c.model, w: 2, on: $("ex-model").checked },
      { k: "recent", label: "Recent 20-day vol", col: c.base, w: 1.5, on: $("ex-recent").checked },
      { k: "hist", label: "Historical average", col: c.hist, w: 1.5, on: $("ex-hist").checked, dash: [5, 4] },
      { k: "ewma", label: "EWMA", col: c.ewma, w: 1.5, on: $("ex-ewma").checked },
    ];
    const ds = series.filter((s) => s.on).map((s) => ({
      label: s.label, data: pick(T[s.k]), borderColor: s.col, backgroundColor: s.col, borderWidth: s.w, pointRadius: 0, pointHoverRadius: 4, borderDash: s.dash || [], tension: 0.15,
    }));
    if (exChart) exChart.destroy();
    exChart = new Chart($("exChart"), {
      type: "line", data: { labels: pick(T.dates), datasets: ds },
      options: {
        maintainAspectRatio: false, interaction: { mode: "index", intersect: false }, animation: false,
        scales: { y: { title: { display: true, text: "Annualised volatility (%)" } }, x: { ticks: { maxTicksLimit: 10 } } },
        plugins: { legend: { position: "bottom" }, tooltip: { callbacks: { label: (ctx) => `${ctx.dataset.label}: ${f2(ctx.raw)}%`, afterBody: (items) => `VIX: ${f2(T.vix[idx[items[0].dataIndex]])}` } } },
      },
    });
    const y = pick(T.actual);
    const rows = [["model", `${FAM} (final)`], ["recent", "Recent 20-day vol"], ["hist", "Historical average"], ["ewma", "EWMA"]].map(([k, lab]) => {
      const p = pick(T[k]);
      return { lab, r: rmse(y, p), a: mae(y, p) };
    });
    const best = Math.min(...rows.map((r) => r.r));
    table($("ex-table"), [{ t: `Forecast (${exAsset.value}, ${idx.length} days)` }, { t: "RMSE", num: true }, { t: "MAE", num: true }, { t: "vs model", num: true }],
      rows.map((r, i) => [r.lab + (r.r === best ? " <span class='tag'>best in window</span>" : ""), f2(r.r), f2(r.a),
        i === 0 ? "–" : `<span class="${rows[0].r < r.r ? "win" : "lose"}">${rows[0].r < r.r ? "model better" : "model worse"} by ${f2(Math.abs(rows[0].r - r.r))}</span>`]), 0);
  }
  ["ex-asset", "ex-window", "ex-model", "ex-recent", "ex-hist", "ex-ewma"].forEach((id) => $(id).addEventListener("change", drawExplorer));
  drawExplorer();

  // ---------------------------------------------------------------- portfolio
  const P = D.portfolio;
  $("universe-note").innerHTML = ASSETS.length < 4
    ? `<b>Asset universe in this build: ${ASSETS.join(", ")}.</b> These are two closely related equity ETFs, so diversification is limited. Running <code>python scripts/run_all.py --refresh-data</code> downloads bonds (TLT) and gold (GLD) as planned in the proposal and rebuilds this page.`
    : `Universe: ${ASSETS.join(", ")}. Mixing stocks, bonds and gold shows how low correlations reduce portfolio risk.`;
  const W = {};
  ASSETS.forEach((a) => (W[a] = 1 / ASSETS.length));
  $("weights").innerHTML = ASSETS.map((a) => `<div class="weight-row"><b>${a}</b><input type="range" min="0" max="100" step="5" id="w-${a}" aria-label="Weight of ${a}"><span class="pct" id="wp-${a}"></span></div>`).join("");
  $("presets").innerHTML = Object.keys(P.presets).map((k) => `<button data-p="${esc(k)}">${esc(k)}</button>`).join("");
  ASSETS.forEach((a) => $(`w-${a}`).addEventListener("input", () => { W[a] = +$(`w-${a}`).value / 100; renderPortfolio(); }));
  $("presets").addEventListener("click", (ev) => {
    const k = ev.target.dataset && ev.target.dataset.p; if (!k) return;
    ASSETS.forEach((a) => (W[a] = P.presets[k][a] || 0)); renderPortfolio();
  });
  let corrKey = "corr_train";
  $("corr-seg").addEventListener("click", (ev) => {
    if (!ev.target.dataset.k) return; corrKey = ev.target.dataset.k;
    [...$("corr-seg").children].forEach((bt) => bt.classList.toggle("on", bt.dataset.k === corrKey)); drawHeat();
  });
  function drawHeat() {
    const M = P[corrKey], n = ASSETS.length;
    const neg = css("--div-neg"), pos = css("--div-pos"), mid = css("--div-mid");
    const mix = (v) => `color-mix(in srgb, ${v >= 0 ? pos : neg} ${Math.round(Math.abs(v) * 85)}%, ${mid})`;
    const el = $("heat");
    el.style.gridTemplateColumns = `50px repeat(${n}, 1fr)`;
    let html = `<div></div>` + ASSETS.map((a) => `<div class="hdr">${a}</div>`).join("");
    for (let i = 0; i < n; i++) {
      html += `<div class="hdr">${ASSETS[i]}</div>`;
      for (let j = 0; j < n; j++) {
        const v = M[i][j];
        html += `<div class="cell" style="background:${mix(v)};color:${Math.abs(v) > 0.6 ? "#fff" : "inherit"}" title="${ASSETS[i]} vs ${ASSETS[j]}: ${v.toFixed(2)}">${v.toFixed(2)}</div>`;
      }
    }
    el.innerHTML = html;
  }
  drawHeat();

  function normW() {
    const s = ASSETS.reduce((a, k) => a + W[k], 0);
    const w = {}; ASSETS.forEach((k) => (w[k] = s > 0 ? W[k] / s : 1 / ASSETS.length)); return w;
  }
  function realised(w) {
    const n = P.return_dates.length; const port = new Array(n).fill(0);
    ASSETS.forEach((a) => { const r = P.returns_pct[a]; for (let i = 0; i < n; i++) port[i] += w[a] * (Math.exp(r[i] / 100) - 1); });
    let wealth = 1, peak = 1, mdd = 0;
    for (const x of port) { wealth *= 1 + x; peak = Math.max(peak, wealth); mdd = Math.min(mdd, wealth / peak - 1); }
    const mean = port.reduce((a, c) => a + c, 0) / n;
    const sd = Math.sqrt(port.reduce((a, c) => a + (c - mean) ** 2, 0) / (n - 1));
    const annRet = Math.pow(wealth, 252 / n) - 1, annVol = sd * Math.sqrt(252);
    return { port, annRet, annVol, sharpe: annRet / annVol, mdd };
  }
  let pfChart;
  function renderPortfolio() {
    const w = normW();
    ASSETS.forEach((a) => { $(`w-${a}`).value = Math.round(W[a] * 100); $(`wp-${a}`).textContent = (w[a] * 100).toFixed(0) + "%"; });
    const r = realised(w);
    $("pf-tiles").innerHTML = [
      ["Annual return", (r.annRet * 100).toFixed(1) + "%"], ["Annual volatility", (r.annVol * 100).toFixed(1) + "%"],
      ["Sharpe (rf = 0)", r.sharpe.toFixed(2)], ["Max drawdown", (r.mdd * 100).toFixed(1) + "%"],
    ].map(([l, v]) => `<div class="tile"><div class="label">${l}</div><div class="value" style="font-size:22px">${v}</div></div>`).join("");

    // forecast vs realised portfolio vol
    const rd = P.return_dates, pos = {}; rd.forEach((d, i) => (pos[d] = i));
    const fd = P.forecast_dates, H = 20;
    const corrAt = (k, a, b) => { if (a === b) return 1; const key = P.corr60[`${a}|${b}`] ? `${a}|${b}` : `${b}|${a}`; return P.corr60[key][k]; };
    const fc = (src, k) => {
      let s = 0;
      for (const a of ASSETS) for (const b of ASSETS) {
        const va = P[src][a][k], vb = P[src][b][k], c = corrAt(k, a, b);
        if (va == null || vb == null || c == null) return null;
        s += w[a] * w[b] * va * vb * c;
      }
      return Math.sqrt(s);
    };
    const labels = [], act = [], pm = [], pb = [];
    fd.forEach((d, k) => {
      const i = pos[d]; if (i == null || i + H >= rd.length) return;
      const win = r.port.slice(i + 1, i + 1 + H).map((x) => Math.log(1 + x));
      const mu = win.reduce((a, c) => a + c, 0) / H;
      const sd = Math.sqrt(win.reduce((a, c) => a + (c - mu) ** 2, 0) / (H - 1));
      labels.push(d); act.push(sd * Math.sqrt(252) * 100); pm.push(fc("pred_model", k)); pb.push(fc("pred_baseline", k));
    });
    const c = C();
    if (pfChart) pfChart.destroy();
    pfChart = new Chart($("pfChart"), {
      type: "line",
      data: { labels, datasets: [
        { label: "Realised portfolio vol (next 20 days)", data: act, borderColor: c.actual, backgroundColor: c.actual, borderWidth: 2.2, pointRadius: 0 },
        { label: `Forecast from ${FAM} asset forecasts`, data: pm, borderColor: c.model, backgroundColor: c.model, borderWidth: 2, pointRadius: 0 },
        { label: "Forecast from recent 20-day vols", data: pb, borderColor: c.base, backgroundColor: c.base, borderWidth: 1.5, pointRadius: 0 },
      ] },
      options: { maintainAspectRatio: false, animation: false, interaction: { mode: "index", intersect: false },
        scales: { y: { title: { display: true, text: "Annualised volatility (%)" } }, x: { ticks: { maxTicksLimit: 10 } } },
        plugins: { legend: { position: "bottom" }, tooltip: { callbacks: { label: (ctx) => `${ctx.dataset.label}: ${f2(ctx.raw)}%` } } } },
    });
    const rm = rmse(act, pm), rb = rmse(act, pb);
    table($("pf-err"), [{ t: "Portfolio vol forecast built from" }, { t: "RMSE", num: true }, { t: "MAE", num: true }],
      [[`${FAM} asset forecasts`, f2(rm), f2(mae(act, pm))], ["Recent 20-day vols", f2(rb), f2(mae(act, pb))]], rm <= rb ? 0 : 1);
  }
  renderPortfolio();

  // ---------------------------------------------------------------- errors & limits
  const CONTEXT = {
    "2024-08": "Yen carry-trade unwind: the VIX briefly spiked (closing at 38.6 on 5 Aug) and the shock faded within days, so a crisis-like forecast overshot.",
    "2024-07": "Mid-July rotation out of mega-cap tech: volatility rose sharply from very low levels, and no feature on the day gave warning.",
    "2024-09": "Early-September growth scare: volatility was elevated on the day but calmed quickly afterwards.",
    "2023-03": "US regional bank failures (SVB): a sudden stress event.",
    "2023-10": "Rate-driven sell-off in October 2023.",
  };
  table($("err-table"), [{ t: "Date" }, { t: "Asset" }, { t: "Realised", num: true }, { t: "Model", num: true }, { t: "Recent vol", num: true }, { t: "VIX", num: true }, { t: "What happened" }],
    R.representative_errors.map((r) => [r.date, r.asset, f1(r.actual), `<span class="${r.error > 0 ? "lose" : "win"}">${f1(r.predicted)}</span>`, f1(r.baseline), f1(r.vix),
      (r.error > 0 ? "Over-predicted. " : "Under-predicted. ") + (CONTEXT[r.date.slice(0, 7)] || "")]));
  $("limit-data").innerHTML = q.source.includes("legacy")
    ? "<b>Data in this build.</b> This build uses an older SPY/QQQ file with unadjusted closes, so dividend days show small artificial returns. Refreshing the data uses dividend-adjusted closes and adds bonds and gold."
    : "<b>Survivorship and adjustments.</b> These are large, long-lived ETFs, and prices are dividend-adjusted by Yahoo Finance. We did not cross-check them against a second data vendor.";

  // ---------------------------------------------------------------- reproducibility
  const rp = R.reproducibility;
  table($("repro-table"), ["Item", "Value"], [
    ["Random seed", rp.seed], ["Python / scikit-learn / pandas / numpy", `${rp.python} / ${rp.sklearn} / ${rp.pandas} / ${rp.numpy}`],
    ["Full pipeline runtime", rp.runtime_seconds + " s"], ["Final model", `${FAM} ${JSON.stringify(R.final_model.params)}, trained through ${R.final_model.trained_through}`],
    ["Saved model", "<code>artifacts/final_model.joblib</code> (scaler + model in one pipeline)"],
    ...(R.training_proof ? [
      ["Is training really happening?", `Yes: ${R.training_proof.total_model_fits} model fits in total (CV screening + validation tuning + final refit). The final fit on ${R.training_proof.training_rows_final_fit.toLocaleString()} rows &times; ${R.training_proof.n_features} features takes ${(R.training_proof.final_fit_seconds * 1000).toFixed(1)} ms because Ridge is a single closed-form linear solve.`],
      ["Shuffled-label control", `Same model trained on randomly shuffled targets: test RMSE ${f2(R.training_proof.shuffled_target_test_RMSE)} vs ${f2(R.training_proof.real_target_test_RMSE)} with the real targets, so the model learns real signal from the features.`],
    ] : []),
    ["Metrics", "<code>artifacts/evaluation_metrics.json</code>, test predictions in <code>artifacts/test_predictions.csv</code>"],
  ]);
  $("gen-date").textContent = "Data generated " + D.generated;

  // nav highlight
  const links = [...document.querySelectorAll(".topnav a")];
  const obs = new IntersectionObserver((ents) => ents.forEach((en) => {
    if (en.isIntersecting) links.forEach((l) => l.classList.toggle("active", l.getAttribute("href") === "#" + en.target.id));
  }), { rootMargin: "-40% 0px -55% 0px" });
  document.querySelectorAll("main section").forEach((s) => obs.observe(s));
})();
