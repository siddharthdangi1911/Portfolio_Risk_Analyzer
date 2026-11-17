import type { Metadata } from "next";
import { HBars } from "@/components/charts/common";
import { FoldChart, TuningCurve } from "@/components/charts/ModelCharts";
import { Card, PageHead } from "@/components/ui";
import { CAT, forecastColor } from "@/lib/colors";
import { EWMA, FAM, FAMILIES, paramText, R, shortName } from "@/lib/data";
import { f1, f2, f3 } from "@/lib/stats";

export const metadata: Metadata = { title: "Model selection" };

const DESC: Record<string, string> = {
  Ridge: "Linear, L2-regularised",
  "Decision tree": "Single tree of if/then rules",
  "Random forest": "Average of many trees",
  "SVR (RBF)": "Kernel support vector regression",
  "Gradient boosting": "scikit-learn boosted trees",
  LightGBM: "Leaf-wise boosted trees",
};

export default function ModelsPage() {
  const S = R.stage1_screening;
  const rows = [...S.results].sort((a, b) => a.cv_rmse_mean - b.cv_rmse_mean);
  const famColor = (n: string) => CAT[FAMILIES.indexOf(n) % CAT.length];
  const sel = S.results.find((r) => r.name === S.selected_family)!;
  const others = S.results.filter((r) => r.kind === "model" && r.name !== sel.name);
  const winsAll = sel.fold_rmse.every((v, i) => others.every((o) => v <= o.fold_rmse[i]));
  const T = R.stage2_tuning;
  const trials = T.all_trials ?? T.top_trials;
  const vc = R.validation_comparison;
  const vkeys = Object.keys(vc).sort((a, b) => vc[a].RMSE - vc[b].RMSE);
  const valBest = vkeys[0];
  const vm = vc[`${FAM} (tuned)`];
  const ewmaRow = S.results.find((r) => r.name === EWMA);

  return (
    <>
      <PageHead eyebrow="Baselines, methods and evaluation design" title="Model selection">
        {FAMILIES.length} models and 3 baselines are compared with {S.folds.length}-fold expanding-window cross-validation on the training years only. Only the winner is tuned and tested.
      </PageHead>

      <div className="grid g2">
        <Card title="Stage 1 · Cross-validation error" sub="Mean RMSE across folds · whiskers show ±1 standard deviation">
          <HBars height="lg"
            rows={rows.map((r) => ({ label: r.kind === "baseline" ? `${shortName(r.name)} (baseline)` : r.name, value: r.cv_rmse_mean, lo: r.cv_rmse_mean - r.cv_rmse_std, hi: r.cv_rmse_mean + r.cv_rmse_std, color: r.kind === "baseline" ? "var(--muted-fill)" : famColor(r.name) }))}
            xLabel="CV RMSE (vol points)" />
        </Card>
        <Card title="Leaderboard">
          <div className="table-wrap">
            <table>
              <thead><tr><th>#</th><th>Model</th><th className="num">CV RMSE</th><th className="num">Time</th></tr></thead>
              <tbody>
                {rows.map((r, i) => (
                  <tr key={r.name} className={r.name === sel.name ? "hl" : ""}>
                    <td className="muted">{i + 1}</td>
                    <td>
                      <b style={{ fontWeight: 600 }}>{shortName(r.name)}</b>
                      {r.kind === "baseline" && <span className="tag">baseline</span>}
                      {r.name === sel.name && <span className="tag accent">selected</span>}
                      <div className="small muted">{r.kind === "model" ? `${DESC[r.name] ?? ""} · ${paramText(r.best_params)}` : "No learning"}</div>
                    </td>
                    <td className="num">{f2(r.cv_rmse_mean)} <span className="small muted">±{f1(r.cv_rmse_std)}</span></td>
                    <td className="num muted">{r.kind === "model" ? `${r.seconds}s` : "–"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>

      <Card title="Error in each fold" sub={`${sel.name} ${winsAll ? "is best in every fold" : "has the lowest mean"} · fold 5 holds the 2020 COVID shock · click a chip to hide a model`} className="mb">
        <FoldChart
          folds={S.folds.map((f) => `Fold ${f.fold} · ${f.score_start.slice(0, 4)}–${f.score_end.slice(2, 4)}`)}
          series={[
            ...FAMILIES.map((f) => ({ name: f, color: famColor(f), data: S.results.find((r) => r.name === f)!.fold_rmse, selected: f === sel.name })),
            ...(ewmaRow ? [{ name: "EWMA baseline", color: "var(--text-3)", data: ewmaRow.fold_rmse, dashed: true }] : []),
          ]}
        />
      </Card>

      <div className="grid g2">
        <Card title={`Stage 2 · Tuning ${T.family} on validation`} sub={`${T.n_configs} settings fit on train, scored on 2021–22 · best: ${paramText(T.best_params)}`}>
          {T.family === "Ridge" ? <TuningCurve trials={trials} /> : null}
          <div className="table-wrap" style={{ marginTop: 8 }}>
            <table>
              <thead><tr><th>Rank</th><th>Setting</th><th className="num">Val RMSE</th></tr></thead>
              <tbody>
                {trials.slice(0, 4).map((t, i) => (
                  <tr key={i} className={i === 0 ? "hl" : ""}><td className="muted">{i + 1}</td><td>{paramText(t.params)}</td><td className="num">{f3(t.val_rmse)}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
        <Card title="Validation 2021–22 vs baselines" sub="All fit on training years only">
          <HBars rows={vkeys.map((k) => ({ label: shortName(k), value: vc[k].RMSE, color: forecastColor(k, FAM), note: `bias ${f2(vc[k].Bias)}` }))} xLabel="RMSE" />
          {valBest !== `${FAM} (tuned)` && (
            <p className="note"><b>{shortName(valBest)}</b> beat {FAM} here ({f2(vc[valBest].RMSE)} vs {f2(vm.RMSE)}). {FAM} under-predicted 2022 (bias {f2(vm.Bias)}) because volatility stayed high all year.</p>
          )}
        </Card>
      </div>
    </>
  );
}
