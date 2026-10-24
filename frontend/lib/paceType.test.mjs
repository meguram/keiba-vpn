// 実行: node --test lib/paceType.test.mjs
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync, writeFileSync, mkdtempSync } from "node:fs";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const ts = createRequire(import.meta.url)("typescript");
const opts = { compilerOptions: { module: ts.ModuleKind.ES2020, target: ts.ScriptTarget.ES2020 } };
const dir = mkdtempSync(join(tmpdir(), "pt-"));
writeFileSync(join(dir, "pt.mjs"), ts.transpileModule(readFileSync(join(here, "paceType.ts"), "utf8"), opts).outputText);
const m = await import(pathToFileURL(join(dir, "pt.mjs")).href);

test("記号と語: ±0.15 で有利/不利、語は ±0.4 で強まる", () => {
  assert.equal(m.fitMark(0.2), "▲");
  assert.equal(m.fitMark(-0.2), "▽");
  assert.equal(m.fitMark(0.1), "－");
  assert.equal(m.fitMark(null), "—");
  assert.equal(m.fitWord(0.5), "有利");
  assert.equal(m.fitWord(0.2), "やや有利");
  assert.equal(m.fitWord(0), "互角");
  assert.equal(m.fitWord(-0.2), "やや不利");
  assert.equal(m.fitWord(-0.6), "不利");
  assert.equal(m.overallWord(0.15), "向く");
  assert.equal(m.overallWord(-0.15), "不向き");
  assert.equal(m.overallWord(0.1), "ふつう");
  assert.equal(m.overallWord(undefined), null);
});

const horse = {
  horse_number: 3,
  by_scenario: {
    high_compact: { fit_score: -0.6 }, high_normal: { fit_score: -0.3 }, high_stretched: { fit_score: 0.0 },
    mid_compact: { fit_score: 0.2 }, mid_normal: { fit_score: 0.2 }, mid_stretched: { fit_score: 0.2 },
    slow_compact: { fit_score: 0.5 }, slow_normal: { fit_score: 0.5 }, slow_stretched: { fit_score: 0.5 },
  },
};

test("ペースごとの fit は 3 隊列を隊列確率で重みづけ（合計が 1 でなくても再正規化）", () => {
  const f = m.paceFitScore(horse, "high", { compact: 2, normal: 0, stretched: 2 });
  assert.ok(Math.abs(f - -0.3) < 1e-9);
  assert.ok(Math.abs(m.paceFitScore(horse, "slow", { compact: 0.2, normal: 0.3, stretched: 0.5 }) - 0.5) < 1e-9);
  assert.equal(m.paceFitScore({ horse_number: 1, by_scenario: {} }, "mid", { compact: 1 }), null);
});

test("総合は 9 シナリオの確率で重みづけ。確率が無ければ null", () => {
  const scen = Object.keys(horse.by_scenario).map((key) => ({ key, prob: key.startsWith("slow") ? 0.2 : key.startsWith("mid") ? 0.6 : 0.2 }));
  const o = m.overallFitScore(horse, scen);
  const want = (0.2 * (-0.6 - 0.3 + 0) + 0.6 * 0.6 + 0.2 * 1.5) / 3.0;      // 9 個の確率の合計は 3.0（0.6+1.8+0.6）。合計で割って平均にする
  assert.ok(Math.abs(o - want) < 1e-9);
  assert.equal(m.overallFitScore(horse, []), null);
});

test("buildFitMap: 馬番で引け、ペース別・総合・最も合う展開・9マスの格子が入る", () => {
  const sf = { scenarios: Object.keys(horse.by_scenario).map((key) => ({ key, label: key, prob: 1 / 9 })), horses: [{ ...horse, sensitivity_label: "展開次第", best_scenario_label: "スロー・中間" }] };
  const pt = { key: "mid_normal", label: "ミドル・中間", probs: {}, pace_probs: { high: 0.2, mid: 0.6, slow: 0.2 }, shape_probs: { compact: 0.3, normal: 0.3, stretched: 0.4 }, style_fit: {} };
  const map = m.buildFitMap(sf, pt);
  const h = map.get(3);
  assert.equal(h.sensitivity, "展開次第");
  assert.equal(h.bestLabel, "スロー・中間");
  assert.equal(h.grid.length, 9);
  assert.equal(m.fitMark(h.paces.high), "▽");
  assert.equal(m.fitMark(h.paces.slow), "▲");
  assert.equal(m.buildFitMap(null, pt), null);
  assert.equal(m.buildFitMap({ scenarios: [], horses: [] }, pt), null);
});

test("積み上げバー: 合計 100 に近い整数、中央ゼロの幅、根拠の札", () => {
  const seg = m.segments({ high: 0.241, mid: 0.567, slow: 0.193 }, ["high", "mid", "slow"]);
  assert.deepEqual(seg.map((s) => s.pct), [24, 57, 19]);
  assert.deepEqual(m.diverging(0.32), { side: "plus", widthPct: 16 });
  assert.deepEqual(m.diverging(-1.4), { side: "minus", widthPct: 50 });
  const b = m.basisTags({ strong_front_runners: 1, front_runners_claim: 2, lap_forecast_used: true });
  assert.deepEqual(b.tags, ["強い逃げ馬 1頭", "ハナ主張 2頭", "ラップ予想を反映"]);
  assert.ok(b.warn.includes("ハイペース"));
  assert.equal(m.basisTags({ front_runners_claim: 1 }).warn, null);
});
