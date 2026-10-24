// 実行: node --test lib/lapForecast.test.mjs
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync, writeFileSync, mkdtempSync } from "node:fs";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const ts = createRequire(import.meta.url)("typescript");
const out = join(mkdtempSync(join(tmpdir(), "lapfc-")), "lapForecast.mjs");
writeFileSync(out, ts.transpileModule(readFileSync(join(here, "lapForecast.ts"), "utf8"), { compilerOptions: { module: ts.ModuleKind.ES2020, target: ts.ScriptTarget.ES2020 } }).outputText);
const m = await import(pathToFileURL(out).href);

const x = (i) => i * 10;
const y = (v) => v * 2;

test("buildPath は null で線を切り、孤立点は点として残す", () => {
  assert.equal(m.buildPath([1, 2, null, 3, null, 4, 5], x, y), "M0.0 2.0L10.0 4.0M30.0 6.0L30.0 6.0M50.0 8.0L60.0 10.0");
  assert.equal(m.buildPath([null, null], x, y), "");
  assert.equal(m.buildPath([], x, y), "");
});

test("buildBand は上下とも入っている連続区間だけを塗る", () => {
  const d = m.buildBand([1, 1, null, 1, 1], [2, 2, 2, 2, 2], x, y);
  assert.equal((d.match(/M/g) ?? []).length, 2);
  assert.ok(d.endsWith("Z"));
  assert.equal(m.buildBand([1, null], [2, 2], x, y), "");
  assert.equal(m.buildBand([1, 1], [null, 2], x, y), "");
});

test("yRange は null を無視し、全部 null なら null、幅が狭くても最小幅を確保", () => {
  assert.equal(m.yRange([[null, null], []]), null);
  const r = m.yRange([[12.0, null, 12.2], [11.9]]);
  assert.ok(r.min < 11.9 && r.max > 12.2);
  const flat = m.yRange([[12.0]]);
  assert.ok(flat.max - flat.min >= 0.6);
});

test("niceTicks は本数が maxTicks 以内で、範囲内の値を返す", () => {
  const t = m.niceTicks(11.4, 13.1, 5);
  assert.ok(t.length >= 2 && t.length <= 5);
  assert.ok(t.every((v) => v >= 11.4 - 1e-9 && v <= 13.1 + 1e-9));
  const wide = m.niceTicks(10, 40, 5);
  assert.ok(wide.length <= 5);
});

test("cumulativeMeters と tickIndices", () => {
  assert.deepEqual(m.cumulativeMeters([160, 200, 200]), [160, 360, 560]);
  assert.deepEqual(m.tickIndices(3, 5), [0, 1, 2]);
  const t = m.tickIndices(12, 5);
  assert.equal(t[t.length - 1], 11);
  assert.ok(t.length <= 5);
  assert.deepEqual(m.tickIndices(0, 5), []);
});

test("lapSummaryText は値のある項目だけを並べる", () => {
  assert.equal(m.lapSummaryText({ n_laps: 5, lap_meters: [], expected_laps: [12], expected_first3f: 35.7, expected_last3f: 34.5, pace_label: "ミドル", pace_prob: 0.5 }),
    "前半3F 35.7秒 ・ 上がり3F 34.5秒 ・ 傾向 ミドル 50%");
  assert.equal(m.lapSummaryText({ n_laps: 5, lap_meters: [], expected_laps: [12], expected_first3f: null, expected_last3f: 34.5 }), "上がり3F 34.5秒");
});

test("isUsableLapForecast は区間数と期待ラップが揃うものだけ true", () => {
  assert.equal(m.isUsableLapForecast(null), false);
  assert.equal(m.isUsableLapForecast({ n_laps: 0, lap_meters: [], expected_laps: [] }), false);
  assert.equal(m.isUsableLapForecast({ n_laps: 2, lap_meters: [200, 200], expected_laps: [null, null] }), false);
  assert.equal(m.isUsableLapForecast({ n_laps: 2, lap_meters: [200, 200], expected_laps: [12.1, null] }), true);
});

test("scenarioColor は名前の語で色を決める", () => {
  assert.notEqual(m.scenarioColor("ハイ", 0), m.scenarioColor("スロー", 0));
  assert.equal(m.scenarioColor("???", 1), m.scenarioColor("ミドル", 9));
});
