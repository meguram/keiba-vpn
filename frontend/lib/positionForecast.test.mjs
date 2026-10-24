// 実行: node --test lib/positionForecast.test.mjs
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync, writeFileSync, mkdtempSync } from "node:fs";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const ts = createRequire(import.meta.url)("typescript");
const out = join(mkdtempSync(join(tmpdir(), "posfc-")), "positionForecast.mjs");
writeFileSync(out, ts.transpileModule(readFileSync(join(here, "positionForecast.ts"), "utf8"), { compilerOptions: { module: ts.ModuleKind.ES2020, target: ts.ScriptTarget.ES2020 } }).outputText);
const m = await import(pathToFileURL(out).href);

test("ハイペースは差し・追込が向き、逃げ・先行が不向き", () => {
  assert.equal(m.paceFit("closer", "high", null).label, "向く");
  assert.equal(m.paceFit("mid", "high", null).label, "向く");
  assert.equal(m.paceFit("front", "high", null).label, "不向き");
  assert.equal(m.paceFit("stalker", "high", null).label, "不向き");
});

test("スローは逃げ・先行が向き、差し・追込が不向き", () => {
  assert.equal(m.paceFit("front", "slow", null).label, "向く");
  assert.equal(m.paceFit("closer", "slow", null).label, "不向き");
});

test("ミドル・不明は脚質だけでは判定せず、追走難度で動く", () => {
  assert.equal(m.paceFit("front", "mid", null).label, "ふつう");
  assert.equal(m.paceFit("front", "unknown", 70).label, "向く");
  assert.equal(m.paceFit("front", "mid", 20).label, "不向き");
  assert.equal(m.paceFit("front", "mid", 45).label, "ふつう");
});

test("脚質×ペースと追走難度が打ち消し合うとふつう", () => {
  assert.equal(m.paceFit("front", "high", 70).label, "ふつう");
  assert.equal(m.paceFit("closer", "high", 20).label, "ふつう");
  assert.equal(m.paceFit("closer", "high", 70).score, 2);
});

test("脚質が無いと判定しない", () => {
  assert.equal(m.paceFit(undefined, "high", 70), null);
});

test("追走のしやすさの境目は API のラベルと同じ", () => {
  assert.equal(m.easeTone(75).word, "非常に楽");
  assert.equal(m.easeTone(60).word, "楽");
  assert.equal(m.easeTone(45).word, "普通");
  assert.equal(m.easeTone(30).word, "やや困難");
  assert.equal(m.easeTone(29.9).word, "困難");
  assert.equal(m.easeTone(null).word, "—");
});

test("buildTdMap は position_flow と entries を馬番で合わせる", () => {
  const mp = m.buildTdMap(
    [{ horse_number: 2, style: "差し", flow_pattern: "粘り込み", stamina_concern: true, stamina: { stamina_label: "激消耗" }, expected_last_3f: { seconds: 34.7, delta_sec: -0.2 } }],
    [{ horse_number: 2, tracking_difficulty: { ease_pct: 30, label: "やや困難" } }, { horse_number: 5, tracking_difficulty: { ease_pct: 80 } }, { tracking_difficulty: { ease_pct: 1 } }],
  );
  assert.equal(mp.size, 2);
  assert.deepEqual(mp.get(2), { ease_pct: 30, ease_label: "やや困難", style: "差し", flow_pattern: "粘り込み", stamina_label: "激消耗", stamina_concern: true, last3f_sec: 34.7, last3f_delta: -0.2 });
  assert.equal(mp.get(5).ease_pct, 80);
  assert.equal(m.buildTdMap(undefined, undefined).size, 0);
});
