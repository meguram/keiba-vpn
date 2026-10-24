// 実行: node --test lib/probCalibration.test.mjs
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync, writeFileSync, mkdtempSync } from "node:fs";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const ts = createRequire(import.meta.url)("typescript");
const out = join(mkdtempSync(join(tmpdir(), "probcal-")), "probCalibration.mjs");
writeFileSync(out, ts.transpileModule(readFileSync(join(here, "probCalibration.ts"), "utf8"), { compilerOptions: { module: ts.ModuleKind.ES2020, target: ts.ScriptTarget.ES2020 } }).outputText);
const m = await import(pathToFileURL(out).href);

test("calState: 無効 / 適用 / 適用なし", () => {
  assert.equal(m.calState(undefined), "off");
  assert.equal(m.calState(null), "off");
  assert.equal(m.calState({ applied: true }), "applied");
  assert.equal(m.calState({ applied: false, reason: "オッズ未確定" }), "skipped");
});

test("modelRows: 元のモデルの確率がある馬だけを馬番順に", () => {
  const rows = m.modelRows([
    { horse_number: 3, win_prob: 0.1, win_prob_model: 0.05 },
    { horse_number: 1, win_prob: 0.3, win_prob_model: 0.4 },
    { horse_number: 2, win_prob: 0.2 },
  ]);
  assert.deepEqual(rows.map((r) => r.horse_number), [1, 3]);
  assert.deepEqual(m.modelRows(undefined), []);
});

test("oddsPhaseLabel", () => {
  assert.equal(m.oddsPhaseLabel("t15"), "発走15分前");
  assert.equal(m.oddsPhaseLabel(null), "");
  assert.equal(m.oddsPhaseLabel("x9"), "x9");
});

test("説明文にオッズ前は補正なしと書いてある", () => {
  assert.match(m.PROB_CAL_EXPLAIN, /市場補正なし/);
  assert.match(m.PROB_CAL_LABEL, /市場補正後/);
});
