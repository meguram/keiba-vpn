import test from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { execFileSync } from "node:child_process";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import ts from "typescript";
import { readFileSync } from "node:fs";

const src = readFileSync(new URL("./growth.ts", import.meta.url), "utf8");
const out = ts.transpileModule(src, { compilerOptions: { module: ts.ModuleKind.ES2020, target: ts.ScriptTarget.ES2020 } }).outputText;
const dir = mkdtempSync(join(tmpdir(), "growth-"));
const file = join(dir, "growth.mjs");
writeFileSync(file, out);
const m = await import(file);

const META = { level_table: [0, 1, 2, 3, 4].map((i) => ({ level: i, share_resid_up: [0.357, 0.435, 0.508, 0.582, 0.677][i] })) };

test("growthView は 5 段階に記号と言葉を返し、範囲外は丸める", () => {
  assert.equal(m.growthView(0).sym, "▼");
  assert.equal(m.growthView(4).word, "大きく成長");
  assert.equal(m.growthView(9).word, "大きく成長");
  assert.equal(m.growthView(null).word, "横ばい");
  assert.equal(new Set(m.GROWTH_VIEW.map((v) => v.sym)).size, 5);
});

test("indexLine は符号と秒（1点=0.1秒）", () => {
  assert.equal(m.indexLine({ index: 1.7 }), "期待より +1.7点（≒ 0.17秒速い）");
  assert.equal(m.indexLine({ index: -2.04 }), "期待より −2.0点（≒ 0.20秒遅い）");
  assert.equal(m.indexLine({ index: 0 }), "期待より ±0点（同じ）");
});

test("levelSummary は段階の実績と全段階の幅", () => {
  assert.equal(m.levelSummary(META, 3), "この段階の馬は、期待より指数が上がった割合 58%（全段階: 下降 36% → 大きく成長 68%）");
  assert.equal(m.levelSummary(null, 3), null);
  assert.equal(m.levelSummary({ level_table: [] }, 3), null);
});

test("contributionRows は寄与の大きい順で ctrl を出さない", () => {
  const rows = m.contributionRows({ contributions: { stage: -2.1, weight: 0.1, oikiri: 1.8, ped: -0.3, ctrl: 5 } });
  assert.deepEqual(rows.map((r) => r.key), ["stage", "oikiri", "ped", "weight"]);
  assert.equal(m.fmtPoints(-0.04), "±0");
});
