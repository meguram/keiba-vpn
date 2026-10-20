// 実行: node --test lib/bracket.test.mjs
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync, writeFileSync, mkdtempSync } from "node:fs";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const ts = createRequire(import.meta.url)("typescript");
const out = join(mkdtempSync(join(tmpdir(), "bracket-")), "bracket.mjs");
writeFileSync(out, ts.transpileModule(readFileSync(join(here, "bracket.ts"), "utf8"), { compilerOptions: { module: ts.ModuleKind.ES2020, target: ts.ScriptTarget.ES2020 } }).outputText);
const m = await import(pathToFileURL(out).href);

const all = (n) => Array.from({ length: n }, (_, i) => m.bracketOf(i + 1, n));

test("8頭以下は 1 頭 1 枠", () => {
  assert.deepEqual(all(8), [1, 2, 3, 4, 5, 6, 7, 8]);
  assert.deepEqual(all(5), [1, 2, 3, 4, 5]);
});

test("9〜15頭は後ろの枠から 2 頭ずつ", () => {
  assert.deepEqual(all(9), [1, 2, 3, 4, 5, 6, 7, 8, 8]);
  assert.deepEqual(all(12), [1, 2, 3, 4, 5, 5, 6, 6, 7, 7, 8, 8]);
  assert.deepEqual(all(15), [1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8]);
});

test("16頭は全枠 2 頭", () => {
  assert.deepEqual(all(16), [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8]);
});

test("17頭は 8 枠が 3 頭、18頭は 7・8 枠が 3 頭", () => {
  assert.deepEqual(all(17), [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 8]);
  assert.deepEqual(all(18), [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 7, 8, 8, 8]);
});

test("頭数ごとの枠内頭数の合計は頭数に一致する", () => {
  for (let n = 1; n <= 18; n++) assert.equal(m.bracketCounts(n).reduce((a, b) => a + b, 0), n);
});

test("API の枠番が優先される。範囲外・馬番不正は null", () => {
  assert.equal(m.bracketOf(1, 18, 5), 5);
  assert.equal(m.bracketOf(6, 12, null), 5);
  assert.equal(m.bracketOf(0, 12), null);
  assert.equal(m.bracketOf(13, 12), null);
  assert.equal(m.bracketOf(3, 0), null);
});

test("枠色と文字色: netkeiba 配色（白・黄・橙・桃は黒文字、黒・赤・青・緑は白文字）", () => {
  const dark = "#111827", white = "#ffffff";
  const want = { 1: dark, 2: white, 3: white, 4: white, 5: dark, 6: white, 7: dark, 8: dark };
  for (const [bn, c] of Object.entries(want)) assert.equal(m.BRACKET[bn].color, c, `${bn}枠の文字色`);
  assert.equal(m.BRACKET[1].bg, "#ffffff");
  assert.equal(m.BRACKET[2].bg, "#111111");
  assert.equal(Object.keys(m.BRACKET).length, 8);
});

test("枠番が無いときは中立色", () => {
  assert.equal(m.bracketColor(null), m.BRACKET_FALLBACK);
  assert.equal(m.bracketColor(9), m.BRACKET_FALLBACK);
  assert.equal(m.bracketChipStyle(3).background, "#e60012");
});
