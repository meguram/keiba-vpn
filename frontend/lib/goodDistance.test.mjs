import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import ts from "typescript";

const src = readFileSync(new URL("./goodDistance.ts", import.meta.url), "utf8");
const out = ts.transpileModule(src, { compilerOptions: { module: ts.ModuleKind.ES2020, target: ts.ScriptTarget.ES2020 } }).outputText;
const dir = mkdtempSync(join(tmpdir(), "gooddistance-"));
const file = join(dir, "goodDistance.mjs");
writeFileSync(file, out);
const m = await import(file);

test("距離は 3 桁区切りの m（1,600m）で、範囲は 〜 でつなぐ", () => {
  assert.equal(m.goodDistanceText({ value: null, short_m: 1400, long_m: 1800, runs: 5 }), "1,400m〜1,800m（過去の走 5 走）");
  assert.equal(m.goodDistanceText({ value: null, short_m: 1600, long_m: 1600, runs: 3 }), "1,600m（過去の走 3 走）");
  assert.equal(m.goodDistanceText({ value: null, short_m: 800, long_m: 1200, runs: 4 }), "800m〜1,200m（過去の走 4 走）");
});

test("3 走未満・走なしは不明（走数は出す）", () => {
  assert.equal(m.goodDistanceText({ value: null, short_m: null, long_m: null, runs: 2 }), "不明（過去の走 2 走。3 走以上で判定）");
  assert.equal(m.goodDistanceText({ value: null, short_m: null, long_m: null }), "不明（過去の走 0 走。3 走以上で判定）");
});

test("fmtM は 10m 単位に丸める", () => {
  assert.equal(m.fmtM(2400), "2,400m");
  assert.equal(m.fmtM(1604), "1,600m");
});
