// 実行: node --test lib/growthCurve.test.mjs
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync, writeFileSync, mkdtempSync } from "node:fs";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const ts = createRequire(import.meta.url)("typescript");
const out = join(mkdtempSync(join(tmpdir(), "growth-")), "growthCurve.mjs");
writeFileSync(out, ts.transpileModule(readFileSync(join(here, "growthCurve.ts"), "utf8"), { compilerOptions: { module: ts.ModuleKind.ES2020, target: ts.ScriptTarget.ES2020 } }).outputText);
const m = await import(pathToFileURL(out).href);

test("weightAxis は min/max に余白を付けて刻みに丸め、体重が無ければ null", () => {
  assert.equal(m.weightAxis([]), null);
  const a = m.weightAxis([{ weight: 470 }, { weight: 492 }]);
  assert.ok(a.lo < 470 && a.hi > 492 && a.lo % 5 === 0 && a.hi % 5 === 0);
  assert.deepEqual(a.ticks.slice(0, 2), [a.lo, a.lo + (a.ticks[1] - a.ticks[0])]);
  const one = m.weightAxis([{ weight: 480 }]);
  assert.ok(one.lo < 480 && one.hi > 480);
  const wide = m.weightAxis([{ weight: 400 }, { weight: 500 }]);
  assert.equal(wide.ticks[1] - wide.ticks[0], 20);
});

test("weightSegments は体重の無い走（breakDates）が間にあるときだけ線を切る", () => {
  const w = [{ date: "20240101" }, { date: "20240301" }, { date: "20240601" }];
  assert.equal(m.weightSegments(w, []).length, 1);
  const segs = m.weightSegments(w, ["20240401"]);
  assert.deepEqual(segs.map((s) => s.length), [2, 1]);
  assert.equal(m.weightSegments(w, ["20231201", "20240701"]).length, 1);
  assert.deepEqual(m.weightSegments([], ["20240101"]), []);
});

test("fmtWeightChange は符号付き（0 は ±0、読めなければ空）", () => {
  assert.deepEqual([4, -2, 0, null, undefined, NaN].map(m.fmtWeightChange), ["+4", "−2", "±0", "", "", ""]);
});

test("pathOf は 2 点未満なら空、2 点以上で M/L", () => {
  assert.equal(m.pathOf([{ x: 1, y: 2 }]), "");
  assert.equal(m.pathOf([{ x: 1, y: 2 }, { x: 3, y: 4 }]), "M1.0,2.0 L3.0,4.0");
});

test("weightBarKind は前走比で増=up・減=down、0・不明=flat", () => {
  assert.deepEqual([6, -4, 0, 0.3, -0.4, null, undefined, NaN].map(m.weightBarKind), ["up", "down", "flat", "flat", "flat", "flat", "flat", "flat"]);
});

test("barWidth は最小間隔の62%を3〜12pxに収め、1本だけなら10px", () => {
  assert.equal(m.barWidth([50]), 10);
  assert.equal(m.barWidth([10, 12]), 3);
  assert.equal(m.barWidth([300, 10, 100]), 12);
  assert.ok(Math.abs(m.barWidth([0, 10, 30]) - 6.2) < 1e-9);
});
