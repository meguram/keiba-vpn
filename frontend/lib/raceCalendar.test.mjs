// 実行: node --test lib/raceCalendar.test.mjs
import test from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { readFileSync, writeFileSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const ts = createRequire(import.meta.url)("typescript");
const out = join(mkdtempSync(join(tmpdir(), "raceCalendar-")), "raceCalendar.mjs");
writeFileSync(out, ts.transpileModule(readFileSync(join(here, "raceCalendar.ts"), "utf8"), { compilerOptions: { module: ts.ModuleKind.ES2020, target: ts.ScriptTarget.ES2020 } }).outputText);
const m = await import(pathToFileURL(out).href);

test("shiftMonth は年をまたぐ", () => {
  assert.equal(m.shiftMonth("202601", -1), "202512");
  assert.equal(m.shiftMonth("202512", 1), "202601");
});

test("clampMonth は範囲に収める", () => {
  assert.equal(m.clampMonth("201912", "202001", "202610"), "202001");
  assert.equal(m.clampMonth("202701", "202001", "202610"), "202610");
});

test("monthGrid は日曜始まりで 7 列", () => {
  const g = m.monthGrid("202610");               // 2026-10-01 は木曜
  assert.ok(g.every((w) => w.length === 7));
  assert.equal(g[0][4], "20261001");
  assert.equal(g[0][3], null);
  assert.equal(g.flat().filter(Boolean).length, 31);
});

test("latestOnOrBefore と resolveDate は未来を避ける", () => {
  assert.equal(m.latestOnOrBefore(["20261024", "20261017", "20261011"], "20261019"), "20261017");
  assert.deepEqual(m.resolveDate("20261024", "20261017", "20261019"), { date: "20261017", future: true });
  assert.deepEqual(m.resolveDate("20210110", "20261017", "20261019"), { date: "20210110", future: false });
  assert.deepEqual(m.resolveDate("abc", "20261017", "20261019"), { date: "20261017", future: false });
});

test("年月の選択肢は範囲内だけ", () => {
  assert.deepEqual(m.yearsBetween("202001", "202610"), [2026, 2025, 2024, 2023, 2022, 2021, 2020]);
  assert.deepEqual(m.monthsOfYear(2026, "202001", "202610"), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
  assert.deepEqual(m.monthsOfYear(2020, "202003", "202610").slice(0, 2), [3, 4]);
});
