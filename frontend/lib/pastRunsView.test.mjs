// 実行: node --test lib/pastRunsView.test.mjs
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
const dir = mkdtempSync(join(tmpdir(), "prv-"));
writeFileSync(join(dir, "em.mjs"), ts.transpileModule(readFileSync(join(here, "expectedMegu.ts"), "utf8"), opts).outputText);
writeFileSync(join(dir, "prv.mjs"), ts.transpileModule(readFileSync(join(here, "pastRunsView.ts"), "utf8").replaceAll('"@/lib/expectedMegu"', '"./em.mjs"'), opts).outputText);
const m = await import(pathToFileURL(join(dir, "prv.mjs")).href);

test("棒の高さ: 20〜80 を 0〜100%、範囲外は丸め、最低高さを残す", () => {
  assert.equal(m.barHeightPct(80), 100);
  assert.equal(m.barHeightPct(50), 50);
  assert.equal(m.barHeightPct(95), 100);
  assert.equal(m.barHeightPct(10), 8);
  assert.equal(m.barHeightPct(NaN), 8);
});

test("棒の色: 突破ライン以上=clear、未満=below、基準なし=plain", () => {
  assert.equal(m.barTone(55, 50), "clear");
  assert.equal(m.barTone(50, 50), "clear");
  assert.equal(m.barTone(49.9, 50), "below");
  assert.equal(m.barTone(60, null), "plain");
});

test("着順の記号", () => {
  assert.equal(m.finishMark(1), "gold");
  assert.equal(m.finishMark(3), "blue");
  assert.equal(m.finishMark(4), "none");
  assert.equal(m.finishMark(undefined), "none");
});

test("古い→新しい、最高の位置（走が1つなら強調しない・同値は新しい方）", () => {
  const recent = [{ date: "2026-09-10", value: 60 }, { date: "2026-08-10", value: 60 }, { date: "2026-07-10", value: 40 }];
  const c = m.chronological(recent);
  assert.deepEqual(c.map((r) => r.date), ["2026-07-10", "2026-08-10", "2026-09-10"]);
  assert.equal(m.bestIndex(c), 2);
  assert.equal(m.bestIndex(c.slice(0, 1)), -1);
});

test("前走の短文: グレードは G1〜G3 のときだけ", () => {
  const s = m.prevSummary({ date: "2026-07-11", value: 62, race_name: "2歳未勝利", finish: 1, pop: 8, grade: "OP" });
  assert.equal(s.text, "前走 7/11 2歳未勝利 1着・8人気");
  assert.equal(s.grade, null);
  assert.equal(m.prevSummary({ date: "2026-07-11", value: 62, race_name: "新潟2歳S", grade: "G3" }).grade, "G3");
  assert.equal(m.prevSummary(undefined), null);
});
