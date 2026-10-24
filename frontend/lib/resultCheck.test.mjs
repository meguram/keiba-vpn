// 実行: node --test lib/resultCheck.test.mjs
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync, writeFileSync, mkdtempSync } from "node:fs";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const ts = createRequire(import.meta.url)("typescript");
const out = join(mkdtempSync(join(tmpdir(), "resultcheck-")), "resultCheck.mjs");
writeFileSync(out, ts.transpileModule(readFileSync(join(here, "resultCheck.ts"), "utf8"), { compilerOptions: { module: ts.ModuleKind.ES2022, target: ts.ScriptTarget.ES2022 } }).outputText);
const m = await import(pathToFileURL(out).href);

test("実際のペースは前後半3Fの差で決まる", () => {
  assert.equal(m.actualPace(35.0, 36.4).level, "high");
  assert.equal(m.actualPace(37.3, 36.4).level, "mid");
  assert.equal(m.actualPace(38.0, 36.4).level, "slow");
  assert.equal(m.actualPace(null, 36.4), null);
});

test("ペースの判定は1段違いが惜しい", () => {
  assert.equal(m.paceVerdict("mid", "mid"), "hit");
  assert.equal(m.paceVerdict("mid", "high"), "near");
  assert.equal(m.paceVerdict("slow", "high"), "miss");
});

test("最初のコーナーの順位と位置取りの差", () => {
  assert.equal(m.firstCornerPos("9-9-14-17"), 9);
  assert.equal(m.firstCornerPos(""), null);
  assert.deepEqual(m.positionVerdict(3, 5), { diff: 2, verdict: "hit" });
  assert.equal(m.positionVerdict(3, 8).verdict, "miss");
  const s = m.summarizePositions([{ no: 1, est: 1, actual: 2 }, { no: 2, est: 2, actual: 1 }, { no: 3, est: 3, actual: 9 }]);
  assert.deepEqual([s.n, s.hit, s.frontPred, s.frontActual], [3, 2, 1, 2]);
});
