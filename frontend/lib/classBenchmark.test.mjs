// 実行: node --test lib/classBenchmark.test.mjs
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync, writeFileSync, mkdtempSync } from "node:fs";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const ts = createRequire(import.meta.url)("typescript");
const out = join(mkdtempSync(join(tmpdir(), "clsbm-")), "classBenchmark.mjs");
writeFileSync(out, ts.transpileModule(readFileSync(join(here, "classBenchmark.ts"), "utf8"), { compilerOptions: { module: ts.ModuleKind.ES2020, target: ts.ScriptTarget.ES2020 } }).outputText);
const { benchmarkSummary, benchmarkTitle, clearsLine, passState } = await import(pathToFileURL(out).href);

const b = { label: "2歳未勝利", key: "2歳|未勝利|芝", level: "surface", surface: "芝", band: null, n: 532, median: 38.3, line: 30.5, p75: 41, period: "2023-2025" };

test("title includes the surface and the distance band only when the benchmark was split", () => {
  assert.equal(benchmarkTitle(b), "2歳未勝利・芝");
  assert.equal(benchmarkTitle({ ...b, band: "中短" }), "2歳未勝利・芝・1301〜1600m");
  assert.equal(benchmarkTitle({ ...b, surface: null }), "2歳未勝利");
});

test("clearsLine is inclusive and safe on missing values", () => {
  assert.equal(clearsLine(30.5, b), true);
  assert.equal(clearsLine(30.4, b), false);
  assert.equal(clearsLine(null, b), false);
  assert.equal(clearsLine(60, null), false);
});

test("passState uses the best of the latest five runs", () => {
  const runs = [{ date: "d", value: 28 }, { date: "d", value: 33.2 }, { date: "d", value: 20 }];
  assert.deepEqual(passState(runs, b), { kind: "above", best: 33.2, gap: 33.2 - 30.5 });
  assert.equal(passState([{ date: "d", value: 25 }], b).kind, "below");
  assert.equal(passState([], b).kind, "none");
  assert.equal(passState(runs, null).kind, "none");
  const six = [...Array(5).fill({ date: "d", value: 10 }), { date: "d", value: 99 }];
  assert.equal(passState(six, b).kind, "below");                                   // 6走目以降は見ない
});

test("summary text", () => {
  assert.equal(benchmarkSummary(b), "2着馬の中央値 38 ／ 突破ライン 31（2023-2025・n=532）");
});
