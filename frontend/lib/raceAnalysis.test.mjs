// 実行: node --test lib/raceAnalysis.test.mjs
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync, writeFileSync, mkdtempSync } from "node:fs";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const ts = createRequire(import.meta.url)("typescript");
const src = ts.transpileModule(readFileSync(join(here, "raceAnalysis.ts"), "utf8"), { compilerOptions: { module: ts.ModuleKind.ES2020, target: ts.ScriptTarget.ES2020 } }).outputText;
const out = join(mkdtempSync(join(tmpdir(), "ra-")), "raceAnalysis.mjs");
writeFileSync(out, src);
const { formatCell, sortValue, columnBlocks, isThin, entriesUnavailable, formatSource } = await import(pathToFileURL(out).href);

test("formatCell: 成績・率・整数、値が無ければダッシュ", () => {
  assert.equal(formatCell([6, 11, 7, 38], "record"), "6-11-7-38");
  assert.equal(formatCell(null, "record"), "–");
  assert.equal(formatCell(0.06666, "pct"), "6.7%");
  assert.equal(formatCell(0, "pct"), "0.0%");              // 0% は値なしではない
  assert.equal(formatCell(null, "pct"), "–");
  assert.equal(formatCell(12.4, "int"), "12");
  assert.equal(formatCell("", "text"), "–");
});

test("sortValue: 成績は 1 着の数で並べる", () => {
  assert.equal(sortValue([6, 11, 7, 38], "record"), "6");
  assert.equal(sortValue(null, "record"), undefined);
  assert.equal(sortValue(0.5, "pct"), undefined);
});

const groups = [{ key: "a", label: "A", n_key: "a_n", min_n: 10 }, { key: "b", label: "B", min_n: 0 }];
const columns = [
  { key: "b_x", label: "x", fmt: "pct", group: "b" },
  { key: "a_x", label: "x", fmt: "pct", group: "a" },
  { key: "a_y", label: "y", fmt: "pct", group: "a" },
  { key: "z", label: "z", fmt: "text" },
  { key: "q", label: "q", fmt: "text", group: "unknown" },
];

test("columnBlocks: 群の定義順に並べ、群の無い列は最後にまとめる", () => {
  const blocks = columnBlocks(columns, groups);
  assert.deepEqual(blocks.map((b) => [b.group?.key ?? null, b.columns.map((c) => c.key)]), [["a", ["a_x", "a_y"]], ["b", ["b_x"]], [null, ["z", "q"]]]);
});

test("isThin: 標本数が min_n 未満の群だけ薄くする", () => {
  assert.equal(isThin(groups[0], { a_n: 9 }), true);
  assert.equal(isThin(groups[0], { a_n: 10 }), false);
  assert.equal(isThin(groups[0], {}), false);
  assert.equal(isThin(groups[1], { b_n: 0 }), false);
  assert.equal(isThin(null, { a_n: 0 }), false);
});

test("entriesUnavailable: 未来日で出馬表の段階が無い／none のレースだけ選べない", () => {
  assert.equal(entriesUnavailable({ entries_phase: null }, "20261031", "20261028"), true);
  assert.equal(entriesUnavailable({}, "20261031", "20261028"), true);
  assert.equal(entriesUnavailable({ entries_phase: "none" }, "20261031", "20261028"), true);
  assert.equal(entriesUnavailable({ entries_phase: "registered" }, "20261031", "20261028"), false);   // 特別登録（暫定）は選べる
  assert.equal(entriesUnavailable({ entries_phase: "final" }, "20261031", "20261028"), false);
  assert.equal(entriesUnavailable({ entries_phase: null }, "20261028", "20261028"), false);           // 当日は選べる
  assert.equal(entriesUnavailable({ entries_phase: null }, "20261025", "20261028"), false);           // 過去は選べる
});

test("formatSource: 提供元・集計期間・件数を 1 文にする。期間や件数が無ければ省く", () => {
  assert.equal(
    formatSource({ provider: "netkeiba.com", dataset: "レース結果", period_from: "2020-01-05", period_to: "2026-10-25", races: 2880, rows: 39639 }),
    "データの出典: netkeiba.com（レース結果） ／ 集計期間: 2020/01/05〜2026/10/25 ／ 対象: 2,880レース・39,639出走",
  );
  assert.equal(formatSource({ provider: "netkeiba.com", dataset: "レース結果", period_from: null, period_to: null, races: 0, rows: 0 }), "データの出典: netkeiba.com（レース結果）");
});
