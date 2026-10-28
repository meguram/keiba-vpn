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
const { formatCell, sortValue, columnBlocks, isThin } = await import(pathToFileURL(out).href);

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
