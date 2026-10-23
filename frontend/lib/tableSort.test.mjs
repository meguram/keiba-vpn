// 実行: node --test lib/tableSort.test.mjs
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync, writeFileSync, mkdtempSync } from "node:fs";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const ts = createRequire(import.meta.url)("typescript");
const out = join(mkdtempSync(join(tmpdir(), "tablesort-")), "tableSort.mjs");
writeFileSync(out, ts.transpileModule(readFileSync(join(here, "tableSort.ts"), "utf8"), { compilerOptions: { module: ts.ModuleKind.ES2022, target: ts.ScriptTarget.ES2022 } }).outputText);
const m = await import(pathToFileURL(out).href);
const n = (t) => m.parseCell(t).n;

test("数値の解釈", () => {
  assert.equal(n("77.7%"), 77.7);
  assert.equal(n("1.6倍"), 1.6);
  assert.equal(n("¥17,470"), 17470);
  assert.equal(n("3番手"), 3);
  assert.equal(n("+0.9"), 0.9);
  assert.equal(n("1.9〜2.8"), 1.9);
  assert.equal(n("1:56.1"), 116.1);
  assert.equal(n("０.８２"), 0.82);          // 全角
  assert.equal(m.parseCell("—").empty, true);
  assert.equal(m.parseCell("◎").n, 0);
});

test("文字列は数値扱いにしない", () => {
  assert.equal(m.parseCell("3歳以上2勝クラス").numericLike, false);
  assert.equal(m.parseCell("モックウマ09").numericLike, false);
});

test("列の判定と比較（値なしは最後）", () => {
  const col = ["35.2", "—", "33.5", "34.1"].map((t) => m.parseCell(t));
  assert.equal(m.isNumericColumn(col), true);
  const asc = [...col].sort((a, b) => m.compareCells(a, b, true, 1)).map((c) => c.s);
  const desc = [...col].sort((a, b) => m.compareCells(a, b, true, -1)).map((c) => c.s);
  assert.deepEqual(asc, ["33.5", "34.1", "35.2", ""]);
  assert.deepEqual(desc, ["35.2", "34.1", "33.5", ""]);
  const names = ["モック2", "あい", "モック1"].map((t) => m.parseCell(t));
  assert.equal(m.isNumericColumn(names), false);
});
