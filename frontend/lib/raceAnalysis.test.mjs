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
const { formatCell, sortValue, columnBlocks, isThin, entriesUnavailable, formatPeriod, heatColor, formatComputedAt, runColor } = await import(pathToFileURL(out).href);

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

test("formatPeriod: 集計期間を何年何月何日からで、件数とあわせて 1 文に。無ければ省く", () => {
  assert.equal(
    formatPeriod({ provider: "x", dataset: "y", period_from: "2019-12-22", period_to: "2026-10-24", races: 23640, rows: 323461 }),
    "集計期間: 2019年12月22日〜2026年10月24日 ／ 対象: 23,640レース・323,461出走",
  );
  assert.equal(formatPeriod({ provider: "x", dataset: "y", period_from: null, period_to: null, races: 0, rows: 0 }), null);
});

test("heatColor: 平均より高いほど緑、低いほど赤、平均付近・値なし・目盛りなしは色なし", () => {
  const sc = { lo: 0.02, mid: 0.08, hi: 0.24 };
  assert.equal(heatColor(0.24, sc), "rgba(34,197,94,0.50)");      // hi 以上 = 最も濃い緑
  assert.equal(heatColor(0.5, sc), "rgba(34,197,94,0.50)");       // 上限で頭打ち
  assert.equal(heatColor(0.02, sc), "rgba(248,81,73,0.50)");      // lo 以下 = 最も濃い赤
  assert.equal(heatColor(0, sc), "rgba(248,81,73,0.50)");
  assert.equal(heatColor(0.08, sc), undefined);                    // 平均ちょうど
  assert.equal(heatColor(0.081, sc), undefined);                   // ほぼ平均
  assert.equal(heatColor(null, sc), undefined);
  assert.equal(heatColor(0.2, null), undefined);
  const mid = heatColor(0.16, sc);                                 // hi と mid の中間 = 半分の濃さ
  assert.equal(mid, "rgba(34,197,94,0.29)");
});

test("formatComputedAt: 事前計算の時刻を月/日 時:分 で。壊れた値は null", () => {
  assert.equal(formatComputedAt("2026-10-28T17:30:05+0900"), "10/28 17:30 時点");
  assert.equal(formatComputedAt(null), null);
  assert.equal(formatComputedAt("x"), null);
});

test("runs: 位置取りの文字化と色（前ほど赤・後ろほど青）", () => {
  assert.equal(formatCell([[3, 16, 0], [1, 8, 1]], "runs"), "3/16 1/8");
  assert.equal(formatCell(null, "runs"), "–");
  assert.equal(runColor(1, 16), "hsl(8 62% 34%)");
  assert.equal(runColor(16, 16), "hsl(215 62% 34%)");
  assert.equal(runColor(1, 1), "hsl(8 62% 34%)");
});
