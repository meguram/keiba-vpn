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
const { formatCell, sortValue, columnBlocks, isThin, entriesUnavailable, formatPeriod, heatColor, formatComputedAt, runColor, layoutPositionMatrix, matrixOverlap } = await import(pathToFileURL(out).href);

// 表の並べ替え（TableSorter が使う比較）。sortValue の値が実際にどう並ぶかを確かめる
async function loadTableSort() {
  const f = join(mkdtempSync(join(tmpdir(), "ra-sort-")), "tableSort.mjs");
  writeFileSync(f, ts.transpileModule(readFileSync(join(here, "tableSort.ts"), "utf8"), { compilerOptions: { module: ts.ModuleKind.ES2020, target: ts.ScriptTarget.ES2020 } }).outputText);
  return import(pathToFileURL(f).href);
}

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

test("sortValue: 脚質は色分けの定義の順（ハナ → 先行(前) → 先行(後ろ) → 差し → 追込）、不明は最後", async () => {
  const { parseCell, compareCells } = await loadTableSort();
  const palette = { "ハナ": "a", "先行(前)": "b", "先行(後ろ)": "c", "差し": "d", "追込": "e" };
  const vals = ["差し", "不明", "追込", "ハナ", "先行(後ろ)", "先行(前)"];
  const cells = vals.map((v) => ({ v, c: parseCell(v, sortValue(v, "text", palette)) }));
  const asc = [...cells].sort((a, b) => compareCells(a.c, b.c, true, 1)).map((x) => x.v);
  const desc = [...cells].sort((a, b) => compareCells(a.c, b.c, true, -1)).map((x) => x.v);
  assert.deepEqual(asc, ["ハナ", "先行(前)", "先行(後ろ)", "差し", "追込", "不明"]);
  assert.deepEqual(desc, ["追込", "差し", "先行(後ろ)", "先行(前)", "ハナ", "不明"]);
  assert.equal(sortValue("差し", "text"), undefined);       // 色分けの無い文字の列は従来どおり
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

// ── 位置取りマトリックス（1 頭身 = 1、lane 0 = ラチ沿い） ──
// N = 10 頭なら mean 0.1 刻みが 1 頭身。中心 c = mean × 10 − 0.5
const H = (num, mean) => ({ key: `k${num}`, num, mean });
const laneOf = (lay) => Object.fromEntries(lay.placed.map((p) => [p.num, p.lane]));

test("matrix: 誰とも重ならない馬はすべてラチ沿い（内から順に並ぶ）", () => {
  const lay = layoutPositionMatrix([H(3, 0.1), H(1, 0.3), H(2, 0.5), H(4, 0.9)], 10);
  assert.deepEqual(laneOf(lay), { 1: 0, 2: 0, 3: 0, 4: 0 });
  assert.equal(lay.lanes, 1);
});

test("matrix: 重なる 2 頭は馬番の小さい馬がラチ側（馬番と前後が逆でも）", () => {
  assert.deepEqual(laneOf(layoutPositionMatrix([H(8, 0.30), H(2, 0.33)], 10)), { 2: 0, 8: 1 });
  assert.deepEqual(laneOf(layoutPositionMatrix([H(2, 0.30), H(8, 0.33)], 10)), { 2: 0, 8: 1 });
  assert.deepEqual(laneOf(layoutPositionMatrix([H(5, 0.5), H(1, 0.5)], 10)), { 1: 0, 5: 1 });   // 完全に同じ位置
});

test("matrix: 互いに重なる 3 頭は馬番順に 0・1・2", () => {
  assert.deepEqual(laneOf(layoutPositionMatrix([H(7, 0.42), H(3, 0.40), H(5, 0.45)], 10)), { 3: 0, 5: 1, 7: 2 });
});

test("matrix: 連鎖（1–2 と 2–3 は重なり、1–3 は重ならない）は 1 と 3 がラチ沿い", () => {
  assert.deepEqual(laneOf(layoutPositionMatrix([H(1, 0.30), H(2, 0.37), H(3, 0.44)], 10)), { 1: 0, 2: 1, 3: 0 });
  // 外のラインが空いても、内のラインで重ならなければ内に入る（2 段目にいる馬の後ろでもラチ沿い）
  assert.deepEqual(laneOf(layoutPositionMatrix([H(1, 0.30), H(2, 0.35), H(3, 0.38), H(4, 0.41)], 10)), { 1: 0, 2: 1, 3: 2, 4: 0 });
});

test("matrix: ちょうど 1 頭身離れる馬は重ならない／少しでも近ければ重なる", () => {
  assert.equal(matrixOverlap(2.5, 3.5), false);
  assert.equal(matrixOverlap(2.5, 3.49), true);
  assert.deepEqual(laneOf(layoutPositionMatrix([H(1, 0.3), H(2, 0.4)], 10)), { 1: 0, 2: 0 });
});

test("matrix: 位置は 0.5〜N−0.5 に収まり、平均の無い馬は置かない", () => {
  const lay = layoutPositionMatrix([H(1, 0.01), H(2, 1.2), H(3, null), { key: "x", num: null, mean: undefined }], 10);
  const c = Object.fromEntries(lay.placed.map((p) => [p.num, p.c]));
  assert.equal(c[1], 0.5);
  assert.equal(c[2], 9.5);
  assert.deepEqual(lay.unknown.map((h) => h.key), ["k3", "x"]);
});

test("matrix: 馬番の無い馬は馬番のある馬の後（外）に置く", () => {
  assert.deepEqual(layoutPositionMatrix([{ key: "z", num: null, mean: 0.3 }, H(9, 0.3)], 10).placed.map((p) => [p.key, p.lane]), [["k9", 0], ["z", 1]]);
});

test("matrix: 乱数の並びでも、同じラインで重ならず・内のラインに入れる馬は内に入っている", () => {
  let seed = 7;
  const rnd = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648);
  for (let t = 0; t < 300; t++) {
    const n = 2 + Math.floor(rnd() * 17);
    const horses = Array.from({ length: n }, (_, i) => H(i + 1, rnd() < 0.1 ? null : rnd()));
    const lay = layoutPositionMatrix(horses, n);
    const byNum = [...lay.placed].sort((a, b) => a.num - b.num);
    for (const p of byNum) {
      // 同じラインの馬とは重ならない
      for (const q of lay.placed) if (q !== p && q.lane === p.lane) assert.equal(matrixOverlap(p.c, q.c), false);
      // 内側のどのラインにも、自分より馬番の小さい重なる馬がいる（だから内に入れない）
      for (let l = 0; l < p.lane; l++) {
        assert.ok(lay.placed.some((q) => q.lane === l && q.num < p.num && matrixOverlap(p.c, q.c)), `馬番${p.num} は lane ${l} に入れたはず`);
      }
    }
    assert.equal(lay.placed.length + lay.unknown.length, n);
  }
});

// ── スマホの表の並べ替え（脚質・位置は列から外すので、選択で並べ替える） ──
test("mobileSortOptions: 脚質（色分け）と位置（range）がある表だけ、馬番と並べて選べる", async () => {
  const { mobileSortOptions } = await import(pathToFileURL(out).href);
  const pp = [
    { key: "style", label: "脚質", fmt: "text", palette: { "ハナ": "a", "追込": "b" } },
    { key: "range", label: "直近5走の位置（前から%）", fmt: "range" },
    { key: "n", label: "走数", fmt: "int" },
    { key: "band_win", label: "勝率", fmt: "pct" },
  ];
  assert.deepEqual(mobileSortOptions(pp).map((o) => o.label), ["馬番", "脚質", "位置（前から）"]);
  assert.deepEqual(mobileSortOptions([{ key: "sire", label: "父", fmt: "text" }, { key: "win", label: "勝率", fmt: "pct" }]), []);
});

test("mobileSortValue: TableSorter と同じ比較で、脚質は定義の順・位置は平均の順・馬番は番号順、値なしは最後", async () => {
  const { mobileSortValue } = await import(pathToFileURL(out).href);
  const { parseCell, compareCells, isNumericColumn } = await loadTableSort();
  const palette = { "ハナ": "a", "先行(前)": "b", "先行(後ろ)": "c", "差し(前)": "d", "差し(後ろ)": "e", "追込": "f" };
  const cols = [{ key: "style", label: "脚質", fmt: "text", palette }, { key: "range", label: "位置", fmt: "range" }];
  const rc = (mean) => ({ lo: mean, hi: mean, mean, pts: [] });
  const es = [
    { e: { key: "a", horse_number: 3 }, c: { style: "差し(前)", range: rc(0.5) } },
    { e: { key: "b", horse_number: 1 }, c: { style: "追込", range: rc(0.9) } },
    { e: { key: "c", horse_number: 7 }, c: { style: "ハナ", range: rc(0.05) } },
    { e: { key: "d", horse_number: 2 }, c: { style: "不明", range: null } },
    { e: { key: "e", horse_number: 5 }, c: { style: "先行(前)", range: rc(0.15) } },
  ];
  const order = (key, dir) => {
    const cells = es.map((x) => parseCell("", mobileSortValue(x.e, x.c, key, cols)));
    const numeric = isNumericColumn(cells);
    assert.ok(numeric);
    return es.map((x, i) => [x, i]).sort((p, q) => compareCells(cells[p[1]], cells[q[1]], numeric, dir) || p[1] - q[1]).map(([x]) => x.e.key);
  };
  assert.deepEqual(order("style", 1), ["c", "e", "a", "b", "d"]);        // ハナ → 先行(前) → 差し(前) → 追込 → 不明
  assert.deepEqual(order("style", -1), ["b", "a", "e", "c", "d"]);       // 降順でも不明は最後
  assert.deepEqual(order("range", 1), ["c", "e", "a", "b", "d"]);        // 平均の前から。位置なしは最後
  assert.equal(mobileSortValue({ key: "x", horse_number: 1 }, {}, "_num", cols), undefined);   // 馬番 = 並べ替えなし（data-sort を付けない）
  assert.equal(mobileSortValue({ key: "x", horse_number: 1 }, {}, "nope", cols), "");
});

test("mobileSortReducer: 見出しで並べ替えたあとも、同じ値を選び直せば並べ直す", async () => {
  const { mobileSortReducer, MOBILE_SORT_INIT } = await import(pathToFileURL(out).href);
  let s = mobileSortReducer(MOBILE_SORT_INIT, { type: "select", key: "style" });
  assert.deepEqual(s, { key: "style", picked: true, run: 1 });
  s = mobileSortReducer(s, { type: "header" });                       // 見出しを押した → 選択は「見出しの順」の表示に
  assert.deepEqual(s, { key: "style", picked: false, run: 1 });
  assert.equal(mobileSortReducer(s, { type: "header" }), s);           // 何度押しても同じ（再描画しない）
  s = mobileSortReducer(s, { type: "select", key: "style" });          // 同じ脚質を選び直す → 並べ直す（run が進む）
  assert.deepEqual(s, { key: "style", picked: true, run: 2 });
});

test("mobileSortShown: 見出しの順の間は選択を空にし、data-sort は付けない（隠れた基準で並ばない）", async () => {
  const { mobileSortShown, mobileSortValue } = await import(pathToFileURL(out).href);
  const cols = [{ key: "style", label: "脚質", fmt: "text", palette: { "ハナ": "a" } }];
  assert.deepEqual(mobileSortShown({ key: "style", picked: true, run: 1 }, "style"), { select: "style", dataKey: "style" });
  const head = mobileSortShown({ key: "style", picked: false, run: 1 }, "style");
  assert.deepEqual(head, { select: "", dataKey: "_num" });
  assert.equal(mobileSortValue({ key: "x", horse_number: 1 }, { style: "ハナ" }, head.dataKey, cols), undefined);
  // 群を切り替えて選択肢に無い値になったら、選択も馬番（effectiveMobileSort の結果）を見せる
  assert.deepEqual(mobileSortShown({ key: "style", picked: true, run: 1 }, "_num"), { select: "_num", dataKey: "_num" });
});

test("effectiveMobileSort: 群・モジュールを切り替えて選択肢に無い値は馬番（並べ替えなし）に戻す", async () => {
  const { effectiveMobileSort, mobileSortOptions } = await import(pathToFileURL(out).href);
  const pp = mobileSortOptions([{ key: "style", label: "脚質", fmt: "text", palette: { "ハナ": "a" } }, { key: "range", label: "位置", fmt: "range" }]);
  assert.equal(effectiveMobileSort(pp, "style"), "style");
  assert.equal(effectiveMobileSort(pp, "_num"), "_num");
  assert.equal(effectiveMobileSort([], "style"), "_num");                       // 選択肢の無い表
  assert.equal(effectiveMobileSort(mobileSortOptions([{ key: "range", label: "位置", fmt: "range" }]), "style"), "_num");
  // range が 2 本あるときは列名で区別する
  const two = mobileSortOptions([{ key: "r1", label: "位置A", fmt: "range" }, { key: "r2", label: "位置B", fmt: "range" }]);
  assert.deepEqual(two.map((o) => o.label), ["馬番", "位置A", "位置B"]);
});
