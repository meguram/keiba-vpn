// 実行: node --test lib/columns.test.mjs
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync, writeFileSync, mkdtempSync, readdirSync } from "node:fs";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const ts = createRequire(import.meta.url)("typescript");
const src = ts.transpileModule(readFileSync(join(here, "columns.ts"), "utf8"), { compilerOptions: { module: ts.ModuleKind.ES2020, target: ts.ScriptTarget.ES2020 } }).outputText;
const out = join(mkdtempSync(join(tmpdir(), "col-")), "columns.mjs");
writeFileSync(out, src);
const { categoryTrail, allCategoryPaths, chartMax, chartRange, chartIsEmpty, chartVariant, cascadeOptions, cascadePick, axis2Max, columnsIn, countIn } = await import(pathToFileURL(out).href);

const dir = join(here, "..", "content", "columns");
const articles = readdirSync(dir).filter((f) => f.endsWith(".json")).map((f) => JSON.parse(readFileSync(join(dir, f), "utf8")));

test("記事の slug は重複せず、カテゴリは実在するパスで、ファイル名と一致する", () => {
  const files = readdirSync(dir).filter((f) => f.endsWith(".json"));
  assert.equal(new Set(articles.map((a) => a.slug)).size, articles.length);
  for (const [i, a] of articles.entries()) {
    assert.equal(files[i], `${a.slug}.json`);
    assert.ok(a.categories.length > 0, a.slug);
    for (const c of a.categories) assert.ok(categoryTrail(c), `${a.slug}: 未定義のカテゴリ ${c}`);
  }
});

test("親カテゴリのページには配下の記事も並ぶ", () => {
  const all = articles;
  const upsetN = countIn(all, "upset");
  assert.equal(upsetN, columnsIn(all, "upset").length);
  assert.ok(upsetN >= countIn(all, "upset/conditions"));
  assert.ok(allCategoryPaths().includes("jockey/distance"));
});

test("グラフ: 縦軸の上限は切りのよい数、値が無ければ空扱い", () => {
  assert.equal(chartMax([{ values: [0.31, 0.44, null] }]), 0.5);
  assert.equal(chartMax([{ values: [0.12, 0.07] }]), 0.15);
  assert.equal(chartMax([{ values: [null, null] }]), 0.5);
  assert.equal(chartIsEmpty([{ values: [null] }, { values: [] }]), true);
  assert.equal(chartIsEmpty([{ values: [null, 0.2] }]), false);
});

test("グラフ: 人気などの数の縦軸と、プルダウンでの系列の切り替え", () => {
  assert.equal(chartMax([{ values: [4.46, 3.9], ci: [[4.41, 4.52], null] }], "num"), 5);
  const block = {
    type: "chart", title: "t", xLabel: "x", yLabel: "y", xs: ["a"], caption: "全体",
    series: [{ name: "s", values: [0.5] }],
    filters: [{ label: "距離", options: ["全体", "短"] }, { label: "芝ダ", options: ["全体", "芝"] }],
    variants: { "短|芝": { series: [{ name: "s", values: [0.3] }], caption: "短芝" }, "短|全体": { series: [{ name: "s", values: [0.4] }] } },
  };
  assert.equal(chartVariant(block, ["全体", "全体"]), block);
  assert.equal(chartVariant(block, ["短", "芝"]).series[0].values[0], 0.3);
  assert.equal(chartVariant(block, ["短", "芝"]).caption, "短芝");
  assert.equal(chartVariant(block, ["短", "全体"]).caption, "全体");             // caption が無い組み合わせは元の説明
  assert.equal(chartVariant(block, ["全体", "芝"]), block);                     // 該当する組み合わせが無ければ元のまま
});

test("プルダウンつきのグラフは、全部の組み合わせに系列がある", () => {
  for (const a of articles) {
    const walk = (bs) => bs.flatMap((b) => (b.type === "accordion" ? walk(b.blocks) : [b]));
    for (const c of walk(a.body).filter((b) => b.type === "chart" && b.filters)) {
      if (c.cascade) {                                                         // 階層のグラフ: 組み合わせは一部だけ。全部の key が選択肢でできていて、選べる先頭が必ず存在する
        for (const k of Object.keys(c.variants ?? {})) k.split("|").forEach((o, i) => assert.ok(c.filters[i].options.includes(o), `${a.slug}: ${k}`));
        assert.ok(Object.keys(c.variants).length > 0);
        continue;
      }
      const keys = c.filters.reduce((acc, f) => acc.flatMap((k) => f.options.map((o) => (k ? `${k}|${o}` : o))), [""]);
      for (const k of keys.slice(1)) assert.ok(c.variants?.[k], `${a.slug} ${c.title}: ${k} が無い`);
    }
  }
});

test("グラフ: 右軸は左軸の補助線にそろう切りのよい上限で、プルダウンで右軸の系列も替わる", () => {
  assert.equal(axis2Max([{ values: [4.46, 3.9], ci: [[4.41, 4.52], null] }], 7, "num"), 7);     // 4.52 / 7 区切り → 1 刻み
  assert.equal(axis2Max([{ values: [0.6] }], 4, "pct"), 0.8);
  const block = {
    type: "chart", title: "t", xLabel: "x", yLabel: "y", xs: ["a"], series: [{ name: "s", values: [0.5] }],
    y2: { label: "人気", format: "num", series: [{ name: "p", values: [4] }] },
    filters: [{ label: "f", options: ["全体", "芝"] }], variants: { 芝: { series: [{ name: "s", values: [0.4] }], y2Series: [{ name: "p", values: [3] }] } },
  };
  assert.equal(chartVariant(block, ["芝"]).y2.series[0].values[0], 3);
  assert.equal(chartVariant(block, ["芝"]).y2.label, "人気");
});

test("cascade: 前の選択に合う選択肢だけを出し、後ろの選択が成り立たなくなれば先頭に寄せる", () => {
  const ser = (v) => [{ name: "a", values: [v] }];
  const block = {
    type: "chart", title: "t", xLabel: "x", yLabel: "y", xs: ["1"], series: ser(0), cascade: true,
    filters: [{ label: "会場", options: ["全場", "東京"] }, { label: "距離", options: ["芝1600", "芝2000", "ダ1200"] }],
    variants: { "全場|芝1600": { series: ser(1) }, "東京|芝1600": { series: ser(2) }, "東京|ダ1200": { series: ser(3) } },
  };
  assert.deepEqual(cascadeOptions(block, ["全場", "芝1600"], 1), ["芝1600"]);
  assert.deepEqual(cascadeOptions(block, ["東京", "芝1600"], 1), ["芝1600", "ダ1200"]);
  assert.deepEqual(cascadePick(block, ["東京", "ダ1200"], 0, "全場"), ["全場", "芝1600"]);
  assert.equal(chartVariant(block, ["全場", "芝1600"]).series[0].values[0], 1);        // cascade では先頭の組み合わせも variants を引く
});

test("chartRange fit: 0 始まりにせず値の範囲に合わせる", () => {
  const r = chartRange([{ values: [12.1, 12.4, null], ci: [[12.0, 12.2], [12.3, 12.5], null] }], "num", true);
  assert.ok(r.min > 11.5 && r.min <= 12.0 && r.max >= 12.5 && r.max < 13);
  assert.equal(chartRange([{ values: [0.1, 0.3] }], "num").min, 0);               // fit なしは今までどおり
});

test("グラフ: variant が xs を持てば横軸もその長さに替わる", () => {
  const block = { type: "chart", title: "t", xLabel: "x", yLabel: "y", xs: ["a", "b", "c"], series: [{ name: "s", values: [1, 2, 3] }],
    filters: [{ label: "距離", options: ["長", "短"] }], variants: { 短: { xs: ["a", "b"], series: [{ name: "s", values: [1, 2] }] } } };
  assert.deepEqual(chartVariant(block, ["短"]).xs, ["a", "b"]);
  assert.deepEqual(chartVariant(block, ["長"]).xs, ["a", "b", "c"]);
});
