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
const { categoryTrail, allCategoryPaths, chartMax, chartIsEmpty, chartVariant, axis2Max, columnsIn, countIn } = await import(pathToFileURL(out).href);

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
