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
const { categoryTrail, allCategoryPaths, chartMax, chartIsEmpty, columnsIn, countIn } = await import(pathToFileURL(out).href);

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
