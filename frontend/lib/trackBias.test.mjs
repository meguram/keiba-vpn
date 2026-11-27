// 実行: node --test lib/trackBias.test.mjs
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync, writeFileSync, mkdtempSync } from "node:fs";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const ts = createRequire(import.meta.url)("typescript");
const out = join(mkdtempSync(join(tmpdir(), "trackbias-")), "trackBias.mjs");
writeFileSync(out, ts.transpileModule(readFileSync(join(here, "trackBias.ts"), "utf8"), { compilerOptions: { module: ts.ModuleKind.ES2020, target: ts.ScriptTarget.ES2020 } }).outputText);
const m = await import(pathToFileURL(out).href);

test("前後の札: 前残り・差し有利・中立", () => {
  assert.deepEqual(m.frontBadge("強い前残り"), { text: "強い前残り", tone: "front", strong: true });
  assert.deepEqual(m.frontBadge("差し有利"), { text: "差し有利", tone: "back", strong: false });
  assert.deepEqual(m.frontBadge("前後は中立"), { text: "前後：中立", tone: "neutral", strong: false });
  assert.equal(m.frontBadge(null), null);
});

test("内外の札: 内有利・外有利・中立", () => {
  assert.deepEqual(m.innerBadge("内有利"), { text: "内有利", tone: "inner", strong: false });
  assert.deepEqual(m.innerBadge("強い外有利"), { text: "強い外有利", tone: "outer", strong: true });
  assert.deepEqual(m.innerBadge("内外は中立"), { text: "内外：中立", tone: "neutral", strong: false });
});

test("前後・内外の札: 強くない前残り・強い差し・強い内・判定なし", () => {
  assert.deepEqual(m.frontBadge("前残り"), { text: "前残り", tone: "front", strong: false });
  assert.deepEqual(m.frontBadge("強い差し有利"), { text: "強い差し有利", tone: "back", strong: true });
  assert.deepEqual(m.innerBadge("強い内有利"), { text: "強い内有利", tone: "inner", strong: true });
  assert.equal(m.frontBadge("判定なし（レース 3 未満）"), null);
  assert.equal(m.innerBadge("判定なし（レース 3 未満）"), null);
});

test("1 行目: 大きな文字は場と面だけ・札は中立でない方だけ", () => {
  assert.equal(m.headline("福島", "芝"), "福島芝");
  assert.equal(m.headline("東京", "ダ"), "東京ダート");
  assert.deepEqual(m.labelBadges("前後は中立", "内有利").map((b) => b.text), ["内有利"]);
  assert.deepEqual(m.labelBadges("前残り", "外有利").map((b) => b.text), ["外有利", "前残り"]);
  assert.deepEqual(m.labelBadges("前後は中立", "内外は中立").map((b) => [b.text, b.tone]), [["中立", "neutral"]]);
  assert.deepEqual(m.labelBadges("判定なし（レース 3 未満）", "判定なし（レース 3 未満）", "判定なし（レース 3 未満）").map((b) => b.text), ["判定なし（レース 3 未満）"]);
});

test("位置の列: 序盤は通過順の最初・4 角は最後", () => {
  assert.equal(m.positionText("6-6-4-2", 15), "序盤 6/15 → 4 角 2/15");
  assert.equal(m.positionText("3", 10), "序盤 3/10 → 4 角 3/10");
  assert.equal(m.positionText("", 10), "—");
  assert.equal(m.positionText(null, 10), "—");
  assert.equal(m.positionText("5-3", null), "—");
  assert.equal(m.positionShort("6-6-4-2", 15), "6→2/15");
  assert.equal(m.earlySortKey("6-6-4-2"), 6);
  assert.equal(m.earlySortKey(""), 999);
});
