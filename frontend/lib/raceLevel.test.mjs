// 実行: node --test lib/raceLevel.test.mjs
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync, writeFileSync, mkdtempSync } from "node:fs";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const ts = createRequire(import.meta.url)("typescript");
const out = join(mkdtempSync(join(tmpdir(), "racelevel-")), "raceLevel.mjs");
writeFileSync(out, ts.transpileModule(readFileSync(join(here, "raceLevel.ts"), "utf8"), { compilerOptions: { module: ts.ModuleKind.ES2020, target: ts.ScriptTarget.ES2020 } }).outputText);
const m = await import(pathToFileURL(out).href);

test("馬場・レースの 5 段", () => {
  const words = ["速い", "やや速い", "標準", "やや時計が掛かる", "時計が掛かる"];
  assert.deepEqual(words.map((w) => m.levelBadge(w, "track").step), [-2, -1, 0, 1, 2]);
  assert.equal(m.levelBadge("速い", "track").strong, true);
  assert.equal(m.levelBadge("やや速い", "track").strong, false);
  assert.equal(m.levelBadge("標準", "track").color, "#94a3b8");
});

test("メンバーの 5 段", () => {
  const words = ["高い", "やや高い", "標準", "やや低い", "低い"];
  assert.deepEqual(words.map((w) => m.levelBadge(w, "member").step), [-2, -1, 0, 1, 2]);
  assert.equal(m.levelBadge("高い", "member").color, m.levelBadge("速い", "track").color);
});

test("色の向き: 速い・高いの側は暖色、遅い・低いの側は寒色、ペースは太字にしない", () => {
  assert.deepEqual([-2, -1, 0, 1, 2].map((k) => m.STEP_COLOR[String(k)]), ["#ef4444", "#f59e0b", "#94a3b8", "#38bdf8", "#3b82f6"]);
  assert.equal(m.levelBadge("ハイ", "pace").color, m.STEP_COLOR["-2"]);
  assert.equal(m.levelBadge("スロー", "pace").color, m.STEP_COLOR["2"]);
  assert.equal(m.levelBadge("ハイ", "pace").strong, false);
});

test("ペースの 3 段は「〜ペース」の札", () => {
  assert.deepEqual(["ハイ", "平均", "スロー"].map((w) => m.levelBadge(w, "pace").text), ["ハイペース", "平均ペース", "スローペース"]);
  assert.deepEqual(["ハイ", "平均", "スロー"].map((w) => m.levelBadge(w, "pace").step), [-2, 0, 2]);
});

test("無い言葉は null、表に無い言葉は灰の札", () => {
  assert.equal(m.levelBadge(null, "track"), null);
  assert.equal(m.levelBadge("", "member"), null);
  assert.deepEqual(m.levelBadge("謎", "track"), { text: "謎", step: null, color: "#94a3b8", strong: false });
  assert.equal(m.levelBadge("速い", "member").step, null);
});

test("秒の文字とペースの補正の分", () => {
  assert.equal(m.secText(-0.84), "−0.8 秒");
  assert.equal(m.secText(0.5), "+0.5 秒");
  assert.equal(m.secText(0), "±0.0 秒");
  assert.equal(m.secText(null), "—");
  assert.equal(m.secText(-0.04), "±0.0 秒");
  assert.equal(m.paceAdjust(0.3, 0.1), 0.2);
  assert.equal(m.paceAdjust(-0.8, -0.3), -0.5);
  assert.equal(m.paceAdjust(null, -0.3), null);
});
