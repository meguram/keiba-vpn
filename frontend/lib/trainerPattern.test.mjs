import test from "node:test";
import assert from "node:assert/strict";
import { writeFileSync, readFileSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import ts from "typescript";

const src = readFileSync(new URL("./trainerPattern.ts", import.meta.url), "utf8");
const out = ts.transpileModule(src, { compilerOptions: { module: ts.ModuleKind.ES2020, target: ts.ScriptTarget.ES2020 } }).outputText;
const file = join(mkdtempSync(join(tmpdir(), "tp-")), "tp.mjs");
writeFileSync(file, out);
const m = await import(file);

// 6 区間: <2, 2-3, 3-5, 5-10, 10-20, 20-50
const A = [10, 8, 2, 3, 1, 4.5, 5.0, [1, 2, 3, 2, 1, 1], [1, 0, 1, 0, 0, 0], [1, 1, 2, 1, 0, 1], [1.5, 0, 3, 0, 0, 0], [1.1, 1.2, 1.3, 0.4, 0, 1.0]];
const B = [5, 5, 1, 0, 1, 2.0, 1.0, [0, 1, 1, 1, 1, 1], [0, 0, 0, 1, 0, 0], [0, 0, 1, 1, 0, 0], [0, 0, 0, 2.0, 0, 0], [0, 0, 0.5, 0.5, 0, 0]];

test("rowStats は勝率・複勝率・回収率と4着以下を返す", () => {
  const s = m.rowStats(A);
  assert.equal(s.c4p, 4);
  assert.equal(s.win, 0.2);
  assert.equal(s.place, 0.6);
  assert.equal(s.winRoi, 0.45);
  assert.equal(m.rowStats([0, 0, 0, 0, 0, 0, 0, [], [], [], [], []]).win, null);
});

test("sumRows は足し合わせ、頭数は null", () => {
  const s = m.sumRows([A, B]);
  assert.equal(s[0], 15);
  assert.equal(s[1], null);
  assert.deepEqual(s[7], [1, 3, 4, 3, 2, 2]);
  assert.equal(s[5], 6.5);
  assert.equal(m.sumRows([]), null);
});

test("binLabels と binStats（合計の回収率は区間の合計から）", () => {
  assert.deepEqual(m.binLabels([2, 3, 5, 10, 20], 50), ["2倍未満", "2-3倍", "3-5倍", "5-10倍", "10-20倍", "20-50倍"]);
  const { bins, total } = m.binStats(A, [2, 3, 5, 10, 20], 50);
  assert.equal(bins[0].winRoi, 1.5);
  assert.equal(bins[2].win, 1 / 3);
  assert.equal(total.n, 10);
  assert.equal(total.w, 2);
  assert.equal(total.t3, 6);
  assert.equal(total.winRoi, 0.45);
});

test("levelsOf は調教師・場所・パターンの3段。調教なしは調教師だけ", () => {
  const t = { p: { "栗東坂路|A2": A, "栗東坂路|B1": B, "美浦坂路|X": B } };
  const lv = m.levelsOf(t, "栗東坂路|A2");
  assert.equal(lv.trainer[0], 20);
  assert.equal(lv.place[0], 15);
  assert.equal(lv.pattern, A);
  const none = m.levelsOf(t, null);
  assert.equal(none.place, null);
  assert.equal(none.trainer[0], 20);
  assert.equal(m.levelsOf(t, "栗東坂路|Z").pattern, null);
  assert.equal(m.levelsOf(undefined, "x"), null);
  assert.deepEqual(m.splitPat("美浦ウッド(南)|W52.0-53.9"), { place: "美浦ウッド(南)", pat: "W52.0-53.9" });
});
