import test from "node:test";
import assert from "node:assert/strict";
import { writeFileSync, readFileSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import ts from "typescript";

const src = readFileSync(new URL("./raceGrade.ts", import.meta.url), "utf8");
const out = ts.transpileModule(src, { compilerOptions: { module: ts.ModuleKind.ES2020, target: ts.ScriptTarget.ES2020 } }).outputText;
const file = join(mkdtempSync(join(tmpdir(), "rg-")), "rg.mjs");
writeFileSync(file, out);
const m = await import(file);

test("gradeOf は一覧の格のアイコンを優先し、無ければ grade を読む", () => {
  assert.equal(m.gradeOf({ list_grade_icon: "Icon_GradeType1", grade: "OP" }), "G1");   // 出馬表の格（菊花賞が OP など）より一覧のアイコン
  assert.equal(m.gradeOf({ list_grade_icon: "Icon_GradeType3" }), "G3");
  assert.equal(m.gradeOf({ list_grade_icon: "Icon_GradeType15" }), "L");
  assert.equal(m.gradeOf({ list_grade_icon: "Icon_GradeType16" }), null);               // 3 勝クラスは印なし
  assert.equal(m.gradeOf({ grade: "G2" }), "G2");
  assert.equal(m.gradeOf({ grade: "OP", race_name: "菊花賞" }), null);                   // DB の OP は当てにしない（アイコンが無ければ印なし）
  assert.equal(m.gradeOf({ grade: "OP", race_name: "第86回菊花賞(G1)" }), "G1");         // レース名の (G1) で補う
  assert.equal(m.gradeOf({ race_name: "東京ハイジャンプ(J.G2)" }), "J.G2");
  assert.equal(m.gradeOf({ grade: "JG1" }), "J.G1");
  assert.equal(m.gradeOf({}), null);
  assert.ok(m.isGraded("G1") && !m.isGraded("L") && !m.isGraded("OP") && !m.isGraded(null));
});

test("shortName・startTimeOf", () => {
  assert.equal(m.shortName("第43回マイルCS(G1)"), "マイルCS");
  assert.equal(m.shortName("東京スポーツ杯2歳ステークス", 6), "東京スポー…");
  assert.equal(m.startTimeOf({ start_time: "9:50" }), "09:50");
  assert.equal(m.startTimeOf({ start_time: "15:40:00" }), "15:40");
  assert.equal(m.startTimeOf({ start_time: null }), null);
});

test("sameRoundIn は同じ番号、無ければ一番近い番号（同じ距離なら小さい方）、選べないレースは飛ばす", () => {
  const R = [
    { race_id: "a11", venue: "東京", round: 11 }, { race_id: "b9", venue: "京都", round: 9 }, { race_id: "b11", venue: "京都", round: 11 },
    { race_id: "b12", venue: "京都", round: 12 }, { race_id: "c10", venue: "福島", round: 10 }, { race_id: "c12", venue: "福島", round: 12 },
  ];
  assert.equal(m.sameRoundIn(R, "京都", 11).race_id, "b11");
  assert.equal(m.sameRoundIn(R, "福島", 11).race_id, "c10");
  assert.equal(m.sameRoundIn(R, "京都", 11, new Set(["b11"])).race_id, "b12");   // 9 と 12 では 12 が近い
  assert.equal(m.sameRoundIn(R, "京都", null).race_id, "b9");
  assert.equal(m.sameRoundIn(R, "中山", 11), undefined);
});
