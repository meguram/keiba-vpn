// 実行: node --test lib/weekDates.test.mjs   （typescript で変換して読み込むので、追加の依存は要らない）
import test from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { readFileSync, writeFileSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const ts = createRequire(import.meta.url)("typescript");
const out = join(mkdtempSync(join(tmpdir(), "weekDates-")), "weekDates.mjs");
writeFileSync(out, ts.transpileModule(readFileSync(join(here, "weekDates.ts"), "utf8"), { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2020 } }).outputText);
const W = await import(pathToFileURL(out).href);

test("曜日・月曜・日付加算", () => {
  assert.equal(W.weekdayOf("20261017"), 6);   // 土
  assert.equal(W.weekdayOf("20261018"), 0);   // 日
  assert.equal(W.mondayOf("20261018"), "20261012");
  assert.equal(W.mondayOf("20261012"), "20261012");
  assert.equal(W.addDays("20261031", 1), "20261101");
  assert.equal(W.addDays("20260101", -1), "20251231");
});

test("今日(JST)は UTC の 15 時以降で翌日になる", () => {
  assert.equal(W.todayJst(new Date("2026-10-16T14:59:00Z")), "20261016");
  assert.equal(W.todayJst(new Date("2026-10-16T15:00:00Z")), "20261017");
});

test("開催ごとにまとめる（土曜がキー。月曜の祝日開催は前の土日と同じ開催）", () => {
  const w = W.groupWeeks(["20261018", "20261011", "20261010", "20261012", "20261017", "20261010"]);
  assert.deepEqual(w.map((x) => x.key), ["20261010", "20261017"]);
  assert.deepEqual(w[0].dates, ["20261010", "20261011", "20261012"]);   // 月祝 10/12 は 10/10-11 と同じ開催
  assert.deepEqual(w[1].dates, ["20261017", "20261018"]);
});

test("開催のキー: 土=自分, 日=前日, 月=2日前, 火=3日前, 水〜金=次の土曜", () => {
  assert.deepEqual(["20261017", "20261018", "20261019", "20261020", "20261021", "20261023"].map(W.meetingKeyOf),
    ["20261017", "20261017", "20261017", "20261017", "20261024", "20261024"]);
});

const DATES = ["20261018", "20261017", "20261012", "20261011", "20261010", "20261004", "20261003"];

test("金曜: 次回=今週末(今週)、前回=先週末(先週)の2回", () => {
  const p = W.pickTwoWeeks(DATES, "20261016");
  assert.equal(p.length, 2);
  assert.deepEqual([p[0].role, p[1].role], ["prev", "next"]);
  assert.deepEqual(p[0].week.dates, ["20261010", "20261011", "20261012"]);
  assert.deepEqual(p[1].week.dates, ["20261017", "20261018"]);
  assert.deepEqual([p[0].rel, p[1].rel], ["先週", "今週"]);
});

test("今日が開催日なら、その開催が次回(今週)", () => {
  const p = W.pickTwoWeeks(DATES, "20261017");
  assert.equal(p[1].week.key, "20261017");
  assert.equal(p[1].rel, "今週");
});

test("月曜の祝日開催の最終日(10/12)も、その開催が次回(開催中=今週)", () => {
  const p = W.pickTwoWeeks(DATES, "20261012");
  assert.equal(p[1].week.key, "20261010");
  assert.equal(p[1].rel, "今週");
  assert.equal(p[0].week.key, "20261003");
});

test("今日以降の開催が無ければ最後の2回", () => {
  const p = W.pickTwoWeeks(DATES, "20261105");
  assert.equal(p.length, 2);
  assert.equal(p[1].week.key, "20261017");
  assert.equal(p[0].week.key, "20261010");
});

test("開催が1回だけなら1回だけ（開催なしの週は出さない）", () => {
  assert.equal(W.pickTwoWeeks(["20260906"], "20261016").length, 1);
  assert.deepEqual(W.pickTwoWeeks([], "20261016"), []);
});

test("先週/今週/来週の表記", () => {
  const p = W.pickTwoWeeks(["20261010", "20261011", "20261024"], "20261016");
  assert.deepEqual(p.map((x) => x.rel), ["先週", "来週"]);      // 10/24 の開催は来週
});

test("既定の日: 今日がその週なら今日以降の最初の開催日、そうでなければ週の最初", () => {
  const wk = { key: "20261017", dates: ["20261017", "20261018"] };
  assert.equal(W.firstDateOf(wk, "20261016"), "20261017");    // 開催前の金曜 → 最初の日
  assert.equal(W.firstDateOf(wk, "20261018"), "20261018");    // 開催中 → 今日
  assert.equal(W.firstDateOf(wk, "20261001"), "20261017");
  assert.equal(W.firstDateOf({ key: "20261010", dates: ["20261010", "20261011", "20261012"] }, "20261016"), "20261010");   // 過去の開催 → 最初の日
});

test("土=青・日=赤・平日開催=祝(赤)", () => {
  assert.deepEqual([W.dayKind("20261017", "20261016").kind, W.dayKind("20261017", "20261016").color], ["sat", "#60a5fa"]);
  assert.equal(W.dayKind("20261018", "20261016").color, "#f87171");
  const h = W.dayKind("20261012", "20261016");
  assert.equal(h.holiday, true); assert.equal(h.label, "月"); assert.equal(h.color, "#f87171");
  assert.equal(W.dayKind("20261016", "20261016").isToday, true);
});

test("表記: 10/10(土)〜12(月) / 月またぎ / 1日だけ", () => {
  assert.equal(W.formatRange(["20261010", "20261011", "20261012"]), "10/10(土)〜12(月)");
  assert.equal(W.formatRange(["20260926", "20261004"]), "9/26(土)〜10/4(日)");
  assert.equal(W.formatRange(["20260906"]), "9/6(日)");
});

test("重賞だけを開催場コード順・R 順に抜く。『第N回』『(G3)』は短縮名から外す", () => {
  const races = [
    { race_id: "202608040511", race_name: "第12回サウジアラビアRC", grade: "G3", venue: "東京", round: 11 },
    { race_id: "202608040507", race_name: "2歳未勝利", grade: "未勝利", venue: "東京", round: 7 },
    { race_id: "202608040611", race_name: "第69回MBS賞スワンS", grade: "G2", venue: "京都", round: 11 },
    { race_id: "202605040111", race_name: "紫苑S(G2)", grade: "G2", venue: "中山", round: 11 },
  ];
  const g = W.pickGradeRaces(races);
  assert.deepEqual(g.map((x) => x.name), ["紫苑S", "サウジアラビアRC", "MBS賞スワンS"]);
  assert.equal(g[1].fullName, "第12回サウジアラビアRC");
  assert.deepEqual(W.venuesOf(races), ["中山", "東京", "京都"]);
});


test("pickPastWeeks: 未来の開催を含めず、今日以前の最新 2 開催を返す（今日が開催日ならその開催を含む）", () => {
  const dates = ["20261003", "20261004", "20261010", "20261011", "20261012", "20261017", "20261018", "20261024", "20261025"];
  const got = W.pickPastWeeks(dates, "20261019");
  assert.deepEqual(got.map((p) => p.week.key), ["20261010", "20261017"]);
  assert.ok(got.every((p) => p.role === "prev" && p.week.dates.every((d) => d <= "20261019")));
  assert.deepEqual(W.pickPastWeeks(dates, "20261018").map((p) => p.week.dates).pop(), ["20261017", "20261018"]);
  assert.deepEqual(W.pickPastWeeks(["20261024"], "20261019"), []);
});
