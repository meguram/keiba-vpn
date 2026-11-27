// 実行: node --test lib/pedigreePersona.test.mjs
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync, writeFileSync, mkdtempSync } from "node:fs";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const ts = createRequire(import.meta.url)("typescript");
const out = join(mkdtempSync(join(tmpdir(), "pp-")), "pp.mjs");
writeFileSync(out, ts.transpileModule(readFileSync(join(here, "pedigreePersona.ts"), "utf8"), { compilerOptions: { module: ts.ModuleKind.ES2020, target: ts.ScriptTarget.ES2020 } }).outputText);
const m = await import(pathToFileURL(out).href);

test("発走の時刻（JST）を過ぎたら欄を出さない", () => {
  const race = { date: "2026-12-06", start_time: "9:50" };
  assert.equal(m.pastStart(race, new Date("2026-12-06T09:49:00+09:00")), false);
  assert.equal(m.pastStart(race, new Date("2026-12-06T09:50:00+09:00")), true);
  assert.equal(m.pastStart({ date: "2026-12-06", start_time: "15:25:00" }, new Date("2026-12-06T15:24:59+09:00")), false);
});

test("発走の時刻が分からなければその日の終わりまで・日付が無ければ出さない", () => {
  assert.equal(m.pastStart({ date: "2026-12-06", start_time: "" }, new Date("2026-12-06T23:58:00+09:00")), false);
  assert.equal(m.pastStart({ date: "2026-12-06" }, new Date("2026-12-07T00:00:00+09:00")), true);
  assert.equal(m.pastStart(null), true);
});

test("一覧の状態の文と部品に、買いを勧める語が無い", () => {
  // Python の rules.BANNED と同じ一覧（tests/research/test_pedigree_persona.py が両方を突き合わせる）
  const banned = ["回収率100%超", "回収率 100% 超", "もうかる", "儲かる", "儲け", "勝てる", "期待値", "妙味", "お宝", "狙い目", "穴", "推奨", "おすすめ",
    "オススメ", "お勧め", "買い", "買える", "注目", "激走", "必ず", "確実"];
  const texts = ["pending", "ready", "none", "unavailable", "not_applicable"].map((s) => m.picksMessage(s) ?? "");
  const src = readFileSync(join(here, "../components/analysis/PedigreePersona.tsx"), "utf8");
  for (const t of [...texts, src]) for (const w of banned) assert.ok(!t.includes(w), `${w} が入った: ${t.slice(0, 40)}`);
});
