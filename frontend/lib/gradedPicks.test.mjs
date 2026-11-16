import test from "node:test";
import assert from "node:assert/strict";
import { writeFileSync, readFileSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import ts from "typescript";

const src = readFileSync(new URL("./gradedPicks.ts", import.meta.url), "utf8");
const out = ts.transpileModule(src, { compilerOptions: { module: ts.ModuleKind.ES2020, target: ts.ScriptTarget.ES2020 } }).outputText;
const file = join(mkdtempSync(join(tmpdir(), "gp-")), "gp.mjs");
writeFileSync(file, out);
const m = await import(file);

test("pct・rec・odds・yen・fin は欠けを「—」にする", () => {
  assert.equal(m.pct(1.9), "190.0%");
  assert.equal(m.pct(null), "—");
  assert.equal(m.rec([1, 0, 0, 1]), "1-0-0-1");
  assert.equal(m.rec(null), "—");
  assert.equal(m.odds({ odds: 3.8 }), "3.8");
  assert.equal(m.odds({ odds: null }), "—");
  assert.equal(m.yen(1200), "1,200");
  assert.equal(m.yen(null), "—");
  assert.equal(m.fin({ fin: 14 }), "14");
});

test("killedTop3 は 3 着以内の消し馬を数える", () => {
  assert.equal(m.killedTop3([{ fin: 3 }, { fin: 4 }, { fin: 1 }]), 2);
  assert.equal(m.killedTop3(null), 0);
});

test("periodsOf は新しい期間から、値のある期間だけ", () => {
  assert.deepEqual(m.periodsOf({ dev: {}, eval: {} }), ["eval", "dev"]);
  assert.deepEqual(m.periodsOf({}), []);
  assert.ok(m.PERIOD_LABEL.dev.includes("当てはめ"));
});

test("horseShort は馬番・馬名と、あれば着順", () => {
  assert.equal(m.horseShort({ hn: 14, name: "マジョレルブルー", fin: 7 }), "14 マジョレルブルー（7着）");
  assert.equal(m.horseShort({ hn: 3, name: null }), "3 （馬名なし）");
});
