// 実行: node --test lib/entriesStatus.test.mjs
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync, writeFileSync, mkdtempSync } from "node:fs";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const ts = createRequire(import.meta.url)("typescript");
const out = join(mkdtempSync(join(tmpdir(), "entries-")), "entriesStatus.mjs");
writeFileSync(out, ts.transpileModule(readFileSync(join(here, "entriesStatus.ts"), "utf8"), { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2020 } }).outputText);
const m = await import(pathToFileURL(out).href);

test("phaseMeta / isProvisional", () => {
  assert.equal(m.phaseMeta("registered").label, "特別登録（暫定）");
  assert.equal(m.phaseMeta("final").label, "出馬表確定");
  assert.equal(m.phaseMeta("unknown").label, "出馬表待ち");
  assert.equal(m.isProvisional("registered"), true);
  assert.equal(m.isProvisional("entries"), true);
  assert.equal(m.isProvisional("final"), false);
  assert.equal(m.isProvisional("weighed"), false);
});

test("fmtUpdated は JST で M/D HH:MM", () => {
  assert.equal(m.fmtUpdated("2026-10-23T16:05:00+09:00"), "10/23 16:05");
  assert.equal(m.fmtUpdated("2026-10-23T07:05:00+00:00"), "10/23 16:05");
  assert.equal(m.fmtUpdated(null), "");
  assert.equal(m.fmtUpdated("broken"), "");
});

test("changesSummary", () => {
  assert.equal(m.changesSummary([]), "");
  assert.equal(m.changesSummary([{ type: "removed" }, { type: "removed" }, { type: "jockey_change" }]), "取消 2頭・騎手変更 1件");
  assert.equal(m.changesSummary([{ type: "added" }]), "");
});

test("summarizeDay", () => {
  const s = m.summarizeDay([{ entries_phase: "registered" }, { entries_phase: "final" }, {}, { entries_phase: "registered" }]);
  assert.deepEqual(s.counts, { registered: 2, final: 1, none: 1 });
  assert.equal(s.withPhase, 3);
  assert.equal(s.slowest, "none");
  assert.equal(m.summarizeDay([]).slowest, null);
});
