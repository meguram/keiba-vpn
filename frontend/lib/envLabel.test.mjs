// 実行: node --test lib/envLabel.test.mjs   （typescript で変換して読み込むので、追加の依存は要らない）
import test from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { readFileSync, writeFileSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const ts = createRequire(import.meta.url)("typescript");
const out = join(mkdtempSync(join(tmpdir(), "envLabel-")), "envLabel.mjs");
writeFileSync(
  out,
  ts.transpileModule(readFileSync(join(here, "envLabel.ts"), "utf8"), { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2020 } }).outputText,
);
const { envBadgeLabel } = await import(pathToFileURL(out).href);

test("prod / 空 / 未設定 は非表示", () => {
  for (const v of ["", " ", "prod", "PROD", "Production", undefined, null]) assert.equal(envBadgeLabel(v), "");
});
test("stg は大文字で出る", () => {
  assert.equal(envBadgeLabel("stg"), "STG");
  assert.equal(envBadgeLabel(" Stg "), "STG");
});
test("長い文字は 8 文字まで", () => {
  assert.equal(envBadgeLabel("staging-environment"), "STAGING-");
});
