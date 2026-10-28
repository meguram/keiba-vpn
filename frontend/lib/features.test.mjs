// 実行: node --test lib/features.test.mjs
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync, writeFileSync, mkdtempSync } from "node:fs";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const ts = createRequire(import.meta.url)("typescript");
const src = ts.transpileModule(readFileSync(join(here, "features.ts"), "utf8"), { compilerOptions: { module: ts.ModuleKind.ES2020, target: ts.ScriptTarget.ES2020 } }).outputText;

// process.env はモジュールの読み込み時に読まれるので、条件ごとに別ファイルとして読み込む
let n = 0;
async function load(env) {
  const saved = { ...process.env };
  Object.assign(process.env, env);
  for (const k of Object.keys(env)) if (env[k] === undefined) delete process.env[k];
  const out = join(mkdtempSync(join(tmpdir(), "features-")), `features${n++}.mjs`);
  writeFileSync(out, src);
  try {
    return await import(pathToFileURL(out).href);
  } finally {
    process.env = saved;
  }
}

test("開発中の画面は next dev では表示される", async () => {
  const m = await load({ NODE_ENV: "development", NEXT_PUBLIC_ENABLE_FEATURES: undefined });
  assert.equal(m.featureEnabled("megu-analysis"), true);
  assert.equal(m.featureEnabled("megu-column"), true);
  assert.equal(m.pathEnabled("/column/category/gate"), true);
});

test("prod ビルドでは、全機能を有効にする指定があっても開発中の画面は出さない", async () => {
  const m = await load({ NODE_ENV: "production", NEXT_PUBLIC_ENV_LABEL: "", NEXT_PUBLIC_ENABLE_FEATURES: "all" });
  assert.equal(m.SHOW_DEV_ONLY, false);
  assert.equal(m.featureEnabled("megu-analysis"), false);
  assert.equal(m.featureEnabled("megu-column"), false);
  assert.equal(m.pathEnabled("/analysis"), false);
  assert.equal(m.pathEnabled("/column/sample-gate-inner"), false);
  assert.equal(m.featureEnabled("predictions"), true);     // 公開中の機能は変わらない
  assert.equal(m.featureEnabled("megu-index"), true);      // all 指定は開発中でない機能には効く
});

test("stg ビルドでは開発中の画面を表示する（prod と違い STG ラベルで判定）", async () => {
  const m = await load({ NODE_ENV: "production", NEXT_PUBLIC_ENV_LABEL: "STG", NEXT_PUBLIC_ENABLE_FEATURES: undefined });
  assert.equal(m.SHOW_DEV_ONLY, true);
  assert.equal(m.featureEnabled("megu-analysis"), true);
  assert.equal(m.featureEnabled("megu-column"), true);
  assert.equal(m.pathEnabled("/column/sample-gate-inner"), true);
  assert.equal(m.featureEnabled("megu-index"), false);     // 他の無効機能は stg でも無効のまま
});
