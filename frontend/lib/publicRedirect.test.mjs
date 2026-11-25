// 実行: node --test lib/publicRedirect.test.mjs
// middleware の転送先がプロキシの内側のホスト名（localhost:3000）にならないこと（2026-11-25 の不具合）。
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync, writeFileSync, mkdtempSync } from "node:fs";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const ts = createRequire(import.meta.url)("typescript");
const out = join(mkdtempSync(join(tmpdir(), "pubredir-")), "publicRedirect.mjs");
writeFileSync(out, ts.transpileModule(readFileSync(join(here, "publicRedirect.ts"), "utf8"), { compilerOptions: { module: ts.ModuleKind.ES2020, target: ts.ScriptTarget.ES2020 } }).outputText);
const { publicRedirectUrl } = await import(pathToFileURL(out).href);
const H = (o) => new Headers(o);

test("nginx の後ろでは公開のホスト名と https を使う", () => {
  const u = publicRedirectUrl(H({ host: "app.megu-keiba.com", "x-forwarded-proto": "https" }), "https://localhost:3000", "/");
  assert.equal(u.toString(), "https://app.megu-keiba.com/");
});

test("クエリを保つ・stg のホスト", () => {
  const u = publicRedirectUrl(H({ host: "stg.megu-keiba.com", "x-forwarded-proto": "https" }), "http://localhost:3000", "/login?next=%2Fcolumn");
  assert.equal(u.toString(), "https://stg.megu-keiba.com/login?next=%2Fcolumn");
});

test("偽装した X-Forwarded-Host は採らない", () => {
  const u = publicRedirectUrl(H({ host: "app.megu-keiba.com", "x-forwarded-host": "evil.example", "x-forwarded-proto": "https" }), "http://localhost:3000", "/");
  assert.equal(u.toString(), "https://app.megu-keiba.com/");
});

test("転送ヘッダーが無い・許可していないホスト・おかしい値は元のオリジン", () => {
  const F = "http://localhost:3000";
  assert.equal(publicRedirectUrl(H({ host: "localhost:3000" }), F, "/").toString(), "http://localhost:3000/");
  assert.equal(publicRedirectUrl(H({ host: "evil.example", "x-forwarded-proto": "https" }), F, "/").toString(), "http://localhost:3000/");
  assert.equal(publicRedirectUrl(H({ host: "megu-keiba.com.evil.example", "x-forwarded-proto": "https" }), F, "/").toString(), "http://localhost:3000/");
  assert.equal(publicRedirectUrl(H({ host: "app.megu-keiba.com", "x-forwarded-proto": "javascript" }), F, "/").toString(), "http://localhost:3000/");
});
