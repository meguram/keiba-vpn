// 実行: node --test lib/adsense.test.mjs   （typescript で変換して読み込むので、追加の依存は要らない）
import test from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { readFileSync, writeFileSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const ts = createRequire(import.meta.url)("typescript");
const out = join(mkdtempSync(join(tmpdir(), "adsense-")), "adsense.mjs");
writeFileSync(
  out,
  ts.transpileModule(readFileSync(join(here, "adsense.ts"), "utf8"), { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2020 } }).outputText,
);
const { shouldLoadAdsense, ADSENSE_CLIENT, ADSENSE_SRC } = await import(pathToFileURL(out).href);

const anon = { loading: false, isMember: false, isAdmin: false };
const free = { loading: false, isMember: false, isAdmin: false }; // ログイン済みでも is_member は false
const paid = { loading: false, isMember: true, isAdmin: false };
const dev = { loading: false, isMember: true, isAdmin: true };

test("匿名・無料会員は本番と stg で読む", () => {
  for (const h of ["app.megu-keiba.com", "stg.megu-keiba.com", "APP.megu-keiba.com"]) {
    assert.equal(shouldLoadAdsense(anon, h), true);
    assert.equal(shouldLoadAdsense(free, h), true);
  }
});
test("有料会員・開発者には読まない", () => {
  assert.equal(shouldLoadAdsense(paid, "app.megu-keiba.com"), false);
  assert.equal(shouldLoadAdsense(dev, "app.megu-keiba.com"), false);
  assert.equal(shouldLoadAdsense({ ...anon, isAdmin: true }, "app.megu-keiba.com"), false);
});
test("区分の確認中・失敗は読まない", () => {
  assert.equal(shouldLoadAdsense({ ...anon, loading: true }, "app.megu-keiba.com"), false);
  assert.equal(shouldLoadAdsense({ ...anon, failed: true }, "app.megu-keiba.com"), false);
});
test("手元・別のホストでは読まない", () => {
  for (const h of ["localhost", "127.0.0.1", "app-new.megu-keiba.com", "megu-keiba.com", ""]) assert.equal(shouldLoadAdsense(anon, h), false);
});
test("止める印 off で誰にも読まない", () => {
  assert.equal(shouldLoadAdsense(anon, "app.megu-keiba.com", "off"), false);
  assert.equal(shouldLoadAdsense(anon, "app.megu-keiba.com", " OFF "), false);
  assert.equal(shouldLoadAdsense(anon, "app.megu-keiba.com", ""), true);
});
test("ads.txt とパブリッシャー ID が同じ", () => {
  const line = readFileSync(join(here, "..", "public", "ads.txt"), "utf8").trim();
  assert.equal(line, `google.com, ${ADSENSE_CLIENT.replace(/^ca-/, "")}, DIRECT, f08c47fec0942fa0`);
  assert.ok(ADSENSE_SRC.endsWith(`client=${ADSENSE_CLIENT}`));
});
