// 実行: node --test lib/maintenance.test.mjs
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync, writeFileSync, mkdtempSync } from "node:fs";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const ts = createRequire(import.meta.url)("typescript");
const out = join(mkdtempSync(join(tmpdir(), "maint-")), "maintenance.mjs");
writeFileSync(out, ts.transpileModule(readFileSync(join(here, "maintenance.ts"), "utf8"), { compilerOptions: { module: ts.ModuleKind.ES2020, target: ts.ScriptTarget.ES2020 } }).outputText);
const m = await import(pathToFileURL(out).href);

const active = (rem) => ({ ...m.NORMAL_STATUS, enabled: true, active: true, phase: "active", remaining_sec: rem, ends_at_label: "04:00", window_start: "2026-10-22T03:00:00+09:00", window_end: "2026-10-22T04:00:00+09:00" });
const notice = (sec) => ({ ...m.NORMAL_STATUS, enabled: true, phase: "notice", starts_in_sec: sec, window_start: "2026-10-22T03:00:00+09:00", window_end: "2026-10-22T04:00:00+09:00", ends_at_label: "04:00" });

test("formatCountdown: 1 時間以上は HH:MM:SS、未満は MM:SS、null は --:--", () => {
  assert.equal(m.formatCountdown(3600), "01:00:00");
  assert.equal(m.formatCountdown(3599), "59:59");
  assert.equal(m.formatCountdown(65), "01:05");
  assert.equal(m.formatCountdown(0), "00:00");
  assert.equal(m.formatCountdown(-5), "00:00");
  assert.equal(m.formatCountdown(null), "--:--");
});

test("secondsLeft: 受信時刻からの経過で数え、端末の時計のずれに影響されない（0 で止まる）", () => {
  assert.equal(m.secondsLeft(3000, 1_000_000, 1_000_000), 3000);
  assert.equal(m.secondsLeft(3000, 1_000_000, 1_010_500), 2990); // 10.5 秒後 → 切り上げで 2990
  assert.equal(m.secondsLeft(10, 0, 20_000), 0);
  assert.equal(m.secondsLeft(null, 0, 1), null);
});

test("livePhase: 事前告知→開催中→終了の境界を、受信後の経過で先取りする", () => {
  const n = { status: notice(120), receivedAt: 0 };
  assert.equal(m.livePhase(n, 60_000), "notice");
  assert.equal(m.livePhase(n, 120_000), "active");
  const a = { status: active(30), receivedAt: 0 };
  assert.equal(m.livePhase(a, 29_000), "active");
  assert.equal(m.livePhase(a, 30_000), "normal");
  assert.equal(m.livePhase({ status: m.NORMAL_STATUS, receivedAt: 0 }, 1e9), "normal");
});

test("approxMinutes / hhmm / windowLabel", () => {
  assert.equal(m.approxMinutes(1800), "あと約30分");
  assert.equal(m.approxMinutes(61), "あと約2分");
  assert.equal(m.approxMinutes(60), "まもなく");
  assert.equal(m.approxMinutes(null), "");
  assert.equal(m.hhmm("2026-10-22T03:00:00+09:00"), "03:00");
  assert.equal(m.hhmm(null), "");
  assert.equal(m.windowLabel(active(100)), "03:00〜04:00");
});

test("parseStatus: API の応答・503 の本文・不正な値", () => {
  const s = m.parseStatus({ enabled: true, active: true, phase: "active", remaining_sec: 100, ends_at_label: "04:00", message: "x", retry_after: 100 });
  assert.equal(s.active, true);
  assert.equal(s.remaining_sec, 100);
  const b = m.parseStatus({ maintenance: true, message: "m", ends_at_label: "04:00", remaining_sec: 50 });
  assert.equal(b.active, true);
  assert.equal(b.phase, "active");
  assert.equal(b.retry_after, 50);
  assert.deepEqual(m.parseStatus(null), m.NORMAL_STATUS);
  assert.deepEqual(m.parseStatus("x"), m.NORMAL_STATUS);
  assert.equal(m.parseStatus({ phase: "weird", remaining_sec: "3" }).phase, "normal");
  assert.equal(m.parseStatus({ phase: "weird", remaining_sec: "3" }).remaining_sec, null);
});

test("isMaintenanceResponse: 503 かつ X-Keiba-Maintenance:1 のときだけ", () => {
  const mk = (status, hv) => ({ status, headers: { get: (k) => (k === "x-keiba-maintenance" ? hv : null) } });
  assert.equal(m.isMaintenanceResponse(mk(503, "1")), true);
  assert.equal(m.isMaintenanceResponse(mk(503, null)), false);
  assert.equal(m.isMaintenanceResponse(mk(500, "1")), false);
});

test("previewStatus（dev のプレビュー）: 種類と残り秒の範囲", () => {
  assert.equal(m.previewStatus(null), null);
  assert.equal(m.previewStatus("x"), null);
  const a = m.previewStatus("active", "99999");
  assert.equal(a.active, true);
  assert.equal(a.remaining_sec, 7200); // 上限
  assert.equal(m.previewStatus("active").remaining_sec, 2400);
  const n = m.previewStatus("notice", "300");
  assert.equal(n.phase, "notice");
  assert.equal(n.starts_in_sec, 300);
  assert.equal(m.previewStatus("ended").phase, "normal");
});

test("safeNext: 同一オリジンの相対パスだけ（オープンリダイレクト防止、/maintenance へは戻さない）", () => {
  assert.equal(m.safeNext("/weekly-predictions?d=1"), "/weekly-predictions?d=1");
  assert.equal(m.safeNext("https://evil.example/"), "/");
  assert.equal(m.safeNext("//evil.example"), "/");
  assert.equal(m.safeNext("/\\evil"), "/");
  assert.equal(m.safeNext("/maintenance"), "/");
  assert.equal(m.safeNext(null), "/");
});
