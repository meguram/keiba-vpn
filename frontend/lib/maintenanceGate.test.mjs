// 実行: node --test lib/maintenanceGate.test.mjs
// middleware の保守の確認を待たせない入れ物（docs/operations/latency_plan_20261118.md の L2a）。
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync, writeFileSync, mkdtempSync } from "node:fs";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const ts = createRequire(import.meta.url)("typescript");
const out = join(mkdtempSync(join(tmpdir(), "maintgate-")), "maintenanceGate.mjs");
writeFileSync(out, ts.transpileModule(readFileSync(join(here, "maintenanceGate.ts"), "utf8"), { compilerOptions: { module: ts.ModuleKind.ES2020, target: ts.ScriptTarget.ES2020 } }).outputText);
const m = await import(pathToFileURL(out).href);

/** 応答を順に返す偽の取得と、進められる時計 */
function harness(responses, capMs) {
  let t = 1_000_000;
  let calls = 0;
  const q = responses.slice();
  const gate = m.createMaintenanceGate(() => {
    calls++;
    const r = q.shift();
    if (r === "hang") return new Promise(() => {});
    if (r === "throw") throw new Error("sync");
    return r instanceof Error ? Promise.reject(r) : Promise.resolve(r);
  }, () => t, capMs);
  return { gate, adv: (ms) => { t += ms; }, calls: () => calls };
}

test("起動直後（値なし）は待つ。期限内は取り直さない", async () => {
  const h = harness([{ active: false }]);
  const r1 = h.gate.read();
  assert.equal(r1.mustWait, true);
  assert.equal(r1.state, null);
  await r1.pending;
  h.adv(1000);
  const r2 = h.gate.read();
  assert.equal(r2.mustWait, false);
  assert.equal(r2.pending, null);
  assert.equal(r2.state.active, false);
  assert.equal(h.calls(), 1);
});

test("期限切れでも猶予内の「保守ではない」は待たない。取り直しは 1 本だけ。済んだら新しい値", async () => {
  const h = harness([{ active: false }, { active: true, remaining_sec: 600, retry_after: 120 }]);
  await h.gate.read().pending;
  h.adv(16_000);
  const a = h.gate.read();
  const b = h.gate.read();
  assert.equal(a.mustWait, false);
  assert.equal(a.state.active, false);
  assert.ok(a.pending);
  assert.equal(a.pending, b.pending);
  assert.equal(h.calls(), 2);
  await a.pending;
  const c = h.gate.read();
  assert.equal(c.state.active, true);
  assert.equal(c.state.retry, 120);
});

test("猶予（期限 + 45 秒）より古い「保守ではない」は待つ（夜に要求が途切れたプロセス）", async () => {
  const h = harness([{ active: false }, { active: true, remaining_sec: 1800 }]);
  await h.gate.read().pending;
  h.adv(m.NORMAL_TTL_MS + m.STALE_GRACE_MS + 1);
  const r = h.gate.read();
  assert.equal(r.mustWait, true);
  assert.equal((await r.pending).active, true);
});

test("期限切れの値が「保守中」なら待つ（終わりを見落とさない）", async () => {
  const h = harness([{ active: true, remaining_sec: 5 }, { active: false }]);
  await h.gate.read().pending;
  h.adv(6000);
  const r = h.gate.read();
  assert.equal(r.mustWait, true);
  assert.equal((await r.pending).active, false);
});

test("取得の失敗（429 など）は前の値を 5 秒だけ使う。前の値が無ければ「保守ではない」", async () => {
  const h = harness([{ active: true, remaining_sec: 600 }, new Error("429")]);
  await h.gate.read().pending;
  h.adv(16_000);
  const st = await h.gate.read().pending;
  assert.equal(st.active, true);
  assert.equal(st.ttl, m.FAIL_TTL_MS);
  const h2 = harness([new Error("502")]);
  const st2 = await h2.gate.read().pending;
  assert.equal(st2.active, false);
  assert.equal(st2.ttl, m.FAIL_TTL_MS);
});

test("失敗が続いても、保守の終わりの時刻を過ぎたら「保守ではない」に倒す（保守画面が出続けない）", async () => {
  const h = harness([{ active: true, remaining_sec: 20 }, new Error("429"), new Error("timeout")]);
  await h.gate.read().pending;
  h.adv(16_000);
  assert.equal((await h.gate.read().pending).active, true);   // 終わりの前の失敗: 前の値
  h.adv(m.FAIL_TTL_MS);
  assert.equal((await h.gate.read().pending).active, false);  // 終わりを過ぎた失敗: 倒す
});

test("終わりの時刻が分かる保守中は、429 が 60 秒を超えて続いても終わりまで保守のまま。延ばす期限は終わりで縮む", async () => {
  const h = harness([{ active: true, remaining_sec: 120 }, ...Array(30).fill(new Error("429"))]);
  await h.gate.read().pending;
  let st;
  for (let i = 0; i < 20; i++) { h.adv(5_000); st = await h.gate.read().pending; }   // 100 秒後も
  assert.equal(st.active, true);
  h.adv(17_000);                       // 終わりの 3 秒前
  st = await h.gate.read().pending;
  assert.equal(st.active, true);
  assert.equal(st.ttl, 3000);
  h.adv(3_000);
  assert.equal((await h.gate.read().pending).active, false);
});

test("終わりの時刻が分からない保守中の値は、最後の成功から 60 秒で前の値を使わない", async () => {
  const h = harness([{ active: true }, new Error("a"), new Error("b"), new Error("c")]);
  await h.gate.read().pending;
  h.adv(16_000);
  assert.equal((await h.gate.read().pending).active, true);
  h.adv(30_000);
  assert.equal((await h.gate.read().pending).active, true);
  h.adv(20_000);
  assert.equal((await h.gate.read().pending).active, false);
});

test("猶予の境目: 期限 + 45 秒の 1ms 前は待たず、過ぎたら待つ", () => {
  const h = harness([{ active: false }]);
  return h.gate.read().pending.then(() => {
    h.adv(m.NORMAL_TTL_MS + m.STALE_GRACE_MS - 1);
    assert.equal(h.gate.read().mustWait, false);
  });
});

test("保守中に不正な応答（null）が返っても、保守は解けない（失敗の道で前の値）", async () => {
  const h = harness([{ active: true, remaining_sec: 600 }, null]);
  await h.gate.read().pending;
  h.adv(16_000);
  assert.equal((await h.gate.read().pending).active, true);
});

test("前の値が「保守ではない」ときの失敗は「保守ではない」のまま、待たずに答え続ける（Cloud Run が冷えている間）", async () => {
  const h = harness([{ active: false }, new Error("a"), new Error("b"), new Error("c")]);
  await h.gate.read().pending;
  for (let i = 0; i < 3; i++) {
    h.adv(30_000);
    const r = h.gate.read();
    assert.equal(r.mustWait, false);   // 期限 + 猶予の内。失敗の値は失敗のたびに数え直すので、待たずに答え続ける
    assert.equal((await r.pending).active, false);
  }
  assert.equal(h.calls(), 4);
});

test("失敗の後は取り直しが解放され、次の期限切れで再び取りに行く", async () => {
  const h = harness([{ active: false }, new Error("x"), { active: false }]);
  await h.gate.read().pending;
  h.adv(16_000);
  const st = await h.gate.read().pending;
  assert.equal(st.active, false);
  h.adv(m.FAIL_TTL_MS + 1);
  await h.gate.read().pending;
  assert.equal(h.calls(), 3);
});

test("取得が終わらない・同期で例外・不正な応答でも、取り直しが残り続けない", async () => {
  const h = harness(["hang", "throw", null], 20);
  const st = await h.gate.read().pending;      // hang → 打ち切り
  assert.equal(st.active, false);
  h.adv(m.FAIL_TTL_MS + 1);
  assert.equal((await h.gate.read().pending).active, false);  // throw
  h.adv(m.FAIL_TTL_MS + 1);
  assert.equal((await h.gate.read().pending).active, false);  // null の応答
  assert.equal(h.calls(), 3);
});

test("保守中の期限は残り秒で縮む（1〜15 秒）", async () => {
  const h = harness([{ active: true, remaining_sec: 3 }]);
  assert.equal((await h.gate.read().pending).ttl, 3000);
});
