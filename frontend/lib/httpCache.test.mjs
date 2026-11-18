// 実行: node --test lib/httpCache.test.mjs
// 前回の内容を先に出すための保存（sessionStorage）の扱い（docs/operations/latency_plan_20261118.md の L1）。
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync, writeFileSync, mkdtempSync } from "node:fs";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const ts = createRequire(import.meta.url)("typescript");
const src = ts.transpileModule(readFileSync(join(here, "httpCache.ts"), "utf8"), { compilerOptions: { module: ts.ModuleKind.ES2020, target: ts.ScriptTarget.ES2020 } }).outputText;

/** 偽の sessionStorage（容量の上限つき。文字数で数える） */
class FakeStorage {
  constructor(limit = Infinity) { this.m = new Map(); this.limit = limit; }
  get length() { return this.m.size; }
  key(i) { return [...this.m.keys()][i] ?? null; }
  getItem(k) { return this.m.has(k) ? this.m.get(k) : null; }
  setItem(k, v) {
    const used = [...this.m.entries()].reduce((n, [kk, vv]) => n + (kk === k ? 0 : kk.length + vv.length), 0);
    if (used + k.length + String(v).length > this.limit) throw new Error("QuotaExceededError");
    this.m.set(k, String(v));
  }
  removeItem(k) { this.m.delete(k); }
}
// Object.keys(sessionStorage) で項目の鍵が出るように、Proxy で包む
function storageProxy(fs) {
  return new Proxy(fs, {
    ownKeys: (t) => [...t.m.keys()],
    getOwnPropertyDescriptor: (t, k) => (t.m.has(k) ? { enumerable: true, configurable: true, value: t.m.get(k) } : undefined),
    get: (t, k) => (typeof t[k] === "function" ? t[k].bind(t) : t[k]),
  });
}

let n = 0;
/** 新しいモジュールの実体（メモリのキャッシュ・tierConfirmed を毎回まっさらにする）と、偽の fetch */
async function fresh({ limit = Infinity, store } = {}) {
  const fs = store ?? new FakeStorage(limit);
  globalThis.window = { sessionStorage: storageProxy(fs) };
  const responses = new Map();
  let calls = 0;
  globalThis.fetch = (url) => {
    calls++;
    const r = responses.get(url);
    if (r instanceof Error) return Promise.reject(r);
    const body = typeof r === "function" ? r() : r;
    return Promise.resolve({ ok: body !== undefined, status: body !== undefined ? 200 : 500, json: async () => body });
  };
  const out = join(mkdtempSync(join(tmpdir(), "httpCache-")), `httpCache${n++}.mjs`);
  writeFileSync(out, src);
  const m = await import(pathToFileURL(out).href);
  return { m, fs, responses, calls: () => calls };
}

const flush = () => new Promise((r) => setTimeout(r, 0));

test("区分を確かめる前（再読み込みの直後）は、前の区分の記録があっても前回の内容を返さない", async () => {
  const a = await fresh();
  a.m.setStoredTier("member");
  a.responses.set("/api/x", { v: 1 });
  await a.m.getJson("/api/x", { persist: true });
  assert.deepEqual(a.m.readStored("/api/x"), { v: 1 });
  // 同じタブで再読み込み（モジュールは新しく、sessionStorage は残る）
  const b = await fresh({ store: a.fs });
  assert.equal(b.m.readStored("/api/x"), null);
  b.m.setStoredTier("member");
  assert.deepEqual(b.m.readStored("/api/x"), { v: 1 });
});

test("区分が変わったら前の内容は捨てる", async () => {
  const a = await fresh();
  a.m.setStoredTier("member");
  a.responses.set("/api/x", { v: 1 });
  await a.m.getJson("/api/x", { persist: true });
  a.m.setStoredTier("anon");
  assert.equal(a.m.readStored("/api/x"), null);
});

test("persist なしの先読みに相乗りしても、persist を求めたら結果を残す（要求は 1 本）", async () => {
  const a = await fresh();
  a.m.setStoredTier("member");
  a.responses.set("/api/race/1", { id: 1 });
  a.m.prefetchJson("/api/race/1", { ttlMs: 30_000 });
  await a.m.getJson("/api/race/1", { ttlMs: 20_000, persist: true });
  await flush();
  assert.equal(a.calls(), 1);
  assert.deepEqual(a.m.readStored("/api/race/1"), { id: 1 });
});

test("要求の間に区分が変わった応答は残さない（ログイン前の匿名の応答を会員の区分で残さない）", async () => {
  const a = await fresh();
  a.m.setStoredTier("anon");
  let release;
  const gate = new Promise((r) => { release = r; });
  globalThis.fetch = () => gate.then(() => ({ ok: true, status: 200, json: async () => ({ anon: true }) }));
  const p = a.m.getJson("/api/y", { persist: true });
  a.m.setStoredTier("member");
  release();
  await p;
  assert.equal(a.m.readStored("/api/y"), null);
});

test("1 件が大きすぎる応答は残さない", async () => {
  const a = await fresh();
  a.m.setStoredTier("member");
  a.responses.set("/api/big", { s: "x".repeat(400_000) });
  await a.m.getJson("/api/big", { persist: true });
  assert.equal(a.m.readStored("/api/big"), null);
});

test("あふれたら古いものから消し、全部は消さない（区分の記録も残る）", async () => {
  const a = await fresh({ limit: 60_000 });
  a.m.setStoredTier("member");
  const realNow = Date.now;
  let t = 1_000_000;
  Date.now = () => t;
  try {
    for (let i = 0; i < 6; i++) {
      t += 1000;
      a.responses.set(`/api/r/${i}`, { s: "y".repeat(12_000) });
      await a.m.getJson(`/api/r/${i}`, { persist: true });
    }
  } finally {
    Date.now = realNow;
  }
  assert.ok(a.m.readStored("/api/r/5", 1e14), "いちばん新しいものは残る");
  assert.equal(a.m.readStored("/api/r/0", 1e14), null, "いちばん古いものは消える");
  assert.equal(a.fs.getItem("jc:tier"), "member");
});

test("失敗は残さない", async () => {
  const a = await fresh();
  a.m.setStoredTier("member");
  await a.m.getJson("/api/fail", { persist: true }).catch(() => undefined);
  assert.equal(a.m.readStored("/api/fail"), null);
});

/** 返る時を自分で決められる fetch */
function heldFetch(body) {
  let release;
  const gate = new Promise((r) => { release = r; });
  globalThis.fetch = () => gate.then(() => ({ ok: true, status: 200, json: async () => body }));
  return () => release();
}

test("最初の読み込み（区分の記録なし）で出した要求は、最初の確認の後に返れば残す", async () => {
  const a = await fresh();
  const release = heldFetch({ v: 1 });
  const p = a.m.getJson("/api/first", { persist: true });
  a.m.setStoredTier("member");
  release();
  await p;
  assert.deepEqual(a.m.readStored("/api/first"), { v: 1 });
});

test("区分の記録なしで出した要求の途中でログインしたら（primeAuthStatus）、匿名の応答を会員の区分で残さない", async () => {
  const a = await fresh();
  const release = heldFetch({ anon: true });
  const p = a.m.getJson("/api/z", { persist: true });
  a.m.primeAuthStatus({ logged_in: true, is_member: true });
  release();
  await p;
  assert.equal(a.m.readStored("/api/z"), null);
});

test("ログアウト（clearStored）の後に返った応答は残さない", async () => {
  const a = await fresh();
  a.m.setStoredTier("member");
  const release = heldFetch({ m: 1 });
  const p = a.m.getJson("/api/w", { persist: true });
  a.m.clearStored();
  a.m.setStoredTier("anon");
  release();
  await p;
  assert.equal(a.m.readStored("/api/w"), null);
});

test("このページで区分を確かめる前は、前の区分の記録があっても書き込まない", async () => {
  const a = await fresh();
  a.m.setStoredTier("member");
  const b = await fresh({ store: a.fs });           // 再読み込み（記録は member のまま、まだ確かめていない）
  b.responses.set("/api/v", { v: 2 });
  await b.m.getJson("/api/v", { persist: true });
  assert.equal(b.fs.getItem("jc:d:/api/v"), null);
});

test("キャッシュに当たったときの書き込みも、区分が変わっていれば残さない", async () => {
  const a = await fresh();
  a.m.setStoredTier("anon");
  const release = heldFetch({ anon: 1 });
  a.m.prefetchJson("/api/h");
  a.m.setStoredTier("member");
  const p = a.m.getJson("/api/h", { persist: true });
  release();
  await p;
  await flush();
  assert.equal(a.m.readStored("/api/h"), null);
});
