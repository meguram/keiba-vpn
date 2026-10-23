"use client";

import { useCallback, useEffect, useState } from "react";
import { useAuthStatus } from "@/lib/hooks/useAuthStatus";

type UserRow = { fp: string; source: "config" | "managed"; label: string | null; enabled: boolean; created_at: string | null; note: string };

const API = "/api/v1/admin/users";

async function call(url: string, method: string, body?: unknown): Promise<{ ok: boolean; error?: string }> {
  try {
    const res = await fetch(url, { method, credentials: "include", headers: { "Content-Type": "application/json" }, body: body ? JSON.stringify(body) : undefined });
    if (res.ok) return { ok: true };
    const j = await res.json().catch(() => ({}));
    return { ok: false, error: j.error || `エラー（${res.status}）` };
  } catch {
    return { ok: false, error: "通信に失敗しました" };
  }
}

export default function AdminUsersPage() {
  const { isAdmin, loading } = useAuthStatus();
  const [users, setUsers] = useState<UserRow[]>([]);
  const [id, setId] = useState("");
  const [pw, setPw] = useState("");
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const [busy, setBusy] = useState(false);
  const [resetFp, setResetFp] = useState<string | null>(null);
  const [newPw, setNewPw] = useState("");

  const load = useCallback(async () => {
    try {
      const res = await fetch(API, { credentials: "include", cache: "no-store" });
      if (res.ok) setUsers((await res.json()).users ?? []);
    } catch { /* 一覧は取れなくても登録はできる */ }
  }, []);
  useEffect(() => { if (isAdmin) void load(); }, [isAdmin, load]);

  async function act(p: Promise<{ ok: boolean; error?: string }>, okText: string) {
    setBusy(true);
    const r = await p;
    setMsg({ ok: r.ok, text: r.ok ? okText : r.error || "失敗しました" });
    setBusy(false);
    if (r.ok) await load();
    return r.ok;
  }

  if (loading) return <div className="mx-auto mt-16 max-w-xl card">読み込み中…</div>;
  if (!isAdmin) return <div className="mx-auto mt-16 max-w-xl card"><p className="font-semibold">管理者専用ページ</p><p className="text-sm" style={{ color: "var(--text-dim)" }}>ユーザーの登録は管理者だけが行えます。</p></div>;

  const input = "w-full rounded border bg-transparent p-2";
  return (
    <div className="mx-auto mt-10 max-w-2xl space-y-6">
      <div className="card">
        <h1 className="mb-1 text-xl font-bold">ユーザー登録</h1>
        <p className="mb-4 text-xs" style={{ color: "var(--text-dim)" }}>
          ログインIDとパスワードは一方向に変換して保存します（元の値は誰にも読み出せません。忘れたときは再設定します）。パスワードは10文字以上、ログインIDは英数字と . _ @ - の3〜64文字です。
        </p>
        <form
          onSubmit={async (e) => { e.preventDefault(); if (await act(call(API, "POST", { id: id.trim(), password: pw }), `登録しました（ログインID: ${id.trim()}）`)) { setId(""); setPw(""); } }}
          className="space-y-3"
        >
          <input className={input} style={{ borderColor: "var(--border)" }} placeholder="ログインID" autoComplete="off" value={id} onChange={(e) => setId(e.target.value)} />
          <input className={input} style={{ borderColor: "var(--border)" }} type="password" placeholder="初期パスワード（10文字以上）" autoComplete="new-password" value={pw} onChange={(e) => setPw(e.target.value)} />
          <button type="submit" className="btn w-full" disabled={busy || !id.trim() || !pw}>{busy ? "処理中…" : "登録する"}</button>
        </form>
        {msg && <p className="mt-3 text-sm" style={{ color: msg.ok ? "var(--ok, #3fb950)" : "var(--err)" }}>{msg.text}</p>}
      </div>

      <div className="card">
        <h2 className="mb-2 text-base font-bold">登録済みユーザー</h2>
        <p className="mb-3 text-xs" style={{ color: "var(--text-dim)" }}>ログインIDは保存していないため一覧には出ません。「指紋」で見分けます（登録した順番や日時で判断できます）。</p>
        {users.length === 0 && <p className="text-sm" style={{ color: "var(--text-dim)" }}>ユーザーはいません</p>}
        <ul className="space-y-2">
          {users.map((u) => (
            <li key={`${u.source}-${u.fp}-${u.label ?? ""}`} className="rounded border p-3 text-sm" style={{ borderColor: "var(--border)" }}>
              <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
                <span className="font-mono">{u.source === "config" ? `（動作確認用 ${u.label}）` : `指紋 ${u.fp}`}</span>
                <span style={{ color: u.enabled ? "#3fb950" : "var(--text-dim)" }}>{u.enabled ? "有効" : "無効"}</span>
                {u.created_at && <span style={{ color: "var(--text-dim)" }}>{u.created_at.slice(0, 16).replace("T", " ")}</span>}
                {u.note && <span style={{ color: "var(--text-dim)" }}>{u.note}</span>}
              </div>
              {u.source === "managed" && (
                <div className="mt-2 flex flex-wrap items-center gap-2">
                  <button className="btn" disabled={busy} onClick={() => act(call(`${API}/${u.fp}/enabled`, "POST", { enabled: !u.enabled }), u.enabled ? "無効にしました（ログイン中でも使えなくなります）" : "有効にしました")}>{u.enabled ? "無効にする" : "有効にする"}</button>
                  <button className="btn" disabled={busy} onClick={() => { setResetFp(resetFp === u.fp ? null : u.fp); setNewPw(""); }}>パスワード再設定</button>
                  <button className="btn" disabled={busy} onClick={() => { if (window.confirm(`指紋 ${u.fp} のユーザーを削除します。元に戻せません。`)) void act(call(`${API}/${u.fp}`, "DELETE"), "削除しました"); }}>削除</button>
                </div>
              )}
              {resetFp === u.fp && (
                <form className="mt-2 flex gap-2" onSubmit={async (e) => { e.preventDefault(); if (await act(call(`${API}/${u.fp}/password`, "POST", { password: newPw }), "パスワードを変更しました")) { setResetFp(null); setNewPw(""); } }}>
                  <input className={input} style={{ borderColor: "var(--border)" }} type="password" placeholder="新しいパスワード（10文字以上）" autoComplete="new-password" value={newPw} onChange={(e) => setNewPw(e.target.value)} />
                  <button type="submit" className="btn" disabled={busy || !newPw}>変更</button>
                </form>
              )}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
