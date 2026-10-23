"use client";

import { useState } from "react";

export default function SignupPage() {
  const [userId, setUserId] = useState("");
  const [password, setPassword] = useState("");
  const [agree, setAgree] = useState(false);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (busy) return;
    setBusy(true);
    setError("");
    try {
      const res = await fetch("/api/v1/billing/signup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ user_id: userId.trim(), password }),
      });
      const d = await res.json().catch(() => ({}));
      if (!res.ok || !d.checkout_url) {
        setError(d.error || "登録を開始できませんでした。時間をおいてもう一度お試しください");
        setBusy(false);
        return;
      }
      window.location.href = d.checkout_url;
    } catch {
      setError("通信に失敗しました。もう一度お試しください");
      setBusy(false);
    }
  }

  const field = "w-full rounded border bg-transparent p-2";
  return (
    <div className="mx-auto mt-16 max-w-sm card">
      <h1 className="mb-1 text-xl font-bold">会員登録（月額プラン）</h1>
      <p className="mb-4 text-sm" style={{ color: "var(--text-dim)" }}>ログインIDとパスワードを決めて、お支払いに進みます。お支払いはカード決済サービス Stripe の画面で行い、カード情報は当サイトには保存されません。</p>
      <form onSubmit={submit} className="space-y-3">
        <input type="text" autoComplete="username" className={field} style={{ borderColor: "var(--border)" }} placeholder="ログインID（英数字と . _ @ - の3〜64文字）" value={userId} onChange={(e) => setUserId(e.target.value)} required />
        <input type="password" autoComplete="new-password" className={field} style={{ borderColor: "var(--border)" }} placeholder="パスワード" value={password} onChange={(e) => setPassword(e.target.value)} required />
        <label className="flex items-start gap-2 text-sm">
          <input type="checkbox" className="mt-1" checked={agree} onChange={(e) => setAgree(e.target.checked)} />
          <span><a href="/legal/terms" target="_blank" rel="noreferrer" style={{ color: "var(--accent)" }}>利用規約</a>・<a href="/legal/privacy" target="_blank" rel="noreferrer" style={{ color: "var(--accent)" }}>プライバシーポリシー</a>・<a href="/legal/tokushoho" target="_blank" rel="noreferrer" style={{ color: "var(--accent)" }}>特定商取引法に基づく表記</a>に同意します</span>
        </label>
        {error && <p style={{ color: "var(--err)" }}>{error}</p>}
        <button type="submit" className="btn w-full" disabled={busy || !agree} aria-busy={busy}>{busy ? "移動中…" : "お支払いに進む"}</button>
      </form>
    </div>
  );
}
