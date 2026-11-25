"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { invalidateJson, primeAuthStatus, type AuthStatusJson } from "@/lib/httpCache";
import { lockedMessage, ROTATED_MESSAGE } from "@/lib/loginMessage";
import { adsenseLoaded } from "@/lib/adsense";

export default function LoginPage() {
  const [userId, setUserId] = useState("");
  const [password, setPassword] = useState("");
  const [remember, setRemember] = useState(true);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const router = useRouter();

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (busy) return;
    setBusy(true);
    setError("");
    try {
      const res = await fetch("/api/v1/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ id: userId.trim(), password, remember }),
      });
      if (!res.ok) {
        if (res.status === 429) {
          const body = (await res.json().catch(() => null)) as { retry_after?: number } | null;
          setError(lockedMessage(body?.retry_after ?? Number(res.headers.get("Retry-After") || 0)));
        } else {
          const body = (await res.json().catch(() => null)) as { reason?: string } | null;
          setError(body?.reason === "password_rotated" ? ROTATED_MESSAGE : "ログインID またはパスワードが正しくありません");
        }
        setBusy(false);
        return;
      }
      const body = (await res.json().catch(() => null)) as { auth?: AuthStatusJson } | null;
      if (body?.auth) primeAuthStatus(body.auth);
      else invalidateJson("/api/v1/auth/status");
      // 匿名の間に広告のスクリプトを読んでいたら、全体を読み直して捨てる（一度読んだスクリプトは外せない。会員の画面に残さない）
      if (adsenseLoaded()) window.location.assign("/");
      else router.push("/");
    } catch {
      setError("通信に失敗しました。もう一度お試しください");
      setBusy(false);
    }
  }

  return (
    <div className="mx-auto mt-16 max-w-sm card">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src="/brand/megu-banner-cream.webp" alt="" width={1200} height={636} className="mb-4 w-full rounded-lg" style={{ aspectRatio: "1200 / 636", objectFit: "cover" }} />
      <h1 className="mb-4 text-xl font-bold">ログイン</h1>
      <form onSubmit={submit} className="space-y-3">
        <input type="text" autoComplete="username" className="w-full rounded border bg-transparent p-2" style={{ borderColor: "var(--border)" }} placeholder="ログインID（管理者は空欄）" value={userId} onChange={(e) => setUserId(e.target.value)} />
        <input type="password" className="w-full rounded border bg-transparent p-2" style={{ borderColor: "var(--border)" }} placeholder="パスワード" value={password} onChange={(e) => setPassword(e.target.value)} />
        <label className="flex items-center gap-2 text-sm">
          <input type="checkbox" checked={remember} onChange={(e) => setRemember(e.target.checked)} />
          30日間ログイン保持
        </label>
        {error && <p style={{ color: "var(--err)" }}>{error}</p>}
        <button type="submit" className="btn w-full" disabled={busy} aria-busy={busy}>{busy ? "ログイン中…" : "ログイン"}</button>
      </form>
    </div>
  );
}
