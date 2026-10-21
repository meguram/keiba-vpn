"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { invalidateJson } from "@/lib/httpCache";

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
        setError("ID またはパスワードが正しくありません");
        setBusy(false);
        return;
      }
      invalidateJson("/api/v1/auth/status");
      router.push("/");
    } catch {
      setError("通信に失敗しました。もう一度お試しください");
      setBusy(false);
    }
  }

  return (
    <div className="mx-auto mt-16 max-w-sm card">
      <h1 className="mb-4 text-xl font-bold">ログイン</h1>
      <form onSubmit={submit} className="space-y-3">
        <input type="text" autoComplete="username" className="w-full rounded border bg-transparent p-2" style={{ borderColor: "var(--border)" }} placeholder="ID（開発者は空欄）" value={userId} onChange={(e) => setUserId(e.target.value)} />
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
