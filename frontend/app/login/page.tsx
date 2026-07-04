"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

export default function LoginPage() {
  const [password, setPassword] = useState("");
  const [remember, setRemember] = useState(true);
  const [error, setError] = useState("");
  const router = useRouter();

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    const res = await fetch("/api/v1/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ password, remember }),
    });
    if (!res.ok) {
      setError("ログイン失敗");
      return;
    }
    router.push("/");
  }

  return (
    <div className="mx-auto mt-16 max-w-sm card">
      <h1 className="mb-4 text-xl font-bold">ログイン</h1>
      <form onSubmit={submit} className="space-y-3">
        <input type="password" className="w-full rounded border bg-transparent p-2" style={{ borderColor: "var(--border)" }} placeholder="パスワード" value={password} onChange={(e) => setPassword(e.target.value)} />
        <label className="flex items-center gap-2 text-sm">
          <input type="checkbox" checked={remember} onChange={(e) => setRemember(e.target.checked)} />
          30日間ログイン保持
        </label>
        {error && <p style={{ color: "var(--err)" }}>{error}</p>}
        <button type="submit" className="btn w-full">ログイン</button>
      </form>
    </div>
  );
}
