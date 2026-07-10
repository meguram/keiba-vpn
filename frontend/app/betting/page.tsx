"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { PageShell } from "@/components/PageShell";
import { USE_MOCK, MOCK_KELLY } from "@/lib/mock";

export default function BettingPage() {
  const [isAdmin, setIsAdmin] = useState<boolean | null>(null);
  const [bankroll, setBankroll] = useState("100000");
  const [result, setResult] = useState<typeof MOCK_KELLY | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (USE_MOCK) {
      setIsAdmin(true);
      return;
    }
    fetch("/api/v1/auth/status", { credentials: "include" })
      .then((r) => (r.ok ? r.json() : { logged_in: false, is_admin: false }))
      .then((d) => setIsAdmin(!!d.logged_in && !!d.is_admin))
      .catch(() => setIsAdmin(false));
  }, []);

  async function optimize() {
    setLoading(true);
    let r: typeof MOCK_KELLY;
    if (USE_MOCK) {
      await new Promise((resolve) => setTimeout(resolve, 400));
      r = { ...MOCK_KELLY, bankroll: Number(bankroll) };
    } else {
      const res = await fetch("/api/v1/betting/optimize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ race_id: "r001", bankroll: Number(bankroll) }),
      });
      r = await res.json();
    }
    setResult(r);
    localStorage.setItem("betting_last_optimize", JSON.stringify({
      race_id:        r.race_id,
      win_prob:       (r.bets as Array<{win_prob?: number}>)[0]?.win_prob ?? null,
      win_odds:       (r.bets as Array<{win_odds?: number}>)[0]?.win_odds ?? null,
      kelly_fraction: r.kelly_fraction,
      bankroll:       r.bankroll,
      saved_at:       new Date().toISOString(),
    }));
    setLoading(false);
  }

  if (isAdmin === null) {
    return (
      <PageShell title="馬券最適化" description="Kelly 基準ポートフォリオ（AN-13）">
        <div className="card flex flex-col items-center gap-4 py-12">
          <p className="text-sm" style={{ color: "var(--text-dim)" }}>認証確認中…</p>
        </div>
      </PageShell>
    );
  }

  if (!isAdmin) {
    return (
      <PageShell title="馬券最適化" description="Kelly 基準ポートフォリオ（AN-13）">
        <div className="card flex flex-col items-center gap-4 py-12">
          <span style={{ fontSize: 40 }}>🔒</span>
          <p className="font-semibold">管理者専用ページ</p>
          <p className="text-sm" style={{ color: "var(--text-dim)" }}>
            馬券最適化機能は管理者限定です。
          </p>
          <Link href="/login" className="btn">ログインする</Link>
        </div>
      </PageShell>
    );
  }

  return (
    <PageShell title="馬券最適化" description="Kelly 基準ポートフォリオ（AN-13）">
      <div className="card space-y-4">
        <div className="flex flex-wrap items-end gap-3">
          <div className="space-y-1">
            <label className="text-xs" style={{ color: "var(--text-dim)" }}>軍資金（円）</label>
            <input
              type="number"
              className="rounded border bg-transparent px-3 py-1.5 text-sm"
              style={{ borderColor: "var(--border)", color: "var(--text)", width: 160 }}
              value={bankroll}
              min={0}
              onChange={(e) => setBankroll(e.target.value)}
            />
          </div>
          <button type="button" className="btn" onClick={optimize} disabled={loading}>
            {loading ? "計算中…" : "最適化"}
          </button>
        </div>
        {result && (
          <>
            <dl className="grid grid-cols-2 gap-3 md:grid-cols-4">
              {[
                { label: "軍資金", value: `¥${result.bankroll.toLocaleString()}` },
                { label: "合計賭け金", value: `¥${result.total_bet.toLocaleString()}` },
                { label: "期待収益", value: `+¥${result.expected_profit.toLocaleString()}`, color: "var(--ok)" },
                { label: "Kelly 比率", value: `${(result.kelly_fraction * 100).toFixed(1)}%`, color: "var(--accent)" },
              ].map((item) => (
                <div key={item.label} className="rounded-lg p-3" style={{ background: "var(--surface2)" }}>
                  <dt className="text-xs" style={{ color: "var(--text-dim)" }}>{item.label}</dt>
                  <dd className="mt-1 font-bold" style={{ color: item.color ?? "var(--text)" }}>{item.value}</dd>
                </div>
              ))}
            </dl>
            <table className="w-full text-sm">
              <thead>
                <tr style={{ color: "var(--text-dim)" }}>
                  <th className="py-2 px-3 text-left">馬名</th>
                  <th className="py-2 px-3 text-center">券種</th>
                  <th className="py-2 px-3 text-right">賭け金</th>
                  <th className="py-2 px-3 text-right">Kelly f</th>
                  <th className="py-2 px-3 text-right">エッジ</th>
                </tr>
              </thead>
              <tbody>
                {result.bets.map((b) => (
                  <tr key={`${b.horse_id}-${b.bet_type}`} style={{ borderTop: "1px solid var(--border)" }}>
                    <td className="py-2 px-3">
                      {b.horse_id
                        ? <a href={`/horse/${b.horse_id}`} target="_blank" rel="noreferrer" style={{ color: "inherit", textDecoration: "none" }}>{b.horse_name}</a>
                        : b.horse_name}
                    </td>
                    <td className="py-2 px-3 text-center">{b.bet_type}</td>
                    <td className="py-2 px-3 text-right">¥{b.stake.toLocaleString()}</td>
                    <td className="py-2 px-3 text-right font-mono">{(b.kelly_f * 100).toFixed(1)}%</td>
                    <td className="py-2 px-3 text-right font-mono" style={{ color: "var(--ok)" }}>
                      +{(b.edge * 100).toFixed(1)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        )}
      </div>
    </PageShell>
  );
}
