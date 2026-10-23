"use client";

import { useEffect, useState } from "react";
import { PendingLink } from "@/components/PendingLink";
import { getJson } from "@/lib/httpCache";

type MarkRow = { key: string; symbol: string; n: number; text: string; win_rate: number | null; top2_rate: number | null; top3_rate: number | null; avg_popularity: number | null; avg_odds: number | null };
type Period = { key: string; label: string; from: string; to: string; data_from: string | null; data_to: string | null; days: number; races: number; status: string; marks: MarkRow[] };
type Card = { as_of: string; year: number; periods: Period[] };

const md = (ymd: string) => `${Number(ymd.slice(4, 6))}/${Number(ymd.slice(6, 8))}`;
const pct = (x: number | null) => (x == null ? "—" : `${(x * 100).toFixed(1)}%`);

/** ホームの『印の成績』: ◎○▲★消 の着順（1着-2着-3着-4着以下）・◎○の勝率・◎○▲★の平均人気/オッズ（確定）を、先週・直近5週間・年（今年のデータが約2か月未満なら去年）で並べる。集計が無ければ何も出さない。 */
export function MarkResultCard() {
  const [card, setCard] = useState<Card | null>(null);

  useEffect(() => {
    let alive = true;
    getJson<Card>("/api/prediction-results/mark-card", { ttlMs: 60_000 })
      .then((d) => { if (alive) setCard(d); })
      .catch(() => { if (alive) setCard(null); });
    return () => { alive = false; };
  }, []);

  if (!card || !card.periods.some((p) => p.races > 0)) return null;

  return (
    <section aria-label="印の成績" style={{ maxWidth: 1100, margin: "0 auto 24px", padding: "0 20px" }}>
      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", marginBottom: 10, flexWrap: "wrap", gap: 6 }}>
        <h2 style={{ margin: 0, fontSize: 16, fontWeight: 700, color: "var(--text)" }}>🎯 印の成績</h2>
        <span style={{ fontSize: 11, color: "var(--text-dim)" }}>1着-2着-3着-着外（印をつけた馬の頭数）</span>
        <PendingLink href="/prediction-results" style={{ fontSize: 12, color: "var(--accent)", textDecoration: "none" }}>予測成績をすべて見る →</PendingLink>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 12 }}>
        {card.periods.map((p) => (
          <div key={p.key} style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 10, padding: "12px 14px" }}>
            <div style={{ display: "flex", alignItems: "baseline", gap: 8, marginBottom: 8 }}>
              <span style={{ fontWeight: 700, fontSize: 14 }}>{p.label}</span>
              <span style={{ fontSize: 11, color: "var(--text-dim)" }}>
                {p.data_from && p.data_to ? `${md(p.data_from)}〜${md(p.data_to)}` : `${md(p.from)}〜${md(p.to)}`}・{p.races}レース
              </span>
            </div>
            {p.races === 0 ? (
              <p style={{ margin: 0, fontSize: 12, color: "var(--text-dim)" }}>集計できるレースがありません</p>
            ) : (
              <div>
                {p.marks.map((m) => {
                  const showWin = m.key === "honmei" || m.key === "pair";          // ◎○ は勝率も出す
                  const showAvg = m.key !== "kesi";                                // ◎○▲★ は平均人気・平均オッズも出す
                  return (
                    <div key={m.key} style={{ borderTop: "1px solid var(--border)", padding: "6px 4px" }}>
                      <div style={{ display: "flex", alignItems: "baseline", gap: 8, fontSize: 13 }}>
                        <span style={{ width: 22, fontWeight: 800 }}>{m.symbol}</span>
                        <span style={{ fontVariantNumeric: "tabular-nums", fontWeight: 700 }}>{m.n ? m.text : "—"}</span>
                        <span style={{ marginLeft: "auto", fontSize: 11, color: "var(--text-dim)", whiteSpace: "nowrap" }}>
                          {m.n ? `${showWin ? `勝率 ${pct(m.win_rate)}・` : ""}複勝率 ${pct(m.top3_rate)}` : ""}
                        </span>
                      </div>
                      {showAvg && m.n > 0 && (m.avg_popularity != null || m.avg_odds != null) && (
                        <div style={{ marginLeft: 30, fontSize: 11, color: "var(--text-dim)" }}>
                          {m.avg_popularity != null ? `平均 ${m.avg_popularity.toFixed(1)}人気` : ""}
                          {m.avg_popularity != null && m.avg_odds != null ? "・" : ""}
                          {m.avg_odds != null ? `平均オッズ ${m.avg_odds.toFixed(1)}倍` : ""}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        ))}
      </div>
    </section>
  );
}
