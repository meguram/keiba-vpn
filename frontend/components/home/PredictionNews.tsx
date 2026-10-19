"use client";

import { useEffect, useState } from "react";
import { PendingLink } from "@/components/PendingLink";
import { getJson } from "@/lib/httpCache";
import { useAuthStatus } from "@/lib/hooks/useAuthStatus";
import type { NewsItem } from "@/lib/predictionResults";

const KIND_META: Record<NewsItem["kind"], { icon: string; accent: string }> = {
  day: { icon: "📰", accent: "var(--accent)" },
  week: { icon: "📈", accent: "#34d399" },
  ana_hit: { icon: "🎯", accent: "#f59e0b" },
  big_return: { icon: "💰", accent: "#f59e0b" },
  big_miss: { icon: "⚠️", accent: "#f87171" },
};

/** ホーム上部の『予測ニュース』: 直近の結果まとめ・今週の累計・注目レースを、記事のカードで並べる。
 *  回収率（仮想購入）は会員向け。未ログインの間は回収率を除いた文言を出す。集計が1件も無ければ何も出さない。 */
export function PredictionNews() {
  const { isMember, loading } = useAuthStatus();
  const [items, setItems] = useState<NewsItem[] | null>(null);

  useEffect(() => {
    let alive = true;
    getJson<{ items: NewsItem[] }>("/api/prediction-results/news?limit=5", { ttlMs: 30_000 })
      .then((d) => { if (alive) setItems(d.items ?? []); })
      .catch(() => { if (alive) setItems([]); });
    return () => { alive = false; };
  }, []);

  if (!items || items.length === 0) return null;

  return (
    <section aria-label="予測ニュース" style={{ maxWidth: 1100, margin: "0 auto 24px", padding: "0 20px" }}>
      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", marginBottom: 10 }}>
        <h2 style={{ margin: 0, fontSize: 16, fontWeight: 700, color: "var(--text)" }}>📰 予測ニュース</h2>
        <PendingLink href="/prediction-results" style={{ fontSize: 12, color: "var(--accent)", textDecoration: "none" }}>
          予測成績をすべて見る →
        </PendingLink>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: 12 }}>
        {items.map((it, i) => {
          const meta = KIND_META[it.kind] ?? KIND_META.day;
          const text = !loading && isMember ? it.subtitle : it.subtitle_public;
          return (
            <PendingLink
              key={`${it.kind}-${i}`}
              href={it.link}
              style={{
                display: "block", textDecoration: "none", color: "var(--text)", background: "var(--surface)",
                border: "1px solid var(--border)", borderLeft: `4px solid ${meta.accent}`, borderRadius: 10, padding: "12px 14px",
                minHeight: 44,
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                <span aria-hidden>{meta.icon}</span>
                <span style={{ fontWeight: 700, fontSize: 14 }}>{it.title}</span>
                <span
                  style={{
                    marginLeft: "auto", fontSize: 11, fontWeight: 700, padding: "1px 8px", borderRadius: 999,
                    color: it.badge === "確定" ? "#4ade80" : "#fbbf24",
                    border: `1px solid ${it.badge === "確定" ? "rgba(74,222,128,0.5)" : "rgba(251,191,36,0.5)"}`,
                  }}
                >
                  {it.badge}
                </span>
              </div>
              <div style={{ fontSize: 13, color: "var(--text-dim)", lineHeight: 1.5 }}>{text}</div>
            </PendingLink>
          );
        })}
      </div>
      {!loading && !isMember && (
        <p style={{ margin: "8px 0 0", fontSize: 11, color: "var(--text-dim)" }}>推奨馬券の回収率は、ログインすると表示されます。</p>
      )}
    </section>
  );
}
