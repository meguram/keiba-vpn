"use client";

import { md, recentMean, type ExpectedMegu, type ExpectedMeguMeta } from "@/lib/expectedMegu";

/* 実測のめぐ指数（直近の過去走）の表示。想定値（今回の指数の推定）は画面に出さない: 予測精度への寄与が小さいため。
   API の expected_megu.recent を使う（古い→新しい）。 */

/** 馬のパネル: 直近の実測指数を日付つきで（新しい順）。平均と、最新の反映走。 */
export function RecentMeguSection({ e, meta }: { e?: ExpectedMegu | null; meta?: ExpectedMeguMeta | null }) {
  const list = e?.recent ?? [];
  if (!list.length) return <div style={{ fontSize: 12, color: "var(--text-dim)" }}>過去のめぐ指数がまだありません（初出走、または未計算）。</div>;
  const mean = recentMean(e);
  return (
    <div style={{ display: "grid", gap: 8 }}>
      <div style={{ display: "flex", alignItems: "baseline", gap: 8, flexWrap: "wrap" }}>
        <span style={{ fontSize: 26, fontWeight: 800, color: "#fff", fontVariantNumeric: "tabular-nums" }}>{list[0].value.toFixed(1)}</span>
        <span style={{ fontSize: 12, color: "var(--text-dim)" }}>前走（{md(list[0].date)}）</span>
        {mean != null && list.length > 1 && <span style={{ fontSize: 12, color: "var(--text)" }}>直近{list.length}走の平均 <b>{mean.toFixed(1)}</b></span>}
      </div>
      <div role="list" aria-label="直近のめぐ指数" style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
        {list.map((r, i) => (
          <span key={r.date + i} role="listitem" title={r.low ? "低信頼（基準タイムのサンプルが少ないなど）" : undefined} style={{ fontSize: 12, padding: "3px 8px", borderRadius: 6, background: "var(--surface2)", border: "1px solid var(--border)", opacity: r.low ? 0.6 : 1, fontVariantNumeric: "tabular-nums" }}>
            <span style={{ color: "var(--text-dim)" }}>{md(r.date)} </span><b>{r.value.toFixed(1)}</b>{r.low && <span style={{ color: "var(--text-dim)" }}> 低信頼</span>}
          </span>
        ))}
      </div>
      <div style={{ fontSize: 11, color: "var(--text-dim)", lineHeight: 1.6 }}>
        実測のめぐ指数（過去の走。1点＝0.1秒、斤量55kg換算）。{meta?.data_through && <>指数データは {md(meta.data_through)} まで。</>}
        {e?.latest_unreflected && <b style={{ color: "#fbbf24" }}> 直近の走（{md(e.latest_past_date)}）は指数に未反映です。</b>}
      </div>
    </div>
  );
}
