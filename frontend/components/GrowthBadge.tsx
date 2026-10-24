"use client";

import { useEffect, useRef, useState } from "react";
import { GROWTH_NOTE, contributionRows, fmtPoints, growthView, indexLine, levelRow, levelSummary, pct, type Growth, type GrowthMeta } from "@/lib/growth";

/* 成長度のバッジ（出馬表の馬名の近く）と、押して開く詳細。記号＋言葉＋色（色だけに頼らない）。 */

export function GrowthBadge({ g, meta, horse }: { g?: Growth | null; meta?: GrowthMeta | null; horse?: string }) {
  const [open, setOpen] = useState(false);
  if (!g) return null;
  const v = growthView(g.stage);
  return (
    <>
      <button
        type="button"
        onClick={(e) => { e.stopPropagation(); setOpen(true); }}
        aria-haspopup="dialog"
        aria-label={`成長度 ${v.word}${g.provisional ? "（暫定）" : ""}。押すと理由を表示`}
        title={`成長度: ${v.word}${g.provisional ? "（暫定・馬体重の発表待ち）" : ""}`}
        style={{ font: "inherit", fontSize: 11, fontWeight: 700, lineHeight: 1.2, display: "inline-flex", alignItems: "center", gap: 3, marginLeft: 6, padding: "0 6px", minHeight: 20, borderRadius: 999, background: v.bg, color: v.color, border: `1px ${g.provisional ? "dashed" : "solid"} ${v.bd}`, cursor: "pointer", verticalAlign: "middle", whiteSpace: "nowrap" }}
      >
        <span aria-hidden>{v.sym}</span><span>{v.word}</span>{g.provisional && <span style={{ fontWeight: 500, opacity: 0.85 }}>暫定</span>}
      </button>
      {open && <GrowthDetail g={g} meta={meta} horse={horse} onClose={() => setOpen(false)} />}
    </>
  );
}

function GrowthDetail({ g, meta, horse, onClose }: { g: Growth; meta?: GrowthMeta | null; horse?: string; onClose: () => void }) {
  const closeRef = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    closeRef.current?.focus();
    const on = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", on);
    return () => window.removeEventListener("keydown", on);
  }, [onClose]);
  const v = growthView(g.stage);
  const rows = contributionRows(g);
  const sum = levelSummary(meta, g.stage);
  const table = meta?.level_table ?? [];
  const noTl = g.coverage && g.coverage.oikiri === false;
  return (
    <div role="dialog" aria-modal="true" aria-label={`${horse ?? ""}の成長度`} onClick={(e) => { e.stopPropagation(); onClose(); }}
      style={{ position: "fixed", inset: 0, zIndex: 70, display: "flex", alignItems: "center", justifyContent: "center", padding: 12, background: "rgba(0,0,0,0.6)" }}>
      <div onClick={(e) => e.stopPropagation()} style={{ width: "min(440px, 100%)", maxHeight: "88vh", overflowY: "auto", background: "var(--bg, #0d1117)", border: "1px solid var(--border)", borderRadius: 14, padding: "14px 14px 16px", color: "var(--text)", fontSize: 13, lineHeight: 1.6 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <strong style={{ fontSize: 15, color: "#f0f6fc", minWidth: 0, overflowWrap: "anywhere" }}>{horse ? `${horse} の成長度` : "成長度"}</strong>
          <button ref={closeRef} type="button" onClick={onClose} aria-label="閉じる" style={{ marginLeft: "auto", minWidth: 44, minHeight: 44, borderRadius: 8, border: "1px solid var(--border)", background: "var(--surface)", color: "var(--text)", fontSize: 16, cursor: "pointer" }}>✕</button>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap", margin: "4px 0 8px" }}>
          <span style={{ display: "inline-flex", alignItems: "center", gap: 6, padding: "4px 12px", borderRadius: 999, background: v.bg, color: v.color, border: `1px solid ${v.bd}`, fontSize: 16, fontWeight: 800 }}>
            <span aria-hidden>{v.sym}</span>{v.word}
          </span>
          <span style={{ color: "var(--text-dim)" }}>{indexLine(g)}</span>
        </div>
        {g.provisional && <p style={{ margin: "0 0 8px", color: "#fbbf24" }}>暫定（馬体重の発表待ち）: 馬体重を除いて計算しています。発表後（レースの約1時間前）に更新されます。</p>}
        {(g.notes ?? []).length > 0
          ? (g.notes ?? []).map((n) => <p key={n} style={{ margin: "0 0 8px", color: "#fbbf24" }}>{n}。</p>)
          : noTl && <p style={{ margin: "0 0 8px", color: "var(--text-dim)" }}>追い切りの時計が取れていないため、追い切りの寄与は平均で置いています。</p>}
        {(g.reasons ?? []).length > 0 && (
          <ul style={{ margin: "0 0 8px", paddingLeft: 18 }}>
            {(g.reasons ?? []).map((r) => <li key={r}>{r}</li>)}
          </ul>
        )}
        {rows.length > 0 && (
          <div style={{ margin: "0 0 10px" }}>
            <div style={{ fontSize: 11, color: "var(--text-dim)", marginBottom: 2 }}>要素ごとの寄与（指数ポイント。主役は追い切りと成長期で、体重と血統の上乗せは小さい）</div>
            {rows.map((r) => (
              <div key={r.key} style={{ display: "flex", justifyContent: "space-between", gap: 8, borderBottom: "1px solid var(--border)", padding: "2px 0" }}>
                <span>{r.label}</span>
                <b style={{ color: r.points > 0.05 ? "#4ade80" : r.points < -0.05 ? "#f87171" : "var(--text-dim)", fontVariantNumeric: "tabular-nums" }}>{fmtPoints(r.points)}</b>
              </div>
            ))}
          </div>
        )}
        {sum && <p style={{ margin: "0 0 6px" }}>{sum}</p>}
        {table.length > 0 && (
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12, margin: "0 0 8px" }}>
            <caption style={{ textAlign: "left", fontSize: 11, color: "var(--text-dim)", paddingBottom: 2 }}>段階ごとの実績（2024〜2025年）</caption>
            <thead><tr style={{ color: "var(--text-dim)", textAlign: "right" }}><th style={{ textAlign: "left", fontWeight: 500 }}>段階</th><th style={{ fontWeight: 500 }}>期待より上がった割合</th><th style={{ fontWeight: 500 }}>平均の上がり幅</th></tr></thead>
            <tbody>
              {table.map((r) => {
                const rv = growthView(r.level);
                const me = r.level === g.stage;
                return (
                  <tr key={r.level} style={{ textAlign: "right", background: me ? rv.bg : undefined, outline: me ? `1px solid ${rv.bd}` : undefined }}>
                    <td style={{ textAlign: "left", color: rv.color, fontWeight: me ? 800 : 500 }}>{rv.sym} {rv.word}</td>
                    <td style={{ fontVariantNumeric: "tabular-nums" }}>{pct(r.share_resid_up)}</td>
                    <td style={{ fontVariantNumeric: "tabular-nums" }}>{r.mean_y_now != null ? `${fmtPoints(r.mean_y_now)}点` : "—"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
        {levelRow(meta, 4) && <p style={{ margin: "0 0 6px", fontSize: 11, color: "var(--text-dim)" }}>「大きく成長」でも、4頭に1頭は指数が下がります。</p>}
        <p style={{ margin: 0, fontSize: 11, color: "var(--text-dim)" }}>{meta?.note ?? GROWTH_NOTE}</p>
      </div>
    </div>
  );
}
