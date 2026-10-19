"use client";

import { CHANGE_LABEL, CONF_LABEL, fmtGap, fmtValue, md, tipOf, type ExpectedMegu, type ExpectedMeguMeta } from "@/lib/expectedMegu";

/** 出馬表の 1 セル: 想定指数（大きく）＋ ±レンジ（広い画面のみ）＋ 条件変化・参考のタグ */
export function ExpectedMeguCell({ e }: { e?: ExpectedMegu | null }) {
  if (!e) return <span style={{ color: "var(--text-dim)" }} title={tipOf(e)}>—</span>;
  const faint = e.confidence !== "normal";
  const tag = e.latest_unreflected ? "未反映" : e.change && e.change !== "none" ? "条件" : faint ? "参考" : null;
  const tagTone = e.change === "large" ? "#fb923c" : "var(--text-dim)";
  return (
    <span title={tipOf(e)} style={{ whiteSpace: "nowrap", opacity: faint ? 0.75 : 1 }}>
      <b style={{ color: "var(--text)", fontVariantNumeric: "tabular-nums" }}>{fmtValue(e)}</b>
      {e.range != null && <span className="pt-hide-sm" style={{ fontSize: 10, color: "var(--text-dim)" }}> ±{e.range.toFixed(0)}</span>}
      {tag && <span style={{ fontSize: 9, marginLeft: 3, padding: "0 4px", borderRadius: 3, border: `1px solid ${tagTone}`, color: tagTone }}>{tag}</span>}
    </span>
  );
}

/** 馬のパネル: 想定めぐ指数の内訳（過去 5 走の平均 → 条件変化の補正 → 想定 ±レンジ）と、最上位との秒差 */
export function ExpectedMeguSection({ e, meta }: { e?: ExpectedMegu | null; meta?: ExpectedMeguMeta | null }) {
  if (!e) return <div style={{ fontSize: 12, color: "var(--text-dim)" }}>過去のめぐ指数がまだありません（初出走、または未計算）。</div>;
  const tone = e.change === "large" ? "#fb923c" : e.change === "some" ? "#fbbf24" : "var(--text-dim)";
  return (
    <div style={{ display: "grid", gap: 8 }}>
      <div style={{ display: "flex", alignItems: "baseline", gap: 8, flexWrap: "wrap" }}>
        <span style={{ fontSize: 26, fontWeight: 800, color: "#fff", fontVariantNumeric: "tabular-nums" }}>{fmtValue(e)}</span>
        {e.range != null && <span style={{ fontSize: 13, color: "var(--text-dim)" }}>±{e.range.toFixed(1)}</span>}
        {e.rank != null && <span style={{ fontSize: 12, color: "var(--text)" }}>このレースで {e.rank} 番手（{fmtGap(e)}）</span>}
      </div>
      <div style={{ fontSize: 12, color: "var(--text-dim)", lineHeight: 1.7 }}>
        過去{e.n_runs}走の加重平均 <b style={{ color: "var(--text)" }}>{(e.base_value ?? e.value).toFixed(1)}</b>
        {(e.adjust ?? []).map((a) => (
          <span key={a.key}> → <span style={{ color: a.delta < 0 ? "#f87171" : "#4ade80" }}>{a.label} {a.delta > 0 ? "+" : ""}{a.delta.toFixed(1)}</span></span>
        ))}
        {(e.adjust ?? []).length > 0 && <> → 想定 <b style={{ color: "#fff" }}>{fmtValue(e)}</b></>}
      </div>
      {(e.reasons?.length || (e.change && e.change !== "none")) ? (
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap", alignItems: "center" }}>
          {e.change && e.change !== "none" && <span style={{ fontSize: 11, fontWeight: 700, color: tone, border: `1px solid ${tone}`, borderRadius: 4, padding: "1px 6px" }}>{CHANGE_LABEL[e.change]}</span>}
          {(e.reasons ?? []).map((r) => <span key={r} style={{ fontSize: 11, color: "var(--text)", background: "var(--surface2)", borderRadius: 4, padding: "1px 6px" }}>{r}</span>)}
          <span style={{ fontSize: 10, color: "var(--text-dim)" }}>レンジは同じ条件変化の場合の実際の誤差（1標準偏差）</span>
        </div>
      ) : null}
      <div style={{ fontSize: 11, color: "var(--text-dim)", lineHeight: 1.6 }}>
        最新の反映走 {md(e.latest_run_date)}
        {e.latest_unreflected && <b style={{ color: "#fbbf24" }}> ・直近の走（{md(e.latest_past_date)}）は指数に未反映</b>}
        {CONF_LABEL[e.confidence] && <> ・{CONF_LABEL[e.confidence]}</>}
        {(e.low_conf_runs ?? 0) > 0 && <> ・低信頼の走 {e.low_conf_runs} 走は重みを半分に</>}
        <br />1点＝0.1秒。<b>過去のめぐ指数からの想定で、AI予測ではありません。</b>
        {meta?.data_through && <> 指数データは {md(meta.data_through)} まで。</>}
        {meta?.model?.eval && <> 条件変化の補正は 2020〜24年で学習し、2025年で検証（誤差 {meta.model.eval.plain_mae?.toFixed(2)} → {meta.model.eval.adjusted_mae?.toFixed(2)}）。</>}
      </div>
    </div>
  );
}
