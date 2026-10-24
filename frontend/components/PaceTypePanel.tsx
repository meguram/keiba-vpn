"use client";

import { useId, useState } from "react";
import {
  CONF_JA, PACE_FACTS, PACE_JA, PACE_ORDER, SCENARIO_NOTE, SHAPE_JA, SHAPE_ORDER, STYLE_ORDER,
  basisTags, diverging, fitMark, fitTone, fitWord, segments,
  type FitTone, type HorseFit, type PaceKey, type PaceType, type ShapeKey,
} from "@/lib/paceType";

/* 展開タイプのパネル（ペースの確率・隊列の確率・脚質ごとの有利度・根拠）と、馬ごとの『ハイ/ミドル/スロー』の記号。
   シナリオ別の有利さは過去の傾向から作った説明用の目安で、勝率・期待値の予測には使っていない。 */

const TONE: Record<FitTone, { fg: string; word: string }> = {
  plus: { fg: "#4ade80", word: "有利" },
  even: { fg: "#94a3b8", word: "互角" },
  minus: { fg: "#f87171", word: "不利" },
};

const PACE_COLOR: Record<PaceKey, string> = { high: "#f87171", mid: "#fbbf24", slow: "#60a5fa" };
const SHAPE_COLOR: Record<ShapeKey, string> = { compact: "#a78bfa", normal: "#94a3b8", stretched: "#2dd4bf" };

function StackBar({ title, items }: { title: string; items: { key: string; label: string; pct: number; color: string }[] }) {
  return (
    <div style={{ display: "grid", gap: 4 }}>
      <span style={{ fontSize: 11, color: "var(--text-dim)" }}>{title}</span>
      <div role="img" aria-label={`${title}: ${items.map((i) => `${i.label}${i.pct}%`).join("、")}`} style={{ display: "flex", height: 14, borderRadius: 7, overflow: "hidden", background: "var(--surface2)", border: "1px solid var(--border)" }}>
        {items.map((i) => <span key={i.key} style={{ width: `${i.pct}%`, background: i.color, opacity: 0.85 }} />)}
      </div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: "2px 12px", fontSize: 12 }}>
        {items.map((i) => (
          <span key={i.key} style={{ display: "inline-flex", alignItems: "center", gap: 4, color: "var(--text)" }}>
            <span aria-hidden style={{ width: 9, height: 9, borderRadius: 2, background: i.color }} />{i.label} <b style={{ fontVariantNumeric: "tabular-nums" }}>{i.pct}%</b>
          </span>
        ))}
      </div>
    </div>
  );
}

function StyleRow({ label, score, lift }: { label: string; score: number; lift?: number }) {
  const t = fitTone(score) ?? "even";
  const d = diverging(score);
  return (
    <div style={{ display: "grid", gridTemplateColumns: "40px minmax(90px,1fr) 92px", alignItems: "center", gap: 8, minHeight: 24 }} title={lift != null ? `この展開での3着内率の期待倍率 ×${lift.toFixed(2)}` : undefined}>
      <b style={{ fontSize: 12, color: "var(--text)" }}>{label}</b>
      <span aria-hidden style={{ position: "relative", height: 10, borderRadius: 5, background: "var(--surface2)", border: "1px solid var(--border)" }}>
        <span style={{ position: "absolute", left: "50%", top: -2, bottom: -2, width: 1, background: "var(--text-dim)", opacity: 0.6 }} />
        <span style={{ position: "absolute", top: 1, bottom: 1, borderRadius: 3, background: TONE[t].fg, opacity: 0.85, width: `${d.widthPct}%`, ...(d.side === "plus" ? { left: "50%" } : { right: "50%" }) }} />
      </span>
      <span style={{ fontSize: 12, fontWeight: 700, color: TONE[t].fg, whiteSpace: "nowrap" }}>{fitWord(score)}<span style={{ fontWeight: 400, color: "var(--text-dim)", marginLeft: 4, fontVariantNumeric: "tabular-nums" }}>{score > 0 ? "+" : score < 0 ? "−" : ""}{Math.abs(score).toFixed(2)}</span></span>
    </div>
  );
}

export function PaceTypePanel({ pt }: { pt: PaceType }) {
  const [open, setOpen] = useState(false);
  const id = useId();
  const paceSeg = segments(pt.pace_probs, PACE_ORDER).map((s) => ({ ...s, label: PACE_JA[s.key as PaceKey], color: PACE_COLOR[s.key as PaceKey] }));
  const shapeSeg = segments(pt.shape_probs, SHAPE_ORDER).map((s) => ({ ...s, label: SHAPE_JA[s.key as ShapeKey], color: SHAPE_COLOR[s.key as ShapeKey] }));
  const { tags, warn } = basisTags(pt.basis);
  const styles = STYLE_ORDER.map((k) => ({ k, v: pt.style_fit?.[k] })).filter((x) => x.v);
  const prob = (p: PaceKey, s: ShapeKey) => Math.round((pt.probs?.[`${p}_${s}`] ?? 0) * 100);
  return (
    <div style={{ display: "grid", gap: 10, margin: "0 0 10px", padding: "10px 12px", borderRadius: 10, background: "var(--surface2)", border: "1px solid var(--border)" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
        <b style={{ fontSize: 14, color: "#f0f6fc" }}>展開タイプ {pt.label}</b>
        {pt.confidence && <span style={{ fontSize: 11, padding: "0 7px", borderRadius: 999, border: "1px solid var(--border)", color: "var(--text-dim)" }}>確度 {CONF_JA[pt.confidence] ?? pt.confidence}</span>}
        <button type="button" aria-expanded={open} aria-controls={id} onClick={() => setOpen((v) => !v)} aria-label="展開タイプの見方を開く"
          style={{ marginLeft: "auto", minWidth: 28, minHeight: 28, borderRadius: 999, border: "1px solid var(--border)", background: "var(--surface)", color: "var(--text-dim)", fontSize: 12, cursor: "pointer" }}>ⓘ</button>
      </div>
      {pt.note && <p style={{ margin: 0, fontSize: 12.5, color: "var(--text)", lineHeight: 1.6 }}>{pt.note}</p>}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(220px,1fr))", gap: 12 }}>
        <StackBar title="ペースの確率" items={paceSeg} />
        <StackBar title="隊列の確率（凝縮＝馬群が固まる／縦長＝ばらける）" items={shapeSeg} />
      </div>
      {styles.length > 0 && (
        <div style={{ display: "grid", gap: 2 }}>
          <span style={{ fontSize: 11, color: "var(--text-dim)" }}>この展開で有利な脚質（右＝有利、左＝不利）</span>
          {styles.map(({ k, v }) => <StyleRow key={k} label={v!.label} score={v!.score} lift={v!.expected_lift} />)}
        </div>
      )}
      {(tags.length > 0 || warn) && (
        <div style={{ display: "flex", flexWrap: "wrap", gap: 6, alignItems: "center" }}>
          {tags.map((t) => <span key={t} style={{ fontSize: 11, padding: "1px 8px", borderRadius: 999, background: "var(--surface)", border: "1px solid var(--border)", color: "var(--text)" }}>{t}</span>)}
          {warn && <span style={{ fontSize: 11, fontWeight: 700, color: "#fbbf24" }}>⚠ {warn}</span>}
        </div>
      )}
      {open && (
        <div id={id} style={{ display: "grid", gap: 8, fontSize: 12, color: "var(--text-dim)", lineHeight: 1.6 }}>
          <table style={{ borderCollapse: "collapse", fontSize: 12 }}>
            <caption style={{ captionSide: "top", textAlign: "left", paddingBottom: 4 }}>9つのシナリオの確率</caption>
            <thead><tr><th style={{ textAlign: "left", fontWeight: 400, padding: "2px 8px 2px 0" }} />{SHAPE_ORDER.map((s) => <th key={s} style={{ fontWeight: 400, padding: "2px 10px", textAlign: "right" }}>{SHAPE_JA[s]}</th>)}</tr></thead>
            <tbody>{PACE_ORDER.map((p) => (
              <tr key={p}><th style={{ textAlign: "left", fontWeight: 400, padding: "2px 8px 2px 0" }}>{PACE_JA[p]}</th>{SHAPE_ORDER.map((s) => <td key={s} style={{ padding: "2px 10px", textAlign: "right", color: "var(--text)", fontVariantNumeric: "tabular-nums" }}>{prob(p, s)}%</td>)}</tr>
            ))}</tbody>
          </table>
          <ul style={{ margin: 0, paddingLeft: 18 }}>{PACE_FACTS.map((f) => <li key={f}>{f}</li>)}</ul>
        </div>
      )}
      <p style={{ margin: 0, fontSize: 11, color: "var(--text-dim)" }}>{SCENARIO_NOTE}</p>
    </div>
  );
}

/** 馬ごとの『ハイ/ミドル/スロー』の記号（有利=▲緑、互角=－灰、不利=▽赤）。 */
export function FitMarks({ fit, size = 12 }: { fit: HorseFit; size?: number }) {
  return (
    <span role="group" aria-label="展開別の有利さ（ハイ・ミドル・スロー）" style={{ display: "inline-flex", alignItems: "center", gap: 7, whiteSpace: "nowrap", fontSize: size }}>
      {PACE_ORDER.map((p) => {
        const s = fit.paces[p];
        const t = fitTone(s);
        return (
          <span key={p} title={`${PACE_JA[p]}ペース: ${fitWord(s)}${s != null ? `（${s > 0 ? "+" : ""}${s.toFixed(2)}）` : ""}`} style={{ display: "inline-flex", alignItems: "center", gap: 2 }}>
            <span style={{ color: "var(--text-dim)", fontSize: size - 1 }}>{PACE_JA[p]}</span>
            <b style={{ color: t ? TONE[t].fg : "var(--text-dim)", fontSize: size + 1 }} aria-label={fitWord(s)}>{fitMark(s)}</b>
          </span>
        );
      })}
    </span>
  );
}

/** 押した詳細: 最も合う展開と、9シナリオごとの有利さ（記号＋数値）。 */
export function FitGrid({ fit }: { fit: HorseFit }) {
  return (
    <div style={{ flexBasis: "100%", display: "grid", gap: 4 }}>
      {fit.bestLabel && <span style={{ color: "var(--text-dim)" }}>最も合う展開 <b style={{ color: "var(--text)" }}>{fit.bestLabel}</b>{fit.sensitivity ? <span style={{ marginLeft: 8 }}>（{fit.sensitivity}）</span> : null}</span>}
      <table style={{ borderCollapse: "collapse", fontSize: 12, width: "max-content" }} aria-label="9シナリオごとの有利さ">
        <thead><tr><th />{SHAPE_ORDER.map((s) => <th key={s} style={{ fontWeight: 400, color: "var(--text-dim)", padding: "1px 10px", textAlign: "center" }}>{SHAPE_JA[s]}</th>)}</tr></thead>
        <tbody>{PACE_ORDER.map((p) => (
          <tr key={p}>
            <th style={{ fontWeight: 400, color: "var(--text-dim)", textAlign: "left", padding: "1px 8px 1px 0" }}>{PACE_JA[p]}</th>
            {SHAPE_ORDER.map((s) => {
              const g = fit.grid.find((x) => x.pace === p && x.shape === s);
              const t = fitTone(g?.score);
              return (
                <td key={s} title={g ? `確率 ${Math.round(g.prob * 100)}%` : undefined} style={{ padding: "1px 10px", textAlign: "center", whiteSpace: "nowrap", color: t ? TONE[t].fg : "var(--text-dim)", fontVariantNumeric: "tabular-nums" }}>
                  <b>{fitMark(g?.score)}</b> {g?.score != null ? `${g.score > 0 ? "+" : g.score < 0 ? "−" : ""}${Math.abs(g.score).toFixed(2)}` : ""}
                </td>
              );
            })}
          </tr>
        ))}</tbody>
      </table>
    </div>
  );
}
