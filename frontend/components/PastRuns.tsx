"use client";

import { useId, useState } from "react";
import { benchmarkSummary, benchmarkTitle, clearsLine, passState } from "@/lib/classBenchmark";
import { md, recentMean, type ClassBenchmark, type ExpectedMegu, type RecentRun } from "@/lib/expectedMegu";
import { barHeightPct, barLabel, barTone, bestIndex, chronological, finishMark, prevSummary } from "@/lib/pastRunsView";

/* 馬柱（競馬新聞の 1 マス風）: 各馬の直近5走を、前走から順に、レース情報と実測のめぐ指数つきで並べる。
   出馬表の各馬の行の直下（広い・中間の表）と、カードの下段（スマホ）に置く。指数は実測（過去走）だけ。今回の想定は出さない。 */

const SLOTS = 5;
const LABELS = ["前走", "2走前", "3走前", "4走前", "5走前"];

/** 出馬表のページの <style> に入れる（どの表示幅でも効く） */
export const PAST_RUNS_CSS = `
.pr-wrap{text-align:left}
.pr-row{display:flex;align-items:center;gap:6px 14px;flex-wrap:wrap;width:100%;min-height:44px;padding:4px 8px;border:1px solid transparent;border-radius:8px;background:transparent;color:var(--text);font:inherit;font-size:12px;text-align:left;cursor:pointer}
.pr-row:hover{background:rgba(96,165,250,0.06)}
.pr-row:focus-visible{outline:2px solid #60a5fa;outline-offset:1px}
.pr-row[aria-expanded="true"]{border-color:rgba(96,165,250,0.45);background:rgba(96,165,250,0.08)}
.pr-lead{display:flex;flex-direction:column;gap:3px;flex:0 0 150px;min-width:0;font-size:11px;color:var(--text-dim);line-height:1.35}
.pr-lead b.t{color:var(--text);font-size:12px}
.pr-bars{display:flex;align-items:flex-end;gap:7px;flex:0 0 auto}
.pr-bar{display:flex;flex-direction:column;align-items:center;gap:2px;width:30px}
.pr-bar .n{font-size:11px;font-weight:700;font-variant-numeric:tabular-nums;color:var(--text-dim);line-height:1}
.pr-bar .track{height:34px;width:16px;display:flex;align-items:flex-end}
.pr-bar .fill{width:100%;border-radius:3px 3px 0 0;min-height:3px}
.pr-bar.clear .fill{background:#4ade80}.pr-bar.clear .n{color:#4ade80}
.pr-bar.below .fill{background:#64748b}
.pr-bar.plain .fill{background:#7c8aa5}
.pr-bar.best .fill{box-shadow:0 0 0 2px #fbbf24}.pr-bar.best .n{color:#fbbf24}
.pr-bar.low{opacity:.55}
.pr-bar .mk{width:12px;height:12px;border-radius:999px;box-sizing:border-box}
.pr-bar .mk.gold{background:#fbbf24}.pr-bar .mk.blue{background:rgba(96,165,250,0.35);border:1px solid #60a5fa}
.pr-prev{flex:1 1 200px;min-width:0;font-size:12px;color:var(--text);line-height:1.4}
.pr-prev small{color:#fbbf24;font-weight:800;margin-left:4px;font-size:11px}
.pr-caret{margin-left:auto;color:var(--text-dim);font-size:11px;white-space:nowrap}
.pr-detail{display:grid;grid-template-columns:repeat(${SLOTS},minmax(0,1fr));gap:6px;padding:6px 8px 4px}
.pr-cell{min-width:0;padding:6px 8px;border-radius:8px;background:var(--surface2);border:1px solid var(--border);display:flex;flex-direction:column;gap:2px;font-size:11px;color:var(--text-dim);line-height:1.35}
.pr-cell.best{border-color:rgba(251,191,36,0.7)}
.pr-line{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.pt-tbl tr.pt-r1 td{border-bottom:none;padding-bottom:3px}
.pt-tbl tr.pt-r2 td{padding:0 6px 8px;border-bottom:1px solid rgba(36,48,73,.75);overflow:visible;text-overflow:clip}
@media (max-width:1099px){.pr-detail{grid-template-columns:repeat(auto-fit,minmax(138px,1fr))}}
@media (max-width:768px){.pr-lead{flex:1 1 100%;flex-direction:row;align-items:center;flex-wrap:wrap;gap:3px 8px}.pr-prev{flex:1 1 100%}}
`;

function finishStyle(f?: number) {
  if (f === 1) return { background: "#fbbf24", color: "#111827", border: "1px solid #fbbf24" };
  if (f === 2 || f === 3) return { background: "rgba(96,165,250,0.22)", color: "#bfdbfe", border: "1px solid rgba(96,165,250,0.6)" };
  return { background: "transparent", color: "var(--text-dim)", border: "1px solid var(--border)" };
}

function RunCell({ r, label, best, bench }: { r: RecentRun; label: string; best: boolean; bench?: ClassBenchmark | null }) {
  const clear = clearsLine(r.value, bench);
  const course = [r.surface ? `${r.surface.slice(0, 1)}${r.distance ?? ""}` : r.distance ? `${r.distance}` : "", r.going ?? "", r.field ? `${r.field}頭` : ""].filter(Boolean).join(" ");
  const where = `${md(r.date)}${r.venue ? ` ${r.venue}` : ""}${r.round ? `${r.round}R` : ""}`;
  const sub = [r.low ? "低信頼" : "", r.l3f != null ? `上${r.l3f.toFixed(1)}` : "", r.passing ? `通過${r.passing}` : "", r.jw != null ? `${r.jw}kg` : "", r.jockey ?? "", r.bw ? `${r.bw}${r.bw_diff != null ? `(${r.bw_diff > 0 ? "+" : ""}${r.bw_diff})` : ""}` : ""].filter(Boolean).join(" ");
  return (
    <div className={`pr-cell${best ? " best" : ""}`} style={{ opacity: r.low ? 0.8 : 1 }}>
      <span className="pr-line"><b style={{ color: "var(--text)" }}>{label}</b> {where}</span>
      <span className="pr-line" style={{ color: "var(--text)", fontWeight: 600 }}>
        {r.race_name ?? "—"}{r.grade && /^G[123]$/.test(r.grade) ? <span style={{ marginLeft: 4, color: "#fbbf24", fontWeight: 800 }}>{r.grade}</span> : null}
      </span>
      <span className="pr-line">{course || "—"}</span>
      <span style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 2 }}>
        <span style={{ ...finishStyle(r.finish), minWidth: 34, textAlign: "center", borderRadius: 5, padding: "0 5px", fontWeight: 800, fontSize: 12, whiteSpace: "nowrap" }}>{r.finish ? `${r.finish}着` : "—"}</span>
        {r.pop ? <span style={{ whiteSpace: "nowrap" }}>{r.pop}人気</span> : null}
        <span style={{ marginLeft: "auto", textAlign: "right", lineHeight: 1, whiteSpace: "nowrap" }}>
          <b style={{ fontSize: 17, color: clear ? "#4ade80" : best ? "#fbbf24" : "#fff", fontVariantNumeric: "tabular-nums" }}>{r.value.toFixed(0)}</b>
          {clear && <span aria-label="突破ライン以上" style={{ marginLeft: 2, fontSize: 11, color: "#4ade80" }}>▲</span>}
        </span>
      </span>
      {sub && <span className="pr-line" title={sub} style={{ fontSize: 11 }}>{sub}</span>}
    </div>
  );
}

/** 1 頭分の直近5走。1 行（推移の棒＋前走の短文）で概要を見せ、押すとその馬の詳細カード（前走から順）を展開する。 */
export function PastRunStrip({ e, bench }: { e?: ExpectedMegu | null; bench?: ClassBenchmark | null }) {
  const [open, setOpen] = useState(false);
  const panelId = useId();
  const newest = (e?.recent ?? []).slice(0, SLOTS);
  const runs = chronological(e?.recent, SLOTS);                    // 古い→新しい（棒グラフ用）
  const best = bestIndex(runs);
  const mean = recentMean(e);
  const ps = passState(newest, bench);
  const prev = prevSummary(newest[0]);
  const line = bench?.line ?? null;
  if (!runs.length) return <div className="pr-wrap"><div className="pr-row" style={{ cursor: "default", opacity: 0.7 }}><div className="pr-lead"><b className="t">直近5走</b><span>初出走、または指数が未計算</span></div></div></div>;
  return (
    <div className="pr-wrap">
      <button type="button" className="pr-row" aria-expanded={open} aria-controls={panelId} onClick={() => setOpen((v) => !v)}
        title="押すと、5走それぞれのレース名・着順・上がりなどを開きます">
        <span className="pr-lead">
          <b className="t">直近5走 <span style={{ fontWeight: 400, color: "var(--text-dim)" }}>平均 {mean!.toFixed(0)}</span></b>
          {bench && ps.kind !== "none" ? (
            <span style={{ alignSelf: "flex-start", fontSize: 11, fontWeight: 700, padding: "0 7px", borderRadius: 999, whiteSpace: "nowrap",
              border: `1px solid ${ps.kind === "above" ? "rgba(74,222,128,0.6)" : "rgba(251,191,36,0.6)"}`, background: ps.kind === "above" ? "rgba(74,222,128,0.12)" : "rgba(251,191,36,0.1)",
              color: ps.kind === "above" ? "#4ade80" : "#fbbf24" }}>
              {ps.kind === "above" ? "基準超え" : "基準未満"} {ps.best!.toFixed(0)}/{bench.line.toFixed(0)}
            </span>
          ) : bench ? <span style={{ fontSize: 11 }}>基準と比べるデータなし</span> : null}
        </span>
        <span className="pr-bars" role="img" aria-label={`指数の推移（古い順）: ${runs.map(barLabel).join("、")}`}>
          {runs.map((r, i) => (
            <span key={r.date + i} className={`pr-bar ${barTone(r.value, line)}${i === best ? " best" : ""}${r.low ? " low" : ""}`} title={barLabel(r)}>
              <span className="n">{r.value.toFixed(0)}</span>
              <span className="track"><span className="fill" style={{ height: `${barHeightPct(r.value)}%` }} /></span>
              <span className={`mk ${finishMark(r.finish)}`} />
            </span>
          ))}
        </span>
        <span className="pr-prev">
          {prev ? <><span className="pr-line" style={{ display: "block" }}>{prev.text}{prev.grade && <small>{prev.grade}</small>}</span></> : null}
        </span>
        <span className="pr-caret" aria-hidden>{open ? "閉じる ▲" : "詳細 ▼"}</span>
      </button>
      {open && (
        <div id={panelId} className="pr-detail" role="group" aria-label="直近5走の詳細（前走から順）">
          {newest.map((r, i) => <RunCell key={i} r={r} label={LABELS[i]} best={i === (best < 0 ? -1 : runs.length - 1 - best)} bench={bench} />)}
        </div>
      )}
    </div>
  );
}

/** 表の上の説明行（1〜2 行）。定義・期間・注意は ⓘ を押したときに出す。 */
export function BenchNote({ bench }: { bench?: ClassBenchmark | null }) {
  const [open, setOpen] = useState(false);
  const id = useId();
  return (
    <div role="note" style={{ margin: "0 10px 8px", padding: "7px 10px", border: bench ? "1px solid rgba(74,222,128,0.35)" : "1px solid var(--border)", background: bench ? "rgba(74,222,128,0.07)" : "var(--surface2)", borderRadius: 8, fontSize: 12, lineHeight: 1.6, color: "var(--text)" }}>
      <span style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
        {bench ? <span><b>このクラスの基準（{benchmarkTitle(bench)}）</b>: {benchmarkSummary(bench)}</span> : <span><b>直近5走の指数</b>（過去の走の実測値）</span>}
        <button type="button" aria-expanded={open} aria-controls={id} onClick={() => setOpen((v) => !v)} aria-label="見方と注意を開く"
          style={{ minWidth: 28, minHeight: 28, borderRadius: 999, border: "1px solid var(--border)", background: "var(--surface2)", color: "var(--text-dim)", fontSize: 12, cursor: "pointer" }}>ⓘ</button>
      </span>
      {open && (
        <span id={id} style={{ display: "block", fontSize: 11, color: "var(--text-dim)", marginTop: 4 }}>
          各馬の行は、指数の推移（左が古く右が最新。棒は同じ目盛り、緑は突破ライン以上、金枠はその馬の最高。丸は1着=金・2〜3着=青）と前走の要約です。押すと5走それぞれの詳細が開きます。
          指数は過去の走の実測めぐ指数（1点＝0.1秒、斤量55kg換算）で、今回の指数の想定は出しません。
          {bench && <> 突破ライン＝そのクラスの2着馬の指数の下位25%（{bench.period ?? ""}）。勝ち馬ではなく2着馬の分布から作った目安で、突破を保証するものではありません。{bench.fallback ? "年齢まで一致する基準が少ないため、クラス全体の基準です。" : ""}</>}
        </span>
      )}
    </div>
  );
}
