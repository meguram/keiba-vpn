"use client";

import { bracketChipStyle } from "@/lib/bracket";
import { benchmarkSummary, benchmarkTitle, clearsLine, passState } from "@/lib/classBenchmark";
import { md, recentMean, type ClassBenchmark, type ExpectedMegu, type RecentRun } from "@/lib/expectedMegu";

/* 馬柱（競馬新聞の 1 マス風）: 各馬の直近5走を、前走から順に、レース情報と実測のめぐ指数つきで並べる。
   指数は実測（過去走）だけ。今回の想定は出さない。 */

export type PastRunHorse = { no: number; name: string; bracket: number | null; e?: ExpectedMegu | null };

const SLOTS = 5;
const LABELS = ["前走", "2走前", "3走前", "4走前", "5走前"];

const CSS = `
.pr-wrap{display:grid;gap:8px}
.pr-row{display:grid;grid-template-columns:150px repeat(${SLOTS},minmax(0,1fr));gap:6px;align-items:stretch;padding:6px;border:1px solid var(--border);border-radius:10px;background:var(--surface)}
.pr-head{display:flex;flex-direction:column;gap:4px;justify-content:center;min-width:0}
.pr-cell{min-width:0;padding:6px 8px;border-radius:8px;background:var(--surface2);border:1px solid var(--border);display:flex;flex-direction:column;gap:2px;font-size:11px;color:var(--text-dim);line-height:1.35}
.pr-row.flash{outline:2px solid #60a5fa;transition:outline-color .6s}
.pr-cell.best{border-color:rgba(251,191,36,0.7);box-shadow:inset 0 0 0 1px rgba(251,191,36,0.25)}
.pr-cell.empty{border-style:dashed;align-items:center;justify-content:center;opacity:.6}
.pr-line{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
@media (max-width:900px){.pr-row{grid-template-columns:repeat(auto-fit,minmax(140px,1fr))}.pr-head{grid-column:1/-1;flex-direction:row;align-items:center;flex-wrap:wrap;gap:8px}.pr-cell.empty{display:none}}
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
  const title = [`${label}`, where, r.race_name, course, r.finish ? `${r.finish}着` : "", r.jockey, r.margin ? `着差 ${r.margin}` : "", r.low ? "（低信頼の指数）" : ""].filter(Boolean).join(" / ");
  return (
    <div className={`pr-cell${best ? " best" : ""}`} title={title} style={{ opacity: r.low ? 0.75 : 1 }}>
      <span className="pr-line"><b style={{ color: "var(--text)" }}>{label}</b> {where}</span>
      <span className="pr-line" style={{ color: "var(--text)", fontWeight: 600 }}>
        {r.race_name ?? "—"}{r.grade && /^G[123]$/.test(r.grade) ? <span style={{ marginLeft: 4, color: "#fbbf24", fontWeight: 800 }}>{r.grade}</span> : null}
      </span>
      <span className="pr-line">{course || "—"}</span>
      <span style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 2 }}>
        <span style={{ ...finishStyle(r.finish), minWidth: 34, textAlign: "center", borderRadius: 5, padding: "0 5px", fontWeight: 800, fontSize: 12 }}>{r.finish ? `${r.finish}着` : "—"}</span>
        {r.pop ? <span>{r.pop}人気</span> : null}
        <span style={{ marginLeft: "auto", textAlign: "right", lineHeight: 1 }}>
          <span style={{ fontSize: 10 }}>指数</span>{" "}
          <b style={{ fontSize: 17, color: clear ? "#4ade80" : best ? "#fbbf24" : "#fff", fontVariantNumeric: "tabular-nums" }}>{r.value.toFixed(0)}</b>
          {clear && <span aria-label="突破ライン以上" title={`突破ライン ${bench!.line.toFixed(0)} 以上`} style={{ marginLeft: 2, fontSize: 11, color: "#4ade80" }}>▲</span>}
        </span>
      </span>
      <span className="pr-line">{[r.l3f != null ? `上${r.l3f.toFixed(1)}` : "", r.passing ? `通過 ${r.passing}` : ""].filter(Boolean).join(" ") || " "}</span>
      <span className="pr-line">{[r.jw != null ? `${r.jw}kg` : "", r.jockey ?? "", r.bw ? `${r.bw}${r.bw_diff != null ? `(${r.bw_diff > 0 ? "+" : ""}${r.bw_diff})` : ""}` : ""].filter(Boolean).join(" ") || " "}</span>
      {r.low && <span style={{ fontSize: 10, color: "#fbbf24" }}>低信頼の指数</span>}
    </div>
  );
}

export function PastRuns({ horses, raceId, bench }: { horses: PastRunHorse[]; raceId: string; bench?: ClassBenchmark | null }) {
  return (
    <div className="pr-wrap" role="list" aria-label="馬柱（直近5走の指数）">
      <style>{CSS}</style>
      {bench && (
        <div role="note" style={{ padding: "8px 10px", border: "1px solid rgba(74,222,128,0.35)", background: "rgba(74,222,128,0.07)", borderRadius: 8, fontSize: 12, lineHeight: 1.6, color: "var(--text)" }}>
          <b>このクラスの基準（{benchmarkTitle(bench)}）</b>: {benchmarkSummary(bench)}
          <span style={{ display: "block", fontSize: 11, color: "var(--text-dim)" }}>
            突破ライン＝そのクラスの2着馬の指数の下位25%。これ以上なら、このクラスの標準的な水準にある目安です。{bench.fallback ? "（年齢まで一致する基準が少ないため、クラス全体の基準です）" : ""}
          </span>
        </div>
      )}
      {horses.map((h) => {
        const runs = (h.e?.recent ?? []).slice(0, SLOTS);
        const mean = recentMean(h.e);
        const bestIdx = runs.length ? runs.reduce((bi, r, i) => (r.value > runs[bi].value ? i : bi), 0) : -1;
        const ps = passState(runs, bench);
        return (
          <div key={h.no} id={`pr-${raceId}-${h.no}`} className="pr-row" role="listitem">
            <div className="pr-head">
              <span style={{ display: "flex", alignItems: "center", gap: 6, minWidth: 0 }}>
                <span style={bracketChipStyle(h.bracket, 24)}>{h.no}</span>
                <strong style={{ fontSize: 13, color: "#f0f6fc", overflowWrap: "anywhere" }}>{h.name}</strong>
              </span>
              {bench && ps.kind !== "none" && (
                <span title={`直近${runs.length}走の最高 ${ps.best!.toFixed(0)} と、このクラスの突破ライン ${bench.line.toFixed(0)} の比較`}
                  style={{ alignSelf: "flex-start", fontSize: 11, fontWeight: 700, padding: "1px 7px", borderRadius: 999, whiteSpace: "nowrap",
                    border: `1px solid ${ps.kind === "above" ? "rgba(74,222,128,0.6)" : "rgba(251,191,36,0.6)"}`, background: ps.kind === "above" ? "rgba(74,222,128,0.12)" : "rgba(251,191,36,0.1)",
                    color: ps.kind === "above" ? "#4ade80" : "#fbbf24" }}>
                  {ps.kind === "above" ? "超えている" : "届かない"} 最高 {ps.best!.toFixed(0)} {ps.kind === "above" ? "≥" : "<"} {bench.line.toFixed(0)}
                </span>
              )}
              {bench && ps.kind === "none" && <span style={{ alignSelf: "flex-start", fontSize: 11, color: "var(--text-dim)", border: "1px dashed var(--border)", borderRadius: 999, padding: "1px 7px" }}>基準と比べるデータなし</span>}
              <span style={{ fontSize: 11, color: "var(--text-dim)" }}>
                {runs.length ? <>直近{runs.length}走の平均 <b style={{ color: "var(--text)" }}>{mean!.toFixed(1)}</b>{runs.length > 1 ? ` ・最高 ${Math.max(...runs.map((r) => r.value)).toFixed(0)}` : ""}</> : "過去のめぐ指数なし"}
              </span>
            </div>
            {Array.from({ length: SLOTS }, (_, i) =>
              runs[i] ? <RunCell key={i} r={runs[i]} label={LABELS[i]} best={i === bestIdx && runs.length > 1} bench={bench} /> : <div key={i} className="pr-cell empty">{i === 0 && !runs.length ? "初出走・未計算" : "—"}</div>,
            )}
          </div>
        );
      })}
      <p style={{ margin: 0, fontSize: 11, color: "var(--text-dim)", lineHeight: 1.6 }}>
        指数は過去の走の実測めぐ指数（1点＝0.1秒、斤量55kg換算。金枠はその馬の直近の最高）。左から前走、2走前…の順。今回の指数の想定は出しません。{bench && <> 緑の数字と▲は、このクラスの突破ライン以上。突破ラインは2着馬の指数の分布から作った目安で、突破を保証するものではありません。</>}
      </p>
    </div>
  );
}
