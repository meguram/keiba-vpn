"use client";

import { useMemo } from "react";
import { bracketChipStyle } from "@/lib/bracket";
import { AnalysisEntry, AnalysisModuleMeta, RangeCell, layoutPositionMatrix } from "@/lib/raceAnalysis";

const LANE_H = 26;      // 1 ライン（ラチから外へ 1 頭分）の高さ(px)

const CSS = `
.pm{margin-top:10px}
.pm-h{display:flex;align-items:baseline;gap:8px;flex-wrap:wrap;font-size:13px;font-weight:700;margin-bottom:4px}
.pm-h small{font-size:11px;font-weight:400;color:var(--text-dim)}
.pm-board{position:relative;border-top:3px solid rgba(229,231,235,.75);border-bottom:1px dashed rgba(139,148,158,.35);background:rgba(36,48,73,.18)}
.pm-rail{position:absolute;top:-17px;left:0;font-size:10px;color:var(--text-dim)}
.pm-chip{position:absolute;display:flex;align-items:stretch;justify-content:center;padding:0 1px;box-sizing:border-box}
.pm-chip>span{width:100%!important;min-width:0!important;padding:0!important;font-size:11px!important}
.pm-ax{display:flex;justify-content:space-between;font-size:10px;color:var(--text-dim);margin-top:2px}
.pm-na{display:flex;align-items:center;gap:4px;flex-wrap:wrap;font-size:11px;color:var(--text-dim);margin-top:6px}
`;

/**
 * 位置取りマトリックス（隊列図）。横は直近の平均の位置（左が先頭）で 1 マス = 1 頭身、縦は上がラチ沿い。
 * 配置は layoutPositionMatrix（馬番の小さい順に、重ならない最もラチ側のラインへ）。
 */
export function PositionMatrix({ entries, module, rangeKey }: { entries: AnalysisEntry[]; module: AnalysisModuleMeta; rangeKey: string }) {
  const byKey = useMemo(() => new Map(entries.map((e) => [e.key, e])), [entries]);
  const lay = useMemo(
    () => layoutPositionMatrix(
      entries.map((e) => ({ key: e.key, num: e.horse_number, mean: (e.analysis?.[module.id]?.[rangeKey] as RangeCell | null | undefined)?.mean ?? null })),
      entries.length,
    ),
    [entries, module.id, rangeKey],
  );
  if (lay.placed.length === 0) return null;
  // 馬番が決まるまでは、内外の順（馬番の小さい馬がラチ側）を決められないので出さない
  if (entries.some((e) => e.horse_number == null)) {
    return <p style={{ margin: "10px 0 0", fontSize: 12, color: "var(--text-dim)" }}>位置取りマトリックスは、馬番が決まってから表示します。</p>;
  }
  const w = 100 / lay.field;
  return (
    <div className="pm">
      <style>{CSS}</style>
      <div className="pm-h">
        位置取りマトリックス
        <small>直近の平均の位置。1マス＝1頭身、上がラチ沿い。重なるときは馬番の小さい馬がラチ側</small>
      </div>
      <div style={{ paddingTop: 16 }}>
        <div
          className="pm-board"
          style={{ height: lay.lanes * LANE_H + 4, backgroundImage: `repeating-linear-gradient(to right, transparent 0, transparent calc(${w}% - 1px), rgba(139,148,158,.12) calc(${w}% - 1px), rgba(139,148,158,.12) ${w}%)` }}
          role="img"
          aria-label="位置取りマトリックス"
        >
          <span className="pm-rail">ラチ（内）</span>
          {lay.placed.map((p) => {
            const e = byKey.get(p.key);
            const cells = e?.analysis?.[module.id];
            const style = typeof cells?.style === "string" ? cells.style : null;
            const mean = (cells?.[rangeKey] as RangeCell | null | undefined)?.mean;
            return (
              <span
                key={p.key}
                className="pm-chip"
                style={{ left: `${(p.c - 0.5) * w}%`, width: `${w}%`, top: p.lane * LANE_H + 2, height: LANE_H - 4 }}
                title={`${e?.horse_number ?? "–"} ${e?.horse_name ?? ""}${style ? ` / ${style}` : ""}${typeof mean === "number" ? ` / 前から${Math.round(mean * 100)}%` : ""}`}
              >
                <span style={bracketChipStyle(e?.bracket_number, LANE_H - 4)}>
                  {e?.horse_number ?? "–"}
                </span>
              </span>
            );
          })}
        </div>
      </div>
      <div className="pm-ax"><span>◀ 先頭（進行方向）</span><span>最後方</span></div>
      {lay.unknown.length > 0 && (
        <div className="pm-na">
          位置データなし:
          {lay.unknown.map((h) => {
            const e = byKey.get(h.key);
            return <span key={h.key} style={bracketChipStyle(e?.bracket_number, 20)} title={e?.horse_name ?? undefined}>{e?.horse_number ?? "–"}</span>;
          })}
        </div>
      )}
    </div>
  );
}
