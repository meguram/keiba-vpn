"use client";

import { Fragment } from "react";
import { OddsDist } from "@/components/OddsDistPopup";
import { bracketChipStyle } from "@/lib/bracket";
import type { AnalysisEntry } from "@/lib/raceAnalysis";
import { patWord, pct, rowStats, splitPat, type TPFigure, type TPHorse, type TPRow } from "@/lib/trainerPattern";

const CSS = `
.tp-scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
.tp{width:100%;border-collapse:collapse;font-size:12px;font-variant-numeric:tabular-nums}
.tp th,.tp td{padding:5px 6px;text-align:right;white-space:nowrap;vertical-align:middle}
.tp thead th{color:var(--text-dim);font-weight:600;font-size:11px;border-bottom:1px solid var(--border)}
.tp .l{text-align:left}
.tp tbody td{border-bottom:1px solid rgba(36,48,73,.35)}
.tp tbody.hs tr:last-child td{border-bottom:1px solid var(--border)}
.tp td.hn{text-align:left;vertical-align:top;padding-top:7px;min-width:9em;max-width:12em;white-space:normal}
.tp td.hn b{display:block;font-size:13px;color:#fff}
.tp td.hn small{display:block;font-size:10.5px;color:var(--text-dim);margin-top:1px}
.tp tr.pt td{background:rgba(59,130,246,.10)}
.tp tr.pt td.lv{font-weight:700;color:var(--accent)}
.tp td.lv{text-align:left;color:var(--text-dim)}
.tp .up{color:#4ade80;font-weight:700}
.tp .na{color:var(--text-dim);text-align:left;white-space:normal}
.tp td.hn{position:sticky;left:0;z-index:1;background:var(--surface)}
@media(max-width:640px){.tp td.hn{min-width:0;max-width:7em;font-size:11px}.tp td.hn b{font-size:12px}.tp .xs{display:none}}
`;

const THIN = 30;

function Cells({ r, edges, maxOdds, title }: { r: TPRow; edges: number[]; maxOdds: number; title: string }) {
  const s = rowStats(r);
  const roi = (v: number | null) => <td className={v != null && v > 1 ? "up" : ""}>{pct(v)}</td>;
  return (
    <>
      <td>{s.n}</td>
      <td className="xs">{s.heads ?? "–"}</td>
      <td>{s.c1}-{s.c2}-{s.c3}-{s.c4p}</td>
      <td>{pct(s.win)}</td>
      <td>{pct(s.place)}</td>
      {roi(s.winRoi)}
      {roi(s.placeRoi)}
      <td style={{ textAlign: "center" }}><OddsDist row={r} edges={edges} maxOdds={maxOdds} title={title} /></td>
    </>
  );
}

function reason(h: TPHorse): string {
  if (!h.trainer_id) return "調教師が不明です。";
  if (!h.levels) return h.pat ? "この調教師は、表（2020〜2025・単勝50倍未満）に出走がありません。" : "調教の記録がなく、調教師も表にありません。";
  return "";
}

export function TrainerPatternTable({ figure, entries }: { figure: TPFigure; entries: AnalysisEntry[] }) {
  const byKey = new Map(entries.map((e) => [e.key, e]));
  const horses = [...figure.horses].sort((a, b) => (a.horse_number ?? 99) - (b.horse_number ?? 99));
  const { edges, max_odds: maxOdds } = figure;
  return (
    <div className="tp-scroll" style={{ marginTop: 8 }}>
      <style>{CSS}</style>
      <table className="tp">
        <thead>
          <tr>
            <th className="l">馬 / 今週のパターン</th><th className="l">段</th><th>出走</th><th className="xs">頭数</th><th>着順内訳</th>
            <th>勝率</th><th>複勝率</th><th>単勝回収率</th><th>複勝回収率</th><th title="オッズ区間別の出走数。カーソルを合わせる／タップで詳細">オッズ分布</th>
          </tr>
        </thead>
        {horses.map((h) => {
          const e = h.key ? byKey.get(h.key) : undefined;
          const lv = h.levels;
          const place = h.pat ? splitPat(h.pat).place : null;
          const pat = h.pat ? patWord(splitPat(h.pat).pat) : null;
          const rows: { cls?: string; label: string; r: TPRow | null; title: string; miss: string }[] = lv ? [
            { label: "調教師全体", r: lv.trainer, title: `${h.trainer_name ?? ""} 全体`, miss: "" },
            { label: place ? `同じ場所（${place}）` : "同じ場所", r: lv.place, title: `${h.trainer_name ?? ""} ${place ?? ""}`, miss: h.pat ? "この場所の出走なし" : "調教の記録なし" },
            { cls: "pt", label: pat ? `同じパターン（${pat}）` : "同じパターン", r: lv.pattern, title: `${h.trainer_name ?? ""} ${place ?? ""} ${pat ?? ""}`, miss: h.pat ? "このパターンの出走なし" : "調教の記録なし" },
          ] : [];
          return (
            <tbody key={h.key ?? h.horse_number} className="hs">
              {rows.length === 0 ? (
                <tr>
                  <td className="hn">
                    <span style={{ ...bracketChipStyle(e?.bracket_number, 22), marginRight: 6 }}>{h.horse_number ?? "–"}</span>
                    <b style={{ display: "inline" }}>{h.horse_name}</b>
                    <small>{h.trainer_name ?? ""}{pat ? ` ／ ${place}・${pat}` : ""}</small>
                  </td>
                  <td className="na" colSpan={9}>{reason(h)}</td>
                </tr>
              ) : rows.map((x, i) => (
                <Fragment key={i}>
                  <tr className={x.cls}>
                    {i === 0 && (
                      <td className="hn" rowSpan={3}>
                        <span style={{ ...bracketChipStyle(e?.bracket_number, 22), marginRight: 6 }}>{h.horse_number ?? "–"}</span>
                        <b style={{ display: "inline" }}>{h.horse_name}</b>
                        <small>{h.trainer_name ?? ""}</small>
                        <small>{pat ? `${place}・${pat}` : "調教の記録なし"}</small>
                      </td>
                    )}
                    <td className="lv">{x.label}</td>
                    {x.r
                      ? <Cells r={x.r} edges={edges} maxOdds={maxOdds} title={x.title} />
                      : <td className="na" colSpan={8}>{x.miss}</td>}
                  </tr>
                </Fragment>
              ))}
            </tbody>
          );
        })}
      </table>
      <p style={{ margin: "6px 0 0", fontSize: 11, color: "var(--text-dim)" }}>
        出走が {THIN} 走に満たない段は参考程度です。表の集計は {figure.y0}〜{figure.y1} 年（単勝 {figure.max_odds} 倍以上は除く）。
      </p>
    </div>
  );
}
