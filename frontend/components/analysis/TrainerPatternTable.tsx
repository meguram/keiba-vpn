"use client";

import { OddsDist } from "@/components/OddsDistPopup";
import { bracketChipStyle } from "@/lib/bracket";
import type { AnalysisEntry } from "@/lib/raceAnalysis";
import { patHint, patWord, pct, rowStats, splitPat, workCell, workReason, type TPFigure, type TPHorse, type TPRow } from "@/lib/trainerPattern";

const CSS = `
.tp-scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
.tp{width:100%;border-collapse:collapse;font-size:12px;font-variant-numeric:tabular-nums}
.tp th,.tp td{padding:5px 6px;text-align:right;white-space:nowrap;vertical-align:middle}
.tp thead th{color:var(--text-dim);font-weight:600;font-size:11px;border-bottom:1px solid var(--border)}
.tp .l{text-align:left}
.tp tbody td{border-bottom:1px solid rgba(36,48,73,.35)}
.tp td.hn{text-align:left;vertical-align:top;padding-top:7px;min-width:9em;max-width:12em;white-space:normal}
.tp td.hn b{display:block;font-size:13px;color:#fff}
.tp td.hn small{display:block;font-size:10.5px;color:var(--text-dim);margin-top:1px}
.tp tr.pt td{background:rgba(59,130,246,.10)}
.tp tr.pt td.lv{font-weight:700;color:var(--accent)}
.tp td.lv{text-align:left;color:var(--text-dim)}
.tp .up{color:#4ade80;font-weight:700}
.tp .na{color:var(--text-dim);text-align:left;white-space:normal}
.tp td.hn{position:sticky;left:0;z-index:1;background:var(--surface)}
.tp tr.pt td.hn{background:linear-gradient(rgba(59,130,246,.10),rgba(59,130,246,.10)),var(--surface)}
.tp tr.thin td:not(.hn):not(.lv),.tp tr.thin td.up{color:var(--text-dim);font-weight:400}
.tp tr.thin td:not(.hn):not(.lv) > *{opacity:.55}
.tp .tag{display:inline-block;font-size:10px;line-height:1.5;padding:0 5px;margin-left:4px;border:1px solid var(--border);border-radius:8px;color:var(--text-dim);white-space:nowrap}
@media(max-width:640px){.tp td.hn{min-width:8.5em;max-width:9em;font-size:11px}.tp td.hn b{font-size:12px}.tp td.lv{white-space:normal;min-width:6.5em}.tp .xs{display:none}}
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

function reason(h: TPHorse, code: string | null): string {
  if (!h.trainer_id) return "調教師が不明です。";
  const noTrainer = "この調教師は、表（2020〜2025・単勝50倍未満）に出走がありません。";
  const work = !h.pat ? workReason(h.wk) ?? "レース週の平日に 4F の追い切りの記録がありません（出走前のレースは前日夕方に取得します）。" : null;
  if (!h.levels) return work ? `${work}${noTrainer}` : noTrainer;
  if (work) return work;
  if (!code) return "追い切りのパターンを判定できませんでした。";
  return "この調教師は、同じ場所・同じパターンでの出走がありません。";
}

/** その馬の調教師が、今回と同じ最終追い切りのパターン（場所・パターン）で出した成績だけを 1 行で見せる。 */
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
            <th className="l">馬 / 調教師</th><th className="l">最終追い切りのパターン</th><th>出走</th><th className="xs">頭数</th><th>着順内訳</th>
            <th>勝率</th><th>複勝率</th><th>単勝回収率</th><th>複勝回収率</th><th title="オッズ区間別の出走数。カーソルを合わせる／タップで詳細">オッズ分布</th>
          </tr>
        </thead>
        <tbody>
          {horses.map((h) => {
            const e = h.key ? byKey.get(h.key) : undefined;
            const place = h.pat ? splitPat(h.pat).place : null;
            const code = h.pat ? splitPat(h.pat).pat : null;
            const pat = code ? patWord(code) : null;
            const hint = patHint(code) || undefined;
            const row = (code && h.levels?.pattern) || null;
            const thin = row != null && rowStats(row).n < THIN;
            return (
              <tr key={h.key ?? h.horse_number} className={row ? (thin ? "pt thin" : "pt") : undefined}
                title={thin ? `出走が ${THIN} 走に満たないので参考程度です` : undefined}>
                <td className="hn">
                  <span style={{ ...bracketChipStyle(e?.bracket_number, 22), marginRight: 6 }}>{h.horse_number ?? "–"}</span>
                  <b style={{ display: "inline" }}>{h.horse_name}</b>
                  <small>{h.trainer_name ?? ""}</small>
                </td>
                {pat
                  ? <td className="lv" title={hint}>{`${place}・${pat}`}</td>
                  : (() => { const c = workCell(h.wk); return <td className="lv">{c.text}{c.tag && <span className="tag">{c.tag}</span>}</td>; })()}
                {row
                  ? <Cells r={row} edges={edges} maxOdds={maxOdds} title={`${h.trainer_name ?? ""} ${place ?? ""} ${pat ?? ""}`} />
                  : <><td className="na" colSpan={7}>{reason(h, code)}</td><td className="xs" /></>}
              </tr>
            );
          })}
        </tbody>
      </table>
      <p style={{ margin: "6px 0 0", fontSize: 11, color: "var(--text-dim)" }}>
        パターン名にカーソルを合わせる（タップ）と定義が出ます。出走が {THIN} 走に満たない行は薄く表示しています（参考程度）。「集計の対象外」は、追い切りはあるものの、集計に使う種別（レース週の月〜金の 4F の時計）ではない馬です。表の集計は {figure.y0}〜{figure.y1} 年（単勝 {figure.max_odds} 倍以上は除く）。
      </p>
    </div>
  );
}
