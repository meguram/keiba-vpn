"use client";

import { useEffect, useMemo, useState } from "react";
import { BenchNote, PAST_RUNS_CSS, PastRunStrip } from "@/components/PastRuns";
import { bracketChipStyle, bracketOf } from "@/lib/bracket";
import type { ClassBenchmark, ExpectedMegu, ExpectedMeguMeta, MeguMark } from "@/lib/expectedMegu";
import { markExplain, markSortValue, meguSortValue, type MeguSortKey, type TodayCourse } from "@/lib/pastRunsView";

/* めぐ指数専用のテーブル。出馬表（勝率・期待値など）とは別のブロックにして、出馬表の行を縦長にしない。
   各馬の直近5走の指数（古→新の棒・着順）と、押すと開く5走の詳細、クラスの突破ラインとの比較。指数は実測（過去走）だけ。 */

export type MeguRow = {
  horse_number?: number;
  horse_name?: string;
  bracket_number?: number | null;
  expected_megu?: ExpectedMegu | null;
  megu_mark?: MeguMark | null;
};

const STORE_KEY = "wp-megu-open";
const SORTS: { key: MeguSortKey; label: string; title: string }[] = [
  { key: "num", label: "馬番", title: "馬番順（既定）" },
  { key: "mark", label: "めぐ印", title: "めぐ印（◎→○→印なし）の順。同じ印の中は集約値の高い順" },
  { key: "mean", label: "平均", title: "直近5走の平均が高い順" },
  { key: "best", label: "最高", title: "直近5走の最高が高い順" },
  { key: "clear", label: "基準との差", title: "今回と同じ条件の走の最高 − 突破ライン が大きい順" },
];

const CSS = `
.mg-sortbar{display:flex;flex-wrap:wrap;align-items:center;gap:6px;padding:0 10px 8px;font-size:12px;color:var(--text-dim)}
.mg-chip{min-height:36px;padding:0 12px;border-radius:999px;border:1px solid var(--border);background:var(--surface2);color:var(--text);font-size:12px;cursor:pointer}
.mg-chip[aria-pressed="true"]{border-color:var(--accent);background:rgba(59,130,246,0.15);color:#fff}
.mg-tbl{width:100%;border-collapse:collapse;font-size:12px}
.mg-tbl th{padding:8px 10px;font-size:11px;font-weight:600;color:var(--text-dim);text-align:left;background:var(--surface2);border-bottom:1px solid var(--border)}
.mg-tbl td{padding:6px 8px;border-bottom:1px solid rgba(36,48,73,.75);vertical-align:middle}
.mg-tbl td.c{text-align:center}
.mg-nm{font-weight:600;color:#fff;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.mg-mark{display:inline-flex;align-items:center;justify-content:center;min-width:44px;height:26px;padding:0 8px;border-radius:999px;font-size:12px;font-weight:700;border:1px solid var(--border);color:var(--text-dim);background:transparent;cursor:help;white-space:nowrap}
.mg-mark.top{color:#fff;background:#b45309;border-color:#f59e0b}
.mg-mark.good{color:#fde68a;background:rgba(245,158,11,.14);border-color:rgba(245,158,11,.6)}
.mg-mark small{font-weight:400;font-size:10px;margin-left:4px;opacity:.85}
.mg-legend{font-size:11px;color:var(--text-dim);padding:2px 10px 8px;line-height:1.6}
.mg-mean{font-variant-numeric:tabular-nums;color:var(--text);font-weight:700;text-align:right;white-space:nowrap}
.mg-mean small{display:block;font-weight:400;font-size:10px;color:var(--text-dim)}
@media (max-width:768px){
  .mg-tbl thead{display:none}
  .mg-tbl,.mg-tbl tbody{display:block}
  .mg-tbl tr{display:grid;grid-template-columns:auto 1fr auto;align-items:center;column-gap:8px;border-bottom:1px solid rgba(36,48,73,.75);padding:6px 0}
  .mg-tbl td{border-bottom:none;padding:2px 6px}
  .mg-tbl td.strip{grid-column:1 / -1;grid-row:2}
  .mg-tbl td.mg-mean{grid-column:3;grid-row:1}
  .mg-tbl td.mgm{grid-column:2;grid-row:1;justify-self:end}
}
`;

export function MeguTable({ preds, bench, today, meta }: { preds: MeguRow[]; bench?: ClassBenchmark | null; today?: TodayCourse | null; meta?: ExpectedMeguMeta | null }) {
  const [open, setOpen] = useState(true);
  const [sort, setSort] = useState<MeguSortKey>("num");
  useEffect(() => {
    try { if (window.localStorage.getItem(STORE_KEY) === "0") setOpen(false); } catch { /* 記憶できなくても動く */ }
  }, []);
  const toggle = () => setOpen((v) => {
    try { window.localStorage.setItem(STORE_KEY, v ? "0" : "1"); } catch { /* 記憶できなくても動く */ }
    return !v;
  });

  const line = bench?.line ?? null;
  const rows = useMemo(() => {
    const base = [...preds].filter((p) => p.horse_number != null);
    const byNum = (a: MeguRow, b: MeguRow) => (a.horse_number ?? 0) - (b.horse_number ?? 0);
    if (sort === "num") return base.sort(byNum);
    const val = (p: MeguRow) => (sort === "mark" ? markSortValue(p.megu_mark) : meguSortValue(p.expected_megu?.recent, sort, line, today));
    return base.sort((a, b) => {
      const va = val(a), vb = val(b);
      if (va == null && vb == null) return byNum(a, b);
      if (va == null) return 1;
      if (vb == null) return -1;
      return vb - va || byNum(a, b);
    });
  }, [preds, sort, line, today]);

  const withRuns = rows.filter((p) => p.expected_megu?.recent?.length);
  if (!withRuns.length) return null;      // 全馬が初出走など、指数がまったく無いレースは表を出さない
  const bestByMean = [...withRuns].sort((a, b) => (meguSortValue(b.expected_megu?.recent, "mean") ?? 0) - (meguSortValue(a.expected_megu?.recent, "mean") ?? 0))[0];
  const bestMean = meguSortValue(bestByMean.expected_megu?.recent, "mean");
  const nTop = rows.filter((p) => p.megu_mark?.key === "top").length;
  const nGood = rows.filter((p) => p.megu_mark?.key === "good").length;
  const anyMarkInfo = rows.some((p) => p.megu_mark);

  return (
    <section aria-label="めぐ指数" style={{ marginTop: 14, border: "1px solid var(--border)", borderRadius: 10, background: "var(--surface)", overflow: "hidden" }}>
      <style>{PAST_RUNS_CSS + CSS}</style>
      <button type="button" aria-expanded={open} onClick={toggle}
        style={{ display: "flex", alignItems: "center", gap: 10, width: "100%", minHeight: 44, padding: "8px 12px", border: "none", background: "transparent", color: "var(--text)", cursor: "pointer", textAlign: "left", font: "inherit" }}>
        <b style={{ fontSize: 14 }}>めぐ指数（直近5走）</b>
        <span style={{ fontSize: 11, color: "var(--text-dim)" }}>
          {withRuns.length}頭{bestMean != null && ` ・ 平均が最高 ${bestByMean.horse_number}番 ${bestMean.toFixed(1)}`}
        </span>
        <span aria-hidden style={{ marginLeft: "auto", fontSize: 11, color: "var(--text-dim)" }}>{open ? "閉じる ▲" : "開く ▼"}</span>
      </button>
      {open && (
        <div>
          <BenchNote bench={bench} />
          {anyMarkInfo && (
            <p className="mg-legend">
              <b>めぐ印</b>: 過去走のめぐ指数（直近5走を、新しい走ほど重く・低信頼の走は半分の重みで集約）が、<b>レース内1位で突破ライン以上＝めぐ◎</b>、<b>2〜3位で突破ライン以上＝めぐ○</b>。
              AI の予測とは別の観点の目安です。過去の検証では◎の3着内率は約48%（出走馬全体の平均の約2倍）ですが、AI の予測が高い馬とかなり重なり、AI の予測に足せる新しい情報は小さいです（印だけで買う根拠にはしないでください）。
              {nTop + nGood > 0 ? ` このレース: ◎${nTop}頭・○${nGood}頭。` : " このレースは印なし（基準なし・出走が少ない・過去走なし・突破ライン未満など）。"}
            </p>
          )}
          <div className="mg-sortbar" role="group" aria-label="めぐ指数の並び替え">
            <span>並び替え</span>
            {SORTS.map((s) => (
              <button key={s.key} type="button" className="mg-chip" title={s.title} aria-pressed={sort === s.key} onClick={() => setSort(s.key)}>{s.label}</button>
            ))}
          </div>
          <table className="mg-tbl">
            <colgroup><col style={{ width: 60 }} /><col style={{ width: "20%" }} />{anyMarkInfo && <col style={{ width: 84 }} />}<col /><col style={{ width: 72 }} /></colgroup>
            <thead>
              <tr><th className="c" style={{ textAlign: "center" }}>枠・馬番</th><th>馬名</th>{anyMarkInfo && <th className="c" style={{ textAlign: "center" }}>めぐ印</th>}<th>直近5走の指数（古→新）</th><th style={{ textAlign: "right" }}>平均</th></tr>
            </thead>
            <tbody>
              {rows.map((p) => {
                const bn = bracketOf(p.horse_number ?? 0, preds.length, p.bracket_number);
                const mean = meguSortValue(p.expected_megu?.recent, "mean");
                return (
                  <tr key={p.horse_number}>
                    <td className="c"><span title={`${bn ?? "-"}枠 ${p.horse_number}番`} style={bracketChipStyle(bn, 24)}>{p.horse_number}</span></td>
                    <td><div className="mg-nm" title={p.horse_name}>{p.horse_name ?? "-"}</div></td>
                    {anyMarkInfo && (
                      <td className="c mgm">
                        <span className={`mg-mark ${p.megu_mark?.key ?? ""}`} title={markExplain(p.megu_mark)} aria-label={markExplain(p.megu_mark)}>
                          {p.megu_mark?.label ?? "—"}
                          {p.megu_mark?.rank != null && <small>{p.megu_mark.rank}位</small>}
                        </span>
                      </td>
                    )}
                    <td className="strip"><PastRunStrip e={p.expected_megu} bench={bench} today={today} /></td>
                    <td className="mg-mean">{mean != null ? <>{mean.toFixed(1)}<small>平均</small></> : "—"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          {meta?.data_through && (
            <p style={{ fontSize: 11, color: "var(--text-dim)", padding: "6px 10px 8px", margin: 0 }}>
              指数データは {meta.data_through.slice(5).replace("-", "/")} まで（直近の週末分は、確定結果の反映が数日遅れます）。
            </p>
          )}
        </div>
      )}
    </section>
  );
}
