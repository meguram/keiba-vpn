"use client";

import { PRED_TTL_MS, fetchRaceList, getJson, hasFreshJson, prefetchJson, readStored } from "@/lib/httpCache";
import { PredPrefetcher } from "@/lib/predPrefetch";
import { useEffect, useState, useCallback, useRef, useMemo } from "react";
import Link from "next/link";
import { USE_MOCK, MOCK_WEEKLY_RACES, getMockRaceDates, getMockPredictions } from "@/lib/mock";
import { featureEnabled } from "@/lib/features";
import { useAuthStatus } from "@/lib/hooks/useAuthStatus";
import { MemberLockNote, PublicModeNote } from "@/components/upgrade/MemberLock";
import { LEVEL_STYLE, RaceValue, prefetchRaceValues, useRaceValuesState, valueTitle } from "@/lib/raceValue";
import { LoadingNotice } from "@/components/LoadingNotice";
import { useAutoRetry } from "@/lib/hooks/useAutoRetry";
import { PredictionSourceBadge } from "@/components/PredictionSourceBadge";
import { WeekPicker } from "@/components/WeekPicker";
import { sameRoundIn } from "@/lib/raceGrade";
import { RaceStrip } from "@/components/RaceStrip";
import { CardTab, EntryCardPanel } from "@/components/EntryCardPanel";
import { MarkNote } from "@/components/MarkNote";
import { MeguTable } from "@/components/MeguTable";
import { WeeklyBetTab } from "@/components/WeeklyBetTab";
import { RaceResult } from "@/components/RaceResult";
import { useRaceResult } from "@/lib/hooks/useRaceResult";
import ProbCalNote from "@/components/ProbCalNote";
import { EntriesDaySummary, EntriesStatusBadge } from "@/components/EntriesStatus";
import type { EntriesStatus } from "@/lib/entriesStatus";
import type { ProbCalibration } from "@/lib/probCalibration";
import { PredExtras } from "@/components/PredExtras";
import type { L3fCoherent, PaceUnified } from "@/components/PredExtras";
import type { LapForecast } from "@/lib/lapForecast";
import { bracketChipStyle, bracketOf } from "@/lib/bracket";
import type { GateBias, PaceForecast, PositionForecast as PosFc, TrackingV2 } from "@/lib/positionForecast";
import type { PaceType, ScenarioForecast } from "@/lib/paceType";
import type { ClassBenchmark, ExpectedMegu, ExpectedMeguMeta } from "@/lib/expectedMegu";
import { firstDateOf, meetingKeyOf, pickTwoWeeks, todayJst, venuesOf } from "@/lib/weekDates";

const SHOW_HORSE_LINK = featureEnabled("horse-detail");
/* ── 型 ── */
import { turnOf } from "@/lib/courseTurn";
import { IntentLink } from "@/components/IntentLink";

type RaceItem = {
  direction?: string;
  race_id: string;
  race_name?: string;
  venue?: string;
  round?: number | string;
  date?: string;
  distance?: string | number;
  surface?: string;
  grade?: string;
  field_size?: number;
  start_time?: string;
  track_condition?: string;
  entries_count?: number;
  list_grade_icon?: string | null;
  /** 出馬表の公示の段階（収集側が判定）。無いレースは出馬表待ち */
  entries_phase?: string | null;
  entries_label?: string | null;
  entries_updated_at?: string | null;
};

const MARK_SYMBOL: Record<string, string> = { honmei: "◎", pair: "○", show_val: "▲", anchor: "△", star: "★", kesi: "消" };
/** 印の並び（ユーザーの指定 2026-11-25）: ◎ > ○ > ▲ > ★ > △ > 無印 > 消 */
const MARK_ORDER: Record<string, number> = { "◎": 0, "○": 1, "▲": 2, "★": 3, "△": 4, "": 5, "消": 6 };
const markOf = (p: { mark?: string; mark_type?: string }) => p.mark || MARK_SYMBOL[p.mark_type ?? ""] || "";

type PredEntry = {
  horse_number?: number;                 // 馬番が未確定の段階（特別登録・出走馬決定）は API が null を返す（画面は『-』を出す）
  /** 馬番が未確定の段階（特別登録・出走馬決定）では true */
  horse_number_provisional?: boolean;
  horse_name?: string;
  horse_id?: string;
  bracket_number?: number | null;
  sex_age?: string | null;
  jockey_name?: string | null;
  jockey_weight?: number | null;
  trainer_name?: string | null;
  weight?: number | null;
  weight_change?: number | null;
  popularity?: number | null;
  /** 印の語と記号。匿名・無料会員には予測の数値の代わりにこれだけが来る（src/api/tier.py） */
  mark_type?: string;
  mark?: string;
  pred_rank?: number;
  composite_rank?: number;
  win_prob?: number | null;
  top2_prob?: number | null;
  top3_prob?: number | null;
  /** 市場補正（KEIBA_PROB_CALIBRATION=1）した馬の、元のモデルの確率 */
  win_prob_model?: number | null;
  top2_prob_model?: number | null;
  top3_prob_model?: number | null;
  ev_win?: number | null;
  ev_place?: number | null;
  /** EV の較正が有効なときの、較正後の値（過去の実績に合わせて補正した期待回収率）。あれば画面はこちらを表示する */
  ev_win_cal?: number | null;
  ev_place_cal?: number | null;
  expected_value?: number | null;
  win_odds?: number | null;
  place_odds_min?: number | null;
  place_odds_max?: number | null;
  buy_tier?: string;
  expected_megu?: ExpectedMegu | null;
  position_forecast?: PosFc | null;
  l3f_coherent?: L3fCoherent | null;
  tracking_v2?: TrackingV2 | null;
};

type PredData = {
  access?: { public_mode?: boolean; withheld?: string[] };
  status?: string;
  has_prediction?: boolean;
  model_description?: string;
  prediction_source?: string;
  prediction_note?: string;
  message?: string;
  total_horses?: number;
  predictions?: PredEntry[];
  expected_megu_meta?: ExpectedMeguMeta | null;
  pace_forecast?: PaceForecast | null;
  pace_unified?: PaceUnified | null;
  class_benchmark?: ClassBenchmark | null;
  lap_forecast?: LapForecast | null;
  pace_type?: PaceType | null;
  scenario_forecast?: ScenarioForecast | null;
  gate_bias?: GateBias | null;
  prob_calibration?: ProbCalibration | null;
  /** 出馬表の公示の段階と最終更新 */
  entries_status?: EntriesStatus | null;
  provisional_entries?: boolean;
  error?: string;
};

/* ── ユーティリティ ── */

function fmtPct(v: number | null | undefined): string {
  if (v == null) return "-";
  return `${(v * 100).toFixed(1)}%`;
}

function gradeChip(grade: string | undefined) {
  if (!grade) return null;
  const s =
    grade === "G1" ? { bg: "rgba(239,68,68,0.18)", color: "#ef4444" } :
    grade === "G2" ? { bg: "rgba(59,130,246,0.18)", color: "#60a5fa" } :
    grade === "G3" ? { bg: "rgba(34,197,94,0.18)", color: "#22c55e" } :
    null;
  if (!s) return null;
  return (
    <span style={{ fontSize: 10, fontWeight: 700, padding: "2px 7px", borderRadius: 4, marginLeft: 6, ...s }}>{grade}</span>
  );
}

/* ── 列での並べ替え（表に出している値だけ） ── */
type SortKey = "num" | "mark" | "name" | "bracket" | "win" | "top3" | "odds" | "podds";
type SortDir = "asc" | "desc";

/** 列の値。数値は数、文字は文字。値が無い（"-" 表示）ときは null（並べ替えでは常に末尾） */
const SORT_VALUE: Record<SortKey, (p: PredEntry) => number | string | null> = {
  num: (p) => p.horse_number ?? null,
  mark: (p) => MARK_ORDER[markOf(p)] ?? MARK_ORDER[""],              // 無印も値（消より前）にする
  name: (p) => p.horse_name ?? null,
  bracket: (p) => p.bracket_number ?? null,
  win: (p) => p.win_prob ?? null,
  top3: (p) => p.top3_prob ?? null,
  odds: (p) => p.win_odds ?? null,
  podds: (p) => p.place_odds_min ?? null,
};
// 『大きいほど良い』列は、最初のクリックを降順にする（勝率・複勝率）
const DESC_FIRST = new Set<SortKey>(["win", "top3"]);

function sortPreds(preds: PredEntry[], key: SortKey | null, dir: SortDir, fallback: (a: PredEntry, b: PredEntry) => number): PredEntry[] {
  const out = [...preds].sort(fallback);                                   // 同じ値のときは既定の並び（安定）
  if (!key) return out;
  const val = SORT_VALUE[key];
  return out.sort((a, b) => {
    const x = val(a), y = val(b);
    if (x == null && y == null) return 0;
    if (x == null) return 1;                                              // 値が無いものは昇順でも降順でも末尾
    if (y == null) return -1;
    const c = typeof x === "number" && typeof y === "number" ? x - y : String(x).localeCompare(String(y), "ja");
    return dir === "asc" ? c : -c;
  });
}

/* AI 予測の表は、どの幅でも同じ 7 列（枠・馬番・印・馬名・AI勝率・AI複勝率・予想オッズ）。期待値は出さない（2026-11-25 ユーザーの判断）。
   予想オッズは 1 列に 2 段（上=単勝・下=複勝）にして、スマホ幅（390px）でも横スクロールなしで収める。
   さらに狭い端末（表の幅 360px 未満）では表の枠の中だけ横に送れ、枠・馬番・印・馬名は左に固定する（sticky）。
   騎手・性齢・斤量・馬体重・連対率・買い判断は表に出さない（馬名を押した情報パネルと、レース詳細で見られる。
   docs/operations/weekly_predictions_table.md）。
   色はサイトの変数（--text・--border・--surface）を使う（サイトはダークのみ。ライトの配色ができたら変数で追従する）。 */
const TABLE_CSS = `
.pt-tap{all:unset;cursor:pointer;display:inline-block;max-width:100%;padding:6px 0;border-bottom:1px dotted rgba(148,163,184,.55);font:inherit;font-weight:inherit;line-height:1.3;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;vertical-align:bottom;box-sizing:border-box}
.pt-tap:hover{border-bottom-color:var(--accent)}
.pt-tap:focus-visible{outline:2px solid var(--accent);outline-offset:2px;border-radius:3px}
.pt-sort{padding:0 !important}
.pt-sortbtn{all:unset;box-sizing:border-box;cursor:pointer;display:flex;align-items:center;justify-content:center;gap:1px;width:100%;min-height:44px;padding:0 3px;font:inherit;color:inherit;overflow:hidden;line-height:1.2}
.pt-sortbtn:hover{background:rgba(148,163,184,.08)}
.pt-sortbtn[data-align="left"]{justify-content:flex-start}
.pt-sortbtn[data-align="right"]{justify-content:flex-end;text-align:right}
.pt-sortbtn:focus-visible{outline:2px solid var(--accent);outline-offset:-2px}
.pt-sortbtn > span:first-child{word-break:keep-all}
.pt-sortbtn small{display:block;font-size:10px;font-weight:400;opacity:.85}
.pt-scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
.pt-tbl{width:100%;min-width:360px;table-layout:fixed;border-collapse:separate;border-spacing:0;font-size:12px}
.pt-tbl th{background:var(--surface2);border-bottom:1px solid var(--border);color:var(--text-dim);font-size:11px;font-weight:600}
.pt-tbl td{overflow:hidden;padding:5px 4px;border-bottom:1px solid var(--border);text-align:center;vertical-align:middle;font-size:12px;background:linear-gradient(var(--rowbg,transparent),var(--rowbg,transparent)),var(--surface)}
.pt-tbl td:nth-child(-n+3){padding-left:1px;padding-right:1px}
.pt-tbl td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.pt-tbl td.sep,.pt-tbl th.sep{border-left:1px solid var(--border)}
.pt-tbl .ln{display:block;line-height:1.35}
.pt-tbl .ln + .ln{font-size:11px;color:var(--text-dim)}
.pt-name{display:flex;align-items:center;gap:4px;min-width:0}
.pt-name > .pt-tap{flex:0 1 auto;min-width:0;font-weight:700;color:var(--text);font-size:12.5px;letter-spacing:-.01em}
.pt-tbl .c-br{width:24px}.pt-tbl .c-no{width:26px}.pt-tbl .c-mk{width:20px}.pt-tbl .c-win{width:46px}.pt-tbl .c-t3{width:46px}.pt-tbl .c-odds{width:58px}
.pt-tbl tr > :nth-child(-n+4){position:sticky;z-index:1}
.pt-tbl tr > :nth-child(1){left:0}.pt-tbl tr > :nth-child(2){left:24px}.pt-tbl tr > :nth-child(3){left:50px}.pt-tbl tr > :nth-child(4){left:70px;box-shadow:1px 0 0 var(--border)}
.pt-tbl thead tr > th:nth-child(-n+4){z-index:2}
.pt-tbl td.pt-bar{background:linear-gradient(to left,var(--c) var(--w),transparent var(--w)) right bottom/100% 3px no-repeat,linear-gradient(var(--rowbg,transparent),var(--rowbg,transparent)),var(--surface)}
.pt-top{color:var(--text) !important;font-weight:800 !important}
.pt-legend{margin:6px 2px 0;font-size:11px;line-height:1.6;color:var(--text-dim)}
.pt-sortbar{display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin:0 0 8px;font-size:12px;color:var(--text-dim)}
.pt-chip{all:unset;box-sizing:border-box;cursor:pointer;min-height:36px;padding:0 12px;display:inline-flex;align-items:center;border:1px solid var(--border);border-radius:999px;font-size:12px;color:var(--text)}
.pt-chip[aria-pressed="true"]{border-color:var(--accent);color:var(--accent);background:rgba(59,130,246,.10)}
.pt-chip:focus-visible,.pt-select:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.pt-select{min-height:36px;padding:0 8px;background:var(--surface2);border:1px solid var(--border);border-radius:8px;color:var(--text);font-size:12px}
@media (hover:none) and (pointer:coarse){.pt-chip,.pt-select{min-height:44px}}
@media (pointer: coarse){ .pt-tap{padding:10px 0} }
@media (min-width:769px){
  .pt-tbl{max-width:880px;font-size:13px}
  .pt-tbl td{padding:7px 8px;font-size:13px}
  .pt-tbl .ln + .ln{font-size:12px}
  .pt-tbl .ev{min-width:38px;padding:0 4px}
  .pt-tbl .c-br{width:40px}.pt-tbl .c-no{width:44px}.pt-tbl .c-mk{width:40px}.pt-tbl .c-win{width:84px}.pt-tbl .c-t3{width:84px}.pt-tbl .c-odds{width:110px}
  .pt-tbl tr > :nth-child(2){left:40px}.pt-tbl tr > :nth-child(3){left:84px}.pt-tbl tr > :nth-child(4){left:124px}
  .pt-name > .pt-tap{font-size:14px}
}
@media (max-width:639px){ .wp-main{padding:12px 6px !important} .wp-race-body{padding:0 4px 10px !important} }
`;

/* ── 予測テーブル ── */
function PredTable({ raceId, preds, hasPred, message, megu }: { raceId: string; preds: PredEntry[]; hasPred: boolean; message?: string; megu?: ExpectedMeguMeta | null }) {
  const [sort, setSort] = useState<{ key: SortKey; dir: SortDir } | null>(null);
  const [card, setCard] = useState<{ horseId: string; tab: CardTab } | null>(null);   // 馬名のタップで開く情報パネル（騎手・調教師のタブもここ）
  const defaultOrder = (a: PredEntry, b: PredEntry) => (a.horse_number ?? 0) - (b.horse_number ?? 0);      // 既定は馬番順
  const sorted = sortPreds(preds, sort?.key ?? null, sort?.dir ?? "asc", defaultOrder);
  const { isMember, loading: authLoading } = useAuthStatus();
  const lock = !authLoading && !isMember;      // 匿名・無料会員は印だけ。予測の数値の列はサーバが落としているので見出しに 🔒 を付ける
  const LK = (label: string) => (lock ? `🔒${label}` : label);

  // 見出しのクリック: 1 回目=その列の向き（既定は昇順、大きいほど良い列は降順）→ 2 回目=逆 → 3 回目=元の並びに戻す
  function toggle(key: SortKey) {
    setSort((cur) => {
      const first: SortDir = DESC_FIRST.has(key) ? "desc" : "asc";
      if (!cur || cur.key !== key) return { key, dir: first };
      if (cur.dir === first) return { key, dir: first === "asc" ? "desc" : "asc" };
      return null;
    });
  }
  const arrow = (key: SortKey) => (sort?.key === key ? (sort.dir === "asc" ? "▲" : "▼") : "");
  const th = (key: SortKey, label: React.ReactNode, opts: { align?: "left" | "right"; title?: string; cls?: string; sub?: string } = {}) => (
    <th
      className={`pt-sort ${opts.cls ?? ""}`}
      style={{ cursor: "pointer", userSelect: "none", color: sort?.key === key ? "var(--accent)" : undefined }}
      aria-sort={sort?.key === key ? (sort.dir === "asc" ? "ascending" : "descending") : "none"}
      title={opts.title ? `${opts.title}（クリックで並べ替え）` : "クリックで並べ替え"}
    >
      <button type="button" onClick={() => toggle(key)} className="pt-sortbtn" data-align={opts.align}>
        <span>{label}{opts.sub && <small>{opts.sub}</small>}</span>
        {sort?.key === key && <span aria-hidden style={{ fontSize: 9 }}>{arrow(key)}</span>}
      </button>
    </th>
  );

  /* 馬名: タップ/クリック（Enter・スペース）で情報パネルを開く。horse_id が無い行は文字のみ。 */
  const nameCell = (p: PredEntry) => {
    const text = p.horse_name;
    if (!text) return <span>-</span>;
    if (!p.horse_id) return <span className="pt-tap" style={{ cursor: "default", borderBottom: "none" }}>{text}</span>;
    return (
      <button
        type="button"
        className="pt-tap"
        aria-haspopup="dialog"
        title="馬の近走・血統・追い切り、騎手・調教師を見る"
        onClick={() => setCard({ horseId: p.horse_id as string, tab: "horse" })}
      >
        {text}
      </button>
    );
  };
  const panelHorses = sorted.filter((p) => p.horse_id).map((p) => ({
    horse_id: p.horse_id as string, horse_number: p.horse_number, horse_name: p.horse_name, bracket_number: p.bracket_number, sex_age: p.sex_age,
    jockey_name: p.jockey_name, jockey_weight: p.jockey_weight, trainer_name: p.trainer_name, weight: p.weight, weight_change: p.weight_change,
    expected_megu: p.expected_megu ?? null, expected_through: megu?.data_through ?? null, expected_meta: megu ?? null,
  }));

  // 総合順位の上位 3 頭は行をうっすら色付け（sticky の列にも同じ色が乗るよう CSS 変数で渡す）
  const rowBgOf = (p: PredEntry): string | undefined => {
    const rank = p.composite_rank ?? p.pred_rank ?? 0;
    return rank === 1 ? "rgba(239,68,68,0.06)" : rank === 2 ? "rgba(59,130,246,0.06)" : rank === 3 ? "rgba(34,197,94,0.05)" : undefined;
  };
  const brOf = (p: PredEntry) => bracketOf(p.horse_number ?? 0, sorted.length, p.bracket_number);
  const wakuChip = (p: PredEntry) => {
    const bn = brOf(p);
    return bn ? <span title={`${bn}枠`} style={bracketChipStyle(bn, 20)}>{bn}</span> : <span>-</span>;
  };
  // 100 倍以上は小数を落とす（複勝の「105-280」がスマホの列幅に収まるように）
  const num1 = (v: number | null | undefined) => (v == null ? "-" : Number(v) >= 100 ? Number(v).toFixed(0) : Number(v).toFixed(1));
  const placeOdds = (p: PredEntry) =>
    p.place_odds_min != null && p.place_odds_max != null ? `${num1(p.place_odds_min)}-${num1(p.place_odds_max)}` : "-";
  const dim = hasPred ? "var(--text)" : "var(--text-dim)";
  // 勝率・複勝率は、このレースの最大を基準にしたうっすらした棒（セルの下辺）を付け、最大の馬は太字にする
  const maxWin = Math.max(0, ...preds.map((p) => p.win_prob ?? 0));
  const maxT3 = Math.max(0, ...preds.map((p) => p.top3_prob ?? 0));
  const probCell = (v: number | null | undefined, max: number, color: string, top: boolean, sep: boolean) => (
    <td
      className={`num ${sep ? "sep" : ""} ${v != null && hasPred && max > 0 ? "pt-bar" : ""} ${top ? "pt-top" : ""}`}
      style={{ color: dim, fontWeight: 600, ...(v != null && max > 0 ? { ["--w" as string]: `${Math.round((v / max) * 100)}%`, ["--c" as string]: color } : {}) } as React.CSSProperties}
    >
      {lock ? "🔒" : fmtPct(v)}
    </td>
  );
  // 並び替えバー: よく使う 3 つはチップ、そのほか（表に出している値だけ）は選択肢から
  const QUICK: { key: SortKey; label: string }[] = [
    { key: "num", label: "馬番" }, { key: "win", label: "勝率" }, { key: "mark", label: "印" },
  ];
  const ALL: { key: SortKey; label: string }[] = [
    { key: "num", label: "馬番" }, { key: "mark", label: "印" }, { key: "name", label: "馬名" }, { key: "bracket", label: "枠" },
    { key: "win", label: "AI勝率" }, { key: "top3", label: "AI複勝率" },
    { key: "odds", label: "単勝オッズ予測" }, { key: "podds", label: "複勝オッズ予測" },
  ];
  const sortBar = (
    <div className="pt-sortbar" role="group" aria-label="並び替え">
      <span>並び替え</span>
      {QUICK.map((q) => (
        <button key={q.key} type="button" className="pt-chip" aria-pressed={sort?.key === q.key} onClick={() => toggle(q.key)}>
          {q.label}<span aria-hidden style={{ marginLeft: 3, fontSize: 10 }}>{arrow(q.key)}</span>
        </button>
      ))}
      <select
        className="pt-select"
        aria-label="そのほかの項目で並べ替え"
        value={sort && !QUICK.some((q) => q.key === sort.key) ? sort.key : ""}
        onChange={(e) => { const k = e.target.value as SortKey | ""; if (k) setSort({ key: k, dir: DESC_FIRST.has(k) ? "desc" : "asc" }); else setSort(null); }}
      >
        <option value="">そのほか…</option>
        {ALL.filter((a) => !QUICK.some((q) => q.key === a.key)).map((a) => <option key={a.key} value={a.key}>{a.label}</option>)}
      </select>
      {sort && <button type="button" className="pt-chip" onClick={() => setSort(null)}>元の並び</button>}
    </div>
  );

  return (
    <div style={{ marginTop: 12, maxWidth: "100%" }}>
      <style>{TABLE_CSS}</style>
      {card && (
        <EntryCardPanel
          raceId={raceId}
          horses={panelHorses}
          horseId={card.horseId}
          tab={card.tab}
          onTab={(tab) => setCard((c) => (c ? { ...c, tab } : c))}
          onHorse={(horseId) => setCard((c) => (c ? { ...c, horseId } : c))}
          onClose={() => setCard(null)}
        />
      )}
      {sortBar}

      <div className="pt-scroll">
        {/* 並べ替えは独自に持つので、共通の TableSorter は効かせない（data-no-sort） */}
        <table className="pt-tbl" data-no-sort>
          <colgroup>
            <col className="c-br" /><col className="c-no" /><col className="c-mk" /><col /><col className="c-win" /><col className="c-t3" /><col className="c-odds" />
          </colgroup>
          <thead>
            <tr>
              {th("bracket", "枠", { title: "枠番（前日に確定）" })}
              {th("num", "馬番")}
              {th("mark", "印", { title: "印（◎○▲★△消）。押すと ◎ > ○ > ▲ > ★ > △ > 無印 > 消 の順（もう一度押すと逆）" })}
              {th("name", "馬名", { align: "left", title: "馬名（押すと近走・血統・騎手・調教師の情報）" })}
              {th("win", <>{lock ? "🔒" : ""}AI<br />勝率</>, { align: "right", cls: "sep", title: lock ? "AI の勝率（会員限定）" : "AI の勝率" })}
              {th("top3", <>{lock ? "🔒" : ""}AI<br />複勝率</>, { align: "right", title: lock ? "AI の複勝率（会員限定）" : "AI の複勝率（3着以内に入る確率）" })}
              {th("odds", "オッズ", { align: "right", cls: "sep", sub: LK("予測 単/複"), title: "オッズの予測値（上段=単勝・下段=複勝の下限-上限）。単勝で並べ替え（複勝は『そのほか』から）" })}
            </tr>
          </thead>
          <tbody>
            {sorted.map((p, i) => {
              const bg = rowBgOf(p);
              return (
                <tr key={p.horse_number ?? `i${i}`} style={bg ? ({ ["--rowbg" as string]: bg } as React.CSSProperties) : undefined}>
                  <td>{wakuChip(p)}</td>
                  <td style={{ fontWeight: 800, fontVariantNumeric: "tabular-nums", color: "var(--text)" }} title={p.horse_number != null ? `${brOf(p) ?? "-"}枠 ${p.horse_number}番` : "馬番は未確定"}>
                    {p.horse_number ?? "-"}
                  </td>
                  <td style={{ fontWeight: 800, color: "#fbbf24" }} title={markOf(p) ? "めぐAI印" : undefined}>
                    {markOf(p) || ""}
                  </td>
                  <td style={{ textAlign: "left" }}>
                    <span className="pt-name">{nameCell(p)}</span>
                  </td>
                  {probCell(p.win_prob, maxWin, "rgba(96,165,250,.75)", hasPred && p.win_prob != null && p.win_prob === maxWin && maxWin > 0, true)}
                  {probCell(p.top3_prob, maxT3, "rgba(52,211,153,.7)", hasPred && p.top3_prob != null && p.top3_prob === maxT3 && maxT3 > 0, false)}
                  <td className="num sep" style={{ color: "var(--text)" }}>
                    <span className="ln">{num1(p.win_odds)}</span>
                    <span className="ln">{placeOdds(p)}</span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {hasPred && <MarkNote />}
      <p className="pt-legend">
        オッズは上段＝単勝、下段＝複勝（複勝オッズは下限-上限）。騎手・調教師・性齢・斤量・馬体重は馬名を押した情報パネルで、連対率は「詳細 →」のレース詳細（広い画面）で見られます。
      </p>

      {!hasPred && (
        <p style={{ fontSize: 11, color: "var(--text-dim)", padding: "8px 10px" }}>
          ※ {message ?? "予測はまだありません。数値は全て「—」です。"}
        </p>
      )}
    </div>
  );
}

/* ── 選んだ 1 レースの表示 ── */
function RacePanel({
  race, pred, loading, onReload, value, date,
}: {
  race: RaceItem;
  date?: string;
  pred: PredData | null;
  loading: boolean;
  onReload: (id: string) => void;
  value?: RaceValue;
}) {
  // 匿名・無料会員の応答には確率が無く印だけが来るので、印でも「予測あり」とみなす
  const hasPred = pred?.has_prediction === true && (pred.predictions?.some((p) => p.win_prob != null || !!p.mark_type) ?? false);
  const preds = pred?.predictions ?? [];
  const heads = race.entries_count ?? race.field_size ?? pred?.total_horses;
  const conds = [
    race.venue,
    race.distance ? `${race.distance}m` : "",
    race.surface,
    race.track_condition ? `馬場 ${race.track_condition}` : "",
    race.start_time ? `${race.start_time} 発走` : "",
    heads ? `${heads}頭` : "",
  ].filter(Boolean);
  const failed = !loading && pred?.status === "error" && preds.length === 0;
  useAutoRetry(failed, () => onReload(race.race_id));
  const { isMember, loading: authLoading } = useAuthStatus();
  const locked = !authLoading && !isMember;
  const [tab, setTab] = useState<"predict" | "bet" | "result">("predict");
  const { result, started } = useRaceResult(race.race_id, race.date || date, race.start_time, !authLoading && !locked);
  const userPicked = useRef(false);
  useEffect(() => { userPicked.current = false; setTab("predict"); }, [race.race_id]);
  // 結果が出たレースを開いたときは、結果を先に見せる（ユーザーが自分でタブを選んだ後は動かさない）
  useEffect(() => { if (result && !userPicked.current) setTab("result"); }, [result?.status, result?.official, race.race_id]); // eslint-disable-line react-hooks/exhaustive-deps
  const predRank = useMemo(() => {
    const m: Record<number, number> = {};
    for (const p of preds) {
      const r = p.composite_rank ?? p.pred_rank;
      if (p.horse_number != null && r) m[p.horse_number] = r;
    }
    return m;
  }, [preds]);
  const tabs: [typeof tab, string][] = [["predict", "AI予測"], ["bet", "サンプル買い目"], ...(result ? [["result", "レース結果"] as [typeof tab, string]] : started && !locked ? [["result", "レース結果（待ち）"] as [typeof tab, string]] : [])];

  return (
    <section
      id={`race-${race.race_id}`}
      aria-label={`${race.venue ?? ""}${race.round ?? ""}R ${race.race_name ?? ""}`}
      aria-busy={loading}
      title={valueTitle(value)}
      style={{
        background: value?.level === "hot" ? "linear-gradient(90deg, rgba(249,115,22,0.10), var(--surface) 40%)" : "var(--surface)",
        border: "1px solid var(--border)", borderLeft: `4px solid ${value ? LEVEL_STYLE[value.level].border : "var(--border)"}`,
        borderRadius: 10, overflow: "hidden", marginBottom: 12,
        boxShadow: value?.level === "hot" ? "0 0 12px rgba(249,115,22,0.25)" : undefined,
      }}
    >
      <div style={{ padding: "14px 18px", display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
        <span style={{ background: "var(--accent)", color: "#fff", fontWeight: 800, fontSize: 13, padding: "3px 10px", borderRadius: 4, minWidth: 40, textAlign: "center" }}>
          {race.round}R
        </span>
        <h2 style={{ margin: 0, fontWeight: 700, fontSize: 16, color: "#fff" }}>
          {race.race_name ?? race.race_id}
          {gradeChip(race.grade)}
        </h2>
        <span style={{ fontSize: 12, color: "var(--text-dim)" }}>{conds.join(" ・ ")}</span>

        {(pred?.entries_status || race.entries_phase) && (
          <EntriesStatusBadge
            status={pred?.entries_status ?? { phase: race.entries_phase, label: race.entries_label ?? undefined, updated_at: race.entries_updated_at }}
          />
        )}

        {value && value.level !== "none" && (
          <span
            style={{
              fontSize: 11, fontWeight: 700, padding: "2px 9px", borderRadius: 6, whiteSpace: "nowrap",
              border: `1px solid ${LEVEL_STYLE[value.level].border}`, background: LEVEL_STYLE[value.level].bg, color: LEVEL_STYLE[value.level].color,
              boxShadow: LEVEL_STYLE[value.level].glow,
            }}
          >
            {value.level === "hot" ? "★ " : ""}{value.tier?.label ? `参考度 ${value.tier.label}` : value.level !== "pass" ? `参考度 ${LEVEL_STYLE[value.level].label.replace(/^★ /, "")}` : "見送り"}
          </span>
        )}


        <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 8 }}>
          {pred && !failed && (
            <span style={{
              fontSize: 10, fontWeight: 600, padding: "2px 7px", borderRadius: 3,
              background: hasPred ? "rgba(34,197,94,0.12)" : "rgba(107,125,149,0.12)",
              color: hasPred ? "var(--ok)" : "var(--text-dim)",
            }}>
              {hasPred ? "✓ 予測済" : pred.provisional_entries ? "確定後に予測" : "未予測"}
            </span>
          )}
          <IntentLink
            href={`/race/${race.race_id}`}
            className="hit44"
            style={{ fontSize: 11, color: "var(--accent)", textDecoration: "none", border: "1px solid rgba(59,130,246,0.3)", padding: "3px 8px", borderRadius: 4 }}
          >
            詳細 →
          </IntentLink>
        </div>
      </div>

      <div className="wp-race-body" style={{ borderTop: "1px solid var(--border)", padding: "0 18px 14px" }}>
        {(loading || (!pred && !failed)) && <LoadingNotice what="予測データ" onRetry={() => onReload(race.race_id)} />}
        {failed && (
          <div role="alert" style={{ margin: "14px 0", background: "rgba(239,68,68,0.1)", border: "1px solid rgba(239,68,68,0.3)", borderRadius: 8, padding: "12px 16px", color: "var(--err)", fontSize: 13, display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
            <span>⚠️ このレースの予測をまだ取得できていません。</span>
            <span style={{ marginLeft: "auto", color: "var(--text-dim)" }}>自動で取り直しています…</span>
          </div>
        )}
        {pred && !failed && (
          <div role="tablist" aria-label="表示の切り替え" style={{ display: "flex", gap: 6, margin: "10px 0 4px", flexWrap: "wrap" }}>
            {tabs.map(([k, label]) => (
              <button
                key={k}
                type="button"
                role="tab"
                aria-selected={tab === k}
                onClick={() => { userPicked.current = true; setTab(k); }}
                style={{
                  minHeight: 44, padding: "0 16px", borderRadius: 8, cursor: "pointer", fontSize: 13, fontWeight: 700,
                  border: `1px solid ${tab === k ? "var(--accent)" : "var(--border)"}`,
                  background: tab === k ? "rgba(59,130,246,0.15)" : "transparent",
                  color: tab === k ? "var(--accent)" : "var(--text-dim)",
                }}
              >
                {k === "bet" && locked ? "🔒 " : ""}{label}
              </button>
            ))}
          </div>
        )}
        {pred && !failed && tab === "bet" && (
          <WeeklyBetTab raceId={race.race_id} hasPred={hasPred} lock={locked} authLoading={authLoading} resultKey={result ? `${result.status}-${result.official ? 1 : 0}` : ""} />
        )}
        {pred && !failed && tab === "result" && (result
          ? <RaceResult result={result} predRank={predRank} />
          : <div role="status" style={{ padding: "16px 0", color: "var(--text-dim)", fontSize: 13 }}>レース結果を待っています。発走の約15分後に速報を取得し、取得でき次第この画面に表示します（1分ごとに自動で確認します）。</div>)}
        {pred && !failed && tab === "predict" && <PredictionSourceBadge source={pred.prediction_source} note={pred.prediction_note} />}
        {pred && !failed && tab === "predict" && locked && preds.length > 0 && <MemberLockNote />}
        {pred && !failed && <PublicModeNote access={pred.access} />}
        {pred && !failed && tab === "predict" && <ProbCalNote pc={pred.prob_calibration} preds={preds} />}
        {pred && !failed && tab === "predict" && (
          <PredTable raceId={race.race_id} preds={preds} hasPred={hasPred} message={pred.message} megu={pred.expected_megu_meta} />
        )}
        {pred && !failed && tab === "predict" && !locked && preds.length > 0 && (
          <MeguTable preds={preds} bench={pred.class_benchmark} meta={pred.expected_megu_meta} today={{ surface: race.surface, distance: Number(race.distance) || undefined }} />
        )}
        {pred && !failed && tab === "predict" && !locked && preds.length > 0 && (
          <PredExtras key={race.race_id} raceId={race.race_id} entries={preds} pace={pred.pace_forecast} paceUnified={pred.pace_unified} lapForecast={pred.lap_forecast} paceType={pred.pace_type} scenarioForecast={pred.scenario_forecast} gate={pred.gate_bias} steeplechase={(race.surface ?? "").startsWith("障")} direction={turnOf(race.direction, race.venue, race.surface, race.distance)} />
        )}
      </div>
    </section>
  );
}

/* ── メインページ ── */
export default function WeeklyPredictionsPage() {
  const [dates, setDates] = useState<string[]>([]);
  const [selectedDate, setSelectedDate] = useState("");
  const [races, setRaces] = useState<RaceItem[]>([]);
  const [preds, setPreds] = useState<Record<string, PredData | null>>({});
  const [loadingPred, setLoadingPred] = useState<Record<string, boolean>>({});
  const [loadingDates, setLoadingDates] = useState(true);
  const [loadingRaces, setLoadingRaces] = useState(false);
  const [racesError, setRacesError] = useState("");
  const [datesError, setDatesError] = useState("");
  const [datesReload, setDatesReload] = useState(0);
  const { values: raceValues, loading: valuesLoading, failed: valuesFailed } = useRaceValuesState(selectedDate);
  const [selectedVenue, setSelectedVenue] = useState("");
  const [selectedRaceId, setSelectedRaceId] = useState("");
  const today = useMemo(() => todayJst(), []);

  /** URL の ?date= ?week= ?venue= ?race= を書き換える（再読込・共有しても同じ表示に戻る） */
  function syncUrl(patch: Record<string, string | null>) {
    const u = new URL(window.location.href);
    for (const [k, v] of Object.entries(patch)) {
      if (v == null || v === "") u.searchParams.delete(k); else u.searchParams.set(k, v);
    }
    window.history.replaceState(null, "", u.toString());
  }

  /* 開催場ごとに 12R を切り替える（URL の ?venue= で共有・再読込しても維持） */
  const venues = useMemo(() => venuesOf(races), [races]);   // JRA の競馬場コード順
  const activeVenue = venues.includes(selectedVenue) ? selectedVenue : (venues[0] ?? "");
  const visibleRaces = useMemo(
    () => races
      .filter((r) => !activeVenue || r.venue === activeVenue)
      .sort((a, b) => Number(a.round ?? 0) - Number(b.round ?? 0)),
    [races, activeVenue],
  );

  /* 表示する 1 レース: ?race= で指定されたレース。無い・この開催場に無いときは、その開催場の最初のレース */
  const activeRace = useMemo(
    () => visibleRaces.find((r) => r.race_id === selectedRaceId) ?? visibleRaces[0],
    [visibleRaces, selectedRaceId],
  );
  const activeRaceId = activeRace?.race_id ?? "";

  useEffect(() => {
    const q = new URLSearchParams(window.location.search);
    const v = q.get("venue");
    if (v) setSelectedVenue(v);
    const dq = q.get("date");
    if (dq && /^\d{8}$/.test(dq) && !USE_MOCK) {         // 開催日一覧・認証の確認を待たずに、開く日のレース一覧と参考度を先に取り始める
      prefetchJson(`/api/race-list/${dq}`, { ttlMs: 60_000 });
      prefetchRaceValues(dq);
    }
    const r = q.get("race");
    if (r) {
      setSelectedRaceId(r);
      if (!USE_MOCK) prefetchJson(`/api/race/${r}/predictions`, { ttlMs: PRED_TTL_MS, persist: true });      // 日付・一覧の取得を待たずに、開くレースの予測を先に取り始める
    }
  }, []);

  function chooseVenue(v: string) {
    // 今と同じレース番号を選ぶ（無ければ一番近い番号）。1R に飛ばさない
    const cur = races.find((r) => r.race_id === activeRaceId);
    const target = sameRoundIn(races, v, cur ? Number(cur.round ?? Number(cur.race_id.slice(-2))) : null);
    setSelectedVenue(v);
    setSelectedRaceId(target?.race_id ?? "");
    syncUrl({ venue: v, race: target?.race_id ?? null });
  }

  function chooseDate(d: string) {
    setSelectedDate(d);
    setSelectedRaceId("");
    syncUrl({ date: d, week: meetingKeyOf(d), race: null });
  }

  function chooseRace(raceId: string) {
    const r = races.find((x) => x.race_id === raceId);
    if (r?.venue && r.venue !== activeVenue) setSelectedVenue(r.venue);
    setSelectedRaceId(raceId);
    syncUrl({ race: raceId, ...(r?.venue ? { venue: r.venue } : {}) });
  }

  /* ?race= で指定されたレースの開催場へ切り替える（レース一覧を読み込んだあと） */
  useEffect(() => {
    if (!selectedRaceId) return;
    const r = races.find((x) => x.race_id === selectedRaceId);
    if (r?.venue) setSelectedVenue(r.venue);
  }, [races, selectedRaceId]);

  /* 開催日一覧 */
  useEffect(() => {
    // 既定の開催日: ?date= → NEXT_PUBLIC_DEFAULT_RACE_DATE（dev は実データを取り込んだ日を既定にできる）→ 次回の開催の最初の開催日
    const initial = (list: string[]) => {
      const fromUrl = new URLSearchParams(window.location.search).get("date") ?? undefined;
      const preferred = fromUrl && list.includes(fromUrl) ? fromUrl : process.env.NEXT_PUBLIC_DEFAULT_RACE_DATE;
      if (preferred && list.includes(preferred)) return preferred;
      const next = pickTwoWeeks(list, today).find((w) => w.role === "next");
      return next ? firstDateOf(next.week, today) : (list[0] ?? "");
    };
    if (USE_MOCK) {
      const mockDates = getMockRaceDates();
      setDates(mockDates);
      setSelectedDate(initial(mockDates));
      setLoadingDates(false);
      return;
    }
    setDatesError("");
    (async () => {
      try {
        const days = Number(process.env.NEXT_PUBLIC_PICKER_PAST_DAYS) || 30;
        // 先週分〜来週分を出すため、今日と先の開催日（14 日先まで）も含める
        const res = await fetch(`/api/scrape-dates?picker_past_days=${days}&picker_future_days=14`, { cache: "no-store" });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const d = await res.json();
        const list: string[] = d.dates ?? d ?? [];
        setDates(list);
        if (list.length) setSelectedDate(initial(list));
        if (list.length) {                                  // 近い開催日の参考度も裏で温めておく（日付を切り替えたときすぐ出る）
          const t0 = new Date();
          const day = (x: string) => Date.UTC(+x.slice(0, 4), +x.slice(4, 6) - 1, +x.slice(6, 8));
          const near = list.filter((x) => Math.abs(day(x) - Date.UTC(t0.getFullYear(), t0.getMonth(), t0.getDate())) <= 7 * 86400000).slice(0, 8);
          setTimeout(() => near.forEach(prefetchRaceValues), 800);
        }
      } catch (e: unknown) {
        setDatesError(`開催日の一覧を取得できませんでした（${e instanceof Error ? e.message : String(e)}）`);
      } finally {
        setLoadingDates(false);
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [datesReload]);

  /* レース一覧（日付変更時） */
  const racesSeq = useRef(0);
  const loadRaces = useCallback(async (date: string) => {
    if (!date) return;
    const seq = ++racesSeq.current;      // 日付を素早く切り替えたとき、遅れて届いた古い一覧で上書きしない
    setLoadingRaces(true);
    setRacesError("");
    setRaces([]);
    setPreds({});
    if (USE_MOCK) {
      await new Promise((r) => setTimeout(r, 300));
      setRaces(MOCK_WEEKLY_RACES.map((r) => ({ ...r, date })));
      setLoadingRaces(false);
      return;
    }
    try {
      const d = (await fetchRaceList(date)) as { races?: RaceItem[] } | RaceItem[];
      if (seq !== racesSeq.current) return;
      const list: RaceItem[] = (Array.isArray(d) ? d : d.races) ?? [];
      setRaces(list);
    } catch (e: unknown) {
      if (seq === racesSeq.current) setRacesError(e instanceof Error ? e.message : String(e));
    } finally {
      if (seq === racesSeq.current) setLoadingRaces(false);
    }
  }, []);

  useEffect(() => {
    if (selectedDate) loadRaces(selectedDate);
  }, [selectedDate, loadRaces]);

  /* 一覧の取得に失敗したら、利用者の操作を待たず裏で取り直す */
  useAutoRetry(!!(datesError || racesError), () => { if (datesError) { setLoadingDates(true); setDatesReload((n) => n + 1); } else loadRaces(selectedDate); }, 8000);

  /* 個別レース予測取得 */
  const loadPred = useCallback(async (raceId: string, force = false) => {
    setLoadingPred((prev) => ({ ...prev, [raceId]: true }));
    if (force) setPreds((prev) => ({ ...prev, [raceId]: null }));
    if (USE_MOCK) {
      await new Promise((r) => setTimeout(r, 200));
      setPreds((prev) => ({ ...prev, [raceId]: getMockPredictions(raceId) }));
      setLoadingPred((prev) => ({ ...prev, [raceId]: false }));
      return;
    }
    // 前回このタブで見た内容があれば先に出して（取得中を出さない）、最新を取れたら差し替える
    const url = `/api/race/${raceId}/predictions`;
    let hadStale = false;
    if (!force) {
      const prev = readStored<PredData>(url);
      if (prev) {
        hadStale = true;
        setPreds((p) => (p[raceId] ? p : { ...p, [raceId]: prev }));
        setLoadingPred((p) => ({ ...p, [raceId]: false }));
      }
    }
    try {
      const d = await getJson<PredData>(url, { ttlMs: PRED_TTL_MS, force, persist: true }).catch((): PredData => ({ status: "error", predictions: [] }));
      if (!(hadStale && d.status === "error")) setPreds((prev) => ({ ...prev, [raceId]: d }));      // 最新が取れなくても、先に出した前回の内容は消さない
    } catch {
      if (!hadStale) setPreds((prev) => ({ ...prev, [raceId]: { status: "error", predictions: [] } }));
    } finally {
      setLoadingPred((prev) => ({ ...prev, [raceId]: false }));
    }
  }, []);

  /* 予測の先読みは全体で同時 1 本（lib/predPrefetch.ts）。選んだレースの取得が先 */
  const prefetcher = useRef<PredPrefetcher | null>(null);
  const pf = useCallback((): PredPrefetcher => {
    if (!prefetcher.current) {
      prefetcher.current = new PredPrefetcher(
        (id) => getJson(`/api/race/${id}/predictions`, { ttlMs: PRED_TTL_MS, persist: true }),
        (id) => hasFreshJson(`/api/race/${id}/predictions`, PRED_TTL_MS),
      );
    }
    return prefetcher.current;
  }, []);

  /* マウスが乗った／触れ始めたレースの予測を先読みする。連続で動かしたときは、止まった（150ms）レースだけ */
  const warmTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => () => {                  // 画面を離れたら、まだ出していない先読みは出さない
    if (warmTimer.current) clearTimeout(warmTimer.current);
    prefetcher.current?.stop();
  }, []);
  const warmPred = useCallback((id: string) => {
    if (USE_MOCK) return;
    if (warmTimer.current) clearTimeout(warmTimer.current);
    warmTimer.current = setTimeout(() => pf().want([id], true), 150);
  }, [pf]);

  /* 選んだレースの予測だけを取得する（素早く切り替えたときは最後に選んだものだけ。取得済みは即表示）。
     表示は activeRaceId の状態だけを見るので、遅れて届いた別レースの応答が画面を上書きすることはない */
  useEffect(() => {
    if (!activeRaceId || preds[activeRaceId] || loadingPred[activeRaceId]) return;
    const id = activeRaceId;
    // 取得済み（先読み済み）なら待たずに表示。そうでなくても、素早い切替の連打をまとめる程度（60ms）だけ待つ
    const wait = hasFreshJson(`/api/race/${id}/predictions`, PRED_TTL_MS) ? 0 : 60;
    const timer = setTimeout(() => { void loadPred(id); }, wait);
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeRaceId, preds]);

  /* 選んだレースの前後（同じ開催場の次の 2 つと前の 1 つ）の予測だけを裏で先に取っておく（←/→・次のレースへの切替が待たずに出る）。
     1 本ずつ・選んだレースの取得の後に。以前はその日の全レース（最大 40 件・2 本ずつ）を取っていて、1 回開くと API が約 37 件出ていた。
     それ以外のレースは、マウスが乗った・触れたときの先読み（warmPred）で温める */
  useEffect(() => {
    if (USE_MOCK) return;
    const q = pf();
    const i = activeRaceId ? visibleRaces.findIndex((r) => r.race_id === activeRaceId) : -1;
    if (i < 0) { q.release(); return; }    // 選んだレースが無い間は、マウスの先読みを止めたままにしない
    const ids = [visibleRaces[i + 1], visibleRaces[i + 2], visibleRaces[i - 1]].map((r) => r?.race_id).filter((id): id is string => !!id);
    q.focus(activeRaceId);                 // 選んだレースが変わったら、まだ出していない先読みを捨て、選んだレースが済むまで次を出さない
    // 選んだレースの取得（loadPred と同じ 1 本に相乗り）が済んでから、前後を 1 本ずつ
    const t1 = setTimeout(() => q.select(getJson(`/api/race/${activeRaceId}/predictions`, { ttlMs: PRED_TTL_MS, persist: true })), 60);
    const t2 = setTimeout(() => q.want(ids), 800);
    return () => { clearTimeout(t1); clearTimeout(t2); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeRaceId, visibleRaces]);

  /* 既定のレース（先頭）が選ばれたときも URL に race を入れる（再読み込み・共有で同じレースが開く） */
  useEffect(() => {
    if (!activeRace) return;
    const cur = new URLSearchParams(window.location.search).get("race");
    if (cur !== activeRace.race_id) syncUrl({ race: activeRace.race_id, ...(activeRace.venue ? { venue: activeRace.venue } : {}) });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeRace?.race_id]);

  /* ←/→ キーで前後のレースへ（入力欄では無効） */
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.altKey || e.ctrlKey || e.metaKey || e.shiftKey) return;
      if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
      const el = e.target as HTMLElement | null;
      if (el && (/^(INPUT|SELECT|TEXTAREA)$/.test(el.tagName) || el.isContentEditable)) return;
      const i = visibleRaces.findIndex((r) => r.race_id === activeRaceId);
      const to = visibleRaces[i + (e.key === "ArrowRight" ? 1 : -1)];
      if (i >= 0 && to) { e.preventDefault(); chooseRace(to.race_id); }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [visibleRaces, activeRaceId, races]);

  return (
    <div style={{ minHeight: "100dvh", background: "var(--bg)", color: "var(--text)" }}>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>

      {/* ページヘッダー */}
      <div className="wp-head" style={{ background: "var(--surface)", borderBottom: "1px solid var(--border)", padding: "18px 24px" }}>
        <div style={{ maxWidth: 1200, margin: "0 auto", display: "flex", alignItems: "center", gap: 12 }}>
          <h1 style={{ fontSize: 18, fontWeight: 700, color: "#f0f6fc", margin: 0 }}>🤖 今週のAI予測</h1>
          <p className="hide-narrow" style={{ fontSize: 12, color: "var(--text-dim)", margin: 0 }}>
            前回と次回の開催（最大2週分）の AI の勝率と印を一覧表示
          </p>
        </div>
      </div>

      {/* 開催日の選択（週単位） */}
      <div className="wp-pick" style={{ background: "var(--surface)", borderBottom: "1px solid var(--border)", padding: "14px 24px" }}>
        <div style={{ maxWidth: 1200, margin: "0 auto", display: "flex", flexDirection: "column", gap: 12 }}>
          <WeekPicker dates={dates} today={today} selectedDate={selectedDate} loading={loadingDates} onSelectDate={chooseDate} />
        </div>
      </div>

      {/* メインコンテンツ */}
      <div className="wp-main" style={{ maxWidth: 1200, margin: "0 auto", padding: "20px 24px" }}>
        {/* 印の凡例は外した（2026-11-25 ユーザーの指摘）。印の並びは印の列の見出しの title に 1 行 */}

        {loadingRaces && <LoadingNotice what="レース一覧" size={36} padding="60px 0" onRetry={() => loadRaces(selectedDate)} />}

        {(datesError || racesError) && (
          <div role="alert" style={{ background: "rgba(239,68,68,0.1)", border: "1px solid rgba(239,68,68,0.3)", borderRadius: 8, padding: "14px 18px", color: "var(--err)", fontSize: 13, marginBottom: 16, display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
            <span>⚠️ {datesError || `レース一覧を取得できませんでした（${racesError}）`}</span>
            <span style={{ marginLeft: "auto", color: "var(--text-dim)" }}>自動で取り直しています…</span>
          </div>
        )}

        {!loadingRaces && races.length === 0 && !racesError && !datesError && selectedDate && (
          <div style={{ textAlign: "center", padding: "60px 0", color: "var(--text-dim)" }}>
            <div style={{ fontSize: 40, marginBottom: 12 }}>📭</div>
            <p>この日程のレース情報がありません</p>
          </div>
        )}

        {!loadingRaces && races.length > 0 && (
          <>
            <EntriesDaySummary races={races} />
            <RaceStrip
              races={races}
              venue={activeVenue}
              onVenue={chooseVenue}
              values={raceValues}
              valuesState={{ loading: valuesLoading, failed: valuesFailed }}
              selectedRaceId={activeRaceId}
              onSelectRace={chooseRace}
              onWarm={warmPred}
            />
            <div style={{ marginBottom: 12, fontSize: 13, color: "var(--text-dim)" }}>
              {selectedDate}{activeVenue ? ` ${activeVenue}` : ""} — {visibleRaces.length}レース中 {activeRace?.round}R を表示
              <span className="hide-narrow" style={{ marginLeft: 8 }}>（← → キーで前後のレース）</span>
            </div>
            {activeRace && (
              <RacePanel
                key={activeRace.race_id}
                race={activeRace}
                pred={preds[activeRace.race_id] ?? null}
                loading={loadingPred[activeRace.race_id] ?? false}
                onReload={(id) => { void loadPred(id, true); }}
                value={raceValues.get(activeRace.race_id)}
                date={selectedDate}
              />
            )}
          </>
        )}
      </div>
    </div>
  );
}
