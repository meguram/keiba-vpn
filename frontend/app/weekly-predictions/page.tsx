"use client";

import { fetchRaceList, getJson, prefetchJson } from "@/lib/httpCache";
import { useEffect, useState, useCallback, useRef, useMemo } from "react";
import Link from "next/link";
import { USE_MOCK, MOCK_WEEKLY_RACES, getMockRaceDates, getMockPredictions } from "@/lib/mock";
import { featureEnabled } from "@/lib/features";
import { LEVEL_STYLE, RaceValue, useRaceValues, valueTitle } from "@/lib/raceValue";
import { PredictionSourceBadge } from "@/components/PredictionSourceBadge";
import { WeekPicker } from "@/components/WeekPicker";
import { RaceStrip } from "@/components/RaceStrip";
import { CardTab, EntryCardPanel } from "@/components/EntryCardPanel";
import { ExpectedMeguCell } from "@/components/ExpectedMegu";
import type { ExpectedMegu, ExpectedMeguMeta } from "@/lib/expectedMegu";
import { firstDateOf, meetingKeyOf, pickTwoWeeks, todayJst, venuesOf } from "@/lib/weekDates";

const SHOW_HORSE_LINK = featureEnabled("horse-detail");
/* ── 型 ── */
type RaceItem = {
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
};

type PredEntry = {
  horse_number?: number;
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
  mark_type?: string;
  pred_rank?: number;
  composite_rank?: number;
  win_prob?: number | null;
  top2_prob?: number | null;
  top3_prob?: number | null;
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
};

type PredData = {
  status?: string;
  has_prediction?: boolean;
  model_description?: string;
  prediction_source?: string;
  prediction_note?: string;
  message?: string;
  total_horses?: number;
  predictions?: PredEntry[];
  expected_megu_meta?: ExpectedMeguMeta | null;
  error?: string;
};

/* ── ユーティリティ ── */
const MARK_META: Record<string, { sym: string; color: string }> = {
  honmei:   { sym: "◎", color: "#ef4444" },
  pair:     { sym: "○", color: "#3b82f6" },
  anchor:   { sym: "△", color: "#22c55e" },
  show_val: { sym: "▲", color: "#f59e0b" },
  star:     { sym: "★", color: "#a78bfa" },
  none:     { sym: "-", color: "var(--text-dim)" },
};

function markSym(p: PredEntry): { sym: string; color: string } {
  if (p.mark_type && MARK_META[p.mark_type]) return MARK_META[p.mark_type];
  return MARK_META.none;
}

function fmtPct(v: number | null | undefined): string {
  if (v == null) return "-";
  return `${(v * 100).toFixed(1)}%`;
}

/** 較正後の期待値（cal=true）の色の閾値は、買い推奨度（強推奨 1.10 / 推奨 1.00 / 様子見 0.90）に合わせる */
function fmtEv(v: number | null | undefined, cal = false): { text: string; color: string; bg: string } {
  if (v == null) return { text: "-", color: "var(--text-dim)", bg: "transparent" };
  const n = Number(v);
  if (cal) {
    if (n >= 1.1) return { text: n.toFixed(2), color: "#22c55e", bg: "rgba(34,197,94,0.10)" };
    if (n >= 1.0) return { text: n.toFixed(2), color: "#60a5fa", bg: "rgba(96,165,250,0.08)" };
    if (n >= 0.9) return { text: n.toFixed(2), color: "var(--text-dim)", bg: "transparent" };
    return { text: n.toFixed(2), color: "#f87171", bg: "rgba(239,68,68,0.07)" };
  }
  if (n >= 1.5) return { text: n.toFixed(2), color: "#22c55e", bg: "rgba(34,197,94,0.10)" };
  if (n >= 1.2) return { text: n.toFixed(2), color: "#4ade80", bg: "rgba(74,222,128,0.08)" };
  if (n >= 1.0) return { text: n.toFixed(2), color: "#60a5fa", bg: "rgba(96,165,250,0.08)" };
  if (n >= 0.8) return { text: n.toFixed(2), color: "var(--text-dim)", bg: "transparent" };
  return { text: n.toFixed(2), color: "#f87171", bg: "rgba(239,68,68,0.07)" };
}

function fmtOdds(v: number | null | undefined): string {
  if (v == null) return "-";
  return `${Number(v).toFixed(1)}倍`;
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

/* ── サマリーバッジ（勝期待値 >= 1.0 の頭数など） ── */
function RaceSummaryBadges({ preds }: { preds: PredEntry[] }) {
  const hasP = preds.filter((p) => p.win_prob != null);
  if (!hasP.length) return null;
  const evOk = hasP.filter((p) => (p.ev_win_cal ?? p.ev_win ?? 0) >= 1.0).length;
  const evGreat = hasP.filter((p) => (p.ev_win_cal != null ? p.ev_win_cal >= 1.1 : (p.ev_win ?? 0) >= 1.3)).length;
  const honmei = hasP.find((p) => p.mark_type === "honmei");
  return (
    <div style={{ display: "flex", gap: 6, flexWrap: "wrap", alignItems: "center" }}>
      {honmei && (
        <span style={{ fontSize: 11, fontWeight: 700, padding: "2px 8px", borderRadius: 4, background: "rgba(239,68,68,0.12)", color: "#ef4444" }}>
          ◎ {honmei.horse_id && SHOW_HORSE_LINK
            ? <a href={`/horse/${honmei.horse_id}`} target="_blank" rel="noreferrer" style={{ color: "inherit", textDecoration: "none" }}>{honmei.horse_name ?? `馬番${honmei.horse_number}`}</a>
            : (honmei.horse_name ?? `馬番${honmei.horse_number}`)
          }
        </span>
      )}
      {evGreat > 0 && (
        <span style={{ fontSize: 11, fontWeight: 600, padding: "2px 8px", borderRadius: 4, background: "rgba(34,197,94,0.10)", color: "#22c55e" }}>
          EV≥1.3: {evGreat}頭
        </span>
      )}
      {evOk > 0 && (
        <span style={{ fontSize: 11, fontWeight: 600, padding: "2px 8px", borderRadius: 4, background: "rgba(59,130,246,0.10)", color: "#60a5fa" }}>
          EV≥1.0: {evOk}頭
        </span>
      )}
    </div>
  );
}

/* 枠の色（JRA: 1白 2黒 3赤 4青 5黄 6緑 7橙 8桃） */
const BRACKET: Record<number, { bg: string; color: string }> = {
  1: { bg: "#f8fafc", color: "#111" }, 2: { bg: "#111827", color: "#fff" }, 3: { bg: "#ef4444", color: "#fff" }, 4: { bg: "#3b82f6", color: "#fff" },
  5: { bg: "#facc15", color: "#111" }, 6: { bg: "#22c55e", color: "#111" }, 7: { bg: "#fb923c", color: "#111" }, 8: { bg: "#f9a8d4", color: "#111" },
};

/** 馬体重「514(+2)」。未発表（前日〜当日の発表前）は "-" */
function fmtWeight(w: number | null | undefined, d: number | null | undefined): string {
  if (!w) return "-";
  return d == null ? `${w}` : `${w}(${d > 0 ? "+" : d < 0 ? "" : "±"}${d})`;
}

/* ── 列での並べ替え ── */
type SortKey = "num" | "name" | "bracket" | "sexage" | "jwt" | "jockey" | "trainer" | "weight" | "emegu" | "win" | "top2" | "top3" | "odds" | "podds" | "evw" | "evp" | "tier";
type SortDir = "asc" | "desc";

function sexAgeKey(v?: string | null): number | null {
  const m = (v ?? "").match(/^(牡|牝|セ|騸)?(\d+)/);
  return m ? Number(m[2]) * 10 + ({ 牡: 1, 牝: 2, セ: 3, 騸: 3 } as Record<string, number>)[m[1] ?? ""] : null;
}

/** 列の値。数値は数、文字は文字。値が無い（"-" 表示）ときは null（並べ替えでは常に末尾） */
const SORT_VALUE: Record<SortKey, (p: PredEntry) => number | string | null> = {
  num: (p) => p.horse_number ?? null,
  name: (p) => p.horse_name ?? null,
  bracket: (p) => p.bracket_number ?? null,
  sexage: (p) => sexAgeKey(p.sex_age),
  jwt: (p) => p.jockey_weight ?? null,
  jockey: (p) => p.jockey_name ?? null,
  trainer: (p) => p.trainer_name ?? null,
  weight: (p) => p.weight ?? null,
  emegu: (p) => p.expected_megu?.value ?? null,
  win: (p) => p.win_prob ?? null,
  top2: (p) => p.top2_prob ?? null,
  top3: (p) => p.top3_prob ?? null,
  odds: (p) => p.win_odds ?? null,
  podds: (p) => p.place_odds_min ?? null,
  evw: (p) => p.ev_win_cal ?? p.ev_win ?? null,
  evp: (p) => p.ev_place_cal ?? p.ev_place ?? p.expected_value ?? null,
  tier: (p) => p.buy_tier ?? null,
};
// 『大きいほど良い』列は、最初のクリックを降順にする（勝率・連対率・複勝率・期待値・馬体重）
const DESC_FIRST = new Set<SortKey>(["win", "top2", "top3", "evw", "evp", "weight", "emegu"]);

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

/* 出馬表の見せ方を幅で切り替える（横スクロールはしない）:
   広い(1100px〜)=列を統合した表 / 中間(769〜1099px)=さらに畳んだ表 / 狭い(〜768px)=1頭1枚のカード。
   3 つとも DOM に置き、CSS の display で切り替える（JS の幅検知なし＝SSR とハイドレーションが一致する）。 */
const TABLE_CSS = `
.pt-tap{all:unset;cursor:pointer;display:inline-block;max-width:100%;padding:6px 2px;border-bottom:1px dotted rgba(148,163,184,.55);font:inherit;font-weight:inherit;line-height:1.3;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;vertical-align:bottom;box-sizing:border-box}
.pt-tap:hover{border-bottom-color:var(--accent)}
.pt-tap:focus-visible{outline:2px solid var(--accent);outline-offset:2px;border-radius:3px}
.pt-sort{padding:0 !important}
.pt-sortbtn{all:unset;box-sizing:border-box;cursor:pointer;display:flex;align-items:center;justify-content:center;gap:2px;width:100%;min-height:44px;padding:0 4px;font:inherit;color:inherit;overflow:hidden}
.pt-sortbtn:hover{background:rgba(148,163,184,.08)}
.pt-sortbtn[data-left]{justify-content:flex-start}
.pt-sortbtn:focus-visible{outline:2px solid var(--accent);outline-offset:-2px}
.pt-v-wide{display:block}.pt-v-mid{display:none}.pt-v-card{display:none}
@media (max-width:1099px){.pt-v-wide{display:none}.pt-v-mid{display:block}}
@media (max-width:768px){.pt-v-mid{display:none}.pt-v-card{display:block}}
.pt-tbl{width:100%;table-layout:fixed;border-collapse:collapse;font-size:12px}
.pt-tbl th,.pt-tbl td{overflow:hidden;text-overflow:ellipsis}
.pt-tbl td{padding:7px 6px;border-bottom:1px solid rgba(36,48,73,.4);text-align:center;vertical-align:middle;font-size:12px}
.pt-tbl td.l{text-align:left}
.pt-tbl .sub{display:block;font-size:10px;color:var(--text-dim);margin-top:1px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-weight:400}
.pt-tbl .sep{border-left:1px solid rgba(36,48,73,.35)}
.pt-sortbar{display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin:0 0 8px;font-size:12px;color:var(--text-dim)}
.pt-chip{all:unset;box-sizing:border-box;cursor:pointer;min-height:36px;padding:0 12px;display:inline-flex;align-items:center;border:1px solid var(--border);border-radius:999px;font-size:12px;color:var(--text)}
.pt-chip[aria-pressed="true"]{border-color:var(--accent);color:var(--accent);background:rgba(59,130,246,.10)}
.pt-chip:focus-visible,.pt-select:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.pt-select{min-height:36px;padding:0 8px;background:var(--surface2);border:1px solid var(--border);border-radius:8px;color:var(--text);font-size:12px}
.pt-card{display:grid;gap:6px;padding:10px 10px 10px 12px;border-bottom:1px solid rgba(36,48,73,.5)}
.pt-card .r1{display:flex;align-items:center;gap:8px;min-width:0}
.pt-card .no{flex:0 0 auto;min-width:28px;height:28px;border-radius:6px;display:inline-flex;align-items:center;justify-content:center;font-weight:800;font-size:14px}
.pt-card .nm{flex:1 1 auto;min-width:0;font-weight:700;color:#fff;font-size:15px}
.pt-card .win{flex:0 0 auto;text-align:right;line-height:1.1}
.pt-card .win b{font-size:20px;font-variant-numeric:tabular-nums}
.pt-card .win small{display:block;font-size:10px;color:var(--text-dim)}
.pt-card .r2{display:flex;flex-wrap:wrap;gap:2px 10px;font-size:12px;color:var(--text-dim);align-items:center}
.pt-card .r3{display:flex;flex-wrap:wrap;gap:6px 14px;align-items:center;font-size:12px}
.pt-bar{display:inline-flex;align-items:center;gap:5px}
.pt-bar i{display:inline-block;width:52px;height:6px;border-radius:3px;background:rgba(148,163,184,.2);position:relative;overflow:hidden}
.pt-bar i::after{content:"";position:absolute;left:0;top:0;bottom:0;width:var(--w);background:var(--c);border-radius:3px}
.pt-badge{display:inline-flex;align-items:center;gap:4px;padding:2px 7px;border-radius:5px;font-weight:700;font-size:12px}
@media (pointer: coarse){ .pt-tap{padding:12px 4px} }
@media (max-width:639px){ .wp-main{padding:12px 6px !important} .wp-race-body{padding:0 4px 10px !important} }
`;

/* ── 予測テーブル ── */
function PredTable({ raceId, preds, hasPred, message, megu }: { raceId: string; preds: PredEntry[]; hasPred: boolean; message?: string; megu?: ExpectedMeguMeta | null }) {
  const TH: React.CSSProperties = {
    padding: "8px 10px", fontSize: 11, fontWeight: 600, color: "var(--text-dim)",
    textAlign: "center", borderBottom: "1px solid var(--border)", whiteSpace: "normal", lineHeight: 1.25, background: "var(--surface2)", overflow: "hidden",
  };
  const [sort, setSort] = useState<{ key: SortKey; dir: SortDir } | null>(null);
  const [card, setCard] = useState<{ horseId: string; tab: CardTab } | null>(null);   // 馬名・騎手・調教師のタップで開く情報パネル
  const defaultOrder = hasPred
    ? (a: PredEntry, b: PredEntry) => (a.composite_rank ?? a.pred_rank ?? 99) - (b.composite_rank ?? b.pred_rank ?? 99)
    : (a: PredEntry, b: PredEntry) => (a.horse_number ?? 0) - (b.horse_number ?? 0);
  const sorted = sortPreds(preds, sort?.key ?? null, sort?.dir ?? "asc", defaultOrder);
  const calibratedEv = preds.some((p) => p.ev_win_cal != null || p.ev_place_cal != null);     // EV の較正が有効なときだけ『(較正後)』と表示する

  // 見出しのクリック: 1 回目=その列の向き（既定は昇順、大きいほど良い列は降順）→ 2 回目=逆 → 3 回目=元の並びに戻す
  function toggle(key: SortKey) {
    setSort((cur) => {
      const first: SortDir = DESC_FIRST.has(key) ? "desc" : "asc";
      if (!cur || cur.key !== key) return { key, dir: first };
      if (cur.dir === first) return { key, dir: first === "asc" ? "desc" : "asc" };
      return null;
    });
  }
  const arrow = (key: SortKey) => (sort?.key === key ? (sort.dir === "asc" ? " ▲" : " ▼") : "");
  const th = (key: SortKey, label: string, style: React.CSSProperties = {}, title?: string, cls?: string) => (
    <th
      className={`pt-sort ${cls ?? ""}`}
      style={{ ...TH, cursor: "pointer", userSelect: "none", color: sort?.key === key ? "var(--accent)" : TH.color, ...style }}
      aria-sort={sort?.key === key ? (sort.dir === "asc" ? "ascending" : "descending") : "none"}
      title={title ? `${title}（クリックで並べ替え）` : "クリックで並べ替え"}
    >
      <button
        type="button"
        onClick={() => toggle(key)}
        className="pt-sortbtn"
        data-left={style.textAlign === "left" ? "" : undefined}
      >
        {label}<span aria-hidden style={{ fontSize: 9, opacity: sort?.key === key ? 1 : 0.35 }}>{arrow(key) || " ↕"}</span>
      </button>
    </th>
  );

  const TD: React.CSSProperties = {
    padding: "8px 10px", borderBottom: "1px solid rgba(36,48,73,0.4)",
    textAlign: "center", verticalAlign: "middle", fontSize: 12,
  };

  /* 馬名・騎手・調教師: タップ/クリック（Enter・スペース）で情報パネルを開く。horse_id が無い行は文字のみ。 */
  const tapCell = (p: PredEntry, tab: CardTab, text: string | null | undefined, color: string) => {
    if (!text) return "-";
    if (!p.horse_id) return text;
    return (
      <button
        type="button"
        className="pt-tap"
        aria-haspopup="dialog"
        title={tab === "horse" ? "馬の近走・血統・追い切りを見る" : tab === "jockey" ? "騎手の成績を見る" : "調教師の成績を見る"}
        onClick={() => setCard({ horseId: p.horse_id as string, tab })}
        style={{ color }}
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

  const rowBgOf = (p: PredEntry) => {
    const rank = p.composite_rank ?? p.pred_rank ?? 0;
    return rank === 1 ? "rgba(239,68,68,0.04)" : rank === 2 ? "rgba(59,130,246,0.04)" : rank === 3 ? "rgba(34,197,94,0.03)" : undefined;
  };
  const evOf = (p: PredEntry) => ({
    w: fmtEv(p.ev_win_cal ?? p.ev_win, p.ev_win_cal != null),
    pl: fmtEv(p.ev_place_cal ?? p.ev_place ?? p.expected_value, p.ev_place_cal != null),
  });
  const bracketChip = (p: PredEntry) =>
    p.bracket_number && BRACKET[p.bracket_number]
      ? <span title={`${p.bracket_number}枠`} style={{ display: "inline-block", minWidth: 20, borderRadius: 4, padding: "1px 0", fontWeight: 700, background: BRACKET[p.bracket_number].bg, color: BRACKET[p.bracket_number].color }}>{p.bracket_number}</span>
      : <span>-</span>;
  const placeOdds = (p: PredEntry) =>
    p.place_odds_min != null && p.place_odds_max != null ? `${Number(p.place_odds_min).toFixed(1)}〜${Number(p.place_odds_max).toFixed(1)}` : "-";
  const dim = hasPred ? "var(--text)" : "var(--text-dim)";
  const evLabel = calibratedEv ? "期待値(勝/複・較正後)" : "期待値(勝/複)";
  const evTitle = "勝期待値で並べ替え（複勝は『並び替え』の選択肢から）";

  // 並び替えバー: よく使う 4 つはチップ、そのほかは選択肢から。どの幅でも使える（表の見出しは主要な列だけ）
  const QUICK: { key: SortKey; label: string }[] = [
    { key: "num", label: "馬番" }, { key: "win", label: "勝率" }, { key: "evw", label: "期待値" }, { key: "emegu", label: "想定指数" },
  ];
  const ALL: { key: SortKey; label: string }[] = [
    { key: "num", label: "馬番" }, { key: "name", label: "馬名" }, { key: "bracket", label: "枠" }, { key: "sexage", label: "性齢" },
    { key: "jwt", label: "斤量" }, { key: "jockey", label: "騎手" }, { key: "trainer", label: "調教師" }, { key: "weight", label: "馬体重" },
    { key: "emegu", label: "想定指数" }, { key: "win", label: "勝率" }, { key: "top2", label: "連対率" }, { key: "top3", label: "複勝率" },
    { key: "odds", label: "想定単勝" }, { key: "podds", label: "想定複勝" }, { key: "evw", label: "勝期待値" }, { key: "evp", label: "複期待値" }, { key: "tier", label: "推奨" },
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

      {/* ── 広い画面（1100px〜）: 関連する列を 2 段にまとめた表 ── */}
      <div className="pt-v-wide">
        <table className="pt-tbl">
          <colgroup>
            <col style={{ width: "5%" }} /><col style={{ width: "22%" }} /><col style={{ width: "4.5%" }} /><col style={{ width: "16%" }} />
            <col style={{ width: "10%" }} /><col style={{ width: "7%" }} /><col style={{ width: "9.5%" }} /><col style={{ width: "11%" }} />
            <col style={{ width: "9%" }} /><col style={{ width: "6%" }} />
          </colgroup>
          <thead>
            <tr>
              {th("num", "馬番")}
              {th("name", "馬名", { textAlign: "left" }, "馬名（下段: 性齢・斤量・馬体重）")}
              {th("bracket", "枠", { borderLeft: "1px solid rgba(36,48,73,0.5)" }, "出馬表（前日に確定。馬体重は当日の発表後）")}
              {th("jockey", "騎手/調教師", { textAlign: "left" }, "騎手で並べ替え（調教師は『そのほか』から）")}
              {th("emegu", "想定指数", { borderLeft: "1px solid rgba(36,48,73,0.5)" }, "過去のめぐ指数からの想定（AI予測ではありません）。1点＝0.1秒")}
              {th("win", "勝率", { borderLeft: "1px solid rgba(36,48,73,0.5)" })}
              {th("top2", "連対/複勝", {}, "連対率で並べ替え（複勝率は『そのほか』から）")}
              {th("odds", "想定単勝/複勝", { borderLeft: "1px solid rgba(36,48,73,0.5)" }, "想定単勝で並べ替え（想定複勝は『そのほか』から）")}
              {th("evw", calibratedEv ? "期待値(較正後)" : "期待値", { borderLeft: "1px solid rgba(36,48,73,0.5)" }, evTitle)}
              {th("tier", "推奨", { borderLeft: "1px solid rgba(36,48,73,0.5)" })}
            </tr>
          </thead>
          <tbody>
            {sorted.map((p, i) => {
              const ev = evOf(p);
              return (
                <tr key={p.horse_number ?? i} style={{ background: rowBgOf(p) }}>
                  <td style={{ fontWeight: 800, color: "#7dd3fc" }}>{p.horse_number ?? "-"}</td>
                  <td className="l" style={{ fontWeight: 600, color: "#fff" }}>
                    {tapCell(p, "horse", p.horse_name, "#fff")}
                    <span className="sub">{[p.sex_age, p.jockey_weight != null ? `${p.jockey_weight}kg` : null, p.weight ? fmtWeight(p.weight, p.weight_change) : null].filter(Boolean).join("　")}</span>
                  </td>
                  <td className="sep">{bracketChip(p)}</td>
                  <td className="l">
                    {tapCell(p, "jockey", p.jockey_name, "var(--text)")}
                    <span className="sub">{p.trainer_name ? <>{tapCell(p, "trainer", p.trainer_name, "var(--text-dim)")}</> : "-"}</span>
                  </td>
                  <td className="sep"><ExpectedMeguCell e={p.expected_megu} /></td>
                  <td className="sep" style={{ fontWeight: hasPred ? 700 : 400, color: dim, fontSize: 13 }}>{fmtPct(p.win_prob)}</td>
                  <td style={{ color: dim }}>{fmtPct(p.top2_prob)}<span className="sub" style={{ color: dim }}>{fmtPct(p.top3_prob)}</span></td>
                  <td className="sep" style={{ color: "var(--text-dim)" }}>{fmtOdds(p.win_odds)}<span className="sub">{placeOdds(p)}</span></td>
                  <td className="sep" style={{ background: ev.w.bg, fontWeight: hasPred ? 700 : 400, color: ev.w.color }}>{ev.w.text}<span className="sub" style={{ color: ev.pl.color }}>{ev.pl.text}</span></td>
                  <td className="sep" style={{ fontSize: 11, color: "var(--text-dim)" }}>{p.buy_tier ?? "-"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* ── 中間（769〜1099px）: 馬名の下に騎手・斤量・体重を畳んだ表 ── */}
      <div className="pt-v-mid">
        <table className="pt-tbl">
          <colgroup>
            <col style={{ width: "8%" }} /><col style={{ width: "36%" }} /><col style={{ width: "13%" }} /><col style={{ width: "15%" }} />
            <col style={{ width: "14%" }} /><col style={{ width: "14%" }} />
          </colgroup>
          <thead>
            <tr>
              {th("num", "馬番")}
              {th("name", "馬名", { textAlign: "left" }, "馬名（下段: 枠・性齢・斤量・騎手・馬体重・調教師）")}
              {th("win", "勝率", { borderLeft: "1px solid rgba(36,48,73,0.5)" }, "勝率で並べ替え（連対率・複勝率は下段、『そのほか』から並べ替え）")}
              {th("emegu", "想定指数", { borderLeft: "1px solid rgba(36,48,73,0.5)" }, "過去のめぐ指数からの想定（AI予測ではありません）。1点＝0.1秒")}
              {th("evw", calibratedEv ? "期待値(較正後)" : "期待値", { borderLeft: "1px solid rgba(36,48,73,0.5)" }, evTitle)}
              {th("odds", "想定単勝", { borderLeft: "1px solid rgba(36,48,73,0.5)" }, "想定単勝で並べ替え（下段: 想定複勝）")}
            </tr>
          </thead>
          <tbody>
            {sorted.map((p, i) => {
              const ev = evOf(p);
              return (
                <tr key={p.horse_number ?? i} style={{ background: rowBgOf(p) }}>
                  <td style={{ fontWeight: 800, color: "#7dd3fc" }}>{p.horse_number ?? "-"}</td>
                  <td className="l" style={{ fontWeight: 600, color: "#fff" }}>
                    {tapCell(p, "horse", p.horse_name, "#fff")}
                    <span className="sub">{[p.bracket_number ? `${p.bracket_number}枠` : null, p.sex_age, p.jockey_weight != null ? `${p.jockey_weight}kg` : null, p.jockey_name, p.weight ? fmtWeight(p.weight, p.weight_change) : null].filter(Boolean).join("　")}</span>
                    <span className="sub">{[p.trainer_name, p.buy_tier && p.buy_tier !== "—" ? `推奨 ${p.buy_tier}` : null].filter(Boolean).join("　")}</span>
                  </td>
                  <td className="sep" style={{ fontWeight: hasPred ? 700 : 400, color: dim, fontSize: 13 }}>{fmtPct(p.win_prob)}<span className="sub" style={{ color: dim }}>{fmtPct(p.top2_prob)} / {fmtPct(p.top3_prob)}</span></td>
                  <td className="sep"><ExpectedMeguCell e={p.expected_megu} /></td>
                  <td className="sep" style={{ background: ev.w.bg, fontWeight: hasPred ? 700 : 400, color: ev.w.color }}>{ev.w.text}<span className="sub" style={{ color: ev.pl.color }}>複 {ev.pl.text}</span></td>
                  <td className="sep" style={{ color: "var(--text-dim)" }}>{fmtOdds(p.win_odds)}<span className="sub">{placeOdds(p)}</span></td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* ── 狭い画面（〜768px）: 1 頭 = 1 枚のカード ── */}
      <div className="pt-v-card" role="list">
        {sorted.map((p, i) => {
          const ev = evOf(p);
          const m = markSym(p);
          const bb = p.bracket_number && BRACKET[p.bracket_number] ? BRACKET[p.bracket_number] : { bg: "var(--surface2)", color: "#7dd3fc" };
          const bar = (v: number | null | undefined, c: string) => (
            <span className="pt-bar"><i style={{ ["--w" as string]: `${Math.min(100, Math.max(0, (v ?? 0) * 100))}%`, ["--c" as string]: c } as React.CSSProperties} /></span>
          );
          return (
            <div key={p.horse_number ?? i} className="pt-card" role="listitem" style={{ background: rowBgOf(p) }}>
              <div className="r1">
                <span className="no" title={p.bracket_number ? `${p.bracket_number}枠` : undefined} style={{ background: bb.bg, color: bb.color }}>{p.horse_number ?? "-"}</span>
                <span className="nm">{tapCell(p, "horse", p.horse_name, "#fff")}</span>
                {m.sym !== "-" && <span title="印" aria-label={`印 ${m.sym}`} style={{ color: m.color, fontWeight: 800, fontSize: 15 }}>{m.sym}</span>}
                <span className="win"><b style={{ color: dim }}>{fmtPct(p.win_prob)}</b><small>勝率</small></span>
              </div>
              <div className="r2">
                {p.sex_age && <span>{p.sex_age}</span>}
                {p.jockey_name && <span>{tapCell(p, "jockey", p.jockey_name, "var(--text)")}{p.jockey_weight != null && ` ${p.jockey_weight}kg`}</span>}
                {p.trainer_name && <span>{tapCell(p, "trainer", p.trainer_name, "var(--text-dim)")}</span>}
                {p.weight ? <span>馬体重 {fmtWeight(p.weight, p.weight_change)}</span> : null}
              </div>
              <div className="r3">
                <span className="pt-bar" title="連対率">連対 {bar(p.top2_prob, "#60a5fa")}<b style={{ color: dim, fontVariantNumeric: "tabular-nums" }}>{fmtPct(p.top2_prob)}</b></span>
                <span className="pt-bar" title="複勝率">複勝 {bar(p.top3_prob, "#34d399")}<b style={{ color: dim, fontVariantNumeric: "tabular-nums" }}>{fmtPct(p.top3_prob)}</b></span>
              </div>
              <div className="r3">
                <span className="pt-badge" title="勝期待値" style={{ background: ev.w.bg, color: ev.w.color }}>期待値 {ev.w.text}</span>
                <span className="pt-badge" title="複勝期待値" style={{ background: ev.pl.bg, color: ev.pl.color }}>複 {ev.pl.text}</span>
                <span style={{ color: "var(--text-dim)" }} title="想定単勝オッズ">単勝 {fmtOdds(p.win_odds)}</span>
                <span style={{ color: "var(--text-dim)" }} title="想定複勝オッズ">複勝 {placeOdds(p)}</span>
                {p.buy_tier && p.buy_tier !== "—" && <span style={{ color: "var(--text-dim)" }}>推奨 {p.buy_tier}</span>}
              </div>
              <div className="r3" style={{ color: "var(--text-dim)" }}>
                <span>想定指数 </span><ExpectedMeguCell e={p.expected_megu} />
              </div>
            </div>
          );
        })}
      </div>

      {megu && (
        <p style={{ fontSize: 11, color: "var(--text-dim)", padding: "6px 10px 0" }}>
          想定指数: {megu.name ?? "過去のめぐ指数からの想定（AI予測ではありません）"}。{megu.data_through ? `指数データは ${megu.data_through.slice(5).replace("-", "/")} まで。` : ""}
          1点＝0.1秒。「条件」は面・距離・クラス・休み明けなどが過去走と違う馬（±は実際の誤差の大きさ）。
        </p>
      )}
      {!hasPred && (
        <p style={{ fontSize: 11, color: "var(--text-dim)", padding: "8px 10px" }}>
          ※ {message ?? "予測未実行。数値は全て「—」です。レース詳細から「AI予測を実行」してください。"}
        </p>
      )}
    </div>
  );
}

/* ── 選んだ 1 レースの表示 ── */
function RacePanel({
  race, pred, loading, onReload, value,
}: {
  race: RaceItem;
  pred: PredData | null;
  loading: boolean;
  onReload: (id: string) => void;
  value?: RaceValue;
}) {
  const hasPred = pred?.has_prediction === true && (pred.predictions?.some((p) => p.win_prob != null) ?? false);
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

        {value && value.level !== "none" && (
          <span
            style={{
              fontSize: 11, fontWeight: 700, padding: "2px 9px", borderRadius: 6, whiteSpace: "nowrap",
              border: `1px solid ${LEVEL_STYLE[value.level].border}`, background: LEVEL_STYLE[value.level].bg, color: LEVEL_STYLE[value.level].color,
              boxShadow: LEVEL_STYLE[value.level].glow,
            }}
          >
            {value.level === "hot" ? "★ " : ""}推奨 {value.expected_roi != null ? `${(value.expected_roi * 100).toFixed(0)}%` : "見送り"}
          </span>
        )}

        {hasPred && <RaceSummaryBadges preds={preds} />}

        <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 8 }}>
          {pred && !failed && (
            <span style={{
              fontSize: 10, fontWeight: 600, padding: "2px 7px", borderRadius: 3,
              background: hasPred ? "rgba(34,197,94,0.12)" : "rgba(107,125,149,0.12)",
              color: hasPred ? "var(--ok)" : "var(--text-dim)",
            }}>
              {hasPred ? "✓ 予測済" : "未予測"}
            </span>
          )}
          <Link
            href={`/race/${race.race_id}`}
            className="hit44"
            style={{ fontSize: 11, color: "var(--accent)", textDecoration: "none", border: "1px solid rgba(59,130,246,0.3)", padding: "3px 8px", borderRadius: 4 }}
          >
            詳細 →
          </Link>
        </div>
      </div>

      <div className="wp-race-body" style={{ borderTop: "1px solid var(--border)", padding: "0 18px 14px" }}>
        {(loading || (!pred && !failed)) && (
          <div role="status" style={{ padding: "20px 0", textAlign: "center", color: "var(--text-dim)", fontSize: 13 }}>
            <div style={{ width: 24, height: 24, border: "2px solid var(--border)", borderTopColor: "var(--accent)", borderRadius: "50%", animation: "spin 1s linear infinite", margin: "0 auto 8px" }} />
            予測データを取得中…
          </div>
        )}
        {failed && (
          <div role="alert" style={{ margin: "14px 0", background: "rgba(239,68,68,0.1)", border: "1px solid rgba(239,68,68,0.3)", borderRadius: 8, padding: "12px 16px", color: "var(--err)", fontSize: 13, display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
            <span>⚠️ このレースの予測を取得できませんでした。</span>
            <button
              type="button"
              onClick={() => onReload(race.race_id)}
              style={{ marginLeft: "auto", minHeight: 44, padding: "0 16px", borderRadius: 8, border: "1px solid var(--accent)", background: "rgba(59,130,246,0.15)", color: "var(--accent)", cursor: "pointer", fontSize: 13 }}
            >
              再読み込み
            </button>
          </div>
        )}
        {pred && !failed && <PredictionSourceBadge source={pred.prediction_source} note={pred.prediction_note} />}
        {pred && !failed && (
          <PredTable raceId={race.race_id} preds={preds} hasPred={hasPred} message={pred.message} megu={pred.expected_megu_meta} />
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
  const raceValues = useRaceValues(selectedDate);
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
    const r = q.get("race");
    if (r) setSelectedRaceId(r);
  }, []);

  function firstRaceOf(v: string): RaceItem | undefined {
    return races.filter((r) => r.venue === v).sort((a, b) => Number(a.round ?? 0) - Number(b.round ?? 0))[0];
  }

  function chooseVenue(v: string) {
    const first = firstRaceOf(v);
    setSelectedVenue(v);
    setSelectedRaceId(first?.race_id ?? "");
    syncUrl({ venue: v, race: first?.race_id ?? null });
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
    try {
      const d = await getJson<PredData>(`/api/race/${raceId}/predictions`, { ttlMs: 15_000, force }).catch((): PredData => ({ status: "error", predictions: [] }));
      setPreds((prev) => ({ ...prev, [raceId]: d }));
    } catch {
      setPreds((prev) => ({ ...prev, [raceId]: { status: "error", predictions: [] } }));
    } finally {
      setLoadingPred((prev) => ({ ...prev, [raceId]: false }));
    }
  }, []);

  /* マウスが乗った／触れ始めたレースの予測を先読みする。連続で動かしたときは、止まった（150ms）レースだけ */
  const warmTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const warmPred = useCallback((id: string) => {
    if (USE_MOCK) return;
    if (warmTimer.current) clearTimeout(warmTimer.current);
    warmTimer.current = setTimeout(() => prefetchJson(`/api/race/${id}/predictions`, { ttlMs: 15_000 }), 150);
  }, []);

  /* 選んだレースの予測だけを取得する（素早く切り替えたときは最後に選んだものだけ。取得済みは即表示）。
     表示は activeRaceId の状態だけを見るので、遅れて届いた別レースの応答が画面を上書きすることはない */
  useEffect(() => {
    if (!activeRaceId || preds[activeRaceId] || loadingPred[activeRaceId]) return;
    const id = activeRaceId;
    const timer = setTimeout(() => { void loadPred(id); }, 120);
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeRaceId, preds]);

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
    <div style={{ minHeight: "100vh", background: "var(--bg)", color: "var(--text)" }}>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>

      {/* ページヘッダー */}
      <div style={{ background: "var(--surface)", borderBottom: "1px solid var(--border)", padding: "18px 24px" }}>
        <div style={{ maxWidth: 1200, margin: "0 auto", display: "flex", alignItems: "center", gap: 12 }}>
          <Link href="/" className="hide-narrow" style={{ fontSize: 12, color: "var(--text-dim)", textDecoration: "none" }}>← ホーム</Link>
          <span className="hide-narrow" style={{ color: "var(--border)" }}>/</span>
          <h1 style={{ fontSize: 18, fontWeight: 700, color: "#f0f6fc", margin: 0 }}>🤖 今週のAI予測</h1>
          <p className="hide-narrow" style={{ fontSize: 12, color: "var(--text-dim)", margin: 0 }}>
            前回と次回の開催（最大2週分）の勝率・期待値を一覧表示
          </p>
        </div>
      </div>

      {/* 開催日の選択（週単位） */}
      <div style={{ background: "var(--surface)", borderBottom: "1px solid var(--border)", padding: "14px 24px" }}>
        <div style={{ maxWidth: 1200, margin: "0 auto", display: "flex", flexDirection: "column", gap: 12 }}>
          <WeekPicker dates={dates} today={today} selectedDate={selectedDate} loading={loadingDates} onSelectDate={chooseDate} />
        </div>
      </div>

      {/* メインコンテンツ */}
      <div className="wp-main" style={{ maxWidth: 1200, margin: "0 auto", padding: "20px 24px" }}>
        {/* 凡例 */}
        <div style={{ marginBottom: 16, padding: "10px 14px", background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 11, color: "var(--text-dim)", display: "flex", gap: 16, flexWrap: "wrap" }}>
          <span>印: ◎1着優位・○連対候補・▲単穴・★中穴妙味は各1頭 / △は馬券内に来るかもしれない馬（複数・3連系の相手候補）</span>
          <span>
            期待値（過去の実績に合わせて較正した期待回収率）: <strong style={{ color: "#22c55e" }}>≥1.10</strong> / <strong style={{ color: "#60a5fa" }}>≥1.00</strong> / <strong style={{ color: "var(--text-dim)" }}>≥0.90</strong> / <strong style={{ color: "#f87171" }}>＜0.90</strong>
          </span>
        </div>

        {loadingRaces && (
          <div style={{ textAlign: "center", padding: "60px 0", color: "var(--text-dim)" }}>
            <div style={{ width: 36, height: 36, border: "3px solid var(--border)", borderTopColor: "var(--accent)", borderRadius: "50%", animation: "spin 1s linear infinite", margin: "0 auto 12px" }} />
            <p style={{ fontSize: 14 }}>レース一覧を読み込み中…</p>
          </div>
        )}

        {(datesError || racesError) && (
          <div role="alert" style={{ background: "rgba(239,68,68,0.1)", border: "1px solid rgba(239,68,68,0.3)", borderRadius: 8, padding: "14px 18px", color: "var(--err)", fontSize: 13, marginBottom: 16, display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
            <span>⚠️ {datesError || `レース一覧を取得できませんでした（${racesError}）`}</span>
            <button
              type="button"
              onClick={() => { if (datesError) { setLoadingDates(true); setDatesReload((n) => n + 1); } else loadRaces(selectedDate); }}
              style={{ marginLeft: "auto", minHeight: 44, padding: "0 16px", borderRadius: 8, border: "1px solid var(--accent)", background: "rgba(59,130,246,0.15)", color: "var(--accent)", cursor: "pointer", fontSize: 13 }}
            >
              再読み込み
            </button>
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
            <RaceStrip
              races={races}
              venue={activeVenue}
              onVenue={chooseVenue}
              values={raceValues}
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
              />
            )}
          </>
        )}
      </div>
    </div>
  );
}
