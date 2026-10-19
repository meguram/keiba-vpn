"use client";

import { fetchRaceList, getJson } from "@/lib/httpCache";
import { useEffect, useState, useCallback, useRef, useMemo } from "react";
import Link from "next/link";
import { USE_MOCK, MOCK_WEEKLY_RACES, getMockRaceDates, getMockPredictions } from "@/lib/mock";
import { featureEnabled } from "@/lib/features";
import { LEVEL_STYLE, RaceValue, useRaceValues, valueTitle } from "@/lib/raceValue";
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
  anchor:   { sym: "✓", color: "#22c55e" },
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

function fmtEv(v: number | null | undefined): { text: string; color: string; bg: string } {
  if (v == null) return { text: "-", color: "var(--text-dim)", bg: "transparent" };
  const n = Number(v);
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
  const evOk = hasP.filter((p) => (p.ev_win ?? 0) >= 1.0).length;
  const evGreat = hasP.filter((p) => (p.ev_win ?? 0) >= 1.3).length;
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
  evw: (p) => p.ev_win ?? null,
  evp: (p) => p.ev_place ?? p.expected_value ?? null,
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

/* 出馬表の表のスマホ最適化: 馬番・馬名を固定表示して横スクロール、重要度の低い列は幅が狭いと隠す（パネルで見られる）。見出しは指で押しやすく */
const TABLE_CSS = `
.pt-tap{all:unset;cursor:pointer;display:inline-block;padding:6px 2px;border-bottom:1px dotted rgba(148,163,184,.55);font:inherit;font-weight:inherit;line-height:1.3}
.pt-tap:hover{border-bottom-color:var(--accent)}
.pt-tap:focus-visible{outline:2px solid var(--accent);outline-offset:2px;border-radius:3px}
.pt-sort{padding:0 !important}
.pt-sortbtn{all:unset;box-sizing:border-box;cursor:pointer;display:flex;align-items:center;justify-content:center;gap:2px;width:100%;min-height:44px;padding:0 10px;font:inherit;color:inherit}
.pt-sortbtn:hover{background:rgba(148,163,184,.08)}
.pt-sortbtn[data-left]{justify-content:flex-start}
.pt-sortbtn:focus-visible{outline:2px solid var(--accent);outline-offset:-2px}
.pt-c1 .pt-sortbtn{padding:0 2px}
@media (max-width: 899px){
  .pt-c1,.pt-c2{position:sticky;background:var(--surface);z-index:1}
  .pt-c1{left:0;min-width:36px;max-width:36px}
  .pt-c2{left:36px;box-shadow:2px 0 0 rgba(36,48,73,.6);max-width:110px;overflow:hidden;text-overflow:ellipsis}
  th.pt-c1,th.pt-c2{background:var(--surface2);z-index:2}
}
@media (max-width: 639px){ .pt-hide-sm{display:none} .wp-main{padding:12px 6px !important} .wp-race-body{padding:0 4px 10px !important} .pt-c2{max-width:128px} th.pt-c2,td.pt-c2{padding-left:6px !important;padding-right:6px !important} }
@media (pointer: coarse){
  .pt-tap{padding:12px 4px}
}
`;

/* ── 予測テーブル ── */
function PredTable({ raceId, preds, hasPred, message, megu }: { raceId: string; preds: PredEntry[]; hasPred: boolean; message?: string; megu?: ExpectedMeguMeta | null }) {
  const TH: React.CSSProperties = {
    padding: "8px 10px", fontSize: 11, fontWeight: 600, color: "var(--text-dim)",
    textAlign: "center", borderBottom: "1px solid var(--border)", whiteSpace: "nowrap", background: "var(--surface2)",
  };
  const [sort, setSort] = useState<{ key: SortKey; dir: SortDir } | null>(null);
  const [card, setCard] = useState<{ horseId: string; tab: CardTab } | null>(null);   // 馬名・騎手・調教師のタップで開く情報パネル
  const defaultOrder = hasPred
    ? (a: PredEntry, b: PredEntry) => (a.composite_rank ?? a.pred_rank ?? 99) - (b.composite_rank ?? b.pred_rank ?? 99)
    : (a: PredEntry, b: PredEntry) => (a.horse_number ?? 0) - (b.horse_number ?? 0);
  const sorted = sortPreds(preds, sort?.key ?? null, sort?.dir ?? "asc", defaultOrder);

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

  return (
    <div style={{ overflowX: "auto", marginTop: 12 }}>
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
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
        <thead>
          <tr>
            {th("num", "馬番", { width: 36, minWidth: 36 }, undefined, "pt-c1")}
            {th("name", "馬名", { textAlign: "left", width: 120 }, undefined, "pt-c2")}
            {th("bracket", "枠", { borderLeft: "1px solid rgba(36,48,73,0.5)" }, "出馬表（前日に確定。馬体重は当日の発表後）")}
            {th("sexage", "性齢")}
            {th("jwt", "斤量", {}, undefined, "pt-hide-sm")}
            {th("jockey", "騎手", { textAlign: "left" })}
            {th("trainer", "調教師", { textAlign: "left" }, undefined, "pt-hide-sm")}
            {th("weight", "馬体重", {}, "当日、発走の約1時間前に発表")}
            {th("emegu", "想定指数", { borderLeft: "1px solid rgba(36,48,73,0.5)" }, "過去のめぐ指数からの想定（AI予測ではありません）。1点＝0.1秒")}
            {th("win", "勝率", { borderLeft: "1px solid rgba(36,48,73,0.5)" })}
            {th("top2", "連対率", {}, undefined, "pt-hide-sm")}
            {th("top3", "複勝率", {}, undefined, "pt-hide-sm")}
            {th("odds", "想定単勝", { borderLeft: "1px solid rgba(36,48,73,0.5)" })}
            {th("podds", "想定複勝", {}, "複勝オッズの下限で並べ替え", "pt-hide-sm")}
            {th("evw", "勝期待値", { borderLeft: "1px solid rgba(36,48,73,0.5)" })}
            {th("evp", "複期待値", {}, undefined, "pt-hide-sm")}
            {th("tier", "推奨", { borderLeft: "1px solid rgba(36,48,73,0.5)", width: 60 }, undefined, "pt-hide-sm")}
          </tr>
        </thead>
        <tbody>
          {sorted.map((p, i) => {
            const rank = p.composite_rank ?? p.pred_rank ?? 0;
            const evW = fmtEv(p.ev_win);
            const evP = fmtEv(p.ev_place ?? p.expected_value);
            const rowBg =
              rank === 1 ? "rgba(239,68,68,0.04)" :
              rank === 2 ? "rgba(59,130,246,0.04)" :
              rank === 3 ? "rgba(34,197,94,0.03)" : undefined;
            return (
              <tr key={p.horse_number ?? i} style={{ background: rowBg }}>
                <td className="pt-c1" style={{ ...TD, fontWeight: 800, color: "#7dd3fc" }}>{p.horse_number ?? "-"}</td>
                <td className="pt-c2" style={{ ...TD, textAlign: "left", fontWeight: 600, color: "#fff", whiteSpace: "nowrap" }}>{tapCell(p, "horse", p.horse_name, "#fff")}</td>
                <td style={{ ...TD, borderLeft: "1px solid rgba(36,48,73,0.3)" }}>
                  {p.bracket_number && BRACKET[p.bracket_number]
                    ? <span style={{ display: "inline-block", minWidth: 20, borderRadius: 4, padding: "1px 0", fontWeight: 700, background: BRACKET[p.bracket_number].bg, color: BRACKET[p.bracket_number].color }}>{p.bracket_number}</span>
                    : "-"}
                </td>
                <td style={TD}>{p.sex_age ?? "-"}</td>
                <td className="pt-hide-sm" style={TD}>{p.jockey_weight ?? "-"}</td>
                <td style={{ ...TD, textAlign: "left", whiteSpace: "nowrap" }}>{tapCell(p, "jockey", p.jockey_name, "var(--text)")}</td>
                <td className="pt-hide-sm" style={{ ...TD, textAlign: "left", whiteSpace: "nowrap" }}>{tapCell(p, "trainer", p.trainer_name, "var(--text)")}</td>
                <td style={{ ...TD, whiteSpace: "nowrap" }}>{fmtWeight(p.weight, p.weight_change)}</td>
                <td style={{ ...TD, borderLeft: "1px solid rgba(36,48,73,0.3)" }}><ExpectedMeguCell e={p.expected_megu} /></td>
                <td style={{ ...TD, borderLeft: "1px solid rgba(36,48,73,0.3)", fontWeight: hasPred ? 600 : 400, color: hasPred ? "var(--text)" : "var(--text-dim)" }}>{fmtPct(p.win_prob)}</td>
                <td className="pt-hide-sm" style={{ ...TD, color: hasPred ? "var(--text)" : "var(--text-dim)" }}>{fmtPct(p.top2_prob)}</td>
                <td className="pt-hide-sm" style={{ ...TD, color: hasPred ? "var(--text)" : "var(--text-dim)" }}>{fmtPct(p.top3_prob)}</td>
                <td style={{ ...TD, borderLeft: "1px solid rgba(36,48,73,0.3)", color: "var(--text-dim)" }}>{fmtOdds(p.win_odds)}</td>
                <td className="pt-hide-sm" style={{ ...TD, color: "var(--text-dim)", fontSize: 11 }}>
                  {p.place_odds_min != null && p.place_odds_max != null
                    ? `${Number(p.place_odds_min).toFixed(1)}〜${Number(p.place_odds_max).toFixed(1)}`
                    : "-"}
                </td>
                <td style={{ ...TD, borderLeft: "1px solid rgba(36,48,73,0.3)", background: evW.bg, fontWeight: hasPred ? 700 : 400, color: evW.color }}>{evW.text}</td>
                <td className="pt-hide-sm" style={{ ...TD, background: evP.bg, fontWeight: hasPred ? 700 : 400, color: evP.color }}>{evP.text}</td>
                <td className="pt-hide-sm" style={{ ...TD, borderLeft: "1px solid rgba(36,48,73,0.3)", fontSize: 11, color: "var(--text-dim)" }}>{p.buy_tier ?? "-"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
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

/* ── レースカード ── */
function RaceCard({
  race,
  pred,
  loading,
  onLoad,
  value,
  selected,
  selectNonce,
}: {
  race: RaceItem;
  selected: boolean;
  selectNonce: number;
  pred: PredData | null;
  loading: boolean;
  onLoad: (id: string) => void;
  value?: RaceValue;
}) {
  const [expanded, setExpanded] = useState(false);
  const cardRef = useRef<HTMLDivElement | null>(null);
  // R チップ（RaceStrip）で選ばれたら、開いて予測を読み込み、画面内に移動する（同じ R をもう一度押しても移動する）
  useEffect(() => {
    if (!selected || selectNonce === 0) return;
    setExpanded(true);
    if (!pred && !loading) onLoad(race.race_id);
    cardRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected, selectNonce]);
  const hasPred = pred?.has_prediction === true && (pred.predictions?.some((p) => p.win_prob != null) ?? false);
  const preds = pred?.predictions ?? [];

  return (
    <div
      ref={cardRef}
      id={`race-${race.race_id}`}
      title={valueTitle(value)}
      style={{
        scrollMarginTop: 230,
        outline: selected ? "2px solid var(--accent)" : undefined,
        background: value?.level === "hot" ? "linear-gradient(90deg, rgba(249,115,22,0.10), var(--surface) 40%)" : "var(--surface)",
        border: "1px solid var(--border)", borderLeft: `4px solid ${value ? LEVEL_STYLE[value.level].border : "var(--border)"}`,
        borderRadius: 10, overflow: "hidden", marginBottom: 12,
        boxShadow: value?.level === "hot" ? "0 0 12px rgba(249,115,22,0.25)" : undefined,
      }}
    >
      {/* ヘッダー */}
      <div
        style={{ padding: "14px 18px", display: "flex", alignItems: "center", gap: 12, cursor: "pointer", flexWrap: "wrap" }}
        onClick={() => {
          setExpanded(!expanded);
          if (!pred && !loading) onLoad(race.race_id);
        }}
      >
        <span style={{ background: "var(--accent)", color: "#fff", fontWeight: 800, fontSize: 13, padding: "3px 10px", borderRadius: 4, minWidth: 40, textAlign: "center" }}>
          {race.round}R
        </span>
        <span style={{ fontWeight: 700, fontSize: 14, color: "#fff" }}>
          {race.race_name ?? race.race_id}
          {gradeChip(race.grade)}
        </span>
        <span style={{ fontSize: 12, color: "var(--text-dim)" }}>
          {race.venue} {race.distance && `${race.distance}m`} {race.surface}
        </span>

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

        {/* 予測サマリー */}
        {hasPred && <RaceSummaryBadges preds={preds} />}

        <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 8 }}>
          {pred && (
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
            onClick={(e) => e.stopPropagation()}
            style={{ fontSize: 11, color: "var(--accent)", textDecoration: "none", border: "1px solid rgba(59,130,246,0.3)", padding: "3px 8px", borderRadius: 4 }}
          >
            詳細 →
          </Link>
          <span style={{ fontSize: 16, color: "var(--text-dim)", userSelect: "none" }}>{expanded ? "▲" : "▼"}</span>
        </div>
      </div>

      {/* 展開コンテンツ */}
      {expanded && (
        <div className="wp-race-body" style={{ borderTop: "1px solid var(--border)", padding: "0 18px 14px" }}>
          {loading && (
            <div style={{ padding: "20px 0", textAlign: "center", color: "var(--text-dim)", fontSize: 13 }}>
              <div style={{ width: 24, height: 24, border: "2px solid var(--border)", borderTopColor: "var(--accent)", borderRadius: "50%", animation: "spin 1s linear infinite", margin: "0 auto 8px" }} />
              予測データを取得中…
            </div>
          )}
          {!loading && !pred && (
            <div style={{ padding: "16px 0", display: "flex", gap: 10, alignItems: "center" }}>
              <button
                style={{ background: "#1e3a5f", color: "#60a5fa", border: "1px solid rgba(59,130,246,0.3)", padding: "6px 14px", borderRadius: 6, fontSize: 12, fontWeight: 600, cursor: "pointer" }}
                onClick={() => onLoad(race.race_id)}
              >
                🤖 予測を取得
              </button>
              <Link href={`/race/${race.race_id}`} style={{ fontSize: 12, color: "var(--accent)" }}>
                レース詳細で予測を実行 →
              </Link>
            </div>
          )}
          {pred && (
            <PredTable raceId={race.race_id} preds={preds} hasPred={hasPred} message={pred.message} megu={pred.expected_megu_meta} />
          )}
        </div>
      )}
    </div>
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
  const [bulkLoading, setBulkLoading] = useState(false);
  const bulkAbort = useRef<AbortController | null>(null);
  const raceValues = useRaceValues(selectedDate);
  const [selectedVenue, setSelectedVenue] = useState("");
  const [selectedRaceId, setSelectedRaceId] = useState("");
  const [selectNonce, setSelectNonce] = useState(0);
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

  useEffect(() => {
    const q = new URLSearchParams(window.location.search);
    const v = q.get("venue");
    if (v) setSelectedVenue(v);
    const r = q.get("race");
    if (r) { setSelectedRaceId(r); setSelectNonce(1); }      // ?race= で開いたときは、そのレースへ移動して開く
  }, []);

  function chooseVenue(v: string) {
    setSelectedVenue(v);
    setSelectedRaceId("");
    syncUrl({ venue: v, race: null });
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
    setSelectNonce((n) => n + 1);
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
  const loadPred = useCallback(async (raceId: string) => {
    setLoadingPred((prev) => ({ ...prev, [raceId]: true }));
    if (USE_MOCK) {
      await new Promise((r) => setTimeout(r, 200));
      setPreds((prev) => ({ ...prev, [raceId]: getMockPredictions(raceId) }));
      setLoadingPred((prev) => ({ ...prev, [raceId]: false }));
      return;
    }
    try {
      const d = await getJson<PredData>(`/api/race/${raceId}/predictions`, { ttlMs: 15_000 }).catch((): PredData => ({ status: "error", predictions: [] }));
      setPreds((prev) => ({ ...prev, [raceId]: d }));
    } catch {
      setPreds((prev) => ({ ...prev, [raceId]: { status: "error", predictions: [] } }));
    } finally {
      setLoadingPred((prev) => ({ ...prev, [raceId]: false }));
    }
  }, []);

  /* 全レース一括取得 */
  async function loadAllPreds() {
    if (!visibleRaces.length) return;
    setBulkLoading(true);
    bulkAbort.current = new AbortController();
    const signal = bulkAbort.current.signal;
    for (const race of visibleRaces) {
      if (signal.aborted) break;
      if (preds[race.race_id]) continue;
      setLoadingPred((prev) => ({ ...prev, [race.race_id]: true }));
      if (USE_MOCK) {
        await new Promise((r) => setTimeout(r, 100));
        if (!signal.aborted) {
          setPreds((prev) => ({ ...prev, [race.race_id]: getMockPredictions(race.race_id) }));
        }
        setLoadingPred((prev) => ({ ...prev, [race.race_id]: false }));
        continue;
      }
      try {
        const res = await fetch(`/api/race/${race.race_id}/predictions`, { signal });
        const d: PredData = res.ok ? await res.json() : { status: "error", predictions: [] };
        setPreds((prev) => ({ ...prev, [race.race_id]: d }));
      } catch { /* ignore abort */ } finally {
        setLoadingPred((prev) => ({ ...prev, [race.race_id]: false }));
      }
    }
    setBulkLoading(false);
  }

  function stopBulk() {
    bulkAbort.current?.abort();
    setBulkLoading(false);
  }

  /* 統計 */
  const visiblePreds = visibleRaces.map((r) => preds[r.race_id]).filter((p): p is PredData => !!p);
  const loadedCount = visiblePreds.length;
  const predOkCount = visiblePreds.filter((p) => p.has_prediction).length;
  const totalEVOk = visiblePreds
    .flatMap((p) => p.predictions ?? [])
    .filter((h) => (h.ev_win ?? 0) >= 1.0).length;

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

      {/* コントロールバー */}
      <div style={{ background: "var(--surface)", borderBottom: "1px solid var(--border)", padding: "12px 24px" }}>
        <div style={{ maxWidth: 1200, margin: "0 auto", display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
          {visibleRaces.length > 0 && (
            <>
              {!bulkLoading ? (
                <button
                  style={{ background: "#1e3a5f", color: "#60a5fa", border: "1px solid rgba(59,130,246,0.3)", padding: "7px 14px", borderRadius: 6, fontSize: 12, fontWeight: 600, cursor: "pointer" }}
                  onClick={loadAllPreds}
                >
                  🤖 {activeVenue} 全{visibleRaces.length}レースの予測を取得
                </button>
              ) : (
                <button
                  style={{ background: "rgba(239,68,68,0.12)", color: "#f87171", border: "1px solid rgba(239,68,68,0.3)", padding: "7px 14px", borderRadius: 6, fontSize: 12, fontWeight: 600, cursor: "pointer" }}
                  onClick={stopBulk}
                >
                  ⏹ 取得を中止
                </button>
              )}
            </>
          )}

          {/* 統計バッジ */}
          {loadedCount > 0 && (
            <div style={{ display: "flex", gap: 8, marginLeft: 4, flexWrap: "wrap" }}>
              <span style={{ fontSize: 11, padding: "3px 8px", borderRadius: 4, background: "rgba(107,125,149,0.12)", color: "var(--text-dim)" }}>
                {loadedCount}/{visibleRaces.length}R 取得済
              </span>
              {predOkCount > 0 && (
                <span style={{ fontSize: 11, padding: "3px 8px", borderRadius: 4, background: "rgba(34,197,94,0.10)", color: "var(--ok)" }}>
                  予測済: {predOkCount}R
                </span>
              )}
              {totalEVOk > 0 && (
                <span style={{ fontSize: 11, padding: "3px 8px", borderRadius: 4, background: "rgba(59,130,246,0.10)", color: "#60a5fa" }}>
                  EV≥1.0: 計{totalEVOk}頭
                </span>
              )}
            </div>
          )}
        </div>
      </div>

      {/* メインコンテンツ */}
      <div className="wp-main" style={{ maxWidth: 1200, margin: "0 auto", padding: "20px 24px" }}>
        {/* 凡例 */}
        <div style={{ marginBottom: 16, padding: "10px 14px", background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 11, color: "var(--text-dim)", display: "flex", gap: 16, flexWrap: "wrap" }}>
          <span>
            期待値: <strong style={{ color: "#22c55e" }}>≥1.3</strong> / <strong style={{ color: "#60a5fa" }}>≥1.0</strong> / <strong style={{ color: "var(--text-dim)" }}>≥0.8</strong> / <strong style={{ color: "#f87171" }}>＜0.8</strong>
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
              selectedRaceId={selectedRaceId}
              onSelectRace={chooseRace}
            />
            <div style={{ marginBottom: 12, fontSize: 13, color: "var(--text-dim)" }}>
              {selectedDate}{activeVenue ? ` ${activeVenue}` : ""} — {visibleRaces.length}レース
              {bulkLoading && <span style={{ marginLeft: 8, color: "var(--accent)" }}>⏳ 取得中…</span>}
            </div>
            {visibleRaces.map((race) => (
              <RaceCard
                key={race.race_id}
                race={race}
                pred={preds[race.race_id] ?? null}
                loading={loadingPred[race.race_id] ?? false}
                onLoad={loadPred}
                value={raceValues.get(race.race_id)}
                selected={race.race_id === selectedRaceId}
                selectNonce={selectNonce}
              />
            ))}
          </>
        )}
      </div>
    </div>
  );
}
