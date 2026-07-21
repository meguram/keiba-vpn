"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import Link from "next/link";
import { USE_MOCK, MOCK_WEEKLY_RACES, getMockRaceDates, getMockMeguPredicted } from "@/lib/mock";

/* ── 型定義 ── */
type RaceItem = {
  race_id: string;
  race_name?: string;
  venue?: string;
  round?: number | string;
  date?: string;
  distance?: string | number;
  surface?: string;
  grade?: string;
  track_condition?: string;
  start_time?: string;
  entries_count?: number;
};

type ConditionChange = {
  type: "none" | "surface" | "distance" | "both";
  label: string | null;
  delta_mean?: number | null;
  delta_std?: number | null;
  transfer_sample_count?: number;
};

type MeguHorse = {
  horse_id: string;
  horse_name: string | null;
  horse_number: number | null;
  bracket_number?: number | null;
  sex_age?: string | null;
  jockey_weight: number | null;
  finish_time_sec: number | null;
  finish_pos?: number | null;
  actual_megu: number | null;
  actual_status?: string | null;
  base_megu: number | null;
  megu_adjusted: number | null;
  weight_megu_delta?: number | null;
  megu_final: number | null;
  megu_gap?: number | null;
  pred_margin_sec?: number | null;
  actual_margin_sec?: number | null;
  condition_change: ConditionChange;
  history: {
    race_id: string;
    race_date: string;
    venue: string;
    surface: string;
    distance: number;
    megu_index: number;
    finish_pos: number | null;
  }[];
};

type RaceInfo = {
  race_name: string | null;
  venue: string | null;
  surface: string | null;
  distance: number | null;
  dist_band: string | null;
  track_condition: string | null;
  grade: string | null;
  race_date: string | null;
};

type RaceLevel = {
  label: string;  // "未勝利級" | "1勝級" | "2勝級" | "3勝級" | "重賞級" | "G1級" | "?"
  field_avg_megu: number | null;
};

type ResultStats = {
  finisher_count: number;
  actual_megu_displayed_count: number;
  megu_coverage_ok?: boolean;
};

type MeguPredicted = {
  race_id: string;
  race_info: RaceInfo;
  race_level?: RaceLevel;
  result_stats?: ResultStats;
  model_version: string;
  index_note?: string;
  horses: MeguHorse[];
};

type SortKey = "horse_number" | "finish_pos" | "jockey_weight" | "finish_time_sec" | "megu_final" | "actual_megu" | "megu_gap" | "pred_margin_sec";
type SortDir = "asc" | "desc";

/* ── ユーティリティ ── */
function parseYmd(s: string): Date {
  return new Date(
    parseInt(s.slice(0, 4), 10),
    parseInt(s.slice(4, 6), 10) - 1,
    parseInt(s.slice(6, 8), 10),
  );
}

function formatYmd(d: Date): string {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}${m}${day}`;
}

/** 今週の土日（Sat/Sun）を返す */
function getThisWeekendDates(base: Date): [Date, Date] {
  const d = new Date(base);
  d.setHours(0, 0, 0, 0);
  const dow = d.getDay();
  if (dow === 6) {
    const sat = new Date(d);
    const sun = new Date(d);
    sun.setDate(sun.getDate() + 1);
    return [sat, sun];
  }
  if (dow === 0) {
    const sat = new Date(d);
    sat.setDate(sat.getDate() - 1);
    const sun = new Date(d);
    return [sat, sun];
  }
  const sat = new Date(d);
  sat.setDate(sat.getDate() + (6 - dow));
  const sun = new Date(sat);
  sun.setDate(sun.getDate() + 1);
  return [sat, sun];
}

/** めぐ指数ページのデフォルト開催日: 当週週末 → 翌週末 → 直近未来 → 直近過去 */
function pickDefaultMeguDate(dates: string[], meguDates?: string[]): string {
  if (!dates.length) return "";
  const set = new Set(dates);
  const meguSet = new Set(meguDates ?? []);
  const today = new Date();
  today.setHours(0, 0, 0, 0);

  const weekendCandidates = (): string[] => {
    const [sat, sun] = getThisWeekendDates(today);
    const nextSat = new Date(sat);
    nextSat.setDate(nextSat.getDate() + 7);
    const nextSun = new Date(sun);
    nextSun.setDate(nextSun.getDate() + 7);
    return [sat, sun, nextSat, nextSun].map(formatYmd);
  };

  // めぐ指数計算済みの直近過去日を最優先
  const meguPast = (meguDates ?? []).filter(d => set.has(d) && parseYmd(d) <= today).sort().reverse();
  if (meguPast.length) return meguPast[0];

  // 今週末（過去分のみ。未来の空データ日を避ける）
  for (const c of weekendCandidates()) {
    if (set.has(c) && parseYmd(c) <= today) return c;
  }

  const past = dates.filter(d => parseYmd(d) <= today).sort().reverse();
  if (past.length) return past[0];

  const future = dates.filter(d => parseYmd(d) >= today).sort();
  if (future.length) return future[0];

  if (meguSet.size) {
    const md = [...meguSet].sort().reverse();
    if (md.length) return md[0];
  }

  return [...dates].sort().reverse()[0];
}
/**
 * レースレベルバッジのスタイル。
 * ラベルの末尾クラス部分（"G1級", "重賞級", "3勝級", "2勝級", "1勝級", "未勝利級"）
 * で色を決定。
 */
const RACE_LEVEL_CLASS_STYLE: Record<string, { color: string; bg: string; border: string }> = {
  "G1級":  { color: "#fbbf24", bg: "rgba(251,191,36,0.18)",  border: "rgba(251,191,36,0.6)" },
  "重賞級": { color: "#f87171", bg: "rgba(239,68,68,0.14)",   border: "rgba(239,68,68,0.5)" },
  "3勝級": { color: "#c084fc", bg: "rgba(192,132,252,0.13)", border: "rgba(192,132,252,0.45)" },
  "2勝級": { color: "#60a5fa", bg: "rgba(96,165,250,0.12)",  border: "rgba(96,165,250,0.4)" },
  "1勝級": { color: "#4ade80", bg: "rgba(74,222,128,0.10)",  border: "rgba(74,222,128,0.35)" },
  "未勝利級":{ color: "#94a3b8", bg: "rgba(148,163,184,0.10)", border: "rgba(148,163,184,0.3)" },
};

function raceLevelStyle(label: string): { color: string; bg: string; border: string } {
  for (const key of Object.keys(RACE_LEVEL_CLASS_STYLE)) {
    if (label.endsWith(key)) return RACE_LEVEL_CLASS_STYLE[key];
  }
  return { color: "var(--text-dim)", bg: "transparent", border: "var(--border)" };
}

/** 古馬基準のめぐ指数閾値（backend _MEGU_THRESHOLDS と同期） */
const MEGU_THRESHOLDS: Record<string, [number, string][]> = {
  芝: [
    [115, "G1級"],
    [108, "重賞級"],
    [103, "3勝級"],
    [99, "2勝級"],
    [93, "1勝級"],
  ],
  ダート: [
    [112, "G1級"],
    [106, "重賞級"],
    [101, "3勝級"],
    [97, "2勝級"],
    [88, "1勝級"],
  ],
};

function meguClassLabel(v: number, surface?: string | null): string {
  const thresholds = MEGU_THRESHOLDS[surface === "芝" ? "芝" : "ダート"];
  for (const [bound, cls] of thresholds) {
    if (v >= bound) return cls;
  }
  return "未勝利級";
}

function meguColor(v: number | null, surface?: string | null): { color: string; bg: string } {
  if (v == null) return { color: "var(--text-dim)", bg: "transparent" };
  const s = raceLevelStyle(meguClassLabel(v, surface));
  return { color: s.color, bg: s.bg };
}

function inferRaceLevelLabel(raceName?: string | null, grade?: string | null): string | null {
  const rn = raceName ?? "";
  const gr = grade ?? "";
  if (rn.includes("未勝利") || gr === "未勝利") return "未勝利級";
  if (rn.includes("新馬") || gr === "新馬") return "未勝利級";
  if (rn.includes("(G1)") || gr === "G1") return "G1級";
  if (rn.includes("(G2)") || gr === "G2") return "重賞級";
  if (rn.includes("(G3)") || gr === "G3" || rn.includes("(L)") || gr === "L") return "重賞級";
  if (rn.includes("(OP)") || gr === "OP" || rn.includes("オープン")) return "3勝級";
  if (rn.includes("(3勝)") || rn.includes("３勝") || gr === "3勝") return "3勝級";
  if (rn.includes("(2勝)") || rn.includes("２勝") || gr === "2勝") return "2勝級";
  if (rn.includes("(1勝)") || rn.includes("１勝") || gr === "1勝") return "1勝級";
  return null;
}

function resolveRaceLevel(data: MeguPredicted | null, race: RaceItem): RaceLevel | null {
  const apiLabel = data?.race_level?.label;
  if (apiLabel && apiLabel !== "?") {
    return data!.race_level!;
  }
  const inferred = inferRaceLevelLabel(
    data?.race_info?.race_name ?? race.race_name,
    data?.race_info?.grade ?? race.grade,
  );
  if (!inferred) return null;
  return { label: inferred, field_avg_megu: data?.race_level?.field_avg_megu ?? null };
}

function RaceLevelBadge({ level }: { level?: RaceLevel }) {
  if (!level || !level.label || level.label === "?") return null;
  const s = raceLevelStyle(level.label);
  const tooltip = level.field_avg_megu != null
    ? `フィールド平均実測めぐ: ${level.field_avg_megu.toFixed(1)}`
    : undefined;
  return (
    <span
      title={tooltip}
      style={{
        fontSize: 11, fontWeight: 800, padding: "2px 8px", borderRadius: 4,
        background: s.bg, color: s.color, border: `1px solid ${s.border}`,
        letterSpacing: "0.04em", cursor: tooltip ? "help" : undefined,
        whiteSpace: "nowrap",
      }}
    >
      {level.label}
    </span>
  );
}

function fmtTime(sec: number | null): string {
  if (sec == null) return "—";
  const m = Math.floor(sec / 60);
  const s = (sec % 60).toFixed(1).padStart(4, "0");
  return m > 0 ? `${m}:${s}` : `${s}s`;
}

/** 勝ち馬とのタイム差（秒）。1着または差 < 0.05s は null */
function fmtMarginBehind(horseSec: number | null, winnerSec: number | null): string | null {
  if (horseSec == null || winnerSec == null) return null;
  const d = horseSec - winnerSec;
  if (d < 0.05) return null;
  return `+${d.toFixed(1)}`;
}

function winnerTimeSec(horses: MeguHorse[]): number | null {
  const winner = horses.find(h => h.finish_pos === 1 && h.finish_time_sec != null);
  if (winner?.finish_time_sec != null) return winner.finish_time_sec;
  const times = horses.map(h => h.finish_time_sec).filter((t): t is number => t != null && t > 0);
  return times.length ? Math.min(...times) : null;
}

function isPastOrTodayYmd(ymd: string): boolean {
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  return parseYmd(ymd) <= today;
}

function predictedMegu(horse: MeguHorse): number | null {
  return horse.megu_final;
}

function isRaceFinisher(horse: MeguHorse): boolean {
  return (
    horse.finish_pos != null
    && horse.finish_pos > 0
    && horse.finish_time_sec != null
    && horse.finish_time_sec > 0
  );
}

/** 実測めぐ列に数値または「圏外」が表示されるか */
function isActualMeguDisplayed(horse: MeguHorse): boolean {
  if (horse.actual_megu != null) return true;
  return horse.actual_status === "valid" || horse.actual_status === "out_of_range";
}

function topMeguScore(horse: MeguHorse): number | null {
  return predictedMegu(horse) ?? horse.actual_megu;
}

function fmtMeguMargin(sec: number | null | undefined): string {
  if (sec == null || sec < 0.05) return "—";
  return `+${sec.toFixed(1)}s`;
}

function fmtMeguGap(gap: number | null | undefined): string {
  if (gap == null) return "—";
  const sign = gap > 0 ? "+" : "";
  return `${sign}${gap.toFixed(1)}`;
}

function finishPosStyle(pos: number | null | undefined): { color: string; fontWeight: number } {
  if (pos === 1) return { color: "#fbbf24", fontWeight: 800 };
  if (pos === 2) return { color: "#94a3b8", fontWeight: 800 };
  if (pos === 3) return { color: "#d97706", fontWeight: 800 };
  return { color: "var(--text-dim)", fontWeight: 400 };
}

function CondBadge({ cc }: { cc: ConditionChange }) {
  if (cc.type === "none" || !cc.label) return null;
  const bg =
    cc.type === "both"     ? "rgba(239,68,68,0.16)" :
    cc.type === "surface"  ? "rgba(251,146,60,0.16)" :
                             "rgba(250,204,21,0.14)";
  const color =
    cc.type === "both"     ? "#f87171" :
    cc.type === "surface"  ? "#fb923c" :
                             "#facc15";
  return (
    <span style={{ fontSize: 10, fontWeight: 700, padding: "1px 6px", borderRadius: 3, background: bg, color, whiteSpace: "nowrap", marginLeft: 5 }}>
      ⚠ {cc.label}
    </span>
  );
}

function gradeChip(grade: string | undefined | null) {
  if (!grade) return null;
  const s =
    grade === "G1" ? { bg: "rgba(239,68,68,0.18)", color: "#ef4444" } :
    grade === "G2" ? { bg: "rgba(59,130,246,0.18)", color: "#60a5fa" } :
    grade === "G3" ? { bg: "rgba(34,197,94,0.18)", color: "#22c55e" } : null;
  if (!s) return null;
  return <span style={{ fontSize: 10, fontWeight: 700, padding: "2px 6px", borderRadius: 3, marginLeft: 4, ...s }}>{grade}</span>;
}

/* 枠色（tracking-difficulty と同一）— bracket_number で色分け、表示は馬番のみ */
/* JRA公式枠色: 1白 2黒 3赤 4青 5黄 6緑 7橙 8桃 */
const WAKU_COLORS: [string, string][] = [
  ["#fff",    "#333"], // 1枠: 白
  ["#1a1a1a", "#fff"], // 2枠: 黒
  ["#c0392b", "#fff"], // 3枠: 赤
  ["#1a5fa0", "#fff"], // 4枠: 青
  ["#e8b800", "#333"], // 5枠: 黄
  ["#1e8449", "#fff"], // 6枠: 緑
  ["#e36c00", "#fff"], // 7枠: 橙
  ["#e0479e", "#fff"], // 8枠: 桃（ピンク）
];

function horseNumberBadgeStyle(bracketNumber: number | null | undefined): React.CSSProperties {
  const bn = bracketNumber != null && bracketNumber >= 1 && bracketNumber <= 8
    ? bracketNumber
    : null;
  const [bg, color] = bn ? WAKU_COLORS[(bn - 1) % 8] : ["#64748b", "#fff"];
  return {
    background: bg,
    color,
    width: 28,
    height: 28,
    borderRadius: "50%",
    display: "inline-flex",
    alignItems: "center",
    justifyContent: "center",
    fontSize: 12,
    fontWeight: 800,
    lineHeight: 1,
    border: bg === "#fff" ? "1px solid #cbd5e1" : "none",
    boxShadow: "0 1px 2px rgba(0,0,0,0.18)",
  };
}

function HorseNumberBadge({
  horseNumber,
  bracketNumber,
}: {
  horseNumber: number | null | undefined;
  bracketNumber?: number | null;
}) {
  if (horseNumber == null) return <span style={{ color: "var(--text-dim)" }}>—</span>;
  return (
    <span style={horseNumberBadgeStyle(bracketNumber)}>
      {horseNumber}
    </span>
  );
}

function formatSexAge(horse: MeguHorse): string {
  if (horse.sex_age?.trim()) return horse.sex_age.trim();
  return "—";
}

/* ── ソートヘッダー ── */
function SortTh({
  label, sortKey, current, dir, onSort, style, title,
}: {
  label: string; sortKey: SortKey; current: SortKey; dir: SortDir;
  onSort: (k: SortKey) => void; style?: React.CSSProperties; title?: string;
}) {
  const active = current === sortKey;
  return (
    <th
      title={title}
      onClick={() => onSort(sortKey)}
      style={{
        padding: "8px 10px", fontSize: 11, fontWeight: 600, whiteSpace: "nowrap",
        color: active ? "var(--accent)" : "var(--text-dim)",
        background: "var(--surface2)", borderBottom: "1px solid var(--border)",
        cursor: "pointer", userSelect: "none", textAlign: "center",
        ...style,
      }}
    >
      {label} <span style={{ opacity: active ? 1 : 0.3 }}>{active ? (dir === "asc" ? "▲" : "▼") : "▼"}</span>
    </th>
  );
}

/* ── 馬行 ── */
function HorseRow({
  horse, rank, sortKey, winnerSec, surface,
}: {
  horse: MeguHorse; rank: number; sortKey: SortKey; winnerSec: number | null;
  surface?: string | null;
}) {
  const [open, setOpen] = useState(false);
  const predicted = predictedMegu(horse);
  const predictedC = meguColor(predicted, surface);
  const actualC = meguColor(horse.actual_megu, surface);
  const cc = horse.condition_change;
  const margin = fmtMarginBehind(horse.finish_time_sec, winnerSec);
  const posStyle = finishPosStyle(horse.finish_pos);
  const TDc: React.CSSProperties = {
    padding: "10px 10px", borderBottom: "1px solid rgba(36,48,73,0.4)",
    textAlign: "center", verticalAlign: "middle", fontSize: 13,
  };

  return (
    <>
      <tr
        style={{
          background: rank === 0 ? "rgba(34,197,94,0.04)" : rank === 1 ? "rgba(96,165,250,0.03)" : undefined,
          cursor: horse.history.length > 0 ? "pointer" : undefined,
        }}
        onClick={() => horse.history.length > 0 && setOpen(!open)}
      >
        {/* 馬番（枠色丸アイコン） */}
        <td style={{ ...TDc, background: sortKey === "horse_number" ? "rgba(59,130,246,0.04)" : undefined }}>
          <HorseNumberBadge horseNumber={horse.horse_number} bracketNumber={horse.bracket_number} />
        </td>

        {/* 馬名 */}
        <td style={{ ...TDc, textAlign: "left", fontWeight: 600 }}>
          <a
            href={`/horse/${horse.horse_id}`}
            target="_blank"
            rel="noreferrer"
            onClick={e => e.stopPropagation()}
            style={{ color: "#fff", textDecoration: "none" }}
          >
            {horse.horse_name ?? horse.horse_id}
          </a>
          <CondBadge cc={cc} />
        </td>

        {/* 性齢 */}
        <td style={{ ...TDc, color: "var(--text-dim)", whiteSpace: "nowrap", fontSize: 12 }}>
          {formatSexAge(horse)}
        </td>

        {/* 斤量 */}
        <td style={{ ...TDc, color: "var(--text-dim)", background: sortKey === "jockey_weight" ? "rgba(59,130,246,0.04)" : undefined }}>
          {horse.jockey_weight != null ? `${horse.jockey_weight}kg` : "—"}
        </td>

        {/* 着順 */}
        <td style={{
          ...TDc,
          color: posStyle.color,
          fontWeight: posStyle.fontWeight,
          background: sortKey === "finish_pos" ? "rgba(59,130,246,0.04)" : undefined,
        }}>
          {horse.finish_pos != null && horse.finish_pos > 0 ? horse.finish_pos : "—"}
        </td>

        {/* 走破タイム */}
        <td style={{ ...TDc, fontFamily: "monospace", background: sortKey === "finish_time_sec" ? "rgba(59,130,246,0.04)" : undefined }}>
          {fmtTime(horse.finish_time_sec)}
          {margin && (
            <span style={{ display: "block", fontSize: 10, fontWeight: 400, color: "var(--text-dim)", opacity: 0.65, lineHeight: 1.2, marginTop: 2 }}>
              {margin}
            </span>
          )}
        </td>

        {/* 想定めぐ指数（補正後・今回斤量込み） */}
        <td
          title="ペース・馬場・斤量・レベル補正後。今回 par・斤量で換算した能力推定。1点=0.1秒"
          style={{
          ...TDc, fontWeight: 800, fontSize: 15,
          color: predictedC.color,
          background: sortKey === "megu_final" ? predictedC.bg || "rgba(59,130,246,0.04)" : predictedC.bg,
        }}>
          {predicted != null ? predicted.toFixed(1) : "—"}
          {horse.pred_margin_sec != null && horse.pred_margin_sec >= 0.05 && (
            <span style={{ display: "block", fontSize: 9, fontWeight: 500, color: "var(--text-dim)", lineHeight: 1.2, marginTop: 2 }}>
              想定{fmtMeguMargin(horse.pred_margin_sec)}
            </span>
          )}
        </td>

        {/* 実測めぐ指数（補正後・当日斤量込み） */}
        <td
          title={
            horse.actual_status === "out_of_range"
              ? "2着基準から外れたため非表示（out_of_range）"
              : "レース後: 走破タイムをペース・馬場・斤量・レベル補正。1点=0.1秒"
          }
          style={{
          ...TDc, fontWeight: 700, fontSize: 14,
          color: horse.actual_megu != null ? actualC.color : "var(--text-dim)",
          background: sortKey === "actual_megu" ? actualC.bg || "rgba(59,130,246,0.04)" : (horse.actual_megu != null ? actualC.bg : undefined),
        }}>
          {horse.actual_megu != null ? horse.actual_megu.toFixed(1) : (horse.actual_status === "out_of_range" ? "圏外" : "—")}
          {horse.actual_margin_sec != null && horse.actual_margin_sec >= 0.05 && (
            <span style={{ display: "block", fontSize: 9, fontWeight: 500, color: "var(--text-dim)", lineHeight: 1.2, marginTop: 2 }}>
              実測{fmtMeguMargin(horse.actual_margin_sec)}
            </span>
          )}
        </td>

        {/* 実測−想定 */}
        <td
          title="実測めぐ − 想定めぐ（+は好走、−は凡走）"
          style={{
            ...TDc,
            fontSize: 12,
            fontWeight: 600,
            color: horse.megu_gap != null
              ? (horse.megu_gap >= 5 ? "#4ade80" : horse.megu_gap <= -5 ? "#f87171" : "var(--text-dim)")
              : "var(--text-dim)",
            background: sortKey === "megu_gap" ? "rgba(59,130,246,0.04)" : undefined,
          }}
        >
          {fmtMeguGap(horse.megu_gap)}
        </td>

        {/* 履歴展開 */}
        <td style={{ ...TDc, width: 24, color: "var(--text-dim)", fontSize: 11 }}>
          {horse.history.length > 0 ? (open ? "▲" : "▼") : ""}
        </td>
      </tr>

      {/* 履歴展開行 */}
      {open && (
        <tr>
          <td colSpan={10} style={{ background: "rgba(12,18,32,0.7)", padding: "8px 16px 12px 40px" }}>
            <div style={{ fontSize: 11, color: "var(--text-dim)", marginBottom: 5 }}>直近{horse.history.length}走</div>
            <table style={{ borderCollapse: "collapse", fontSize: 11 }}>
              <tbody>
                {horse.history.map((h) => {
                  const c = meguColor(h.megu_index, h.surface === "芝" ? "芝" : surface);
                  return (
                    <tr key={h.race_id}>
                      <td style={{ padding: "3px 8px", color: "var(--text-dim)" }}>{h.race_date}</td>
                      <td style={{ padding: "3px 8px" }}>{h.venue}</td>
                      <td style={{ padding: "3px 8px", color: "var(--text-dim)" }}>{h.surface}{h.distance}m</td>
                      <td style={{ padding: "3px 8px", color: "var(--text-dim)" }}>{h.finish_pos != null ? `${h.finish_pos}着` : "—"}</td>
                      <td style={{ padding: "3px 10px", fontWeight: 700, color: c.color, background: c.bg, textAlign: "right" }}>
                        {h.megu_index.toFixed(1)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </td>
        </tr>
      )}
    </>
  );
}

/* ── レースカード ── */
function RaceCard({
  race, data, loading, onLoad, defaultExpanded = false,
}: {
  race: RaceItem;
  data: MeguPredicted | null;
  loading: boolean;
  onLoad: (id: string) => void;
  defaultExpanded?: boolean;
}) {
  const [expanded, setExpanded] = useState(defaultExpanded);

  useEffect(() => {
    if (defaultExpanded && !data && !loading) {
      onLoad(race.race_id);
    }
  }, [defaultExpanded, data, loading, onLoad, race.race_id]);
  const [sortKey, setSortKey] = useState<SortKey>("finish_pos");
  const [sortDir, setSortDir] = useState<SortDir>("asc");

  const ri = data?.race_info;
  const horses = data?.horses ?? [];
  const raceLevel = resolveRaceLevel(data, race);
  const surface = race.surface ?? ri?.surface ?? null;
  const winSec = winnerTimeSec(horses);
  const predCount = horses.filter(h => h.megu_final != null).length;
  const actualCount = horses.filter(h => isActualMeguDisplayed(h)).length;
  const finisherCount = data?.result_stats?.finisher_count ?? horses.filter(isRaceFinisher).length;
  const meguCoverageOk = data?.result_stats?.megu_coverage_ok ?? (finisherCount === 0 || finisherCount === actualCount);

  function handleSort(key: SortKey) {
    if (sortKey === key) {
      setSortDir(d => d === "asc" ? "desc" : "asc");
    } else {
      setSortKey(key);
      setSortDir(
        key === "horse_number" || key === "finish_pos" || key === "finish_time_sec" ? "asc" : "desc"
      );
    }
  }

  const sorted = [...horses].sort((a, b) => {
    let va: number | null;
    let vb: number | null;
    if (sortKey === "horse_number") {
      va = a.horse_number ?? null; vb = b.horse_number ?? null;
    } else if (sortKey === "finish_pos") {
      va = a.finish_pos != null && a.finish_pos > 0 ? a.finish_pos : null;
      vb = b.finish_pos != null && b.finish_pos > 0 ? b.finish_pos : null;
    } else if (sortKey === "jockey_weight") {
      va = a.jockey_weight ?? null; vb = b.jockey_weight ?? null;
    } else if (sortKey === "finish_time_sec") {
      va = a.finish_time_sec ?? null; vb = b.finish_time_sec ?? null;
    } else if (sortKey === "megu_final") {
      va = predictedMegu(a); vb = predictedMegu(b);
    } else if (sortKey === "actual_megu") {
      va = a.actual_megu; vb = b.actual_megu;
    } else if (sortKey === "megu_gap") {
      va = a.megu_gap ?? null; vb = b.megu_gap ?? null;
    } else if (sortKey === "pred_margin_sec") {
      va = a.pred_margin_sec ?? null; vb = b.pred_margin_sec ?? null;
    } else {
      va = predictedMegu(a); vb = predictedMegu(b);
    }

    let cmp = 0;
    if (va == null && vb == null) cmp = 0;
    else if (va == null) cmp = 1;
    else if (vb == null) cmp = -1;
    else cmp = sortDir === "asc" ? va - vb : vb - va;

    if (cmp !== 0) return cmp;
    // 同値・着順未確定時は馬番で安定ソート
    return (a.horse_number ?? 999) - (b.horse_number ?? 999);
  });

  const topHorse = horses.length > 0
    ? [...horses].sort((a, b) => {
        const va = topMeguScore(a) ?? -999;
        const vb = topMeguScore(b) ?? -999;
        return vb - va;
      })[0]
    : null;

  const TH: React.CSSProperties = {
    padding: "8px 10px", fontSize: 11, fontWeight: 600, color: "var(--text-dim)",
    background: "var(--surface2)", borderBottom: "1px solid var(--border)", whiteSpace: "nowrap",
    textAlign: "left",
  };

  return (
    <div style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 10, overflow: "hidden", marginBottom: 12 }}>

      {/* ── カードヘッダー ── */}
      <div
        style={{ padding: "14px 18px", cursor: "pointer" }}
        onClick={() => { setExpanded(!expanded); if (!data && !loading) onLoad(race.race_id); }}
      >
        {/* 1行目: ラウンド・レース名・グレード */}
        <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap", marginBottom: 6 }}>
          <span style={{ background: "var(--accent)", color: "#fff", fontWeight: 800, fontSize: 13, padding: "3px 10px", borderRadius: 4, minWidth: 38, textAlign: "center" }}>
            {race.round}R
          </span>
          <span style={{ fontWeight: 700, fontSize: 15, color: "#f0f6fc" }}>
            {ri?.race_name ?? race.race_name ?? race.race_id}
            {gradeChip(ri?.grade ?? race.grade)}
          </span>
          {raceLevel && <RaceLevelBadge level={raceLevel} />}
          {topHorse && topMeguScore(topHorse) != null && (
            <span style={{ fontSize: 11, padding: "2px 8px", borderRadius: 4, background: "rgba(34,197,94,0.10)", color: "#22c55e", marginLeft: 4 }}>
              Top: {topHorse.horse_name ?? "—"}{" "}
              {(topMeguScore(topHorse) ?? 0).toFixed(1)}
            </span>
          )}
          <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 8 }}>
            {data && horses.length > 0 && (
              <span
                title={`入着${finisherCount}頭 / 実測めぐ表示${actualCount}頭（圏外含む）`}
                style={{
                  fontSize: 10, fontWeight: 600, padding: "2px 7px", borderRadius: 3,
                  background: meguCoverageOk ? "rgba(34,197,94,0.12)" : "rgba(251,146,60,0.12)",
                  color: meguCoverageOk ? "var(--ok)" : "#fb923c",
                }}
              >
                入着{finisherCount} ／ 実測{actualCount}{predCount > 0 ? ` 想定${predCount}` : ""}
              </span>
            )}
            {loading && !data && (
              <span style={{ fontSize: 10, color: "var(--text-dim)" }}>取得中…</span>
            )}
            <Link href={`/race/${race.race_id}`} target="_blank" rel="noreferrer" onClick={e => e.stopPropagation()} style={{ fontSize: 11, color: "var(--accent)", textDecoration: "none", border: "1px solid rgba(59,130,246,0.3)", padding: "3px 8px", borderRadius: 4 }}>
              詳細 →
            </Link>
            <span style={{ fontSize: 14, color: "var(--text-dim)", userSelect: "none" }}>{expanded ? "▲" : "▼"}</span>
          </div>
        </div>

        {/* 2行目: レース情報ピル（race-listのDB情報を優先、fallbackはmegu API） */}
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center" }}>
          {(ri?.venue ?? race.venue) && (
            <span style={{ fontSize: 11, padding: "2px 8px", borderRadius: 12, background: "rgba(36,48,73,0.6)", color: "var(--text-dim)" }}>
              📍 {ri?.venue ?? race.venue}
            </span>
          )}
          {(race.surface || ri?.surface || race.distance || ri?.distance) && (
            <span style={{ fontSize: 11, padding: "2px 8px", borderRadius: 12, background: "rgba(36,48,73,0.6)", color: "var(--text-dim)" }}>
              {race.surface ?? ri?.surface}{race.distance ?? ri?.distance}m
            </span>
          )}
          {(race.track_condition ?? ri?.track_condition) && (
            <span style={{ fontSize: 11, padding: "2px 8px", borderRadius: 12, background: "rgba(36,48,73,0.6)", color: "var(--text-dim)" }}>
              馬場: {race.track_condition ?? ri?.track_condition}
            </span>
          )}
          {race.start_time && (
            <span style={{ fontSize: 11, padding: "2px 8px", borderRadius: 12, background: "rgba(36,48,73,0.6)", color: "var(--text-dim)" }}>
              🕐 {race.start_time}
            </span>
          )}
          {race.entries_count != null && (
            <span style={{ fontSize: 11, padding: "2px 8px", borderRadius: 12, background: "rgba(36,48,73,0.6)", color: "var(--text-dim)" }}>
              {race.entries_count}頭立
            </span>
          )}
        </div>
      </div>

      {/* ── 展開コンテンツ ── */}
      {expanded && (
        <div style={{ borderTop: "1px solid var(--border)" }}>
          {loading && (
            <div style={{ padding: "20px 0", textAlign: "center", color: "var(--text-dim)", fontSize: 13 }}>
              <div style={{ width: 24, height: 24, border: "2px solid var(--border)", borderTopColor: "var(--accent)", borderRadius: "50%", animation: "spin 1s linear infinite", margin: "0 auto 8px" }} />
              データを取得中…
            </div>
          )}
          {!loading && !data && (
            <div style={{ padding: "14px 18px" }}>
              <button
                style={{ background: "#1e3a5f", color: "#60a5fa", border: "1px solid rgba(59,130,246,0.3)", padding: "6px 14px", borderRadius: 6, fontSize: 12, fontWeight: 600, cursor: "pointer" }}
                onClick={() => onLoad(race.race_id)}
              >
                📊 めぐ指数を取得
              </button>
            </div>
          )}
          {data && horses.length === 0 && (
            <p style={{ padding: "14px 18px", fontSize: 12, color: "var(--text-dim)" }}>出走馬のデータがありません。</p>
          )}
          {data && horses.length > 0 && !meguCoverageOk && (
            <p style={{ padding: "8px 18px", fontSize: 11, color: "#fb923c", background: "rgba(251,146,60,0.08)" }}>
              入着{finisherCount}頭に対し実測めぐ表示が{actualCount}頭です（未計算またはデータ欠損）。
            </p>
          )}
          {data && horses.length > 0 && predCount === 0 && actualCount === 0 && (
            <p style={{ padding: "10px 18px", fontSize: 11, color: "var(--text-dim)", background: "rgba(107,125,149,0.06)" }}>
              このレースは実測めぐ指数が未計算です（surface/distance 欠損・パータイム不足など）。
            </p>
          )}
          {data && horses.length > 0 && predCount === 0 && actualCount > 0 && (
            <p style={{ padding: "8px 18px", fontSize: 11, color: "var(--text-dim)", background: "rgba(59,130,246,0.06)" }}>
              実測めぐ {actualCount}頭（想定めぐは履歴不足等で未表示の場合があります）。
            </p>
          )}
          {data && horses.length > 0 && (
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                <thead>
                  <tr>
                    <SortTh label="馬番" sortKey="horse_number" current={sortKey} dir={sortDir} onSort={handleSort} style={{ width: 52 }} />
                    <th style={{ ...TH, width: 160 }}>馬名</th>
                    <th style={{ ...TH, width: 52, textAlign: "center" }}>性齢</th>
                    <SortTh label="斤量" sortKey="jockey_weight" current={sortKey} dir={sortDir} onSort={handleSort} style={{ width: 68 }} />
                    <SortTh label="着順" sortKey="finish_pos" current={sortKey} dir={sortDir} onSort={handleSort} style={{ width: 44 }} />
                    <SortTh label="走破タイム" sortKey="finish_time_sec" current={sortKey} dir={sortDir} onSort={handleSort} style={{ width: 100 }} />
                    <SortTh label="想定めぐ" sortKey="megu_final" current={sortKey} dir={sortDir} onSort={handleSort} style={{ width: 96 }} title="補正後・今回斤量込み" />
                    <SortTh label="実測めぐ" sortKey="actual_megu" current={sortKey} dir={sortDir} onSort={handleSort} style={{ width: 96 }} title="補正後・当日斤量込み" />
                    <SortTh label="差" sortKey="megu_gap" current={sortKey} dir={sortDir} onSort={handleSort} style={{ width: 52 }} title="実測−想定" />
                    <th style={{ ...TH, width: 24 }} />
                  </tr>
                </thead>
                <tbody>
                  {sorted.map((horse, i) => (
                    <HorseRow key={horse.horse_id} horse={horse} rank={i} sortKey={sortKey} winnerSec={winSec} surface={surface} />
                  ))}
                </tbody>
              </table>
              <p style={{ fontSize: 10, color: "var(--text-dim)", padding: "5px 14px 10px", lineHeight: 1.5 }}>
                {data.index_note ?? "想定・実測ともペース・馬場・斤量・レベル補正後（1点=0.1秒）。"}
                想定着差は想定めぐ1位との秒差。差列は実測−想定（+好走／−凡走）。
                {horses.some(h => h.condition_change.type !== "none") && " ⚠ = 芝↔ダート or 距離±600m超の条件転換。"}
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/* ── メインページ ── */
export default function MeguIndexPage() {
  const [dates, setDates] = useState<string[]>([]);
  const [selectedDate, setSelectedDate] = useState("");
  const [races, setRaces] = useState<RaceItem[]>([]);
  const [meguMap, setMeguMap] = useState<Record<string, MeguPredicted | null>>({});
  const [loadingMegu, setLoadingMegu] = useState<Record<string, boolean>>({});
  const [loadingDates, setLoadingDates] = useState(true);
  const [datesError, setDatesError] = useState("");
  const [loadingRaces, setLoadingRaces] = useState(false);
  const [racesError, setRacesError] = useState("");

  const selectedDateRef = useRef("");
  const meguLoadGenRef = useRef(0);
  const meguLoadedRef = useRef<Set<string>>(new Set());

  useEffect(() => {
    selectedDateRef.current = selectedDate;
  }, [selectedDate]);

  useEffect(() => {
    if (USE_MOCK) {
      const list = getMockRaceDates();
      setDates(list);
      if (list.length) setSelectedDate(pickDefaultMeguDate(list));
      setLoadingDates(false);
      return;
    }
    (async () => {
      try {
        const [resMegu, resFast, resFull] = await Promise.all([
          fetch("/api/megu-index-dates", { cache: "no-store" }),
          fetch("/api/scrape-dates?picker_past_days=120", { cache: "no-store" }),
          fetch("/api/scrape-dates?filter=meeting", { cache: "no-store" }),
        ]);
        const dMegu = resMegu.ok ? await resMegu.json() : {};
        const dFast = resFast.ok ? await resFast.json() : {};
        const dFull = resFull.ok ? await resFull.json() : {};
        const megu: string[] = Array.isArray(dMegu?.dates) ? dMegu.dates : [];
        const fast: string[] = Array.isArray(dFast?.dates) ? dFast.dates : [];
        const full: string[] = Array.isArray(dFull?.dates) ? dFull.dates : [];
        const merged = [...new Set([...megu, ...full, ...fast])]
          .filter(d => /^\d{8}$/.test(d) && isPastOrTodayYmd(d))
          .sort()
          .reverse();
        const list = merged.length
          ? merged
          : [...new Set([...megu, ...full, ...fast])].filter(d => /^\d{8}$/.test(d)).sort().reverse();
        if (!list.length) throw new Error("開催日データが空です（API・FastAPI を確認してください）");
        setDates(list);
        setSelectedDate(pickDefaultMeguDate(list, megu.filter(isPastOrTodayYmd)));
      } catch (e: unknown) {
        setDatesError(e instanceof Error ? e.message : String(e));
      } finally {
        setLoadingDates(false);
      }
    })();
  }, []);

  const loadRaces = useCallback(async (date: string) => {
    if (!date) return;
    const loadGen = ++meguLoadGenRef.current;
    setLoadingRaces(true);
    setRacesError("");
    try {
      if (USE_MOCK) {
        const list = MOCK_WEEKLY_RACES.map(r => ({
          race_id: r.race_id,
          race_name: r.race_name,
          venue: r.venue,
          round: r.round,
          distance: r.distance,
          surface: r.surface,
          grade: r.grade,
        }));
        if (loadGen !== meguLoadGenRef.current || selectedDateRef.current !== date) return;
        meguLoadedRef.current = new Set();
        setMeguMap({});
        setLoadingMegu({});
        setRaces(list);
      } else {
        const res = await fetch(`/api/race-list/${date}`, { cache: "no-store" });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const d = await res.json();
        const list = Array.isArray(d.races ?? d) ? (d.races ?? d) : [];
        if (loadGen !== meguLoadGenRef.current || selectedDateRef.current !== date) return;
        meguLoadedRef.current = new Set();
        setMeguMap({});
        setLoadingMegu({});
        setRaces(list);
      }
    } catch (e: unknown) {
      if (loadGen !== meguLoadGenRef.current || selectedDateRef.current !== date) return;
      setRacesError(e instanceof Error ? e.message : String(e));
      setRaces([]);
    } finally {
      if (loadGen === meguLoadGenRef.current && selectedDateRef.current === date) {
        setLoadingRaces(false);
      }
    }
  }, []);

  useEffect(() => {
    if (selectedDate) loadRaces(selectedDate);
  }, [selectedDate, loadRaces]);

  const loadMegu = useCallback(async (raceId: string, date: string, loadGen: number) => {
    if (meguLoadedRef.current.has(raceId)) return;
    setLoadingMegu(prev => ({ ...prev, [raceId]: true }));
    try {
      if (USE_MOCK) {
        await new Promise(r => setTimeout(r, 300));
        if (loadGen !== meguLoadGenRef.current || selectedDateRef.current !== date) return;
        const d = getMockMeguPredicted(raceId) as MeguPredicted;
        meguLoadedRef.current.add(raceId);
        setMeguMap(prev => ({ ...prev, [raceId]: d }));
        return;
      }
      const res = await fetch(`/api/v1/races/${raceId}/megu-index-predicted`, { cache: "no-store" });
      if (loadGen !== meguLoadGenRef.current || selectedDateRef.current !== date) return;
      const d: MeguPredicted = res.ok ? await res.json() : {
        race_id: raceId,
        race_info: { race_name: null, venue: null, surface: null, distance: null, dist_band: null, track_condition: null, grade: null, race_date: null },
        model_version: "",
        horses: [],
      };
      meguLoadedRef.current.add(raceId);
      setMeguMap(prev => ({ ...prev, [raceId]: d }));
    } catch {
      if (loadGen !== meguLoadGenRef.current || selectedDateRef.current !== date) return;
      // 失敗時は null にせず再試行可能に（loaded セットに入れない）
    } finally {
      if (loadGen === meguLoadGenRef.current && selectedDateRef.current === date) {
        setLoadingMegu(prev => ({ ...prev, [raceId]: false }));
      }
    }
  }, []);

  useEffect(() => {
    if (!races.length || loadingRaces || !selectedDate) return;
    const loadGen = meguLoadGenRef.current;
    const date = selectedDate;
    let cancelled = false;
    const pending = races.map(r => r.race_id).filter(id => !meguLoadedRef.current.has(id));
    const concurrency = 6;

    async function worker() {
      while (!cancelled) {
        const raceId = pending.shift();
        if (!raceId) break;
        if (meguLoadedRef.current.has(raceId)) continue;
        await loadMegu(raceId, date, loadGen);
      }
    }

    void Promise.all(Array.from({ length: Math.min(concurrency, pending.length || 1) }, worker));
    return () => { cancelled = true; };
  }, [races, loadingRaces, selectedDate, loadMegu]);

  async function loadAll() {
    const loadGen = meguLoadGenRef.current;
    const date = selectedDate;
    for (const race of races) {
      if (meguLoadedRef.current.has(race.race_id)) continue;
      await loadMegu(race.race_id, date, loadGen);
    }
  }

  const handleLoadMegu = useCallback((raceId: string) => {
    if (!selectedDate) return;
    void loadMegu(raceId, selectedDate, meguLoadGenRef.current);
  }, [selectedDate, loadMegu]);

  const loadedCount = races.filter(r => meguMap[r.race_id] != null).length;
  const dataCount = Object.values(meguMap).filter(m => (m?.horses?.length ?? 0) > 0).length;
  const coverageOkCount = Object.values(meguMap).filter(m =>
    m?.result_stats?.megu_coverage_ok ?? (
      (m?.horses ?? []).filter(isRaceFinisher).length === 0
      || (m?.horses ?? []).filter(isActualMeguDisplayed).length
        === (m?.horses ?? []).filter(isRaceFinisher).length
    )
  ).length;
  const changedCount = Object.values(meguMap).flatMap(m => m?.horses ?? []).filter(h => h.condition_change.type !== "none").length;

  const SEL: React.CSSProperties = {
    background: "var(--surface2)", border: "1px solid var(--border)", color: "var(--text)",
    padding: "7px 12px", borderRadius: 6, fontSize: 13,
  };

  return (
    <div style={{ minHeight: "100vh", background: "var(--bg)", color: "var(--text)" }}>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>

      {/* ページヘッダー */}
      <div style={{ background: "var(--surface)", borderBottom: "1px solid var(--border)", padding: "18px 24px" }}>
        <div style={{ maxWidth: 1100, margin: "0 auto", display: "flex", alignItems: "center", gap: 12 }}>
          <Link href="/" style={{ fontSize: 12, color: "var(--text-dim)", textDecoration: "none" }}>← ホーム</Link>
          <span style={{ color: "var(--border)" }}>/</span>
          <h1 style={{ fontSize: 18, fontWeight: 700, color: "#f0f6fc", margin: 0 }}>📊 今週のめぐ指数</h1>
          <p style={{ fontSize: 12, color: "var(--text-dim)", margin: 0 }}>
            補正後指数（1点=0.1秒）で着差想定。想定・実測とも斤量込み
          </p>
        </div>
      </div>

      <div style={{ maxWidth: 1100, margin: "12px auto 0", padding: "0 24px" }}>
        <p style={{ fontSize: 11, color: "var(--text-dim)", margin: 0, lineHeight: 1.5, padding: "8px 12px", background: "rgba(59,130,246,0.08)", borderRadius: 6, border: "1px solid rgba(59,130,246,0.2)" }}>
          想定めぐ＝過去走の補正済み能力を今回条件（par・斤量）に換算。
          実測めぐ＝当日走破を同じ補正体系で評価。
          指数差10点≈1秒。想定着差は想定1位との秒差です。
        </p>
      </div>

      {/* コントロールバー */}
      <div style={{ background: "var(--surface)", borderBottom: "1px solid var(--border)", padding: "12px 24px" }}>
        <div style={{ maxWidth: 1100, margin: "0 auto", display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
          <label style={{ fontSize: 12, color: "var(--text-dim)" }}>開催日:</label>
          <select style={SEL} value={selectedDate} onChange={e => setSelectedDate(e.target.value)} disabled={loadingDates}>
            {loadingDates ? <option>読み込み中…</option> :
              dates.length === 0 ? <option>データなし</option> :
              dates.map(d => {
                const pd = parseYmd(d);
                const dow = pd.getDay();
                const label = dow === 6 ? `${d} (土)` : dow === 0 ? `${d} (日)` : d;
                return <option key={d} value={d}>{label}</option>;
              })}
          </select>
          {selectedDate && !loadingDates && (
            <span style={{ fontSize: 11, color: "var(--text-dim)" }}>
              ※ めぐ指数は出走歴のある馬のみ算出
            </span>
          )}

          {datesError && (
            <span style={{ fontSize: 11, color: "var(--err)" }}>⚠ {datesError}</span>
          )}

          {races.length > 0 && (
            <button style={{ background: "#1e3a5f", color: "#60a5fa", border: "1px solid rgba(59,130,246,0.3)", padding: "7px 14px", borderRadius: 6, fontSize: 12, fontWeight: 600, cursor: "pointer" }} onClick={loadAll}>
              📊 全{races.length}レースを取得
            </button>
          )}

          {loadedCount > 0 && (
            <div style={{ display: "flex", gap: 8, marginLeft: 4 }}>
              <span style={{ fontSize: 11, padding: "3px 8px", borderRadius: 4, background: "rgba(107,125,149,0.12)", color: "var(--text-dim)" }}>
                {loadedCount}/{races.length}R 取得済
              </span>
              {dataCount > 0 && (
                <span style={{ fontSize: 11, padding: "3px 8px", borderRadius: 4, background: "rgba(34,197,94,0.10)", color: "var(--ok)" }}>
                  データあり: {dataCount}R
                </span>
              )}
              {dataCount > 0 && coverageOkCount < dataCount && (
                <span style={{ fontSize: 11, padding: "3px 8px", borderRadius: 4, background: "rgba(251,146,60,0.12)", color: "#fb923c" }}>
                  実測不足: {dataCount - coverageOkCount}R
                </span>
              )}
              {changedCount > 0 && (
                <span style={{ fontSize: 11, padding: "3px 8px", borderRadius: 4, background: "rgba(251,146,60,0.12)", color: "#fb923c" }}>
                  ⚠ 条件代わり: {changedCount}頭
                </span>
              )}
            </div>
          )}
        </div>
      </div>

      {/* メインコンテンツ */}
      <div style={{ maxWidth: 1100, margin: "0 auto", padding: "20px 24px" }}>

        {/* 凡例 */}
        <div style={{ marginBottom: 16, padding: "8px 14px", background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 11, color: "var(--text-dim)", display: "flex", gap: 14, flexWrap: "wrap", alignItems: "center" }}>
          <span>指数色（クラス基準・古馬閾値）:</span>
          {Object.entries(RACE_LEVEL_CLASS_STYLE).map(([label, s]) => (
            <span key={label}>
              <strong style={{ color: s.color, background: s.bg, padding: "1px 6px", borderRadius: 3 }}>{label}</strong>
            </span>
          ))}
          <span style={{ borderLeft: "1px solid var(--border)", paddingLeft: 10 }}>
            ⚠ <strong style={{ color: "#f87171" }}>赤</strong>=両方変更
            <strong style={{ color: "#fb923c", marginLeft: 6 }}>橙</strong>=芝↔ダート
            <strong style={{ color: "#facc15", marginLeft: 6 }}>黄</strong>=距離±600m超
          </span>
        </div>

        {loadingRaces && (
          <div style={{ textAlign: "center", padding: "60px 0", color: "var(--text-dim)" }}>
            <div style={{ width: 36, height: 36, border: "3px solid var(--border)", borderTopColor: "var(--accent)", borderRadius: "50%", animation: "spin 1s linear infinite", margin: "0 auto 12px" }} />
            <p style={{ fontSize: 14 }}>レース一覧を読み込み中…</p>
          </div>
        )}

        {racesError && (
          <div style={{ background: "rgba(239,68,68,0.1)", border: "1px solid rgba(239,68,68,0.3)", borderRadius: 8, padding: "14px 18px", color: "var(--err)", fontSize: 13, marginBottom: 16 }}>
            ⚠️ {racesError}
          </div>
        )}

        {!loadingRaces && races.length === 0 && !racesError && selectedDate && (
          <div style={{ textAlign: "center", padding: "60px 0", color: "var(--text-dim)" }}>
            <div style={{ fontSize: 40, marginBottom: 12 }}>📭</div>
            <p>この日程のレース情報がありません</p>
          </div>
        )}

        {!loadingRaces && races.length > 0 && (
          <>
            <div style={{ marginBottom: 12, fontSize: 13, color: "var(--text-dim)" }}>
              {selectedDate} — {races.length}レース
            </div>
            {races.map((race, idx) => (
              <RaceCard
                key={race.race_id}
                race={race}
                data={meguMap[race.race_id] ?? null}
                loading={loadingMegu[race.race_id] ?? false}
                onLoad={handleLoadMegu}
                defaultExpanded={idx === 0}
              />
            ))}
          </>
        )}
      </div>
    </div>
  );
}
