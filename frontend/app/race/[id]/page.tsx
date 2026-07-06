"use client";

import { useEffect, useState, useCallback } from "react";
import { useParams } from "next/navigation";

/* ── 型 ── */
type RaceEntry = {
  horse_number?: number | string;
  bracket_number?: number | string;
  horse_name?: string;
  horse_id?: string;
  sex_age?: string;
  sex?: string;
  age?: string;
  jockey_name?: string;
  jockey?: string;
  trainer_name?: string;
  trainer?: string;
  jockey_weight?: number | string;
  weight?: number | string;
  finish_position?: number | string;
  position?: number | string;
  time?: string;
  passing_order?: string;
  last3f?: number | string;
  win_odds?: number;
  [k: string]: unknown;
};

type RaceOddsEntry = { horse_number?: number | string; odds?: number };
type RaceData = {
  race_id?: string;
  race_name?: string;
  venue?: string;
  round?: number | string;
  date?: string;
  surface?: string;
  distance?: number | string;
  direction?: string;
  weather?: string;
  track_condition?: string;
  start_time?: string;
  grade?: string;
  race_shutuba?: { entries?: RaceEntry[] };
  race_result?: { entries?: RaceEntry[]; payoffs?: unknown[] };
  race_odds?: { win?: RaceOddsEntry[]; entries?: RaceOddsEntry[] };
  [k: string]: unknown;
};

type PredEntry = {
  horse_number?: number;
  horse_name?: string;
  horse_id?: string;
  mark_type?: string;
  recommendation?: string;
  pred_rank?: number;
  composite_rank?: number;
  pred_score?: number | null;
  composite_score?: number | null;
  win_prob?: number | null;
  top2_prob?: number | null;
  top3_prob?: number | null;
  ev_win?: number | null;
  ev_place?: number | null;
  expected_value?: number | null;
  buy_tier?: string;
  win_odds?: number | null;
  place_odds_min?: number | null;
  place_odds_max?: number | null;
  [k: string]: unknown;
};

type PredData = {
  status?: string;
  has_prediction?: boolean;
  model_type?: string;
  model_description?: string;
  total_horses?: number;
  elapsed_sec?: number;
  predictions?: PredEntry[];
  feature_highlights?: { label?: string; direction?: string; magnitude?: string }[];
  bet_suggestion?: unknown;
  [k: string]: unknown;
};

/* ── ユーティリティ ── */
function pick(e: RaceEntry, ...keys: string[]): string {
  for (const k of keys) {
    const v = (e as Record<string, unknown>)[k];
    if (v !== undefined && v !== null && v !== "") return String(v);
  }
  return "";
}

function fmtOdds(v: number | null | undefined): { text: string; cls: string } {
  if (v == null) return { text: "—", cls: "" };
  const n = Number(v);
  if (n < 5) return { text: n.toFixed(1), cls: "text-red-400" };
  if (n < 15) return { text: n.toFixed(1), cls: "text-yellow-400" };
  return { text: n.toFixed(1), cls: "text-gray-400" };
}

const MARK_META: Record<string, { sym: string; color: string; bg: string }> = {
  honmei:   { sym: "◎", color: "#ef4444", bg: "rgba(239,68,68,0.12)" },
  pair:     { sym: "○", color: "#3b82f6", bg: "rgba(59,130,246,0.12)" },
  anchor:   { sym: "✓", color: "#22c55e", bg: "rgba(34,197,94,0.12)" },
  show_val: { sym: "▲", color: "#f59e0b", bg: "rgba(245,158,11,0.12)" },
  star:     { sym: "★", color: "#a78bfa", bg: "rgba(167,139,250,0.12)" },
  none:     { sym: "—", color: "var(--text-dim)", bg: "transparent" },
};

function markMeta(p: PredEntry) {
  if (p.mark_type && MARK_META[p.mark_type]) return MARK_META[p.mark_type];
  const r = String(p.recommendation ?? "");
  if (r.includes("◎") || r.includes("1着")) return MARK_META.honmei;
  if (r.includes("○") || r.includes("2連")) return MARK_META.pair;
  return MARK_META.none;
}

function gradeStyle(grade: string | undefined): React.CSSProperties {
  if (!grade) return {};
  if (grade === "G1") return { background: "rgba(239,68,68,0.2)", color: "#ef4444" };
  if (grade === "G2") return { background: "rgba(59,130,246,0.2)", color: "#60a5fa" };
  if (grade === "G3") return { background: "rgba(34,197,94,0.2)", color: "#22c55e" };
  return { background: "rgba(245,158,11,0.2)", color: "var(--warn)" };
}

function evColor(v: number | null | undefined): string {
  if (v == null) return "var(--text-dim)";
  if (v >= 1.3) return "var(--ok)";
  if (v >= 1.0) return "var(--accent)";
  if (v >= 0.8) return "var(--text-dim)";
  return "var(--err)";
}

/* ── タブ ── */
type Tab = "shutuba" | "result" | "predict" | "horses";

const TABS: { id: Tab; label: string }[] = [
  { id: "shutuba", label: "出馬表" },
  { id: "result", label: "レース結果" },
  { id: "predict", label: "🤖 AI予測" },
  { id: "horses", label: "出走馬詳細" },
];

const TH: React.CSSProperties = {
  padding: "10px 12px", fontSize: 12, fontWeight: 600, color: "var(--text-dim)",
  textAlign: "center", borderBottom: "1px solid var(--border)", whiteSpace: "nowrap",
};
const TD: React.CSSProperties = {
  padding: "10px 12px", borderBottom: "1px solid rgba(36,48,73,0.5)",
  textAlign: "center", verticalAlign: "middle",
};

/* ── めぐ指数（モック） ── */
/** 時刻文字列 "1:23.4" や "83.4" を秒に変換 */
function parseTimeSec(t: string | undefined | null): number | null {
  if (!t) return null;
  const s = String(t).trim();
  const m = s.match(/^(\d+):(\d+(?:\.\d+)?)$/);
  if (m) return parseInt(m[1]) * 60 + parseFloat(m[2]);
  const n = parseFloat(s);
  return isNaN(n) ? null : n;
}

/**
 * めぐ指数（モック）
 * 走破タイムとレース内最速タイムの差から暫定的な指数を算出。
 * ※ 正式なロジックは後日置き換え予定
 */
function calcMeguIndex(timeSec: number | null, bestSec: number | null, fieldSize: number): number | null {
  if (timeSec == null || bestSec == null || timeSec <= 0) return null;
  const diff = timeSec - bestSec; // 秒差（0=1着タイム）
  // ベース: 勝ちタイム=100, 頭数・距離にかかわらず1秒差≒-4pt（暫定）
  const raw = 100 - diff * 4.0;
  // フィールドサイズが大きいほど上位の評価を引き上げる（暫定補正）
  const fieldBonus = diff < 0.3 ? (fieldSize - 8) * 0.3 : 0;
  return Math.max(50, Math.min(120, Math.round((raw + fieldBonus) * 10) / 10));
}

function meguColor(v: number | null): { color: string; bg: string } {
  if (v == null) return { color: "var(--text-dim)", bg: "transparent" };
  if (v >= 105) return { color: "#22c55e", bg: "rgba(34,197,94,0.10)" };
  if (v >= 98)  return { color: "#4ade80", bg: "rgba(74,222,128,0.07)" };
  if (v >= 92)  return { color: "#60a5fa", bg: "rgba(96,165,250,0.07)" };
  if (v >= 82)  return { color: "var(--text-dim)", bg: "transparent" };
  return { color: "#f87171", bg: "rgba(239,68,68,0.07)" };
}

/* ── コンポーネント ── */
export default function RaceDetailPage() {
  const { id: raceId } = useParams<{ id: string }>();
  const [tab, setTab] = useState<Tab>("shutuba");
  const [raceData, setRaceData] = useState<RaceData | null>(null);
  const [predData, setPredData] = useState<PredData | null>(null);
  const [tdData, setTdData] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState("");
  const [predicting, setPredicting] = useState(false);

  const loadRace = useCallback(async () => {
    if (!raceId) return;
    setLoading(true);
    setLoadError("");
    try {
      const [rRes, tdRes] = await Promise.allSettled([
        fetch(`/api/race/${raceId}`),
        fetch(`/api/race/${raceId}/tracking-difficulty`),
      ]);
      if (rRes.status !== "fulfilled" || !rRes.value.ok) throw new Error("レースデータの取得に失敗しました");
      const rd: RaceData = await rRes.value.json();
      setRaceData(rd);
      if (tdRes.status === "fulfilled" && tdRes.value.ok) {
        const td = await tdRes.value.json();
        if (!td.error) setTdData(td);
      }
    } catch (e: unknown) {
      setLoadError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [raceId]);

  const loadPrediction = useCallback(async () => {
    if (!raceId) return;
    try {
      const res = await fetch(`/api/race/${raceId}/predictions`);
      if (res.ok) setPredData(await res.json());
    } catch { /* ignore */ }
  }, [raceId]);

  useEffect(() => {
    loadRace();
  }, [loadRace]);

  useEffect(() => {
    if (raceData) loadPrediction();
  }, [raceData, loadPrediction]);

  async function runPrediction() {
    if (!raceId) return;
    setPredicting(true);
    try {
      const res = await fetch(`/api/race/${raceId}/predict`, { method: "POST" });
      const data = await res.json();
      if (data.status === "success" || data.predictions?.length) setPredData(data);
    } catch { /* ignore */ } finally {
      setPredicting(false);
    }
  }

  /* ── ローディング ── */
  if (loading) {
    return (
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", minHeight: "60vh", flexDirection: "column", gap: 12 }}>
        <div style={{ width: 40, height: 40, border: "3px solid var(--border)", borderTopColor: "var(--accent)", borderRadius: "50%", animation: "spin 1s linear infinite" }} />
        <p style={{ color: "var(--text-dim)", fontSize: 14 }}>レースデータを読み込み中…</p>
        <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
      </div>
    );
  }

  if (loadError) {
  return (
      <div style={{ padding: 24, textAlign: "center" }}>
        <div style={{ fontSize: 32, marginBottom: 12 }}>⚠️</div>
        <p style={{ color: "var(--err)", marginBottom: 12 }}>{loadError}</p>
        <a href="/monitor" style={{ color: "var(--accent)" }}>← モニターへ</a>
      </div>
    );
  }

  const rd = raceData;

  /* ── レースヘッダー ── */
  const Header = rd ? (
    <div style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 12, padding: "24px 32px", marginBottom: 24 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 12 }}>
        <span style={{ background: "var(--accent)", color: "#fff", fontWeight: 800, fontSize: 16, padding: "6px 14px", borderRadius: 6, minWidth: 50, textAlign: "center" }}>
          {rd.venue} {rd.round}R
        </span>
        {rd.grade && rd.grade !== "未勝利" && rd.grade !== "新馬" && (
          <span style={{ fontWeight: 700, fontSize: 13, padding: "4px 12px", borderRadius: 4, ...gradeStyle(rd.grade) }}>{rd.grade}</span>
        )}
        <a
          href={`https://race.netkeiba.com/race/shutuba.html?race_id=${raceId}`}
          target="_blank" rel="noopener noreferrer"
          style={{ fontSize: 12, color: "var(--accent)", textDecoration: "none", marginLeft: "auto" }}
        >
          🔗 netkeiba
        </a>
      </div>
      <h1 style={{ fontSize: 26, fontWeight: 700, color: "#fff", marginBottom: 8 }}>{rd.race_name ?? raceId}</h1>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 12, fontSize: 13, color: "var(--text-dim)" }}>
        {[rd.date, rd.surface && `馬場: ${rd.surface}`, rd.distance && `${rd.distance}m`, rd.direction, rd.weather && `☁ ${rd.weather}`, rd.track_condition && `馬場: ${rd.track_condition}`, rd.start_time && `🕐 ${rd.start_time}`]
          .filter(Boolean)
          .map((t, i) => (
            <span key={i} style={{ background: "var(--surface2)", border: "1px solid var(--border)", borderRadius: 4, padding: "3px 10px", fontSize: 12 }}>{t}</span>
        ))}
      </div>
    </div>
  ) : null;

  /* ── 出馬表 ── */
  const shutubaEntries = [...(rd?.race_shutuba?.entries ?? [])].sort((a, b) => (Number(a.horse_number) || 99) - (Number(b.horse_number) || 99));
  const oddsMap: Record<string, number> = {};
  (rd?.race_odds?.win ?? rd?.race_odds?.entries ?? []).forEach((o) => {
    if (o.horse_number != null && o.odds != null) oddsMap[String(o.horse_number)] = o.odds;
  });

  const ShutubaTbl = shutubaEntries.length === 0 ? (
    <div style={{ textAlign: "center", padding: 40, color: "var(--text-dim)" }}>📋 出馬表データがありません</div>
  ) : (
    <div style={{ overflowX: "auto" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 10, overflow: "hidden", fontSize: 13 }}>
        <thead style={{ background: "var(--surface2)" }}>
          <tr>{["枠", "馬番", "馬名", "性齢", "斤量", "騎手", "調教師", "単勝"].map(h => <th key={h} style={TH}>{h}</th>)}</tr>
        </thead>
        <tbody>
          {shutubaEntries.map((e, i) => {
            const hn = String(e.horse_number ?? e.number ?? i + 1);
            const oddVal = e.win_odds ?? oddsMap[hn];
            const { text: oddsText, cls: oddsColorClass } = fmtOdds(oddVal as number | undefined);
            return (
              <tr key={hn} style={{ transition: "background 0.12s" }}>
                <td style={TD}>{e.bracket_number ?? "—"}</td>
                <td style={{ ...TD, fontWeight: 800, color: "#7dd3fc" }}>{hn}</td>
                <td style={{ ...TD, fontWeight: 600, color: "#fff", textAlign: "left" }}>
                  {e.horse_id ? <a href={`#horse-${e.horse_id}`} style={{ color: "#fff" }}>{e.horse_name ?? "—"}</a> : (e.horse_name ?? "—")}
                </td>
                <td style={{ ...TD, fontSize: 12, color: "var(--text-dim)" }}>{pick(e, "sex_age") || `${pick(e, "sex")}${pick(e, "age")}` || "—"}</td>
                <td style={TD}>{pick(e, "jockey_weight", "weight", "impost") || "—"}</td>
                <td style={{ ...TD, fontSize: 12, color: "var(--text-dim)" }}>{pick(e, "jockey_name", "jockey") || "—"}</td>
                <td style={{ ...TD, fontSize: 12, color: "var(--text-dim)" }}>{pick(e, "trainer_name", "trainer") || "—"}</td>
                <td style={{ ...TD, fontWeight: 600, color: oddsColorClass.includes("red") ? "#f87171" : oddsColorClass.includes("yellow") ? "#fbbf24" : "var(--text-dim)" }}>{oddsText}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );

  /* ── レース結果 ── */
  const resultEntries = [...(rd?.race_result?.entries ?? [])].sort((a, b) => (Number(a.finish_position ?? a.position ?? 99)) - (Number(b.finish_position ?? b.position ?? 99)));

  // めぐ指数計算用: フィールド内最速タイムを求める
  const allTimes = resultEntries.map((e) => parseTimeSec(e.time as string | undefined)).filter((v): v is number => v != null && v > 0);
  const bestTime = allTimes.length ? Math.min(...allTimes) : null;
  const fieldSize = resultEntries.length;

  const ResultTbl = resultEntries.length === 0 ? (
    <div style={{ textAlign: "center", padding: 40, color: "var(--text-dim)" }}>🏆 レース結果データがありません</div>
  ) : (
    <div style={{ overflowX: "auto" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 10, overflow: "hidden", fontSize: 13 }}>
        <thead style={{ background: "var(--surface2)" }}>
          <tr>
            {["着順", "枠", "馬番", "馬名", "騎手", "タイム", "通過順", "上り3F", "単勝"].map(h => <th key={h} style={TH}>{h}</th>)}
            <th style={{ ...TH, borderLeft: "2px solid rgba(167,139,250,0.3)", color: "#a78bfa" }}>
              めぐ指数
              <span style={{ fontSize: 9, verticalAlign: "super", marginLeft: 2, opacity: 0.7 }}>β</span>
            </th>
          </tr>
        </thead>
        <tbody>
          {resultEntries.map((e, i) => {
            const pos = Number(e.finish_position ?? e.position ?? 99);
            const placeColor = pos === 1 ? "#fbbf24" : pos === 2 ? "#94a3b8" : pos === 3 ? "#d97706" : "var(--text)";
            const tSec = parseTimeSec(e.time as string | undefined);
            const megu = calcMeguIndex(tSec, bestTime, fieldSize);
            const { color: meguColor_, bg: meguBg } = meguColor(megu);
            return (
              <tr key={i} style={{ transition: "background 0.12s" }}>
                <td style={{ ...TD, fontWeight: pos <= 3 ? 800 : 400, color: placeColor, fontSize: pos <= 3 ? 15 : 13 }}>{pos <= 0 || pos >= 99 ? "—" : pos}</td>
                <td style={TD}>{e.bracket_number ?? "—"}</td>
                <td style={{ ...TD, fontWeight: 800, color: "#7dd3fc" }}>{String(e.horse_number ?? e.number ?? "—")}</td>
                <td style={{ ...TD, fontWeight: 600, color: "#fff", textAlign: "left" }}>{e.horse_name ?? "—"}</td>
                <td style={{ ...TD, fontSize: 12, color: "var(--text-dim)" }}>{pick(e, "jockey_name", "jockey") || "—"}</td>
                <td style={TD}>{e.time ?? "—"}</td>
                <td style={{ ...TD, fontSize: 11 }}>{e.passing_order ?? "—"}</td>
                <td style={TD}>{e.last3f ?? "—"}</td>
                <td style={{ ...TD, color: "var(--text-dim)", fontSize: 12 }}>{e.win_odds != null ? `${Number(e.win_odds).toFixed(1)}` : "—"}</td>
                <td
                  style={{ ...TD, borderLeft: "2px solid rgba(167,139,250,0.15)", fontWeight: megu != null ? 700 : 400, color: meguColor_, background: meguBg }}
                  title="レースパフォーマンス指数（めぐ指数）β版 - 走破タイムベースの暫定算出"
                >
                  {megu != null ? megu.toFixed(1) : "—"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {/* めぐ指数の注釈 */}
      <div style={{ padding: "8px 12px", fontSize: 11, color: "var(--text-dim)", display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
        <span style={{ color: "#a78bfa", fontWeight: 600 }}>めぐ指数 β</span>
        <span>走破タイムを基にしたレースパフォーマンス指数（暫定モック算出）。正式ロジックは後日更新予定。</span>
        <span style={{ borderLeft: "1px solid var(--border)", paddingLeft: 10 }}>
          <strong style={{ color: "#22c55e" }}>≥105</strong> 優秀 /
          <strong style={{ color: "#60a5fa", marginLeft: 4 }}>98〜</strong> 良好 /
          <strong style={{ color: "var(--text-dim)", marginLeft: 4 }}>82〜</strong> 標準 /
          <strong style={{ color: "#f87171", marginLeft: 4 }}>＜82</strong> 低調
        </span>
                </div>
              </div>
  );

  /* ── AI予測 ── */
  const preds = predData?.predictions ?? [];
  const hasPred = predData?.has_prediction === true && preds.some((p) => p.pred_score != null || p.composite_score != null);
  const sortedPreds = hasPred
    ? [...preds].sort((a, b) => (a.composite_rank ?? a.pred_rank ?? 99) - (b.composite_rank ?? b.pred_rank ?? 99))
    : preds;

  const PredPanel = (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 16, flexWrap: "wrap" }}>
        <button
          style={{ background: "#1e3a5f", color: "#60a5fa", border: "1px solid rgba(59,130,246,0.3)", padding: "7px 14px", borderRadius: 6, fontSize: 12, fontWeight: 600, cursor: predicting ? "not-allowed" : "pointer", opacity: predicting ? 0.6 : 1 }}
          disabled={predicting}
          onClick={runPrediction}
        >
          {predicting ? "⏳ 予測実行中…" : hasPred ? "🔄 予測を再実行" : "🤖 AI予測を実行"}
        </button>
        {predData?.model_description && (
          <span style={{ fontSize: 11, padding: "4px 10px", borderRadius: 4, background: "rgba(59,130,246,0.12)", color: "#60a5fa" }}>
            {hasPred ? "🧠" : "📊"} {predData.model_description}
          </span>
        )}
        {hasPred && predData?.elapsed_sec != null && (
          <span style={{ fontSize: 11, color: "var(--text-dim)" }}>⏱ {predData.elapsed_sec}秒 / {predData.total_horses}頭</span>
        )}
      </div>

      {!hasPred && (
        <div style={{ marginBottom: 12, fontSize: 13, color: "var(--text-dim)", padding: "10px 14px", borderRadius: 8, background: "rgba(36,48,73,0.4)" }}>
          予測結果がありません。「AI予測を実行」で算出できます。
            </div>
          )}

      {sortedPreds.length > 0 ? (
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 10, overflow: "hidden", fontSize: 12 }}>
            <thead style={{ background: "var(--surface2)" }}>
              <tr>
                <th style={{ ...TH, textAlign: "center" }} colSpan={2}>出走馬</th>
                <th style={{ ...TH, borderLeft: "1px solid var(--border)" }} colSpan={3}>予測確率</th>
                <th style={{ ...TH, borderLeft: "1px solid var(--border)" }} colSpan={2}>想定オッズ</th>
                <th style={{ ...TH, borderLeft: "1px solid var(--border)" }} colSpan={2}>期待値</th>
                <th style={{ ...TH, borderLeft: "1px solid var(--border)" }} colSpan={2}>印</th>
              </tr>
              <tr>
                {["馬番","馬名","勝率","連対率","複勝率","単勝","複勝","勝期待値","複期待値","推奨","印"].map((h, i) => (
                  <th key={h} style={{ ...TH, borderLeft: [2,5,7,9,11].includes(i) ? "1px solid rgba(36,48,73,0.5)" : undefined }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sortedPreds.map((p, i) => {
                const mark = markMeta(p);
                const rank = p.composite_rank ?? p.pred_rank ?? 0;
                const rowBg = rank === 1 ? "rgba(239,68,68,0.04)" : rank === 2 ? "rgba(59,130,246,0.04)" : rank === 3 ? "rgba(34,197,94,0.04)" : undefined;
                return (
                  <tr key={p.horse_number ?? i} style={{ background: rowBg, transition: "background 0.12s" }}>
                    <td style={{ ...TD, fontWeight: 800, color: "#7dd3fc" }}>{p.horse_number ?? "—"}</td>
                    <td style={{ ...TD, fontWeight: 600, color: "#fff", textAlign: "left", whiteSpace: "nowrap" }}>{p.horse_name ?? "—"}</td>
                    <td style={{ ...TD, borderLeft: "1px solid rgba(36,48,73,0.3)" }}>{p.win_prob != null ? `${(p.win_prob * 100).toFixed(1)}%` : "—"}</td>
                    <td style={TD}>{p.top2_prob != null ? `${(p.top2_prob * 100).toFixed(1)}%` : "—"}</td>
                    <td style={TD}>{p.top3_prob != null ? `${(p.top3_prob * 100).toFixed(1)}%` : "—"}</td>
                    <td style={{ ...TD, borderLeft: "1px solid rgba(36,48,73,0.3)", color: "var(--text-dim)" }}>{p.win_odds != null ? `${Number(p.win_odds).toFixed(1)}倍` : "—"}</td>
                    <td style={{ ...TD, color: "var(--text-dim)", fontSize: 11 }}>{p.place_odds_min != null && p.place_odds_max != null ? `${Number(p.place_odds_min).toFixed(1)}〜${Number(p.place_odds_max).toFixed(1)}` : "—"}</td>
                    <td style={{ ...TD, borderLeft: "1px solid rgba(36,48,73,0.3)", color: evColor(p.ev_win), fontWeight: 600 }}>{p.ev_win != null ? p.ev_win.toFixed(2) : "—"}</td>
                    <td style={{ ...TD, color: evColor(p.ev_place ?? p.expected_value), fontWeight: 600 }}>{(p.ev_place ?? p.expected_value) != null ? Number(p.ev_place ?? p.expected_value).toFixed(2) : "—"}</td>
                    <td style={{ ...TD, borderLeft: "1px solid rgba(36,48,73,0.3)", fontSize: 11 }}>{p.buy_tier ?? "—"}</td>
                    <td style={{ ...TD, fontSize: 18, fontWeight: 800, color: mark.color }}>{mark.sym}</td>
                </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : (
        <div style={{ textAlign: "center", padding: 40, color: "var(--text-dim)" }}>出馬表がないため予測表を表示できません</div>
      )}

      {hasPred && predData?.feature_highlights?.length ? (
        <div style={{ marginTop: 16, padding: "12px 16px", background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 8 }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: "var(--text-dim)", marginBottom: 8 }}>特徴量ハイライト</div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
            {predData.feature_highlights.map((f, i) => (
              <span key={i} style={{ fontSize: 11, padding: "3px 8px", borderRadius: 4, background: f.direction === "positive" ? "rgba(34,197,94,0.12)" : "rgba(239,68,68,0.12)", color: f.direction === "positive" ? "var(--ok)" : "var(--err)" }}>
                {f.label} {f.magnitude ? `(${f.magnitude})` : ""}
              </span>
            ))}
          </div>
        </div>
      ) : null}

      <p style={{ fontSize: 11, color: "var(--text-dim)", marginTop: 12 }}>
        印: ◎1着優位・★中穴は各1頭まで / ○2連相手・✓3列紐・▲複勝妙味は複数可 / —回目に入れない。期待値: 緑≥1.3 / 青≥1.0 / 赤＜0.8
      </p>
    </div>
  );

  /* ── 出走馬詳細（追走難度との統合） ── */
  const tdEntries = (tdData as Record<string, unknown> & { entries?: unknown[] })?.entries ?? [];
  const HorseDetails = tdEntries.length === 0 ? (
    <div style={{ textAlign: "center", padding: 40 }}>
      <p style={{ color: "var(--text-dim)", marginBottom: 12 }}>追走難度データがありません</p>
      <a href={`/tracking-difficulty`} style={{ color: "var(--accent)", fontSize: 13 }}>
        → 追走難度分析ページで詳細を確認
      </a>
    </div>
  ) : (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill,minmax(280px,1fr))", gap: 12 }}>
      {(tdEntries as TdEntry[]).map((e: TdEntry) => {
        const td = (e.tracking_difficulty as { ease_pct?: number; ease_label?: string; flow_position?: string }) ?? {};
        const pct = td.ease_pct ?? 50;
        const color = pct >= 75 ? "var(--ok)" : pct >= 60 ? "var(--accent)" : pct >= 45 ? "var(--text-dim)" : pct >= 30 ? "var(--warn)" : "var(--err)";
        return (
          <div key={e.horse_id ?? e.horse_number} style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 8, padding: "12px 14px" }}>
            <div style={{ fontWeight: 700, color: "#fff", fontSize: 13, marginBottom: 8 }}>
              <span style={{ color: "#7dd3fc", marginRight: 6 }}>{e.horse_number}.</span>{e.horse_name}
            </div>
            <div style={{ fontSize: 11, marginBottom: 6 }}>
              <span style={{ color: "var(--text-dim)" }}>追走容易度: </span>
              <span style={{ fontWeight: 700, color }}>{pct.toFixed(1)}%</span>
              {td.ease_label && <span style={{ marginLeft: 6, padding: "1px 6px", borderRadius: 3, fontSize: 10, fontWeight: 600, background: color === "var(--ok)" ? "rgba(34,197,94,0.15)" : "rgba(107,125,149,0.12)", color }}>{td.ease_label}</span>}
            </div>
            <div style={{ height: 6, background: "rgba(36,48,73,0.5)", borderRadius: 3, overflow: "hidden", marginBottom: 6 }}>
              <div style={{ height: "100%", width: `${Math.min(100, Math.max(2, pct))}%`, background: color, borderRadius: 3 }} />
            </div>
            {td.flow_position && (
              <div style={{ fontSize: 11, color: "var(--text-dim)" }}>想定位置: <span style={{ fontWeight: 600, color: "#7dd3fc" }}>{td.flow_position}</span></div>
            )}
          </div>
        );
      })}
    </div>
  );

  type TdEntry = {
    horse_number?: number;
    horse_id?: string;
    horse_name?: string;
    tracking_difficulty?: { ease_pct?: number; ease_label?: string; flow_position?: string };
  };

  /* ── レンダー ── */
  return (
    <div style={{ minHeight: "100vh", background: "var(--bg)", color: "var(--text)", padding: 24 }}>
      <a href="/monitor" style={{ display: "inline-flex", alignItems: "center", gap: 6, fontSize: 13, color: "var(--text-dim)", marginBottom: 16, textDecoration: "none" }}>
        ← モニターへ
      </a>

      {Header}

      {/* タブバー */}
      <div style={{ display: "flex", gap: 4, marginBottom: 20, borderBottom: "1px solid var(--border)", paddingBottom: 0 }}>
        {TABS.map(({ id, label }) => (
          <button
            key={id}
            onClick={() => setTab(id)}
            style={{
              padding: "10px 20px",
              fontSize: 14,
              fontWeight: 500,
              color: tab === id ? "var(--accent)" : "var(--text-dim)",
              cursor: "pointer",
              background: "none",
              border: "none",
              borderBottom: `2px solid ${tab === id ? "var(--accent)" : "transparent"}`,
              transition: "all 0.15s",
            } as React.CSSProperties}
          >
            {label}
          </button>
        ))}
      </div>

      {/* タブコンテンツ */}
      {tab === "shutuba" && ShutubaTbl}
      {tab === "result" && ResultTbl}
      {tab === "predict" && PredPanel}
      {tab === "horses" && HorseDetails}
    </div>
  );
}
