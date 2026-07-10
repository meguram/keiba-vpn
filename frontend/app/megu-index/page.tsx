"use client";

import { useEffect, useState, useCallback } from "react";
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
  jockey_weight: number | null;
  finish_time_sec: number | null;
  actual_megu: number | null;
  base_megu: number | null;
  megu_adjusted: number | null;
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

type MeguPredicted = {
  race_id: string;
  race_info: RaceInfo;
  model_version: string;
  horses: MeguHorse[];
};

type SortKey = "horse_number" | "jockey_weight" | "finish_time_sec" | "megu_adjusted";
type SortDir = "asc" | "desc";

/* ── ユーティリティ ── */
function meguColor(v: number | null): { color: string; bg: string } {
  if (v == null) return { color: "var(--text-dim)", bg: "transparent" };
  if (v >= 105) return { color: "#22c55e", bg: "rgba(34,197,94,0.10)" };
  if (v >= 98)  return { color: "#4ade80", bg: "rgba(74,222,128,0.07)" };
  if (v >= 92)  return { color: "#60a5fa", bg: "rgba(96,165,250,0.07)" };
  if (v >= 82)  return { color: "var(--text-dim)", bg: "transparent" };
  return { color: "#f87171", bg: "rgba(239,68,68,0.07)" };
}

function fmtTime(sec: number | null): string {
  if (sec == null) return "—";
  const m = Math.floor(sec / 60);
  const s = (sec % 60).toFixed(1).padStart(4, "0");
  return m > 0 ? `${m}:${s}` : `${s}s`;
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

/* ── ソートヘッダー ── */
function SortTh({
  label, sortKey, current, dir, onSort, style,
}: {
  label: string; sortKey: SortKey; current: SortKey; dir: SortDir;
  onSort: (k: SortKey) => void; style?: React.CSSProperties;
}) {
  const active = current === sortKey;
  return (
    <th
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
  horse, rank, sortKey,
}: {
  horse: MeguHorse; rank: number; sortKey: SortKey;
}) {
  const [open, setOpen] = useState(false);
  const main = horse.actual_megu ?? horse.megu_adjusted ?? horse.base_megu;
  const mainC = meguColor(main);
  const cc = horse.condition_change;
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
        {/* 馬番 */}
        <td style={{ ...TDc, fontWeight: 800, color: "#7dd3fc", background: sortKey === "horse_number" ? "rgba(59,130,246,0.04)" : undefined }}>
          {horse.horse_number ?? "—"}
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

        {/* 斤量 */}
        <td style={{ ...TDc, color: "var(--text-dim)", background: sortKey === "jockey_weight" ? "rgba(59,130,246,0.04)" : undefined }}>
          {horse.jockey_weight != null ? `${horse.jockey_weight}kg` : "—"}
        </td>

        {/* 走破タイム */}
        <td style={{ ...TDc, fontFamily: "monospace", background: sortKey === "finish_time_sec" ? "rgba(59,130,246,0.04)" : undefined }}>
          {fmtTime(horse.finish_time_sec)}
        </td>

        {/* めぐ指数 */}
        <td style={{
          ...TDc, fontWeight: 800, fontSize: 15,
          color: mainC.color, background: sortKey === "megu_adjusted" ? mainC.bg || "rgba(59,130,246,0.04)" : mainC.bg,
        }}>
          {main != null ? main.toFixed(1) : "—"}
          {horse.actual_megu != null && (
            <span style={{ fontSize: 9, fontWeight: 500, color: "#22c55e", display: "block", lineHeight: 1 }}>実測</span>
          )}
          {horse.actual_megu == null && horse.base_megu != null && (
            <span style={{ fontSize: 9, fontWeight: 400, color: "var(--text-dim)", display: "block", lineHeight: 1 }}>
              直近平均 {horse.base_megu.toFixed(1)}
            </span>
          )}
        </td>

        {/* 履歴展開 */}
        <td style={{ ...TDc, width: 24, color: "var(--text-dim)", fontSize: 11 }}>
          {horse.history.length > 0 ? (open ? "▲" : "▼") : ""}
        </td>
      </tr>

      {/* 履歴展開行 */}
      {open && (
        <tr>
          <td colSpan={6} style={{ background: "rgba(12,18,32,0.7)", padding: "8px 16px 12px 40px" }}>
            <div style={{ fontSize: 11, color: "var(--text-dim)", marginBottom: 5 }}>直近{horse.history.length}走</div>
            <table style={{ borderCollapse: "collapse", fontSize: 11 }}>
              <tbody>
                {horse.history.map((h) => {
                  const c = meguColor(h.megu_index);
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
  race, data, loading, onLoad,
}: {
  race: RaceItem;
  data: MeguPredicted | null;
  loading: boolean;
  onLoad: (id: string) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const [sortKey, setSortKey] = useState<SortKey>("megu_adjusted");
  const [sortDir, setSortDir] = useState<SortDir>("desc");

  const ri = data?.race_info;
  const horses = data?.horses ?? [];

  function handleSort(key: SortKey) {
    if (sortKey === key) {
      setSortDir(d => d === "asc" ? "desc" : "asc");
    } else {
      setSortKey(key);
      setSortDir(key === "horse_number" ? "asc" : key === "finish_time_sec" ? "asc" : "desc");
    }
  }

  const sorted = [...horses].sort((a, b) => {
    let va: number | null, vb: number | null;
    if (sortKey === "horse_number") {
      va = a.horse_number; vb = b.horse_number;
    } else if (sortKey === "jockey_weight") {
      va = a.jockey_weight; vb = b.jockey_weight;
    } else if (sortKey === "finish_time_sec") {
      va = a.finish_time_sec; vb = b.finish_time_sec;
    } else {
      va = a.actual_megu ?? a.megu_adjusted ?? a.base_megu;
      vb = b.actual_megu ?? b.megu_adjusted ?? b.base_megu;
    }
    if (va == null && vb == null) return 0;
    if (va == null) return 1;
    if (vb == null) return -1;
    return sortDir === "asc" ? va - vb : vb - va;
  });

  const topHorse = horses.length > 0
    ? [...horses].sort((a, b) => {
        const va = a.actual_megu ?? a.megu_adjusted ?? -999;
        const vb = b.actual_megu ?? b.megu_adjusted ?? -999;
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
          {topHorse && (
            <span style={{ fontSize: 11, padding: "2px 8px", borderRadius: 4, background: "rgba(34,197,94,0.10)", color: "#22c55e", marginLeft: 4 }}>
              Top: {topHorse.horse_name ?? "—"}{" "}
              {((topHorse.actual_megu ?? topHorse.megu_adjusted) ?? 0).toFixed(1)}
            </span>
          )}
          <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 8 }}>
            {data && (
              <span style={{ fontSize: 10, fontWeight: 600, padding: "2px 7px", borderRadius: 3, background: horses.length > 0 ? "rgba(34,197,94,0.12)" : "rgba(107,125,149,0.12)", color: horses.length > 0 ? "var(--ok)" : "var(--text-dim)" }}>
                {horses.length > 0 ? `✓ ${horses.length}頭` : "データなし"}
              </span>
            )}
            <Link href={`/race/${race.race_id}`} onClick={e => e.stopPropagation()} style={{ fontSize: 11, color: "var(--accent)", textDecoration: "none", border: "1px solid rgba(59,130,246,0.3)", padding: "3px 8px", borderRadius: 4 }}>
              詳細 →
            </Link>
            <span style={{ fontSize: 14, color: "var(--text-dim)", userSelect: "none" }}>{expanded ? "▲" : "▼"}</span>
          </div>
        </div>

        {/* 2行目: レース情報ピル */}
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          {(ri?.venue ?? race.venue) && (
            <span style={{ fontSize: 11, padding: "2px 8px", borderRadius: 12, background: "rgba(36,48,73,0.6)", color: "var(--text-dim)" }}>
              📍 {ri?.venue ?? race.venue}
            </span>
          )}
          {(ri?.surface || ri?.distance || race.surface || race.distance) && (
            <span style={{ fontSize: 11, padding: "2px 8px", borderRadius: 12, background: "rgba(36,48,73,0.6)", color: "var(--text-dim)" }}>
              {ri?.surface ?? race.surface}{ri?.distance ?? race.distance}m
            </span>
          )}
          {ri?.track_condition && (
            <span style={{ fontSize: 11, padding: "2px 8px", borderRadius: 12, background: "rgba(36,48,73,0.6)", color: "var(--text-dim)" }}>
              馬場: {ri.track_condition}
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
          {data && horses.length > 0 && (
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                <thead>
                  <tr>
                    <SortTh label="馬番" sortKey="horse_number" current={sortKey} dir={sortDir} onSort={handleSort} style={{ width: 52 }} />
                    <th style={{ ...TH, width: 160 }}>馬名</th>
                    <SortTh label="斤量" sortKey="jockey_weight" current={sortKey} dir={sortDir} onSort={handleSort} style={{ width: 68 }} />
                    <SortTh label="走破タイム" sortKey="finish_time_sec" current={sortKey} dir={sortDir} onSort={handleSort} style={{ width: 100 }} />
                    <SortTh label="めぐ指数" sortKey="megu_adjusted" current={sortKey} dir={sortDir} onSort={handleSort} style={{ width: 100 }} />
                    <th style={{ ...TH, width: 24 }} />
                  </tr>
                </thead>
                <tbody>
                  {sorted.map((horse, i) => (
                    <HorseRow key={horse.horse_id} horse={horse} rank={i} sortKey={sortKey} />
                  ))}
                </tbody>
              </table>
              <p style={{ fontSize: 10, color: "var(--text-dim)", padding: "5px 14px 10px" }}>
                めぐ指数: 実測値があれば実測、なければ直近3走平均 ± 条件転換補正
                {horses.some(h => h.condition_change.type !== "none") && " ／ ⚠ = 条件代わり(±600m超 or 芝↔ダート)"}
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
  const [loadingRaces, setLoadingRaces] = useState(false);
  const [racesError, setRacesError] = useState("");

  useEffect(() => {
    if (USE_MOCK) {
      const list = getMockRaceDates();
      setDates(list);
      if (list.length) setSelectedDate(list[0]);
      setLoadingDates(false);
      return;
    }
    (async () => {
      try {
        const res = await fetch("/api/scrape-dates?picker_past_days=30", { cache: "no-store" });
        if (!res.ok) return;
        const d = await res.json();
        const list: string[] = d.dates ?? d ?? [];
        setDates(list);
        if (list.length) setSelectedDate(list[0]);
      } catch { /* ignore */ } finally {
        setLoadingDates(false);
      }
    })();
  }, []);

  const loadRaces = useCallback(async (date: string) => {
    if (!date) return;
    setLoadingRaces(true);
    setRacesError("");
    setRaces([]);
    setMeguMap({});
    try {
      if (USE_MOCK) {
        setRaces(MOCK_WEEKLY_RACES.map(r => ({
          race_id: r.race_id,
          race_name: r.race_name,
          venue: r.venue,
          round: r.round,
          distance: r.distance,
          surface: r.surface,
          grade: r.grade,
        })));
      } else {
        const res = await fetch(`/api/race-list/${date}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const d = await res.json();
        setRaces(d.races ?? d ?? []);
      }
    } catch (e: unknown) {
      setRacesError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoadingRaces(false);
    }
  }, []);

  useEffect(() => {
    if (selectedDate) loadRaces(selectedDate);
  }, [selectedDate, loadRaces]);

  const loadMegu = useCallback(async (raceId: string) => {
    setLoadingMegu(prev => ({ ...prev, [raceId]: true }));
    try {
      if (USE_MOCK) {
        await new Promise(r => setTimeout(r, 300));
        const d = getMockMeguPredicted(raceId) as MeguPredicted;
        setMeguMap(prev => ({ ...prev, [raceId]: d }));
        return;
      }
      const res = await fetch(`/api/v1/races/${raceId}/megu-index-predicted`);
      const d: MeguPredicted = res.ok ? await res.json() : {
        race_id: raceId,
        race_info: { race_name: null, venue: null, surface: null, distance: null, dist_band: null, track_condition: null, grade: null, race_date: null },
        model_version: "",
        horses: [],
      };
      setMeguMap(prev => ({ ...prev, [raceId]: d }));
    } catch {
      setMeguMap(prev => ({ ...prev, [raceId]: null }));
    } finally {
      setLoadingMegu(prev => ({ ...prev, [raceId]: false }));
    }
  }, []);

  async function loadAll() {
    for (const race of races) {
      if (meguMap[race.race_id]) continue;
      await loadMegu(race.race_id);
    }
  }

  const loadedCount = Object.keys(meguMap).length;
  const dataCount = Object.values(meguMap).filter(m => (m?.horses?.length ?? 0) > 0).length;
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
          <p style={{ fontSize: 12, color: "var(--text-dim)", margin: 0 }}>馬番 / 馬名 / 斤量 / 走破タイム / めぐ指数</p>
        </div>
      </div>

      {/* コントロールバー */}
      <div style={{ background: "var(--surface)", borderBottom: "1px solid var(--border)", padding: "12px 24px" }}>
        <div style={{ maxWidth: 1100, margin: "0 auto", display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
          <label style={{ fontSize: 12, color: "var(--text-dim)" }}>開催日:</label>
          <select style={SEL} value={selectedDate} onChange={e => setSelectedDate(e.target.value)} disabled={loadingDates}>
            {loadingDates ? <option>読み込み中…</option> :
              dates.length === 0 ? <option>データなし</option> :
              dates.map(d => <option key={d} value={d}>{d}</option>)}
          </select>

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
        <div style={{ marginBottom: 16, padding: "8px 14px", background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 11, color: "var(--text-dim)", display: "flex", gap: 14, flexWrap: "wrap" }}>
          <span>めぐ指数:</span>
          <span><strong style={{ color: "#22c55e" }}>≥105</strong></span>
          <span><strong style={{ color: "#4ade80" }}>≥98</strong></span>
          <span><strong style={{ color: "#60a5fa" }}>≥92</strong></span>
          <span><strong style={{ color: "var(--text-dim)" }}>≥82</strong></span>
          <span><strong style={{ color: "#f87171" }}>＜82</strong></span>
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
            {races.map(race => (
              <RaceCard
                key={race.race_id}
                race={race}
                data={meguMap[race.race_id] ?? null}
                loading={loadingMegu[race.race_id] ?? false}
                onLoad={loadMegu}
              />
            ))}
          </>
        )}
      </div>
    </div>
  );
}
