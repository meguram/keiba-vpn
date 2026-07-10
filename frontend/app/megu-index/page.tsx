"use client";

import { useEffect, useState, useCallback } from "react";
import Link from "next/link";

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

type MeguEntry = {
  horse_id: string;
  horse_name?: string;
  horse_number?: number;
  megu_index: number;
};

type MeguData = {
  race_id: string;
  megu_index: MeguEntry[];
  source?: string;
};

function meguColor(v: number | null): { color: string; bg: string } {
  if (v == null) return { color: "var(--text-dim)", bg: "transparent" };
  if (v >= 105) return { color: "#22c55e", bg: "rgba(34,197,94,0.10)" };
  if (v >= 98)  return { color: "#4ade80", bg: "rgba(74,222,128,0.07)" };
  if (v >= 92)  return { color: "#60a5fa", bg: "rgba(96,165,250,0.07)" };
  if (v >= 82)  return { color: "var(--text-dim)", bg: "transparent" };
  return { color: "#f87171", bg: "rgba(239,68,68,0.07)" };
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

function MeguTable({ entries }: { entries: MeguEntry[] }) {
  const TH: React.CSSProperties = {
    padding: "7px 10px", fontSize: 11, fontWeight: 600, color: "var(--text-dim)",
    textAlign: "center", borderBottom: "1px solid var(--border)", background: "var(--surface2)", whiteSpace: "nowrap",
  };
  const TD: React.CSSProperties = {
    padding: "8px 10px", borderBottom: "1px solid rgba(36,48,73,0.4)",
    textAlign: "center", verticalAlign: "middle", fontSize: 13,
  };

  return (
    <div style={{ overflowX: "auto", marginTop: 12 }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
        <thead>
          <tr>
            <th style={{ ...TH, width: 40 }}>順位</th>
            <th style={{ ...TH, width: 44 }}>馬番</th>
            <th style={{ ...TH, textAlign: "left" }}>馬名</th>
            <th style={{ ...TH, width: 100 }}>めぐ指数</th>
          </tr>
        </thead>
        <tbody>
          {entries.map((e, i) => {
            const c = meguColor(e.megu_index);
            return (
              <tr key={e.horse_id} style={{ background: i === 0 ? "rgba(34,197,94,0.04)" : i === 1 ? "rgba(96,165,250,0.03)" : undefined }}>
                <td style={{ ...TD, fontWeight: 700, color: i < 3 ? "#f0f6fc" : "var(--text-dim)" }}>{i + 1}</td>
                <td style={{ ...TD, fontWeight: 700, color: "#7dd3fc" }}>{e.horse_number ?? "—"}</td>
                <td style={{ ...TD, textAlign: "left", fontWeight: 600, color: "#fff" }}>{e.horse_name ?? e.horse_id}</td>
                <td style={{ ...TD, fontWeight: 800, color: c.color, background: c.bg }}>
                  {e.megu_index.toFixed(1)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function RaceCard({
  race,
  megu,
  loading,
  onLoad,
}: {
  race: RaceItem;
  megu: MeguData | null;
  loading: boolean;
  onLoad: (id: string) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const hasData = (megu?.megu_index?.length ?? 0) > 0;

  return (
    <div style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 10, overflow: "hidden", marginBottom: 12 }}>
      <div
        style={{ padding: "14px 18px", display: "flex", alignItems: "center", gap: 12, cursor: "pointer", flexWrap: "wrap" }}
        onClick={() => {
          setExpanded(!expanded);
          if (!megu && !loading) onLoad(race.race_id);
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

        {hasData && megu && (
          <span style={{ fontSize: 11, padding: "2px 8px", borderRadius: 4, background: "rgba(34,197,94,0.10)", color: "#22c55e" }}>
            Top: {megu.megu_index[0]?.horse_name ?? "—"} {megu.megu_index[0]?.megu_index.toFixed(1)}
          </span>
        )}

        <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 8 }}>
          {megu && (
            <span style={{
              fontSize: 10, fontWeight: 600, padding: "2px 7px", borderRadius: 3,
              background: hasData ? "rgba(34,197,94,0.12)" : "rgba(107,125,149,0.12)",
              color: hasData ? "var(--ok)" : "var(--text-dim)",
            }}>
              {hasData ? `✓ ${megu.megu_index.length}頭` : "データなし"}
            </span>
          )}
          <Link
            href={`/race/${race.race_id}`}
            onClick={(e) => e.stopPropagation()}
            style={{ fontSize: 11, color: "var(--accent)", textDecoration: "none", border: "1px solid rgba(59,130,246,0.3)", padding: "3px 8px", borderRadius: 4 }}
          >
            詳細 →
          </Link>
          <span style={{ fontSize: 16, color: "var(--text-dim)", userSelect: "none" }}>{expanded ? "▲" : "▼"}</span>
        </div>
      </div>

      {expanded && (
        <div style={{ borderTop: "1px solid var(--border)", padding: "0 18px 14px" }}>
          {loading && (
            <div style={{ padding: "20px 0", textAlign: "center", color: "var(--text-dim)", fontSize: 13 }}>
              <div style={{ width: 24, height: 24, border: "2px solid var(--border)", borderTopColor: "var(--accent)", borderRadius: "50%", animation: "spin 1s linear infinite", margin: "0 auto 8px" }} />
              めぐ指数を取得中…
            </div>
          )}
          {!loading && !megu && (
            <div style={{ padding: "16px 0", display: "flex", gap: 10, alignItems: "center" }}>
              <button
                style={{ background: "#1e3a5f", color: "#60a5fa", border: "1px solid rgba(59,130,246,0.3)", padding: "6px 14px", borderRadius: 6, fontSize: 12, fontWeight: 600, cursor: "pointer" }}
                onClick={() => onLoad(race.race_id)}
              >
                📊 めぐ指数を取得
              </button>
            </div>
          )}
          {megu && !hasData && (
            <p style={{ padding: "14px 0", fontSize: 12, color: "var(--text-dim)" }}>
              このレースのめぐ指数データがありません。計算スクリプトを実行してください。
            </p>
          )}
          {megu && hasData && <MeguTable entries={megu.megu_index} />}
        </div>
      )}
    </div>
  );
}

export default function MeguIndexPage() {
  const [dates, setDates] = useState<string[]>([]);
  const [selectedDate, setSelectedDate] = useState("");
  const [races, setRaces] = useState<RaceItem[]>([]);
  const [meguMap, setMeguMap] = useState<Record<string, MeguData | null>>({});
  const [loadingMegu, setLoadingMegu] = useState<Record<string, boolean>>({});
  const [loadingDates, setLoadingDates] = useState(true);
  const [loadingRaces, setLoadingRaces] = useState(false);
  const [racesError, setRacesError] = useState("");

  useEffect(() => {
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
      const res = await fetch(`/api/race-list/${date}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const d = await res.json();
      setRaces(d.races ?? d ?? []);
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
    setLoadingMegu((prev) => ({ ...prev, [raceId]: true }));
    try {
      const res = await fetch(`/api/v1/races/${raceId}/megu-index`);
      const d: MeguData = res.ok ? await res.json() : { race_id: raceId, megu_index: [] };
      setMeguMap((prev) => ({ ...prev, [raceId]: d }));
    } catch {
      setMeguMap((prev) => ({ ...prev, [raceId]: { race_id: raceId, megu_index: [] } }));
    } finally {
      setLoadingMegu((prev) => ({ ...prev, [raceId]: false }));
    }
  }, []);

  async function loadAll() {
    for (const race of races) {
      if (meguMap[race.race_id]) continue;
      await loadMegu(race.race_id);
    }
  }

  const loadedCount = Object.keys(meguMap).length;
  const dataCount = Object.values(meguMap).filter((m) => (m?.megu_index?.length ?? 0) > 0).length;

  const SEL: React.CSSProperties = {
    background: "var(--surface2)", border: "1px solid var(--border)", color: "var(--text)",
    padding: "7px 12px", borderRadius: 6, fontSize: 13,
  };

  return (
    <div style={{ minHeight: "100vh", background: "var(--bg)", color: "var(--text)" }}>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>

      <div style={{ background: "var(--surface)", borderBottom: "1px solid var(--border)", padding: "18px 24px" }}>
        <div style={{ maxWidth: 1200, margin: "0 auto", display: "flex", alignItems: "center", gap: 12 }}>
          <Link href="/" style={{ fontSize: 12, color: "var(--text-dim)", textDecoration: "none" }}>← ホーム</Link>
          <span style={{ color: "var(--border)" }}>/</span>
          <h1 style={{ fontSize: 18, fontWeight: 700, color: "#f0f6fc", margin: 0 }}>📊 今週のめぐ指数</h1>
          <p style={{ fontSize: 12, color: "var(--text-dim)", margin: 0 }}>
            開催日ごとのめぐ指数ランキングを一覧表示
          </p>
        </div>
      </div>

      <div style={{ background: "var(--surface)", borderBottom: "1px solid var(--border)", padding: "12px 24px" }}>
        <div style={{ maxWidth: 1200, margin: "0 auto", display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
          <label style={{ fontSize: 12, color: "var(--text-dim)" }}>開催日:</label>
          <select style={SEL} value={selectedDate} onChange={(e) => setSelectedDate(e.target.value)} disabled={loadingDates}>
            {loadingDates ? <option>読み込み中…</option> :
              dates.length === 0 ? <option>データなし</option> :
              dates.map((d) => <option key={d} value={d}>{d}</option>)}
          </select>

          {races.length > 0 && (
            <button
              style={{ background: "#1e3a5f", color: "#60a5fa", border: "1px solid rgba(59,130,246,0.3)", padding: "7px 14px", borderRadius: 6, fontSize: 12, fontWeight: 600, cursor: "pointer" }}
              onClick={loadAll}
            >
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
            </div>
          )}
        </div>
      </div>

      <div style={{ maxWidth: 1200, margin: "0 auto", padding: "20px 24px" }}>
        <div style={{ marginBottom: 16, padding: "10px 14px", background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 11, color: "var(--text-dim)", display: "flex", gap: 16, flexWrap: "wrap" }}>
          <span>めぐ指数の色:</span>
          <span><strong style={{ color: "#22c55e" }}>≥105</strong> 高評価</span>
          <span><strong style={{ color: "#4ade80" }}>≥98</strong></span>
          <span><strong style={{ color: "#60a5fa" }}>≥92</strong></span>
          <span><strong style={{ color: "var(--text-dim)" }}>≥82</strong> 平均</span>
          <span><strong style={{ color: "#f87171" }}>＜82</strong> 低評価</span>
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
            {races.map((race) => (
              <RaceCard
                key={race.race_id}
                race={race}
                megu={meguMap[race.race_id] ?? null}
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
