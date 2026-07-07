"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import Link from "next/link";
import { USE_MOCK, MOCK_WEEKLY_RACES, getMockRaceDates, getMockPredictions } from "@/lib/mock";

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
};

type PredData = {
  status?: string;
  has_prediction?: boolean;
  model_description?: string;
  total_horses?: number;
  predictions?: PredEntry[];
  error?: string;
};

/* ── ユーティリティ ── */
const MARK_META: Record<string, { sym: string; color: string }> = {
  honmei:   { sym: "◎", color: "#ef4444" },
  pair:     { sym: "○", color: "#3b82f6" },
  anchor:   { sym: "✓", color: "#22c55e" },
  show_val: { sym: "▲", color: "#f59e0b" },
  star:     { sym: "★", color: "#a78bfa" },
  none:     { sym: "—", color: "var(--text-dim)" },
};

function markSym(p: PredEntry): { sym: string; color: string } {
  if (p.mark_type && MARK_META[p.mark_type]) return MARK_META[p.mark_type];
  return MARK_META.none;
}

function fmtPct(v: number | null | undefined): string {
  if (v == null) return "—";
  return `${(v * 100).toFixed(1)}%`;
}

function fmtEv(v: number | null | undefined): { text: string; color: string; bg: string } {
  if (v == null) return { text: "—", color: "var(--text-dim)", bg: "transparent" };
  const n = Number(v);
  if (n >= 1.5) return { text: n.toFixed(2), color: "#22c55e", bg: "rgba(34,197,94,0.10)" };
  if (n >= 1.2) return { text: n.toFixed(2), color: "#4ade80", bg: "rgba(74,222,128,0.08)" };
  if (n >= 1.0) return { text: n.toFixed(2), color: "#60a5fa", bg: "rgba(96,165,250,0.08)" };
  if (n >= 0.8) return { text: n.toFixed(2), color: "var(--text-dim)", bg: "transparent" };
  return { text: n.toFixed(2), color: "#f87171", bg: "rgba(239,68,68,0.07)" };
}

function fmtOdds(v: number | null | undefined): string {
  if (v == null) return "—";
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
          ◎ {honmei.horse_name ?? `馬番${honmei.horse_number}`}
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

/* ── 予測テーブル ── */
function PredTable({ preds, hasPred }: { preds: PredEntry[]; hasPred: boolean }) {
  const sorted = hasPred
    ? [...preds].sort((a, b) => (a.composite_rank ?? a.pred_rank ?? 99) - (b.composite_rank ?? b.pred_rank ?? 99))
    : [...preds].sort((a, b) => (a.horse_number ?? 0) - (b.horse_number ?? 0));

  const TH: React.CSSProperties = {
    padding: "8px 10px", fontSize: 11, fontWeight: 600, color: "var(--text-dim)",
    textAlign: "center", borderBottom: "1px solid var(--border)", whiteSpace: "nowrap", background: "var(--surface2)",
  };
  const TD: React.CSSProperties = {
    padding: "8px 10px", borderBottom: "1px solid rgba(36,48,73,0.4)",
    textAlign: "center", verticalAlign: "middle", fontSize: 12,
  };

  return (
    <div style={{ overflowX: "auto", marginTop: 12 }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
        <thead>
          <tr>
            <th style={{ ...TH, width: 36 }}>印</th>
            <th style={{ ...TH, width: 36 }}>馬番</th>
            <th style={{ ...TH, textAlign: "left", width: 120 }}>馬名</th>
            <th style={{ ...TH, borderLeft: "1px solid rgba(36,48,73,0.5)" }}>勝率</th>
            <th style={TH}>連対率</th>
            <th style={TH}>複勝率</th>
            <th style={{ ...TH, borderLeft: "1px solid rgba(36,48,73,0.5)" }}>想定単勝</th>
            <th style={TH}>想定複勝</th>
            <th style={{ ...TH, borderLeft: "1px solid rgba(36,48,73,0.5)" }}>勝期待値</th>
            <th style={TH}>複期待値</th>
            <th style={{ ...TH, borderLeft: "1px solid rgba(36,48,73,0.5)", width: 60 }}>推奨</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((p, i) => {
            const rank = p.composite_rank ?? p.pred_rank ?? 0;
            const mark = markSym(p);
            const evW = fmtEv(p.ev_win);
            const evP = fmtEv(p.ev_place ?? p.expected_value);
            const rowBg =
              rank === 1 ? "rgba(239,68,68,0.04)" :
              rank === 2 ? "rgba(59,130,246,0.04)" :
              rank === 3 ? "rgba(34,197,94,0.03)" : undefined;
            return (
              <tr key={p.horse_number ?? i} style={{ background: rowBg }}>
                <td style={{ ...TD, fontSize: 16, fontWeight: 800, color: mark.color }}>{mark.sym}</td>
                <td style={{ ...TD, fontWeight: 800, color: "#7dd3fc" }}>{p.horse_number ?? "—"}</td>
                <td style={{ ...TD, textAlign: "left", fontWeight: 600, color: "#fff", whiteSpace: "nowrap" }}>
                  {p.horse_id ? (
                    <Link href={`#horse-${p.horse_id}`} style={{ color: "#fff", textDecoration: "none" }}>{p.horse_name ?? "—"}</Link>
                  ) : (p.horse_name ?? "—")}
                </td>
                <td style={{ ...TD, borderLeft: "1px solid rgba(36,48,73,0.3)", fontWeight: hasPred ? 600 : 400, color: hasPred ? "var(--text)" : "var(--text-dim)" }}>{fmtPct(p.win_prob)}</td>
                <td style={{ ...TD, color: hasPred ? "var(--text)" : "var(--text-dim)" }}>{fmtPct(p.top2_prob)}</td>
                <td style={{ ...TD, color: hasPred ? "var(--text)" : "var(--text-dim)" }}>{fmtPct(p.top3_prob)}</td>
                <td style={{ ...TD, borderLeft: "1px solid rgba(36,48,73,0.3)", color: "var(--text-dim)" }}>{fmtOdds(p.win_odds)}</td>
                <td style={{ ...TD, color: "var(--text-dim)", fontSize: 11 }}>
                  {p.place_odds_min != null && p.place_odds_max != null
                    ? `${Number(p.place_odds_min).toFixed(1)}〜${Number(p.place_odds_max).toFixed(1)}`
                    : "—"}
                </td>
                <td style={{ ...TD, borderLeft: "1px solid rgba(36,48,73,0.3)", background: evW.bg, fontWeight: hasPred ? 700 : 400, color: evW.color }}>{evW.text}</td>
                <td style={{ ...TD, background: evP.bg, fontWeight: hasPred ? 700 : 400, color: evP.color }}>{evP.text}</td>
                <td style={{ ...TD, borderLeft: "1px solid rgba(36,48,73,0.3)", fontSize: 11, color: "var(--text-dim)" }}>{p.buy_tier ?? "—"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {!hasPred && (
        <p style={{ fontSize: 11, color: "var(--text-dim)", padding: "8px 10px" }}>
          ※ 予測未実行。数値は全て「—」です。レース詳細から「AI予測を実行」してください。
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
}: {
  race: RaceItem;
  pred: PredData | null;
  loading: boolean;
  onLoad: (id: string) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const hasPred = pred?.has_prediction === true && (pred.predictions?.some((p) => p.win_prob != null) ?? false);
  const preds = pred?.predictions ?? [];

  return (
    <div style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 10, overflow: "hidden", marginBottom: 12 }}>
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
        <div style={{ borderTop: "1px solid var(--border)", padding: "0 18px 14px" }}>
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
            <PredTable preds={preds} hasPred={hasPred} />
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
  const [bulkLoading, setBulkLoading] = useState(false);
  const bulkAbort = useRef<AbortController | null>(null);

  /* 開催日一覧 */
  useEffect(() => {
    if (USE_MOCK) {
      const mockDates = getMockRaceDates();
      setDates(mockDates);
      setSelectedDate(mockDates[0]);
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

  /* レース一覧（日付変更時） */
  const loadRaces = useCallback(async (date: string) => {
    if (!date) return;
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
      const res = await fetch(`/api/race-list/${date}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const d = await res.json();
      const list: RaceItem[] = d.races ?? d ?? [];
      setRaces(list);
    } catch (e: unknown) {
      setRacesError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoadingRaces(false);
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
      const res = await fetch(`/api/race/${raceId}/predictions`);
      const d: PredData = res.ok ? await res.json() : { status: "error", predictions: [] };
      setPreds((prev) => ({ ...prev, [raceId]: d }));
    } catch {
      setPreds((prev) => ({ ...prev, [raceId]: { status: "error", predictions: [] } }));
    } finally {
      setLoadingPred((prev) => ({ ...prev, [raceId]: false }));
    }
  }, []);

  /* 全レース一括取得 */
  async function loadAllPreds() {
    if (!races.length) return;
    setBulkLoading(true);
    bulkAbort.current = new AbortController();
    const signal = bulkAbort.current.signal;
    for (const race of races) {
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
  const loadedCount = Object.keys(preds).length;
  const predOkCount = Object.values(preds).filter((p) => p?.has_prediction).length;
  const totalEVOk = Object.values(preds)
    .flatMap((p) => p?.predictions ?? [])
    .filter((h) => (h.ev_win ?? 0) >= 1.0).length;

  const SEL: React.CSSProperties = {
    background: "var(--surface2)", border: "1px solid var(--border)", color: "var(--text)",
    padding: "7px 12px", borderRadius: 6, fontSize: 13,
  };

  return (
    <div style={{ minHeight: "100vh", background: "var(--bg)", color: "var(--text)" }}>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>

      {/* devモック通知バナー */}
      {USE_MOCK && (
        <div style={{ background: "rgba(88,166,255,0.1)", borderBottom: "1px solid rgba(88,166,255,0.25)", padding: "8px 24px", fontSize: 12, color: "#58a6ff", textAlign: "center" }}>
          🔧 開発モード（モックデータ表示中）— 実際のレースデータではありません
        </div>
      )}

      {/* ページヘッダー */}
      <div style={{ background: "var(--surface)", borderBottom: "1px solid var(--border)", padding: "18px 24px" }}>
        <div style={{ maxWidth: 1200, margin: "0 auto", display: "flex", alignItems: "center", gap: 12 }}>
          <Link href="/" style={{ fontSize: 12, color: "var(--text-dim)", textDecoration: "none" }}>← ホーム</Link>
          <span style={{ color: "var(--border)" }}>/</span>
          <h1 style={{ fontSize: 18, fontWeight: 700, color: "#f0f6fc", margin: 0 }}>🤖 今週のAI予測</h1>
          <p style={{ fontSize: 12, color: "var(--text-dim)", margin: 0 }}>
            開催日ごとの勝率・期待値を一覧表示
          </p>
        </div>
      </div>

      {/* コントロールバー */}
      <div style={{ background: "var(--surface)", borderBottom: "1px solid var(--border)", padding: "12px 24px" }}>
        <div style={{ maxWidth: 1200, margin: "0 auto", display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
          <label style={{ fontSize: 12, color: "var(--text-dim)" }}>開催日:</label>
          <select
            style={SEL}
            value={selectedDate}
            onChange={(e) => setSelectedDate(e.target.value)}
            disabled={loadingDates}
          >
            {loadingDates ? <option>読み込み中…</option> :
              dates.length === 0 ? <option>データなし</option> :
              dates.map((d) => <option key={d} value={d}>{d}</option>)
            }
          </select>

          {races.length > 0 && (
            <>
              {!bulkLoading ? (
                <button
                  style={{ background: "#1e3a5f", color: "#60a5fa", border: "1px solid rgba(59,130,246,0.3)", padding: "7px 14px", borderRadius: 6, fontSize: 12, fontWeight: 600, cursor: "pointer" }}
                  onClick={loadAllPreds}
                >
                  🤖 全{races.length}レースの予測を取得
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
                {loadedCount}/{races.length}R 取得済
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
      <div style={{ maxWidth: 1200, margin: "0 auto", padding: "20px 24px" }}>
        {/* 凡例 */}
        <div style={{ marginBottom: 16, padding: "10px 14px", background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 11, color: "var(--text-dim)", display: "flex", gap: 16, flexWrap: "wrap" }}>
          <span><strong style={{ color: "#ef4444" }}>◎</strong> 1着優位</span>
          <span><strong style={{ color: "#3b82f6" }}>○</strong> 2連相手</span>
          <span><strong style={{ color: "#22c55e" }}>✓</strong> 3列紐</span>
          <span><strong style={{ color: "#f59e0b" }}>▲</strong> 複勝妙味</span>
          <span><strong style={{ color: "#a78bfa" }}>★</strong> 中穴</span>
          <span style={{ borderLeft: "1px solid var(--border)", paddingLeft: 12 }}>
            期待値: <strong style={{ color: "#22c55e" }}>≥1.3</strong> / <strong style={{ color: "#60a5fa" }}>≥1.0</strong> / <strong style={{ color: "var(--text-dim)" }}>≥0.8</strong> / <strong style={{ color: "#f87171" }}>＜0.8</strong>
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
              {bulkLoading && <span style={{ marginLeft: 8, color: "var(--accent)" }}>⏳ 取得中…</span>}
            </div>
            {races.map((race) => (
              <RaceCard
                key={race.race_id}
                race={race}
                pred={preds[race.race_id] ?? null}
                loading={loadingPred[race.race_id] ?? false}
                onLoad={loadPred}
              />
            ))}
          </>
        )}
      </div>
    </div>
  );
}
