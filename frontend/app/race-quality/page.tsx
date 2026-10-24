"use client";

import { useState, useCallback } from "react";
import { RacePicker } from "@/components/RacePicker";
import { USE_MOCK, getMockRaceQuality, getMockRaceQualityDay } from "@/lib/mock";

/* ── 型 ── */
type AxisInfo = { label_ja?: string; p?: number; description?: string };
type RaceQualityRace = {
  race_id: string;
  race_name?: string;
  venue?: string;
  distance?: string | number;
  surface?: string;
  track_condition?: string;
  segment_key?: string;
  n_runners?: number;
  r2_fit?: number;
  probs?: number[];
  axes?: AxisInfo[];
  pace_shape?: { grind_index?: number; burst_index?: number; lap_evenness?: number };
};

type DayData = {
  date?: string;
  day_summary?: { axes?: AxisInfo[]; probs?: number[]; n_races?: number };
  by_segment?: Record<string, { n_races?: number; probs?: number[]; axes?: AxisInfo[] }>;
  races?: RaceQualityRace[];
  skipped?: number;
};

type SingleData = {
  race_id?: string;
  race_name?: string;
  axes?: AxisInfo[];
  probs?: number[];
  r2_fit?: number;
  n_runners?: number;
  error?: string;
};

/* ── 軸カラーパレット（9軸） ── */
const COLORS = [
  "#2dd4bf", "#a78bfa", "#f472b6", "#fb923c",
  "#4ade80", "#facc15", "#94a3b8", "#f87171", "#475569",
];

/* ── StackedBar ── */
function StackedBar({ probs, labels, height = 40 }: { probs: number[]; labels: string[]; height?: number }) {
  const total = probs.reduce((s, v) => s + Math.max(0, v), 0) || 1;
  return (
    <div style={{ display: "flex", height, borderRadius: 8, overflow: "hidden", boxShadow: "inset 0 0 0 1px rgba(42,49,66,0.8)" }}>
      {probs.map((v, i) => {
        const pct = (Math.max(0, v) / total) * 100;
        if (pct < 0.5) return null;
        return (
          <div
            key={i}
            style={{ flex: `0 0 ${pct}%`, background: COLORS[i % COLORS.length], display: "flex", alignItems: "center", justifyContent: "center" }}
            title={`${labels[i] ?? `軸${i + 1}`}: ${(pct).toFixed(1)}%`}
          >
            {pct >= 8 && <span style={{ fontSize: 9, fontWeight: 700, color: "rgba(0,0,0,0.75)" }}>{pct.toFixed(0)}%</span>}
          </div>
        );
      })}
    </div>
  );
}

function AxisLegend({ axes, probs }: { axes: AxisInfo[]; probs: number[] }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill,minmax(220px,1fr))", gap: "6px 10px", fontSize: 11, color: "var(--dim)", marginTop: 10 }}>
      {axes.map((a, i) => (
        <div key={i} style={{ display: "flex", alignItems: "flex-start", gap: 8 }}>
          <span style={{ width: 10, height: 10, borderRadius: 2, background: COLORS[i % COLORS.length], flexShrink: 0, marginTop: 2 }} />
          <span style={{ color: "var(--text-dim)" }}>
            {a.label_ja ?? `軸${i + 1}`} — <strong style={{ color: "var(--text)" }}>{((probs[i] ?? 0) * 100).toFixed(1)}%</strong>
          </span>
        </div>
      ))}
    </div>
  );
}

/* ── PaceShape ── */
function PaceStr(ps: { grind_index?: number; burst_index?: number; lap_evenness?: number } | undefined) {
  if (!ps) return "—";
  const g = ps.grind_index != null ? (ps.grind_index * 100).toFixed(0) : "—";
  const b = ps.burst_index != null ? (ps.burst_index * 100).toFixed(0) : "—";
  const e = ps.lap_evenness != null ? (ps.lap_evenness * 100).toFixed(0) : "—";
  return `${g}/${b}/${e}`;
}

/* ── メインコンポーネント ── */
export default function RaceQualityPage() {
  const [selectedRaceId, setSelectedRaceId] = useState("");
  const [selectedDate, setSelectedDate] = useState("");
  const [dayData, setDayData] = useState<DayData | null>(null);
  const [singleData, setSingleData] = useState<SingleData | null>(null);
  const [loadingDay, setLoadingDay] = useState(false);
  const [loadingSingle, setLoadingSingle] = useState(false);
  const [dayError, setDayError] = useState("");
  const [singleError, setSingleError] = useState("");

  const handleRaceSelect = useCallback((id: string, _label: string) => {
    setSelectedRaceId(id);
    setSingleData(null);
    setSingleError("");
    if (USE_MOCK) {
      setSingleData(getMockRaceQuality(id) as unknown as SingleData);
    }
  }, []);

  async function loadDay() {
    setLoadingDay(true);
    setDayError("");
    setDayData(null);
    if (USE_MOCK) {
      await new Promise((r) => setTimeout(r, 300));
      const date = selectedDate || new Date().toISOString().slice(0, 10);
      setDayData(getMockRaceQualityDay(date) as unknown as DayData);
      setLoadingDay(false);
      return;
    }
    if (!selectedDate) { setLoadingDay(false); return; }
    try {
      const res = await fetch(`/api/race-quality/day?date=${encodeURIComponent(selectedDate)}`);
      const j = await res.json();
      if (!res.ok) throw new Error(j.error ?? res.statusText);
      setDayData(j);
    } catch (e: unknown) {
      setDayError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoadingDay(false);
    }
  }

  async function runSingle() {
    if (!selectedRaceId) return;
    setLoadingSingle(true);
    setSingleError("");
    setSingleData(null);
    if (USE_MOCK) {
      await new Promise((r) => setTimeout(r, 300));
      setSingleData(getMockRaceQuality(selectedRaceId) as unknown as SingleData);
      setLoadingSingle(false);
      return;
    }
    try {
      const res = await fetch(`/api/race-quality/race?race_id=${encodeURIComponent(selectedRaceId)}`);
      const j = await res.json();
      if (!res.ok) throw new Error(j.error ?? res.statusText);
      setSingleData(j);
    } catch (e: unknown) {
      setSingleError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoadingSingle(false);
    }
  }

  const panelStyle: React.CSSProperties = {
    background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 12, padding: "16px 18px", marginBottom: 16,
  };
  const h2Style: React.CSSProperties = {
    fontSize: 11, textTransform: "uppercase", letterSpacing: "0.16em", color: "var(--text-dim)", marginBottom: 12, fontWeight: 600,
  };

  const axes0 = dayData?.day_summary?.axes ?? [];
  const probs0 = dayData?.day_summary?.probs ?? [];

  return (
    <div style={{ minHeight: "100vh", background: "var(--bg)", color: "var(--text)" }}>
      {/* ヘッダー */}
      <div style={{ background: "var(--surface)", borderBottom: "1px solid var(--border)", padding: "16px 24px" }}>
        <h1 style={{ fontSize: 20, fontWeight: 700, color: "#f0f6fc", marginBottom: 6 }}>📊 レース質分析</h1>
        <p style={{ fontSize: 12, color: "var(--text-dim)", lineHeight: 1.68, maxWidth: 900 }}>
          <strong style={{ color: "var(--text)" }}>目的:</strong> 開催日・各レースが「どのタイプの馬に寄っていたか」を確率で可視化します。
          教師ラベルを使わず、<strong style={{ color: "var(--text)" }}>血統因子・タイム指数・バロメーター・過去成績</strong>から作った 8 タイプの馬側スコアと
          <strong style={{ color: "var(--text)" }}>実着順</strong>を NNLS で整合させます。
        </p>
      </div>

      {/* 日付/会場/レース選択 */}
      <RacePicker
        onRaceSelect={handleRaceSelect}
        analyzeLabel="1レースを分析"
        onAnalyze={runSingle}
        analyzing={loadingSingle}
        statusMsg={loadingSingle ? "分析中…" : singleError || ""}
        extraControls={
          <button
            style={{ background: "transparent", border: "1px solid rgba(167,139,250,0.35)", color: "var(--purple)", padding: "7px 14px", borderRadius: 8, fontSize: 12, fontWeight: 600, cursor: loadingDay ? "not-allowed" : "pointer", opacity: loadingDay ? 0.5 : 1 }}
            disabled={loadingDay}
            onClick={() => {
              const dateEl = document.querySelector<HTMLSelectElement>('[data-date-select]');
              if (dateEl) setSelectedDate(dateEl.value);
              loadDay();
            }}
          >
            {loadingDay ? "読み込み中…" : "開催日まとめを読み込み"}
          </button>
        }
      />

      <div style={{ maxWidth: 1220, margin: "0 auto", padding: "20px 16px 56px" }}>
        {/* 1レース結果 */}
        {(singleData || singleError) && (
          <div style={panelStyle}>
            <h2 style={h2Style}>1レース分析結果</h2>
            {singleError && <p style={{ color: "var(--err)", fontSize: 12 }}>⚠️ {singleError}</p>}
            {singleData && (
              <>
                <div style={{ marginBottom: 12 }}>
                  <a href={`/race/${singleData.race_id}`} style={{ color: "var(--teal)", fontWeight: 700, fontSize: 14, textDecoration: "none" }}>
                    {singleData.race_name ?? singleData.race_id}
                  </a>
                  <span style={{ fontSize: 11, color: "var(--text-dim)", marginLeft: 12 }}>
                    {singleData.n_runners}頭 / R²={singleData.r2_fit}
                  </span>
                </div>
                {singleData.axes && singleData.probs && (
                  <>
                    <StackedBar probs={singleData.probs} labels={singleData.axes.map((a) => a.label_ja ?? "")} height={48} />
                    <AxisLegend axes={singleData.axes} probs={singleData.probs} />
                  </>
                )}
              </>
            )}
          </div>
        )}

        {/* 開催日エラー */}
        {dayError && (
          <div style={{ ...panelStyle, border: "1px solid rgba(239,68,68,0.3)" }}>
            <p style={{ color: "var(--err)", fontSize: 13 }}>⚠️ {dayError}</p>
          </div>
        )}

        {/* 開催日まとめ */}
        {dayData && (
          <>
            {/* 日全体の積み上げバー */}
            {axes0.length > 0 && probs0.length > 0 && (
              <div style={panelStyle}>
                <h2 style={h2Style}>その日の全体（√頭数加重平均）</h2>
                <div style={{ marginBottom: 8, fontSize: 12, color: "var(--text-dim)" }}>
                  解析 {dayData.day_summary?.n_races}R · 日付 {dayData.date}
                </div>
                <StackedBar probs={probs0} labels={axes0.map((a) => a.label_ja ?? "")} height={40} />
                <AxisLegend axes={axes0} probs={probs0} />
              </div>
            )}

            {/* セグメント別 */}
            {dayData.by_segment && Object.keys(dayData.by_segment).length > 0 && (
              <div style={panelStyle}>
                <h2 style={h2Style}>距離・芝/ダ別（セグメント別平均）</h2>
                <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                  {Object.entries(dayData.by_segment)
                    .filter(([, s]) => s?.n_races)
                    .sort(([a], [b]) => a.localeCompare(b))
                    .map(([key, seg]) => {
                      const segAxes = seg.axes ?? axes0;
                      const segProbs = seg.probs ?? [];
                      return (
                        <div key={key}>
                          <div style={{ fontSize: 13, fontWeight: 600, color: "var(--text)", marginBottom: 6 }}>
                            {key} <span style={{ fontSize: 11, color: "var(--text-dim)" }}>({seg.n_races}R)</span>
                          </div>
                          {segProbs.length > 0 && (
                            <StackedBar probs={segProbs} labels={segAxes.map((a) => a.label_ja ?? "")} height={26} />
                          )}
                        </div>
                      );
                    })}
                </div>
              </div>
            )}

            {/* レース一覧 */}
            {dayData.races && dayData.races.length > 0 && (
              <div style={panelStyle}>
                <h2 style={h2Style}>レース一覧</h2>
                <p style={{ fontSize: 11, color: "var(--text-dim)", marginBottom: 8 }}>
                  <strong>消</strong>=消耗寄り / <strong>溜</strong>=溜め瞬発寄り / <strong>均</strong>=均速リズム（消/溜/均 0〜100目安）
                </p>
                <div style={{ overflowX: "auto" }}>
                  <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                    <thead>
                      <tr>
                        {["条件帯", "場所", "距離・馬場", "消/溜/均", "レース", "頭", "R²", "上位傾向", "分布"].map((h) => (
                          <th key={h} style={{ padding: "8px 10px", textAlign: "left", color: "var(--text-dim)", fontWeight: 600, borderBottom: "1px solid var(--border)", whiteSpace: "nowrap", fontSize: 11 }}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {dayData.races.map((rc) => {
                        const rcAxes = rc.axes ?? axes0;
                        const rcProbs = rc.probs ?? [];
                        const top3 = [...rcAxes]
                          .map((a, i) => ({ label: a.label_ja ?? `軸${i + 1}`, p: rcProbs[i] ?? 0 }))
                          .sort((a, b) => b.p - a.p)
                          .slice(0, 3)
                          .map((a) => a.label)
                          .join(" > ");
                        return (
                          <tr key={rc.race_id} style={{ borderBottom: "1px solid rgba(36,48,73,0.4)" }}>
                            <td style={{ padding: "8px 10px", fontSize: 11, whiteSpace: "nowrap", color: "var(--text-dim)" }}>{rc.segment_key ?? "—"}</td>
                            <td style={{ padding: "8px 10px" }}>{rc.venue ?? "—"}</td>
                            <td style={{ padding: "8px 10px", fontSize: 11 }}>
                              {rc.distance}{rc.surface ? ` ${rc.surface}` : ""}
                              {rc.track_condition && <span style={{ display: "block", fontSize: 10, color: "var(--text-dim)" }}>{rc.track_condition}</span>}
                            </td>
                            <td style={{ padding: "8px 10px", fontSize: 11, color: "var(--text-dim)", fontFamily: "monospace" }}>{PaceStr(rc.pace_shape)}</td>
                            <td style={{ padding: "8px 10px" }}>
                              <a href={`/race/${rc.race_id}`} style={{ color: "var(--teal)", textDecoration: "none" }}>
                                {rc.race_name ?? rc.race_id}
                              </a>
                            </td>
                            <td style={{ padding: "8px 10px", textAlign: "center" }}>{rc.n_runners ?? "—"}</td>
                            <td style={{ padding: "8px 10px", textAlign: "center", fontSize: 11 }}>{rc.r2_fit ?? "—"}</td>
                            <td style={{ padding: "8px 10px", fontSize: 11, color: "var(--text)" }}>{top3 || "—"}</td>
                            <td style={{ padding: "8px 10px", minWidth: 100 }}>
                              {rcProbs.length > 0 && (
                                <StackedBar probs={rcProbs} labels={rcAxes.map((a) => a.label_ja ?? "")} height={16} />
                              )}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </>
        )}

        {/* 初期プレースホルダー */}
        {!dayData && !dayError && !singleData && !singleError && !loadingDay && !loadingSingle && (
          <div style={{ textAlign: "center", padding: "60px 0", color: "var(--text-dim)" }}>
            <div style={{ fontSize: 40, marginBottom: 16 }}>📊</div>
            <p style={{ fontSize: 14 }}>開催日・会場・レースを選んで「1レースを分析」または「開催日まとめを読み込み」を押してください</p>
          </div>
        )}

        {(loadingDay || loadingSingle) && (
          <div style={{ textAlign: "center", padding: "60px 0", color: "var(--text-dim)" }}>
            <div style={{ width: 40, height: 40, border: "3px solid var(--border)", borderTopColor: "var(--teal)", borderRadius: "50%", animation: "spin 1s linear infinite", margin: "0 auto 16px" }} />
            <p style={{ fontSize: 14 }}>{loadingDay ? "全日計算中…" : "分析中…"}</p>
            <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
          </div>
        )}
      </div>
    </div>
  );
}
