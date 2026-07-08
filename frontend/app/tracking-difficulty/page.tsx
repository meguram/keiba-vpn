"use client";

import { useState, useCallback } from "react";
import { RacePicker } from "@/components/RacePicker";
import { USE_MOCK, getMockTdData, getMockRaceQuality } from "@/lib/mock";

/* ── 型定義 ── */
type TdEntry = {
  horse_number: number;
  horse_id: string;
  horse_name: string;
  bracket_number?: number;
  tracking_difficulty?: {
    ease_score?: number;
    ease_pct?: number;
    ease_label?: string;
    flow_position?: string;
    flow_sub?: string;
    t1f_norm?: number;
    expected_last3f?: { seconds?: number; delta_sec?: number; rank?: number; label?: string };
  };
  profile?: { style?: string; style_jra?: string };
  prev_race?: Record<string, unknown>;
  position_flow?: Record<string, unknown>;
};

type TdData = {
  race_name?: string;
  race_id?: string;
  entries?: TdEntry[];
  pace_prediction?: {
    pace_type?: string;
    pace_comment?: string;
    fps?: Record<string, unknown>;
  };
  error?: string;
};

type RaceQualityData = {
  axes?: { label_ja?: string }[];
  probs?: number[];
  r2_fit?: number | string;
  n_runners?: number;
  pace_shape?: { grind_index?: number; burst_index?: number; lap_evenness?: number };
  error?: string;
};

/* ── ユーティリティ ── */
function easeColor(pct: number | null | undefined): string {
  if (pct == null) return "var(--text-dim)";
  if (pct >= 75) return "var(--ok)";
  if (pct >= 60) return "var(--accent)";
  if (pct >= 45) return "var(--text-dim)";
  if (pct >= 30) return "var(--warn)";
  return "var(--err)";
}

function easeLabel(label: string | undefined): { bg: string; color: string } {
  if (label === "非常に楽") return { bg: "rgba(34,197,94,0.15)", color: "var(--ok)" };
  if (label === "楽") return { bg: "rgba(59,130,246,0.15)", color: "var(--accent)" };
  if (label === "普通") return { bg: "rgba(107,125,149,0.1)", color: "var(--text-dim)" };
  if (label === "やや困難") return { bg: "rgba(245,158,11,0.15)", color: "var(--warn)" };
  return { bg: "rgba(239,68,68,0.15)", color: "var(--err)" };
}

const WAKU_COLORS: [string, string][] = [
  ["#fff", "#555"], ["#000", "#fff"], ["#c0392b", "#fff"], ["#2471a3", "#fff"],
  ["#f1c40f", "#333"], ["#27ae60", "#fff"], ["#e67e22", "#fff"], ["#f39c12", "#333"],
];

function wakuStyle(bn: number): React.CSSProperties {
  const [bg, color] = WAKU_COLORS[(bn - 1) % 8] ?? ["#888", "#fff"];
  return { background: bg, color, width: 26, height: 26, borderRadius: "50%", display: "inline-flex", alignItems: "center", justifyContent: "center", fontSize: 11, fontWeight: 800 };
}

function styleChip(style: string) {
  const map: Record<string, { bg: string; color: string }> = {
    逃げ: { bg: "rgba(239,68,68,0.15)", color: "var(--err)" },
    先行: { bg: "rgba(245,158,11,0.15)", color: "var(--warn)" },
    差し: { bg: "rgba(59,130,246,0.15)", color: "var(--accent)" },
    追い込み: { bg: "rgba(167,139,250,0.15)", color: "#a78bfa" },
  };
  const s = map[style] ?? { bg: "rgba(107,125,149,0.12)", color: "var(--text-dim)" };
  return s;
}

/* ── レース質分析 ユーティリティ ── */
const RQ_COLORS = [
  "#2dd4bf", "#a78bfa", "#f472b6", "#fb923c",
  "#4ade80", "#facc15", "#94a3b8", "#f87171", "#475569",
];

function RqStackedBar({ probs, labels }: { probs: number[]; labels: string[] }) {
  const total = probs.reduce((s, v) => s + Math.max(0, v), 0) || 1;
  return (
    <div style={{ display: "flex", height: 40, borderRadius: 8, overflow: "hidden", boxShadow: "inset 0 0 0 1px rgba(42,49,66,0.8)" }}>
      {probs.map((v, i) => {
        const pct = (Math.max(0, v) / total) * 100;
        if (pct < 0.5) return null;
        return (
          <div
            key={i}
            style={{ flex: `0 0 ${pct}%`, background: RQ_COLORS[i % RQ_COLORS.length], display: "flex", alignItems: "center", justifyContent: "center" }}
            title={`${labels[i] ?? `軸${i + 1}`}: ${pct.toFixed(1)}%`}
          >
            {pct >= 8 && <span style={{ fontSize: 9, fontWeight: 700, color: "rgba(0,0,0,0.75)" }}>{pct.toFixed(0)}%</span>}
          </div>
        );
      })}
    </div>
  );
}

function RqAxisLegend({ axes, probs }: { axes: { label_ja?: string }[]; probs: number[] }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill,minmax(200px,1fr))", gap: "5px 10px", fontSize: 11, marginTop: 10 }}>
      {axes.map((a, i) => (
        <div key={i} style={{ display: "flex", alignItems: "flex-start", gap: 6 }}>
          <span style={{ width: 9, height: 9, borderRadius: 2, background: RQ_COLORS[i % RQ_COLORS.length], flexShrink: 0, marginTop: 2 }} />
          <span style={{ color: "var(--text-dim)" }}>
            {a.label_ja ?? `軸${i + 1}`} — <strong style={{ color: "var(--text)" }}>{((probs[i] ?? 0) * 100).toFixed(1)}%</strong>
          </span>
        </div>
      ))}
    </div>
  );
}

function paceShapeStr(ps: { grind_index?: number; burst_index?: number; lap_evenness?: number }): string {
  const g = ps.grind_index != null ? (ps.grind_index * 100).toFixed(0) : "—";
  const b = ps.burst_index != null ? (ps.burst_index * 100).toFixed(0) : "—";
  const e = ps.lap_evenness != null ? (ps.lap_evenness * 100).toFixed(0) : "—";
  return `${g} / ${b} / ${e}`;
}

/* ── メインコンポーネント ── */
export default function TrackingDifficultyPage() {
  const [selectedRaceId, setSelectedRaceId] = useState("");
  const [raceLabel, setRaceLabel] = useState("");
  const [analyzing, setAnalyzing] = useState(false);
  const [status, setStatus] = useState("");
  const [data, setData] = useState<TdData | null>(null);
  const [rqData, setRqData] = useState<RaceQualityData | null>(null);
  const [sortMode, setSortMode] = useState<"predicted" | "horse_number" | "ease">("predicted");
  const [error, setError] = useState("");

  const handleRaceSelect = useCallback((id: string, label: string) => {
    setSelectedRaceId(id);
    setRaceLabel(label);
    setData(null);
    setRqData(null);
    setError("");
    setStatus("");
    if (USE_MOCK) analyze(id);
  }, []);

  async function analyze(raceId: string) {
    setAnalyzing(true);
    setError("");
    setStatus("分析中…");
    setData(null);
    setRqData(null);
    if (USE_MOCK) {
      await new Promise((r) => setTimeout(r, 500));
      setData(getMockTdData(raceId) as TdData);
      setRqData(getMockRaceQuality(raceId) as RaceQualityData);
      setStatus("完了");
      setAnalyzing(false);
      return;
    }
    try {
      const [tdRes, rqRes] = await Promise.all([
        fetch(`/api/race/${raceId}/tracking-difficulty`),
        fetch(`/api/race-quality/race?race_id=${encodeURIComponent(raceId)}`),
      ]);
      if (!tdRes.ok) {
        let msg = `HTTP ${tdRes.status}`;
        try {
          const body = await tdRes.json();
          if (body.status === "not_precomputed") msg = "このレースの追走難度はまだ事前計算されていません。";
          else if (body.error) msg = body.error;
        } catch { /* ignore */ }
        throw new Error(msg);
      }
      const result: TdData = await tdRes.json();
      if (result.error && (!result.entries?.length)) throw new Error(result.error);
      setData(result);
      if (rqRes.ok) {
        const rq: RaceQualityData = await rqRes.json();
        if (!rq.error) setRqData(rq);
      }
      setStatus("完了");
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg);
      setStatus(`エラー: ${msg}`);
    } finally {
      setAnalyzing(false);
    }
  }

  /* ── 並び替え ── */
  const entries = (data?.entries ?? []).slice().sort((a, b) => {
    if (sortMode === "horse_number") return (a.horse_number ?? 0) - (b.horse_number ?? 0);
    if (sortMode === "ease") {
      const pa = a.tracking_difficulty?.ease_pct ?? 50;
      const pb = b.tracking_difficulty?.ease_pct ?? 50;
      return pb - pa;
    }
    const pa = a.tracking_difficulty?.ease_pct ?? 50;
    const pb = b.tracking_difficulty?.ease_pct ?? 50;
    return pb - pa;
  });

  return (
    <div style={{ minHeight: "100vh", background: "var(--bg)", color: "var(--text)" }}>
      {/* ページヘッダー */}
      <div style={{ background: "var(--surface)", borderBottom: "1px solid var(--border)", padding: "16px 24px", display: "flex", alignItems: "center", gap: 16 }}>
        <div>
          <h1 style={{ fontSize: 16, fontWeight: 700, color: "#fff" }}>追走難度分析</h1>
          <span style={{ fontSize: 12, color: "var(--text-dim)" }}>
            前走T1F・位置取り分布・コース比較・ペース予測から追走難度（脚がたまる度）を算出
          </span>
        </div>
      </div>

      {/* コントロール */}
      <RacePicker
        onRaceSelect={handleRaceSelect}
        onAnalyze={analyze}
        analyzeLabel="分析実行"
        analyzing={analyzing}
        statusMsg={status}
        extraControls={
          selectedRaceId ? (
            <a
              href={`/race/${selectedRaceId}`}
              style={{ fontSize: 12, color: "var(--accent)", textDecoration: "none", border: "1px solid var(--accent)", padding: "5px 12px", borderRadius: 5 }}
            >
              📊 レース詳細
            </a>
          ) : undefined
        }
      />

      {/* メインコンテンツ */}
      <div style={{ padding: "20px 24px", maxWidth: 1400, margin: "0 auto" }}>
        {!data && !error && (
          <div style={{ textAlign: "center", padding: "60px 0", color: "var(--text-dim)", fontSize: 14 }}>
            レースを選択して「分析実行」をクリックしてください
          </div>
        )}

        {error && (
          <div style={{ background: "rgba(239,68,68,0.1)", border: "1px solid rgba(239,68,68,0.3)", borderRadius: 8, padding: "14px 18px", color: "var(--err)", fontSize: 13 }}>
            ⚠️ {error}
          </div>
        )}

        {analyzing && (
          <div style={{ textAlign: "center", padding: "60px 0", color: "var(--text-dim)" }}>
            <div style={{ fontSize: 32, marginBottom: 12 }}>⏳</div>
            <div style={{ fontSize: 14 }}>追走難度を計算中…</div>
            <div style={{ fontSize: 12, marginTop: 8, color: "var(--text-dim)" }}>
              出馬表・戦績の取得 → 追走難度・ペース・位置取りの計算 → 結果の表示
            </div>
          </div>
        )}

        {data && entries.length > 0 && (
          <>
            {/* レースメタ */}
            <div style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 8, padding: "14px 18px", marginBottom: 20, display: "flex", gap: 20, flexWrap: "wrap", alignItems: "center" }}>
              <span style={{ fontSize: 16, fontWeight: 700, color: "#fff" }}>{data.race_name ?? raceLabel}</span>
              <a href={`/race/${selectedRaceId}`} style={{ marginLeft: "auto", display: "inline-flex", alignItems: "center", gap: 6, fontSize: 12, fontWeight: 700, color: "#fff", padding: "6px 14px", borderRadius: 6, background: "linear-gradient(135deg,rgba(34,197,94,0.35),rgba(34,197,94,0.15))", border: "1px solid rgba(34,197,94,0.45)", textDecoration: "none" }}>
                📊 レース結果・AI予測
              </a>
            </div>

            {/* ゲート配置 × 追走容易度 */}
            <div style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 8, padding: "16px 18px", marginBottom: 20 }}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12, flexWrap: "wrap", gap: 8 }}>
                <h3 style={{ fontSize: 13, fontWeight: 600, color: "#a78bfa" }}>ゲート配置 × 追走容易度</h3>
                <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12, color: "var(--text-dim)" }}>
                  並び替え:
                  <select
                    value={sortMode}
                    onChange={(e) => setSortMode(e.target.value as typeof sortMode)}
                    style={{ background: "var(--surface2)", border: "1px solid var(--border)", color: "var(--text)", padding: "5px 10px", borderRadius: 5, fontSize: 12, cursor: "pointer" }}
                  >
                    <option value="predicted">追走容易度順</option>
                    <option value="horse_number">馬番順</option>
                    <option value="ease">スコア順</option>
                  </select>
                </div>
              </div>
              <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                  <thead>
                    <tr>
                      {["枠", "番", "馬名", "脚質", "想定位置", "追走容易度"].map((h) => (
                        <th key={h} style={{ textAlign: "left", padding: "8px 10px", color: "var(--text-dim)", fontWeight: 600, borderBottom: "1px solid var(--border)", fontSize: 11, whiteSpace: "nowrap" }}>{h}</th>
                      ))}
                      <th style={{ padding: "8px 10px", color: "var(--text-dim)", fontWeight: 600, borderBottom: "1px solid var(--border)", fontSize: 11 }}>容易度スコア</th>
                      <th style={{ padding: "8px 10px", color: "var(--text-dim)", fontWeight: 600, borderBottom: "1px solid var(--border)", fontSize: 11 }}>ラベル</th>
                    </tr>
                  </thead>
                  <tbody>
                    {entries.map((e) => {
                      const td = e.tracking_difficulty ?? {};
                      const pct = td.ease_pct ?? 50;
                      const color = easeColor(pct);
                      const el = easeLabel(td.ease_label);
                      const sty = styleChip(e.profile?.style ?? "");
                      return (
                        <tr key={e.horse_id ?? e.horse_number} style={{ cursor: "pointer" }}>
                          <td style={{ padding: "9px 10px", borderBottom: "1px solid rgba(36,48,73,0.45)", verticalAlign: "middle" }}>
                            <span style={wakuStyle(e.bracket_number ?? e.horse_number)}>{e.bracket_number ?? e.horse_number}</span>
                          </td>
                          <td style={{ padding: "9px 10px", borderBottom: "1px solid rgba(36,48,73,0.45)", textAlign: "center", verticalAlign: "middle" }}>
                            <span style={{ fontWeight: 800, fontSize: 13, color: "#7dd3fc" }}>{e.horse_number}</span>
                          </td>
                          <td style={{ padding: "9px 10px", borderBottom: "1px solid rgba(36,48,73,0.45)", fontWeight: 600, color: "#fff", verticalAlign: "middle", whiteSpace: "nowrap" }}>
                            <a href={`/race/${selectedRaceId}`} style={{ color: "#fff", textDecoration: "none" }}>{e.horse_name}</a>
                          </td>
                          <td style={{ padding: "9px 10px", borderBottom: "1px solid rgba(36,48,73,0.45)", verticalAlign: "middle" }}>
                            {e.profile?.style && (
                              <span style={{ fontSize: 10, padding: "2px 6px", borderRadius: 3, fontWeight: 600, display: "inline-block", ...sty }}>{e.profile.style}</span>
                            )}
                          </td>
                          <td style={{ padding: "9px 10px", borderBottom: "1px solid rgba(36,48,73,0.45)", verticalAlign: "middle" }}>
                            <div style={{ fontWeight: 800, color: "#7dd3fc", fontSize: 13 }}>{td.flow_position ?? "—"}</div>
                            {td.flow_sub && <div style={{ fontSize: 10, color: "var(--text-dim)", marginTop: 2 }}>{td.flow_sub}</div>}
                          </td>
                          <td style={{ padding: "9px 10px", borderBottom: "1px solid rgba(36,48,73,0.45)", verticalAlign: "middle", minWidth: 160 }}>
                            <div style={{ position: "relative", height: 22, background: "var(--surface2)", border: "1px solid var(--border)", borderRadius: 4, overflow: "hidden" }}>
                              <div style={{ height: "100%", width: `${Math.min(100, Math.max(2, pct))}%`, background: color, borderRadius: "3px 0 0 3px", transition: "width 0.35s" }} />
                              <span style={{ position: "absolute", right: 8, top: "50%", transform: "translateY(-50%)", fontSize: 11, fontWeight: 800, color: "#fff", textShadow: "0 1px 2px rgba(0,0,0,0.6)" }}>{pct.toFixed(0)}%</span>
                            </div>
                          </td>
                          <td style={{ padding: "9px 10px", borderBottom: "1px solid rgba(36,48,73,0.45)", verticalAlign: "middle" }}>
                            {td.ease_label && (
                              <span style={{ fontSize: 12, fontWeight: 800, padding: "4px 10px", borderRadius: 6, ...el }}>{td.ease_label}</span>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>

            {/* ペース予測 */}
            {data.pace_prediction && (
              <div style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 8, padding: "14px 18px", marginBottom: 20 }}>
                <h3 style={{ fontSize: 13, fontWeight: 600, color: "#7dd3fc", marginBottom: 10 }}>ペース予測</h3>
                <div style={{ display: "flex", gap: 16, flexWrap: "wrap", alignItems: "center" }}>
                  {data.pace_prediction.pace_type && (
                    <span style={{ fontWeight: 800, fontSize: 15, padding: "5px 14px", borderRadius: 6, background: "rgba(59,130,246,0.15)", color: "var(--accent)" }}>
                      {data.pace_prediction.pace_type}
                    </span>
                  )}
                  {data.pace_prediction.pace_comment && (
                    <span style={{ fontSize: 13, color: "var(--text-dim)" }}>{data.pace_prediction.pace_comment}</span>
                  )}
                </div>
              </div>
            )}

            {/* レース質分析 */}
            {rqData && rqData.axes && rqData.probs && (
              <div style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 8, padding: "14px 18px", marginBottom: 20 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 12, flexWrap: "wrap" }}>
                  <h3 style={{ fontSize: 13, fontWeight: 600, color: "#2dd4bf" }}>レース質分析</h3>
                  <span style={{ fontSize: 11, color: "var(--text-dim)" }}>
                    過去の統計 × 出走メンバー × ペース予測から想定されるレース傾向
                  </span>
                  <div style={{ marginLeft: "auto", display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
                    {rqData.pace_shape && (
                      <span style={{ fontSize: 11, color: "var(--text-dim)", fontFamily: "monospace" }}>
                        <span style={{ color: "var(--text)" }}>消/溜/均</span> {paceShapeStr(rqData.pace_shape)}
                      </span>
                    )}
                    {rqData.r2_fit != null && (
                      <span style={{ fontSize: 11, color: "var(--text-dim)" }}>
                        R²={typeof rqData.r2_fit === "number" ? rqData.r2_fit.toFixed(2) : rqData.r2_fit}
                      </span>
                    )}
                    {rqData.n_runners != null && (
                      <span style={{ fontSize: 11, color: "var(--text-dim)" }}>{rqData.n_runners}頭</span>
                    )}
                  </div>
                </div>
                <RqStackedBar probs={rqData.probs} labels={rqData.axes.map((a) => a.label_ja ?? "")} />
                <RqAxisLegend axes={rqData.axes} probs={rqData.probs} />
              </div>
            )}

            {/* 馬別カードグリッド */}
            <div style={{ marginBottom: 24 }}>
              <h3 style={{ fontSize: 14, fontWeight: 700, color: "#fff", marginBottom: 14 }}>馬別詳細カード</h3>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill,minmax(320px,1fr))", gap: 14 }}>
                {entries.map((e) => {
                  const td = e.tracking_difficulty ?? {};
                  const pct = td.ease_pct ?? 50;
                  const color = easeColor(pct);
                  const el = easeLabel(td.ease_label);
                  const sty = styleChip(e.profile?.style ?? "");
                  const l3f = td.expected_last3f;
                  return (
                    <div key={e.horse_id ?? e.horse_number} style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 10, overflow: "hidden" }}>
                      <div style={{ padding: "12px 14px", display: "flex", alignItems: "center", gap: 10, borderBottom: "1px solid var(--border)" }}>
                        <span style={wakuStyle(e.bracket_number ?? e.horse_number)}>{e.bracket_number ?? e.horse_number}</span>
                        <div style={{ flex: 1, minWidth: 0 }}>
                          <div style={{ fontSize: 14, fontWeight: 700, color: "#fff", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{e.horse_name}</div>
                          {e.profile?.style && (
                            <span style={{ fontSize: 10, padding: "2px 6px", borderRadius: 3, fontWeight: 600, display: "inline-block", marginTop: 2, ...sty }}>{e.profile.style}</span>
                          )}
                        </div>
                        <div style={{ fontSize: 18, fontWeight: 800, padding: "4px 12px", borderRadius: 8, textAlign: "center", minWidth: 60, ...el }}>
                          {td.ease_label ?? "—"}
                        </div>
                      </div>
                      <div style={{ padding: "12px 14px" }}>
                        {/* 追走容易度バー */}
                        <div style={{ marginBottom: 10 }}>
                          <div style={{ fontSize: 10, fontWeight: 600, color: "var(--text-dim)", marginBottom: 5, textTransform: "uppercase", letterSpacing: "0.5px" }}>追走容易度</div>
                          <div style={{ position: "relative", height: 18, background: "rgba(36,48,73,0.5)", borderRadius: 3, overflow: "hidden" }}>
                            <div style={{ height: "100%", width: `${Math.min(100, Math.max(2, pct))}%`, background: color, borderRadius: 3, transition: "width 0.6s" }} />
                          </div>
                          <div style={{ fontSize: 11, fontWeight: 700, color, marginTop: 4 }}>{pct.toFixed(1)}%</div>
                        </div>
                        {/* 想定位置 */}
                        {td.flow_position && (
                          <div style={{ marginBottom: 8 }}>
                            <div style={{ fontSize: 10, fontWeight: 600, color: "var(--text-dim)", marginBottom: 3, textTransform: "uppercase", letterSpacing: "0.5px" }}>想定位置</div>
                            <div style={{ fontSize: 12, fontWeight: 700, color: "#7dd3fc" }}>{td.flow_position} {td.flow_sub && <span style={{ fontSize: 10, color: "var(--text-dim)", fontWeight: 400 }}>{td.flow_sub}</span>}</div>
                          </div>
                        )}
                        {/* 想定上り3F */}
                        {l3f?.seconds != null && l3f.seconds > 0 && (
                          <div style={{ marginBottom: 8 }}>
                            <div style={{ fontSize: 10, fontWeight: 600, color: "var(--text-dim)", marginBottom: 3, textTransform: "uppercase", letterSpacing: "0.5px" }}>想定上り3F</div>
                            <div style={{ fontSize: 12, fontWeight: 800, color: (l3f.delta_sec ?? 0) <= -0.1 ? "var(--ok)" : (l3f.delta_sec ?? 0) >= 0.3 ? "var(--err)" : "var(--text-dim)" }}>
                              {Number(l3f.seconds).toFixed(1)}秒 {l3f.rank != null && <span style={{ fontSize: 10, color: "var(--text-dim)", fontWeight: 400 }}>（{l3f.rank}位）</span>}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
