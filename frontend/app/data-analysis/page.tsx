"use client";

import { useState, useCallback, useMemo } from "react";

/* ── 型 ── */
type AnalysisType = "distribution" | "scatter" | "ranking" | "time_series";
type AggFunc = "avg" | "count" | "sum" | "min" | "max";

type ResultRow = {
  x?: string | number;
  y?: number | null;
  count?: number;
  label?: string;
  group?: string;
};

type AnalysisResult = {
  analysis_type: string;
  x_field: string;
  y_field: string;
  y_agg: string;
  group_by?: string | null;
  rows: ResultRow[];
  total_rows: number;
  meta: { x_label: string; y_label: string; group_label?: string | null };
};

/* ── フィールド定義 ── */
const NUMERIC_FIELDS = [
  { key: "finish_pos", label: "着順" },
  { key: "finish_time_sec", label: "タイム（秒）" },
  { key: "last_3f_sec", label: "上がり3F（秒）" },
  { key: "weight", label: "馬体重（kg）" },
  { key: "jockey_weight", label: "斤量" },
  { key: "win_prob", label: "AI勝率予測" },
  { key: "place_prob", label: "AI複勝率予測" },
  { key: "expected_win_roi", label: "期待ROI（単勝）" },
  { key: "expected_show_roi", label: "期待ROI（複勝）" },
];

const CATEGORICAL_FIELDS = [
  { key: "surface", label: "馬場（芝/ダート）" },
  { key: "track_condition", label: "馬場状態" },
  { key: "grade", label: "グレード" },
  { key: "venue", label: "競馬場" },
  { key: "race_class", label: "クラス" },
  { key: "pace_category", label: "ペース（H/M/S）" },
  { key: "distance_bucket", label: "距離帯（200m単位）" },
  { key: "month", label: "月" },
  { key: "year", label: "年" },
];

const AGG_OPTIONS: { value: AggFunc; label: string }[] = [
  { value: "avg", label: "平均" },
  { value: "count", label: "件数" },
  { value: "sum", label: "合計" },
  { value: "min", label: "最小" },
  { value: "max", label: "最大" },
];

const GROUP_COLORS = [
  "#2dd4bf", "#a78bfa", "#f472b6", "#fb923c", "#4ade80",
  "#facc15", "#60a5fa", "#f87171", "#e879f9", "#34d399",
];

/* ── スタイル ── */
const s = {
  label: { fontSize: 11, fontWeight: 600, color: "var(--text-dim)", display: "block", marginBottom: 4 } as React.CSSProperties,
  select: {
    width: "100%", padding: "7px 10px",
    background: "rgba(255,255,255,0.04)", border: "1px solid var(--border)",
    borderRadius: 6, color: "var(--text)", fontSize: 12.5, outline: "none",
  } as React.CSSProperties,
  input: {
    width: "100%", padding: "7px 10px",
    background: "rgba(255,255,255,0.04)", border: "1px solid var(--border)",
    borderRadius: 6, color: "var(--text)", fontSize: 12.5, outline: "none",
  } as React.CSSProperties,
};

/* ── チャートヘルパー ── */
function colorOf(group: string | undefined, keys: string[]): string {
  if (!group || !keys.length) return "#2dd4bf";
  const i = keys.indexOf(group);
  return GROUP_COLORS[(i >= 0 ? i : 0) % GROUP_COLORS.length];
}

/* ── 水平棒グラフ（分布・ランキング） ── */
function BarChart({ rows, meta, groupKeys }: {
  rows: ResultRow[];
  meta: AnalysisResult["meta"];
  groupKeys: string[];
}) {
  const hasGroup = groupKeys.length > 0;
  const xKeys = [...new Set(rows.map(r => String(r.x ?? r.label ?? "")))].slice(0, 40);
  const maxY = Math.max(...rows.map(r => r.y ?? 0), 0.0001);
  const barH = Math.max(12, Math.min(28, 400 / Math.max(1, rows.length)));
  const PAD = { left: 170, right: 90, top: 16, bottom: 36 };
  const W = 620;

  const byX: Record<string, ResultRow[]> = {};
  rows.forEach(r => {
    const k = String(r.x ?? r.label ?? "");
    if (!byX[k]) byX[k] = [];
    byX[k].push(r);
  });

  let yOff = PAD.top;
  const items: React.ReactNode[] = [];

  xKeys.forEach((xk, xi) => {
    const xRows = byX[xk] ?? [];
    const blockH = xRows.length * (barH + 2);

    items.push(
      <text key={`lbl-${xi}`}
        x={PAD.left - 8}
        y={yOff + blockH / 2 + 4}
        textAnchor="end" fill="#9ca3af" fontSize={11}>
        {xk.length > 18 ? xk.slice(0, 17) + "…" : xk}
      </text>
    );

    xRows.forEach((row, ri) => {
      const bw = Math.max(2, ((row.y ?? 0) / maxY) * (W - PAD.left - PAD.right));
      const by = yOff + ri * (barH + 2);
      const col = colorOf(row.group, groupKeys);
      items.push(
        <g key={`bar-${xi}-${ri}`}>
          <rect x={PAD.left} y={by} width={bw} height={barH} fill={col} opacity={0.85} rx={2} />
          {bw > 50 ? (
            <text x={PAD.left + bw - 5} y={by + barH / 2 + 4}
              textAnchor="end" fill="#fff" fontSize={9} fontWeight={600}>
              {(row.y ?? 0).toFixed(3)}
            </text>
          ) : (
            <text x={PAD.left + bw + 5} y={by + barH / 2 + 4}
              fill={col} fontSize={9} fontWeight={600}>
              {(row.y ?? 0).toFixed(3)}
            </text>
          )}
          {hasGroup && row.group && (
            <text x={PAD.left + bw - 5} y={by + barH / 2 + 4}
              textAnchor="end" fill="rgba(255,255,255,0.0)" fontSize={0}>
              {row.group}
            </text>
          )}
          {/* count tooltip area */}
          <title>{`${xk}${row.group ? " / " + row.group : ""}: ${(row.y ?? 0).toFixed(4)} (n=${row.count ?? 0})`}</title>
        </g>
      );
    });

    yOff += blockH + 6;
  });

  const totalH = yOff + PAD.bottom;
  const gridVals = [0, 0.25, 0.5, 0.75, 1];

  return (
    <svg viewBox={`0 0 ${W} ${totalH}`} style={{ width: "100%", display: "block" }}>
      {/* Grid */}
      {gridVals.map(f => {
        const gx = PAD.left + f * (W - PAD.left - PAD.right);
        return (
          <g key={f}>
            <line x1={gx} y1={PAD.top - 4} x2={gx} y2={totalH - PAD.bottom}
              stroke={f === 0 ? "#4b5563" : "#374151"} strokeWidth={f === 0 ? 1.5 : 1} />
            <text x={gx} y={totalH - PAD.bottom + 14}
              textAnchor="middle" fill="#6b7280" fontSize={9}>
              {(f * maxY).toFixed(f === 0 ? 0 : 3)}
            </text>
          </g>
        );
      })}
      {items}
      {/* Y-axis label */}
      <text x={W - PAD.right + 4} y={totalH - PAD.bottom + 14}
        fill="#6b7280" fontSize={9}>{meta.y_label}</text>
    </svg>
  );
}

/* ── 散布図 ── */
function ScatterChart({ rows, meta, groupKeys }: {
  rows: ResultRow[];
  meta: AnalysisResult["meta"];
  groupKeys: string[];
}) {
  const W = 620, H = 420;
  const P = { top: 20, bottom: 50, left: 56, right: 20 };
  const cW = W - P.left - P.right;
  const cH = H - P.top - P.bottom;

  const xVals = rows.map(r => r.x as number).filter(v => v != null && isFinite(v));
  const yVals = rows.map(r => r.y as number).filter(v => v != null && isFinite(v));
  if (!xVals.length) return <p style={{ color: "var(--text-dim)", padding: 24 }}>データがありません</p>;

  const minX = Math.min(...xVals), maxX = Math.max(...xVals);
  const minY = Math.min(...yVals), maxY = Math.max(...yVals);
  const rX = maxX - minX || 1, rY = maxY - minY || 1;

  const cx = (v: number) => P.left + ((v - minX) / rX) * cW;
  const cy = (v: number) => P.top + cH - ((v - minY) / rY) * cH;

  const ticks = [0, 0.25, 0.5, 0.75, 1];

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", display: "block" }}>
      {ticks.map(f => (
        <g key={`gy-${f}`}>
          <line x1={P.left} y1={P.top + f * cH} x2={W - P.right} y2={P.top + f * cH}
            stroke="#374151" strokeWidth={1} />
          <text x={P.left - 4} y={P.top + f * cH + 4} textAnchor="end" fill="#6b7280" fontSize={9}>
            {(maxY - f * rY).toFixed(3)}
          </text>
        </g>
      ))}
      {ticks.map(f => (
        <g key={`gx-${f}`}>
          <line x1={P.left + f * cW} y1={P.top} x2={P.left + f * cW} y2={P.top + cH}
            stroke="#374151" strokeWidth={1} />
          <text x={P.left + f * cW} y={P.top + cH + 14} textAnchor="middle" fill="#6b7280" fontSize={9}>
            {(minX + f * rX).toFixed(3)}
          </text>
        </g>
      ))}
      <line x1={P.left} y1={P.top} x2={P.left} y2={P.top + cH} stroke="#4b5563" strokeWidth={1.5} />
      <line x1={P.left} y1={P.top + cH} x2={W - P.right} y2={P.top + cH} stroke="#4b5563" strokeWidth={1.5} />
      {rows.slice(0, 3000).map((r, i) => (
        <circle key={i}
          cx={cx(r.x as number)} cy={cy(r.y as number)} r={3}
          fill={colorOf(r.group, groupKeys)} opacity={0.55}>
          <title>{`x=${r.x}, y=${r.y}${r.group ? `, ${r.group}` : ""}`}</title>
        </circle>
      ))}
      <text x={W / 2} y={H - 4} textAnchor="middle" fill="#9ca3af" fontSize={10}>{meta.x_label}</text>
      <text x={10} y={H / 2} textAnchor="middle" fill="#9ca3af" fontSize={10}
        transform={`rotate(-90,10,${H / 2})`}>{meta.y_label}</text>
    </svg>
  );
}

/* ── 折れ線グラフ（時系列） ── */
function LineChart({ rows, meta, groupKeys }: {
  rows: ResultRow[];
  meta: AnalysisResult["meta"];
  groupKeys: string[];
}) {
  const W = 620, H = 360;
  const P = { top: 20, bottom: 50, left: 60, right: 20 };
  const cW = W - P.left - P.right;
  const cH = H - P.top - P.bottom;
  if (!rows.length) return <p style={{ color: "var(--text-dim)", padding: 24 }}>データがありません</p>;

  const allX = [...new Set(rows.map(r => String(r.x ?? "")))].sort();
  const ys = rows.map(r => r.y ?? 0);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const rY = maxY - minY || 1;

  const cx = (xi: number) => P.left + (xi / Math.max(1, allX.length - 1)) * cW;
  const cy = (v: number) => P.top + cH - ((v - minY) / rY) * cH;

  const series: Record<string, ResultRow[]> = {};
  rows.forEach(r => {
    const k = r.group ?? "__all";
    if (!series[k]) series[k] = [];
    series[k].push(r);
  });

  const tickStep = Math.max(1, Math.ceil(allX.length / 10));

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", display: "block" }}>
      {[0, 0.25, 0.5, 0.75, 1].map(f => (
        <g key={f}>
          <line x1={P.left} y1={P.top + f * cH} x2={W - P.right} y2={P.top + f * cH}
            stroke="#374151" strokeWidth={1} />
          <text x={P.left - 4} y={P.top + f * cH + 4} textAnchor="end" fill="#6b7280" fontSize={9}>
            {(maxY - f * rY).toFixed(3)}
          </text>
        </g>
      ))}
      {allX.filter((_, i) => i % tickStep === 0).map((xk) => {
        const xi = allX.indexOf(xk);
        return (
          <text key={xk} x={cx(xi)} y={P.top + cH + 16}
            textAnchor="middle" fill="#6b7280" fontSize={8}>
            {xk}
          </text>
        );
      })}
      <line x1={P.left} y1={P.top} x2={P.left} y2={P.top + cH} stroke="#4b5563" strokeWidth={1.5} />
      <line x1={P.left} y1={P.top + cH} x2={W - P.right} y2={P.top + cH} stroke="#4b5563" strokeWidth={1.5} />
      {Object.entries(series).map(([gk, gRows]) => {
        const sorted = [...gRows].sort((a, b) => String(a.x ?? "").localeCompare(String(b.x ?? "")));
        const pts = sorted.map(r => [cx(allX.indexOf(String(r.x ?? ""))), cy(r.y ?? 0)] as [number, number]);
        if (!pts.length) return null;
        const d = pts.map((p, i) => `${i === 0 ? "M" : "L"}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" ");
        const col = colorOf(gk === "__all" ? undefined : gk, groupKeys);
        return (
          <g key={gk}>
            <path d={d} fill="none" stroke={col} strokeWidth={2} />
            {pts.map((p, i) => <circle key={i} cx={p[0]} cy={p[1]} r={3} fill={col} />)}
          </g>
        );
      })}
      <text x={W / 2} y={H - 4} textAnchor="middle" fill="#9ca3af" fontSize={10}>月</text>
      <text x={10} y={H / 2} textAnchor="middle" fill="#9ca3af" fontSize={10}
        transform={`rotate(-90,10,${H / 2})`}>{meta.y_label}</text>
    </svg>
  );
}

/* ── 凡例 ── */
function Legend({ groupKeys, label }: { groupKeys: string[]; label?: string | null }) {
  if (!groupKeys.length) return null;
  return (
    <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginTop: 12 }}>
      {label && <span style={{ fontSize: 11, color: "var(--text-dim)", marginRight: 4 }}>{label}:</span>}
      {groupKeys.map((k, i) => (
        <span key={k} style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 12 }}>
          <span style={{ width: 10, height: 10, borderRadius: "50%", background: GROUP_COLORS[i % GROUP_COLORS.length], flexShrink: 0 }} />
          {k}
        </span>
      ))}
    </div>
  );
}

/* ── データテーブル ── */
function DataTable({ rows, analysisType, meta }: {
  rows: ResultRow[];
  analysisType: string;
  meta: AnalysisResult["meta"];
}) {
  const cols = analysisType === "ranking"
    ? ["馬名", meta.y_label, "件数"]
    : analysisType === "time_series"
    ? ["期間", meta.y_label, "件数", ...(rows[0]?.group !== undefined ? ["グループ"] : [])]
    : [meta.x_label, meta.y_label, "件数", ...(rows[0]?.group !== undefined ? [meta.group_label ?? "グループ"] : [])];

  const display = rows.slice(0, 100);

  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12.5 }}>
        <thead>
          <tr style={{ borderBottom: "1px solid var(--border)" }}>
            {cols.map(c => (
              <th key={c} style={{ padding: "6px 12px", textAlign: "left", color: "var(--text-dim)", fontWeight: 600, whiteSpace: "nowrap" }}>
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {display.map((r, i) => {
            const xVal = analysisType === "ranking" ? r.label : String(r.x ?? "");
            return (
              <tr key={i} style={{ borderBottom: "1px solid rgba(36,48,73,0.4)" }}>
                <td style={{ padding: "5px 12px", color: "var(--text)" }}>{xVal}</td>
                <td style={{ padding: "5px 12px", color: "var(--accent)", fontWeight: 600 }}>
                  {r.y != null ? r.y.toFixed(4) : "—"}
                </td>
                <td style={{ padding: "5px 12px", color: "var(--text-dim)" }}>{r.count ?? "—"}</td>
                {r.group !== undefined && (
                  <td style={{ padding: "5px 12px", color: "var(--text-dim)" }}>{r.group}</td>
                )}
              </tr>
            );
          })}
        </tbody>
      </table>
      {rows.length > 100 && (
        <p style={{ fontSize: 11, color: "var(--text-dim)", padding: "8px 12px" }}>
          {rows.length}件中 100件表示
        </p>
      )}
    </div>
  );
}

/* ── メインページ ── */
export default function DataAnalysisPage() {
  const [analysisType, setAnalysisType] = useState<AnalysisType>("distribution");
  const [yField, setYField] = useState("finish_pos");
  const [xField, setXField] = useState("surface");
  const [yAgg, setYAgg] = useState<AggFunc>("avg");
  const [groupBy, setGroupBy] = useState("");
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");
  const [surface, setSurface] = useState("");
  const [distMin, setDistMin] = useState("");
  const [distMax, setDistMax] = useState("");
  const [grade, setGrade] = useState("");
  const [limit, setLimit] = useState("2000");
  const [showFilters, setShowFilters] = useState(false);

  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastQuery, setLastQuery] = useState<string>("");

  const runAnalysis = useCallback(async () => {
    setLoading(true);
    setError(null);

    const params = new URLSearchParams({ analysis_type: analysisType, y_field: yField, y_agg: yAgg, limit });
    if (analysisType !== "ranking" && analysisType !== "time_series") params.set("x_field", xField);
    if (groupBy) params.set("group_by", groupBy);
    if (dateFrom) params.set("date_from", dateFrom);
    if (dateTo) params.set("date_to", dateTo);
    if (surface) params.set("surface", surface);
    if (distMin) params.set("distance_min", distMin);
    if (distMax) params.set("distance_max", distMax);
    if (grade) params.set("grade", grade);

    const url = `/api/v1/data-analysis/query?${params}`;
    setLastQuery(url);

    try {
      const res = await fetch(url);
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.error ?? `HTTP ${res.status}`);
      }
      const data: AnalysisResult = await res.json();
      setResult(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "不明なエラー");
    } finally {
      setLoading(false);
    }
  }, [analysisType, yField, xField, yAgg, groupBy, dateFrom, dateTo, surface, distMin, distMax, grade, limit]);

  const groupKeys = useMemo(() => {
    if (!result?.rows.length || !result.group_by) return [];
    return [...new Set(result.rows.map(r => r.group ?? ""))].filter(Boolean);
  }, [result]);

  const showX = analysisType !== "ranking" && analysisType !== "time_series";
  const showAgg = true;

  return (
    <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column", background: "var(--bg)", color: "var(--text)" }}>

      {/* ページヘッダー */}
      <div style={{ background: "var(--surface)", borderBottom: "1px solid var(--border)", padding: "14px 24px" }}>
        <h1 style={{ fontSize: 16, fontWeight: 700, color: "#fff", marginBottom: 2 }}>📊 詳細データ分析</h1>
        <p style={{ fontSize: 12, color: "var(--text-dim)" }}>レース結果・AI予測・血統データを軸とした多次元分析 — Target Frontier スタイル</p>
      </div>

      <div style={{ display: "flex", flex: 1, overflow: "auto" }}>

        {/* ── 左コントロールパネル ── */}
        <aside style={{
          width: 264, flexShrink: 0, background: "var(--surface)", borderRight: "1px solid var(--border)",
          padding: "16px 14px", display: "flex", flexDirection: "column", gap: 14, overflowY: "auto",
        }}>

          {/* 分析タイプ */}
          <div>
            <span style={s.label}>分析タイプ</span>
            {(["distribution", "scatter", "ranking", "time_series"] as AnalysisType[]).map(t => {
              const labels: Record<AnalysisType, string> = {
                distribution: "分布分析", scatter: "散布図", ranking: "ランキング", time_series: "時系列",
              };
              const descs: Record<AnalysisType, string> = {
                distribution: "カテゴリ別集計値",
                scatter: "2指標の相関",
                ranking: "馬別パフォーマンス",
                time_series: "月別トレンド",
              };
              const active = analysisType === t;
              return (
                <button key={t} onClick={() => setAnalysisType(t)}
                  style={{
                    display: "block", width: "100%", textAlign: "left",
                    padding: "7px 10px", marginBottom: 3, borderRadius: 6, border: "none",
                    background: active ? "rgba(59,130,246,0.15)" : "rgba(255,255,255,0.03)",
                    color: active ? "var(--accent)" : "var(--text)",
                    cursor: "pointer",
                  }}>
                  <div style={{ fontWeight: active ? 700 : 400, fontSize: 13 }}>{labels[t]}</div>
                  <div style={{ fontSize: 10.5, color: "var(--text-dim)", marginTop: 1 }}>{descs[t]}</div>
                </button>
              );
            })}
          </div>

          <hr style={{ border: "none", borderTop: "1px solid var(--border)" }} />

          {/* Y軸 */}
          <div>
            <label style={s.label}>Y 軸（指標）</label>
            <select value={yField} onChange={e => setYField(e.target.value)} style={s.select}>
              {NUMERIC_FIELDS.map(f => <option key={f.key} value={f.key}>{f.label}</option>)}
            </select>
          </div>

          {/* 集計方法 */}
          {showAgg && (
            <div>
              <label style={s.label}>集計方法</label>
              <select value={yAgg} onChange={e => setYAgg(e.target.value as AggFunc)} style={s.select}>
                {AGG_OPTIONS.map(a => <option key={a.value} value={a.value}>{a.label}</option>)}
              </select>
            </div>
          )}

          {/* X軸 */}
          {showX && (
            <div>
              <label style={s.label}>X 軸（カテゴリ）</label>
              <select value={xField} onChange={e => setXField(e.target.value)} style={s.select}>
                {analysisType === "scatter"
                  ? [...CATEGORICAL_FIELDS, ...NUMERIC_FIELDS].map(f => <option key={f.key} value={f.key}>{f.label}</option>)
                  : CATEGORICAL_FIELDS.map(f => <option key={f.key} value={f.key}>{f.label}</option>)
                }
              </select>
            </div>
          )}

          {/* グループ分け */}
          <div>
            <label style={s.label}>グループ（色分け・任意）</label>
            <select value={groupBy} onChange={e => setGroupBy(e.target.value)} style={s.select}>
              <option value="">なし</option>
              {CATEGORICAL_FIELDS.map(f => <option key={f.key} value={f.key}>{f.label}</option>)}
            </select>
          </div>

          <hr style={{ border: "none", borderTop: "1px solid var(--border)" }} />

          {/* フィルター */}
          <div>
            <button onClick={() => setShowFilters(v => !v)}
              style={{ background: "none", border: "none", color: "var(--text-dim)", fontSize: 12, cursor: "pointer", display: "flex", alignItems: "center", gap: 4, padding: 0, marginBottom: 8 }}>
              <span style={{ display: "inline-block", transform: showFilters ? "rotate(90deg)" : "none", transition: "transform .2s" }}>▶</span>
              フィルター {showFilters ? "▲" : "▼"}
            </button>
            {showFilters && (
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                <div style={{ display: "flex", gap: 6 }}>
                  <div style={{ flex: 1 }}>
                    <label style={s.label}>開始日</label>
                    <input type="date" value={dateFrom} onChange={e => setDateFrom(e.target.value)} style={s.input} />
                  </div>
                  <div style={{ flex: 1 }}>
                    <label style={s.label}>終了日</label>
                    <input type="date" value={dateTo} onChange={e => setDateTo(e.target.value)} style={s.input} />
                  </div>
                </div>
                <div>
                  <label style={s.label}>馬場（カンマ区切り）</label>
                  <input type="text" value={surface} onChange={e => setSurface(e.target.value)}
                    placeholder="芝,ダート" style={s.input} />
                </div>
                <div style={{ display: "flex", gap: 6 }}>
                  <div style={{ flex: 1 }}>
                    <label style={s.label}>距離 下限</label>
                    <input type="number" value={distMin} onChange={e => setDistMin(e.target.value)}
                      placeholder="1200" style={s.input} />
                  </div>
                  <div style={{ flex: 1 }}>
                    <label style={s.label}>距離 上限</label>
                    <input type="number" value={distMax} onChange={e => setDistMax(e.target.value)}
                      placeholder="3600" style={s.input} />
                  </div>
                </div>
                <div>
                  <label style={s.label}>グレード（カンマ区切り）</label>
                  <input type="text" value={grade} onChange={e => setGrade(e.target.value)}
                    placeholder="G1,G2,G3" style={s.input} />
                </div>
                <div>
                  <label style={s.label}>最大件数</label>
                  <select value={limit} onChange={e => setLimit(e.target.value)} style={s.select}>
                    {["500", "1000", "2000", "5000"].map(v => <option key={v} value={v}>{Number(v).toLocaleString()}</option>)}
                  </select>
                </div>
              </div>
            )}
          </div>

          {/* 実行ボタン */}
          <button onClick={runAnalysis} disabled={loading}
            style={{
              padding: "10px 0", background: loading ? "rgba(59,130,246,0.3)" : "var(--accent)",
              color: "#fff", border: "none", borderRadius: 7, fontSize: 14, fontWeight: 700,
              cursor: loading ? "not-allowed" : "pointer", width: "100%", marginTop: "auto",
            }}>
            {loading ? "クエリ実行中…" : "▶ 分析実行"}
          </button>

          {/* フィルターリセット */}
          <button onClick={() => { setDateFrom(""); setDateTo(""); setSurface(""); setDistMin(""); setDistMax(""); setGrade(""); setLimit("2000"); }}
            style={{ padding: "6px 0", background: "transparent", border: "1px solid var(--border)", color: "var(--text-dim)", borderRadius: 6, fontSize: 12, cursor: "pointer" }}>
            フィルターをリセット
          </button>
        </aside>

        {/* ── メインエリア ── */}
        <main style={{ flex: 1, padding: "20px 24px", overflowY: "auto", minWidth: 0 }}>

          {/* 初期状態 */}
          {!result && !loading && !error && (
            <div style={{ textAlign: "center", padding: "60px 20px", color: "var(--text-dim)" }}>
              <div style={{ fontSize: 48, marginBottom: 16 }}>📊</div>
              <p style={{ fontSize: 15, marginBottom: 8 }}>左パネルで分析条件を設定し、「分析実行」を押してください</p>
              <p style={{ fontSize: 12 }}>
                対象データ: race_results × races × prediction_results<br />
                PostgreSQL からリアルタイムに集計します
              </p>
            </div>
          )}

          {/* エラー */}
          {error && (
            <div style={{ padding: "16px 20px", background: "rgba(239,68,68,0.1)", border: "1px solid rgba(239,68,68,0.3)", borderRadius: 8, color: "var(--err)", marginBottom: 16 }}>
              <strong>エラー:</strong> {error}
            </div>
          )}

          {/* 結果 */}
          {result && (
            <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>

              {/* サマリーバー */}
              <div style={{ display: "flex", gap: 20, flexWrap: "wrap", alignItems: "center" }}>
                <div style={{ fontSize: 13, color: "var(--text)" }}>
                  <span style={{ fontWeight: 700, color: "var(--accent)", fontSize: 20 }}>
                    {result.total_rows.toLocaleString()}
                  </span>
                  <span style={{ color: "var(--text-dim)", fontSize: 12, marginLeft: 4 }}>件</span>
                </div>
                <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                  {[
                    { label: "分析", value: { distribution: "分布", scatter: "散布", ranking: "ランキング", time_series: "時系列" }[result.analysis_type] },
                    { label: "Y軸", value: result.meta.y_label },
                    ...(result.x_field && result.analysis_type !== "ranking" && result.analysis_type !== "time_series"
                      ? [{ label: "X軸", value: result.meta.x_label }] : []),
                    ...(result.group_by ? [{ label: "グループ", value: result.meta.group_label ?? result.group_by }] : []),
                  ].map(({ label, value }) => (
                    <span key={label} style={{ fontSize: 11, padding: "2px 8px", borderRadius: 4, background: "rgba(59,130,246,0.1)", color: "var(--accent)", border: "1px solid rgba(59,130,246,0.2)" }}>
                      {label}: {value}
                    </span>
                  ))}
                </div>
              </div>

              {/* チャート */}
              <div style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 10, padding: "16px 20px" }}>
                <h2 style={{ fontSize: 13, fontWeight: 600, color: "#fff", marginBottom: 14 }}>
                  {result.analysis_type === "distribution" && "分布グラフ"}
                  {result.analysis_type === "scatter" && "散布図"}
                  {result.analysis_type === "ranking" && "ランキング"}
                  {result.analysis_type === "time_series" && "時系列グラフ"}
                </h2>
                {result.total_rows === 0 ? (
                  <p style={{ color: "var(--text-dim)", padding: "20px 0" }}>該当データがありません。フィルター条件を見直してください。</p>
                ) : result.analysis_type === "scatter" ? (
                  <ScatterChart rows={result.rows} meta={result.meta} groupKeys={groupKeys} />
                ) : result.analysis_type === "time_series" ? (
                  <LineChart rows={result.rows} meta={result.meta} groupKeys={groupKeys} />
                ) : (
                  <BarChart rows={result.rows} meta={result.meta} groupKeys={groupKeys} />
                )}
                <Legend groupKeys={groupKeys} label={result.meta.group_label} />
              </div>

              {/* データテーブル */}
              <div style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 10, overflow: "hidden" }}>
                <div style={{ padding: "12px 16px", borderBottom: "1px solid var(--border)", display: "flex", alignItems: "center", gap: 10 }}>
                  <h2 style={{ fontSize: 13, fontWeight: 600, color: "#fff" }}>データテーブル</h2>
                  <span style={{ fontSize: 11, color: "var(--text-dim)" }}>最大100行表示</span>
                </div>
                <DataTable rows={result.rows} analysisType={result.analysis_type} meta={result.meta} />
              </div>

              {/* クエリ詳細 */}
              <details style={{ fontSize: 11, color: "var(--text-dim)" }}>
                <summary style={{ cursor: "pointer" }}>クエリ詳細</summary>
                <code style={{ display: "block", marginTop: 6, padding: "8px 12px", background: "rgba(0,0,0,0.3)", borderRadius: 6, wordBreak: "break-all" }}>
                  {lastQuery}
                </code>
              </details>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
