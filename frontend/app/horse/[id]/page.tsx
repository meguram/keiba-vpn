"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";

/* ── 型定義 ── */
type Tab = "pedigree" | "results" | "index" | "growth";

type RaceHistory = {
  race_id?: string;
  race_date?: string;
  venue?: string;
  race_name?: string;
  surface?: string;
  distance?: number | string;
  track_condition?: string;
  finish_pos?: number | string;
  finish_position?: number | string;
  time?: string;
  finish_time_sec?: number;
  jockey?: string;
  jockey_name?: string;
  grade?: string;
  weight?: number | string;
  weight_change?: number | string;
  [key: string]: unknown;
};

type HorseInfo = {
  horse_name?: string;
  sex?: string;
  age?: number | string;
  birth_year?: number | string;
  sire?: string;
  dam?: string;
  dam_sire?: string;
  trainer?: string;
  owner?: string;
  [key: string]: unknown;
};

type PedNode = {
  name?: string;
  id?: string;
  sire?: PedNode;
  dam?: PedNode;
  [key: string]: unknown;
};

type HorseDetail = {
  horse_id: string;
  info?: HorseInfo;
  race_history?: RaceHistory[];
  ped5?: PedNode;
  [key: string]: unknown;
};

type MeguHistory = {
  race_id: string;
  race_date: string;
  venue: string;
  surface: string;
  distance: number;
  track_condition: string;
  megu_index: number;
  finish_time_sec?: number | null;
};

type GrowthPoint = {
  age?: number;
  year?: number;
  label?: string;
  avg_perf?: number;
  avg_megu?: number;
  count?: number;
  [key: string]: unknown;
};

/* ── めぐ指数色 ── */
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
    grade === "G3" ? { bg: "rgba(34,197,94,0.18)", color: "#22c55e" } : null;
  if (!s) return null;
  return <span style={{ fontSize: 10, fontWeight: 700, padding: "2px 6px", borderRadius: 3, ...s }}>{grade}</span>;
}

/* ── スピン ── */
function Spinner() {
  return (
    <div style={{ textAlign: "center", padding: "48px 0", color: "var(--text-dim)" }}>
      <div style={{ width: 32, height: 32, border: "2px solid var(--border)", borderTopColor: "var(--accent)", borderRadius: "50%", animation: "spin 1s linear infinite", margin: "0 auto 10px" }} />
      <p style={{ fontSize: 13 }}>読み込み中…</p>
    </div>
  );
}

/* ── 血統ツリー（再帰） ── */
function PedTree({ node, depth = 0 }: { node: PedNode; depth?: number }) {
  if (!node) return null;
  const name = node.name ?? "—";
  const indent = depth * 20;
  const isRoot = depth === 0;
  return (
    <div style={{ marginLeft: indent, marginBottom: 3 }}>
      <span style={{
        fontSize: isRoot ? 14 : 13 - Math.min(depth, 2),
        fontWeight: isRoot ? 700 : depth <= 1 ? 600 : 400,
        color: depth === 0 ? "#f0f6fc" : depth === 1 ? "#e2e8f0" : "var(--text-dim)",
      }}>
        {depth > 0 && <span style={{ color: "var(--border)", marginRight: 4 }}>{"└─".padStart(depth * 2)}</span>}
        {name}
      </span>
      {node.sire && <PedTree node={node.sire} depth={depth + 1} />}
      {node.dam && <PedTree node={node.dam} depth={depth + 1} />}
    </div>
  );
}

/* ── 血統タブ ── */
function PedigreeTab({ detail, loading }: { detail: HorseDetail | null; loading: boolean }) {
  if (loading) return <Spinner />;
  if (!detail) return <p style={{ color: "var(--text-dim)", fontSize: 13 }}>データを取得できませんでした。</p>;

  const info = detail.info ?? {};
  const ped = detail.ped5;

  const InfoRow = ({ label, value }: { label: string; value: string | number | undefined | null }) => (
    <div style={{ display: "flex", borderBottom: "1px solid var(--border)", padding: "9px 0" }}>
      <span style={{ width: 90, fontSize: 12, color: "var(--text-dim)", flexShrink: 0 }}>{label}</span>
      <span style={{ fontSize: 13, fontWeight: 500 }}>{value ?? "—"}</span>
    </div>
  );

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
      {/* 基本情報 */}
      <div>
        <h3 style={{ fontSize: 14, fontWeight: 700, color: "#f0f6fc", marginBottom: 12 }}>基本情報</h3>
        <InfoRow label="馬名" value={info.horse_name} />
        <InfoRow label="性別" value={info.sex} />
        <InfoRow label="生年" value={info.birth_year} />
        <InfoRow label="父" value={info.sire} />
        <InfoRow label="母" value={info.dam} />
        <InfoRow label="母父" value={info.dam_sire} />
        <InfoRow label="調教師" value={info.trainer} />
        <InfoRow label="馬主" value={info.owner} />
      </div>

      {/* 血統ツリー */}
      <div>
        <h3 style={{ fontSize: 14, fontWeight: 700, color: "#f0f6fc", marginBottom: 12 }}>血統</h3>
        {ped ? (
          <div style={{ background: "var(--surface2)", borderRadius: 8, padding: "14px 16px", fontFamily: "monospace", lineHeight: 1.8 }}>
            <PedTree node={ped} />
          </div>
        ) : (
          <div style={{ background: "var(--surface2)", borderRadius: 8, padding: "14px 16px", fontSize: 13 }}>
            {info.sire && <div style={{ marginBottom: 4 }}><span style={{ color: "var(--text-dim)", marginRight: 6 }}>父</span><strong>{String(info.sire)}</strong></div>}
            {info.dam && <div style={{ marginBottom: 4 }}><span style={{ color: "var(--text-dim)", marginRight: 6 }}>母</span><strong>{String(info.dam)}</strong></div>}
            {info.dam_sire && <div><span style={{ color: "var(--text-dim)", marginRight: 6 }}>母父</span><strong>{String(info.dam_sire)}</strong></div>}
            {!info.sire && !info.dam && <p style={{ color: "var(--text-dim)", fontSize: 12 }}>血統データがありません。</p>}
          </div>
        )}
      </div>
    </div>
  );
}

/* ── 過去成績タブ ── */
function ResultsTab({ detail, loading }: { detail: HorseDetail | null; loading: boolean }) {
  if (loading) return <Spinner />;
  if (!detail) return <p style={{ color: "var(--text-dim)", fontSize: 13 }}>データを取得できませんでした。</p>;

  const history = detail.race_history ?? [];
  if (history.length === 0) return <p style={{ color: "var(--text-dim)", fontSize: 13, padding: "24px 0" }}>過去成績データがありません。</p>;

  const TH: React.CSSProperties = { padding: "8px 10px", fontSize: 11, fontWeight: 600, color: "var(--text-dim)", textAlign: "left", borderBottom: "1px solid var(--border)", background: "var(--surface2)", whiteSpace: "nowrap" };
  const TD: React.CSSProperties = { padding: "9px 10px", borderBottom: "1px solid rgba(36,48,73,0.4)", fontSize: 12, verticalAlign: "middle" };

  return (
    <div style={{ overflowX: "auto" }}>
      <p style={{ fontSize: 12, color: "var(--text-dim)", marginBottom: 10 }}>{history.length}走の記録</p>
      <table style={{ width: "100%", borderCollapse: "collapse" }}>
        <thead>
          <tr>
            <th style={TH}>日付</th>
            <th style={TH}>競馬場</th>
            <th style={TH}>レース名</th>
            <th style={{ ...TH, textAlign: "center" }}>着順</th>
            <th style={TH}>条件</th>
            <th style={TH}>タイム</th>
            <th style={TH}>騎手</th>
          </tr>
        </thead>
        <tbody>
          {history.map((r, i) => {
            const pos = Number(r.finish_pos ?? r.finish_position ?? 99);
            const posColor = pos === 1 ? "#fbbf24" : pos === 2 ? "#94a3b8" : pos === 3 ? "#d97706" : "var(--text-dim)";
            const raceLink = r.race_id ? `/race/${r.race_id}` : undefined;
            return (
              <tr key={r.race_id ?? i}>
                <td style={{ ...TD, color: "var(--text-dim)" }}>{r.race_date ?? "—"}</td>
                <td style={TD}>{r.venue ?? "—"}</td>
                <td style={TD}>
                  <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
                    {raceLink ? (
                      <Link href={raceLink} style={{ color: "var(--accent)", textDecoration: "none" }}>
                        {r.race_name ?? r.race_id ?? "—"}
                      </Link>
                    ) : (r.race_name ?? "—")}
                    {gradeChip(r.grade as string | undefined)}
                  </div>
                </td>
                <td style={{ ...TD, textAlign: "center", fontWeight: pos <= 3 ? 800 : 400, color: posColor, fontSize: pos <= 3 ? 14 : 12 }}>
                  {pos >= 99 ? "—" : `${pos}着`}
                </td>
                <td style={{ ...TD, fontSize: 11, color: "var(--text-dim)" }}>
                  {r.surface}{r.distance && `${r.distance}m`} {r.track_condition}
                </td>
                <td style={TD}>{r.time ?? (r.finish_time_sec ? `${r.finish_time_sec}s` : "—")}</td>
                <td style={{ ...TD, fontSize: 11, color: "var(--text-dim)" }}>{r.jockey_name ?? r.jockey ?? "—"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/* ── 指数推移タブ ── */
function IndexTab({ horseId }: { horseId: string }) {
  const [data, setData] = useState<MeguHistory[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    if (loaded) return;
    setLoaded(true);
    setLoading(true);
    fetch(`/api/v1/horse/${horseId}/megu-index-history`)
      .then(r => r.ok ? r.json() : null)
      .then(d => setData(d?.history ?? []))
      .catch(() => setData([]))
      .finally(() => setLoading(false));
  }, [horseId, loaded]);

  if (loading) return <Spinner />;

  if (!data || data.length === 0) {
    return <p style={{ color: "var(--text-dim)", fontSize: 13, padding: "24px 0" }}>めぐ指数データがありません。</p>;
  }

  const TH: React.CSSProperties = { padding: "8px 10px", fontSize: 11, fontWeight: 600, color: "var(--text-dim)", textAlign: "left", borderBottom: "1px solid var(--border)", background: "var(--surface2)", whiteSpace: "nowrap" };
  const TD: React.CSSProperties = { padding: "9px 10px", borderBottom: "1px solid rgba(36,48,73,0.4)", fontSize: 12 };

  const avg = data.reduce((s, r) => s + r.megu_index, 0) / data.length;
  const max = Math.max(...data.map(r => r.megu_index));
  const recent3 = data.slice(0, 3);
  const recentAvg = recent3.reduce((s, r) => s + r.megu_index, 0) / recent3.length;

  return (
    <div>
      {/* サマリーバッジ */}
      <div style={{ display: "flex", gap: 12, marginBottom: 16, flexWrap: "wrap" }}>
        {[
          { label: "全走平均", value: avg.toFixed(1), c: meguColor(avg) },
          { label: "直近3走平均", value: recentAvg.toFixed(1), c: meguColor(recentAvg) },
          { label: "最高値", value: max.toFixed(1), c: meguColor(max) },
        ].map(({ label, value, c }) => (
          <div key={label} style={{ background: "var(--surface2)", border: "1px solid var(--border)", borderRadius: 8, padding: "10px 16px", textAlign: "center" }}>
            <div style={{ fontSize: 11, color: "var(--text-dim)", marginBottom: 4 }}>{label}</div>
            <div style={{ fontSize: 22, fontWeight: 800, color: c.color }}>{value}</div>
          </div>
        ))}
        <div style={{ background: "var(--surface2)", border: "1px solid var(--border)", borderRadius: 8, padding: "10px 16px", textAlign: "center" }}>
          <div style={{ fontSize: 11, color: "var(--text-dim)", marginBottom: 4 }}>計測走数</div>
          <div style={{ fontSize: 22, fontWeight: 800, color: "#7dd3fc" }}>{data.length}</div>
        </div>
      </div>

      {/* 簡易スパークライン（CSS バーチャート） */}
      <div style={{ background: "var(--surface2)", borderRadius: 8, padding: "12px 16px", marginBottom: 16 }}>
        <div style={{ fontSize: 11, color: "var(--text-dim)", marginBottom: 8 }}>めぐ指数推移（古い順 →）</div>
        <div style={{ display: "flex", alignItems: "flex-end", gap: 3, height: 60 }}>
          {[...data].reverse().map((r, i) => {
            const pct = Math.max(5, Math.min(100, ((r.megu_index - 60) / 60) * 100));
            const c = meguColor(r.megu_index);
            return (
              <div key={i} title={`${r.race_date} ${r.megu_index.toFixed(1)}`}
                style={{ flex: 1, height: `${pct}%`, background: c.color, borderRadius: "2px 2px 0 0", minWidth: 4, opacity: 0.85 }} />
            );
          })}
        </div>
      </div>

      {/* 詳細テーブル */}
      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead>
            <tr>
              <th style={TH}>日付</th>
              <th style={TH}>競馬場</th>
              <th style={TH}>条件</th>
              <th style={{ ...TH, textAlign: "right" }}>めぐ指数</th>
              <th style={TH}>走破タイム</th>
            </tr>
          </thead>
          <tbody>
            {data.map((r) => {
              const c = meguColor(r.megu_index);
              return (
                <tr key={r.race_id}>
                  <td style={{ ...TD, color: "var(--text-dim)" }}>{r.race_date}</td>
                  <td style={TD}>{r.venue}</td>
                  <td style={{ ...TD, fontSize: 11, color: "var(--text-dim)" }}>{r.surface}{r.distance}m {r.track_condition}</td>
                  <td style={{ ...TD, textAlign: "right", fontWeight: 800, color: c.color, background: c.bg }}>{r.megu_index.toFixed(1)}</td>
                  <td style={{ ...TD, color: "var(--text-dim)" }}>{r.finish_time_sec != null ? `${r.finish_time_sec}s` : "—"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* ── 成長曲線タブ ── */
function GrowthTab({ horseId }: { horseId: string }) {
  const [data, setData] = useState<GrowthPoint[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    if (loaded) return;
    setLoaded(true);
    setLoading(true);
    fetch(`/api/v1/horse/${horseId}/growth-curve`)
      .then(r => r.ok ? r.json() : null)
      .then(d => {
        const points: GrowthPoint[] = d?.points ?? d?.data ?? (Array.isArray(d) ? d : []);
        setData(points);
      })
      .catch(() => setData([]))
      .finally(() => setLoading(false));
  }, [horseId, loaded]);

  if (loading) return <Spinner />;

  if (!data || data.length === 0) {
    return (
      <div>
        <p style={{ color: "var(--text-dim)", fontSize: 13, padding: "12px 0" }}>
          この馬の成長曲線データがありません。
        </p>
        <Link href="/growth-curve" style={{ fontSize: 12, color: "var(--accent)", display: "inline-flex", alignItems: "center", gap: 4 }}>
          → 成長曲線ページ（全馬）を見る
        </Link>
      </div>
    );
  }

  const maxVal = Math.max(...data.map(p => p.avg_megu ?? p.avg_perf ?? 0));

  return (
    <div>
      <div style={{ fontSize: 11, color: "var(--text-dim)", marginBottom: 12 }}>年齢別パフォーマンス推移</div>
      {/* バーチャート */}
      <div style={{ background: "var(--surface2)", borderRadius: 8, padding: "16px", marginBottom: 16 }}>
        <div style={{ display: "flex", alignItems: "flex-end", gap: 8, height: 80 }}>
          {data.map((p, i) => {
            const val = p.avg_megu ?? p.avg_perf ?? 0;
            const pct = maxVal > 0 ? (val / maxVal) * 100 : 50;
            const c = meguColor(val as number);
            return (
              <div key={i} style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", gap: 3 }}>
                <div style={{ fontSize: 9, color: "var(--text-dim)" }}>{typeof val === "number" ? val.toFixed(0) : val}</div>
                <div style={{ width: "100%", height: `${pct}%`, background: c.color, borderRadius: "3px 3px 0 0", opacity: 0.9, minHeight: 4 }} />
                <div style={{ fontSize: 10, color: "var(--text-dim)", whiteSpace: "nowrap" }}>{p.label ?? `${p.age}歳`}</div>
              </div>
            );
          })}
        </div>
      </div>
      {/* 数値テーブル */}
      <table style={{ borderCollapse: "collapse", fontSize: 12 }}>
        <thead>
          <tr>
            {["年齢/期", "指数平均", "サンプル数"].map(h => (
              <th key={h} style={{ padding: "6px 12px", fontSize: 11, fontWeight: 600, color: "var(--text-dim)", borderBottom: "1px solid var(--border)", background: "var(--surface2)", textAlign: "left" }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((p, i) => {
            const val = p.avg_megu ?? p.avg_perf ?? 0;
            const c = meguColor(val as number);
            return (
              <tr key={i}>
                <td style={{ padding: "8px 12px", borderBottom: "1px solid rgba(36,48,73,0.4)", fontSize: 12 }}>{p.label ?? `${p.age}歳`}</td>
                <td style={{ padding: "8px 12px", borderBottom: "1px solid rgba(36,48,73,0.4)", fontWeight: 700, color: c.color }}>{typeof val === "number" ? val.toFixed(1) : val}</td>
                <td style={{ padding: "8px 12px", borderBottom: "1px solid rgba(36,48,73,0.4)", color: "var(--text-dim)" }}>{p.count ?? "—"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/* ── メインページ ── */
export default function HorsePage() {
  const { id: horseId } = useParams<{ id: string }>();
  const [tab, setTab] = useState<Tab>("pedigree");
  const [detail, setDetail] = useState<HorseDetail | null>(null);
  const [loadingDetail, setLoadingDetail] = useState(true);

  useEffect(() => {
    if (!horseId) return;
    fetch(`/api/horse/${horseId}/detail`)
      .then(r => r.ok ? r.json() : null)
      .then(d => setDetail(d))
      .catch(() => setDetail(null))
      .finally(() => setLoadingDetail(false));
  }, [horseId]);

  const horseName = detail?.info?.horse_name ?? horseId;

  const TABS: { id: Tab; label: string; icon: string }[] = [
    { id: "pedigree", label: "血統", icon: "🧬" },
    { id: "results", label: "過去成績", icon: "🏆" },
    { id: "index", label: "指数推移", icon: "📊" },
    { id: "growth", label: "成長曲線", icon: "📈" },
  ];

  return (
    <div style={{ minHeight: "100vh", background: "var(--bg)", color: "var(--text)" }}>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>

      {/* ヘッダー */}
      <div style={{ background: "var(--surface)", borderBottom: "1px solid var(--border)", padding: "18px 24px" }}>
        <div style={{ maxWidth: 1000, margin: "0 auto" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
            <Link href="/" style={{ fontSize: 12, color: "var(--text-dim)", textDecoration: "none" }}>← ホーム</Link>
          </div>
          <div style={{ display: "flex", alignItems: "baseline", gap: 14 }}>
            <h1 style={{ fontSize: 22, fontWeight: 800, color: "#f0f6fc", margin: 0 }}>
              🐴 {loadingDetail ? "…" : horseName}
            </h1>
            {detail?.info?.sex && (
              <span style={{ fontSize: 13, color: "var(--text-dim)" }}>
                {detail.info.sex}{detail.info.age ? ` ${detail.info.age}歳` : ""}
              </span>
            )}
            <span style={{ fontSize: 11, color: "var(--text-dim)", fontFamily: "monospace" }}>{horseId}</span>
          </div>
          {detail?.info?.sire && (
            <p style={{ fontSize: 12, color: "var(--text-dim)", margin: "4px 0 0" }}>
              父 {String(detail.info.sire)}
              {detail.info.dam && ` / 母 ${String(detail.info.dam)}`}
              {detail.info.dam_sire && ` (母父 ${String(detail.info.dam_sire)})`}
            </p>
          )}
        </div>
      </div>

      {/* タブ */}
      <div style={{ background: "var(--surface)", borderBottom: "1px solid var(--border)" }}>
        <div style={{ maxWidth: 1000, margin: "0 auto", display: "flex" }}>
          {TABS.map(t => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              style={{
                padding: "12px 22px",
                fontSize: 13,
                fontWeight: tab === t.id ? 700 : 400,
                color: tab === t.id ? "var(--accent)" : "var(--text-dim)",
                background: "none",
                border: "none",
                borderBottom: tab === t.id ? "2px solid var(--accent)" : "2px solid transparent",
                cursor: "pointer",
                display: "flex",
                alignItems: "center",
                gap: 6,
              }}
            >
              <span>{t.icon}</span>{t.label}
            </button>
          ))}
        </div>
      </div>

      {/* コンテンツ */}
      <div style={{ maxWidth: 1000, margin: "0 auto", padding: "24px" }}>
        {tab === "pedigree" && <PedigreeTab detail={detail} loading={loadingDetail} />}
        {tab === "results"  && <ResultsTab  detail={detail} loading={loadingDetail} />}
        {tab === "index"    && horseId && <IndexTab  horseId={horseId} />}
        {tab === "growth"   && horseId && <GrowthTab horseId={horseId} />}
      </div>
    </div>
  );
}
