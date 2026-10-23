"use client";

import { PACE_JA, actualPace, paceVerdict, type PaceLevel, type PositionSummary } from "@/lib/resultCheck";

const TONE = { hit: { fg: "#86efac", bd: "rgba(34,197,94,0.5)", bg: "rgba(34,197,94,0.12)", word: "的中" }, near: { fg: "#fcd34d", bd: "rgba(251,191,36,0.5)", bg: "rgba(251,191,36,0.10)", word: "惜しい" }, miss: { fg: "#fca5a5", bd: "rgba(239,68,68,0.5)", bg: "rgba(239,68,68,0.12)", word: "外れ" } } as const;

export function Verdict({ v }: { v: keyof typeof TONE }) {
  const t = TONE[v];
  return <span style={{ padding: "1px 8px", borderRadius: 999, border: `1px solid ${t.bd}`, background: t.bg, color: t.fg, fontSize: 12, fontWeight: 700, whiteSpace: "nowrap" }}>{t.word}</span>;
}

const normLevel = (l: string | undefined): PaceLevel | null => (l === "high" || l === "mid" || l === "slow" ? l : null);

/** レース結果の上に出す『展開の答え合わせ』: ペース（予想 vs 前後半3Fの実際）と位置取り（想定の番手 vs 最初のコーナーの順位）。会員向け。 */
export function ResultCheckPanel({ paceLevel, paceLabel, first3f, second3f, pos }: {
  paceLevel: string | undefined; paceLabel?: string; first3f?: number | null; second3f?: number | null; pos: PositionSummary;
}) {
  const pl = normLevel(paceLevel);
  const act = actualPace(first3f, second3f);
  const pv = pl && act ? paceVerdict(pl, act.level) : null;
  if (!pv && pos.n === 0) return null;
  const cell = { flex: "1 1 280px", minWidth: 0 } as const;
  const dim = { color: "var(--text-dim)", fontSize: 12 } as const;
  return (
    <section aria-label="展開の答え合わせ" style={{ display: "flex", gap: 20, flexWrap: "wrap", padding: "10px 14px", marginBottom: 10, background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 10 }}>
      <div style={cell}>
        <div style={dim}>ペースの答え合わせ</div>
        {pl && act ? (
          <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap", marginTop: 2 }}>
            <span>予想 <b>{paceLabel || PACE_JA[pl]}</b></span><span style={dim}>→</span>
            <span>実際 <b>{PACE_JA[act.level]}</b></span>
            {pv && <Verdict v={pv} />}
            <span style={dim} title="前半3F − 後半3F。−1.0秒以下=ハイ、+1.0秒以上=スロー（目安）">前半3F {first3f} − 後半3F {second3f} ＝ {act.diff > 0 ? "+" : ""}{act.diff}</span>
          </div>
        ) : <div style={{ ...dim, marginTop: 2 }}>予想または前後半3Fのデータがありません</div>}
      </div>
      <div style={cell}>
        <div style={dim}>位置取りの答え合わせ（想定の番手 vs 最初のコーナーの順位）</div>
        {pos.n > 0 ? (
          <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap", marginTop: 2 }}>
            <span>±2番手以内 <b>{pos.hit}</b>/{pos.n}頭</span>
            <span style={dim}>平均ズレ {pos.avgAbsDiff}番手</span>
            {pos.frontPred != null && <span style={dim}>先頭の想定 <b style={{ color: "var(--text)" }}>{pos.frontPred}番</b></span>}
            {pos.frontActual != null && <span style={dim}>実際の先頭 <b style={{ color: "var(--text)" }}>{pos.frontActual}番</b></span>}
          </div>
        ) : <div style={{ ...dim, marginTop: 2 }}>通過順位のデータがありません</div>}
      </div>
    </section>
  );
}
