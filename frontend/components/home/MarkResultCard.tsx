"use client";

import { useEffect, useState } from "react";
import { PendingLink } from "@/components/PendingLink";
import { getJson } from "@/lib/httpCache";

type MarkRow = {
  key: string; symbol: string; n: number; text: string;
  f1: number; f2: number; f3: number; f4p: number;
  win_rate: number | null; top2_rate: number | null; top3_rate: number | null;
  avg_popularity: number | null; avg_odds: number | null;
  win_avg_popularity: number | null; win_avg_odds: number | null;     // 1着だったときの平均人気・平均オッズ
  races_with: number; races_in3: number; races_in3_rate: number | null;     // その印の馬が出たレース数と、そのうち1頭でも馬券内（3着以内）に来たレース数
};
type Period = { key: string; label: string; from: string; to: string; data_from: string | null; data_to: string | null; days: number; races: number; status: string; marks: MarkRow[] };
type Card = { as_of: string; year: number; periods: Period[] };

const md = (ymd: string) => `${Number(ymd.slice(4, 6))}/${Number(ymd.slice(6, 8))}`;
const pct = (x: number | null) => (x == null ? "—" : `${(x * 100).toFixed(1)}%`);

// 印ごとの色（競馬新聞の印の印象に寄せた、暗い背景で読める明るさ）
const MARK_COLOR: Record<string, string> = { honmei: "#f87171", pair: "#60a5fa", show_val: "#fbbf24", star: "#c4b5fd", kesi: "#94a3b8" };
// 着順の色（金・銀・銅・着外）
const SEGMENTS: { k: "f1" | "f2" | "f3" | "f4p"; label: string; color: string }[] = [
  { k: "f1", label: "1着", color: "#fbbf24" },
  { k: "f2", label: "2着", color: "#cbd5e1" },
  { k: "f3", label: "3着", color: "#b45309" },
  { k: "f4p", label: "着外", color: "rgba(148,163,184,0.22)" },
];

const CSS = `
.mrc{max-width:1100px;margin:0 auto 16px;padding:0 20px}
.mrc-panel{background:var(--surface);border:1px solid var(--border);border-radius:12px;overflow:hidden}
.mrc-head{display:flex;align-items:center;gap:8px;padding:10px 12px 0}
.mrc-head h2{margin:0;font-size:14px;font-weight:700;color:var(--text)}
.mrc-head .mrc-link{margin-left:auto;font-size:11px;color:var(--accent);text-decoration:none;white-space:nowrap}
.mrc-tabs{display:flex;gap:4px;padding:8px 12px 0}
.mrc-tab{appearance:none;flex:1;border:1px solid var(--border);background:transparent;color:var(--text-dim);font-size:12px;font-weight:600;padding:4px 6px;border-radius:999px;cursor:pointer;min-height:28px;white-space:nowrap;transition:all .15s}
.mrc-tab:hover{color:var(--text);border-color:rgba(148,163,184,.5)}
.mrc-tab[aria-pressed="true"]{background:rgba(88,166,255,.14);border-color:rgba(88,166,255,.55);color:#9ecbff}
.mrc-meta{padding:6px 12px 8px;font-size:11px;color:var(--text-dim)}
.mrc-meta b{color:var(--text);font-weight:700}
.mrc-row{display:grid;grid-template-columns:28px 1fr;column-gap:10px;align-items:start;padding:7px 12px;border-top:1px solid var(--border)}
.mrc-sym{display:grid;place-items:center;width:26px;height:26px;border-radius:50%;font-size:14px;font-weight:800;border:1.5px solid;line-height:1;margin-top:1px}
.mrc-line{display:flex;align-items:baseline;gap:8px;min-width:0}
.mrc-rec{font-size:13px;font-weight:700;font-variant-numeric:tabular-nums;color:var(--text);white-space:nowrap}
.mrc-win{margin-left:auto;font-size:11px;color:var(--text-dim);font-variant-numeric:tabular-nums;white-space:nowrap}
.mrc-win b{color:var(--text);font-size:12px}
.mrc-bar{display:flex;height:6px;border-radius:3px;overflow:hidden;margin:4px 0 3px;background:rgba(148,163,184,.12)}
.mrc-bar>i{display:block;height:100%}
.mrc-sub{font-size:11px;color:var(--text-dim);font-variant-numeric:tabular-nums;line-height:1.4}
.mrc-sub b{color:var(--text);font-weight:600}
.mrc-legend{display:flex;gap:10px;flex-wrap:wrap;padding:7px 12px 9px;border-top:1px solid var(--border);font-size:10px;color:var(--text-dim)}
.mrc-legend i{display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:4px;vertical-align:-1px}
.mrc-empty{padding:14px 12px;font-size:12px;color:var(--text-dim)}
@media (max-width:767px){.mrc{padding:0 12px}}
`;

function Row({ m }: { m: MarkRow }) {
  const color = MARK_COLOR[m.key] ?? "var(--text-dim)";
  const showWin = m.key !== "kesi";                          // ◎○▲★ は勝率も出す
  const showAvg = m.key !== "kesi";                          // ◎○▲★ は平均人気・平均オッズも出す
  const none = m.n === 0;
  const avg = showAvg && m.avg_popularity != null && m.avg_odds != null ? `${m.avg_popularity.toFixed(1)}人気・${m.avg_odds.toFixed(1)}倍` : null;
  const winAvg = showAvg && m.win_avg_popularity != null && m.win_avg_odds != null ? `${m.win_avg_popularity.toFixed(1)}人気・${m.win_avg_odds.toFixed(1)}倍` : null;
  return (
    <div className="mrc-row">
      <span className="mrc-sym" style={{ color, borderColor: color, background: `${color}1f` }} aria-label={m.symbol}>{m.symbol}</span>
      <div style={{ minWidth: 0 }}>
        <div className="mrc-line">
          <span className="mrc-rec" title="1着-2着-3着-着外（頭数）">{none ? "—" : m.text}</span>
          {showWin && !none && <span className="mrc-win">勝率 <b>{pct(m.win_rate)}</b></span>}
        </div>
        {!none && (
          <>
            <div className="mrc-bar" aria-hidden>
              {SEGMENTS.map((s) => (m[s.k] > 0 ? <i key={s.k} style={{ width: `${(m[s.k] / m.n) * 100}%`, background: s.color }} title={`${s.label} ${m[s.k]}頭`} /> : null))}
            </div>
            <div className="mrc-sub">複勝率 <b>{pct(m.top3_rate)}</b>{avg ? <>　平均 <b>{avg}</b></> : null}</div>
            {winAvg && <div className="mrc-sub">1着時 平均 <b>{winAvg}</b></div>}
            {m.key === "kesi" && m.races_with > 0 && <div className="mrc-sub">馬券内に来たレース <b>{m.races_in3}</b> / {m.races_with}R（<b>{pct(m.races_in3_rate)}</b>）</div>}
          </>
        )}
      </div>
    </div>
  );
}

/** ホームの『印の成績』（サイドバー向けの小型版）: ◎○▲★消 の着順（1着-2着-3着-着外）を、先週・直近5週間・年（今年のデータが約2か月未満なら去年）の切り替えで見せる。
 *  着順の割合は色付きの横棒、◎○▲★は勝率と確定の平均人気・平均オッズ。集計が無ければ何も出さない。 */
export function MarkResultCard() {
  const [card, setCard] = useState<Card | null>(null);
  const [sel, setSel] = useState("recent_5w");

  useEffect(() => {
    let alive = true;
    getJson<Card>("/api/prediction-results/mark-card", { ttlMs: 60_000 })
      .then((d) => { if (alive) setCard(d); })
      .catch(() => { if (alive) setCard(null); });
    return () => { alive = false; };
  }, []);

  if (!card || !card.periods.some((p) => p.races > 0)) return null;
  const cur = card.periods.find((p) => p.key === sel && p.races > 0) ?? card.periods.find((p) => p.races > 0) ?? card.periods[0];

  return (
    <section aria-label="印の成績" className="mrc">
      <style>{CSS}</style>
      <div className="mrc-panel">
        <div className="mrc-head">
          <h2>🎯 印の成績</h2>
          <PendingLink href="/prediction-results" className="mrc-link">成績をすべて見る →</PendingLink>
        </div>
        <div className="mrc-tabs" role="group" aria-label="期間">
          {card.periods.map((p) => (
            <button key={p.key} type="button" className="mrc-tab" aria-pressed={p.key === cur.key} disabled={p.races === 0} style={p.races === 0 ? { opacity: 0.4, cursor: "default" } : undefined} onClick={() => setSel(p.key)}>
              {p.label}
            </button>
          ))}
        </div>
        <div className="mrc-meta">
          <b>{cur.data_from && cur.data_to ? `${md(cur.data_from)}〜${md(cur.data_to)}` : `${md(cur.from)}〜${md(cur.to)}`}</b>・{cur.races}レース・着順は 1着-2着-3着-着外
        </div>
        {cur.races === 0 ? (
          <div className="mrc-empty">集計できるレースがありません</div>
        ) : (
          <div role="list" aria-label={`${cur.label}の印の成績`}>
            {cur.marks.map((m) => <Row key={m.key} m={m} />)}
          </div>
        )}
        <div className="mrc-legend">
          {SEGMENTS.map((s) => <span key={s.k}><i style={{ background: s.color }} />{s.label}</span>)}
        </div>
      </div>
    </section>
  );
}
