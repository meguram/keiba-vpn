"use client";

import { useEffect, useState } from "react";
import { PendingLink } from "@/components/PendingLink";
import { getJson } from "@/lib/httpCache";

type MarkRow = {
  key: string; symbol: string; n: number; text: string;
  f1: number; f2: number; f3: number; f4p: number;
  win_rate: number | null; top2_rate: number | null; top3_rate: number | null;
  avg_popularity: number | null; avg_odds: number | null;
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
.mrc{max-width:1100px;margin:0 auto 24px;padding:0 20px}
.mrc-head{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:10px}
.mrc-head h2{margin:0;font-size:16px;font-weight:700;color:var(--text)}
.mrc-head .mrc-link{margin-left:auto;font-size:12px;color:var(--accent);text-decoration:none}
.mrc-panel{background:var(--surface);border:1px solid var(--border);border-radius:14px;overflow:hidden}
.mrc-tabs{display:flex;gap:6px;padding:12px 14px 0;flex-wrap:wrap}
.mrc-tab{appearance:none;border:1px solid var(--border);background:transparent;color:var(--text-dim);font-size:13px;font-weight:600;padding:6px 14px;border-radius:999px;cursor:pointer;min-height:34px;transition:all .15s}
.mrc-tab:hover{color:var(--text);border-color:rgba(148,163,184,.5)}
.mrc-tab[aria-pressed="true"]{background:rgba(88,166,255,.14);border-color:rgba(88,166,255,.55);color:#9ecbff}
.mrc-meta{padding:8px 16px 10px;font-size:12px;color:var(--text-dim)}
.mrc-meta b{color:var(--text);font-weight:700}
.mrc-grid{display:grid;grid-template-columns:48px minmax(190px,1fr) repeat(4,78px);align-items:center;column-gap:12px}
.mrc-th{padding:6px 16px;font-size:11px;color:var(--text-dim);border-top:1px solid var(--border);background:rgba(148,163,184,.05)}
.mrc-th>span{text-align:right}
.mrc-th>span:nth-child(-n+2){text-align:left}
.mrc-row{padding:11px 16px;border-top:1px solid var(--border)}
.mrc-sym{display:grid;place-items:center;width:34px;height:34px;border-radius:50%;font-size:18px;font-weight:800;border:1.5px solid;line-height:1}
.mrc-rec{min-width:0}
.mrc-rec .t{font-size:14px;font-weight:700;font-variant-numeric:tabular-nums;color:var(--text);letter-spacing:.01em}
.mrc-bar{display:flex;height:8px;border-radius:4px;overflow:hidden;margin-top:6px;background:rgba(148,163,184,.12)}
.mrc-bar>i{display:block;height:100%}
.mrc-m{text-align:right;font-variant-numeric:tabular-nums}
.mrc-m .v{font-size:15px;font-weight:700;color:var(--text)}
.mrc-m .v.dim{color:var(--text-dim);font-weight:500}
.mrc-m .l{display:none;font-size:10px;color:var(--text-dim)}
.mrc-legend{display:flex;gap:14px;flex-wrap:wrap;padding:9px 16px 12px;border-top:1px solid var(--border);font-size:11px;color:var(--text-dim)}
.mrc-legend i{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:5px;vertical-align:-1px}
.mrc-empty{padding:18px 16px;font-size:13px;color:var(--text-dim)}
.mrc-m-wrap{display:contents}
@media (max-width:640px){
  .mrc{padding:0 12px}
  .mrc-th{display:none}
  .mrc-grid{grid-template-columns:40px 1fr;row-gap:10px}
  .mrc-row{padding:12px 14px}
  .mrc-m-wrap{display:grid;grid-column:1 / -1;grid-template-columns:repeat(4,1fr);gap:6px}
  .mrc-m{text-align:left}
  .mrc-m .l{display:block}
}
`;

function Metric({ label, value, dim }: { label: string; value: string; dim?: boolean }) {
  return (
    <div className="mrc-m">
      <div className="l">{label}</div>
      <div className={`v${dim ? " dim" : ""}`}>{value}</div>
    </div>
  );
}

function Row({ m }: { m: MarkRow }) {
  const color = MARK_COLOR[m.key] ?? "var(--text-dim)";
  const showWin = m.key === "honmei" || m.key === "pair";    // ◎○ は勝率も出す
  const showAvg = m.key !== "kesi";                          // ◎○▲★ は平均人気・平均オッズも出す
  const none = m.n === 0;
  return (
    <div className="mrc-grid mrc-row" role="row">
      <span className="mrc-sym" style={{ color, borderColor: color, background: `${color}1f` }} aria-label={m.symbol}>{m.symbol}</span>
      <div className="mrc-rec">
        <div className="t" title="1着-2着-3着-着外（頭数）">{none ? "—" : m.text}</div>
        {!none && (
          <div className="mrc-bar" aria-hidden>
            {SEGMENTS.map((s) => (m[s.k] > 0 ? <i key={s.k} style={{ width: `${(m[s.k] / m.n) * 100}%`, background: s.color }} title={`${s.label} ${m[s.k]}頭`} /> : null))}
          </div>
        )}
      </div>
      <div className="mrc-m-wrap">
        <Metric label="勝率" value={none || !showWin ? "—" : pct(m.win_rate)} dim={!showWin || none} />
        <Metric label="複勝率" value={none ? "—" : pct(m.top3_rate)} />
        <Metric label="平均人気" value={none || !showAvg || m.avg_popularity == null ? "—" : m.avg_popularity.toFixed(1)} dim={!showAvg || none} />
        <Metric label="平均オッズ" value={none || !showAvg || m.avg_odds == null ? "—" : `${m.avg_odds.toFixed(1)}倍`} dim={!showAvg || none} />
      </div>
    </div>
  );
}

/** ホームの『印の成績』: ◎○▲★消 の着順（1着-2着-3着-着外）を、先週・直近5週間・年（今年のデータが約2か月未満なら去年）の切り替えで見せる。
 *  着順の割合は色付きの横棒、◎○は勝率、◎○▲★は確定の平均人気・平均オッズ。集計が無ければ何も出さない。 */
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
      <div className="mrc-head">
        <h2>🎯 印の成績</h2>
        <PendingLink href="/prediction-results" className="mrc-link">予測成績をすべて見る →</PendingLink>
      </div>
      <div className="mrc-panel">
        <div className="mrc-tabs" role="group" aria-label="期間">
          {card.periods.map((p) => (
            <button key={p.key} type="button" className="mrc-tab" aria-pressed={p.key === cur.key} disabled={p.races === 0} style={p.races === 0 ? { opacity: 0.4, cursor: "default" } : undefined} onClick={() => setSel(p.key)}>
              {p.label}
            </button>
          ))}
        </div>
        <div className="mrc-meta">
          <b>{cur.data_from && cur.data_to ? `${md(cur.data_from)}〜${md(cur.data_to)}` : `${md(cur.from)}〜${md(cur.to)}`}</b>・{cur.races}レース
          <span>　着順は 1着-2着-3着-着外（印をつけた馬の頭数）</span>
        </div>
        {cur.races === 0 ? (
          <div className="mrc-empty">集計できるレースがありません</div>
        ) : (
          <div role="table" aria-label={`${cur.label}の印の成績`}>
            <div className="mrc-grid mrc-th" role="row">
              <span>印</span><span>着順（1-2-3-着外）</span><span>勝率</span><span>複勝率</span><span>平均人気</span><span>平均オッズ</span>
            </div>
            {cur.marks.map((m) => <Row key={m.key} m={m} />)}
            <div className="mrc-legend">
              {SEGMENTS.map((s) => <span key={s.k}><i style={{ background: s.color }} />{s.label}</span>)}
              <span style={{ marginLeft: "auto" }}>平均人気・オッズは確定の値</span>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}
