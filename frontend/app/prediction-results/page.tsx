"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { PageShell } from "@/components/PageShell";
import { PendingLink } from "@/components/PendingLink";
import { getJson } from "@/lib/httpCache";
import { useAuthStatus } from "@/lib/hooks/useAuthStatus";
import {
  FINISH_META, MARK_META, SCOPES, dayLabel, pct, periodKey, periodLabel, shiftDate, yen,
  type Counts, type MarkKey, type PredictionResults, type RaceRecord, type SeriesPoint,
} from "@/lib/predictionResults";

type DaysInfo = { days: { date: string; status: string; races: number }[]; latest: string | null };

const card: React.CSSProperties = { background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 12, padding: "14px 16px" };
const dim: React.CSSProperties = { color: "var(--text-dim)", fontSize: 12 };

function StatusBadge({ status }: { status: "final" | "provisional" }) {
  const final = status === "final";
  return (
    <span style={{ fontSize: 12, fontWeight: 700, padding: "2px 10px", borderRadius: 999, color: final ? "#4ade80" : "#fbbf24",
      border: `1px solid ${final ? "rgba(74,222,128,0.5)" : "rgba(251,191,36,0.5)"}` }}>
      {final ? "確定" : "暫定"}
    </span>
  );
}

function Tile({ label, value, sub, locked }: { label: string; value: string; sub?: string; locked?: boolean }) {
  return (
    <div style={{ ...card, minWidth: 0 }}>
      <div style={dim}>{label}</div>
      {locked ? (
        <div style={{ fontSize: 13, color: "var(--text-dim)", marginTop: 8 }}>
          🔒 <Link href="/login" style={{ color: "var(--accent)" }}>ログイン</Link>で表示
        </div>
      ) : (
        <div style={{ fontSize: 26, fontWeight: 800, marginTop: 2, color: "#f0f6fc" }}>{value}</div>
      )}
      {sub && !locked && <div style={dim}>{sub}</div>}
    </div>
  );
}

/** ◎○★消 の着順分布（1着/2着/3着/4着以下）を、件数つきの積み上げ棒で。色だけに頼らず、各区間に数字とラベルを出す。 */
const EMPTY_COUNTS: Counts = { n: 0, f1: 0, f2: 0, f3: 0, f4p: 0 };

function FinishDist({ mark, c }: { mark: MarkKey; c: Counts }) {
  const m = MARK_META[mark];
  return (
    <div style={{ ...card }}>
      <div style={{ display: "flex", alignItems: "baseline", gap: 8, marginBottom: 8 }}>
        <span style={{ fontSize: 22, color: m.color, fontWeight: 800 }} aria-hidden>{m.sym}</span>
        <span style={{ fontWeight: 700 }}>{m.label}</span>
        <span style={{ ...dim, marginLeft: "auto" }}>{c.n} 頭</span>
      </div>
      {c.n === 0 ? (
        <div style={dim}>この期間に印のついた馬の結果がありません</div>
      ) : (
        <>
          <div role="img" aria-label={`${m.sym}の着順: ` + FINISH_META.map((f) => `${f.label}${c[f.key]}頭`).join("、")}
            style={{ display: "flex", height: 28, borderRadius: 6, overflow: "hidden", background: "var(--surface2)" }}>
            {FINISH_META.map((f) => c[f.key] > 0 && (
              <div key={f.key} style={{ width: `${(c[f.key] / c.n) * 100}%`, background: f.color, display: "flex", alignItems: "center",
                justifyContent: "center", fontSize: 12, fontWeight: 700, color: "#fff", minWidth: 0 }}>
                {c[f.key] / c.n >= 0.08 ? c[f.key] : ""}
              </div>
            ))}
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 6, marginTop: 8 }}>
            {FINISH_META.map((f) => (
              <div key={f.key} style={{ textAlign: "center" }}>
                <div style={{ fontSize: 11, color: "var(--text-dim)", display: "flex", alignItems: "center", justifyContent: "center", gap: 4 }}>
                  <span style={{ width: 8, height: 8, borderRadius: 2, background: f.color, display: "inline-block" }} aria-hidden />{f.label}
                </div>
                <div style={{ fontWeight: 700, fontSize: 15 }}>{c[f.key]}<span style={{ fontSize: 11, color: "var(--text-dim)" }}>（{pct(c[f.key] / c.n)}）</span></div>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

/** 日別推移の折れ線（回収率・本命の1着率）。外部ライブラリなしの SVG。 */
function Trend({ series, showRoi }: { series: SeriesPoint[]; showRoi: boolean }) {
  const W = 640, H = 200, L = 40, R = 12, T = 12, B = 30;
  if (series.length < 2) return <div style={dim}>推移は 2 日分以上の集計がそろうと表示されます。</div>;
  const roiVals = showRoi ? series.map((s) => s.roi).filter((v): v is number => v != null) : [];
  const hwVals = series.map((s) => s.honmei_win).filter((v): v is number => v != null);
  const maxY = Math.max(1.2, ...roiVals, ...hwVals) * 1.05;
  const x = (i: number) => L + (i / (series.length - 1)) * (W - L - R);
  const y = (v: number) => T + (1 - v / maxY) * (H - T - B);
  const line = (key: "roi" | "honmei_win") =>
    series.map((s, i) => (s[key] == null ? null : `${x(i).toFixed(1)},${y(s[key] as number).toFixed(1)}`)).filter(Boolean).join(" ");
  const ticks = [0, 0.5, 1, Math.floor(maxY * 2) / 2].filter((v, i, a) => v <= maxY && a.indexOf(v) === i);
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="日別の推移" style={{ maxWidth: "100%" }}>
        {ticks.map((v) => (
          <g key={v}>
            <line x1={L} x2={W - R} y1={y(v)} y2={y(v)} stroke={v === 1 ? "#64748b" : "var(--border)"} strokeDasharray={v === 1 ? "4 3" : undefined} />
            <text x={L - 6} y={y(v) + 4} fontSize="11" fill="var(--text-dim)" textAnchor="end">{Math.round(v * 100)}%</text>
          </g>
        ))}
        {series.map((s, i) => (
          <text key={s.date} x={x(i)} y={H - 8} fontSize="11" fill="var(--text-dim)" textAnchor="middle">
            {series.length > 8 && i % 2 ? "" : dayLabel(s.date).slice(0, -3)}
          </text>
        ))}
        {showRoi && <polyline points={line("roi")} fill="none" stroke="#34d399" strokeWidth="2.5" />}
        <polyline points={line("honmei_win")} fill="none" stroke="#60a5fa" strokeWidth="2.5" />
        {series.map((s, i) => (
          <g key={s.date}>
            {showRoi && s.roi != null && <circle cx={x(i)} cy={y(s.roi)} r="3.5" fill="#34d399"><title>{dayLabel(s.date)} 回収率 {pct(s.roi)}（{s.status === "final" ? "確定" : "暫定"}）</title></circle>}
            {s.honmei_win != null && <circle cx={x(i)} cy={y(s.honmei_win)} r="3.5" fill="#60a5fa"><title>{dayLabel(s.date)} 本命1着率 {pct(s.honmei_win)}</title></circle>}
          </g>
        ))}
      </svg>
      <div style={{ display: "flex", gap: 14, fontSize: 12, color: "var(--text-dim)", flexWrap: "wrap" }}>
        <span><b style={{ color: "#60a5fa" }}>━</b> ◎の1着率</span>
        {showRoi && <span><b style={{ color: "#34d399" }}>━</b> 仮想購入の回収率（破線＝100%）</span>}
      </div>
    </div>
  );
}

function FinishChip({ pos }: { pos: number | null | undefined }) {
  if (!pos) return <span style={dim}>—</span>;
  const hit = pos <= 3;
  return (
    <span style={{ display: "inline-block", minWidth: 34, textAlign: "center", fontWeight: 700, fontSize: 12, padding: "1px 6px", borderRadius: 6,
      color: pos === 1 ? "#111827" : "var(--text)", background: pos === 1 ? "#f59e0b" : hit ? "rgba(148,163,184,0.25)" : "var(--surface2)" }}>
      {pos}着
    </span>
  );
}

function RaceCard({ r, member }: { r: RaceRecord; member: boolean }) {
  if (!r.included) {
    return (
      <div style={{ ...card, opacity: 0.7 }}>
        <div style={{ fontWeight: 700 }}>{r.venue}{r.round}R {r.race_name}</div>
        <div style={dim}>{r.reason === "no_result" ? "結果がまだありません" : "予測がありません"}（集計に含めていません）</div>
      </div>
    );
  }
  const b = r.bets;
  return (
    <PendingLink href={`/race/${r.race_id}`} style={{ ...card, display: "block", textDecoration: "none", color: "var(--text)" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
        <span style={{ background: "#3b82f6", color: "#fff", borderRadius: 6, padding: "1px 8px", fontWeight: 700, fontSize: 12 }}>{r.venue}{r.round}R</span>
        <span style={{ fontWeight: 700, fontSize: 14 }}>{r.race_name}</span>
        {r.result_status === "provisional" && <span style={{ ...dim, marginLeft: "auto" }}>暫定</span>}
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "auto 1fr", gap: "4px 10px", marginTop: 8, alignItems: "center", fontSize: 13 }}>
        <span style={dim}>予測1位</span>
        <span>{r.top1?.horse_number}番 {r.top1?.horse_name} <FinishChip pos={r.top1?.finish} /> <span style={dim}>勝率{pct(r.top1?.win_prob)}</span></span>
        {(["honmei", "pair", "star", "kesi"] as MarkKey[]).map((m) => (r.marks?.[m]?.length ? (
          <span key={m} style={{ display: "contents" }}>
            <span style={{ color: MARK_META[m].color, fontWeight: 800 }}>{MARK_META[m].sym}</span>
            <span>
              {r.marks![m].map((p, i) => (
                <span key={p.horse_number} style={{ marginRight: 8, whiteSpace: "nowrap" }}>
                  {i > 0 ? "" : ""}{p.horse_number}番 {p.horse_name} <FinishChip pos={p.finish} />
                </span>
              ))}
            </span>
          </span>
        ) : null))}
        <span style={dim}>結果</span>
        <span>{(r.result_top3 ?? []).map((x) => `${x.finish}着 ${x.horse_number}番`).join(" / ")}</span>
        {member && b && b.total_bet > 0 && (
          <>
            <span style={dim}>仮想購入</span>
            <span style={{ fontWeight: 700, color: b.total_return >= b.total_bet ? "#34d399" : "var(--text)" }}>
              {yen(b.total_bet)} → {yen(b.total_return)}（{pct(b.total_return / b.total_bet)}）{b.estimated ? <span style={dim}> 概算</span> : null}
            </span>
          </>
        )}
      </div>
    </PendingLink>
  );
}

export default function PredictionResultsPage() {
  const { isMember, loading: authLoading } = useAuthStatus();
  const member = !authLoading && isMember;
  const [scope, setScope] = useState<PredictionResults["scope"]>("day");
  const [date, setDate] = useState<string>("");
  const [days, setDays] = useState<DaysInfo | null>(null);
  const [data, setData] = useState<PredictionResults | null>(null);
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);

  // URL（?scope=&date=）から復元。無ければ最新の集計日
  useEffect(() => {
    const q = new URLSearchParams(window.location.search);
    const sc = q.get("scope");
    if (sc && SCOPES.some((s) => s.id === sc)) setScope(sc as PredictionResults["scope"]);
    getJson<DaysInfo>("/api/prediction-results/days", { ttlMs: 20_000 })
      .then((d) => { setDays(d); setDate(/^\d{8}$/.test(q.get("date") ?? "") ? (q.get("date") as string) : (d.latest ?? "")); })
      .catch(() => setErr("集計の一覧を取得できませんでした。"));
  }, []);

  const load = useCallback((force = false) => {
    if (!date) return;
    setBusy(true); setErr("");
    getJson<PredictionResults>(`/api/prediction-results?scope=${scope}&date=${date}`, { ttlMs: 15_000, force })
      .then((d) => { setData(d); const u = new URL(window.location.href); u.searchParams.set("scope", scope); u.searchParams.set("date", date); window.history.replaceState(null, "", u.toString()); })
      .catch(() => setErr("予測成績を取得できませんでした。"))
      .finally(() => setBusy(false));
  }, [scope, date]);
  useEffect(() => { load(); }, [load]);

  const latest = days?.latest ?? "";
  const canNext = !!latest && !!date && periodKey(scope, shiftDate(date, scope, 1)) <= periodKey(scope, latest);
  const kpi = data?.kpi;
  const empty = !!data && data.summary.races_settled === 0;
  const races = useMemo(() => (data?.races ?? []), [data]);

  return (
    <PageShell title="予測成績" description="予測と実際の結果の突き合わせ。毎日、最終レースの後に集計します（確定結果が入ると自動で『確定』に更新）。" note={false}>
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center" }}>
        <div role="tablist" aria-label="集計の期間" style={{ display: "inline-flex", border: "1px solid var(--border)", borderRadius: 10, overflow: "hidden" }}>
          {SCOPES.map((s) => (
            <button key={s.id} role="tab" aria-selected={scope === s.id} onClick={() => setScope(s.id)}
              style={{ padding: "8px 16px", minHeight: 44, border: "none", cursor: "pointer", fontSize: 14, fontWeight: 700,
                background: scope === s.id ? "var(--accent)" : "var(--surface)", color: scope === s.id ? "#fff" : "var(--text)" }}>
              {s.label}
            </button>
          ))}
        </div>
        <div style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
          <button aria-label="前へ" onClick={() => date && setDate(shiftDate(date, scope, -1))} style={navBtn}>◀</button>
          <span style={{ minWidth: 150, textAlign: "center", fontWeight: 700 }}>{data ? periodLabel(data) : date ? dayLabel(date) : "—"}</span>
          <button aria-label="次へ" disabled={!canNext} onClick={() => date && setDate(shiftDate(date, scope, 1))} style={{ ...navBtn, opacity: canNext ? 1 : 0.35 }}>▶</button>
          {latest && date !== latest && <button onClick={() => setDate(latest)} style={{ ...navBtn, width: "auto", padding: "0 12px", fontSize: 13 }}>最新</button>}
        </div>
        {data && !empty && <StatusBadge status={data.status} />}
        {busy && <span style={dim}>読み込み中…</span>}
      </div>

      {err && (
        <div role="alert" style={{ ...card, borderColor: "var(--err)", marginTop: 12 }}>
          ⚠️ {err} <button onClick={() => load(true)} style={{ marginLeft: 8, padding: "6px 14px", minHeight: 36, cursor: "pointer" }}>再読み込み</button>
        </div>
      )}

      {days && days.days.length === 0 && !err && (
        <div style={{ ...card, marginTop: 12 }}>まだ集計がありません。開催日の最終レースの後に、自動で作られます。</div>
      )}

      {empty && <div style={{ ...card, marginTop: 12 }}>この期間は集計できるレースがありません（開催が無い、または予測・結果がそろっていません）。</div>}

      {data && kpi && !empty && (
        <div style={{ display: "grid", gap: 14, marginTop: 14 }}>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: 10 }}>
            <Tile label="◎の1着率" value={pct(kpi.honmei.win)} sub={`${kpi.honmei.n}頭中`} />
            <Tile label="◎の複勝圏（3着以内）" value={pct(kpi.honmei.top3)} sub={`連対 ${pct(kpi.honmei.top2)}`} />
            <Tile label="予測1位の1着率" value={pct(kpi.top1.win)} sub={`${kpi.top1.n}レース`} />
            <Tile label="仮想購入の回収率" value={pct(kpi.roi)} sub={`${yen(kpi.invested)} → ${yen(kpi.returned)}`} locked={!member} />
            <Tile label="的中レース（購入）" value={`${kpi.races_hit}/${kpi.races_bought}`} sub="推奨馬券をどれか当てた" locked={!member} />
            <Tile label="確率の精度" value={kpi.winner_logloss == null ? "—" : kpi.winner_logloss.toFixed(3)} sub="勝ち馬の対数損失（小さいほど良い）" />
          </div>

          <section aria-label="印ごとの着順分布">
            <h2 style={h2}>◎・○・★・消 の着順分布</h2>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 10 }}>
              {(["honmei", "pair", "star", "kesi"] as MarkKey[]).map((m) => <FinishDist key={m} mark={m} c={data.summary.marks[m] ?? EMPTY_COUNTS} />)}
            </div>
            <p style={{ ...dim, margin: "6px 0 0" }}>◎=1着優位（各レース1頭まで）・○=2連系の相手・★=10〜30倍帯で期待値が高い中穴（各レース1頭まで）・消=3着内に来る確率が低い馬（10人気以下で複勝確率8%以下。複数頭付くことも付かないこともある。消は3着以内に来る割合が低いほど当たりで、目標は5%以下）。印の定義は予測画面と同じです。</p>
          </section>

          {scope !== "day" && (
            <section aria-label="日別の推移">
              <h2 style={h2}>日別の推移</h2>
              <div style={card}><Trend series={data.series} showRoi={member} /></div>
            </section>
          )}

          {member && Object.keys(kpi.by_type).length > 0 && (
            <section aria-label="券種別の仮想購入">
              <h2 style={h2}>券種別の仮想購入（推奨馬券を上限 2,000 円/Rで買った場合）</h2>
              <div style={{ ...card, overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                  <thead><tr style={{ color: "var(--text-dim)", textAlign: "right" }}>
                    <th style={{ textAlign: "left", padding: "4px 6px" }}>券種</th><th>点数</th><th>的中</th><th>投資</th><th>払戻</th><th>回収率</th></tr></thead>
                  <tbody>
                    {Object.entries(kpi.by_type).map(([bt, v]) => (
                      <tr key={bt} style={{ borderTop: "1px solid var(--border)", textAlign: "right" }}>
                        <td style={{ textAlign: "left", padding: "6px", fontWeight: 700 }}>{v.label ?? bt}</td>
                        <td>{v.n}</td><td>{v.hit}</td><td>{yen(v.invested)}</td><td>{yen(v.returned)}</td>
                        <td style={{ fontWeight: 700, color: (v.roi ?? 0) >= 1 ? "#34d399" : "var(--text)" }}>{pct(v.roi)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          )}

          {data.highlights.length > 0 && (
            <section aria-label="注目レース">
              <h2 style={h2}>注目レース</h2>
              <div style={{ display: "grid", gap: 8 }}>
                {data.highlights.map((h, i) => (
                  <PendingLink key={i} href={`/race/${h.race_id}`} style={{ ...card, textDecoration: "none", color: "var(--text)", padding: "10px 14px", minHeight: 44 }}>
                    <span aria-hidden>{h.kind === "ana_hit" ? "🎯 " : h.kind === "big_return" ? "💰 " : "⚠️ "}</span>{h.text}
                  </PendingLink>
                ))}
              </div>
            </section>
          )}

          {scope === "day" && races.length > 0 && (
            <section aria-label="レース別の予測と結果">
              <h2 style={h2}>レース別の『予測 → 結果』</h2>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: 10 }}>
                {races.map((r) => <RaceCard key={r.race_id} r={r} member={member} />)}
              </div>
            </section>
          )}

          <div style={{ ...dim, display: "grid", gap: 4 }}>
            {data.status === "provisional" && <div>『暫定』は速報の結果での集計です。確定結果・確定払戻が入ると自動で『確定』に更新されます。払戻は、公式の払戻が無い間はオッズからの概算です。</div>}
            {data.prediction_sources && (data.prediction_sources.backfill || data.prediction_sources.shadow) ? <div>バックフィル／shadow で算出した予測を含みます（レース当日に出した予測ではありません）。</div> : null}
            {data.notes.map((n, i) => <div key={i}>{n}</div>)}
            {data.summary.races_no_result + data.summary.races_no_prediction > 0 && (
              <div>集計に含めていないレース: 結果なし {data.summary.races_no_result}・予測なし {data.summary.races_no_prediction}</div>
            )}
          </div>
        </div>
      )}
    </PageShell>
  );
}

const h2: React.CSSProperties = { fontSize: 16, fontWeight: 700, margin: "0 0 8px" };
const navBtn: React.CSSProperties = { width: 44, height: 44, border: "1px solid var(--border)", borderRadius: 10, background: "var(--surface)", color: "var(--text)", cursor: "pointer", fontSize: 14 };
