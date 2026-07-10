"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { PageShell } from "@/components/PageShell";
import { USE_MOCK, MOCK_SIMULATION_PARAMS } from "@/lib/mock";
import {
  Chart as ChartJS,
  CategoryScale, LinearScale, LogarithmicScale,
  PointElement, LineElement, BarElement,
  Filler, Tooltip, Legend,
} from "chart.js";
import { Line, Bar } from "react-chartjs-2";

ChartJS.register(
  CategoryScale, LinearScale, LogarithmicScale,
  PointElement, LineElement, BarElement,
  Filler, Tooltip, Legend,
);

// ── Types ────────────────────────────────────────────────────────────────────

interface SimParams {
  bankroll: number;
  winProb: number;
  winOdds: number;
  kellyMult: number;
  nRaces: number;
  nTrials: number;
  ruinPct: number;
}

interface SimResult {
  fStar: number; fBet: number; edge: number;
  medianFinal: number; profitRate: number; ruinRate: number;
  medianMaxDD: number; growthRate: number; ruinLevel: number;
  labels: string[];
  bands: { p10: number[]; p25: number[]; p50: number[]; p75: number[]; p90: number[] };
  finalDist: number[];
}

// ── Pure simulation ──────────────────────────────────────────────────────────

function qSorted(s: number[], p: number) {
  const idx = (p / 100) * (s.length - 1);
  const lo = Math.floor(idx), hi = Math.ceil(idx);
  return s[lo] + (s[hi] - s[lo]) * (idx - lo);
}

function runSim(p: SimParams): SimResult {
  const { bankroll, winProb, winOdds, kellyMult, nRaces, nTrials, ruinPct } = p;
  const b = winOdds - 1;
  const fStar = b > 0 ? Math.max(0, (winProb * winOdds - 1) / b) : 0;
  const fBet = kellyMult * fStar;
  const ruinLevel = bankroll * ruinPct;

  const bks = new Float64Array(nTrials).fill(bankroll);
  const peaks = new Float64Array(nTrials).fill(bankroll);
  const maxDDs = new Float64Array(nTrials);
  const ruined = new Uint8Array(nTrials);

  const step = Math.max(1, Math.ceil(nRaces / 100));
  const bandIdx: number[] = [];
  const bands: SimResult["bands"] = { p10: [], p25: [], p50: [], p75: [], p90: [] };

  function captureBand(i: number) {
    bandIdx.push(i);
    const s = Array.from(bks).sort((a, c) => a - c);
    bands.p10.push(qSorted(s, 10)); bands.p25.push(qSorted(s, 25));
    bands.p50.push(qSorted(s, 50)); bands.p75.push(qSorted(s, 75));
    bands.p90.push(qSorted(s, 90));
  }

  captureBand(0);

  for (let n = 1; n <= nRaces; n++) {
    for (let m = 0; m < nTrials; m++) {
      if (ruined[m]) continue;
      const bk = bks[m];
      const stake = fBet * bk;
      let next = Math.random() < winProb ? bk + stake * b : bk - stake;
      if (next < 0) next = 0;
      bks[m] = next;
      if (next > peaks[m]) peaks[m] = next;
      const dd = peaks[m] > 0 ? (peaks[m] - next) / peaks[m] : 0;
      if (dd > maxDDs[m]) maxDDs[m] = dd;
      if (next <= ruinLevel) ruined[m] = 1;
    }
    if (n % step === 0 || n === nRaces) captureBand(n);
  }

  const finals = Array.from(bks).sort((a, c) => a - c);
  const ddArr  = Array.from(maxDDs).sort((a, c) => a - c);

  return {
    fStar, fBet,
    edge: winProb * winOdds - 1,
    medianFinal: qSorted(finals, 50),
    profitRate:  finals.filter(x => x > bankroll).length / nTrials,
    ruinRate:    ruined.reduce((s, v) => s + v, 0) / nTrials,
    medianMaxDD: qSorted(ddArr, 50),
    growthRate:  qSorted(finals, 50) / bankroll - 1,
    ruinLevel, labels: bandIdx.map(String), bands, finalDist: finals,
  };
}

function buildHist(sorted: number[], nBins: number, ruinLevel: number, bankroll: number) {
  const min = sorted[0] ?? 0, max = sorted[sorted.length - 1] ?? 1;
  const w = max > min ? (max - min) / nBins : 1;
  const counts = new Array<number>(nBins).fill(0);
  for (const v of sorted) counts[Math.min(Math.floor((v - min) / w), nBins - 1)]++;
  return {
    labels: Array.from({ length: nBins }, (_, i) =>
      `¥${Math.round(min + (i + 0.5) * w).toLocaleString("ja-JP")}`),
    counts,
    colors: Array.from({ length: nBins }, (_, i) => {
      const c = min + (i + 0.5) * w;
      return c <= ruinLevel ? "rgba(239,68,68,0.75)" : c <= bankroll ? "rgba(249,115,22,0.75)" : "rgba(34,197,94,0.75)";
    }),
  };
}

// ── Helpers ──────────────────────────────────────────────────────────────────

const FMT = new Intl.NumberFormat("ja-JP");

function NumInput({ label, value, onChange, min, max, step }: {
  label: string; value: number; onChange: (v: number) => void;
  min: number; max: number; step: number;
}) {
  return (
    <div className="space-y-1">
      <label className="text-xs" style={{ color: "var(--text-dim)" }}>{label}</label>
      <input type="number" min={min} max={max} step={step} value={value}
             onChange={e => onChange(Number(e.target.value))}
             className="w-full rounded border bg-transparent px-3 py-1.5 text-sm"
             style={{ borderColor: "var(--border)", color: "var(--text)" }} />
    </div>
  );
}

function SliderInput({ label, value, onChange, min, max, step, unit }: {
  label: string; value: number; onChange: (v: number) => void;
  min: number; max: number; step: number; unit?: string;
}) {
  return (
    <div className="space-y-1">
      <label className="text-xs" style={{ color: "var(--text-dim)" }}>{label}</label>
      <div className="flex items-center gap-2">
        <input type="range" min={min} max={max} step={step} value={value}
               onChange={e => onChange(Number(e.target.value))} className="flex-1" />
        <input type="number" min={min} max={max} step={step} value={value}
               onChange={e => onChange(Number(e.target.value))}
               className="w-16 rounded border bg-transparent px-2 py-1 text-xs"
               style={{ borderColor: "var(--border)", color: "var(--text)" }} />
        {unit && <span className="text-xs" style={{ color: "var(--text-dim)" }}>{unit}</span>}
      </div>
    </div>
  );
}

// ── Page ─────────────────────────────────────────────────────────────────────

export default function BettingSimulationPage() {
  const [isAdmin, setIsAdmin] = useState<boolean | null>(null);
  const [bankroll, setBankroll] = useState(MOCK_SIMULATION_PARAMS.bankroll);
  const [winProbPct, setWinProbPct] = useState(MOCK_SIMULATION_PARAMS.win_prob * 100);
  const [winOdds, setWinOdds] = useState(MOCK_SIMULATION_PARAMS.win_odds);
  const [kellyMult, setKellyMult] = useState(MOCK_SIMULATION_PARAMS.kelly_fraction);
  const [nRaces, setNRaces] = useState(MOCK_SIMULATION_PARAMS.n_races);
  const [nTrials, setNTrials] = useState(MOCK_SIMULATION_PARAMS.n_trials);
  const [ruinPct, setRuinPct] = useState(MOCK_SIMULATION_PARAMS.ruin_threshold * 100);
  const [result, setResult] = useState<SimResult | null>(null);
  const [running, setRunning] = useState(false);
  const [logScale, setLogScale] = useState(false);
  const [toast, setToast] = useState<string | null>(null);

  useEffect(() => {
    if (USE_MOCK) { setIsAdmin(true); return; }
    fetch("/api/v1/auth/status", { credentials: "include" })
      .then(r => r.ok ? r.json() : { logged_in: false, is_admin: false })
      .then(d => setIsAdmin(!!d.logged_in && !!d.is_admin))
      .catch(() => setIsAdmin(false));
  }, []);

  const winProb = winProbPct / 100;
  const b = winOdds - 1;
  const fStar = b > 0 ? Math.max(0, (winProb * winOdds - 1) / b) : 0;
  const fBet = kellyMult * fStar;
  const edge = winProb * winOdds - 1;

  function showToast(msg: string) { setToast(msg); setTimeout(() => setToast(null), 3000); }

  function handleRun() {
    setRunning(true);
    setTimeout(() => {
      setResult(runSim({ bankroll, winProb, winOdds, kellyMult, nRaces, nTrials, ruinPct: ruinPct / 100 }));
      setRunning(false);
    }, 10);
  }

  function handleLoadStorage() {
    const raw = typeof window !== "undefined" ? localStorage.getItem("betting_last_optimize") : null;
    if (!raw) { showToast("保存された最適化結果がありません"); return; }
    try {
      const d = JSON.parse(raw) as { race_id?: string; bankroll?: number; win_prob?: number | null; win_odds?: number | null; kelly_fraction?: number };
      if (d.bankroll)       setBankroll(d.bankroll);
      if (d.win_prob)        setWinProbPct(d.win_prob * 100);
      if (d.win_odds)        setWinOdds(d.win_odds);
      if (d.kelly_fraction)  setKellyMult(d.kelly_fraction);
      showToast(`最適化結果 (${d.race_id ?? "—"}) から引き継ぎました`);
    } catch { showToast("引き継ぎに失敗しました"); }
  }

  if (isAdmin === null) return (
    <PageShell title="馬券シミュレーション" description="モンテカルロ軍資金シミュレーション（AREA-12）">
      <div className="card flex flex-col items-center gap-4 py-12">
        <p className="text-sm" style={{ color: "var(--text-dim)" }}>認証確認中…</p>
      </div>
    </PageShell>
  );

  if (!isAdmin) return (
    <PageShell title="馬券シミュレーション" description="モンテカルロ軍資金シミュレーション（AREA-12）">
      <div className="card flex flex-col items-center gap-4 py-12">
        <span style={{ fontSize: 40 }}>🔒</span>
        <p className="font-semibold">管理者専用ページ</p>
        <p className="text-sm" style={{ color: "var(--text-dim)" }}>馬券シミュレーション機能は管理者限定です。</p>
        <Link href="/login" className="btn">ログインする</Link>
      </div>
    </PageShell>
  );

  // Chart data
  const nBins = Math.ceil(Math.log2(nTrials)) + 1;
  const hist  = result ? buildHist(result.finalDist, nBins, result.ruinLevel, bankroll) : null;

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const trajData: any = result ? {
    labels: result.labels,
    datasets: [
      { data: result.bands.p10, fill: false, borderColor: "transparent", pointRadius: 0 },
      { data: result.bands.p90, fill: { target: 0 }, backgroundColor: "rgba(59,130,246,0.12)", borderColor: "transparent", pointRadius: 0 },
      { data: result.bands.p25, fill: false, borderColor: "transparent", pointRadius: 0 },
      { data: result.bands.p75, fill: { target: 2 }, backgroundColor: "rgba(59,130,246,0.22)", borderColor: "transparent", pointRadius: 0 },
      { label: "p50", data: result.bands.p50, fill: false, borderColor: "#3b82f6", borderWidth: 2.5, pointRadius: 0 },
      { data: result.bands.p50.map(() => bankroll), fill: false, borderColor: "rgba(148,163,184,0.45)", borderWidth: 1.5, borderDash: [5, 4], pointRadius: 0 },
      { data: result.bands.p50.map(() => result.ruinLevel), fill: false, borderColor: "rgba(239,68,68,0.45)", borderWidth: 1.5, borderDash: [5, 4], pointRadius: 0 },
    ],
  } : null;

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const trajOptions: any = {
    responsive: true, maintainAspectRatio: false, animation: false,
    plugins: { legend: { display: false }, tooltip: { callbacks: {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      label: (ctx: any) => ctx.datasetIndex === 4 ? `p50: ¥${FMT.format(Math.round(ctx.parsed.y))}` : "",
    }}},
    scales: {
      x: { ticks: { maxTicksLimit: 10, color: "rgba(148,163,184,0.7)" }, grid: { color: "rgba(148,163,184,0.08)" } },
      y: {
        type: logScale ? "logarithmic" : "linear",
        ticks: { color: "rgba(148,163,184,0.7)", callback: (v: unknown) => typeof v === "number" ? `¥${FMT.format(Math.round(v))}` : v },
        grid: { color: "rgba(148,163,184,0.08)" },
      },
    },
  };

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const histData: any = hist ? { labels: hist.labels, datasets: [{ data: hist.counts, backgroundColor: hist.colors, borderWidth: 0 }] } : null;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const histOptions: any = {
    responsive: true, maintainAspectRatio: false, animation: false,
    plugins: { legend: { display: false } },
    scales: {
      x: { ticks: { color: "rgba(148,163,184,0.7)", maxRotation: 45, maxTicksLimit: 8 }, grid: { display: false } },
      y: { ticks: { color: "rgba(148,163,184,0.7)" }, grid: { color: "rgba(148,163,184,0.08)" } },
    },
  };

  const kpis = result ? [
    { label: "中央値 最終軍資金", value: `¥${FMT.format(Math.round(result.medianFinal))}`, color: result.medianFinal > bankroll ? "var(--ok)" : "var(--warn)" },
    { label: "利益達成率", value: `${(result.profitRate * 100).toFixed(1)}%`, color: result.profitRate >= 0.6 ? "var(--ok)" : result.profitRate >= 0.4 ? "var(--warn)" : "var(--err)" },
    { label: "破産確率", value: `${(result.ruinRate * 100).toFixed(1)}%`, color: result.ruinRate <= 0.05 ? "var(--ok)" : result.ruinRate <= 0.15 ? "var(--warn)" : "var(--err)" },
    { label: "中央値 最大DD", value: `${(result.medianMaxDD * 100).toFixed(1)}%`, color: result.medianMaxDD <= 0.3 ? "var(--ok)" : result.medianMaxDD <= 0.5 ? "var(--warn)" : "var(--err)" },
    { label: "期待成長率", value: `${result.growthRate >= 0 ? "+" : ""}${(result.growthRate * 100).toFixed(1)}%`, color: result.growthRate >= 0 ? "var(--ok)" : "var(--err)" },
  ] : null;

  return (
    <PageShell title="馬券シミュレーション" description="モンテカルロ軍資金シミュレーション（AREA-12）">
      {toast && (
        <div className="fixed top-4 right-4 z-50 rounded-lg px-4 py-2 text-sm shadow-lg"
             style={{ background: "var(--surface2)", color: "var(--text)", border: "1px solid var(--border)" }}>
          {toast}
        </div>
      )}

      <div className="flex flex-col gap-4 md:flex-row md:items-start">
        {/* ── Config Panel ── */}
        <div className="card space-y-4 md:w-72 md:flex-shrink-0">
          <h2 className="text-sm font-semibold">シミュレーション設定</h2>

          {/* Edge / fStar preview */}
          <div className="rounded-lg p-3 text-xs space-y-1.5" style={{ background: "var(--surface2)" }}>
            <div className="flex justify-between">
              <span style={{ color: "var(--text-dim)" }}>期待値 (edge)</span>
              <span style={{ color: edge > 0 ? "var(--ok)" : "var(--err)", fontWeight: 700 }}>
                {edge > 0 ? "+" : ""}{(edge * 100).toFixed(1)}% {edge > 0 ? "✓" : "⚠"}
              </span>
            </div>
            <div className="flex justify-between">
              <span style={{ color: "var(--text-dim)" }}>フルKelly f*</span>
              <span>{(fStar * 100).toFixed(2)}%</span>
            </div>
            <div className="flex justify-between">
              <span style={{ color: "var(--text-dim)" }}>実賭けフラクション</span>
              <span style={{ color: "var(--accent)", fontWeight: 700 }}>{(fBet * 100).toFixed(2)}%</span>
            </div>
          </div>

          <NumInput label="初期軍資金（円）" value={bankroll} onChange={setBankroll} min={1000} max={10000000} step={1000} />
          <SliderInput label="期待勝率（%）" value={winProbPct} onChange={setWinProbPct} min={1} max={99} step={0.5} />
          <SliderInput label="単勝オッズ（倍）" value={winOdds} onChange={setWinOdds} min={1.1} max={50} step={0.1} />

          {/* Kelly multiplier */}
          <div className="space-y-1">
            <label className="text-xs" style={{ color: "var(--text-dim)" }}>Kelly 倍率 (α)</label>
            <div className="flex items-center gap-2">
              <input type="range" min={0.05} max={1} step={0.05} value={kellyMult}
                     onChange={e => setKellyMult(Number(e.target.value))} className="flex-1" />
              <input type="number" min={0.01} max={1} step={0.01} value={kellyMult}
                     onChange={e => setKellyMult(Number(e.target.value))}
                     className="w-14 rounded border bg-transparent px-2 py-1 text-xs"
                     style={{ borderColor: "var(--border)", color: "var(--text)" }} />
            </div>
            <div className="flex gap-1.5">
              {([["1/10", 0.1], ["1/4", 0.25], ["1/2", 0.5], ["Full", 1.0]] as [string, number][]).map(([lbl, v]) => (
                <button key={lbl} type="button" onClick={() => setKellyMult(v)}
                        className="rounded px-2 py-0.5 text-xs"
                        style={{
                          background: kellyMult === v ? "rgba(59,130,246,0.2)" : "transparent",
                          border: `1px solid ${kellyMult === v ? "var(--accent)" : "var(--border)"}`,
                          color: kellyMult === v ? "var(--accent)" : "var(--text-dim)",
                        }}>
                  {lbl}
                </button>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1">
              <label className="text-xs" style={{ color: "var(--text-dim)" }}>レース数</label>
              <select value={nRaces} onChange={e => setNRaces(Number(e.target.value))}
                      className="w-full rounded border bg-transparent px-2 py-1.5 text-xs"
                      style={{ borderColor: "var(--border)", color: "var(--text)", background: "var(--surface)" }}>
                {[50, 100, 200, 500].map(n => <option key={n} value={n}>{n}</option>)}
              </select>
            </div>
            <div className="space-y-1">
              <label className="text-xs" style={{ color: "var(--text-dim)" }}>試行回数</label>
              <select value={nTrials} onChange={e => setNTrials(Number(e.target.value))}
                      className="w-full rounded border bg-transparent px-2 py-1.5 text-xs"
                      style={{ borderColor: "var(--border)", color: "var(--text)", background: "var(--surface)" }}>
                {[500, 1000, 2000].map(n => <option key={n} value={n}>{n}</option>)}
              </select>
            </div>
          </div>

          <SliderInput label="破産閾値（初期軍資金の %）" value={ruinPct} onChange={setRuinPct} min={1} max={50} step={1} unit="%" />

          <button type="button" className="btn w-full" onClick={handleRun} disabled={running}>
            {running ? "計算中…" : "▶ シミュレーション実行"}
          </button>

          <button type="button" onClick={handleLoadStorage}
                  className="w-full rounded border px-3 py-1.5 text-xs"
                  style={{ borderColor: "var(--border)", color: "var(--text-dim)" }}>
            📊 馬券最適化から引き継ぎ
          </button>
        </div>

        {/* ── Results Panel ── */}
        <div className="min-w-0 flex-1 space-y-4">
          {!result ? (
            <div className="card flex items-center justify-center py-16">
              <p className="text-sm" style={{ color: "var(--text-dim)" }}>
                設定を入力して「▶ シミュレーション実行」を押してください
              </p>
            </div>
          ) : (
            <>
              {edge <= 0 && (
                <div className="rounded-lg px-4 py-2 text-sm"
                     style={{ background: "rgba(239,68,68,0.1)", color: "var(--err)", border: "1px solid rgba(239,68,68,0.3)" }}>
                  ⚠ 期待値がマイナス（edge = {(edge * 100).toFixed(1)}%）。このパラメータでは長期的に損失が確定します。
                </div>
              )}

              {/* KPI cards */}
              <dl className="grid grid-cols-2 gap-3 md:grid-cols-5">
                {kpis!.map(kpi => (
                  <div key={kpi.label} className="rounded-lg p-3" style={{ background: "var(--surface2)" }}>
                    <dt className="text-xs leading-snug" style={{ color: "var(--text-dim)" }}>{kpi.label}</dt>
                    <dd className="mt-1 text-sm font-bold" style={{ color: kpi.color }}>{kpi.value}</dd>
                  </div>
                ))}
              </dl>

              {/* Trajectory chart */}
              <div className="card space-y-2">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-semibold">
                    軍資金推移 <span className="font-normal text-xs" style={{ color: "var(--text-dim)" }}>（{FMT.format(nTrials)} 試行）</span>
                  </h3>
                  <button type="button" onClick={() => setLogScale(l => !l)}
                          className="rounded border px-2 py-0.5 text-xs"
                          style={{ borderColor: "var(--border)", color: "var(--text-dim)" }}>
                    {logScale ? "対数 ON" : "対数 OFF"}
                  </button>
                </div>
                <div style={{ height: 280 }}>
                  <Line data={trajData} options={trajOptions} />
                </div>
                <div className="flex flex-wrap gap-4 text-xs" style={{ color: "var(--text-dim)" }}>
                  <span className="flex items-center gap-1.5">
                    <span style={{ width: 20, height: 3, background: "#3b82f6", display: "inline-block", borderRadius: 2 }} />
                    p50 中央値
                  </span>
                  <span className="flex items-center gap-1.5">
                    <span style={{ width: 16, height: 10, background: "rgba(59,130,246,0.22)", display: "inline-block" }} />
                    p25–p75
                  </span>
                  <span className="flex items-center gap-1.5">
                    <span style={{ width: 16, height: 10, background: "rgba(59,130,246,0.12)", display: "inline-block" }} />
                    p10–p90
                  </span>
                  <span className="flex items-center gap-1.5">
                    <span style={{ width: 18, height: 0, borderTop: "2px dashed rgba(148,163,184,0.5)", display: "inline-block" }} />
                    初期軍資金
                  </span>
                  <span className="flex items-center gap-1.5">
                    <span style={{ width: 18, height: 0, borderTop: "2px dashed rgba(239,68,68,0.5)", display: "inline-block" }} />
                    破産閾値
                  </span>
                </div>
              </div>

              {/* Distribution histogram */}
              <div className="card space-y-2">
                <h3 className="text-sm font-semibold">
                  最終軍資金の分布 <span className="font-normal text-xs" style={{ color: "var(--text-dim)" }}>（{FMT.format(nTrials)} 試行）</span>
                </h3>
                <div style={{ height: 200 }}>
                  <Bar data={histData} options={histOptions} />
                </div>
                <div className="flex gap-4 text-xs" style={{ color: "var(--text-dim)" }}>
                  <span><span style={{ color: "rgba(34,197,94,1)" }}>■</span> 利益（＞初期）</span>
                  <span><span style={{ color: "rgba(249,115,22,1)" }}>■</span> 損失</span>
                  <span><span style={{ color: "rgba(239,68,68,1)" }}>■</span> 破産</span>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </PageShell>
  );
}
