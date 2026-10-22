"use client";

import { classifyFetchThrow, classifyHttpFailure, classifyRenderError, isSessionRedirect, sessionExpired } from "@/lib/betFetch";
import { Fragment, useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { USE_MOCK, getMockBetRecommendation } from "@/lib/mock";
import { BetGroup, groupBets, nameMap, trioPartnerNums } from "@/lib/betGrouping";

type OddsMode = "auto" | "actual" | "predicted";
type OddsPhase = "latest" | "t60" | "t15";

type Candidate = {
  bet_type: string;
  pair?: number[];
  predicted_odds?: number;
  odds_gap?: number;
  pair_label?: string;
  horse_names?: string[];
  odds?: number;
  prob?: number;
  ev?: number;
  /** EV の較正が有効なときの、較正後の期待値（過去の実績に合わせて補正した期待回収率）。取得後に ev へ入れ替え、元の値は ev_raw に残す */
  ev_calibrated?: number;
  ev_calibration_validated?: boolean;
  ev_raw?: number;
  kelly_fraction?: number;
  bet_amount: number;
  expected_return?: number;
};

type ProbabilityModel = { kind: string; alpha2?: number; alpha3?: number; fitted_to?: string[] };

type Recommendation = {
  probability_model?: ProbabilityModel | null;
  odds_source?: "actual" | "predicted" | null;
  odds_phase?: string | null;
  odds_as_of?: string | null;
  odds_note?: string | null;
  available_odds?: { phase: string; as_of?: string | null }[];
  total_bet: number;
  expected_return: number;
  roi_pct?: number;
  expected_roi_calibrated?: number | null;
  expected_return_calibrated?: number;
  roi_range_calibrated?: [number, number] | null;
  roi_calibrated_capped?: boolean;
  /** 取得後の整形で立てる: 期待値・期待払戻・期待回収率が較正後の値になっている */
  calibrated?: boolean;
  candidates: Candidate[];
};

/** EV の較正が有効なら、買い目の EV・期待払戻を較正後の値に入れ替える（グルーピング・色分けは入れ替え後の値を使う） */
function applyCalibration(rec: Recommendation): Recommendation {
  if (rec.expected_roi_calibrated == null) return rec;
  return {
    ...rec,
    calibrated: true,
    expected_return: rec.expected_return_calibrated ?? rec.expected_return,
    candidates: rec.candidates.map((c) => ({ ...c, ev_raw: c.ev, ev: c.ev_calibrated ?? c.ev })),
  };
}

const BET_LABEL: Record<string, string> = {
  tansho: "単勝",
  fukusho: "複勝",
  umaren: "馬連",
  wide: "ワイド",
  umatan: "馬単",
  sanrenpuku: "3連複",
};

/** 1 レースの購入合計の既定上限（円）。バックエンドの DEFAULT_MAX_TOTAL_BET と揃える。0 / 空欄は上限なし */
const DEFAULT_MAX_TOTAL_BET = 2000;
const CAP_STORAGE_KEY = "betting_max_total_bet";

/** 表示順（券種ごとにまとめる）と見出しの色 */
const BET_ORDER = ["tansho", "fukusho", "umaren", "wide", "umatan", "sanrenpuku"];
const BET_COLOR: Record<string, string> = {
  tansho: "#f59e0b", fukusho: "#22c55e", umaren: "#60a5fa", wide: "#a78bfa", umatan: "#f472b6", sanrenpuku: "#fb7185",
};

const MODE_LABEL: Record<OddsMode, string> = { auto: "自動", actual: "実オッズ", predicted: "予測オッズ" };
const PHASE_LABEL: Record<OddsPhase, string> = { latest: "最新", t60: "T-60（1時間前）", t15: "T-15（15分前）" };

function hhmm(iso?: string | null): string {
  if (!iso) return "";
  const m = iso.match(/T(\d{2}:\d{2})/);
  return m ? m[1] : "";
}

const TH: React.CSSProperties = { padding: "8px 10px", textAlign: "right", fontSize: 12, color: "var(--text-dim)", fontWeight: 600 };
const TD: React.CSSProperties = { padding: "8px 10px", textAlign: "right", fontSize: 13, borderTop: "1px solid var(--border)" };

function Tag({ children }: { children: React.ReactNode }) {
  return <span style={{ fontSize: 11, color: "var(--text-dim)", border: "1px solid var(--border)", borderRadius: 4, padding: "0 5px", marginRight: 6 }}>{children}</span>;
}

function Arrow() {
  return <span style={{ margin: "0 8px 0 2px", color: "var(--text-dim)" }}>→</span>;
}

const NOWRAP_TH: React.CSSProperties = { ...TH, whiteSpace: "nowrap" };
const NOWRAP_TD: React.CSSProperties = { ...TD, whiteSpace: "nowrap" };

function evColor(ev?: number, cal = false): string {
  if (ev == null) return "var(--text-dim)";
  if (cal) return ev >= 1.1 ? "#22c55e" : ev >= 1.0 ? "#60a5fa" : ev >= 0.9 ? "var(--text-dim)" : "#f87171";      // 較正後: 買い推奨度の閾値に合わせる
  return ev >= 1.3 ? "#22c55e" : ev >= 1.0 ? "#60a5fa" : "#f87171";
}

/** AI 予測（確率）× オッズ → 期待値 → Kelly 配分の推奨馬券。計算本体はサーバ（/api/v1/betting/optimize）。 */
/** 馬番 → 印（買い目の各馬番に印を添えるための表示用。週間予測の『推奨馬券』タブで使う） */
export type MarkBySeq = Record<number, { sym: string; color: string; kesi?: boolean }>;

export function BetRecommend({ raceId, hasPrediction, marks }: { raceId: string; hasPrediction: boolean; marks?: MarkBySeq }) {
  const [bankroll, setBankroll] = useState("100000");
  const [cap, setCap] = useState(String(DEFAULT_MAX_TOTAL_BET));
  const [capReady, setCapReady] = useState(false);
  const [open, setOpen] = useState<Record<string, boolean>>({});
  const [rec, setRec] = useState<Recommendation | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [errReload, setErrReload] = useState(false);
  const [needLogin, setNeedLogin] = useState(false);
  const [mode, setMode] = useState<OddsMode>("auto");
  const [phase, setPhase] = useState<OddsPhase>("latest");

  // 前回指定した購入上限を引き継ぐ（未設定なら既定の 2000 円）
  useEffect(() => {
    try {
      const saved = localStorage.getItem(CAP_STORAGE_KEY);
      if (saved !== null) setCap(saved);
    } catch {
      /* localStorage が使えない環境では既定値のまま */
    }
    setCapReady(true);
  }, []);

  const run = useCallback(async () => {
    setError(null);
    setErrReload(false);
    setNeedLogin(false);
    const amount = Number(bankroll);
    if (!Number.isFinite(amount) || amount < 100) {
      setError("資金は 100 円以上で入力してください");
      return;
    }
    const capNum = cap.trim() === "" ? 0 : Number(cap);
    if (!Number.isFinite(capNum) || capNum < 0 || (capNum > 0 && capNum < 100)) {
      setError("購入上限は 100 円以上で入力してください（空欄か 0 で上限なし）");
      return;
    }
    const maxTotalBet = capNum === 0 ? null : capNum;
    try {
      localStorage.setItem(CAP_STORAGE_KEY, cap);
    } catch {
      /* 保存できなくても計算は続ける */
    }
    setLoading(true);
    try {
      if (USE_MOCK) {
        await new Promise((r) => setTimeout(r, 300));
        setRec(getMockBetRecommendation(raceId, amount, mode, phase, maxTotalBet) as Recommendation);
        return;
      }
      let res: Response;
      try {
        res = await fetch("/api/v1/betting/optimize", {
          method: "POST",
          credentials: "include",
          redirect: "manual", // ログイン画面（別オリジン）へ飛ばされたら、例外ではなく『セッション切れ』として検知する
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            race_id: raceId,
            bankroll: amount,
            max_total_bet: maxTotalBet,
            odds_mode: mode,
            odds_phase: phase,
            bet_types: mode === "predicted" ? ["tansho", "fukusho"] : ["tansho", "fukusho", "umaren", "wide", "sanrenpuku"],
          }),
        });
      } catch (e) {
        const f = classifyFetchThrow(typeof navigator !== "undefined" ? navigator.onLine : undefined);
        console.warn("[BetRecommend] fetch が例外:", e);
        setError(f.message);
        setErrReload(f.reload);
        return;
      }
      if (isSessionRedirect(res)) {
        const f = sessionExpired();
        setError(f.message);
        setErrReload(f.reload);
        return;
      }
      const body = await res.json().catch(() => ({}));
      if (res.status === 401) {
        setNeedLogin(true);
        return;
      }
      if (!res.ok) {
        const f = classifyHttpFailure(res.status, body?.error);
        setError(f.message);
        setErrReload(f.reload);
        return;
      }
      try {
        setRec(applyCalibration(body as Recommendation));
      } catch (e) {
        console.error("[BetRecommend] 応答の取り込みで例外:", e, body);
        const f = classifyRenderError();
        setError(f.message);
        setErrReload(f.reload);
      }
    } finally {
      setLoading(false);
    }
  }, [raceId, bankroll, cap, mode, phase]);

  useEffect(() => {
    setRec(null);
    if (hasPrediction && capReady) run();
    // 予測が揃ったとき・レース/オッズ条件が変わったときに自動計算（資金の編集では再計算しない）
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [raceId, hasPrediction, capReady, mode, phase]);

  // 買い目の馬番に添える印（馬番が印を持たないときは何も出さない）
  const markChip = (n: number) => {
    const m = marks?.[n];
    if (!m) return null;
    return (
      <span key={`mk-${n}`} aria-label={`${n}番の印 ${m.sym}`} style={{ color: m.color, fontWeight: 800, marginRight: 2, ...(m.kesi ? { border: "1px solid #94a3b8", borderRadius: 3, padding: "0 3px", fontSize: 11 } : {}) }}>
        {m.sym}
      </span>
    );
  };

  if (!hasPrediction) {
    return (
      <div style={{ textAlign: "center", padding: 40, color: "var(--text-dim)" }}>
        AI 予測がまだありません。「🤖 AI予測・期待値」タブで予測を実行すると、推奨馬券を計算します。
      </div>
    );
  }

  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12, flexWrap: "wrap" }}>
        <span style={{ fontSize: 12, color: "var(--text-dim)" }}>オッズ</span>
        {(Object.keys(MODE_LABEL) as OddsMode[]).map((m) => (
          <button
            key={m}
            onClick={() => setMode(m)}
            style={{ padding: "5px 12px", borderRadius: 6, fontSize: 12, cursor: "pointer", border: "1px solid var(--border)",
              background: mode === m ? "#1e3a5f" : "transparent", color: mode === m ? "#60a5fa" : "var(--text-dim)", fontWeight: mode === m ? 700 : 500 }}
          >
            {MODE_LABEL[m]}
          </button>
        ))}
        {mode !== "predicted" && (
          <select
            value={phase}
            onChange={(e) => setPhase(e.target.value as OddsPhase)}
            style={{ marginLeft: 8, padding: "5px 8px", background: "var(--surface2)", border: "1px solid var(--border)", borderRadius: 6, color: "var(--text)", fontSize: 12 }}
          >
            {(Object.keys(PHASE_LABEL) as OddsPhase[]).map((ph) => (
              <option key={ph} value={ph}>{PHASE_LABEL[ph]}</option>
            ))}
          </select>
        )}
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14, flexWrap: "wrap" }}>
        <label style={{ fontSize: 12, color: "var(--text-dim)" }}>
          資金（円）
          <input
            type="number"
            min={100}
            step={1000}
            value={bankroll}
            onChange={(e) => setBankroll(e.target.value)}
            style={{ marginLeft: 8, width: 120, padding: "6px 8px", background: "var(--surface2)", border: "1px solid var(--border)", borderRadius: 6, color: "var(--text)", fontSize: 13 }}
          />
        </label>
        <label style={{ fontSize: 12, color: "var(--text-dim)" }} title="1 レースの購入合計の上限。Kelly が示す額がこれより小さいときはその額。空欄か 0 で上限なし">
          1R 購入上限（円）
          <input
            type="number"
            min={0}
            step={500}
            value={cap}
            placeholder="上限なし"
            onChange={(e) => setCap(e.target.value)}
            style={{ marginLeft: 8, width: 100, padding: "6px 8px", background: "var(--surface2)", border: "1px solid var(--border)", borderRadius: 6, color: "var(--text)", fontSize: 13 }}
          />
        </label>
        <button
          onClick={run}
          disabled={loading}
          style={{ background: "#1e3a5f", color: "#60a5fa", border: "1px solid rgba(59,130,246,0.3)", padding: "7px 14px", borderRadius: 6, fontSize: 12, fontWeight: 600, cursor: loading ? "not-allowed" : "pointer", opacity: loading ? 0.6 : 1 }}
        >
          {loading ? "⏳ 計算中…" : "🔄 再計算"}
        </button>
      </div>

      {needLogin && (
        <div style={{ padding: "12px 16px", borderRadius: 8, background: "rgba(59,130,246,0.08)", border: "1px solid rgba(59,130,246,0.25)", fontSize: 13 }}>
          🔒 推奨馬券はログインすると表示されます。{" "}
          <Link href={`/login?next=/race/${raceId}`} style={{ color: "var(--accent)" }}>ログインする →</Link>
        </div>
      )}
      {error && (
        <div role="alert" style={{ padding: "12px 16px", borderRadius: 8, background: "rgba(239,68,68,0.08)", color: "var(--err)", fontSize: 13, display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
          <span style={{ flex: "1 1 240px" }}>{error}</span>
          <button type="button" onClick={() => run()} style={{ minHeight: 44, padding: "0 14px", borderRadius: 8, border: "1px solid var(--border)", background: "transparent", color: "var(--text)", cursor: "pointer", fontSize: 13 }}>
            再試行
          </button>
          {errReload && (
            <button type="button" onClick={() => window.location.reload()} style={{ minHeight: 44, padding: "0 14px", borderRadius: 8, border: "1px solid var(--accent)", background: "rgba(59,130,246,0.15)", color: "var(--accent)", cursor: "pointer", fontSize: 13 }}>
              ページを再読み込み
            </button>
          )}
        </div>
      )}

      {rec && (
        <>
          <div style={{ marginBottom: 12, display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap", fontSize: 12 }}>
            {rec.odds_source === "predicted" ? (
              <span style={{ padding: "3px 10px", borderRadius: 4, background: "rgba(251,191,36,0.12)", color: "#fbbf24" }}>
                予測オッズ（暫定・前日時点のモデル出力）
              </span>
            ) : rec.odds_source === "actual" ? (
              <span style={{ padding: "3px 10px", borderRadius: 4, background: rec.odds_phase === "t15" ? "rgba(34,197,94,0.12)" : "rgba(251,191,36,0.12)", color: rec.odds_phase === "t15" ? "#22c55e" : "#fbbf24" }}>
                {rec.odds_phase === "t60" || rec.odds_phase === "t15"
                  ? `実オッズ ${rec.odds_phase === "t60" ? "T-60" : "T-15"} ${hhmm(rec.odds_as_of)} 時点${rec.odds_phase === "t60" ? "（暫定）" : ""}`
                  : "実オッズ（取得済みの最新値）"}
              </span>
            ) : null}
            {rec.odds_note && <span style={{ color: "var(--text-dim)" }}>{rec.odds_note}</span>}
          </div>
          <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginBottom: 14 }}>
            {[
              ["購入合計", `¥${rec.total_bet.toLocaleString()}`],
              ["期待払戻", `¥${Math.round(rec.expected_return).toLocaleString()}`],
              [rec.calibrated ? "期待回収率（較正後）" : "期待回収率", rec.total_bet > 0 ? `${((rec.expected_return / rec.total_bet) * 100).toFixed(rec.calibrated ? 0 : 1)}%${rec.roi_calibrated_capped ? "以上" : ""}` : "—"],
            ].map(([k, v]) => (
              <div key={k} style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 8, padding: "10px 16px" }}>
                <div style={{ fontSize: 11, color: "var(--text-dim)" }}>{k}</div>
                <div style={{ fontSize: 18, fontWeight: 700, color: "#fff" }}>{v}</div>
              </div>
            ))}
          </div>

          {rec.calibrated && rec.roi_range_calibrated && (
            <p style={{ fontSize: 11, color: "var(--text-dim)", margin: "-6px 0 12px", lineHeight: 1.7 }}>
              期待回収率（較正後）は目安です。このルールで買った場合の過去の実現回収率は、年によって {Math.round(rec.roi_range_calibrated[0] * 100)}〜{Math.round(rec.roi_range_calibrated[1] * 100)}% と動きました（2022〜2026 年）。
            </p>
          )}

          {rec.candidates.length === 0 ? (
            <div style={{ textAlign: "center", padding: 32, color: "var(--text-dim)", background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 10 }}>
              期待値が基準（1.05）を超える買い目がありません。このレースは見送りが妥当です。
            </div>
          ) : (
            <div style={{ overflowX: "auto" }}>
              {(() => {
                const groups = groupBets(rec.candidates);
                const names = nameMap(rec.candidates);
                const showPredicted = rec.candidates.some((c) => c.predicted_odds != null);
                const cols = showPredicted ? 7 : 5;
                const sections = BET_ORDER.map((bt) => ({ bt, gs: groups.filter((g) => g.bet_type === bt) })).filter((s) => s.gs.length > 0);
                const chip = (n: number, extra?: string) => (
                  <span key={`${n}-${extra ?? ""}`} style={{ display: "inline-flex", alignItems: "baseline", gap: 3, marginRight: 6, padding: "1px 7px", borderRadius: 5, background: "rgba(125,211,252,0.12)", color: "#7dd3fc", fontWeight: 800 }}>
                    {markChip(n)}{n}
                    {names.get(n) && <span style={{ fontWeight: 400, fontSize: 11, color: "var(--text-dim)" }}>{names.get(n)}</span>}
                    {extra && <span style={{ fontWeight: 600, fontSize: 11, color: "#fff" }}>{extra}</span>}
                  </span>
                );
                const describe = (g: BetGroup) => {
                  if (g.kind === "single") {
                    const c = g.items[0];
                    return (
                      <>
                        {(c.pair ?? []).map((n) => markChip(n))}
                        <span style={{ color: "#7dd3fc", fontWeight: 800, marginRight: 8 }}>{c.pair_label}</span>
                        <span style={{ color: "var(--text-dim)", fontSize: 12 }}>{(c.horse_names ?? []).filter(Boolean).join(" - ")}</span>
                      </>
                    );
                  }
                  const uniform = g.partners.every((p) => p.amount === g.partners[0].amount);
                  const partners = g.partners.map((p) => chip(p.num, uniform ? undefined : `¥${p.amount.toLocaleString()}`));
                  const uni = uniform && g.partners[0] ? <span style={{ fontSize: 12, color: "var(--text-dim)" }}>各 ¥{g.partners[0].amount.toLocaleString()}</span> : null;
                  if (g.bet_type === "sanrenpuku" && g.kind === "axis1") {
                    return <><Tag>軸1頭</Tag>{chip(g.axis[0])}<Arrow />{trioPartnerNums(g).map((n) => chip(n))}<span style={{ fontSize: 12, color: "var(--text-dim)" }}>の2頭 ボックス</span></>;
                  }
                  if (g.bet_type === "sanrenpuku") {
                    return <><Tag>軸2頭</Tag>{g.axis.map((n) => chip(n))}<Arrow />{partners}{uni}</>;
                  }
                  if (g.kind === "axis1_first") return <><Tag>1着</Tag>{chip(g.axis[0])}<Arrow /><Tag>2着</Tag>{partners}{uni}</>;
                  if (g.kind === "axis1_second") return <><Tag>2着</Tag>{chip(g.axis[0])}<Arrow /><Tag>1着</Tag>{partners}{uni}</>;
                  return <><Tag>軸</Tag>{chip(g.axis[0])}<Arrow />{partners}{uni}</>;
                };
                return (
                  <>
                    {/* 券種ごとの点数と小計 */}
                    <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 10 }}>
                      {sections.map((s) => (
                        <span key={s.bt} style={{ fontSize: 12, padding: "4px 10px", borderRadius: 6, border: `1px solid ${BET_COLOR[s.bt]}55`, color: BET_COLOR[s.bt], background: `${BET_COLOR[s.bt]}14` }}>
                          {BET_LABEL[s.bt] ?? s.bt} {s.gs.reduce((n, g) => n + g.items.length, 0)}点 ¥{s.gs.reduce((n, g) => n + g.total, 0).toLocaleString()}
                        </span>
                      ))}
                    </div>
                    <table style={{ width: "100%", borderCollapse: "collapse", background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 10, overflow: "hidden" }}>
                      <thead style={{ background: "var(--surface2)" }}>
                        <tr>
                          <th style={{ ...TH, textAlign: "left" }}>買い目</th>
                          <th style={NOWRAP_TH}>点数</th>
                          <th style={NOWRAP_TH}>オッズ</th>
                          <th style={NOWRAP_TH}>{rec.calibrated ? "期待値（較正後）" : "期待値"}</th>
                          <th style={NOWRAP_TH}>合計</th>
                        </tr>
                      </thead>
                      {sections.map((s) => (
                        <tbody key={s.bt}>
                          <tr style={{ background: `${BET_COLOR[s.bt]}14` }}>
                            <td colSpan={5} style={{ padding: "8px 12px", borderTop: "1px solid var(--border)", borderLeft: `4px solid ${BET_COLOR[s.bt]}`, fontWeight: 700, fontSize: 13, color: BET_COLOR[s.bt] }}>
                              {BET_LABEL[s.bt] ?? s.bt}
                              <span style={{ marginLeft: 10, fontWeight: 500, fontSize: 12, color: "var(--text-dim)" }}>
                                {s.gs.reduce((n, g) => n + g.items.length, 0)}点 ／ 小計 ¥{s.gs.reduce((n, g) => n + g.total, 0).toLocaleString()}
                              </span>
                            </td>
                          </tr>
                          {s.gs.map((g) => {
                            const multi = g.items.length > 1;
                            const isOpen = !!open[g.key];
                            return (
                              <Fragment key={g.key}>
                                <tr onClick={multi ? () => setOpen((o) => ({ ...o, [g.key]: !o[g.key] })) : undefined} style={{ cursor: multi ? "pointer" : "default" }}>
                                  <td style={{ ...TD, textAlign: "left" }}>
                                    {multi && <span style={{ marginRight: 6, color: "var(--text-dim)", fontSize: 11 }}>{isOpen ? "▼" : "▶"}</span>}
                                    {describe(g)}
                                  </td>
                                  <td style={NOWRAP_TD}>{g.items.length}点</td>
                                  <td style={NOWRAP_TD}>{g.oddsMin ? (g.oddsMin === g.oddsMax || !multi ? `${g.oddsMin.toFixed(1)}倍` : `${g.oddsMin.toFixed(1)}〜${g.oddsMax.toFixed(1)}倍`) : "—"}</td>
                                  <td style={{ ...NOWRAP_TD, fontWeight: 700, color: evColor(g.evAvg, rec.calibrated) }} title={multi ? "金額加重の平均" : undefined}>{g.evAvg.toFixed(2)}</td>
                                  <td style={{ ...NOWRAP_TD, fontWeight: 700, color: "#fff" }}>¥{g.total.toLocaleString()}</td>
                                </tr>
                                {multi && isOpen && (
                                  <tr>
                                    <td colSpan={5} style={{ padding: "4px 12px 10px 34px", background: "rgba(255,255,255,0.02)", borderTop: "1px dashed var(--border)" }}>
                                      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                                        <thead>
                                          <tr style={{ color: "var(--text-dim)" }}>
                                            <th style={{ ...TH, textAlign: "left", fontWeight: 500 }}>買い目</th>
                                            <th style={{ ...TH, fontWeight: 500 }}>的中確率</th>
                                            <th style={{ ...TH, fontWeight: 500 }}>オッズ</th>
                                            {showPredicted && <th style={{ ...TH, fontWeight: 500 }}>予測オッズ</th>}
                                            {showPredicted && <th style={{ ...TH, fontWeight: 500 }}>乖離</th>}
                                            <th style={{ ...TH, fontWeight: 500 }}>期待値</th>
                                            <th style={{ ...TH, fontWeight: 500 }}>金額</th>
                                          </tr>
                                        </thead>
                                        <tbody>
                                          {g.items.map((c, i) => (
                                            <tr key={`${c.pair_label}-${i}`}>
                                              <td style={{ ...TD, textAlign: "left", borderTop: "1px solid var(--border)" }}>
                                                {(c.pair ?? []).map((n) => markChip(n))}
                                                <span style={{ color: "#7dd3fc", fontWeight: 800, marginRight: 8 }}>{c.pair_label}</span>
                                                <span style={{ color: "var(--text-dim)" }}>{(c.horse_names ?? []).filter(Boolean).join(" - ")}</span>
                                              </td>
                                              <td style={{ ...TD, borderTop: "1px solid var(--border)" }}>{c.prob != null ? `${(c.prob * 100).toFixed(1)}%` : "—"}</td>
                                              <td style={{ ...TD, borderTop: "1px solid var(--border)" }}>{c.odds != null ? `${c.odds.toFixed(1)}倍` : "—"}</td>
                                              {showPredicted && <td style={{ ...TD, borderTop: "1px solid var(--border)", color: "var(--text-dim)" }}>{c.predicted_odds != null ? `${c.predicted_odds.toFixed(1)}倍` : "—"}</td>}
                                              {showPredicted && <td style={{ ...TD, borderTop: "1px solid var(--border)", color: c.odds_gap == null ? "var(--text-dim)" : c.odds_gap > 0 ? "#22c55e" : "#f87171" }}>{c.odds_gap != null ? `${c.odds_gap > 0 ? "+" : ""}${(c.odds_gap * 100).toFixed(0)}%` : "—"}</td>}
                                              <td style={{ ...TD, borderTop: "1px solid var(--border)", fontWeight: 700, color: evColor(c.ev, rec.calibrated) }} title={c.ev_raw != null ? `未較正の期待値 ${c.ev_raw.toFixed(2)}${c.ev_calibration_validated === false ? "（この券種の較正は目安）" : ""}` : undefined}>{c.ev != null ? c.ev.toFixed(2) : "—"}</td>
                                              <td style={{ ...TD, borderTop: "1px solid var(--border)", fontWeight: 700, color: "#fff" }}>¥{c.bet_amount.toLocaleString()}</td>
                                            </tr>
                                          ))}
                                        </tbody>
                                      </table>
                                    </td>
                                  </tr>
                                )}
                              </Fragment>
                            );
                          })}
                        </tbody>
                      ))}
                    </table>
                  </>
                );
              })()}
            </div>
          )}
        </>
      )}

      {rec?.probability_model && (
        <p style={{ fontSize: 11, color: "var(--text-dim)", marginTop: 12, marginBottom: 0 }} title="docs/html/modeling/chapters/exotic_bet_probabilities.html">
          確率モデル:{" "}
          {rec.probability_model.kind === "discounted_harville"
            ? `馬連・ワイド・3連複は 割引Harville（α2=${rec.probability_model.alpha2}, α3=${rec.probability_model.alpha3}）で、勝率・${
                rec.probability_model.fitted_to?.length ? "連対率・複勝率に整合させて" : "連対率・複勝率が無いため Harville のまま"
              }算出`
            : "pred_score の softmax + Harville（従来）"}
        </p>
      )}
      <p style={{ fontSize: 11, color: "var(--text-dim)", marginTop: 12, lineHeight: 1.7 }}>
        {rec?.calibrated
          ? "期待値（較正後）= 的中確率 × オッズを、過去の確定オッズ・確定払戻での実現回収率に合わせて補正した値（買い目の選定は補正前の期待値 1.05 以上）。補正前の値は各買い目にマウスを載せると表示。ワイド・馬連などは年による変動が大きく、較正は目安。"
          : "期待値 = 的中確率 × オッズ（1.0 超でプラス期待）。"}予測オッズ基準の期待値は「モデル対モデル」で市場に対する優位性ではなく、買う価値の判断は実オッズが出てから行う。推奨金額は 1/4 Kelly 基準の比率で、1R の購入合計が購入上限（既定 2,000 円）に収まるよう 100 円単位で配分。
        確率は AI モデルの出力です。期待値の較正は過去データ（確定オッズ・確定払戻）での目安で、的中や回収を保証するものではありません。
      </p>
    </div>
  );
}
