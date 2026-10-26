"use client";

import { useState, type CSSProperties } from "react";
import { BetResultBanner, ResultCell, type BetResult, type RaceBetResult } from "@/components/BetResult";
import { TIER_STYLE, type PaidLeg, type PaidTier } from "@/components/PaidBets";

/** 既定の推奨馬券（目標払戻つきの 3連複）。購入額は『どれか 1 つ当たれば払戻が予算の約 M 倍』になるように決める。
 *  倍率・払戻は想定オッズ（AI の予測）で、実オッズは使わない・出さない。 */
export type TargetBet = {
  bet_type: string; bet_label: string; combo: number[]; combo_label: string; legs: PaidLeg[];
  yen: number; est_multiple: number; est_payout_yen: number; hit_rate_pct: number; result?: BetResult;
};
export type TargetRec = {
  strategy: "target"; view: string; race_id: string; status: "bet" | "skip"; skip_reason: string | null;
  race_tier: PaidTier; legend: PaidTier[]; bets: TargetBet[]; total_yen: number; budget_yen: number; target_multiple: number;
  target_payout_yen?: number; any_hit_rate_pct?: number; max_total_bet: number | null; disclaimer: string; evidence?: string; note?: string;
  odds_basis?: "sec77" | "predicted" | "own"; result?: RaceBetResult | null;
  default_plan?: string; race_ev?: number | null; plans?: TargetPlan[];
  /** 見送り（推奨度 0）のレース。サーバが買い目・他プラン・金額を返さない */
  skipped?: boolean;
  /** 買い目全体を購入したときの期待払戻（円）と期待回収率（倍）。AI の確率×想定オッズの値（実オッズではない） */
  expected_payout_yen?: number; portfolio_ev?: number | null;
};
export type TargetPlan = {
  id: string; label: string; note: string; evidence: string; kind: string; status: "bet" | "skip"; skip_reason: string | null;
  bets: TargetBet[]; total_yen: number; any_hit_rate_pct?: number; race_ev?: number | null; target_payout_yen?: number; level: number;
  expected_payout_yen?: number; portfolio_ev?: number | null;
};

const TARGET_DESC: Record<number, string> = {
  0: "見送り（レースの期待値が水準未満、または条件を満たす買い目なし）", 1: "レースの期待値は水準を超えた程度", 2: "レースの期待値はやや高い", 3: "レースの期待値が高い", 4: "レースの期待値が特に高い（それでも当たらないレースが大半）",
};
const MARK_COLOR: Record<string, string> = { "◎": "#f87171", "○": "#60a5fa", "▲": "#fbbf24", "★": "#fb923c", "△": "#a78bfa", "消": "#94a3b8" };
const TH: CSSProperties = { padding: "5px 8px", textAlign: "right", fontSize: 12, color: "var(--text-dim)", fontWeight: 600, whiteSpace: "nowrap" };
const TD: CSSProperties = { padding: "5px 8px", textAlign: "right", fontSize: 12.5, borderTop: "1px solid var(--border)", verticalAlign: "middle" };

function Badge({ tier }: { tier: PaidTier }) {
  const s = TIER_STYLE[tier.level] ?? TIER_STYLE[0];
  return (
    <span title={`推奨度 ${tier.level}: ${tier.label}（${TARGET_DESC[tier.level] ?? ""}）`}
      style={{ display: "inline-block", padding: "2px 10px", borderRadius: 999, fontSize: 14, fontWeight: 800, color: s.color, background: s.bg, border: `1px solid ${s.border}` }}>
      {tier.label}
    </span>
  );
}

export function TargetBets({
  rec, cap, setCap, onRerun, loading,
}: { rec: TargetRec; cap: string; setCap: (v: string) => void; onRerun: () => void; loading: boolean }) {
  const [pid, setPid] = useState<string | null>(null);
  const plan = rec.plans?.find((p) => p.id === (pid ?? rec.default_plan)) ?? null;
  const view = plan ? { ...rec, bets: plan.bets, status: plan.status, skip_reason: plan.skip_reason, total_yen: plan.total_yen, any_hit_rate_pct: plan.any_hit_rate_pct,
    target_payout_yen: plan.target_payout_yen, evidence: plan.evidence, race_ev: plan.race_ev,
    expected_payout_yen: plan.expected_payout_yen, portfolio_ev: plan.portfolio_ev, result: plan.id === rec.default_plan ? rec.result : null } : rec;
  const hasResult = !!view.result;
  const yen = (n: number) => `${n.toLocaleString()}円`;
  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap", marginBottom: 8 }}>
        <span style={{ fontSize: 12.5, color: "var(--text-dim)" }}>推奨度</span>
        <Badge tier={rec.race_tier} />
        <span style={{ fontSize: 12, color: "var(--text-dim)" }}>{TARGET_DESC[rec.race_tier.level] ?? ""}</span>
        <details style={{ display: "inline-block" }}>
          <summary style={{ cursor: "pointer", fontSize: 12, color: "var(--accent)", listStyle: "none" }}>見方</summary>
          <div style={{ position: "absolute", zIndex: 5, marginTop: 4, padding: 10, maxWidth: 520, background: "var(--surface2)", border: "1px solid var(--border)", borderRadius: 8, boxShadow: "0 8px 24px rgba(0,0,0,.4)" }}>
            <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
              {rec.legend.map((t) => (
                <span key={t.level} style={{ display: "inline-flex", alignItems: "center", gap: 6, fontSize: 12, color: "var(--text-dim)" }}><Badge tier={t} />{TARGET_DESC[t.level] ?? ""}</span>
              ))}
            </div>
            <div style={{ fontSize: 11.5, color: "var(--text-dim)", marginTop: 6, lineHeight: 1.6 }}>
              推奨度は、想定オッズ（AI の予測）で計算した『このレースの買い方の期待回収率』の高さの目安です。実際のオッズとは異なり、回収率を保証するものではありません。{rec.note ? ` ${rec.note}` : ""}
            </div>
          </div>
        </details>
        <span style={{ marginLeft: "auto", display: "inline-flex", alignItems: "center", gap: 6 }}>
          <label style={{ fontSize: 12, color: "var(--text-dim)" }} htmlFor="target-cap">購入上限</label>
          <input id="target-cap" value={cap} onChange={(e) => setCap(e.target.value)} inputMode="numeric" placeholder="なし" style={{ width: 76, padding: "3px 7px", borderRadius: 6, border: "1px solid var(--border)", background: "transparent", color: "inherit", fontSize: 16 }} />
          <button onClick={onRerun} disabled={loading} style={{ padding: "3px 11px", borderRadius: 6, fontSize: 12, cursor: "pointer", border: "1px solid var(--border)", background: "transparent", color: "var(--text)" }}>{loading ? "計算中…" : "再計算"}</button>
        </span>
      </div>

      {!rec.skipped && <div style={{ fontSize: 12.5, color: "var(--text-dim)", marginBottom: 8, lineHeight: 1.6 }}>
        予算 <b style={{ color: "var(--text)" }}>{yen(rec.budget_yen)}</b> に対し、どれか1つ当たれば払戻が約 <b style={{ color: "var(--text)" }}>{rec.target_multiple}倍（{yen(rec.target_payout_yen ?? rec.budget_yen * rec.target_multiple)}）</b>になる購入額にしています（3連複・最大3点）。
      </div>}

      {rec.plans && rec.plans.length > 1 && (
        <div role="tablist" aria-label="プラン" style={{ display: "flex", gap: 6, overflowX: "auto", paddingBottom: 6, marginBottom: 6 }}>
          {rec.plans.map((p) => (
            <button key={p.id} role="tab" aria-selected={p.id === plan?.id} onClick={() => setPid(p.id)} title={p.note}
              style={{ flex: "0 0 auto", padding: "5px 12px", borderRadius: 999, fontSize: 12.5, cursor: "pointer", whiteSpace: "nowrap", minHeight: 32,
                border: `1px solid ${p.id === plan?.id ? "var(--accent)" : "var(--border)"}`, background: p.id === plan?.id ? "rgba(96,165,250,0.18)" : "transparent", color: "var(--text)" }}>
              {p.label}{p.status === "skip" ? "（見送り）" : ""}
            </button>
          ))}
        </div>
      )}
      {plan?.note && <div style={{ fontSize: 12, color: "var(--text-dim)", marginBottom: 6 }}>{plan.note}{view.race_ev != null && ` レース期待値 ${view.race_ev.toFixed(2)}（想定）。`}</div>}

      {view.result && <BetResultBanner result={view.result} />}

      {view.status === "skip" || view.bets.length === 0 ? (
        <div style={{ padding: 18, textAlign: "center", color: "var(--text-dim)", border: "1px dashed var(--border)", borderRadius: 8, fontSize: 13 }}>
          このレースは見送りです。{view.skip_reason ?? ""}
        </div>
      ) : (
        <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", border: "1px solid var(--border)", borderRadius: 8, overflow: "hidden", background: "var(--surface)" }}>
          <thead style={{ background: "var(--surface2)" }}>
            <tr>
              <th style={{ ...TH, textAlign: "left" }}>買い目</th>
              <th style={TH}>購入額</th>
              <th style={TH} title="想定オッズ（AI の予測）。実際のオッズではありません">想定倍率</th>
              <th style={TH} title="購入額×想定倍率。当たったときの目安">想定払戻</th>
              <th style={TH} title="この組み合わせが的中する確率（AI の予測）">的中率</th>
              {hasResult && <th style={TH}>結果</th>}
            </tr>
          </thead>
          <tbody>
            {view.bets.map((b) => (
              <tr key={b.bet_type + b.combo.join("-")} style={{ background: b.result?.hit ? "rgba(251,191,36,0.08)" : undefined }}>
                <td style={{ ...TD, textAlign: "left" }} data-sort={b.combo.join(" ")}>
                  <span style={{ fontSize: 11.5, color: "var(--text-dim)", marginRight: 6 }}>{b.bet_label}</span><span style={{ fontWeight: 800, marginRight: 8 }}>{b.combo_label}</span>
                  <span style={{ display: "flex", flexWrap: "wrap", gap: "0 10px", fontSize: 12, color: "var(--text-dim)", marginTop: 2 }}>
                    {b.legs.map((l) => (
                      <span key={l.horse_number} style={{ whiteSpace: "nowrap" }}><span style={{ color: MARK_COLOR[l.mark] ?? "inherit", fontWeight: 800, marginRight: 1 }}>{l.mark}</span>{l.horse_name}</span>
                    ))}
                  </span>
                </td>
                <td style={{ ...TD, whiteSpace: "nowrap" }} data-sort={b.yen}>{yen(b.yen)}</td>
                <td style={{ ...TD, whiteSpace: "nowrap" }} data-sort={b.est_multiple}>{b.est_multiple.toFixed(1)}倍</td>
                <td style={{ ...TD, whiteSpace: "nowrap" }} data-sort={b.est_payout_yen}>約{yen(b.est_payout_yen)}</td>
                <td style={{ ...TD, whiteSpace: "nowrap" }} data-sort={b.hit_rate_pct}>{b.hit_rate_pct.toFixed(1)}%</td>
                {hasResult && <ResultCell hit={b.result ? b.result.hit : null} payout={b.result?.payout ?? 0} />}
              </tr>
            ))}
          </tbody>
          <tfoot>
            <tr>
              <td colSpan={hasResult ? 6 : 5} style={{ padding: "6px 10px", textAlign: "right", fontSize: 13, borderTop: "1px solid var(--border)" }}>
                合計 <b>{yen(view.total_yen)}</b>
                {view.any_hit_rate_pct != null && <span style={{ color: "var(--text-dim)", marginLeft: 8, fontSize: 11.5 }}>（どれかが当たる確率 約{view.any_hit_rate_pct}%）</span>}
              </td>
            </tr>
            {view.portfolio_ev != null && view.expected_payout_yen != null && (
              <tr>
                <td colSpan={hasResult ? 6 : 5} style={{ padding: "2px 10px 8px", textAlign: "right", fontSize: 13 }}>
                  期待払戻 <b>約{yen(view.expected_payout_yen)}</b>
                  <span style={{ marginLeft: 8 }}>期待回収率 <b style={{ color: view.portfolio_ev >= 1 ? "var(--green, #4ade80)" : "var(--red, #f87171)" }}>{Math.round(view.portfolio_ev * 100)}%</b></span>
                  <span style={{ color: "var(--text-dim)", marginLeft: 8, fontSize: 11.5 }}>（AI の確率×想定オッズによる値。実際の回収を保証しません）</span>
                </td>
              </tr>
            )}
          </tfoot>
        </table>
        </div>
      )}

      <div style={{ marginTop: 8, fontSize: 11, lineHeight: 1.6, color: "var(--text-dim)" }}>
        {rec.disclaimer}{view.evidence ? ` ${view.evidence}` : ""}
      </div>
    </div>
  );
}
