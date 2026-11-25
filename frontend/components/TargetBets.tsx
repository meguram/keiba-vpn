"use client";

import type { CSSProperties } from "react";
import { BetResultBanner, ResultCell, type BetResult, type RaceBetResult } from "@/components/BetResult";
import { TIER_STYLE, type PaidLeg, type PaidTier } from "@/components/PaidBets";

/** 既定のサンプル買い目。購入額は『どれか 1 つ当たれば払戻が予算の約 M 倍』になるように決める。
 *  買い目は優先順（priority。1 が最優先）。想定オッズ（AI の予測）・期待値・期待回収率は画面に出さない（2026-11-25 のユーザーの判断）。
 *  実オッズは使わない・出さない。 */
export type TargetBet = {
  bet_type: string; bet_label: string; combo: number[]; combo_label: string; legs: PaidLeg[];
  yen: number; est_multiple: number; est_payout_yen: number; hit_rate_pct: number; ev_line_multiple?: number | null; priority?: number; result?: BetResult;
};
export type TargetRec = {
  strategy: "target"; view: string; race_id: string; status: "bet" | "skip"; skip_reason: string | null;
  race_tier: PaidTier; legend: PaidTier[]; bets: TargetBet[]; total_yen: number; budget_yen: number; target_multiple: number;
  target_payout_yen?: number; any_hit_rate_pct?: number; max_total_bet: number | null; disclaimer: string; evidence?: string; note?: string;
  odds_basis?: "sec77" | "predicted" | "own"; result?: RaceBetResult | null;
  default_plan?: string; race_ev?: number | null; plans?: TargetPlan[];
  /** 見送り（参考度 0）のレース。サーバが買い目・他プラン・金額を返さない */
  skipped?: boolean;
  /** 買い目全体を購入したときの期待払戻（円）と期待回収率（倍）。AI の確率×想定オッズの値（実オッズではない） */
  expected_payout_yen?: number; portfolio_ev?: number | null;
};
export type TargetPlan = {
  id: string; label: string; note: string; evidence?: string; kind: string; status: "bet" | "skip"; skip_reason: string | null;
  bets: TargetBet[]; total_yen: number; any_hit_rate_pct?: number; race_ev?: number | null; target_payout_yen?: number; level: number;
  expected_payout_yen?: number; portfolio_ev?: number | null;
};

const TARGET_DESC: Record<number, string> = {
  0: "見送り（このレースはサンプル買い目なし）", 1: "参考度 1", 2: "参考度 2", 3: "参考度 3", 4: "参考度 4（それでも当たらないレースが大半）",
};
const TH: CSSProperties = { padding: "5px 8px", textAlign: "right", fontSize: 12, color: "var(--text-dim)", fontWeight: 600, whiteSpace: "nowrap" };
const TD: CSSProperties = { padding: "5px 8px", textAlign: "right", fontSize: 12.5, borderTop: "1px solid var(--border)", verticalAlign: "middle" };

function Badge({ tier }: { tier: PaidTier }) {
  const s = TIER_STYLE[tier.level] ?? TIER_STYLE[0];
  return (
    <span title={`参考度 ${tier.level}: ${tier.label}（${TARGET_DESC[tier.level] ?? ""}）`}
      style={{ display: "inline-block", padding: "2px 10px", borderRadius: 999, fontSize: 14, fontWeight: 800, color: s.color, background: s.bg, border: `1px solid ${s.border}` }}>
      {tier.label}
    </span>
  );
}

export function TargetBets({ rec }: { rec: TargetRec }) {
  const plan = rec.plans?.find((p) => p.id === rec.default_plan) ?? null;
  const view = plan ? { ...rec, bets: plan.bets, status: plan.status, skip_reason: plan.skip_reason, total_yen: plan.total_yen, any_hit_rate_pct: plan.any_hit_rate_pct,
    target_payout_yen: plan.target_payout_yen, race_ev: plan.race_ev,
    expected_payout_yen: plan.expected_payout_yen, portfolio_ev: plan.portfolio_ev, result: plan.id === rec.default_plan ? rec.result : null } : rec;
  const hasResult = !!view.result;
  const yen = (n: number) => `${n.toLocaleString()}円`;
  const n = view.bets.length;
  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap", marginBottom: 8 }}>
        <span style={{ fontSize: 12.5, color: "var(--text-dim)" }}>参考度</span>
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
              参考度は、AI の確率と想定オッズから見た、このレースの買い方の参考の段階です（高いほど参考度が高い）。回収を約束しません。{rec.note ? ` ${rec.note}` : ""}
            </div>
          </div>
        </details>
      </div>

      {!rec.skipped && <div style={{ fontSize: 12.5, color: "var(--text-dim)", marginBottom: 8, lineHeight: 1.6 }}>
        サンプル買い目は最大5パターンです（1パターン＝1つの馬券）。<b style={{ color: "var(--text)" }}>条件を満たしたパターンをすべて100円ずつ買った場合</b>で集計します。
        パターンは優先順（1 が最優先）です。
        結果が出たあとは、確定オッズが想定オッズ以上だったパターンだけを購入したものとして集計します。
      </div>}

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
              <th style={TH} title="優先順（1 が最優先）">パターン</th>
              <th style={{ ...TH, textAlign: "left" }}>買い目</th>
              <th style={TH} title="この組み合わせが的中する確率（AI の予測）">的中率</th>
              {hasResult && <th style={TH} title="確定オッズが想定オッズ以上なら条件を満たし、集計の対象になります">条件</th>}
              {hasResult && <th style={TH} title="100円あたりの払戻">結果（100円）</th>}
            </tr>
          </thead>
          <tbody>
            {view.bets.map((b, i) => (
              <tr key={b.bet_type + b.combo.join("-")} style={{ background: b.result?.hit && b.result.passed !== false ? "rgba(251,191,36,0.08)" : undefined, opacity: b.result?.passed === false ? 0.55 : undefined }}>
                <td style={{ ...TD, fontWeight: 700 }} data-sort={b.priority ?? i + 1}>{b.priority ?? i + 1}</td>
                <td style={{ ...TD, textAlign: "left" }} data-sort={b.combo.join(" ")}>
                  <span style={{ fontSize: 11.5, color: "var(--text-dim)", marginRight: 6 }}>{b.bet_label}</span><span style={{ fontWeight: 800, marginRight: 8 }}>{b.combo_label}</span>
                  <span style={{ display: "flex", flexWrap: "wrap", gap: "0 10px", fontSize: 12, color: "var(--text-dim)", marginTop: 2 }}>
                    {b.legs.map((l) => (
                      <span key={l.horse_number} style={{ whiteSpace: "nowrap" }}>{l.horse_name}</span>
                    ))}
                  </span>
                </td>
                <td style={{ ...TD, whiteSpace: "nowrap" }} data-sort={b.hit_rate_pct}>{b.hit_rate_pct.toFixed(1)}%</td>
                {hasResult && (
                  <td style={{ ...TD, whiteSpace: "nowrap" }} data-sort={b.result?.passed == null ? "" : b.result.passed ? "1" : "0"}
                    title={b.result?.actual_multiple != null ? `確定オッズ ${b.result.actual_multiple.toFixed(1)}倍` : undefined}>
                    {b.result?.passed == null ? <span style={{ color: "var(--text-dim)" }}>—</span>
                      : b.result.passed ? <span style={{ color: "var(--green, #4ade80)", fontWeight: 700 }}>○ {b.result.actual_multiple != null ? `${b.result.actual_multiple.toFixed(1)}倍` : ""}</span>
                      : <span style={{ color: "var(--text-dim)" }}>× {b.result.actual_multiple != null ? `${b.result.actual_multiple.toFixed(1)}倍` : "未取得"}</span>}
                  </td>
                )}
                {hasResult && <ResultCell hit={b.result ? b.result.hit : null} payout={b.result?.payout ?? 0} />}
              </tr>
            ))}
          </tbody>
          <tfoot>
            <tr>
              <td colSpan={hasResult ? 5 : 3} style={{ padding: "6px 10px", textAlign: "right", fontSize: 13, borderTop: "1px solid var(--border)" }}>
                全{n}パターンを100円ずつ購入した場合（{yen(n * 100)}）
                {view.any_hit_rate_pct != null && <span style={{ color: "var(--text-dim)", marginLeft: 8, fontSize: 11.5 }}>（どれかが当たる確率 約{view.any_hit_rate_pct}%）</span>}
              </td>
            </tr>
          </tfoot>
        </table>
        </div>
      )}

      <div style={{ marginTop: 8, fontSize: 11, lineHeight: 1.6, color: "var(--text-dim)" }}>
        {rec.disclaimer}
      </div>
    </div>
  );
}
