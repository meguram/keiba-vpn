"use client";

import type { CSSProperties } from "react";
import { BetResultBanner, ResultCell, type BetResult, type RaceBetResult } from "@/components/BetResult";

/** 会員向けの推奨馬券（有料ゾーン）。サーバーが『レース単位の推奨度』と『買い目の区分』だけを返す。オッズ・期待値・確率・金額の計算値は来ない。 */
export type PaidTier = { level: number; label: string; short?: string };
export type PaidLeg = { horse_number: number; horse_name: string; mark: string };
export type PaidBet = {
  bet_type: string; bet_label: string; combo: number[]; combo_label: string; legs: PaidLeg[];
  role: "main" | "cover"; role_label: string; units: number; yen: number;
  result?: BetResult;
};
export type PaidRec = {
  view: "paid"; race_id: string; race_tier: PaidTier; bets: PaidBet[]; total_yen: number; total_units: number; unit_yen: number;
  max_total_bet: number | null; legend: PaidTier[]; disclaimer: string; note?: string; basis?: "A" | "B" | "C";
  result?: RaceBetResult | null;
};

export const TIER_STYLE: Record<number, { color: string; bg: string; border: string; desc: string }> = {
  0: { color: "#8b9bb4", bg: "transparent", border: "#35445f", desc: "買わないほうがよいレース" },
  1: { color: "#bfdbfe", bg: "rgba(96,165,250,0.14)", border: "#60a5fa", desc: "期待回収率が約100%（損益分岐のあたり）" },
  2: { color: "#bbf7d0", bg: "rgba(34,197,94,0.18)", border: "#22c55e", desc: "プラスが見込める" },
  3: { color: "#fde68a", bg: "rgba(245,158,11,0.20)", border: "#f59e0b", desc: "プラスの見込みが高め" },
  4: { color: "#fff", bg: "linear-gradient(135deg,#f97316,#ef4444)", border: "#fb923c", desc: "その中でも特に高い（目安）" },
};

/** 方式 A（確率のみ）の推奨度は『確信度』の段階。値段（期待回収率）の説明を出さない */
const CONFIDENCE_DESC: Record<number, string> = {
  0: "本命が抜けておらず、確信度が低い", 1: "確信度がやや低い", 2: "確信度は標準", 3: "確信度がやや高い", 4: "本命が抜けており、確信度が高い",
};
export function tierDesc(level: number, basis?: string): string {
  if (basis === "A") return CONFIDENCE_DESC[level] ?? "";
  return (TIER_STYLE[level] ?? TIER_STYLE[0]).desc;
}


export function TierBadge({ tier, size = 14 }: { tier: PaidTier; size?: number }) {
  const s = TIER_STYLE[tier.level] ?? TIER_STYLE[0];
  return (
    <span
      title={`推奨度 ${tier.level}: ${tier.label}（${s.desc}）`}
      style={{ display: "inline-block", padding: "2px 10px", borderRadius: 999, fontSize: size, fontWeight: 800, color: s.color, background: s.bg, border: `1px solid ${s.border}` }}
    >
      {tier.label}
    </span>
  );
}

const ORDER = ["tansho", "fukusho", "umaren", "wide", "umatan", "sanrenpuku"];

const BET_COLOR: Record<string, string> = { tansho: "#f59e0b", fukusho: "#22c55e", umaren: "#60a5fa", wide: "#a78bfa", umatan: "#f472b6", sanrenpuku: "#fb7185" };
const PTH: CSSProperties = { padding: "5px 10px", textAlign: "right", fontSize: 12, color: "var(--text-dim)", fontWeight: 600, whiteSpace: "nowrap" };
const PTD: CSSProperties = { padding: "5px 10px", textAlign: "right", fontSize: 12.5, borderTop: "1px solid var(--border)", verticalAlign: "middle" };

export function PaidBets({
  rec, cap, setCap, onRerun, loading,
}: { rec: PaidRec; cap: string; setCap: (v: string) => void; onRerun: () => void; loading: boolean }) {
  const rows = ORDER.flatMap((bt) => rec.bets.filter((b) => b.bet_type === bt));
  const hasResult = !!rec.result;
  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap", marginBottom: 8 }}>
        <span style={{ fontSize: 12.5, color: "var(--text-dim)" }}>推奨度</span>
        <TierBadge tier={rec.race_tier} size={14} />
        <span style={{ fontSize: 12, color: "var(--text-dim)" }}>{tierDesc(rec.race_tier.level, rec.basis)}</span>
        <details style={{ display: "inline-block" }}>
          <summary style={{ cursor: "pointer", fontSize: 12, color: "var(--accent)", listStyle: "none" }}>見方</summary>
          <div style={{ position: "absolute", zIndex: 5, marginTop: 4, padding: 10, maxWidth: 520, background: "var(--surface2)", border: "1px solid var(--border)", borderRadius: 8, boxShadow: "0 8px 24px rgba(0,0,0,.4)" }}>
            <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
              {rec.legend.map((t) => (
                <span key={t.level} style={{ display: "inline-flex", alignItems: "center", gap: 6, fontSize: 12, color: "var(--text-dim)" }}><TierBadge tier={t} size={12} />{tierDesc(t.level, rec.basis)}</span>
              ))}
            </div>
            <div style={{ fontSize: 11.5, color: "var(--text-dim)", marginTop: 6, lineHeight: 1.6 }}>{rec.note ?? "推奨度は、推奨する買い目全体（組み合わせ）の見通しを段階にした目安です。オッズや期待値の数値は表示しません。"}</div>
          </div>
        </details>
        <span style={{ marginLeft: "auto", display: "inline-flex", alignItems: "center", gap: 6 }}>
          <label style={{ fontSize: 12, color: "var(--text-dim)" }} htmlFor="paid-cap">購入上限</label>
          <input id="paid-cap" value={cap} onChange={(e) => setCap(e.target.value)} inputMode="numeric" placeholder="なし" style={{ width: 76, padding: "3px 7px", borderRadius: 6, border: "1px solid var(--border)", background: "transparent", color: "inherit", fontSize: 12.5 }} />
          <button onClick={onRerun} disabled={loading} style={{ padding: "3px 11px", borderRadius: 6, fontSize: 12, cursor: "pointer", border: "1px solid var(--border)", background: "transparent", color: "var(--text)" }}>{loading ? "計算中…" : "再計算"}</button>
        </span>
      </div>

      {rec.result && <BetResultBanner result={rec.result} />}

      {rec.bets.length === 0 ? (
        <div style={{ padding: 18, textAlign: "center", color: "var(--text-dim)", border: "1px dashed var(--border)", borderRadius: 8, fontSize: 13 }}>
          このレースは見送りの水準です。買い目は出しません。
        </div>
      ) : (
        <table style={{ width: "100%", borderCollapse: "collapse", border: "1px solid var(--border)", borderRadius: 8, overflow: "hidden", background: "var(--surface)" }}>
          <thead style={{ background: "var(--surface2)" }}>
            <tr>
              <th style={{ ...PTH, textAlign: "left" }}>券種</th>
              <th style={{ ...PTH, textAlign: "left" }}>買い目</th>
              <th style={{ ...PTH, textAlign: "center" }}>区分</th>
              <th style={PTH}>金額</th>
              {hasResult && <th style={PTH}>結果</th>}
            </tr>
          </thead>
          <tbody>
            {rows.map((b, i) => (
              <tr key={`${b.bet_type}-${b.combo.join("-")}`} style={{ background: b.result?.hit ? "rgba(251,191,36,0.08)" : undefined }}>
                <td data-sort={b.bet_label} style={{ ...PTD, textAlign: "left", whiteSpace: "nowrap", fontWeight: 700, color: BET_COLOR[b.bet_type], borderLeft: `3px solid ${BET_COLOR[b.bet_type] ?? "var(--border)"}` }}>
                  {i === 0 || rows[i - 1].bet_type !== b.bet_type ? b.bet_label : <span style={{ opacity: 0.35 }}>〃</span>}
                </td>
                <td style={{ ...PTD, textAlign: "left" }} data-sort={b.combo.join(" ")}>
                  <span style={{ fontWeight: 800, marginRight: 8 }}>{b.combo_label}</span>
                  <span style={{ display: "inline-flex", flexWrap: "wrap", gap: "0 10px", fontSize: 12, color: "var(--text-dim)" }}>
                    {b.legs.map((l) => (
                      <span key={l.horse_number}>{l.horse_name}</span>
                    ))}
                  </span>
                </td>
                <td style={{ ...PTD, textAlign: "center" }}><span style={{ fontSize: 11, padding: "0 7px", borderRadius: 4, ...chipStyle(b.role) }}>{b.role_label}</span></td>
                <td style={{ ...PTD, whiteSpace: "nowrap" }}>{b.yen.toLocaleString()}円</td>
                {hasResult && <ResultCell hit={b.result ? b.result.hit : null} payout={b.result?.payout ?? 0} />}
              </tr>
            ))}
          </tbody>
          <tfoot>
            <tr>
              <td colSpan={hasResult ? 5 : 4} style={{ padding: "6px 10px", textAlign: "right", fontSize: 13, borderTop: "1px solid var(--border)" }}>
                合計 <b>{rec.total_yen.toLocaleString()}円</b>
                <span style={{ color: "var(--text-dim)", marginLeft: 8, fontSize: 11.5 }}>（1口 {rec.unit_yen}円・本線は多め／押さえは少なめ）</span>
              </td>
            </tr>
          </tfoot>
        </table>
      )}

      <div style={{ marginTop: 8, fontSize: 11, lineHeight: 1.6, color: "var(--text-dim)" }}>{rec.disclaimer}</div>
    </div>
  );
}

function chipStyle(role: "main" | "cover"): CSSProperties {
  return role === "main" ? { background: "#1e3a5f", color: "#93c5fd", border: "1px solid #3b82f6" } : { background: "transparent", color: "var(--text-dim)", border: "1px solid var(--border)" };
}
