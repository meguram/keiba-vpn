"use client";

import type { CSSProperties } from "react";

/** 会員向けの推奨馬券（有料ゾーン）。サーバーが『レース単位の推奨度』と『買い目の区分』だけを返す。オッズ・期待値・確率・金額の計算値は来ない。 */
export type PaidTier = { level: number; label: string; short?: string };
export type PaidLeg = { horse_number: number; horse_name: string; mark: string };
export type PaidBet = {
  bet_type: string; bet_label: string; combo: number[]; combo_label: string; legs: PaidLeg[];
  role: "main" | "cover"; role_label: string; units: number; yen: number;
};
export type PaidRec = {
  view: "paid"; race_id: string; race_tier: PaidTier; bets: PaidBet[]; total_yen: number; total_units: number; unit_yen: number;
  max_total_bet: number | null; legend: PaidTier[]; disclaimer: string; note?: string; basis?: "A" | "B" | "C";
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

const MARK_COLOR: Record<string, string> = { "◎": "#f87171", "○": "#60a5fa", "▲": "#fbbf24", "★": "#fb923c", "△": "#a78bfa", "消": "#94a3b8" };

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

export function PaidBets({
  rec, cap, setCap, onRerun, loading,
}: { rec: PaidRec; cap: string; setCap: (v: string) => void; onRerun: () => void; loading: boolean }) {
  const groups = ORDER.map((bt) => ({ bt, bets: rec.bets.filter((b) => b.bet_type === bt) })).filter((g) => g.bets.length > 0);
  const chip = (role: "main" | "cover", label: string): CSSProperties =>
    role === "main"
      ? { background: "#1e3a5f", color: "#93c5fd", border: "1px solid #3b82f6" }
      : { background: "transparent", color: "var(--text-dim)", border: "1px solid var(--border)" };
  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap", marginBottom: 12 }}>
        <span style={{ fontSize: 13, color: "var(--text-dim)" }}>このレースの推奨度</span>
        <TierBadge tier={rec.race_tier} size={15} />
        <span style={{ fontSize: 12, color: "var(--text-dim)" }}>{tierDesc(rec.race_tier.level, rec.basis)}</span>
      </div>

      <details style={{ marginBottom: 12 }}>
        <summary style={{ cursor: "pointer", fontSize: 12, color: "var(--text-dim)" }}>推奨度の見方</summary>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginTop: 8 }}>
          {rec.legend.map((t) => (
            <span key={t.level} style={{ display: "inline-flex", alignItems: "center", gap: 6, fontSize: 12, color: "var(--text-dim)" }}>
              <TierBadge tier={t} size={12} />{tierDesc(t.level, rec.basis)}
            </span>
          ))}
        </div>
        <div style={{ fontSize: 12, color: "var(--text-dim)", marginTop: 6 }}>
          {rec.note ?? "推奨度は、推奨する買い目全体（組み合わせ）の見通しを段階にした目安です。オッズや期待値の数値は表示しません。"}
        </div>
      </details>

      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12, flexWrap: "wrap" }}>
        <label style={{ fontSize: 12, color: "var(--text-dim)" }} htmlFor="paid-cap">購入上限（円）</label>
        <input
          id="paid-cap" value={cap} onChange={(e) => setCap(e.target.value)} inputMode="numeric" placeholder="空欄・0 で上限なし"
          style={{ width: 120, padding: "5px 8px", borderRadius: 6, border: "1px solid var(--border)", background: "transparent", color: "inherit", fontSize: 13 }}
        />
        <button onClick={onRerun} disabled={loading} style={{ padding: "5px 12px", borderRadius: 6, fontSize: 12, cursor: "pointer", border: "1px solid var(--border)", background: "transparent", color: "var(--text-dim)" }}>
          {loading ? "計算中…" : "再計算"}
        </button>
      </div>

      {rec.bets.length === 0 ? (
        <div style={{ padding: 24, textAlign: "center", color: "var(--text-dim)", border: "1px dashed var(--border)", borderRadius: 8 }}>
          このレースは見送りの水準です。買い目は出しません。
        </div>
      ) : (
        <div style={{ display: "grid", gap: 10 }}>
          {groups.map((g) => (
            <div key={g.bt} style={{ border: "1px solid var(--border)", borderRadius: 8, overflow: "hidden" }}>
              <div style={{ padding: "6px 12px", fontSize: 12, fontWeight: 700, background: "rgba(148,163,184,0.08)" }}>{g.bets[0].bet_label}</div>
              {g.bets.map((b) => (
                <div key={`${g.bt}-${b.combo.join("-")}`} style={{ display: "flex", alignItems: "center", gap: 10, padding: "8px 12px", borderTop: "1px solid var(--border)", flexWrap: "wrap" }}>
                  <span style={{ minWidth: 70, fontWeight: 700 }}>{b.combo_label}</span>
                  <span style={{ flex: 1, fontSize: 13, display: "flex", flexWrap: "wrap", gap: 10 }}>
                    {b.legs.map((l) => (
                      <span key={l.horse_number}>
                        <span style={{ color: MARK_COLOR[l.mark] ?? "inherit", fontWeight: 800, marginRight: 2 }}>{l.mark}</span>
                        {l.horse_number} {l.horse_name}
                      </span>
                    ))}
                  </span>
                  <span style={{ fontSize: 11, padding: "1px 8px", borderRadius: 4, ...chip(b.role, b.role_label) }}>{b.role_label}</span>
                  <span style={{ minWidth: 64, textAlign: "right", fontSize: 13 }}>{b.yen.toLocaleString()}円</span>
                </div>
              ))}
            </div>
          ))}
          <div style={{ textAlign: "right", fontSize: 13 }}>
            合計 <b>{rec.total_yen.toLocaleString()}円</b>
            <span style={{ color: "var(--text-dim)", marginLeft: 8, fontSize: 12 }}>（1口 {rec.unit_yen}円。本線は多め、押さえは少なめ）</span>
          </div>
        </div>
      )}

      <div style={{ marginTop: 14, fontSize: 11, lineHeight: 1.7, color: "var(--text-dim)" }}>{rec.disclaimer}</div>
    </div>
  );
}
