"use client";

import type { CSSProperties } from "react";
import { BetResultBanner, ResultCell } from "@/components/BetResult";
import type { TargetRec } from "@/components/TargetBets";

/** 第 2 段のモデル（市場とのずれ）と凍結した買い方（第 2 回）の推奨（試行中）。
 *  サーバは既定方策 target と同じ形（strategy="target"・sub_strategy="stage2_r2"）で返し、判定の中身を stage2 に入れる。
 *  見送りのレースは理由を出す。回収率は確かめ中であることを必ず表示し、検証の回収率の数字は出さない（試行中のあいだ。
 *  レースの結果が出た後の BetResultBanner は、そのレース 1 つの実績で、検証の値ではないので出してよい）。
 *  文言の取り決めは tests/frontend/test_stage2_bets_text.py と tests/api/test_stage2_live.py の両方で縛る。 */
export type Stage2Horse = { horse_number: number; p_win: number | null; q: number | null; r: number | null; pop: number | null; odds: number | null };
export type Stage2Info = {
  frozen_hash?: string; status?: "buy" | "skip"; skip_reason?: string | null; skip_label?: string | null;
  r_max?: number | null; q_r_max?: number; c_class?: string; field?: number;
  honmei?: Stage2Horse | null; as_of?: string; odds_t60_at?: string | null; model?: string; horses?: Stage2Horse[];
};
export type Stage2Rec = TargetRec & { sub_strategy: "stage2_r2"; trial?: boolean; stage2: Stage2Info };

export function isStage2(rec: unknown): rec is Stage2Rec {
  return !!rec && typeof rec === "object" && (rec as { sub_strategy?: string }).sub_strategy === "stage2_r2";
}

const TH: CSSProperties = { padding: "5px 8px", textAlign: "right", fontSize: 12, color: "var(--text-dim)", fontWeight: 600, whiteSpace: "nowrap" };
const TD: CSSProperties = { padding: "5px 8px", textAlign: "right", fontSize: 12.5, borderTop: "1px solid var(--border)", verticalAlign: "middle" };

function hhmm(iso?: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  return d.toLocaleTimeString("ja-JP", { hour: "2-digit", minute: "2-digit", timeZone: "Asia/Tokyo" });
}

export function Stage2Bets({ rec }: { rec: Stage2Rec }) {
  const s = rec.stage2 ?? {};
  const buy = rec.status === "bet" && rec.bets.length > 0;
  const h = s.honmei ?? null;
  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap", marginBottom: 8 }}>
        <span style={{ display: "inline-block", padding: "2px 10px", borderRadius: 999, fontSize: 12.5, fontWeight: 800, color: "#f59e0b", border: "1px solid #f59e0b" }}>
          試行中
        </span>
        <span style={{ fontSize: 12, color: "var(--text-dim)" }}>回収率は確かめ中です</span>
      </div>
      <div style={{ fontSize: 12.5, color: "var(--text-dim)", marginBottom: 8, lineHeight: 1.6 }}>
        {rec.evidence}
        <span style={{ marginLeft: 6 }}>判定は発走 60 分前のオッズ{s.odds_t60_at ? `（${hhmm(s.odds_t60_at)} 時点）` : ""}によるもので、直前のオッズとは異なります。</span>
      </div>

      {rec.result && <BetResultBanner result={rec.result} />}

      {!buy ? (
        <div style={{ padding: 18, textAlign: "center", color: "var(--text-dim)", border: "1px dashed var(--border)", borderRadius: 8, fontSize: 13 }}>
          {s.skip_label ?? rec.skip_reason ?? "このレースは見送りです。"}
        </div>
      ) : (
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", border: "1px solid var(--border)", borderRadius: 8, overflow: "hidden", background: "var(--surface)" }}>
            <thead style={{ background: "var(--surface2)" }}>
              <tr>
                <th style={{ ...TH, textAlign: "left" }}>買い目</th>
                <th style={TH} title="第 2 段のモデルの勝率">勝率</th>
                <th style={TH} title="発走 60 分前の単勝オッズ">オッズ（T-60）</th>
                <th style={TH} title="発走 60 分前の人気">人気（T-60）</th>
                <th style={TH} title="r = log(モデルの勝率 / 市場の勝率)。大きいほど、市場よりモデルが高く見ている">市場とのずれ r</th>
                <th style={TH}>金額</th>
                {rec.result && <th style={TH} title="100円あたりの払戻">結果（100円）</th>}
              </tr>
            </thead>
            <tbody>
              {rec.bets.map((b) => (
                <tr key={b.bet_type + b.combo.join("-")}>
                  <td style={{ ...TD, textAlign: "left" }} data-sort={b.combo.join(" ")}>
                    <span style={{ fontSize: 11.5, color: "var(--text-dim)", marginRight: 6 }}>{b.bet_label}</span>
                    <span style={{ fontWeight: 800, marginRight: 8 }}>◎ {b.combo_label}</span>
                    <span style={{ fontSize: 12, color: "var(--text-dim)" }}>{b.legs.map((l) => l.horse_name).join("・")}</span>
                  </td>
                  <td style={TD} data-sort={h?.p_win ?? 0}>{h?.p_win != null ? `${(h.p_win * 100).toFixed(1)}%` : "—"}</td>
                  <td style={TD} data-sort={h?.odds ?? 0}>{h?.odds != null ? `${h.odds.toFixed(1)}倍` : "—"}</td>
                  <td style={TD} data-sort={h?.pop ?? 99}>{h?.pop != null ? `${h.pop}番人気` : "—"}</td>
                  <td style={{ ...TD, fontWeight: 700 }} data-sort={h?.r ?? 0}>{h?.r != null ? h.r.toFixed(2) : "—"}</td>
                  <td style={TD} data-sort={b.yen}>{b.yen.toLocaleString()}円</td>
                  {rec.result && <ResultCell hit={b.result ? b.result.hit : null} payout={b.result?.payout ?? 0} />}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <div style={{ fontSize: 11, color: "var(--text-dim)", marginTop: 6 }}>
        {rec.note} 判定 {hhmm(s.as_of)}・買い方 {s.frozen_hash ?? ""}{s.model ? `・モデル ${s.model}` : ""}
      </div>
    </div>
  );
}
