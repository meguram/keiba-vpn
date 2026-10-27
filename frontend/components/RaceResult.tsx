"use client";

import type { CSSProperties } from "react";
import type { RaceResultData } from "@/lib/hooks/useRaceResult";

const TH: CSSProperties = { padding: "5px 8px", textAlign: "right", fontSize: 12, color: "var(--text-dim)", fontWeight: 600, whiteSpace: "nowrap" };
const TD: CSSProperties = { padding: "5px 8px", textAlign: "right", fontSize: 12.5, borderTop: "1px solid var(--border)", verticalAlign: "middle", whiteSpace: "nowrap" };
const POS_COLOR: Record<number, string> = { 1: "#fbbf24", 2: "#cbd5e1", 3: "#d97706" };
const yen = (n: number) => `${n.toLocaleString("ja-JP")}円`;

/** 予測の順位（馬番 → 順位）。着順と並べて、当たり外れの目安に使う */
export type PredRankMap = Record<number, number>;

/** 週間予測の『レース結果』タブ。着順表（AI の予想順位つき）と払戻表。速報は払戻が揃うまで『速報』と出す。 */
export function RaceResult({ result, predRank }: { result: RaceResultData; predRank?: PredRankMap }) {
  const finish = result.finish ?? [];
  const payoff = result.payoff ?? [];
  const provisional = result.status === "provisional" || !result.official;
  const hasPredRank = !!predRank && Object.keys(predRank).length > 0;
  return (
    <div style={{ padding: "12px 0" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap", marginBottom: 8, fontSize: 12.5 }}>
        <b style={{ color: "var(--text)" }}>レース結果</b>
        <span style={{ padding: "1px 8px", borderRadius: 999, fontSize: 11, border: `1px solid ${provisional ? "#fbbf24" : "#34d399"}`, color: provisional ? "#fbbf24" : "#34d399" }}>
          {provisional ? "速報（払戻・確定は取得後に更新）" : "確定"}
        </span>
        {result.fetched_at && <span style={{ color: "var(--text-dim)", fontSize: 11.5 }}>取得 {result.fetched_at}</span>}
      </div>

      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", border: "1px solid var(--border)", background: "var(--surface)" }}>
          <thead style={{ background: "var(--surface2)" }}>
            <tr>
              <th style={TH}>着順</th>
              <th style={TH}>馬番</th>
              <th style={{ ...TH, textAlign: "left" }}>馬名</th>
              <th style={{ ...TH, textAlign: "left" }}>騎手</th>
              <th style={TH}>タイム</th>
              <th style={TH}>着差</th>
              <th style={TH}>上がり3F</th>
              <th style={TH}>人気</th>
              <th style={TH}>単勝</th>
              {hasPredRank && <th style={TH} title="AI の予測（勝率順）での順位">AI予想順位</th>}
            </tr>
          </thead>
          <tbody>
            {finish.map((f) => {
              const pr = f.horse_number != null ? predRank?.[f.horse_number] : undefined;
              return (
                <tr key={`${f.pos}-${f.horse_number}`} style={{ background: f.pos <= 3 ? "rgba(251,191,36,0.05)" : undefined }}>
                  <td style={{ ...TD, fontWeight: 800, color: POS_COLOR[f.pos] }} data-sort={f.pos}>{f.pos}</td>
                  <td style={TD} data-sort={f.horse_number ?? 0}>{f.horse_number ?? "-"}</td>
                  <td style={{ ...TD, textAlign: "left", fontWeight: 700 }}>{f.horse_name}</td>
                  <td style={{ ...TD, textAlign: "left" }}>{f.jockey_name}</td>
                  <td style={TD}>{f.finish_time || "—"}</td>
                  <td style={TD}>{f.margin || "—"}</td>
                  <td style={TD} data-sort={f.last_3f ?? 999}>{f.last_3f != null ? f.last_3f.toFixed(1) : "—"}</td>
                  <td style={TD} data-sort={f.popularity ?? 999}>{f.popularity ?? "—"}</td>
                  <td style={TD} data-sort={f.odds ?? 0}>{f.odds != null ? `${f.odds.toFixed(1)}倍` : "—"}</td>
                  {hasPredRank && <td style={TD} data-sort={pr ?? 999}>{pr ?? "—"}</td>}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <h4 style={{ margin: "16px 0 6px", fontSize: 13, color: "var(--text)" }}>払戻（100円あたり）</h4>
      {payoff.length === 0 ? (
        <div style={{ padding: 12, border: "1px dashed var(--border)", borderRadius: 8, color: "var(--text-dim)", fontSize: 12.5 }}>
          払戻はまだ取得できていません（確定の結果が公開されると表示します）。
        </div>
      ) : (
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", border: "1px solid var(--border)", background: "var(--surface)" }} data-no-sort>
            <thead style={{ background: "var(--surface2)" }}>
              <tr>
                <th style={{ ...TH, textAlign: "left" }}>券種</th>
                <th style={{ ...TH, textAlign: "left" }}>組合せ</th>
                <th style={TH}>払戻</th>
                <th style={TH}>人気</th>
              </tr>
            </thead>
            <tbody>
              {payoff.flatMap((g) =>
                g.rows.map((r, i) => (
                  <tr key={`${g.bet_label}-${r.numbers}-${i}`}>
                    <td style={{ ...TD, textAlign: "left", fontWeight: 700 }}>{i === 0 ? g.bet_label : ""}</td>
                    <td style={{ ...TD, textAlign: "left" }}>{r.numbers}</td>
                    <td style={{ ...TD, fontWeight: 700 }}>{yen(r.payout)}</td>
                    <td style={TD}>{r.popularity != null ? `${r.popularity}人気` : "—"}</td>
                  </tr>
                )),
              )}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
