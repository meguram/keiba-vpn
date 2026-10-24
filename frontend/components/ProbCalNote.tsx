"use client";

import { PROB_CAL_EXPLAIN, PROB_CAL_LABEL, calState, modelRows, oddsPhaseLabel, type ProbCalibration, type ProbModelFields } from "@/lib/probCalibration";

const pct = (v?: number | null) => (v == null ? "—" : `${(v * 100).toFixed(1)}%`);

/** 確率が市場補正後かどうかの表示。補正済み: ラベル + ⓘ の説明 + 元のモデルの確率（押すと見える）。補正なし: 小さな注記。機能が無効（prob_calibration なし）: 何も出さない。 */
export default function ProbCalNote({ pc, preds }: { pc?: ProbCalibration | null; preds?: ProbModelFields[] }) {
  const st = calState(pc);
  if (st === "off") return null;
  if (st === "skipped") {
    return (
      <p data-testid="prob-cal-skipped" style={{ fontSize: 11, color: "var(--text-dim)", margin: "6px 0" }} title={PROB_CAL_EXPLAIN}>
        確率: AI予測（市場補正なし）{pc?.reason ? ` — ${pc.reason}` : ""}
      </p>
    );
  }
  const rows = modelRows(preds);
  const phase = oddsPhaseLabel(pc?.odds_phase);
  return (
    <div data-testid="prob-cal-applied" style={{ fontSize: 11, color: "var(--text-dim)", margin: "6px 0" }}>
      <span style={{ color: "#93c5fd", fontWeight: 600 }}>確率: {PROB_CAL_LABEL}</span>
      <span role="img" aria-label="説明" title={PROB_CAL_EXPLAIN} style={{ marginLeft: 4, cursor: "help" }}>ⓘ</span>
      {phase && <span style={{ marginLeft: 6 }}>（{phase}のオッズで補正）</span>}
      {rows.length > 0 && (
        <details style={{ marginTop: 4 }}>
          <summary style={{ cursor: "pointer" }}>モデルの確率を見る</summary>
          <p style={{ margin: "6px 0", lineHeight: 1.6 }}>{PROB_CAL_EXPLAIN}</p>
          <table style={{ borderCollapse: "collapse", fontSize: 11, marginTop: 4 }}>
            <thead>
              <tr>
                <th style={{ textAlign: "left", padding: "2px 8px" }}>馬番・馬名</th>
                <th style={{ padding: "2px 8px" }}>勝率 モデル→補正後</th>
                <th style={{ padding: "2px 8px" }}>複勝率 モデル→補正後</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((p) => (
                <tr key={p.horse_number}>
                  <td style={{ padding: "2px 8px" }}>{p.horse_number} {p.horse_name ?? ""}</td>
                  <td style={{ padding: "2px 8px", textAlign: "center" }}>{pct(p.win_prob_model)} → <b>{pct(p.win_prob)}</b></td>
                  <td style={{ padding: "2px 8px", textAlign: "center" }}>{pct(p.top3_prob_model)} → <b>{pct(p.top3_prob)}</b></td>
                </tr>
              ))}
            </tbody>
          </table>
        </details>
      )}
    </div>
  );
}
