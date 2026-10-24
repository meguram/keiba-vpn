"use client";

import { BetRecommend, type MarkBySeq } from "@/components/BetRecommend";
import { MemberLockCard } from "@/components/upgrade/MemberLock";
import { MARK_META, MARK_ORDER } from "@/lib/marks";

export type BetTabPred = {
  horse_number?: number | null;
  horse_name?: string;
  mark_type?: string;
  win_prob?: number | null;
  top3_prob?: number | null;
  win_odds?: number | null;
  ev_win?: number | null;
  ev_win_cal?: number | null;
};

const TH: React.CSSProperties = { padding: "6px 8px", fontSize: 11, fontWeight: 500, color: "var(--text-dim)", textAlign: "right", whiteSpace: "nowrap", borderBottom: "1px solid var(--border)" };
const TD: React.CSSProperties = { padding: "6px 8px", fontSize: 13, textAlign: "right", whiteSpace: "nowrap", borderTop: "1px solid var(--border)" };

const pct = (v?: number | null) => (v == null ? "-" : `${(v * 100).toFixed(1)}%`);

/** 週間予測の『推奨馬券』タブ: 印の一覧（◎○▲★△消の順）と、印つきの買い目 */
export function WeeklyBetTab({ raceId, hasPred, preds, lock, authLoading }: { raceId: string; hasPred: boolean; preds: BetTabPred[]; lock: boolean; authLoading: boolean }) {
  if (authLoading) {
    return <div role="status" style={{ padding: "16px 0", color: "var(--text-dim)", fontSize: 13 }}>ログイン状態を確認中…</div>;
  }
  if (lock) return <MemberLockCard title="推奨馬券" />;

  const marked = preds
    .filter((p) => p.mark_type && MARK_ORDER[p.mark_type])
    .sort((a, b) => (MARK_ORDER[a.mark_type as string] - MARK_ORDER[b.mark_type as string]) || ((a.horse_number ?? 99) - (b.horse_number ?? 99)));
  const marks: MarkBySeq = {};
  for (const p of marked) {
    if (p.horse_number == null) continue;
    const m = MARK_META[p.mark_type as string];
    marks[p.horse_number] = { sym: m.sym, color: m.color, kesi: p.mark_type === "kesi" };
  }

  return (
    <div style={{ padding: "12px 0" }}>
      <h3 style={{ margin: "0 0 6px", fontSize: 14, color: "#fff" }}>めぐAI印</h3>
      {marked.length === 0 ? (
        <div style={{ color: "var(--text-dim)", fontSize: 13, padding: "6px 0 12px" }}>このレースは、印の付いた馬がまだありません（予測またはオッズが揃うと付きます）。</div>
      ) : (
        <div style={{ overflowX: "auto", marginBottom: 14 }}>
          <table style={{ width: "100%", borderCollapse: "collapse", minWidth: 420 }}>
            <thead>
              <tr>
                <th style={{ ...TH, textAlign: "center" }}>印</th>
                <th style={TH}>馬番</th>
                <th style={{ ...TH, textAlign: "left" }}>馬名</th>
                <th style={TH}>勝率</th>
                <th style={TH}>複勝率</th>
                <th style={TH}>想定単勝</th>
                <th style={TH}>勝期待値</th>
              </tr>
            </thead>
            <tbody>
              {marked.map((p) => {
                const m = MARK_META[p.mark_type as string];
                const ev = p.ev_win_cal ?? p.ev_win;
                return (
                  <tr key={`${p.horse_number}-${p.mark_type}`}>
                    <td style={{ ...TD, textAlign: "center", color: m.color, fontWeight: 800, fontSize: 16 }}>{m.sym}</td>
                    <td style={TD}>{p.horse_number ?? "-"}</td>
                    <td style={{ ...TD, textAlign: "left", color: "#fff" }}>{p.horse_name ?? "-"}</td>
                    <td style={TD}>{pct(p.win_prob)}</td>
                    <td style={TD}>{pct(p.top3_prob)}</td>
                    <td style={TD}>{p.win_odds != null ? `${Number(p.win_odds).toFixed(1)}倍` : "-"}</td>
                    <td style={{ ...TD, fontWeight: 700 }}>{ev != null ? Number(ev).toFixed(2) : "-"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
      <h3 style={{ margin: "0 0 6px", fontSize: 14, color: "#fff" }}>推奨馬券（印つき）</h3>
      <BetRecommend raceId={raceId} hasPrediction={hasPred} marks={marks} />
    </div>
  );
}
