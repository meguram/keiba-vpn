"use client";

/* 推奨馬券の結果（仮想購入）。サーバーが結果の出たレースの買い目に result を付ける（src/api/v1/delegates.attach_bet_results）。 */

export type BetResult = { hit: boolean; payout: number; estimated?: boolean; actual_multiple?: number | null; passed?: boolean | null };
export type RaceBetResult = {
  status: "final" | "provisional"; official: boolean; estimated: boolean;
  finish: { pos: number; no: number }[]; invested: number; returned: number; roi: number | null; hits: number; n_bets: number;
  /** パターン別の推奨: 条件（確定オッズ ≥ 想定倍率）を満たしたパターンを stake_yen ずつ買った集計。all は全パターンを買った場合 */
  stake_yen?: number; condition?: string; condition_known?: boolean; n_patterns?: number; all?: { invested: number; returned: number; hits: number };
};

const yen = (n: number) => `¥${Math.round(n).toLocaleString("ja-JP")}`;
const POS_COLOR: Record<number, string> = { 1: "#fbbf24", 2: "#cbd5e1", 3: "#d97706" };

/** 結果の要約（着順・的中数・購入→払戻・回収率）を 1 行で。パターン別の推奨（stake_yen あり）は『条件を満たしたパターンを 100 円ずつ』の集計 */
export function BetResultBanner({ result }: { result: RaceBetResult }) {
  const win = (result.roi ?? 0) >= 1;
  const flat = result.stake_yen != null;
  const tone = result.hits === 0 ? { bg: "rgba(148,163,184,0.08)", bd: "var(--border)" } : win ? { bg: "rgba(34,197,94,0.10)", bd: "rgba(34,197,94,0.45)" } : { bg: "rgba(251,191,36,0.08)", bd: "rgba(251,191,36,0.4)" };
  const allRoi = result.all && result.all.invested > 0 ? result.all.returned / result.all.invested : null;
  return (
    <div role="status" style={{ marginBottom: 8, padding: "5px 10px", borderRadius: 8, background: tone.bg, border: `1px solid ${tone.bd}`, fontSize: 12.5 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
        <b style={{ color: "#f0f6fc" }}>結果</b>
        {result.status === "provisional" && <span style={{ fontSize: 11, color: "#fbbf24" }}>速報</span>}
        <span style={{ display: "inline-flex", gap: 4 }}>
          {result.finish.map((f) => (
            <span key={f.pos} style={{ padding: "0 7px", borderRadius: 999, border: `1px solid ${POS_COLOR[f.pos] ?? "var(--border)"}`, color: POS_COLOR[f.pos], fontSize: 12, fontWeight: 700 }}>{f.pos}着 {f.no}</span>
          ))}
        </span>
        <span style={{ color: "var(--text-dim)" }}>｜</span>
        {flat && <span>条件を満たした <b>{result.n_bets}</b>/{result.n_patterns}パターンを{result.stake_yen}円ずつ</span>}
        <span>的中 <b>{result.hits}</b>/{result.n_bets}点</span>
        <span style={{ color: "var(--text-dim)" }}>｜</span>
        <span style={{ fontVariantNumeric: "tabular-nums" }}>{yen(result.invested)} → <b>{yen(result.returned)}</b></span>
        <b style={{ fontSize: 15, color: result.roi == null ? "var(--text-dim)" : win ? "#4ade80" : "#f87171" }}>{result.roi == null ? "—" : `${Math.round(result.roi * 100)}%`}</b>
        {(!result.official || result.estimated) && <span style={{ fontSize: 11, color: "var(--text-dim)" }}>（払戻は概算を含む）</span>}
        <span style={{ marginLeft: "auto", fontSize: 11, color: "var(--text-dim)" }}>仮想購入（実際の購入ではありません）</span>
      </div>
      {flat && (
        <div style={{ marginTop: 3, fontSize: 11.5, color: "var(--text-dim)" }}>
          {result.condition_known
            ? "条件＝確定オッズが想定オッズ以上。条件を満たさなかったパターンは購入しなかったものとして集計しています。"
            : "確定オッズが取得できていないため、条件の判定ができません。全パターンを購入した集計を表示しています。"}
          {result.all && result.condition_known && result.all.invested > 0 && (
            <span> 参考：全{result.n_patterns}パターンを{result.stake_yen}円ずつ購入した場合 {yen(result.all.invested)} → {yen(result.all.returned)}（{allRoi == null ? "—" : `${Math.round(allRoi * 100)}%`}）</span>
          )}
        </div>
      )}
    </div>
  );
}

/** 表の『結果』セル。hit が null なら未精算（—）。data-sort は払戻額（外れは 0、未精算は空） */
export function ResultCell({ hit, payout, hits, n }: { hit: boolean | null; payout: number; hits?: number; n?: number }) {
  return (
    <td data-sort={hit == null ? "" : String(hit ? payout : 0)} style={{ padding: "5px 10px", textAlign: "right", whiteSpace: "nowrap", fontSize: 12.5, borderTop: "1px solid var(--border)" }}>
      {hit == null ? <span style={{ color: "var(--text-dim)" }}>—</span>
        : hit ? <span title="的中" style={{ fontWeight: 800, color: "#1a1204", background: "linear-gradient(135deg,#fde68a,#f59e0b)", borderRadius: 4, padding: "1px 7px" }}>的中 {yen(payout)}{n && n > 1 ? `（${hits}/${n}）` : ""}</span>
        : <span style={{ color: "var(--text-dim)" }}>×</span>}
    </td>
  );
}
