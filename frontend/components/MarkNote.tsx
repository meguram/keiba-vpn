/** 印のそばに置く固定の文（rules-council R0-10 の R3・統括役の承認 2026-11-25）。短く、ギャンブルを助長しない書き方 */
export function MarkNote() {
  return (
    <p style={{ margin: "6px 2px", fontSize: 11, lineHeight: 1.6, color: "var(--text-dim)" }}>
      ◎ は AI の評価 1 位。勝つ馬の予言ではありません。印どおりに全部買うと、長い目では控除率（約 2 割）の分だけ損をするのがふつうです。
    </p>
  );
}
