"use client";

import Link from "next/link";

/* 会員限定の項目を案内する表示。サーバは匿名の応答から該当項目を落としている（表示を隠しているだけではない）。 */
export const MEMBER_ONLY_ITEMS = "連対率・想定オッズ・期待値・めぐAI印・買い判断・展開予想・推奨馬券";

/** 予測表の上に出す 1 行の案内 */
export function MemberLockNote() {
  return (
    <p data-testid="member-lock-note" style={{ margin: "8px 0", padding: "8px 12px", fontSize: 12, borderRadius: 8, background: "rgba(59,130,246,0.08)", border: "1px solid rgba(59,130,246,0.25)", color: "var(--text-dim)", lineHeight: 1.6 }}>
      🔒 勝率・複勝率は誰でも見られます。{MEMBER_ONLY_ITEMS}は会員限定です。{" "}
      <Link href="/login" style={{ color: "var(--accent)" }}>ログイン</Link>
    </p>
  );
}

/** タブ全体が会員限定のとき（推奨馬券） */
export function MemberLockCard({ title }: { title: string }) {
  return (
    <div data-testid="member-lock-card" style={{ textAlign: "center", padding: "40px 16px", border: "1px solid var(--border)", borderRadius: 10, background: "var(--surface)" }}>
      <div style={{ fontSize: 40, marginBottom: 8 }}>🔒</div>
      <h2 style={{ fontSize: 16, fontWeight: 700, margin: "0 0 8px" }}>{title}は会員限定です</h2>
      <p style={{ fontSize: 13, color: "var(--text-dim)", margin: "0 0 16px", lineHeight: 1.6 }}>
        勝率・複勝率は誰でも見られます。想定オッズ・期待値・めぐAI印・推奨馬券は、ログインした会員の方にお見せしています。
      </p>
      <Link href="/login" className="btn">ログイン</Link>
    </div>
  );
}
