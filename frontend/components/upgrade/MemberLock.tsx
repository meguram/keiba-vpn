"use client";

import Link from "next/link";

/* 会員限定の項目を案内する表示。サーバは匿名の応答から該当項目を落としている（表示を隠しているだけではない）。 */
export const MEMBER_ONLY_ITEMS = "連対率・想定オッズ・期待値・買い判断・展開予想・推奨馬券";

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
        勝率・複勝率は誰でも見られます。想定オッズ・期待値・推奨馬券は、ログインした会員の方にお見せしています。
      </p>
      <Link href="/login" className="btn">ログイン</Link>
    </div>
  );
}

/** 公開準備中の案内。サーバが公開モードで B（事実）・O（市場オッズ由来）を落としたときだけ出す（access.public_mode）。 */
export type PublicAccess = { public_mode?: boolean; withheld?: string[] } | null | undefined;

export function PublicModeNote({ access }: { access: PublicAccess }) {
  if (!access?.public_mode) return null;
  const w = access.withheld ?? [];
  const parts = [w.includes("B") ? "出馬表・結果の詳細" : null, w.includes("O") ? "オッズ・期待値・買い目" : null, "指数・パドック・調教・過去走"].filter(Boolean);
  return (
    <p data-testid="public-mode-note" style={{ margin: "8px 0", padding: "8px 12px", fontSize: 12, borderRadius: 8, background: "rgba(234,179,8,0.08)", border: "1px solid rgba(234,179,8,0.3)", color: "var(--text-dim)", lineHeight: 1.6 }}>
      ℹ️ 公開準備中のため、{parts.join("・")}は表示していません。勝率・複勝率などの自前の予測は表示します。
    </p>
  );
}
