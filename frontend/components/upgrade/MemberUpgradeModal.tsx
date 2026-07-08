"use client";

import Link from "next/link";

type Props = {
  onClose: () => void;
};

export function MemberUpgradeModal({ onClose }: Props) {
  return (
    <div
      onClick={onClose}
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(0,0,0,0.7)",
        zIndex: 200,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: "20px",
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          background: "var(--surface)",
          border: "1px solid var(--border)",
          borderRadius: 16,
          padding: "36px 32px",
          maxWidth: 480,
          width: "100%",
          position: "relative",
        }}
      >
        {/* Close button */}
        <button
          onClick={onClose}
          style={{
            position: "absolute",
            top: 14,
            right: 16,
            background: "none",
            border: "none",
            color: "var(--text-dim)",
            fontSize: 20,
            cursor: "pointer",
            lineHeight: 1,
          }}
          aria-label="閉じる"
        >
          ×
        </button>

        {/* Lock icon + heading */}
        <div style={{ textAlign: "center", marginBottom: 24 }}>
          <div style={{ fontSize: 44, marginBottom: 12 }}>🔒</div>
          <h2
            style={{
              fontSize: 20,
              fontWeight: 700,
              color: "var(--text)",
              marginBottom: 8,
            }}
          >
            会員限定コンテンツ
          </h2>
          <p style={{ fontSize: 14, color: "var(--text-dim)", lineHeight: 1.6 }}>
            このページは会員限定です。月額プランにご加入いただくと
            すべての機能をご利用いただけます。
          </p>
        </div>

        {/* Feature list */}
        <ul
          style={{
            listStyle: "none",
            padding: 0,
            margin: "0 0 28px",
            background: "var(--surface2)",
            borderRadius: 10,
            overflow: "hidden",
          }}
        >
          {[
            "AI予測（全馬の勝率・複勝率・回収率）",
            "血統分析・種牡馬メモ",
            "詳細データ分析",
            "馬券最適化",
          ].map((feat) => (
            <li
              key={feat}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 10,
                padding: "11px 18px",
                borderBottom: "1px solid var(--border)",
                fontSize: 14,
                color: "var(--text)",
              }}
            >
              <span style={{ color: "var(--ok)", fontWeight: 700, flexShrink: 0 }}>✓</span>
              {feat}
            </li>
          ))}
        </ul>

        {/* Action buttons */}
        <div style={{ display: "flex", gap: 10 }}>
          <Link
            href="/login"
            style={{
              flex: 1,
              textAlign: "center",
              padding: "11px 0",
              borderRadius: 8,
              fontSize: 14,
              fontWeight: 600,
              border: "1px solid var(--border)",
              color: "var(--text)",
              textDecoration: "none",
              background: "var(--surface2)",
            }}
          >
            ログイン
          </Link>
          <Link
            href="/#pricing"
            onClick={onClose}
            style={{
              flex: 1,
              textAlign: "center",
              padding: "11px 0",
              borderRadius: 8,
              fontSize: 14,
              fontWeight: 600,
              background: "var(--accent)",
              color: "#fff",
              textDecoration: "none",
              border: "none",
            }}
          >
            料金プランを見る
          </Link>
        </div>
      </div>
    </div>
  );
}
