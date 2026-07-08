"use client";

import Link from "next/link";
import { useState } from "react";
import { useSearchParams } from "next/navigation";
import {
  ADMIN_CATEGORIES,
  CategorySection,
  NavCardItem,
  PUBLIC_CATEGORIES,
} from "@/components/home/homeData";
import { useAuthStatus } from "@/lib/hooks/useAuthStatus";
import { MemberUpgradeModal } from "@/components/upgrade/MemberUpgradeModal";

function NavCard({
  card,
}: {
  card: NavCardItem;
}) {
  const inner = (
    <>
      <span className="card-icon">{card.icon}</span>
      <div className="card-title">
        {card.title} <span className="arrow">→</span>
      </div>
      <div className="card-desc">{card.desc}</div>
      <div className="card-tags">
        {card.tags.map((tag) => (
          <span key={tag} className="tag">
            {tag}
          </span>
        ))}
      </div>
    </>
  );

  return (
    <Link href={card.href} className="nav-card" style={{ ["--card-accent" as string]: card.accent }}>
      {inner}
    </Link>
  );
}

function CategoryBlock({
  section,
}: {
  section: CategorySection;
}) {
  return (
    <div className="category" style={{ ["--cat-color" as string]: section.catColor }}>
      <div className="category-header">
        <span className="category-num" style={section.numStyle ? { background: section.numStyle } : undefined}>
          {section.num}
        </span>
        <span className="category-title">{section.title}</span>
        {section.badge && <span className="category-badge">{section.badge}</span>}
        <span className="category-desc">{section.desc}</span>
      </div>
      <div className={`nav-grid${section.singleRow ? " single-card-row" : ""}`}>
        {section.cards.map((card) => (
          <NavCard key={card.title} card={card} />
        ))}
      </div>
    </div>
  );
}

export function HomeDashboard() {
  const { isAdmin, isMember } = useAuthStatus();
  const searchParams = useSearchParams();
  const [showUpgradeModal, setShowUpgradeModal] = useState(
    searchParams.get("upgrade") === "1"
  );

  return (
    <div className="home-root">
      {showUpgradeModal && (
        <MemberUpgradeModal onClose={() => setShowUpgradeModal(false)} />
      )}

      <div className="hero">
        <div className="hero-icon-wrap">
          <span className="icon-glow" />
          <span className="icon-ring" />
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src="/data/image/icon.jpg" alt="icon" className="icon" />
        </div>
        <h1 className="hero-catch">🏇 めぐ競馬</h1>
      </div>

      <div className="main">
        {PUBLIC_CATEGORIES.map((section) => (
          <CategoryBlock key={section.num} section={section} />
        ))}

        {isAdmin && (
          <>
            {ADMIN_CATEGORIES.map((section) => (
              <CategoryBlock key={section.num} section={section} />
            ))}
          </>
        )}
      </div>

      {/* Pricing section — visible to non-members only */}
      {!isMember && (
        <section
          id="pricing"
          style={{
            maxWidth: 480,
            margin: "48px auto 32px",
            padding: "0 20px",
          }}
        >
          <div
            style={{
              background: "var(--surface)",
              border: "1px solid var(--border)",
              borderRadius: 16,
              padding: "32px 28px",
              textAlign: "center",
            }}
          >
            <h2
              style={{
                fontSize: 20,
                fontWeight: 700,
                color: "var(--text)",
                marginBottom: 8,
              }}
            >
              月額プラン
            </h2>
            <p
              style={{
                fontSize: 32,
                fontWeight: 800,
                color: "var(--accent)",
                margin: "16px 0 4px",
              }}
            >
              ¥ ----<span style={{ fontSize: 16, fontWeight: 500, color: "var(--text-dim)" }}>/月</span>
            </p>
            <p style={{ fontSize: 13, color: "var(--text-dim)", marginBottom: 24 }}>
              すべての機能が使い放題
            </p>
            <ul
              style={{
                listStyle: "none",
                padding: 0,
                margin: "0 0 24px",
                background: "var(--surface2)",
                borderRadius: 10,
                overflow: "hidden",
                textAlign: "left",
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
                    padding: "10px 16px",
                    borderBottom: "1px solid var(--border)",
                    fontSize: 13,
                    color: "var(--text)",
                  }}
                >
                  <span style={{ color: "var(--ok)", fontWeight: 700, flexShrink: 0 }}>✓</span>
                  {feat}
                </li>
              ))}
            </ul>
            <Link
              href="/login"
              style={{
                display: "block",
                width: "100%",
                padding: "12px 0",
                borderRadius: 8,
                fontSize: 15,
                fontWeight: 600,
                background: "var(--accent)",
                color: "#fff",
                textDecoration: "none",
                textAlign: "center",
              }}
            >
              お問い合わせ
            </Link>
          </div>
        </section>
      )}

      <footer>
        ML-AutoPilot Keiba &mdash; Multi-Agent AI Prediction System &mdash; データソース:{" "}
        <a href="https://db.netkeiba.com/" target="_blank" rel="noreferrer">
          netkeiba.com
        </a>
      </footer>
    </div>
  );
}
