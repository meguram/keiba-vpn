"use client";

import Link from "next/link";
import { useState } from "react";
import { useSearchParams } from "next/navigation";
import {
  ADMIN_CATEGORIES,
  CategorySection,
  NavCardItem,
  PUBLIC_CATEGORIES,
  TabbedCardItem,
  isTabbedCard,
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

function TabbedNavCard({ card }: { card: TabbedCardItem }) {
  const [activeIdx, setActiveIdx] = useState(0);
  const active = card.tabs[activeIdx];

  return (
    <div
      className="nav-card"
      style={{ ["--card-accent" as string]: active.accent, padding: 0, cursor: "default" }}
    >
      {/* Tab bar */}
      <div
        style={{
          display: "flex",
          borderBottom: "1px solid var(--border)",
          background: "var(--surface2)",
          borderRadius: "var(--card-radius, 12px) var(--card-radius, 12px) 0 0",
          overflow: "hidden",
        }}
      >
        {card.tabs.map((tab, i) => (
          <button
            key={tab.href}
            type="button"
            onClick={() => setActiveIdx(i)}
            style={{
              flex: 1,
              padding: "9px 4px",
              fontSize: 11,
              fontWeight: i === activeIdx ? 700 : 500,
              color: i === activeIdx ? tab.accent : "var(--text-dim)",
              background: "transparent",
              border: "none",
              borderBottom: i === activeIdx ? `2px solid ${tab.accent}` : "2px solid transparent",
              cursor: "pointer",
              transition: "color .15s, border-color .15s",
              whiteSpace: "nowrap",
              overflow: "hidden",
              textOverflow: "ellipsis",
            }}
          >
            {tab.icon} {tab.title.replace("今週の", "")}
          </button>
        ))}
      </div>

      {/* Card body — navigates to active tab's page */}
      <Link
        href={active.href}
        style={{
          display: "flex",
          flexDirection: "column",
          gap: 6,
          padding: "16px",
          textDecoration: "none",
          color: "inherit",
        }}
      >
        <span className="card-icon">{active.icon}</span>
        <div className="card-title">
          {active.title} <span className="arrow">→</span>
        </div>
        <div className="card-desc">{active.desc}</div>
        <div className="card-tags">
          {active.tags.map((tag) => (
            <span key={tag} className="tag">{tag}</span>
          ))}
        </div>
      </Link>
    </div>
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
        {section.cards.map((card) =>
          isTabbedCard(card) ? (
            <TabbedNavCard key={card.title} card={card} />
          ) : (
            <NavCard key={card.title} card={card} />
          )
        )}
      </div>
    </div>
  );
}

function DevToolbar({ isAdmin }: { isAdmin: boolean }) {
  if (!isAdmin) return null;

  const monitorUrl =
    typeof window !== "undefined"
      ? (process.env.NEXT_PUBLIC_MONITOR_URL ||
          `${window.location.protocol}//${window.location.hostname}:9090`)
      : (process.env.NEXT_PUBLIC_MONITOR_URL || "http://localhost:9090");

  return (
    <div
      style={{
        position: "fixed",
        bottom: 20,
        right: 20,
        zIndex: 1000,
        display: "flex",
        flexDirection: "column",
        alignItems: "flex-end",
        gap: 8,
      }}
    >
      <a
        href={monitorUrl}
        target="_blank"
        rel="noreferrer"
        title="開発者監視ポータルを開く"
        style={{
          display: "flex",
          alignItems: "center",
          gap: 7,
          padding: "8px 14px",
          background: "rgba(30,30,40,0.92)",
          border: "1px solid rgba(100,120,200,0.4)",
          borderRadius: 10,
          color: "#a0b4e8",
          fontSize: 12,
          fontWeight: 600,
          textDecoration: "none",
          backdropFilter: "blur(8px)",
          boxShadow: "0 2px 12px rgba(0,0,0,0.4)",
          transition: "border-color .15s, color .15s",
          cursor: "pointer",
        }}
        onMouseEnter={(e) => {
          (e.currentTarget as HTMLElement).style.borderColor = "rgba(120,160,255,0.7)";
          (e.currentTarget as HTMLElement).style.color = "#c8d8ff";
        }}
        onMouseLeave={(e) => {
          (e.currentTarget as HTMLElement).style.borderColor = "rgba(100,120,200,0.4)";
          (e.currentTarget as HTMLElement).style.color = "#a0b4e8";
        }}
      >
        <span style={{ fontSize: 14 }}>🖥️</span>
        監視ポータル
        <span style={{ fontSize: 10, opacity: 0.6 }}>↗</span>
      </a>
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

      <DevToolbar isAdmin={isAdmin} />

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
