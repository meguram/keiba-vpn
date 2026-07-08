"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import {
  ADMIN_CATEGORIES,
  CategorySection,
  NavCardItem,
  PUBLIC_CATEGORIES,
} from "@/components/home/homeData";
import { USE_MOCK } from "@/lib/mock";

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
  const [isAdmin, setIsAdmin] = useState(false);

  useEffect(() => {
    if (USE_MOCK) return; // devモックではadmin不要（管理者セクションは非表示）
    fetch("/api/v1/auth/status", { credentials: "include" })
      .then((r) => (r.ok ? r.json() : { logged_in: false, is_admin: false }))
      .then((d) => setIsAdmin(!!d.logged_in && !!d.is_admin))
      .catch(() => setIsAdmin(false));
  }, []);

  return (
    <div className="home-root">
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

      <footer>
        ML-AutoPilot Keiba &mdash; Multi-Agent AI Prediction System &mdash; データソース:{" "}
        <a href="https://db.netkeiba.com/" target="_blank" rel="noreferrer">
          netkeiba.com
        </a>
      </footer>
    </div>
  );
}
