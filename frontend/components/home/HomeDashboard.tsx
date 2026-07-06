"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import {
  ADMIN_CATEGORIES,
  CategorySection,
  NavCardItem,
  PUBLIC_CATEGORIES,
} from "@/components/home/homeData";

type GcsStatus = "green" | "orange" | "red";

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
  const [gcsStatus, setGcsStatus] = useState<GcsStatus>("green");
  const [gcsLabel, setGcsLabel] = useState("GCS");
  const [raceInfo, setRaceInfo] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/v1/auth/status", { credentials: "include" })
      .then((r) => (r.ok ? r.json() : { logged_in: false, is_admin: false }))
      .then((d) => setIsAdmin(!!d.logged_in && !!d.is_admin))
      .catch(() => setIsAdmin(false));
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch("/api/scrape-dates", { cache: "no-store" });
        const data = await res.json();
        if (cancelled) return;
        const dates: string[] = data.dates || [];
        if (data.gcs_enabled !== undefined) {
          if (data.gcs_enabled) {
            setGcsStatus("green");
            setGcsLabel("GCS 接続中");
          } else {
            setGcsStatus("orange");
            setGcsLabel("GCS 未接続");
          }
        }
        if (dates.length > 0) {
          setRaceInfo(`${dates.length}日分のデータ`);
        }
      } catch {
        if (!cancelled) {
          setGcsStatus("red");
          setGcsLabel("GCS ?");
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="home-root">
      <div className="hero">
        <h1>
          <span className="icon-wrap">
            <span className="icon-glow" />
            <span className="icon-ring" />
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src="/data/image/icon.jpg" alt="Meguro" className="icon" />
          </span>{" "}
          ML-<span className="accent">AutoPilot</span> Keiba
        </h1>
        <p className="subtitle">MULTI-AGENT HORSE RACING AI PREDICTION SYSTEM</p>
        <div className="status-row">
          <span className="status-chip">
            <span className={`dot ${gcsStatus}`} /> <span>{gcsLabel}</span>
          </span>
          <span className="status-chip">
            <span className="dot green" /> <span>Server OK</span>
          </span>
          {raceInfo && (
            <span className="status-chip">
              <span className="dot green" /> <span>{raceInfo}</span>
            </span>
          )}
        </div>
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
