"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState, useRef } from "react";

type NavItem = { label: string; href: string };
type NavGroup = { label: string; href?: string; color: string; children?: NavItem[]; adminOnly?: boolean };

const NAV_GROUPS: NavGroup[] = [
  { label: "Home", href: "/", color: "#c8d6e5" },
  {
    label: "分析", color: "#58a6ff",
    children: [
      { label: "追走難度分析", href: "/tracking-difficulty" },
      { label: "レース質分析", href: "/race-quality" },
      { label: "成長曲線", href: "/growth-curve" },
      { label: "馬場速度", href: "/track-speed" },
      { label: "適性3D", href: "/note-aptitude-race" },
    ],
  },
  {
    label: "血統", color: "#bc8cff",
    children: [
      { label: "血統研究", href: "/bloodline" },
      { label: "血統ベクトル", href: "/bloodline-vector" },
      { label: "メタクラスタ", href: "/bloodline-cluster" },
      { label: "血統マップ", href: "/pedigree-map" },
      { label: "血統構成分析", href: "/pedigree-race-stats" },
      { label: "MSTN遺伝子", href: "/myostatin" },
    ],
  },
  {
    label: "AI予測", color: "#3fb950",
    children: [
      { label: "🤖 今週のAI予測", href: "/weekly-predictions" },
    ],
  },
  {
    label: "💰 馬券最適化", color: "#3fb950",
    href: "/betting",
    adminOnly: true,
  },
];

function DropdownMenu({ group, path }: { group: NavGroup; path: string }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const isActive = group.children?.some((c) => path.startsWith(c.href)) ?? false;

  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  return (
    <div ref={ref} style={{ position: "relative" }}>
      <button
        onClick={() => setOpen(!open)}
        style={{
          background: "none", border: "none", cursor: "pointer",
          fontSize: 13, color: isActive ? group.color : "var(--text-dim)",
          padding: "0 4px", display: "flex", alignItems: "center", gap: 3,
        }}
      >
        {group.label}
        <span style={{ fontSize: 9, opacity: 0.6 }}>{open ? "▲" : "▼"}</span>
      </button>
      {open && (
        <div style={{
          position: "absolute", top: "calc(100% + 8px)", left: 0, zIndex: 100,
          background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 8,
          minWidth: 160, boxShadow: "0 8px 24px rgba(0,0,0,0.4)", overflow: "hidden",
        }}>
          {group.children?.map((c) => (
            <Link
              key={c.href} href={c.href}
              onClick={() => setOpen(false)}
              style={{
                display: "block", padding: "9px 14px", fontSize: 13,
                color: path.startsWith(c.href) ? group.color : "var(--text-dim)",
                textDecoration: "none", transition: "background 0.1s",
                background: path.startsWith(c.href) ? "rgba(59,130,246,0.06)" : "transparent",
              }}
              onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.background = "rgba(59,130,246,0.08)"; (e.currentTarget as HTMLElement).style.color = "#fff"; }}
              onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.background = path.startsWith(c.href) ? "rgba(59,130,246,0.06)" : "transparent"; (e.currentTarget as HTMLElement).style.color = path.startsWith(c.href) ? group.color : "var(--text-dim)"; }}
            >
              {c.label}
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}

export function Nav() {
  const path = usePathname();
  const [isAdmin, setIsAdmin] = useState(false);
  const [loggedIn, setLoggedIn] = useState(false);

  useEffect(() => {
    fetch("/api/v1/auth/status", { credentials: "include" })
      .then((r) => (r.ok ? r.json() : { logged_in: false, is_admin: false }))
      .then((d) => {
        setLoggedIn(!!d.logged_in);
        setIsAdmin(!!d.logged_in && !!d.is_admin);
      })
      .catch(() => { setLoggedIn(false); setIsAdmin(false); });
  }, [path]);

  return (
    <nav className="sticky top-0 z-50 flex h-11 items-center gap-4 border-b px-4" style={{ background: "var(--surface)", borderColor: "var(--border)" }}>
      <Link href="/" className="font-bold text-white text-sm">keiba-vpn</Link>

      {NAV_GROUPS.map((g) => {
        if (g.adminOnly && !isAdmin) return null;

        if (g.children) {
          return <DropdownMenu key={g.label} group={g} path={path} />;
        }

        return (
          <Link
            key={g.href}
            href={g.href!}
            style={{ color: path === "/" && g.href === "/" ? g.color : path.startsWith(g.href!) && g.href !== "/" ? g.color : "var(--text-dim)", fontSize: 13 }}
          >
            {g.label}
          </Link>
        );
      })}

      <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 8 }}>
        {!loggedIn && (
          <Link href="/login" className="text-sm" style={{ color: "var(--text-dim)" }}>ログイン</Link>
        )}
        {loggedIn && (
          <span className="text-sm" style={{ color: "var(--text-dim)", fontSize: 12 }}>管理者</span>
        )}
      </div>
    </nav>
  );
}

export function AnalysisNote() {
  return (
    <p className="analysis-note">
      ※ この統計はリアルタイム集計です。AI モデルが予測に使用した時点の特徴量とは異なる場合があります。
    </p>
  );
}
