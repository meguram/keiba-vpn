"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { DevToolbar } from "@/components/DevToolbar";

const groups = [
  { label: "Home", href: "/", color: "#c8d6e5" },
  { label: "AI予測", href: "/tracking-difficulty", color: "#58a6ff" },
  { label: "血統", href: "/bloodline-cluster", color: "#bc8cff" },
  { label: "データ分析", href: "/track-speed", color: "#39d2c0" },
  { label: "馬券", href: "/betting", color: "#3fb950" },
];

export function Nav() {
  const path = usePathname();
  const [loggedIn, setLoggedIn] = useState(false);

  useEffect(() => {
    fetch("/api/v1/auth/status", { credentials: "include" })
      .then((r) => (r.ok ? r.json() : { logged_in: false }))
      .then((d) => setLoggedIn(!!d.logged_in))
      .catch(() => setLoggedIn(false));
  }, [path]);

  return (
    <nav className="sticky top-0 z-50 flex h-11 items-center gap-4 border-b px-4" style={{ background: "var(--surface)", borderColor: "var(--border)" }}>
      <Link href="/" className="font-bold text-white">keiba-vpn</Link>
      {groups.map((g) => (
        <Link key={g.href} href={g.href} style={{ color: path.startsWith(g.href) && g.href !== "/" ? g.color : "var(--text-dim)" }} className="text-sm">
          {g.label}
        </Link>
      ))}
      <DevToolbar loggedIn={loggedIn} />
      {!loggedIn && (
        <Link href="/login" className="ml-auto text-sm" style={{ color: "var(--text-dim)" }}>
          ログイン
        </Link>
      )}
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
