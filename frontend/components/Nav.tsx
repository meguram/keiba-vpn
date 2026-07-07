"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { USE_MOCK } from "@/lib/mock";

export function Nav() {
  const path = usePathname();
  const [loggedIn, setLoggedIn] = useState(false);
  const [isAdmin, setIsAdmin] = useState(false);

  useEffect(() => {
    if (USE_MOCK) return; // mockモードではFlask認証不要
    fetch("/api/v1/auth/status", { credentials: "include" })
      .then((r) => (r.ok ? r.json() : { logged_in: false, is_admin: false }))
      .then((d) => {
        setLoggedIn(!!d.logged_in);
        setIsAdmin(!!d.logged_in && !!d.is_admin);
      })
      .catch(() => { setLoggedIn(false); setIsAdmin(false); });
  }, [path]);

  const isHome = path === "/";

  return (
    <>
      {USE_MOCK && (
        <div style={{
          background: "rgba(88,166,255,0.07)",
          borderBottom: "1px solid rgba(88,166,255,0.18)",
          padding: "5px 20px",
          fontSize: 11,
          color: "#58a6ff",
          textAlign: "center",
          letterSpacing: "0.01em",
        }}>
          🔧 開発モード（モックデータ表示中）— 実際のレースデータではありません
        </div>
      )}
      <nav style={{
        position: "sticky",
        top: 0,
        zIndex: 50,
        display: "flex",
        height: 44,
        alignItems: "center",
        padding: "0 20px",
        background: "#0d1117",
        borderBottom: "1px solid #21262d",
      }}>
        <Link href="/" style={{
          fontWeight: 700,
          fontSize: 14,
          color: "#f0f6fc",
          textDecoration: "none",
          letterSpacing: "-0.3px",
          display: "flex",
          alignItems: "center",
          gap: 8,
        }}>
          <span style={{ fontSize: 16 }}>🏇</span>
          <span>keiba-vpn</span>
        </Link>

        {!isHome && (
          <Link href="/" style={{
            marginLeft: 16,
            fontSize: 12,
            color: "#6e7681",
            textDecoration: "none",
            display: "flex",
            alignItems: "center",
            gap: 4,
            transition: "color 0.15s",
          }}
          onMouseEnter={(e) => (e.currentTarget.style.color = "#c9d1d9")}
          onMouseLeave={(e) => (e.currentTarget.style.color = "#6e7681")}
          >
            ← ホーム
          </Link>
        )}

        <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 12 }}>
          {isAdmin && (
            <Link href="/betting" style={{
              fontSize: 12,
              color: "#3fb950",
              textDecoration: "none",
              padding: "4px 10px",
              borderRadius: 6,
              border: "1px solid rgba(63,185,80,0.3)",
              transition: "all 0.15s",
            }}
            onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.background = "rgba(63,185,80,0.1)"; }}
            onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.background = "transparent"; }}
            >
              💰 馬券
            </Link>
          )}
          {loggedIn ? (
            <span style={{
              fontSize: 12,
              color: "#6e7681",
              padding: "4px 10px",
              borderRadius: 6,
              border: "1px solid #21262d",
            }}>
              管理者
            </span>
          ) : (
            <Link href="/login" style={{
              fontSize: 12,
              color: "#58a6ff",
              textDecoration: "none",
              padding: "4px 12px",
              borderRadius: 6,
              border: "1px solid rgba(88,166,255,0.3)",
              transition: "all 0.15s",
            }}
            onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.background = "rgba(88,166,255,0.1)"; }}
            onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.background = "transparent"; }}
            >
              ログイン
            </Link>
          )}
        </div>
      </nav>
    </>
  );
}

export function AnalysisNote() {
  return (
    <p className="analysis-note">
      ※ この統計はリアルタイム集計です。AI モデルが予測に使用した時点の特徴量とは異なる場合があります。
    </p>
  );
}
