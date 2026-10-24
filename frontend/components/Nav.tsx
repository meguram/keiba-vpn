"use client";

import { fetchAuthStatus } from "@/lib/httpCache";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { USE_MOCK } from "@/lib/mock";
import { isNavActive, visibleNavItems } from "@/lib/siteNav";

export function Nav() {
  const path = usePathname();
  const [loggedIn, setLoggedIn] = useState(false);
  const [userName, setUserName] = useState("");
  const [isAdmin, setIsAdmin] = useState(false);
  const [paid, setPaid] = useState(false);

  useEffect(() => {
    if (USE_MOCK) {
      // mockモードではFlask認証不要。dev の自動ログインが有効ならログイン済み表示にする
      const auto = process.env.NEXT_PUBLIC_DEV_AUTO_LOGIN === "1";
      setLoggedIn(auto);
      setUserName(auto ? "開発者" : "");
      setIsAdmin(auto);
      setPaid(auto);
      return;
    }
    fetchAuthStatus()
      .then((d) => {
        setLoggedIn(!!d.logged_in);
        setUserName(d.logged_in ? (d.user_name || (d.is_admin ? "管理者" : "会員")) : "");
        setIsAdmin(!!d.logged_in && !!d.is_admin);
        setPaid(!!d.logged_in && !!d.is_member);
      })
      .catch(() => { /* 一時的な通信失敗・サーバエラーでは、直前のログイン表示を保つ */ });
  }, [path]);

  const navItems = visibleNavItems();

  const logout = async () => {
    try {
      await fetch("/api/v1/auth/logout", { method: "POST", credentials: "include" });
    } catch { /* 通信失敗でも画面は更新する（Cookie が残っていれば、まだログイン中と表示される） */ }
    window.location.href = "/";
  };

  return (
    <>
      {/* ハニーポット: 人間には見えず押せない（画面外・読み上げ対象外・フォーカス不可）。機械的にリンクをたどる収集プログラムだけが踏む。
          robots.txt で /api/ を拒否しているので、守る相手は踏まない。踏むと API 側が即ブロックする（config/anti_scraping.json） */}
      <a
        href="/api/archive/all-predictions"
        rel="nofollow noindex noarchive"
        tabIndex={-1}
        aria-hidden="true"
        style={{ position: "absolute", left: -9999, top: 0, width: 1, height: 1, overflow: "hidden" }}
      >
        全予測データ
      </a>
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
      <nav className="site-nav" style={{
        position: "sticky",
        top: 0,
        zIndex: 50,
        display: "flex",
        height: 44,
        alignItems: "center",
        background: "#0d1117",
        borderBottom: "1px solid #21262d",
      }}>
        <Link href="/" className="hit44 nav-brand" style={{
          fontWeight: 700,
          fontSize: 14,
          color: "#f0f6fc",
          textDecoration: "none",
          letterSpacing: "-0.3px",
          display: "flex",
          alignItems: "center",
          gap: 8,
        }}>
          <img src="/megu-icon.png" alt="" width={22} height={22} style={{ borderRadius: "50%", display: "block" }} />
          <span>めぐ競馬</span>
        </Link>

        {navItems.length > 0 && (
          <div className="nav-links" role="navigation" aria-label="メインメニュー">
            {navItems.map((it) => {
              const active = isNavActive(it, path);
              return (
                <Link
                  key={it.id}
                  href={it.href}
                  className={`hit44 nav-link${active ? " is-active" : ""}`}
                  aria-current={active ? "page" : undefined}
                >
                  {it.label}
                </Link>
              );
            })}
          </div>
        )}

        <div className="nav-right" style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 12 }}>
          <span className="nav-user" title={loggedIn ? "ログイン中のユーザー" : "ログインしていません"} style={{
            fontSize: 12,
            color: loggedIn ? "#c9d1d9" : "#6e7681",
            padding: "4px 10px",
            borderRadius: 6,
            border: "1px solid #21262d",
            maxWidth: 160,
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          }}>
            {loggedIn ? `👤 ${userName}${isAdmin ? "" : paid ? "（有料会員）" : "（無料）"}` : "ゲスト"}
          </span>
          {isAdmin && (
            <Link href="/admin/users" className="hit44 nav-btn" style={{
              fontSize: 12,
              color: "#3fb950",
              textDecoration: "none",
              padding: "4px 12px",
              borderRadius: 6,
              border: "1px solid rgba(63,185,80,0.35)",
              transition: "all 0.15s",
            }}
            onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.background = "rgba(63,185,80,0.1)"; }}
            onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.background = "transparent"; }}
            >
              ユーザ登録
            </Link>
          )}
          {loggedIn ? (
            <button type="button" onClick={logout} className="hit44 nav-btn" style={{
              fontSize: 12,
              color: "#f0883e",
              background: "transparent",
              cursor: "pointer",
              padding: "4px 12px",
              borderRadius: 6,
              border: "1px solid rgba(240,136,62,0.35)",
              transition: "all 0.15s",
            }}
            onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.background = "rgba(240,136,62,0.1)"; }}
            onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.background = "transparent"; }}
            >
              ログアウト
            </button>
          ) : (
            <Link href="/login" className="hit44 nav-btn" style={{
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
