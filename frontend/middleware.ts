import { NextRequest, NextResponse } from "next/server";
import { pathEnabled } from "@/lib/features";
import { MAINTENANCE_PATH, STATUS_URL } from "@/lib/maintenance";

// 開発者限定ページ。src/api/auth.py の DEV_ONLY_PAGES と同じ一覧（変更時は両方を揃える）。
const DEV_ONLY_PAGES = [
  "/monitor",
  "/data-viewer",
  "/queue-status",
  "/server-logs",
  "/scrape-upcoming",
  "/betting",
  "/admin/users",
  "/track-speed/dev",
];

const DEV_COOKIE = "keiba_dev_session";

// dev の自動ログイン（NEXT_PUBLIC_DEV_AUTO_LOGIN=1）。この PC からの直アクセス（Host が localhost 系）のときだけ。
// tcpexposer など公開ドメイン経由は Host が公開ドメインになるので対象外。バックエンドも同じ条件で判定する（src/api/auth.py）。
const DEV_AUTO_LOGIN = process.env.NEXT_PUBLIC_DEV_AUTO_LOGIN === "1";
const LOCAL_HOSTS = new Set(["localhost", "127.0.0.1", "[::1]", "::1"]);

function isLocalDirectRequest(request: NextRequest): boolean {
  if (!DEV_AUTO_LOGIN) return false;
  const hostname = (h: string | null) => (h ?? "").split(",")[0].trim().toLowerCase().replace(/:\d+$/, "");
  const forwarded = request.headers.get("x-forwarded-host");
  if (forwarded && !LOCAL_HOSTS.has(hostname(forwarded))) return false;
  if (!LOCAL_HOSTS.has(hostname(request.headers.get("host")))) return false;
  const xff = request.headers.get("x-forwarded-for");
  // IPv4 射影（::ffff:127.0.0.1）も含めてループバックだけを許可
  const isLoopback = (x: string) => {
    const s = x.trim().replace(/^\[|\]$/g, "").replace(/^::ffff:/i, "");
    return s === "::1" || /^127\./.test(s);
  };
  if (xff && !xff.split(",").every(isLoopback)) return false;
  return true;
}

function isDevOnly(pathname: string): boolean {
  return DEV_ONLY_PAGES.some((p) => pathname === p || pathname.startsWith(p + "/"));
}

// 脆弱性スキャナ・収集プログラムが探す典型的なパス。このサイトには存在しないので、踏んだ時点で API 側のハニーポットに回してブロックする。
const SCANNER_PROBES = [
  "/.env",
  "/.git",
  "/wp-login.php",
  "/wp-admin",
  "/xmlrpc.php",
  "/phpmyadmin",
  "/vendor/phpunit",
  "/cgi-bin",
  "/admin/backup",
  "/backup.zip",
];

function isScannerProbe(pathname: string): boolean {
  const p = pathname.toLowerCase();
  return SCANNER_PROBES.some((x) => p === x || p.startsWith(x + "/") || p.startsWith(x + "."));
}

// ── メンテナンス時間帯（毎日 JST 03:00–04:00）。判定の正は API（/api/maintenance/status）。15 秒だけメモリに持つ ──
const MAINTENANCE_PASS_PREFIXES = ["/api/", "/_next/", "/data/image/", "/static/", "/favicon", "/login", "/logout", MAINTENANCE_PATH];
let maintCache: { at: number; ttl: number; active: boolean; retry: number } | null = null;

async function maintenanceState(): Promise<{ active: boolean; retry: number }> {
  const now = Date.now();
  if (maintCache && now - maintCache.at < maintCache.ttl) return maintCache;
  const base = process.env.KEIBA_FASTAPI_URL;
  if (!base) return { active: false, retry: 0 };
  try {
    const ctl = new AbortController();
    const t = setTimeout(() => ctl.abort(), 1500);
    const res = await fetch(`${base}${STATUS_URL}`, { cache: "no-store", signal: ctl.signal });
    clearTimeout(t);
    if (!res.ok) throw new Error(String(res.status));
    const j = (await res.json()) as { active?: boolean; remaining_sec?: number | null; retry_after?: number };
    const active = j.active === true;
    const rem = typeof j.remaining_sec === "number" ? j.remaining_sec : 15;
    maintCache = { at: now, ttl: active ? Math.max(1000, Math.min(15_000, rem * 1000)) : 15_000, active, retry: j.retry_after ?? 60 };
  } catch {
    // 取れないときは通常扱い（API が止まっていれば各画面の共通ハンドラが切り替える）
    maintCache = { at: now, ttl: 5_000, active: false, retry: 0 };
  }
  return maintCache;
}

export async function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;

  if (isScannerProbe(pathname)) {
    const url = request.nextUrl.clone();
    url.pathname = "/api/_hp/probe";
    url.search = "";
    return NextResponse.rewrite(url);
  }

  if (!MAINTENANCE_PASS_PREFIXES.some((p) => pathname === p || pathname.startsWith(p))) {
    const st = await maintenanceState();
    // 管理者のセッション Cookie がある人は通す（本当の検証は API 側。偽の Cookie では API が 503 を返し、共通ハンドラがメンテナンス画面にする）
    if (st.active && !request.cookies.get(DEV_COOKIE)) {
      const url = request.nextUrl.clone();
      url.pathname = MAINTENANCE_PATH;
      url.search = "";
      return NextResponse.rewrite(url, {
        status: 503,
        headers: { "Retry-After": String(st.retry || 60), "Cache-Control": "no-store, max-age=0", "X-Robots-Tag": "noindex", "X-Keiba-Maintenance": "1" },
      });
    }
  }

  // 無効化中の機能（lib/features.ts）への直リンクはホームへ戻す
  if (!pathEnabled(pathname)) {
    const url = request.nextUrl.clone();
    url.pathname = "/";
    url.search = "";
    return NextResponse.redirect(url);
  }

  if (!isDevOnly(pathname)) return NextResponse.next();

  if (isLocalDirectRequest(request)) return NextResponse.next();

  // Edge ではクッキーの有無だけを見る。署名検証と実データの認可は API 側（auth.py）が行う。
  if (request.cookies.get(DEV_COOKIE)) return NextResponse.next();

  const url = request.nextUrl.clone();
  url.pathname = "/login";
  url.search = "";
  url.searchParams.set("next", pathname);
  return NextResponse.redirect(url);
}

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};
