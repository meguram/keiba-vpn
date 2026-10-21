"use client";

import { usePathname } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  BLOCK_EVENT, MAINTENANCE_PATH, STATUS_URL, approxMinutes, formatCountdown, hhmm, isMaintenanceResponse, livePhase, parseStatus,
  previewStatus, safeNext, secondsLeft, windowLabel,
  type MaintenanceStatus, type Stamped,
} from "@/lib/maintenance";

/* ── ステータスの取得（サーバ時刻基準）。端末の時計は使わず、受信時刻からの経過で残りを数える ── */

async function fetchStatus(): Promise<MaintenanceStatus | null> {
  try {
    const res = await fetch(STATUS_URL, { cache: "no-store", headers: { "x-keiba-maintenance-poll": "1" } });
    if (!res.ok) return null;
    return parseStatus(await res.json());
  } catch {
    return null;
  }
}

let hookInstalled = false;
/** fetch を 1 回だけ包み、メンテナンスの 503（X-Keiba-Maintenance）や API の 502/504 を検知して全画面共通で切り替える。 */
function installFetchHook() {
  if (hookInstalled || typeof window === "undefined") return;
  hookInstalled = true;
  const orig = window.fetch.bind(window);
  window.fetch = async (...args: Parameters<typeof fetch>) => {
    const res = await orig(...args);
    try {
      const req = args[0];
      const url = typeof req === "string" ? req : req instanceof URL ? req.href : (req as Request).url;
      const isPoll = url.includes(STATUS_URL);
      if (!isPoll) {
        if (isMaintenanceResponse(res)) {
          res.clone().json().then((body) => window.dispatchEvent(new CustomEvent(BLOCK_EVENT, { detail: { body } }))).catch(() =>
            window.dispatchEvent(new CustomEvent(BLOCK_EVENT, { detail: { body: null } })));
        } else if ((res.status === 502 || res.status === 504) && url.includes("/api/")) {
          window.dispatchEvent(new CustomEvent(BLOCK_EVENT, { detail: { recheck: true } }));
        }
      }
    } catch { /* 検知の失敗で本来の通信を壊さない */ }
    return res;
  };
}

/* ── 表示 ── */

export function MaintenanceScreen({ stamped, now, onRecheck, ended, nextUrl }: {
  stamped: Stamped; now: number; onRecheck: () => void; ended: boolean; nextUrl: string;
}) {
  const s = stamped.status;
  const left = secondsLeft(s.remaining_sec, stamped.receivedAt, now);
  const minutes = left == null ? "" : approxMinutes(left);
  const label = windowLabel(s);
  return (
    <div className="mt-screen" data-testid="maintenance-screen">
      <div className="mt-card">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img className="mt-logo" src="/data/image/icon.jpg" alt="" width={56} height={56} onError={(e) => { e.currentTarget.style.display = "none"; }} />
        {ended ? (
          <>
            <h1 className="mt-title">メンテナンスが終了しました</h1>
            <p className="mt-lead" role="status" aria-live="polite">まもなく元の画面に戻ります。</p>
            <a className="mt-btn" href={nextUrl}>今すぐ戻る</a>
          </>
        ) : (
          <>
            <h1 className="mt-title">メンテナンス中です</h1>
            <p className="mt-lead">{s.message || "ただいまメンテナンス中です。終了までしばらくお待ちください。"}</p>
            {label && (
              <p className="mt-window">
                終了予定 <b>{s.ends_at_label || hhmm(s.ends_at)}</b>（日本時間）
                <span className="mt-dim"> ／ メンテナンス時間 {label}</span>
              </p>
            )}
            <div className="mt-count" aria-hidden="true">
              <span className="mt-count-num">{formatCountdown(left)}</span>
              <span className="mt-count-cap">終了まで</span>
            </div>
            <p className="mt-sr" role="status" aria-live="polite">{left == null ? "メンテナンス中です。" : left <= 60 ? "メンテナンス中です。まもなく終了します。" : `メンテナンス中です。終了まで${minutes.replace("あと", "")}です。`}</p>
            <p className="mt-note">終了すると、この画面は自動で元のページに戻ります。ご不便をおかけして申し訳ありません。</p>
            <button type="button" className="mt-btn mt-btn-ghost" onClick={onRecheck}>状態を再確認する</button>
          </>
        )}
      </div>
    </div>
  );
}

export function MaintenanceBanner({ stamped, now }: { stamped: Stamped; now: number }) {
  const s = stamped.status;
  const left = secondsLeft(s.starts_in_sec, stamped.receivedAt, now);
  return (
    <div className="mt-banner" role="status" aria-live="polite" data-testid="maintenance-banner">
      <span className="mt-banner-ico" aria-hidden="true">⚠</span>
      <span>
        まもなくメンテナンスを行います（{windowLabel(s) ? `${windowLabel(s)} 日本時間` : "まもなく"}）。{left != null && left > 60 ? `${approxMinutes(left)}で始まります。` : ""}この間はご利用いただけません。
      </span>
    </div>
  );
}

/* ── 全体を包む（layout から使う） ── */

export function MaintenanceShell({ chrome, children }: { chrome: React.ReactNode; children: React.ReactNode }) {
  const path = usePathname();
  const [stamped, setStamped] = useState<Stamped | null>(null);
  const [now, setNow] = useState(() => Date.now());
  const [preview, setPreview] = useState<MaintenanceStatus | null>(null);
  const [ended, setEnded] = useState(false);
  const wasBlocked = useRef(false);
  const lastRecheck = useRef(0);

  const refresh = useCallback(async () => {
    const st = await fetchStatus();
    if (st) setStamped({ status: st, receivedAt: Date.now() });
  }, []);

  // 初回: fetch の包み・プレビュー（dev のみ ?maintenance=notice|active|ended）・状態取得
  useEffect(() => {
    installFetchHook();
    if (process.env.NODE_ENV !== "production") {
      const q = new URLSearchParams(window.location.search);
      const pv = previewStatus(q.get("maintenance"), q.get("remaining"));
      if (pv) { setPreview(pv); setStamped({ status: pv, receivedAt: Date.now() }); return; }
    }
    refresh();
  }, [refresh]);

  // 画面の更新（秒）・ポーリング（通常 60 秒、事前告知/開催中は 15 秒、終了後の確認は 5 秒）
  useEffect(() => {
    const tick = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(tick);
  }, []);
  const phase = stamped ? livePhase(stamped, now) : "normal";
  useEffect(() => {
    if (preview) return;
    const every = phase === "normal" ? 60_000 : 15_000;
    const t = setInterval(refresh, every);
    return () => clearInterval(t);
  }, [phase, preview, refresh]);

  // 開始・終了の境界を跨いだら、すぐ取り直す（ローカルの経過で先取りした表示を、サーバの状態で確定する）
  const prevPhase = useRef(phase);
  useEffect(() => {
    if (prevPhase.current !== phase && !preview) refresh();
    prevPhase.current = phase;
  }, [phase, preview, refresh]);

  // API が 503（メンテナンス）/502/504 を返したら、全画面共通で切り替える
  useEffect(() => {
    const onBlock = (e: Event) => {
      const d = (e as CustomEvent).detail as { body?: unknown; recheck?: boolean } | undefined;
      if (d?.body !== undefined) {
        const st = parseStatus(d.body ?? { maintenance: true });
        setStamped({ status: st.active ? st : { ...st, active: true, phase: "active" }, receivedAt: Date.now() });
      } else if (d?.recheck && Date.now() - lastRecheck.current > 3000) {
        lastRecheck.current = Date.now();
        refresh();
      }
    };
    window.addEventListener(BLOCK_EVENT, onBlock);
    return () => window.removeEventListener(BLOCK_EVENT, onBlock);
  }, [refresh]);

  const blocked = !!stamped && phase === "active";
  // 開催中→終了を検出したら、案内を出して再読み込み（保持中の画面状態に古い失敗が残らないように）
  useEffect(() => {
    if (blocked) { wasBlocked.current = true; return; }
    if (wasBlocked.current && !preview) {
      wasBlocked.current = false;
      setEnded(true);
      const t = setTimeout(() => window.location.reload(), 1500);
      return () => clearTimeout(t);
    }
  }, [blocked, preview]);

  if (path === MAINTENANCE_PATH) return <>{children}</>; // 専用ページは自前で表示する

  if (blocked || ended) {
    return (
      <MaintenanceScreen
        stamped={stamped ?? { status: previewStatus("active") as MaintenanceStatus, receivedAt: Date.now() }}
        now={now}
        onRecheck={refresh}
        ended={ended && !blocked}
        nextUrl={typeof window !== "undefined" ? window.location.pathname + window.location.search : "/"}
      />
    );
  }
  return (
    <>
      {phase === "notice" && stamped && <MaintenanceBanner stamped={stamped} now={now} />}
      {chrome}
      <main className="site-main">{children}</main>
    </>
  );
}

/* ── 専用ページ（/maintenance。Next.js の middleware がメンテナンス中の全ページをここへ書き換える） ── */

export function MaintenancePage() {
  const [stamped, setStamped] = useState<Stamped | null>(null);
  const [now, setNow] = useState(() => Date.now());
  const [ended, setEnded] = useState(false);
  const [next, setNext] = useState("/");
  const [preview, setPreview] = useState(false);

  const refresh = useCallback(async () => {
    const st = await fetchStatus();
    if (st) setStamped({ status: st, receivedAt: Date.now() });
  }, []);

  useEffect(() => {
    const q = new URLSearchParams(window.location.search);
    // middleware の書き換えでは URL が元のページのまま。/maintenance を直接開いたときだけ ?next= を見る
    const here = window.location.pathname;
    setNext(here === MAINTENANCE_PATH ? safeNext(q.get("next")) : safeNext(here + window.location.search));
    if (process.env.NODE_ENV !== "production") {
      const pv = previewStatus(q.get("maintenance") ?? q.get("preview"), q.get("remaining"));
      if (pv) { setPreview(true); setStamped({ status: pv, receivedAt: Date.now() }); return; }
    }
    refresh();
  }, [refresh]);
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, []);
  const phase = stamped ? livePhase(stamped, now) : "active";
  useEffect(() => {
    if (preview) return;
    const t = setInterval(refresh, phase === "active" ? 15_000 : 5_000);
    return () => clearInterval(t);
  }, [phase, preview, refresh]);
  useEffect(() => {
    if (preview || !stamped || phase === "active") return;
    // 状態が「通常」と確定してから戻る（先取りの 0 秒では戻らず、サーバの確認を待つ）
    if (!stamped.status.active && stamped.status.phase !== "active") {
      setEnded(true);
      const t = setTimeout(() => window.location.replace(next), 1200);
      return () => clearTimeout(t);
    }
  }, [phase, stamped, preview, next]);

  const st = stamped ?? { status: previewStatus("active") as MaintenanceStatus, receivedAt: Date.now() };
  if (preview && stamped?.status.phase === "normal") return <MaintenanceScreen stamped={st} now={now} onRecheck={refresh} ended nextUrl={next} />;
  return <MaintenanceScreen stamped={st} now={now} onRecheck={refresh} ended={ended} nextUrl={next} />;
}
