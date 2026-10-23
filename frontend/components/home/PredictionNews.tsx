"use client";

import { useEffect, useRef, useState } from "react";
import { PendingLink } from "@/components/PendingLink";
import { getJson } from "@/lib/httpCache";
import { useAuthStatus } from "@/lib/hooks/useAuthStatus";
import type { NewsItem } from "@/lib/predictionResults";

const KIND_META: Record<NewsItem["kind"], { icon: string; accent: string }> = {
  day: { icon: "📰", accent: "var(--accent)" },
  week: { icon: "📈", accent: "#34d399" },
  ana_hit: { icon: "🎯", accent: "#f59e0b" },
  big_return: { icon: "💰", accent: "#f59e0b" },
  big_miss: { icon: "⚠️", accent: "#f87171" },
};

const INTERVAL_MS = 6000;

const CSS = `
.pn{max-width:1100px;margin:0 auto 16px;padding:0 20px}
.pn-head{display:flex;align-items:center;gap:10px;margin-bottom:8px;flex-wrap:wrap}
.pn-head h2{margin:0;font-size:16px;font-weight:700;color:var(--text)}
.pn-ctrl{display:flex;align-items:center;gap:4px}
.pn-btn{appearance:none;border:1px solid var(--border);background:transparent;color:var(--text-dim);width:26px;height:26px;border-radius:50%;cursor:pointer;font-size:13px;line-height:1;display:grid;place-items:center}
.pn-btn:hover{color:var(--text);border-color:rgba(148,163,184,.55)}
.pn-dot{appearance:none;border:0;padding:0;width:18px;height:18px;background:transparent;cursor:pointer;display:grid;place-items:center}
.pn-dot::before{content:"";display:block;width:7px;height:7px;border-radius:50%;background:rgba(148,163,184,.4);transition:all .2s}
.pn-dot[aria-current="true"]::before{width:18px;border-radius:4px;background:var(--accent)}
.pn-link{margin-left:auto;font-size:12px;color:var(--accent);text-decoration:none}
.pn-card{overflow:hidden;background:var(--surface);border:1px solid var(--border);border-radius:10px}
.pn-track{display:flex;transition:transform .55s cubic-bezier(.4,0,.2,1)}
.pn-slide{flex:0 0 100%;min-width:0;box-sizing:border-box;display:block;text-decoration:none;color:var(--text);padding:10px 14px;border-left:4px solid var(--pn-accent,var(--accent));min-height:68px}
.pn-top{display:flex;align-items:center;gap:8px;margin-bottom:3px}
.pn-title{font-weight:700;font-size:14px;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.pn-badge{margin-left:auto;flex:none;font-size:11px;font-weight:700;padding:1px 8px;border-radius:999px;border:1px solid}
.pn-sub{font-size:13px;color:var(--text-dim);line-height:1.5;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
@media (prefers-reduced-motion:reduce){.pn-track{transition:none}}
@media (max-width:640px){.pn{padding:0 12px}}
`;

/** ホーム上部の『予測ニュース』: 直近の結果まとめ・今週の累計・注目レースを、1 枚のカードの中でスライドショーのように流す（省スペース）。
 *  自動で 6 秒ごとに切り替え、マウスを乗せている間・フォーカス中・動きを減らす設定のときは止める。矢印・点で手動でも切り替えられる。
 *  回収率（仮想購入）は会員向け。未ログインの間は回収率を除いた文言を出す。集計が1件も無ければ何も出さない。 */
export function PredictionNews() {
  const { isMember, loading } = useAuthStatus();
  const [items, setItems] = useState<NewsItem[] | null>(null);
  const [idx, setIdx] = useState(0);
  const [paused, setPaused] = useState(false);
  const [reduced, setReduced] = useState(false);
  const touchX = useRef<number | null>(null);

  useEffect(() => {
    let alive = true;
    getJson<{ items: NewsItem[] }>("/api/prediction-results/news?limit=5", { ttlMs: 30_000 })
      .then((d) => { if (alive) setItems(d.items ?? []); })
      .catch(() => { if (alive) setItems([]); });
    return () => { alive = false; };
  }, []);

  useEffect(() => {
    setReduced(typeof window !== "undefined" && !!window.matchMedia?.("(prefers-reduced-motion: reduce)").matches);
  }, []);

  const n = items?.length ?? 0;
  useEffect(() => {
    if (n < 2 || paused || reduced) return;
    const t = setInterval(() => setIdx((i) => (i + 1) % n), INTERVAL_MS);
    return () => clearInterval(t);
  }, [n, paused, reduced]);

  if (!items || n === 0) return null;
  const go = (i: number) => setIdx(((i % n) + n) % n);

  return (
    <section aria-label="予測ニュース" aria-roledescription="carousel" className="pn">
      <style>{CSS}</style>
      <div className="pn-head">
        <h2>📰 予測ニュース</h2>
        {n > 1 && (
          <div className="pn-ctrl">
            <button type="button" className="pn-btn" aria-label="前のニュース" onClick={() => go(idx - 1)}>‹</button>
            {items.map((_, i) => (
              <button key={i} type="button" className="pn-dot" aria-label={`${i + 1} 件目`} aria-current={i === idx} onClick={() => go(i)} />
            ))}
            <button type="button" className="pn-btn" aria-label="次のニュース" onClick={() => go(idx + 1)}>›</button>
          </div>
        )}
        <PendingLink href="/prediction-results" className="pn-link">予測成績をすべて見る →</PendingLink>
      </div>
      <div
        className="pn-card"
        onMouseEnter={() => setPaused(true)}
        onMouseLeave={() => setPaused(false)}
        onFocus={() => setPaused(true)}
        onBlur={() => setPaused(false)}
        onTouchStart={(e) => { touchX.current = e.touches[0].clientX; setPaused(true); }}
        onTouchEnd={(e) => {
          const x0 = touchX.current;
          touchX.current = null;
          if (x0 != null && Math.abs(e.changedTouches[0].clientX - x0) > 40) go(idx + (e.changedTouches[0].clientX < x0 ? 1 : -1));
          setTimeout(() => setPaused(false), 4000);
        }}
      >
        <div className="pn-track" style={{ transform: `translateX(-${idx * 100}%)` }} aria-live={paused ? "polite" : "off"}>
          {items.map((it, i) => {
            const meta = KIND_META[it.kind] ?? KIND_META.day;
            const text = !loading && isMember ? it.subtitle : it.subtitle_public;
            const ok = it.badge === "確定";
            return (
              <PendingLink
                key={`${it.kind}-${i}`}
                href={it.link}
                className="pn-slide"
                style={{ ["--pn-accent" as string]: meta.accent }}
                aria-hidden={i !== idx}
                tabIndex={i === idx ? 0 : -1}
                aria-label={`${i + 1} / ${n}`}
              >
                <div className="pn-top">
                  <span aria-hidden>{meta.icon}</span>
                  <span className="pn-title">{it.title}</span>
                  <span className="pn-badge" style={{ color: ok ? "#4ade80" : "#fbbf24", borderColor: ok ? "rgba(74,222,128,0.5)" : "rgba(251,191,36,0.5)" }}>{it.badge}</span>
                </div>
                <div className="pn-sub">{text}</div>
              </PendingLink>
            );
          })}
        </div>
      </div>
      {!loading && !isMember && (
        <p style={{ margin: "6px 0 0", fontSize: 11, color: "var(--text-dim)" }}>推奨馬券の回収率は、ログインすると表示されます。</p>
      )}
    </section>
  );
}
