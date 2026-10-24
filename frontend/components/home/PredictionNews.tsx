"use client";

import { useCallback, useEffect, useRef, useState } from "react";
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

const INTERVAL_MS = 5000;
const GAP = 10;

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
@media (hover:none) and (pointer:coarse){.pn-btn{width:40px;height:40px;font-size:16px}.pn-dot{width:26px;height:40px}}
.pn-link{margin-left:auto;font-size:12px;color:var(--accent);text-decoration:none}
.pn-viewport{overflow:hidden;width:calc(100% + ${GAP}px);margin-right:-${GAP}px}
.pn-track{display:flex;will-change:transform}
.pn-track.anim{transition:transform .65s cubic-bezier(.4,0,.2,1)}
.pn-cell{flex:0 0 calc(100% / var(--v));min-width:0;box-sizing:border-box;padding-right:${GAP}px}
.pn-slide{display:block;height:100%;box-sizing:border-box;text-decoration:none;color:var(--text);background:var(--surface);border:1px solid var(--border);border-left:4px solid var(--pn-accent,var(--accent));border-radius:10px;padding:10px 14px;min-height:76px}
.pn-top{display:flex;align-items:center;gap:8px;margin-bottom:3px}
.pn-title{font-weight:700;font-size:14px;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.pn-badge{margin-left:auto;flex:none;font-size:11px;font-weight:700;padding:1px 8px;border-radius:999px;border:1px solid}
.pn-sub{font-size:13px;color:var(--text-dim);line-height:1.5;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
@media (prefers-reduced-motion:reduce){.pn-track.anim{transition:none}}
@media (max-width:767px){.pn{padding:0 12px}}
`;

/** ホーム上部の『速報』: PC は半幅のカードを 2 枚並べ、スマホは 1 枚で、右から左へ 1 枚ずつ流す。最後のカードのあとも止まらず、先頭へ切れ目なく流れ続ける（先頭ぶんのコピーを末尾に足し、流れ終えた瞬間に動きなしで巻き戻す）。
 *  自動で 5 秒ごとに切り替え、マウスを乗せている間・フォーカス中・動きを減らす設定のときは止める。矢印・点・スワイプで手動でも切り替えられる。
 *  回収率（仮想購入）は会員向け。未ログインの間は回収率を除いた文言を出す。集計が1件も無ければ何も出さない。 */
export function PredictionNews() {
  const { isMember, loading } = useAuthStatus();
  const [items, setItems] = useState<NewsItem[] | null>(null);
  const [idx, setIdx] = useState(0);          // いちばん左に見えているカード（0..n。n はコピーの先頭＝実質 0）
  const [anim, setAnim] = useState(true);
  const [v, setV] = useState(1);              // 同時に見えるカード数（PC 2・スマホ 1）
  const [paused, setPaused] = useState(false);
  const [reduced, setReduced] = useState(false);
  const touchX = useRef<number | null>(null);

  useEffect(() => {
    let alive = true;
    getJson<{ items: NewsItem[] }>("/api/prediction-results/news?limit=6", { ttlMs: 30_000 })
      .then((d) => { if (alive) setItems(d.items ?? []); })
      .catch(() => { if (alive) setItems([]); });
    return () => { alive = false; };
  }, []);

  useEffect(() => {
    const mq = window.matchMedia?.("(min-width: 768px)");
    const upd = () => setV(mq?.matches ? 2 : 1);
    upd();
    mq?.addEventListener?.("change", upd);
    setReduced(!!window.matchMedia?.("(prefers-reduced-motion: reduce)").matches);
    return () => mq?.removeEventListener?.("change", upd);
  }, []);

  const n = items?.length ?? 0;
  const slide = n > v;                           // 見える枚数より多いときだけ流す
  const after2Frames = (f: () => void) => requestAnimationFrame(() => requestAnimationFrame(f));

  const next = useCallback(() => { if (slide) { setAnim(true); setIdx((i) => Math.min(i + 1, n)); } }, [slide, n]);
  const prev = useCallback(() => {
    if (!slide) return;
    if (idx > 0) { setAnim(true); setIdx(idx - 1); return; }
    setAnim(false); setIdx(n);                   // 先頭から戻るときは、末尾のコピーへ動きなしで移ってから 1 枚戻す
    after2Frames(() => { setAnim(true); setIdx(n - 1); });
  }, [slide, idx, n]);
  const goTo = (i: number) => { setAnim(true); setIdx(i); };

  useEffect(() => {
    if (!slide || paused || reduced) return;
    const t = setInterval(next, INTERVAL_MS);
    return () => clearInterval(t);
  }, [slide, paused, reduced, next]);

  // v が変わる・件数が変わったときは位置をそろえる
  useEffect(() => { setAnim(false); setIdx(0); after2Frames(() => setAnim(true)); }, [v, n]);

  if (!items || n === 0) return null;
  const cells = slide ? [...items, ...items.slice(0, v)] : items;
  const active = idx % n;
  const onEnd = (e: React.TransitionEvent<HTMLDivElement>) => {
    if (e.target !== e.currentTarget || e.propertyName !== "transform" || idx < n) return;
    setAnim(false); setIdx(idx - n);             // コピーまで流れ切ったら、動きなしで先頭へ戻す（見た目は連続）
    after2Frames(() => setAnim(true));
  };
  const visible = (i: number) => i >= idx && i < idx + v;

  return (
    <section aria-label="速報" aria-roledescription="carousel" className="pn" style={{ ["--v" as string]: slide ? v : Math.max(Math.min(n, v), 1) }}>
      <style>{CSS}</style>
      <div className="pn-head">
        <h2>📰 速報</h2>
        {slide && (
          <div className="pn-ctrl">
            <button type="button" className="pn-btn" aria-label="前の速報" onClick={prev}>‹</button>
            {items.map((_, i) => (
              <button key={i} type="button" className="pn-dot" aria-label={`${i + 1} 件目`} aria-current={i === active} onClick={() => goTo(i)} />
            ))}
            <button type="button" className="pn-btn" aria-label="次の速報" onClick={next}>›</button>
          </div>
        )}
        <PendingLink href="/prediction-results" className="pn-link">予測成績をすべて見る →</PendingLink>
      </div>
      <div
        className="pn-viewport"
        onMouseEnter={() => setPaused(true)}
        onMouseLeave={() => setPaused(false)}
        onFocus={() => setPaused(true)}
        onBlur={() => setPaused(false)}
        onTouchStart={(e) => { touchX.current = e.touches[0].clientX; setPaused(true); }}
        onTouchEnd={(e) => {
          const x0 = touchX.current;
          touchX.current = null;
          if (x0 != null && Math.abs(e.changedTouches[0].clientX - x0) > 40) (e.changedTouches[0].clientX < x0 ? next : prev)();
          setTimeout(() => setPaused(false), 4000);
        }}
      >
        <div className={`pn-track${anim ? " anim" : ""}`} style={{ transform: `translateX(-${slide ? (idx * 100) / v : 0}%)` }} onTransitionEnd={onEnd} aria-live={paused ? "polite" : "off"}>
          {cells.map((it, i) => {
            const meta = KIND_META[it.kind] ?? KIND_META.day;
            const text = !loading && isMember ? it.subtitle : it.subtitle_public;
            const ok = it.badge === "確定";
            const show = !slide || (visible(i) && i < idx + v);
            return (
              <div className="pn-cell" key={`${it.kind}-${i}`}>
                <PendingLink
                  href={it.link}
                  className="pn-slide"
                  style={{ ["--pn-accent" as string]: meta.accent }}
                  aria-hidden={!show}
                  tabIndex={show ? 0 : -1}
                >
                  <div className="pn-top">
                    <span aria-hidden>{meta.icon}</span>
                    <span className="pn-title">{it.title}</span>
                    <span className="pn-badge" style={{ color: ok ? "#4ade80" : "#fbbf24", borderColor: ok ? "rgba(74,222,128,0.5)" : "rgba(251,191,36,0.5)" }}>{it.badge}</span>
                  </div>
                  <div className="pn-sub">{text}</div>
                </PendingLink>
              </div>
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
