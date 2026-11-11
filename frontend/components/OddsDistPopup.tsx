"use client";

import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { binStats, pct, type TPRow } from "@/lib/trainerPattern";

const RED = "#ef4444";
const YELLOW = "#facc15";
const BLUE = "#60a5fa";

const CSS = `
.odp{position:fixed;z-index:60;background:var(--surface);color:var(--text);border:1px solid var(--border);border-radius:8px;padding:8px 10px;font-size:12px;line-height:1.5;box-shadow:0 4px 14px rgba(0,0,0,.5);max-width:calc(100vw - 8px);pointer-events:none}
.odp .h{font-weight:700;margin-bottom:4px;white-space:normal}
.odp .lg{color:var(--text-dim);margin-bottom:3px;font-size:11px}
.odp .lg i{display:inline-block;width:10px;height:8px;margin:0 3px 0 6px;border-radius:2px}
.odp .r{display:flex;align-items:center;gap:6px;white-space:nowrap;font-variant-numeric:tabular-nums}
.odp .r>span:first-child{width:62px}
.odp .bs{width:120px;flex:none;display:flex;border-radius:2px;overflow:hidden}
.odp .bs i{display:block;height:10px}
.odp .n{width:92px;color:var(--text-dim);text-align:right}
.odp .c{width:56px;text-align:right;font-weight:700}
.odp .hd{color:var(--text-dim);font-size:11px}
.odp .hd>span{font-weight:400!important}
.odp .tt{border-top:1px solid var(--border);margin-top:2px;padding-top:2px}
.odp .w{color:${RED}}
.odp .y{color:${YELLOW}}
.odp .ro{color:var(--text-dim)}
.odp .ro.up{color:#4ade80}
@media(max-width:560px){.odp{font-size:11px;padding:6px 7px}.odp .bs{display:none}.odp .r{gap:3px}.odp .r>span:first-child{width:50px}.odp .n{width:74px}.odp .c{width:50px}.odp .hd .c{font-size:10px;letter-spacing:-.5px}}
`;

function Body({ row, edges, maxOdds, title }: { row: TPRow; edges: number[]; maxOdds: number; title: string }) {
  const { bins, total } = binStats(row, edges, maxOdds);
  const m = Math.max(...bins.map((b) => b.n), 1);
  const seg = (c: number, col: string) => (c > 0 ? <i style={{ width: (c / m) * 120, background: col }} /> : null);
  const roi = (v: number | null) => <span className={"c ro" + (v != null && v > 1 ? " up" : "")}>{pct(v)}</span>;
  return (
    <>
      <div className="h">{title} のオッズ分布（{total.n}走・3着以内 {total.t3}・1着 {total.w}）</div>
      <div className="lg">
        棒の長さ＝区間の出走数。<i style={{ background: RED }} />1着 <i style={{ background: YELLOW }} />2・3着 <i style={{ background: BLUE }} />4着以下。
        右は区間ごとの率と回収率（100% 超は緑）
      </div>
      <div className="r hd">
        <span>オッズ</span><span className="bs" /><span className="n">割合（走数）</span>
        <span className="c">勝率</span><span className="c">複勝率</span><span className="c">単勝回収率</span><span className="c">複勝回収率</span>
      </div>
      {bins.map((b) => (
        <div className="r" key={b.label}>
          <span>{b.label}</span>
          <span className="bs">{seg(b.w, RED)}{seg(b.t3 - b.w, YELLOW)}{seg(b.n - b.t3, BLUE)}</span>
          <span className="n">{pct(b.share)}（{b.n}）</span>
          <span className="c w">{pct(b.win)}</span><span className="c y">{pct(b.place)}</span>
          {roi(b.winRoi)}{roi(b.placeRoi)}
        </div>
      ))}
      <div className="r tt">
        <span>合計</span><span className="bs" /><span className="n">{total.n}走</span>
        <span className="c w">{pct(total.win)}</span><span className="c y">{pct(total.place)}</span>
        {roi(total.winRoi)}{roi(total.placeRoi)}
      </div>
    </>
  );
}

/**
 * 区間別の出走数を小さな積み棒（赤=1着・黄=2・3着・青=4着以下）にし、カーソルを合わせる／タップすると詳細を出す。
 * 計算は行の配列だけ。通信はしない。
 */
export function OddsDist({ row, edges, maxOdds, title }: { row: TPRow; edges: number[]; maxOdds: number; title: string }) {
  const [pos, setPos] = useState<{ x: number; y: number } | null>(null);
  const [pinned, setPinned] = useState(false);
  const box = useRef<HTMLDivElement | null>(null);
  const el = useRef<HTMLSpanElement | null>(null);
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  useEffect(() => {
    if (!pinned) return;
    const off = (e: Event) => { if (!el.current?.contains(e.target as Node)) { setPinned(false); setPos(null); } };
    document.addEventListener("pointerdown", off);
    window.addEventListener("scroll", off, true);
    return () => { document.removeEventListener("pointerdown", off); window.removeEventListener("scroll", off, true); };
  }, [pinned]);

  const { bins } = binStats(row, edges, maxOdds);
  const m = Math.max(...bins.map((b) => b.n), 1);
  const BW = 8;
  const H = 20;

  const place = (cx: number, cy: number) => {
    const w = box.current?.offsetWidth ?? 420;
    const h = box.current?.offsetHeight ?? 220;
    let l = cx + 14, t = cy + 14;
    if (l + w > window.innerWidth - 8) l = cx - w - 14;
    if (t + h > window.innerHeight - 8) t = cy - h - 14;
    setPos({ x: Math.max(4, l), y: Math.max(4, t) });
  };

  return (
    <span ref={el} style={{ display: "inline-block", cursor: "help", lineHeight: 0 }}
      onPointerEnter={(e) => { if (e.pointerType === "mouse") place(e.clientX, e.clientY); }}
      onPointerMove={(e) => { if (e.pointerType === "mouse" && !pinned) place(e.clientX, e.clientY); }}
      onPointerLeave={(e) => { if (e.pointerType === "mouse" && !pinned) setPos(null); }}
      onClick={(e) => { e.stopPropagation(); const r = e.currentTarget.getBoundingClientRect(); if (pinned) { setPinned(false); setPos(null); } else { setPinned(true); place(r.left, r.bottom - 14); } }}
      role="img" aria-label={`${title} のオッズ分布`}>
      <svg width={bins.length * BW} height={H} viewBox={`0 0 ${bins.length * BW} ${H}`}>
        {bins.map((b, i) => {
          const hh = Math.max(b.n ? 2 : 0, Math.min(1, b.n / m) * (H - 2));
          const hw = b.n ? (b.w / b.n) * hh : 0, h3 = b.n ? ((b.t3 - b.w) / b.n) * hh : 0, h4 = hh - hw - h3;
          return (
            <g key={i} transform={`translate(${i * BW},0)`}>
              <rect x={0} y={H - h4} width={BW - 1} height={h4} fill={BLUE} />
              <rect x={0} y={H - h4 - h3} width={BW - 1} height={h3} fill={YELLOW} />
              <rect x={0} y={H - h4 - h3 - hw} width={BW - 1} height={hw} fill={RED} />
            </g>
          );
        })}
      </svg>
      {mounted && pos && createPortal(
        <div ref={box} className="odp" style={{ left: pos.x, top: pos.y }}>
          <style>{CSS}</style>
          <Body row={row} edges={edges} maxOdds={maxOdds} title={title} />
        </div>,
        document.body,
      )}
    </span>
  );
}
