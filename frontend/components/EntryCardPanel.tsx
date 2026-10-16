"use client";

import { ReactNode, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { CardStat, CondStat, EntityCard, EntryCard, HorseCard, PastRun, cachedEntryCard, fetchEntryCard } from "@/lib/entryCard";
import { ViewportMode, useViewportMode } from "@/lib/useViewportMode";

/* 出馬表の馬名・騎手・調教師をタップしたときのパネル。
   PC: 右のドロワー（情報は多め・近5走は表）。スマホ: 下から出るボトムシート（要約を最上部・近5走はカード・残りは折りたたみ）。 */

export type CardTab = "horse" | "jockey" | "trainer";
export type PanelHorse = { horse_id: string; horse_number?: number; horse_name?: string | null };

const TABS: { id: CardTab; label: string }[] = [{ id: "horse", label: "馬" }, { id: "jockey", label: "騎手" }, { id: "trainer", label: "調教師" }];
const NODATA = <span style={{ color: "var(--text-dim)" }}>データなし</span>;

/* ── 小さな部品 ── */
const pct = (v: number | null | undefined) => (v == null ? "-" : `${v.toFixed(1)}%`);
const md = (ymd: string | undefined | null) => (ymd && ymd.length === 8 ? `${Number(ymd.slice(4, 6))}/${Number(ymd.slice(6, 8))}` : "-");

function Chip({ label, value, tone = "neutral" }: { label: string; value: ReactNode; tone?: "good" | "warn" | "neutral" }) {
  const c = tone === "good" ? "#4ade80" : tone === "warn" ? "#fbbf24" : "var(--text)";
  return (
    <div className="ec-chip">
      <div className="ec-chip-l">{label}</div>
      <div className="ec-chip-v" style={{ color: c }}>{value}</div>
    </div>
  );
}

function Section({ title, children, collapsible, defaultOpen = true }: { title: string; children: ReactNode; collapsible?: boolean; defaultOpen?: boolean }) {
  if (collapsible) {
    return (
      <details className="ec-sec" open={defaultOpen}>
        <summary className="ec-sec-t ec-sum">{title}</summary>
        <div className="ec-sec-b">{children}</div>
      </details>
    );
  }
  return (
    <section className="ec-sec">
      <h3 className="ec-sec-t">{title}</h3>
      <div className="ec-sec-b">{children}</div>
    </section>
  );
}

function finishStyle(f: number | string | null | undefined): React.CSSProperties {
  const n = Number(f);
  if (n === 1) return { color: "#fbbf24", fontWeight: 800 };
  if (n === 2 || n === 3) return { color: "#60a5fa", fontWeight: 700 };
  return { color: "var(--text-dim)" };
}

/** 勝率・複勝率と母数。少数（10 走未満）は『少数』を併記して誤解させない */
function StatLine({ s, label }: { s: CardStat & { has_data?: boolean }; label: string }) {
  const has = s.starts != null && s.has_data !== false;
  const few = has && (s.starts ?? 0) < 10;
  return (
    <div className="ec-stat">
      <div className="ec-stat-l">{label}</div>
      {has ? (
        <>
          <div className="ec-stat-n">{s.starts}走<span className="ec-dim"> {s.wins}-{(s.top3 ?? 0) - (s.wins ?? 0)}-{(s.starts ?? 0) - (s.top3 ?? 0)}</span>{few && <span className="ec-few">少数</span>}</div>
          <div className="ec-stat-r">
            <span>勝率 <b>{pct(s.win_rate)}</b></span>
            <span>複勝率 <b>{pct(s.top3_rate)}</b></span>
          </div>
          <div className="ec-bar" aria-hidden><div style={{ width: `${Math.min(100, (s.top3_rate ?? 0) / 0.6)}%` }} /></div>
        </>
      ) : (
        <div className="ec-stat-n">{NODATA}</div>
      )}
    </div>
  );
}

/* ── 馬 ── */
function RunsTable({ runs }: { runs: PastRun[] }) {
  return (
    <div className="ec-scroll-x">
      <table className="ec-table">
        <thead>
          <tr><th>日付</th><th>場・レース</th><th>距離</th><th>着</th><th>人気</th><th>上がり3F</th><th>通過</th><th>騎手</th><th>馬体重</th></tr>
        </thead>
        <tbody>
          {runs.map((r, i) => (
            <tr key={`${r.date}-${i}`}>
              <td>{md(r.date)}</td>
              <td className="ec-l">{r.venue ?? "-"} {r.race_name ?? ""}</td>
              <td>{r.surface ?? ""}{r.distance ?? "-"}{r.going ? ` ${r.going}` : ""}</td>
              <td style={finishStyle(r.finish)}>{r.finish ?? "-"}</td>
              <td>{r.popularity ?? "-"}</td>
              <td>{r.last3f ?? "-"}</td>
              <td>{r.passing ?? "-"}</td>
              <td className="ec-l">{r.jockey ?? "-"}</td>
              <td>{r.weight ? `${r.weight}${r.weight_change != null ? `(${r.weight_change > 0 ? "+" : r.weight_change < 0 ? "" : "±"}${r.weight_change})` : ""}` : "-"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function RunCards({ runs }: { runs: PastRun[] }) {
  return (
    <ol className="ec-runs">
      {runs.map((r, i) => (
        <li key={`${r.date}-${i}`} className="ec-run">
          <div className="ec-run-top">
            <span className="ec-run-date">{md(r.date)}</span>
            <span className="ec-run-name">{r.venue ?? ""} {r.race_name ?? ""}</span>
            <span className="ec-run-fin" style={finishStyle(r.finish)}>{r.finish ?? "-"}<small>着</small></span>
          </div>
          <div className="ec-run-sub">
            {r.surface ?? ""}{r.distance ?? "-"}{r.going ? ` ${r.going}` : ""} ・ {r.field ? `${r.field}頭` : "-"} {r.post ? `${r.post}番` : ""} {r.popularity ? `${r.popularity}人気` : ""}
          </div>
          <div className="ec-run-sub">
            上がり {r.last3f ?? "-"} ・ 通過 {r.passing ?? "-"} ・ {r.jockey ?? "-"}{r.weight ? ` ・ ${r.weight}${r.weight_change != null ? `(${r.weight_change > 0 ? "+" : ""}${r.weight_change})` : ""}kg` : ""}
          </div>
        </li>
      ))}
    </ol>
  );
}

function TrainingBlock({ h }: { h: HorseCard }) {
  const t = h.training;
  if (!t) return <>{NODATA}</>;
  const tone = t.evaluation === "A" ? "#4ade80" : t.evaluation === "B" ? "#60a5fa" : "var(--text-dim)";
  return (
    <div className="ec-kv">
      {t.training_date && <div><span className="ec-k">日付</span>{t.training_date}</div>}
      <div><span className="ec-k">評価</span><b style={{ color: tone }}>{t.evaluation ?? "-"}</b>{t.impression ? <span className="ec-dim"> ・ {t.impression}</span> : null}</div>
      {(t.course || t.condition || t.rider) && <div><span className="ec-k">馬場・騎乗</span>{[t.course, t.condition, t.rider].filter(Boolean).join(" ・ ")}</div>}
      {t.comment && <div><span className="ec-k">コメント</span>{t.comment}</div>}
      {t.lap_times && <div className="ec-mono"><span className="ec-k">ラップ</span>{t.lap_times}</div>}
    </div>
  );
}

function HorseView({ h, mode }: { h: HorseCard; mode: ViewportMode }) {
  const mobile = mode === "mobile";
  const sc = h.same_conditions;
  const surf = sc.find((s) => s.label.startsWith("この馬場"));
  const ch = h.changes;
  const jockeyTone = ch.jockey_changed === true ? "warn" : "neutral";
  return (
    <>
      <div className="ec-chips">
        <Chip label="この馬場の複勝率" value={surf && surf.starts ? `${pct(surf.top3_rate)}（${surf.starts}走）` : "データなし"} tone={surf && (surf.top3_rate ?? 0) >= 40 ? "good" : "neutral"} />
        <Chip label="前走から" value={ch.layoff_days != null ? `${ch.layoff_days}日${ch.distance_change ? `・${ch.distance_change > 0 ? "+" : ""}${ch.distance_change}m` : ""}` : "データなし"} />
        <Chip label="騎手" value={ch.jockey_changed == null ? "不明" : ch.jockey_changed ? `乗り替わり${ch.previous_jockey ? `（前走 ${ch.previous_jockey}）` : ""}` : "継続騎乗"} tone={jockeyTone} />
        <Chip label="最終追い切り" value={h.training ? `${h.training.evaluation ?? "-"}${h.training.training_date ? `（${h.training.training_date.replace(/^\d+\//, "")}）` : ""}` : "データなし"} tone={h.training?.evaluation === "A" ? "good" : "neutral"} />
      </div>

      <Section title={`近${h.recent.length || 5}走`}>
        {h.recent.length === 0 ? NODATA : mobile ? <RunCards runs={h.recent} /> : <RunsTable runs={h.recent} />}
      </Section>

      <Section title="今回と同じ条件での成績（過去走）">
        {sc.length === 0 ? NODATA : <div className="ec-stats">{sc.map((s) => <StatLine key={s.label} s={s} label={s.label} />)}</div>}
      </Section>

      {h.trainer_comment && (
        <Section title="厩舎コメント（レース前）">
          <p className="ec-p">{h.trainer_comment.comment}</p>
        </Section>
      )}

      <Section title="最終追い切り" collapsible={mobile} defaultOpen={!mobile}>
        <TrainingBlock h={h} />
      </Section>

      <Section title="血統・プロフィール" collapsible={mobile} defaultOpen={!mobile}>
        {!h.profile ? NODATA : (
          <div className="ec-kv">
            <div><span className="ec-k">父</span>{h.profile.sire ?? "-"}</div>
            <div><span className="ec-k">母</span>{h.profile.dam ?? "-"}（母父 {h.profile.dam_sire ?? "-"}）</div>
            <div><span className="ec-k">通算</span>{h.profile.career ?? "-"}</div>
            <div><span className="ec-k">馬主</span>{h.profile.owner ?? "-"}</div>
          </div>
        )}
      </Section>

      {h.weights.length > 0 && (
        <Section title="馬体重の推移" collapsible={mobile} defaultOpen={false}>
          <div className="ec-weights">
            {[...h.weights].reverse().map((w) => <span key={w.date}>{md(w.date)} <b>{w.weight}</b>{w.change != null ? `(${w.change > 0 ? "+" : ""}${w.change})` : ""}</span>)}
          </div>
        </Section>
      )}
    </>
  );
}

/* ── 騎手・調教師 ── */
function EntityView({ e, kind, mode, horseName }: { e: EntityCard; kind: "jockey" | "trainer"; mode: ViewportMode; horseName?: string | null }) {
  const mobile = mode === "mobile";
  const jk = kind === "jockey";
  if (e.status === "no_data" && e.conditions.every((c) => !c.has_data)) {
    return (
      <>
        <p className="ec-p">{NODATA}（この{jk ? "騎手" : "調教師"}の集計がまだありません）</p>
        {e.profile && <Profile e={e} />}
        {e.comment && <Section title="この馬の厩舎コメント"><p className="ec-p">{e.comment.comment}</p></Section>}
      </>
    );
  }
  const wh = e.with_horse;
  const r90 = e.recent["cal90"], r365 = e.recent["cal365"], r10 = e.recent["roll10"];
  return (
    <>
      <div className="ec-chips">
        {jk && <Chip label={`${horseName ?? "この馬"}との関係`} value={wh ? (wh.changed === true ? `乗り替わり${wh.previous_jockey ? `（前走 ${wh.previous_jockey}）` : ""}` : wh.changed === false ? "継続騎乗" : "不明") : "データなし"} tone={wh?.changed === true ? "warn" : "neutral"} />}
        {jk && <Chip label="この馬への騎乗" value={wh ? `近${wh.checked}走で${wh.rides_in_last5}回${wh.rides_in_last5 ? `（${wh.wins}勝・複勝${wh.top3}）` : ""}` : "データなし"} />}
        {r90 && <Chip label="直近90日" value={r90.starts ? `${r90.wins}勝/${r90.starts}（複勝率 ${pct(r90.top3_rate)}）` : "データなし"} />}
        {e.conditions[0] && <Chip label={e.conditions[0].label} value={e.conditions[0].has_data ? `複勝率 ${pct(e.conditions[0].top3_rate)}（${e.conditions[0].starts}走）` : "データなし"} tone={(e.conditions[0].top3_rate ?? 0) >= 35 ? "good" : "neutral"} />}
      </div>

      <Section title="今回のレース条件での成績（2020年以降）">
        <div className="ec-stats">{e.conditions.map((c: CondStat) => <StatLine key={c.label} s={c} label={c.label} />)}</div>
      </Section>

      <Section title="最近の調子">
        <div className="ec-stats">
          {[r90, r365, r10].filter(Boolean).map((r) => <StatLine key={r.label} s={r} label={r.label} />)}
          {e.career && <StatLine s={e.career} label={e.career.label} />}
        </div>
      </Section>

      {!jk && e.comment && <Section title="この馬の厩舎コメント（レース前）"><p className="ec-p">{e.comment.comment}</p></Section>}

      {e.profile && <Section title="プロフィール" collapsible={mobile} defaultOpen={!mobile}><Profile e={e} /></Section>}

      <p className="ec-note">
        集計: {e.as_of ?? "-"} 時点の通算（キャッシュ）。{e.includes_after_race ? "このレース当日より後の結果を含みます。" : ""}母数が小さい値は参考程度に。
      </p>
    </>
  );
}

function Profile({ e }: { e: EntityCard }) {
  const p = e.profile;
  if (!p) return null;
  const c = p.career;
  return (
    <div className="ec-kv">
      {p.affiliation && <div><span className="ec-k">所属</span>{p.affiliation}{p.debut_year ? `（${p.debut_year}）` : ""}</div>}
      {c && <div><span className="ec-k">通算（中央）</span>{c.wins}勝/{c.rides}（勝率 {pct(c.win_rate)}・連対率 {pct(c.top2_rate)}・複勝率 {pct(c.top3_rate)}）</div>}
      {p.this_year_text && <div><span className="ec-k">今年</span>{p.this_year_text}</div>}
      {p.g1_wins_text && <div><span className="ec-k">G1</span>{p.g1_wins_text}</div>}
      {p.graded_wins_text && <div><span className="ec-k">重賞</span>{p.graded_wins_text}</div>}
    </div>
  );
}

/* ── パネル本体 ── */
const CSS = `
.ec-root{position:fixed;inset:0;z-index:1000;background:rgba(2,6,23,.55)}
.ec-drawer{position:fixed;top:0;right:0;bottom:0;width:min(480px,92vw);background:var(--surface);border-left:1px solid var(--border);box-shadow:-12px 0 32px rgba(0,0,0,.45);display:flex;flex-direction:column;animation:ecIn .16s ease-out}
.ec-sheet{position:fixed;left:0;right:0;bottom:0;max-height:85vh;background:var(--surface);border-top:1px solid var(--border);border-radius:16px 16px 0 0;box-shadow:0 -12px 32px rgba(0,0,0,.5);display:flex;flex-direction:column;animation:ecUp .2s ease-out;touch-action:pan-y}
@keyframes ecIn{from{transform:translateX(24px);opacity:.4}to{transform:none;opacity:1}}
@keyframes ecUp{from{transform:translateY(40px);opacity:.4}to{transform:none;opacity:1}}
.ec-grab{width:44px;height:5px;border-radius:3px;background:var(--border);margin:8px auto 2px}
.ec-head{display:flex;align-items:center;gap:8px;padding:8px 12px 6px;border-bottom:1px solid var(--border)}
.ec-title{flex:1;min-width:0;font-weight:800;font-size:16px;color:#fff;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ec-title small{font-weight:500;color:var(--text-dim);font-size:11px;margin-left:6px}
.ec-btn{background:var(--surface2);border:1px solid var(--border);color:var(--text);border-radius:8px;min-width:36px;min-height:36px;font-size:15px;cursor:pointer}
.ec-btn:disabled{opacity:.35;cursor:default}
.ec-sheet .ec-btn{min-width:44px;min-height:44px}
.ec-tabs{display:flex;gap:6px;padding:8px 12px 0}
.ec-tab{flex:1;background:transparent;border:0;border-bottom:3px solid transparent;color:var(--text-dim);font-size:14px;font-weight:600;padding:10px 6px;min-height:44px;cursor:pointer}
.ec-tab[aria-selected=true]{color:var(--accent);border-bottom-color:var(--accent)}
.ec-body{overflow-y:auto;padding:10px 12px 20px;-webkit-overflow-scrolling:touch;overscroll-behavior:contain}
.ec-chips{display:grid;grid-template-columns:repeat(2,1fr);gap:8px;margin-bottom:8px}
.ec-drawer .ec-chips{grid-template-columns:repeat(2,1fr)}
.ec-chip{background:var(--surface2);border:1px solid var(--border);border-radius:10px;padding:8px 10px;min-width:0}
.ec-chip-l{font-size:10.5px;color:var(--text-dim)}
.ec-chip-v{font-size:13px;font-weight:700;margin-top:2px;overflow-wrap:anywhere}
.ec-sec{margin:12px 0}
.ec-sec-t{margin:0 0 6px;font-size:12.5px;font-weight:700;color:var(--text-dim);letter-spacing:.02em}
.ec-sum{cursor:pointer;min-height:44px;display:flex;align-items:center;list-style:none}
.ec-sum::before{content:"▸";margin-right:6px;transition:transform .12s}
details[open]>.ec-sum::before{transform:rotate(90deg)}
.ec-scroll-x{overflow-x:auto}
.ec-table{width:100%;border-collapse:collapse;font-size:12px}
.ec-table th{font-size:10.5px;color:var(--text-dim);font-weight:600;padding:5px 6px;border-bottom:1px solid var(--border);white-space:nowrap}
.ec-table td{padding:6px;border-bottom:1px solid rgba(36,48,73,.4);text-align:center;white-space:nowrap}
.ec-table .ec-l{text-align:left}
.ec-runs{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:8px}
.ec-run{background:var(--surface2);border:1px solid var(--border);border-radius:10px;padding:9px 11px}
.ec-run-top{display:flex;align-items:baseline;gap:8px}
.ec-run-date{font-size:12px;color:var(--text-dim);min-width:34px}
.ec-run-name{flex:1;min-width:0;font-weight:700;font-size:13.5px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.ec-run-fin{font-size:20px;font-weight:800;line-height:1}
.ec-run-fin small{font-size:10px;margin-left:1px}
.ec-run-sub{font-size:12px;color:var(--text-dim);margin-top:3px}
.ec-stats{display:grid;grid-template-columns:1fr;gap:8px}
.ec-drawer .ec-stats{grid-template-columns:1fr 1fr}
.ec-stat{background:var(--surface2);border:1px solid var(--border);border-radius:10px;padding:8px 10px}
.ec-stat-l{font-size:11px;color:var(--text-dim)}
.ec-stat-n{font-size:14px;font-weight:700;margin-top:2px}
.ec-stat-r{display:flex;gap:12px;font-size:12px;color:var(--text-dim);margin-top:2px}
.ec-stat-r b{color:var(--text)}
.ec-bar{height:4px;background:var(--border);border-radius:2px;margin-top:6px;overflow:hidden}
.ec-bar>div{height:100%;background:var(--accent)}
.ec-few{margin-left:6px;font-size:10px;color:#fbbf24;border:1px solid rgba(251,191,36,.4);border-radius:4px;padding:0 5px;font-weight:500}
.ec-dim{color:var(--text-dim);font-weight:400}
.ec-kv{display:flex;flex-direction:column;gap:5px;font-size:13px}
.ec-k{display:inline-block;min-width:78px;color:var(--text-dim);font-size:11.5px}
.ec-mono{font-family:ui-monospace,monospace;font-size:11.5px;overflow-wrap:anywhere}
.ec-p{margin:0;font-size:13px;line-height:1.65}
.ec-weights{display:flex;flex-wrap:wrap;gap:10px;font-size:12.5px}
.ec-note{margin:14px 0 0;font-size:11px;color:var(--text-dim);line-height:1.6}
.ec-skel{height:56px;border-radius:10px;background:linear-gradient(90deg,var(--surface2),var(--border),var(--surface2));background-size:200% 100%;animation:ecSk 1.1s linear infinite;margin-bottom:8px}
@keyframes ecSk{from{background-position:200% 0}to{background-position:-200% 0}}
@media (prefers-reduced-motion:reduce){.ec-drawer,.ec-sheet,.ec-skel{animation:none}}
`;

type Props = {
  raceId: string; horses: PanelHorse[]; horseId: string; tab: CardTab;
  onTab: (t: CardTab) => void; onHorse: (horseId: string) => void; onClose: () => void;
};

export function EntryCardPanel({ raceId, horses, horseId, tab, onTab, onHorse, onClose }: Props) {
  const mode = useViewportMode();
  const [card, setCard] = useState<EntryCard | null>(() => cachedEntryCard(raceId, horseId) ?? null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(!card);
  const [dragY, setDragY] = useState(0);
  const panelRef = useRef<HTMLElement | null>(null);
  const closeRef = useRef<HTMLButtonElement | null>(null);
  const startY = useRef<number | null>(null);
  const opener = useRef<Element | null>(null);

  const load = useCallback(() => {
    const hit = cachedEntryCard(raceId, horseId);
    setError("");
    if (hit) { setCard(hit); setLoading(false); return; }
    setLoading(true);
    let alive = true;
    fetchEntryCard(raceId, horseId)
      .then((c) => alive && setCard(c))
      .catch((e: unknown) => alive && setError(e instanceof Error ? e.message : String(e)))
      .finally(() => alive && setLoading(false));
    return () => { alive = false; };
  }, [raceId, horseId]);
  useEffect(() => load(), [load]);

  // 開いたら閉じるボタンにフォーカス、閉じたら元のセルに戻す。スマホは背景のスクロールを止める
  useEffect(() => {
    opener.current = document.activeElement;
    closeRef.current?.focus();
    const prev = document.body.style.overflow;
    if (mode === "mobile") document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = prev;
      if (opener.current instanceof HTMLElement) opener.current.focus();
    };
  }, [mode]);

  const idx = Math.max(0, horses.findIndex((h) => h.horse_id === horseId));
  const cur = horses[idx];
  const prev = horses[idx - 1], next = horses[idx + 1];

  function onKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Escape") { e.stopPropagation(); onClose(); return; }
    if (e.key === "Tab" && panelRef.current) {                       // フォーカスをパネル内に留める
      const els = [...panelRef.current.querySelectorAll<HTMLElement>("button:not([disabled]), summary, [tabindex='0']")];
      if (!els.length) return;
      const first = els[0], last = els[els.length - 1];
      if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
      else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
    }
  }
  function onTabKey(e: React.KeyboardEvent) {
    const i = TABS.findIndex((t) => t.id === tab);
    if (e.key === "ArrowRight") { e.preventDefault(); onTab(TABS[(i + 1) % TABS.length].id); }
    if (e.key === "ArrowLeft") { e.preventDefault(); onTab(TABS[(i + TABS.length - 1) % TABS.length].id); }
  }
  // 下スワイプで閉じる（スマホ。つまみ・見出しの部分）
  const swipe = mode === "mobile" ? {
    onTouchStart: (e: React.TouchEvent) => { startY.current = e.touches[0].clientY; },
    onTouchMove: (e: React.TouchEvent) => { if (startY.current != null) setDragY(Math.max(0, e.touches[0].clientY - startY.current)); },
    onTouchEnd: () => { if (dragY > 90) onClose(); setDragY(0); startY.current = null; },
  } : {};

  const title = cur ? `${cur.horse_number ?? ""} ${cur.horse_name ?? card?.horse?.name ?? ""}`.trim() : (card?.horse?.name ?? "");
  const body = useMemo(() => {
    if (loading) return <div aria-busy="true" aria-label="読み込み中">{[0, 1, 2, 3].map((i) => <div key={i} className="ec-skel" />)}</div>;
    if (error || !card) return (
      <div role="alert" className="ec-p">
        取得できませんでした（{error || "不明なエラー"}）。
        <div><button type="button" className="ec-btn" style={{ marginTop: 8, padding: "0 14px" }} onClick={() => load()}>再読み込み</button></div>
      </div>
    );
    if (tab === "horse") return <HorseView h={card.horse} mode={mode} />;
    return <EntityView e={tab === "jockey" ? card.jockey : card.trainer} kind={tab} mode={mode} horseName={card.horse?.name} />;
  }, [loading, error, card, tab, mode, load]);

  if (typeof document === "undefined") return null;
  return createPortal(
    <div className="ec-root" onMouseDown={(e) => { if (e.target === e.currentTarget) onClose(); }} onTouchStart={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <style>{CSS}</style>
      <aside
        ref={panelRef as React.RefObject<HTMLElement>}
        role="dialog" aria-modal="true" aria-label={`${title} の情報`}
        className={mode === "mobile" ? "ec-sheet" : "ec-drawer"}
        style={mode === "mobile" && dragY ? { transform: `translateY(${dragY}px)`, transition: "none" } : undefined}
        onKeyDown={onKeyDown}
      >
        <div {...swipe}>
          {mode === "mobile" && <div className="ec-grab" aria-hidden />}
          <div className="ec-head">
            <button type="button" className="ec-btn" aria-label="前の馬" disabled={!prev} onClick={() => prev && onHorse(prev.horse_id)}>‹</button>
            <div className="ec-title">{title}<small>{card?.horse?.sex_age ?? ""}{card?.race?.name ? ` ・ ${card.race.name}` : ""}</small></div>
            <button type="button" className="ec-btn" aria-label="次の馬" disabled={!next} onClick={() => next && onHorse(next.horse_id)}>›</button>
            <button type="button" className="ec-btn" ref={closeRef} aria-label="閉じる" onClick={onClose}>✕</button>
          </div>
        </div>
        <div className="ec-tabs" role="tablist" aria-label="表示する情報" onKeyDown={onTabKey}>
          {TABS.map((t) => (
            <button key={t.id} type="button" role="tab" id={`ec-tab-${t.id}`} aria-selected={tab === t.id} aria-controls="ec-body" tabIndex={tab === t.id ? 0 : -1} className="ec-tab" onClick={() => onTab(t.id)}>
              {t.label}
            </button>
          ))}
        </div>
        <div className="ec-body" id="ec-body" role="tabpanel" aria-labelledby={`ec-tab-${tab}`}>{body}</div>
      </aside>
    </div>,
    document.body,
  );
}

