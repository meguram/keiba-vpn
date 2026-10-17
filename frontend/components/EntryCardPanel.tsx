"use client";

import { ReactNode, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { CardStat, CondStat, EntityCard, EntryCard, HorseCard, PastRun, cachedEntryCard, fetchEntryCard } from "@/lib/entryCard";
import {
  BASE_TOP3, BASE_WIN, BRACKET, CLASS_STYLE, FEW_HORSE, FEW_PERSON, TONE, Tag, Tone,
  distanceChange, finishBadge, finishBarHeight, formTag, layoffTag, parseLap, passingList, placeTag, popGap, splitRaceName, styleOf, surfaceStyle, trainingGrade,
} from "@/lib/entryCardUi";
import { ViewportMode, useViewportMode } from "@/lib/useViewportMode";

/* 出馬表の馬名・騎手・調教師をタップしたときのパネル。
   PC: 右のドロワー（要約タイル → 近5走の表 → 条件別ゲージ）。スマホ: 下から出るボトムシート（要約タイルを最上部・近5走はカード・詳細は折りたたみ）。
   「良い/悪い」は色だけに頼らず、◎△▽ と語（得意・標準・苦手・少数）を必ず併記する。 */

export type CardTab = "horse" | "jockey" | "trainer";
export type PanelHorse = {
  horse_id: string; horse_number?: number; horse_name?: string | null;
  bracket_number?: number | null; sex_age?: string | null; jockey_name?: string | null; jockey_weight?: number | null;
  trainer_name?: string | null; weight?: number | null; weight_change?: number | null;
};

const TABS: { id: CardTab; label: string }[] = [{ id: "horse", label: "馬" }, { id: "jockey", label: "騎手" }, { id: "trainer", label: "調教師" }];

/* ── 小さな部品 ── */
const pct = (v: number | null | undefined) => (v == null ? "-" : `${v.toFixed(1)}%`);
const num1 = (v: number | null | undefined) => (v == null ? "-" : v.toFixed(1));
const md = (ymd: string | undefined | null) => (ymd && ymd.length === 8 ? `${Number(ymd.slice(4, 6))}/${Number(ymd.slice(6, 8))}` : "-");
const yr = (ymd: string | undefined | null) => (ymd && ymd.length === 8 ? ymd.slice(0, 4) : "");
const signed = (v: number) => `${v > 0 ? "+" : v < 0 ? "-" : "±"}${Math.abs(v)}`;
const GAUGE_MAX = BASE_TOP3 * 3;   // 目盛りは平均の 3 倍まで。平均の縦線はいつも左から 1/3 の位置

/** API の氏名と出馬表の表記のうち、情報の多い（長い）ほうを使う */
const pickName = (api: string | null | undefined, row: string | null | undefined) => {
  const a = (api ?? "").trim(), r = (row ?? "").trim();
  return a.length >= r.length ? a || r : r;
};

function splitLabel(l: string): { main: string; sub: string } {
  const m = l.match(/^([^（(]+)[（(](.*)[）)]$/);
  return m ? { main: m[1].trim(), sub: m[2].trim() } : { main: l, sub: "" };
}

function Empty({ text = "データなし", note, compact }: { text?: string; note?: string; compact?: boolean }) {
  return (
    <div className={`ec-empty${compact ? " c" : ""}`}>
      <b>— {text}</b>
      {note && <small>{note}</small>}
    </div>
  );
}

function TagChip({ tag }: { tag: Tag }) {
  const t = TONE[tag.tone];
  return <span className="ec-tag" style={{ color: t.fg, background: t.bg, borderColor: t.bd }}>{tag.text}</span>;
}

function Tile({ label, big, unit, word, tag, sub, tone }: { label: string; big: ReactNode; unit?: string; word?: boolean; tag?: Tag; sub?: ReactNode; tone: Tone }) {
  const t = TONE[tone];
  return (
    <div className="ec-tile" style={{ borderLeftColor: t.fg, background: `linear-gradient(135deg, ${t.bg}, var(--surface2) 70%)` }}>
      <div className="ec-tile-h"><span className="ec-tile-l">{label}</span></div>
      <div className={`ec-tile-v${word ? " w" : ""}`} style={{ color: t.fg }}>
        <span className="ec-tile-n">{big}{unit && <small>{unit}</small>}</span>
        {tag && <TagChip tag={tag} />}
      </div>
      {sub && <div className="ec-tile-s">{sub}</div>}
    </div>
  );
}

function Section({ title, hint, children, collapsible, defaultOpen = true }: { title: string; hint?: string; children: ReactNode; collapsible?: boolean; defaultOpen?: boolean }) {
  if (collapsible) {
    return (
      <details className="ec-sec" open={defaultOpen}>
        <summary className="ec-sec-t ec-sum">{title}{hint && <small>{hint}</small>}</summary>
        <div className="ec-sec-b">{children}</div>
      </details>
    );
  }
  return (
    <section className="ec-sec">
      <h3 className="ec-sec-t">{title}{hint && <small>{hint}</small>}</h3>
      <div className="ec-sec-b">{children}</div>
    </section>
  );
}

function FinishBadge({ f, size = "m" }: { f: number | string | null | undefined; size?: "m" | "l" }) {
  const b = finishBadge(f);
  return (
    <span className={`ec-fin ${size}`} role="img" aria-label={b.label} title={b.label} style={{ background: b.bg, color: b.fg, borderColor: b.bd }}>{b.text}</span>
  );
}

function ClassChip({ cls }: { cls: string }) {
  if (!cls) return null;
  const s = CLASS_STYLE[cls] ?? { bg: "rgba(148,163,184,0.18)", fg: "#cbd5e1" };
  return <span className="ec-cls" style={{ background: s.bg, color: s.fg }}>{cls}</span>;
}

function SurfaceChip({ s, children }: { s: string | null | undefined; children?: ReactNode }) {
  if (!s) return <>{children}</>;
  const st = surfaceStyle(s);
  return <span className="ec-surf" style={{ background: st.bg, color: st.fg }}>{st.text}{children}</span>;
}

/** 勝率・複勝率と母数。平均の縦線つきのゲージ。母数が少ないときは評価せず『少数』を前面に出す */
function Gauge({ s, label, few, marker, markerLabel }: { s: CardStat & { has_data?: boolean }; label: string; few: number; marker: number; markerLabel: string }) {
  const { main, sub } = splitLabel(label);
  const has = s.starts != null && s.starts > 0 && s.has_data !== false;
  if (!has) {
    return (
      <div className="ec-stat ec-g">
        <div className="ec-g-top"><span className="ec-g-l">{main}</span></div>
        {sub && <div className="ec-g-sub">{sub}</div>}
        <Empty compact text="データなし" note="この条件の出走がありません" />
      </div>
    );
  }
  const tag = placeTag(s.top3_rate, s.starts, few);
  const t = TONE[tag.tone];
  const isFew = (s.starts ?? 0) < few;
  const rate = s.top3_rate ?? 0;
  const diff = rate - marker;
  const dTone: Tone = isFew ? "muted" : diff >= 3 ? "good" : diff <= -3 ? "warn" : "ok";
  return (
    <div className="ec-stat ec-g" style={{ borderTopColor: t.bd }}>
      <div className="ec-g-top"><span className="ec-g-l">{main}</span><TagChip tag={tag} /></div>
      {sub && <div className="ec-g-sub">{sub}</div>}
      <div className="ec-g-main">
        <span className="ec-g-runs"><b>{s.starts}</b>走中</span>
        <span className="ec-g-big" style={{ color: t.fg }}>{num1(s.top3_rate)}<small>%</small></span>
        <span className="ec-g-unit">複勝率</span>
      </div>
      <div className="ec-g-bar" role="img" aria-label={`複勝率 ${pct(s.top3_rate)}、${markerLabel} ${marker}%`}>
        <i className={isFew ? "few" : ""} style={{ width: `${Math.min(100, (rate / GAUGE_MAX) * 100)}%`, background: t.fg }} />
        <u style={{ left: `${(marker / GAUGE_MAX) * 100}%` }} />
      </div>
      <div className="ec-g-foot">
        <span>勝{s.wins ?? 0}・3着内{s.top3 ?? 0}</span>
        <span style={{ color: TONE[dTone].fg }}>{markerLabel}比 {diff >= 0 ? "▲" : "▼"}{Math.abs(diff).toFixed(1)}pt</span>
      </div>
    </div>
  );
}

/* ── 馬 ── */
function Trend({ runs }: { runs: PastRun[] }) {
  const rs = [...runs].reverse();
  const nums = runs.map((r) => Number(r.finish)).filter((n) => Number.isFinite(n));
  const avg = nums.length ? nums.reduce((a, b) => a + b, 0) / nums.length : null;
  const in3 = nums.filter((n) => n <= 3).length;
  const color = (k: string, bg: string) => (k === "medal" ? bg : k === "mid" ? "#60a5fa" : k === "low" ? "#64748b" : "#fb923c");
  return (
    <div className="ec-trend">
      <div className="ec-trend-bars" role="img" aria-label={`近${rs.length}走の着順（古い順）: ${rs.map((r) => finishBadge(r.finish).label).join("、")}`}>
        {rs.map((r, i) => {
          const b = finishBadge(r.finish);
          return (
            <div className="ec-tb" key={`${r.date}-${i}`}>
              <div className="ec-tb-col">
                <span className="ec-tb-n" style={{ color: b.kind === "odd" ? "#fb923c" : "var(--text)" }}>{b.text}</span>
                <i style={{ height: `${Math.round(finishBarHeight(r.finish) * 0.5)}px`, background: color(b.kind, b.bg) }} />
              </div>
              <em>{md(r.date)}</em>
            </div>
          );
        })}
      </div>
      <div className="ec-trend-s">
        <div><small>平均着順</small><b>{avg == null ? "-" : avg.toFixed(1)}</b></div>
        <div><small>3着内</small><b>{in3}<span>/{runs.length}</span></b></div>
        <div className="ec-trend-h">左が古く、右が新しい</div>
      </div>
    </div>
  );
}

function Pass({ p, field }: { p: string | null | undefined; field: number | null | undefined }) {
  const l = passingList(p);
  if (!l.length) return <span className="ec-dim">-</span>;
  const f = field || 16;
  return (
    <span className="ec-pass" aria-label={`通過順 ${p}`}>
      {l.map((n, i) => <i key={i} style={{ background: n / f <= 0.25 ? "#1d4ed8" : n / f <= 0.6 ? "#475569" : "#334155" }}>{n}</i>)}
    </span>
  );
}

type Cur = { surface?: string | null; distance?: number | null };
const isSame = (r: PastRun, cur?: Cur) => !!cur && !!cur.distance && r.distance === cur.distance && (r.surface ?? "").slice(0, 1) === (cur.surface ?? "").slice(0, 1);

function wtText(r: PastRun) {
  return r.weight ? `${r.weight}kg${r.weight_change != null ? `(${signed(r.weight_change)})` : ""}` : "-";
}

function Last3f({ r, best }: { r: PastRun; best: number | null }) {
  if (r.last3f == null) return <span className="ec-dim">-</span>;
  const isBest = best != null && r.last3f === best;
  return <span className={`ec-l3${isBest ? " best" : ""}`} title={isBest ? "表示中の走で最速の上がり" : undefined}>{r.last3f.toFixed(1)}{isBest && <small>最速</small>}</span>;
}

function PopArrow({ r, full }: { r: PastRun; full?: boolean }) {
  if (!r.popularity) return <span className="ec-dim">-</span>;
  const g = popGap(r.popularity, r.finish);
  const d = Math.abs(r.popularity - Number(r.finish));
  return (
    <span className="ec-pop">
      <span>{r.popularity}人気</span>
      {g && <span className="ec-gap" style={{ color: TONE[g.tone].fg }} title={g.text}>{g.arrow}{g.arrow !== "＝" ? d : ""}{full && <small> {g.text}</small>}</span>}
    </span>
  );
}

function RunsTable({ runs, cur }: { runs: PastRun[]; cur?: Cur }) {
  const best = Math.min(...runs.map((r) => r.last3f ?? Infinity));
  const b = Number.isFinite(best) ? best : null;
  return (
    <div className="ec-scroll-x">
      <table className="ec-table">
        <thead>
          <tr><th>日付</th><th className="ec-l">レース・コース</th><th>着順</th><th>人気→着</th><th>上がり3F</th><th>通過</th><th className="ec-l">騎手・馬体重</th></tr>
        </thead>
        <tbody>
          {runs.map((r, i) => {
            const rn = splitRaceName(r.race_name);
            const dc = r.distance && cur?.distance ? r.distance - cur.distance : null;
            return (
              <tr key={`${r.date}-${i}`}>
                <td><b>{md(r.date)}</b><small>{yr(r.date)}</small></td>
                <td className="ec-l">
                  <div className="ec-rn"><ClassChip cls={rn.cls} /><span>{r.venue ?? ""} {rn.name}</span></div>
                  <div className="ec-rs">
                    <SurfaceChip s={r.surface} />{r.distance ? `${r.distance}m` : "-"}{r.going ? ` ${r.going}` : ""}{r.field ? ` ・${r.field}頭` : ""}
                    {isSame(r, cur) ? <em className="ec-same">同距離</em> : dc ? <em className="ec-dc">今回比{dc > 0 ? "↑" : "↓"}{Math.abs(dc)}m</em> : null}
                  </div>
                </td>
                <td><FinishBadge f={r.finish} /></td>
                <td><PopArrow r={r} /></td>
                <td><Last3f r={r} best={b} /></td>
                <td><Pass p={r.passing} field={r.field} /><small className="ec-st">{styleOf(r.passing, r.field)}</small></td>
                <td className="ec-l"><div>{r.jockey ?? "-"}</div><div className="ec-rs">{wtText(r)}</div></td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function RunCards({ runs, cur }: { runs: PastRun[]; cur?: Cur }) {
  const best = Math.min(...runs.map((r) => r.last3f ?? Infinity));
  const b = Number.isFinite(best) ? best : null;
  return (
    <ol className="ec-runs">
      {runs.map((r, i) => {
        const rn = splitRaceName(r.race_name);
        const dc = r.distance && cur?.distance ? r.distance - cur.distance : null;
        return (
          <li key={`${r.date}-${i}`} className="ec-run">
            <FinishBadge f={r.finish} size="l" />
            <div className="ec-run-b">
              <div className="ec-run-top">
                <span className="ec-run-date">{md(r.date)}</span>
                <ClassChip cls={rn.cls} />
                <span className="ec-run-name">{r.venue ?? ""} {rn.name}</span>
              </div>
              <div className="ec-run-sub">
                <SurfaceChip s={r.surface} />{r.distance ? `${r.distance}m` : "-"}{r.going ? ` ${r.going}` : ""} ・ {r.field ? `${r.field}頭` : "-"}
                {isSame(r, cur) ? <em className="ec-same">同距離</em> : dc ? <em className="ec-dc">今回比{dc > 0 ? "↑" : "↓"}{Math.abs(dc)}m</em> : null}
              </div>
              <div className="ec-run-facts">
                <PopArrow r={r} full />
                <span>上がり <Last3f r={r} best={b} /></span>
              </div>
              <div className="ec-run-sub">
                通過 <Pass p={r.passing} field={r.field} /> <small className="ec-st">{styleOf(r.passing, r.field)}</small>
                <span className="ec-run-j">{r.jockey ?? "-"} ・ {wtText(r)}</span>
              </div>
            </div>
          </li>
        );
      })}
    </ol>
  );
}

function LapChart({ lap }: { lap: string }) {
  const L = parseLap(lap);
  if (L.length < 2) return <div className="ec-mono">{lap}</div>;
  const n = L.length;
  const speeds = L.map((x) => 1 / x.lap);
  const lo = Math.min(...speeds), hi = Math.max(...speeds);
  const accel = L.every((x, i) => i === 0 || x.lap <= L[i - 1].lap);
  return (
    <div className="ec-lap">
      <div className="ec-lap-bars" role="img" aria-label={`追い切りラップ ${L.map((x, i) => `${n - i}F ${x.lap}秒`).join("、")}`}>
        {L.map((x, i) => (
          <div className="ec-lb" key={i}>
            <b>{x.lap.toFixed(1)}</b>
            <div className="ec-lb-col"><i style={{ height: `${hi === lo ? 70 : 38 + ((speeds[i] - lo) / (hi - lo)) * 62}%` }} /></div>
            <span>{n - i}F</span><small>{x.cum.toFixed(1)}</small>
          </div>
        ))}
      </div>
      <div className="ec-lap-n">
        {accel ? <b className="ec-lap-t" style={{ color: TONE.good.fg }}>◎ 加速ラップ</b> : <b className="ec-lap-t" style={{ color: TONE.muted.fg }}>△ 加速ではない</b>}
        棒が高いほど速い（上の数字は各1Fのラップ、下は累計）
      </div>
    </div>
  );
}

function Quote({ text, who, note }: { text: string; who?: string | null; note?: string | null }) {
  return (
    <figure className="ec-quote">
      <span className="ec-q" aria-hidden>“</span>
      <blockquote>{text}</blockquote>
      <figcaption>{who ? `${who}師` : "厩舎"}・レース前コメント{note ? ` ・評価 ${note}` : ""}</figcaption>
    </figure>
  );
}

function TrainingBlock({ h }: { h: HorseCard }) {
  const t = h.training;
  if (!t) return <Empty text="追い切りデータなし" />;
  const g = trainingGrade(t.evaluation);
  return (
    <div className="ec-train">
      <div className="ec-train-h">
        <span className="ec-grade" style={{ background: g.bg, color: g.fg, borderColor: g.letter === "-" ? "var(--border)" : g.bg }} aria-label={`評価 ${g.letter}`}>{g.letter}</span>
        <div>
          <div><b style={{ color: TONE[g.tone].fg }}>{g.word}</b>{t.impression ? <span className="ec-dim"> ・ {t.impression}</span> : null}</div>
          <div className="ec-rs">{[t.training_date, t.course, t.condition, t.rider].filter(Boolean).join(" ・ ") || "コース・日付の記載なし"}</div>
        </div>
      </div>
      {t.lap_times && <LapChart lap={t.lap_times} />}
      {t.comment && <p className="ec-p ec-train-c">{t.comment}</p>}
    </div>
  );
}

function HorseView({ h, mode, cur, jockeyName }: { h: HorseCard; mode: ViewportMode; cur?: Cur; jockeyName?: string | null }) {
  const mobile = mode === "mobile";
  const sc = h.same_conditions.filter((s) => (s.starts ?? 0) > 0);
  const pick = sc.length ? [...sc].sort((a, b) => (b.starts ?? 0) - (a.starts ?? 0))[0] : null;
  const ch = h.changes;
  const lay = layoffTag(ch.layoff_days);
  const dist = distanceChange(ch.distance_change);
  const g = trainingGrade(h.training?.evaluation);
  const pickTag = pick ? placeTag(pick.top3_rate, pick.starts, FEW_HORSE) : null;
  const trainerName = h.trainer_comment?.trainer;

  return (
    <>
      <div className="ec-tiles">
        {pick && pickTag ? (
          <Tile label={`${splitLabel(pick.label).main} 複勝率`} tone={pickTag.tone} tag={pickTag} big={num1(pick.top3_rate)} unit="%" sub={<><b>{pick.starts}走中</b>勝{pick.wins}・3着内{pick.top3}</>} />
        ) : (
          <Tile label="今回条件の複勝率" tone="muted" big="—" word sub="同条件の出走なし" />
        )}
        <Tile
          label="前走から" tone={lay.tone} tag={lay} big={ch.layoff_days != null ? ch.layoff_days : "—"} unit={ch.layoff_days != null ? "日" : undefined}
          sub={<><b>{dist.arrow} {dist.text}</b></>}
        />
        <Tile
          label="騎手" tone={ch.jockey_changed == null ? "muted" : ch.jockey_changed ? "warn" : "ok"} word
          big={ch.jockey_changed == null ? "不明" : ch.jockey_changed ? "▽乗り替わり" : "△継続騎乗"}
          sub={ch.jockey_changed ? <>前走 <b>{ch.previous_jockey ?? "-"}</b> → 今回 <b>{jockeyName ?? "-"}</b></> : ch.jockey_changed === false ? <><b>{jockeyName ?? ""}</b> が続けて騎乗</> : "前走の騎手が不明"}
        />
        <Tile
          label="最終追い切り" tone={g.tone} tag={h.training ? { text: g.word, tone: g.tone } : undefined}
          big={h.training ? <span className="ec-grade big" style={{ background: g.bg, color: g.fg, borderColor: g.letter === "-" ? "var(--border)" : g.bg }}>{g.letter}</span> : "—"} word={!h.training}
          sub={h.training ? [h.training.training_date, h.training.course].filter(Boolean).join(" ・ ") || "日付・コースの記載なし" : "追い切りデータなし"}
        />
      </div>

      <Section title={`近${h.recent.length || 5}走`} hint={h.recent.length ? "着順・人気・上がり" : undefined}>
        {h.recent.length === 0 ? <Empty text="近走データなし" note="初出走、またはまだ取り込まれていません" /> : (
          <>
            <Trend runs={h.recent} />
            {mobile ? <RunCards runs={h.recent} cur={cur} /> : <RunsTable runs={h.recent} cur={cur} />}
          </>
        )}
      </Section>

      <Section title="今回と同じ条件での成績" hint="この馬の過去走">
        {sc.length === 0 && h.same_conditions.length === 0 ? <Empty /> : (
          <>
            <div className="ec-stats">{h.same_conditions.map((s) => <Gauge key={s.label} s={s} label={s.label} few={FEW_HORSE} marker={BASE_TOP3} markerLabel="全体平均" />)}</div>
            <p className="ec-axis">棒の長さ＝複勝率（右端 {GAUGE_MAX}%）。縦線は全体平均 {BASE_TOP3}%。5走未満は『少数』で参考扱い。</p>
          </>
        )}
      </Section>

      {h.trainer_comment && (
        <Section title="厩舎コメント">
          <Quote text={h.trainer_comment.comment} who={trainerName} note={h.trainer_comment.evaluation} />
        </Section>
      )}

      <Section title="最終追い切り" collapsible={mobile} defaultOpen={!mobile}>
        <TrainingBlock h={h} />
      </Section>

      <Section title="血統・プロフィール" collapsible={mobile} defaultOpen={!mobile}>
        {!h.profile ? <Empty /> : (
          <div className="ec-kv">
            <div><span className="ec-k">父</span>{h.profile.sire ? <b>{h.profile.sire}</b> : <span className="ec-dim">データなし</span>}</div>
            <div><span className="ec-k">母</span>{h.profile.dam ? <b>{h.profile.dam}</b> : <span className="ec-dim">データなし</span>}{h.profile.dam_sire ? <span className="ec-dim">（母父 {h.profile.dam_sire}）</span> : null}</div>
            <div><span className="ec-k">通算</span><b>{h.profile.career ?? "-"}</b></div>
            <div><span className="ec-k">馬主</span>{h.profile.owner ?? "-"}</div>
          </div>
        )}
      </Section>

      {h.weights.length > 0 && (
        <Section title="馬体重の推移" collapsible defaultOpen={false}>
          <div className="ec-weights">
            {[...h.weights].reverse().map((w) => (
              <span key={w.date} className="ec-w">
                <small>{md(w.date)}</small><b>{w.weight}</b>
                {w.change != null && <em style={{ color: Math.abs(w.change) >= 10 ? TONE.warn.fg : "var(--ec-dim)" }}>{w.change > 0 ? "▲" : w.change < 0 ? "▼" : "＝"}{Math.abs(w.change)}</em>}
              </span>
            ))}
          </div>
        </Section>
      )}
    </>
  );
}

/* ── 騎手・調教師 ── */
function Strip({ s }: { s: CardStat }) {
  const n = s.starts ?? 0, w = s.wins ?? 0, t3 = Math.max(0, (s.top3 ?? 0) - w), o = Math.max(0, n - w - t3);
  const dots = [...Array(w).fill("w"), ...Array(t3).fill("p"), ...Array(o).fill("o")] as string[];
  return (
    <div className="ec-strip">
      <div className="ec-strip-d" role="img" aria-label={`直近${n}走: 1着${w}回、2・3着${t3}回、着外${o}回`}>
        {dots.map((d, i) => <span key={i} className={`d ${d}`}>{d === "w" ? "●" : d === "p" ? "◐" : "○"}</span>)}
      </div>
      <div className="ec-strip-l"><span><span className="d w">●</span>1着 {w}</span><span><span className="d p">◐</span>2・3着 {t3}</span><span><span className="d o">○</span>着外 {o}</span><em>（順不同）</em></div>
    </div>
  );
}

function RateBars({ c }: { c: { win_rate?: number | null; top2_rate?: number | null; top3_rate?: number | null } }) {
  const rows = [
    { l: "勝率", v: c.win_rate, base: BASE_WIN }, { l: "連対率", v: c.top2_rate, base: BASE_WIN * 2 }, { l: "複勝率", v: c.top3_rate, base: BASE_TOP3 },
  ].filter((r) => r.v != null);
  return (
    <div className="ec-rb">
      {rows.map((r) => {
        const v = r.v as number, up = v >= r.base;
        return (
          <div className="ec-rb-r" key={r.l}>
            <span className="ec-rb-l">{r.l}</span>
            <div className="ec-g-bar" role="img" aria-label={`${r.l} ${pct(v)}（平均 ${r.base}%）`}>
              <i style={{ width: `${Math.min(100, (v / (r.base * 3)) * 100)}%`, background: up ? TONE.good.fg : TONE.ok.fg }} />
              <u style={{ left: "33.3%" }} />
            </div>
            <b>{pct(v)}</b>
            <span className="ec-rb-d" style={{ color: up ? TONE.good.fg : "var(--ec-dim)" }}>{up ? "▲" : "▼"}{Math.abs(v - r.base).toFixed(1)}</span>
          </div>
        );
      })}
      <p className="ec-axis">縦線は全体平均（勝率{BASE_WIN}%・連対率{BASE_WIN * 2}%・複勝率{BASE_TOP3}%）。右の矢印は平均との差(pt)。</p>
    </div>
  );
}

function WhoBlock({ e, kind, name, horse }: { e: EntityCard; kind: "jockey" | "trainer"; name: string; horse: { no?: number; name: string; bracket?: number | null; weight?: number | null } }) {
  const jk = kind === "jockey";
  const bs = horse.bracket ? BRACKET[horse.bracket] : undefined;
  const p = e.profile;
  const meta = [e.id ? `ID ${e.id}` : "", p?.affiliation ?? "", p?.debut_year ?? ""].filter(Boolean);
  return (
    <div className={`ec-who ${kind}`}>
      <div className="ec-who-1">
        <span className={`ec-role ${kind}`}>{jk ? "騎手" : "調教師"}</span>
        <div className="ec-who-n">{name || "（名前の記載なし）"}</div>
      </div>
      {meta.length > 0 && <div className="ec-who-m">{meta.join(" ・ ")}</div>}
      <div className="ec-who-h">
        <span>{jk ? "今回の騎乗馬" : "今回の管理馬"}</span>
        {horse.no != null && <span className="ec-no s" style={bs ? { background: bs.bg, color: bs.color } : undefined}>{horse.no}</span>}
        <b>{horse.name}</b>
        {jk && horse.weight != null && <span className="ec-who-w">斤量 {horse.weight}kg</span>}
      </div>
    </div>
  );
}

function EntityView({ e, kind, mode, name, horse }: { e: EntityCard; kind: "jockey" | "trainer"; mode: ViewportMode; name: string; horse: { no?: number; name: string; bracket?: number | null; weight?: number | null } }) {
  const mobile = mode === "mobile";
  const jk = kind === "jockey";
  const who = jk ? "騎手" : "調教師";
  if (e.status === "no_data" && e.conditions.every((c) => !c.has_data)) {
    return (
      <>
        <WhoBlock e={e} kind={kind} name={name} horse={horse} />
        <Empty text={`この${who}の集計がありません`} note="集計が取り込まれるまで表示できません" />
        {e.profile && <Section title="プロフィール"><Profile e={e} /></Section>}
        {e.comment && <Section title="この馬の厩舎コメント"><Quote text={e.comment.comment} who={e.name} note={e.comment.evaluation} /></Section>}
      </>
    );
  }
  const wh = e.with_horse;
  const r90 = e.recent["cal90"], r365 = e.recent["cal365"], r10 = e.recent["roll10"];
  const form = formTag(r90, e.career);
  const c0 = e.conditions[0];
  const c0tag = c0 && c0.has_data !== false ? placeTag(c0.top3_rate, c0.starts, FEW_PERSON) : null;
  const c2 = e.conditions[2];
  const c2tag = c2 && c2.has_data !== false ? placeTag(c2.top3_rate, c2.starts, FEW_PERSON) : null;
  const careerRate = e.career?.top3_rate ?? BASE_TOP3;
  const specials = e.conditions
    .filter((c) => c.has_data !== false)
    .map((c) => ({ c, tag: placeTag(c.top3_rate, c.starts, FEW_PERSON) }))
    .filter((x) => x.tag.tone === "good" || x.tag.tone === "warn");
  const pc = e.profile?.career ?? null;

  const condTile = (c: CondStat | undefined, tag: Tag | null, fallback: string) =>
    c && tag && c.has_data !== false ? (
      <Tile label={`${splitLabel(c.label).main} 複勝率`} tone={tag.tone} tag={tag} big={num1(c.top3_rate)} unit="%" sub={<><b>{c.starts}走中</b>勝{c.wins}・3着内{c.top3}</>} />
    ) : (
      <Tile label={fallback} tone="muted" big="—" word sub="この条件の出走なし" />
    );

  return (
    <>
      <WhoBlock e={e} kind={kind} name={name} horse={horse} />
      <div className="ec-tiles">
        {r90 && r90.starts ? (
          <Tile label="直近90日 複勝率" tone={form.tone} tag={form.delta == null ? undefined : form} big={num1(r90.top3_rate)} unit="%" sub={<><b>{r90.starts}走中</b>勝{r90.wins}・3着内{r90.top3}</>} />
        ) : <Tile label="直近90日 複勝率" tone="muted" big="—" word sub="直近の騎乗なし" />}
        {condTile(c0, c0tag, "この競馬場 複勝率")}
        {jk ? (
          <Tile
            label="この馬との関係" word
            tone={wh?.changed === true ? "warn" : wh?.changed === false ? "ok" : "muted"}
            big={!wh || wh.changed == null ? "不明" : wh.changed ? "▽乗り替わり" : "△継続騎乗"}
            sub={wh ? (wh.changed ? <>前走 <b>{wh.previous_jockey ?? "-"}</b>／近{wh.checked}走の騎乗 {wh.rides_in_last5}回</> : <>近{wh.checked}走で <b>{wh.rides_in_last5}回</b> 騎乗{wh.rides_in_last5 ? `（${wh.wins}勝・3着内${wh.top3}）` : ""}</>) : "騎乗履歴なし"}
          />
        ) : condTile(c2, c2tag, "この距離 複勝率")}
        <Tile
          label="調子の判定" word tone={form.tone}
          big={form.text}
          sub={form.delta == null ? "直近90日が30走未満のため判定しません" : <>90日 <b>{pct(r90?.top3_rate)}</b> vs 通算 <b>{pct(e.career?.top3_rate)}</b>（{form.delta >= 0 ? "▲" : "▼"}{Math.abs(form.delta).toFixed(1)}pt）</>}
        />
      </div>

      <Section title="得意・苦手" hint="30走以上の条件だけ判定">
        <div className="ec-sp">
          {specials.length === 0 ? <span className="ec-sp-n">目立つ得意・苦手はありません（標準、または30走未満）</span> : specials.map(({ c, tag }) => {
            const t = TONE[tag.tone];
            return <span key={c.label} className="ec-sp-c" style={{ color: t.fg, background: t.bg, borderColor: t.bd }}><b>{tag.text}</b> {splitLabel(c.label).main}（{splitLabel(c.label).sub || ""}）<em>{pct(c.top3_rate)}</em></span>;
          })}
        </div>
      </Section>

      <Section title="今回のレース条件での成績" hint="2020年以降">
        <div className="ec-stats">{e.conditions.map((c: CondStat) => <Gauge key={c.label} s={c} label={c.label} few={FEW_PERSON} marker={careerRate} markerLabel="通算" />)}</div>
        <p className="ec-axis">棒の長さ＝複勝率（右端 {GAUGE_MAX}%）。縦線は{e.career ? `この${who}の通算 ${careerRate}%` : `全体平均 ${BASE_TOP3}%`}。30走未満は『少数』。</p>
      </Section>

      <Section title="最近の調子" hint="通算との比較">
        <div className="ec-stats">
          {[r90, r365].filter(Boolean).map((r) => <Gauge key={r.label} s={r} label={r.label} few={FEW_PERSON} marker={careerRate} markerLabel="通算" />)}
        </div>
        {r10 && r10.starts ? <div className="ec-ten"><div className="ec-ten-t">直近10走の内訳</div><Strip s={r10} /></div> : null}
      </Section>

      {pc && (
        <Section title="通算の成績" hint={`${pc.rides ?? "-"}乗 ${pc.wins ?? "-"}勝（中央）`}>
          <RateBars c={pc} />
        </Section>
      )}

      {!jk && e.comment && <Section title="この馬の厩舎コメント"><Quote text={e.comment.comment} who={e.name} note={e.comment.evaluation} /></Section>}

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
      {p.affiliation && <div><span className="ec-k">所属</span>{p.affiliation}{p.debut_year ? <span className="ec-dim">（{p.debut_year}）</span> : ""}</div>}
      {c && <div><span className="ec-k">通算（中央）</span><b>{c.wins}勝/{c.rides}</b><span className="ec-dim">（勝率 {pct(c.win_rate)}・連対率 {pct(c.top2_rate)}・複勝率 {pct(c.top3_rate)}）</span></div>}
      {p.this_year_text && <div><span className="ec-k">今年</span>{p.this_year_text}</div>}
      {p.g1_wins_text && <div><span className="ec-k">G1</span>{p.g1_wins_text}</div>}
      {p.graded_wins_text && <div><span className="ec-k">重賞</span>{p.graded_wins_text}</div>}
    </div>
  );
}

/* ── パネル本体 ── */
const CSS = `
.ec-root{--ec-dim:#94a6bf;position:fixed;inset:0;z-index:1000;background:rgba(2,6,23,.6)}
.ec-drawer{position:fixed;top:0;right:0;bottom:0;width:min(640px,94vw);background:var(--surface);border-left:1px solid var(--border);box-shadow:-16px 0 40px rgba(0,0,0,.5);display:flex;flex-direction:column;animation:ecIn .16s ease-out}
.ec-sheet{position:fixed;left:0;right:0;bottom:0;max-height:90vh;background:var(--surface);border-top:1px solid var(--border);border-radius:18px 18px 0 0;box-shadow:0 -14px 36px rgba(0,0,0,.55);display:flex;flex-direction:column;animation:ecUp .2s ease-out;touch-action:pan-y}
@keyframes ecIn{from{transform:translateX(24px);opacity:.4}to{transform:none;opacity:1}}
@keyframes ecUp{from{transform:translateY(40px);opacity:.4}to{transform:none;opacity:1}}
.ec-grab{width:44px;height:5px;border-radius:3px;background:#3a4a68;margin:8px auto 2px}
.ec-head{display:flex;align-items:center;gap:8px;padding:10px 12px 6px}
.ec-id{flex:1;min-width:0;display:flex;align-items:center;gap:10px}
.ec-no{flex:none;display:inline-flex;align-items:center;justify-content:center;width:42px;height:42px;border-radius:11px;font-size:22px;font-weight:800;border:2px solid rgba(255,255,255,.35);box-shadow:0 2px 6px rgba(0,0,0,.4);font-variant-numeric:tabular-nums}
.ec-id-t{min-width:0}
.ec-title{font-weight:800;font-size:21px;line-height:1.2;color:#fff;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ec-race{display:flex;align-items:center;gap:6px;margin-top:3px;font-size:12px;color:var(--ec-dim);white-space:nowrap;overflow:hidden}
.ec-race>span:last-child{overflow:hidden;text-overflow:ellipsis}
.ec-btn{flex:none;background:var(--surface2);border:1px solid var(--border);color:var(--text);border-radius:10px;min-width:40px;min-height:40px;font-size:17px;cursor:pointer}
.ec-btn:hover:not(:disabled){border-color:var(--accent)}
.ec-btn:disabled{opacity:.35;cursor:default}
.ec-sheet .ec-btn{min-width:44px;min-height:44px}
.ec-role{flex:none;display:inline-flex;align-items:center;justify-content:center;min-width:52px;height:42px;padding:0 8px;border-radius:11px;font-size:14px;font-weight:800;color:#fff;box-shadow:0 2px 6px rgba(0,0,0,.4)}
.ec-role.jockey{background:#1d4ed8}
.ec-role.trainer{background:#6d28d9}
.ec-no.s{width:20px;height:20px;border-radius:6px;font-size:12px;border-width:1px;box-shadow:none;background:var(--surface2);color:var(--text)}
.ec-mc.on{border-color:var(--accent);background:rgba(59,130,246,.18)}
.ec-who{border:1px solid var(--border);border-left-width:5px;border-radius:12px;padding:10px 12px;margin-bottom:12px;background:linear-gradient(135deg,rgba(59,130,246,.12),var(--surface2) 70%);box-shadow:0 2px 8px rgba(0,0,0,.25)}
.ec-who.trainer{border-left-color:#8b5cf6;background:linear-gradient(135deg,rgba(139,92,246,.14),var(--surface2) 70%)}
.ec-who.jockey{border-left-color:#3b82f6}
.ec-who-1{display:flex;align-items:center;gap:10px}
.ec-who-n{font-size:26px;font-weight:800;color:#fff;line-height:1.15;min-width:0;overflow-wrap:anywhere}
.ec-who-m{font-size:12px;color:var(--ec-dim);margin-top:5px}
.ec-who-h{display:flex;align-items:center;flex-wrap:wrap;gap:6px;margin-top:7px;padding-top:7px;border-top:1px solid var(--border);font-size:13px}
.ec-who-h>span:first-child{font-size:11.5px;color:var(--ec-dim);font-weight:700}
.ec-who-w{margin-left:auto;font-weight:700}
.ec-meta{display:flex;flex-wrap:wrap;gap:6px;padding:2px 12px 8px}
.ec-sheet .ec-meta{flex-wrap:nowrap;overflow-x:auto;scrollbar-width:none}
.ec-mc{flex:none;display:inline-flex;align-items:baseline;gap:5px;background:var(--surface2);border:1px solid var(--border);border-radius:8px;padding:3px 8px;font-size:13px;font-weight:700;color:#e6edf6;white-space:nowrap}
.ec-mc small{font-size:10px;font-weight:600;color:var(--ec-dim)}
.ec-mc em{font-style:normal;font-size:11.5px;font-weight:700}
.ec-tabs{display:flex;gap:4px;padding:0 12px;border-bottom:1px solid var(--border)}
.ec-tab{flex:1;min-width:0;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:1px;line-height:1.2;background:transparent;border:0;border-bottom:3px solid transparent;color:var(--ec-dim);font-size:15px;font-weight:700;padding:7px 4px;min-height:48px;cursor:pointer}
.ec-tab small{max-width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:11px;font-weight:600;color:var(--ec-dim)}
.ec-tab[aria-selected=true] small{color:#cbd5e1}
.ec-tab[aria-selected=true]{color:#fff;border-bottom-color:var(--accent);background:linear-gradient(180deg,transparent,rgba(59,130,246,.12))}
.ec-body{overflow-y:auto;padding:12px 12px 24px;-webkit-overflow-scrolling:touch;overscroll-behavior:contain}
.ec-tiles{display:grid;grid-template-columns:repeat(2,1fr);gap:8px;margin-bottom:14px}
.ec-tile{border:1px solid var(--border);border-left-width:4px;border-radius:12px;padding:9px 11px 10px;min-width:0;box-shadow:0 2px 8px rgba(0,0,0,.25)}
.ec-tile-h{display:flex;align-items:center;gap:6px}
.ec-tile-l{font-size:11.5px;color:var(--ec-dim);font-weight:700}
.ec-tile-v{font-size:30px;font-weight:800;line-height:1.1;margin-top:4px;font-variant-numeric:tabular-nums;display:flex;align-items:center;flex-wrap:wrap;gap:4px 8px}
.ec-tile-n{display:inline-flex;align-items:baseline}
.ec-tile-v small{font-size:14px;font-weight:700;margin-left:2px}
.ec-tile-v .ec-tag{margin-left:auto}
.ec-tile-v.w{font-size:21px;line-height:1.3}
.ec-tile-s{font-size:12px;color:var(--ec-dim);margin-top:4px;line-height:1.4}
.ec-tile-s b{color:var(--text);margin-right:4px}
.ec-tag{display:inline-flex;align-items:center;font-size:11px;font-weight:700;border:1px solid;border-radius:999px;padding:1px 8px;white-space:nowrap}
.ec-grade{display:inline-flex;align-items:center;justify-content:center;width:34px;height:34px;border-radius:9px;font-size:19px;font-weight:800;border:2px solid;flex:none}
.ec-grade.big{width:46px;height:46px;font-size:28px;border-radius:12px}
.ec-sec{margin:16px 0}
.ec-sec-t{display:flex;align-items:baseline;gap:8px;margin:0 0 8px;font-size:15px;font-weight:800;color:#e6edf6}
.ec-sec-t::before{content:"";align-self:center;width:4px;height:15px;border-radius:2px;background:var(--accent)}
.ec-sec-t small{font-size:11px;font-weight:500;color:var(--ec-dim)}
.ec-sum{cursor:pointer;min-height:44px;align-items:center;list-style:none}
.ec-sum::after{content:"▾";margin-left:auto;color:var(--ec-dim);transition:transform .12s}
details:not([open])>.ec-sum::after{transform:rotate(-90deg)}
.ec-sum::-webkit-details-marker{display:none}
.ec-trend{display:flex;gap:12px;align-items:stretch;background:var(--surface2);border:1px solid var(--border);border-radius:12px;padding:10px 12px;margin-bottom:10px}
.ec-trend-bars{flex:1;display:flex;gap:8px;align-items:flex-end;min-width:0}
.ec-tb{flex:1;display:flex;flex-direction:column;align-items:center;gap:2px;min-width:0}
.ec-tb-n{font-size:13px;font-weight:800;font-variant-numeric:tabular-nums}
.ec-tb-col{width:100%;height:78px;display:flex;flex-direction:column;align-items:center;justify-content:flex-end;gap:3px}
.ec-tb-col i{display:block;width:70%;max-width:30px;border-radius:5px 5px 2px 2px;min-height:5px;flex:none}
.ec-tb em{font-style:normal;font-size:10.5px;color:var(--ec-dim)}
.ec-trend-s{flex:none;display:flex;flex-direction:column;justify-content:center;gap:5px;padding-left:12px;border-left:1px solid var(--border);min-width:84px}
.ec-trend-s small{display:block;font-size:10.5px;color:var(--ec-dim)}
.ec-trend-s b{font-size:19px;font-weight:800;color:#fff}
.ec-trend-s b span{font-size:12px;color:var(--ec-dim);font-weight:600}
.ec-trend-h{font-size:10px;color:var(--ec-dim);line-height:1.3}
.ec-fin{flex:none;display:inline-flex;align-items:center;justify-content:center;border-radius:50%;border:2px solid;font-weight:800;font-variant-numeric:tabular-nums;line-height:1}
.ec-fin.m{width:30px;height:30px;font-size:14px}
.ec-fin.l{width:40px;height:40px;font-size:19px}
.ec-cls{display:inline-block;font-size:10px;font-weight:800;border-radius:4px;padding:1px 5px;margin-right:5px;vertical-align:1px}
.ec-surf{display:inline-block;font-size:10.5px;font-weight:800;border-radius:4px;padding:0 5px;margin-right:4px;vertical-align:1px}
.ec-same,.ec-dc{font-style:normal;font-size:10.5px;font-weight:700;margin-left:6px;border-radius:4px;padding:0 5px}
.ec-same{color:#bfdbfe;background:rgba(59,130,246,.25)}
.ec-dc{color:var(--ec-dim);border:1px solid var(--border)}
.ec-scroll-x{overflow-x:auto}
.ec-table{width:100%;border-collapse:separate;border-spacing:0;font-size:12.5px;background:var(--surface2);border:1px solid var(--border);border-radius:12px;overflow:hidden}
.ec-table th{font-size:10.5px;color:var(--ec-dim);font-weight:700;padding:7px 6px;border-bottom:1px solid var(--border);white-space:nowrap;background:rgba(255,255,255,.02)}
.ec-table td{padding:8px 5px;border-bottom:1px solid rgba(36,48,73,.6);text-align:center;vertical-align:middle}
.ec-table tbody tr:last-child td{border-bottom:0}
.ec-table td small{display:block;font-size:10px;color:var(--ec-dim)}
.ec-table .ec-l{text-align:left}
.ec-rn{font-weight:700;color:#e6edf6;white-space:nowrap}
.ec-rs{font-size:11.5px;color:var(--ec-dim);margin-top:2px;white-space:nowrap}
.ec-pop{display:inline-flex;flex-direction:column;align-items:center;gap:1px;font-size:11.5px}
.ec-gap{font-weight:800;font-size:12.5px}
.ec-gap small{display:inline;font-size:11px;font-weight:600}
.ec-l3{font-weight:700;font-variant-numeric:tabular-nums}
.ec-l3.best{color:#4ade80;font-weight:800;text-decoration:underline;text-underline-offset:3px}
.ec-l3 small{display:block;font-size:9.5px;text-decoration:none;color:#4ade80}
.ec-pass{display:inline-flex;gap:2px}
.ec-pass i{font-style:normal;min-width:19px;padding:1px 3px;border-radius:4px;font-size:11px;font-weight:700;color:#fff;text-align:center}
.ec-st{display:block;font-size:10px;color:var(--ec-dim)}
.ec-run .ec-st{display:inline;font-size:11.5px;margin-left:4px}
.ec-runs{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:8px}
.ec-run{display:flex;gap:10px;align-items:flex-start;background:var(--surface2);border:1px solid var(--border);border-radius:12px;padding:10px 11px}
.ec-run-b{flex:1;min-width:0}
.ec-run-top{display:flex;align-items:baseline;gap:6px;min-width:0}
.ec-run-top .ec-cls{margin-right:0}
.ec-run-date{font-size:12px;font-weight:700;color:var(--ec-dim);flex:none}
.ec-run-name{flex:1;min-width:0;font-weight:800;font-size:14px;color:#fff;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.ec-run-sub{font-size:12px;color:var(--ec-dim);margin-top:4px;line-height:1.5}
.ec-run-facts{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:12.5px;margin-top:4px;color:var(--text)}
.ec-run-j{display:block;margin-top:2px}
.ec-run .ec-l3{font-size:14px}
.ec-run .ec-l3 small{display:inline;margin-left:3px}
.ec-run .ec-pop{flex-direction:row;gap:5px}
.ec-stats{display:grid;grid-template-columns:1fr;gap:8px}
.ec-drawer .ec-stats{grid-template-columns:1fr 1fr}
.ec-stat{background:var(--surface2);border:1px solid var(--border);border-top-width:3px;border-radius:12px;padding:9px 11px 10px}
.ec-g-top{display:flex;align-items:center;justify-content:space-between;gap:6px}
.ec-g-l{font-size:13px;font-weight:800;color:#e6edf6}
.ec-g-sub{font-size:11px;color:var(--ec-dim);margin-top:1px}
.ec-g-main{display:flex;align-items:baseline;gap:8px;margin-top:6px}
.ec-g-runs{font-size:13px;color:var(--ec-dim)}
.ec-g-runs b{font-size:21px;color:#fff;font-weight:800;margin-right:1px}
.ec-g-big{font-size:21px;font-weight:800;margin-left:auto;font-variant-numeric:tabular-nums}
.ec-g-big small{font-size:12px;margin-left:1px}
.ec-g-unit{font-size:11px;color:var(--ec-dim)}
.ec-g-bar{position:relative;height:9px;background:#0f1522;border:1px solid var(--border);border-radius:5px;margin-top:6px;overflow:visible}
.ec-g-bar i{display:block;height:100%;border-radius:4px}
.ec-g-bar i.few{background-image:repeating-linear-gradient(135deg,rgba(0,0,0,.35) 0 4px,transparent 4px 8px)!important;opacity:.75}
.ec-g-bar u{position:absolute;top:-4px;bottom:-4px;width:2px;background:#e6edf6;border-radius:1px;text-decoration:none}
.ec-g-foot{display:flex;justify-content:space-between;gap:8px;font-size:11px;color:var(--ec-dim);margin-top:5px}
.ec-g-foot span:last-child{font-weight:700;white-space:nowrap}
.ec-axis{margin:8px 0 0;font-size:11px;color:var(--ec-dim);line-height:1.5}
.ec-empty{display:flex;flex-direction:column;gap:2px;align-items:flex-start;border:1px dashed #3a4a68;border-radius:10px;padding:12px 14px;color:var(--ec-dim)}
.ec-empty b{font-size:13px;font-weight:700}
.ec-empty small{font-size:11.5px}
.ec-empty.c{padding:8px 10px;margin-top:6px}
.ec-sp{display:flex;flex-wrap:wrap;gap:6px}
.ec-sp-c{display:inline-flex;align-items:baseline;gap:5px;border:1px solid;border-radius:10px;padding:5px 9px;font-size:12.5px}
.ec-sp-c b{font-weight:800}
.ec-sp-c em{font-style:normal;font-weight:800}
.ec-sp-n{font-size:12.5px;color:var(--ec-dim);border:1px dashed #3a4a68;border-radius:10px;padding:7px 10px}
.ec-ten{margin-top:10px;background:var(--surface2);border:1px solid var(--border);border-radius:12px;padding:9px 12px}
.ec-ten-t{font-size:12px;font-weight:700;color:var(--ec-dim);margin-bottom:4px}
.ec-strip-d{display:flex;gap:4px;font-size:22px;line-height:1}
.d.w{color:#fbbf24}.d.p{color:#60a5fa}.d.o{color:#64748b}
.ec-strip-l{display:flex;flex-wrap:wrap;gap:4px 12px;font-size:11.5px;color:var(--ec-dim);margin-top:6px}
.ec-strip-l .d{margin-right:3px}
.ec-strip-l em{font-style:normal}
.ec-rb{display:flex;flex-direction:column;gap:9px;background:var(--surface2);border:1px solid var(--border);border-radius:12px;padding:11px 12px}
.ec-rb-r{display:grid;grid-template-columns:48px 1fr 56px 52px;align-items:center;gap:8px}
.ec-rb-r .ec-g-bar{margin-top:0}
.ec-rb-l{font-size:12.5px;font-weight:700}
.ec-rb-r b{font-size:14px;text-align:right;font-variant-numeric:tabular-nums}
.ec-rb-d{font-size:11.5px;font-weight:700}
.ec-train{background:var(--surface2);border:1px solid var(--border);border-radius:12px;padding:10px 12px}
.ec-train-h{display:flex;gap:10px;align-items:center}
.ec-train-c{margin-top:8px;color:var(--text)}
.ec-lap{margin-top:10px}
.ec-lap-bars{display:flex;gap:10px;align-items:flex-end}
.ec-lb{flex:1;display:flex;flex-direction:column;align-items:center;gap:2px;min-width:0}
.ec-lb b{font-size:13px;font-variant-numeric:tabular-nums}
.ec-lb-col{width:100%;height:52px;display:flex;align-items:flex-end;justify-content:center}
.ec-lb-col i{display:block;width:62%;max-width:34px;border-radius:5px 5px 2px 2px;background:linear-gradient(180deg,#60a5fa,#2563eb)}
.ec-lb span{font-size:11.5px;font-weight:700}
.ec-lb small{font-size:10px;color:var(--ec-dim)}
.ec-lap-n{margin-top:8px;font-size:11px;color:var(--ec-dim);line-height:1.5}
.ec-lap-t{display:block;font-size:12.5px}
.ec-quote{position:relative;margin:0;background:linear-gradient(135deg,rgba(59,130,246,.12),var(--surface2) 65%);border:1px solid var(--border);border-left:4px solid var(--accent);border-radius:12px;padding:12px 14px 10px 40px}
.ec-q{position:absolute;left:11px;top:-2px;font-size:44px;line-height:1;font-family:Georgia,serif;color:var(--accent);opacity:.8}
.ec-quote blockquote{margin:0;font-size:14px;line-height:1.75;color:#e6edf6}
.ec-quote figcaption{margin-top:8px;font-size:11.5px;color:var(--ec-dim);text-align:right}
.ec-dim{color:var(--ec-dim);font-weight:400}
.ec-kv{display:flex;flex-direction:column;gap:7px;font-size:13px;background:var(--surface2);border:1px solid var(--border);border-radius:12px;padding:10px 12px}
.ec-k{display:inline-block;min-width:84px;color:var(--ec-dim);font-size:11.5px;font-weight:600}
.ec-mono{font-family:ui-monospace,monospace;font-size:11.5px;overflow-wrap:anywhere}
.ec-p{margin:0;font-size:13px;line-height:1.65}
.ec-weights{display:flex;flex-wrap:wrap;gap:6px}
.ec-w{display:inline-flex;align-items:baseline;gap:5px;background:var(--surface2);border:1px solid var(--border);border-radius:8px;padding:4px 8px;font-size:13px}
.ec-w small{font-size:10.5px;color:var(--ec-dim)}
.ec-w em{font-style:normal;font-size:11.5px;font-weight:700}
.ec-note{margin:16px 0 0;font-size:11px;color:var(--ec-dim);line-height:1.6}
.ec-skel{height:84px;border-radius:12px;background:linear-gradient(90deg,var(--surface2),var(--border),var(--surface2));background-size:200% 100%;animation:ecSk 1.1s linear infinite}
.ec-skel-g{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:12px}
@keyframes ecSk{from{background-position:200% 0}to{background-position:-200% 0}}
.ec-sheet .ec-tile-v{font-size:26px}
.ec-sheet .ec-tile-v.w{font-size:19px}
.ec-sheet .ec-no.s{width:20px;height:20px;font-size:12px}
.ec-sheet .ec-who-n{font-size:23px}
.ec-sheet .ec-title{font-size:17px;white-space:normal;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical}
.ec-sheet .ec-no{width:38px;height:38px;font-size:20px}
@media (prefers-reduced-motion:reduce){.ec-drawer,.ec-sheet,.ec-skel{animation:none}.ec-sum::after{transition:none}}
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

  const horseName = cur?.horse_name ?? card?.horse?.name ?? "";
  const no = cur?.horse_number;
  const jockeyFull = pickName(card?.jockey?.name ?? card?.jockey?.profile?.name, cur?.jockey_name);
  const trainerFull = pickName(card?.trainer?.name ?? card?.trainer?.profile?.name, cur?.trainer_name);
  const jockeyName = jockeyFull || null;
  const trainerName = trainerFull || null;
  const subject = tab === "jockey" ? jockeyFull : tab === "trainer" ? trainerFull : horseName;
  const tabName: Record<CardTab, string> = { horse: horseName, jockey: jockeyFull, trainer: trainerFull };
  const title = tab === "horse" ? `${no ?? ""} ${horseName}`.trim() : `${tab === "jockey" ? "騎手" : "調教師"} ${subject}`;
  const sexAge = cur?.sex_age ?? card?.horse?.sex_age ?? null;
  const weight = cur?.weight ?? card?.horse?.weight ?? null;
  const wchg = cur?.weight_change ?? card?.horse?.weight_change ?? null;
  const jw = cur?.jockey_weight ?? card?.jockey?.weight_today ?? null;
  const bracket = cur?.bracket_number ?? null;
  const bs = bracket ? BRACKET[bracket] : undefined;
  const rn = card?.race ? splitRaceName(card.race.name, card.race.grade) : null;
  const curRace: Cur | undefined = card?.race ? { surface: card.race.surface, distance: card.race.distance } : undefined;

  const body = useMemo(() => {
    if (loading) return <div aria-busy="true" aria-label="読み込み中"><div className="ec-skel-g">{[0, 1, 2, 3].map((i) => <div key={i} className="ec-skel" />)}</div><div className="ec-skel" /></div>;
    if (error || !card) return (
      <div role="alert" className="ec-p">
        取得できませんでした（{error || "不明なエラー"}）。
        <div><button type="button" className="ec-btn" style={{ marginTop: 8, padding: "0 14px" }} onClick={() => load()}>再読み込み</button></div>
      </div>
    );
    if (tab === "horse") return <HorseView h={card.horse} mode={mode} cur={curRace} jockeyName={jockeyName} />;
    return <EntityView e={tab === "jockey" ? card.jockey : card.trainer} kind={tab} mode={mode} name={tab === "jockey" ? jockeyFull : trainerFull} horse={{ no, name: horseName, bracket, weight: jw }} />;
  }, [loading, error, card, tab, mode, load, curRace?.surface, curRace?.distance, jockeyName, jockeyFull, trainerFull, no, horseName, bracket, jw]);   // eslint-disable-line react-hooks/exhaustive-deps

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
            <div className="ec-id">
              {tab === "horse" ? (
                no != null && (
                  <span className="ec-no" title={bracket ? `${bracket}枠 ${no}番` : `${no}番`} style={bs ? { background: bs.bg, color: bs.color } : { background: "var(--surface2)", color: "var(--text)" }}>{no}</span>
                )
              ) : (
                <span className={`ec-role ${tab}`}>{tab === "jockey" ? "騎手" : "調教師"}</span>
              )}
              <div className="ec-id-t">
                <div className="ec-title">{subject || "（名前の記載なし）"}</div>
                {tab === "horse" ? (
                  card?.race && (
                    <div className="ec-race">
                      <span><SurfaceChip s={card.race.surface} />{card.race.distance ? `${card.race.distance}m` : ""}</span>
                      {rn?.cls && <ClassChip cls={rn.cls} />}
                      <span>{card.race.venue ?? ""} {rn?.name ?? ""}</span>
                    </div>
                  )
                ) : (
                  <div className="ec-race">
                    <span>この馬</span>
                    {no != null && <span className="ec-no s" style={bs ? { background: bs.bg, color: bs.color } : undefined}>{no}</span>}
                    <span>{horseName}</span>
                  </div>
                )}
              </div>
            </div>
            <button type="button" className="ec-btn" aria-label="次の馬" disabled={!next} onClick={() => next && onHorse(next.horse_id)}>›</button>
            <button type="button" className="ec-btn" ref={closeRef} aria-label="閉じる" onClick={onClose}>✕</button>
          </div>
          <div className="ec-meta" aria-label="出走情報">
            {bracket ? <span className="ec-mc"><small>枠</small>{bracket}枠</span> : null}
            {sexAge && <span className="ec-mc"><small>性齢</small>{sexAge}</span>}
            {jw != null && <span className="ec-mc"><small>斤量</small>{jw}</span>}
            {weight ? (
              <span className="ec-mc"><small>馬体重</small>{weight}kg
                {wchg != null && <em style={{ color: Math.abs(wchg) >= 10 ? TONE.warn.fg : "var(--ec-dim)" }}>{wchg > 0 ? "▲" : wchg < 0 ? "▼" : "＝"}{Math.abs(wchg)}</em>}
              </span>
            ) : null}
            {jockeyName && <span className={`ec-mc${tab === "jockey" ? " on" : ""}`}><small>騎手</small>{jockeyName}</span>}
            {trainerName && <span className={`ec-mc${tab === "trainer" ? " on" : ""}`}><small>調教師</small>{trainerName}</span>}
          </div>
        </div>
        <div className="ec-tabs" role="tablist" aria-label="表示する情報" onKeyDown={onTabKey}>
          {TABS.map((t) => (
            <button key={t.id} type="button" role="tab" id={`ec-tab-${t.id}`} aria-selected={tab === t.id} aria-controls="ec-body" tabIndex={tab === t.id ? 0 : -1} className="ec-tab" onClick={() => onTab(t.id)}>
              <span>{t.label}</span>
              {tabName[t.id] ? <small>{tabName[t.id]}</small> : null}
            </button>
          ))}
        </div>
        <div className="ec-body" id="ec-body" role="tabpanel" aria-labelledby={`ec-tab-${tab}`}>{body}</div>
      </aside>
    </div>,
    document.body,
  );
}
