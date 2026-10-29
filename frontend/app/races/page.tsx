"use client";

import { PendingLink } from "@/components/PendingLink";
import { fetchRaceList } from "@/lib/httpCache";
import { useCallback, useEffect, useMemo, useState } from "react";
import { PageShell } from "@/components/PageShell";
import { GradeChip, WeekPicker } from "@/components/WeekPicker";
import { RaceCalendar } from "@/components/RaceCalendar";
import { latestOnOrBefore, resolveDate } from "@/lib/raceCalendar";
import { USE_MOCK, MOCK_WEEKLY_RACES, getMockRaceDates } from "@/lib/mock";
import { useAuthStatus } from "@/lib/hooks/useAuthStatus";
import { MemberUpgradeModal } from "@/components/upgrade/MemberUpgradeModal";
import { formatMD, todayJst, weekdayOf, WEEKDAY_JA } from "@/lib/weekDates";
import { LoadingNotice } from "@/components/LoadingNotice";
import { useAutoRetry } from "@/lib/hooks/useAutoRetry";
import { LEVEL_STYLE, useRaceValues, valueTitle, type RaceValue } from "@/lib/raceValue";
import { roiText, useBetsSummary, yen, type BetsSummary, type RaceBet, type RatedCounts } from "@/lib/betsSummary";

/* レース一覧（過去の開催日ごと・開催場ごと）。未来の開催はここには出さない（今日まで。今日が開催日ならその日は含める）。
   データは GCS/ホストの開催日別一覧（/api/scrape-dates と /api/race-list/<日付>）。DB（/api/v1/races）は使わない:
   DB の取り込みが古いと、レース名が文字化けしたまま出ていた（stg, 2026-10-16）。 */

type Row = {
  race_id: string;
  round: number;
  venue: string;
  race_name: string;
  grade?: string;
  surface?: string;
  distance?: number;
  start_time?: string;
  entries_count?: number;
  track_condition?: string;
  has_result?: boolean;
};

const PAST_DAYS = Number(process.env.NEXT_PUBLIC_PICKER_PAST_DAYS) || 30;
const GRADES = new Set(["G1", "G2", "G3"]);

function toRow(x: Record<string, unknown>): Row {
  const id = String(x.race_id);
  return {
    race_id: id,
    round: Number(x.round ?? Number(id.slice(-2))),
    venue: String(x.venue ?? ""),
    race_name: String(x.race_name ?? ""),
    grade: (x.grade as string | undefined) ?? undefined,
    surface: (x.surface as string | undefined) ?? undefined,
    distance: x.distance ? Number(x.distance) : undefined,
    start_time: (x.start_time as string | undefined) ?? undefined,
    entries_count: x.entries_count ? Number(x.entries_count) : x.field_size ? Number(x.field_size) : undefined,
    track_condition: (x.track_condition as string | undefined) ?? undefined,
    has_result: Boolean(x.has_race_result || x.result_viewable),
  };
}

/** 「芝1600m」「ダ1200m」「障2880m」 */
function course(r: Row): string {
  const s = (r.surface ?? "").startsWith("ダ") ? "ダ" : (r.surface ?? "").startsWith("障") ? "障" : (r.surface ?? "").slice(0, 1);
  return r.distance ? `${s}${r.distance}m` : s;
}

/** race_id（YYYY PP KK DD RR）から「2回 6日目」 */
function kaiNichi(raceId: string): string {
  const kai = Number(raceId.slice(6, 8)), nichi = Number(raceId.slice(8, 10));
  return kai && nichi ? `${kai}回 ${nichi}日目` : "";
}

/** 日・週の回収率の帯（有料会員のみ）。仮想購入＝推奨馬券を保存済みの最新オッズで買ったとみなした結果 */
function RoiBar({ b }: { b: BetsSummary }) {
  const emptyText = (c?: RatedCounts) =>
    c && c.unrated > 0 && c.rated === 0 ? `推奨度の記録がまだありません（未記録 ${c.unrated}R${c.skipped ? `・見送り ${c.skipped}R` : ""}）`
      : c && c.settled > 0 ? `推奨度が標準以上のレースがありません（見送り ${c.skipped}R）` : "集計できるレースがありません";
  const cell = (label: string, x: { roi: number | null; invested: number; returned: number; races_bought: number; races_hit: number; counts?: RatedCounts }, sub?: string) => (
    <div style={{ flex: "1 1 220px", minWidth: 0 }}>
      <div style={{ fontSize: 11, color: "var(--text-dim)" }}>{label}{sub ? <span style={{ marginLeft: 6 }}>{sub}</span> : null}</div>
      {x.races_bought > 0 ? (
        <div style={{ display: "flex", alignItems: "baseline", gap: 8, flexWrap: "wrap" }}>
          <span style={{ fontSize: 22, fontWeight: 800, fontVariantNumeric: "tabular-nums", color: (x.roi ?? 0) >= 1 ? "#4ade80" : "#f87171" }}>{roiText(x.roi)}</span>
          <span style={{ fontSize: 12, color: "var(--text-dim)", fontVariantNumeric: "tabular-nums" }}>{yen(x.invested)} → {yen(x.returned)}・的中 {x.races_hit}/{x.races_bought}R</span>
        </div>
      ) : <div style={{ fontSize: 13, color: "var(--text-dim)", padding: "4px 0" }}>{emptyText(x.counts)}</div>}
    </div>
  );
  const w = b.week;
  return (
    <section aria-label="推奨馬券の回収率" title={b.note} style={{ display: "flex", gap: 20, flexWrap: "wrap", padding: "10px 14px", marginBottom: 12, background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 12 }}>
      {cell(`この日の回収率・標準以上（${b.label}）`, b.day, b.day.status === "final" ? "" : "暫定")}
      {cell(`この週の回収率・標準以上（${formatMD(w.from)}〜${formatMD(w.to)}）`, w, w.status === "final" ? `${w.days.length}日分` : `${w.days.length}日分・暫定`)}
      <div style={{ flexBasis: "100%", fontSize: 10, color: "var(--text-dim)" }}>推奨度が標準以上のレースの推奨馬券の仮想購入（見送りのレースは含めません。保存済みの最新オッズで購入したとみなし、払戻は確定払戻）。実際の購入ではありません。</div>
    </section>
  );
}

function VenueBlock({ venue, rows, values, bets }: { venue: string; rows: Row[]; values: Map<string, RaceValue>; bets: Record<string, RaceBet> | null }) {
  const sorted = [...rows].sort((a, b) => a.round - b.round);
  const g = sorted.filter((r) => r.grade && GRADES.has(r.grade)).length;
  const cond = sorted.find((r) => r.track_condition)?.track_condition;
  return (
    <section aria-label={`${venue}のレース`} style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 12, overflow: "hidden" }}>
      <header style={{ display: "flex", alignItems: "baseline", gap: 10, padding: "10px 14px", background: "var(--surface2)", borderBottom: "1px solid var(--border)", flexWrap: "wrap" }}>
        <h2 style={{ margin: 0, fontSize: 16, fontWeight: 700, color: "#f0f6fc" }}>{venue}</h2>
        <span style={{ fontSize: 12, color: "var(--text-dim)" }}>{kaiNichi(sorted[0].race_id)}</span>
        <span style={{ fontSize: 12, color: "var(--text-dim)" }}>{sorted.length}R</span>
        {g > 0 && <span style={{ fontSize: 11, fontWeight: 700, color: "#fbbf24" }}>重賞 {g}</span>}
        {cond && <span style={{ marginLeft: "auto", fontSize: 11, color: "var(--text-dim)" }}>馬場 {cond}</span>}
      </header>
      <ul style={{ listStyle: "none", margin: 0, padding: 0 }}>
        {sorted.map((r) => {
          const isG = !!r.grade && GRADES.has(r.grade);
          const v = values.get(r.race_id);
          const lv = v && v.level !== "none" ? v.level : null;       // 買い目の推奨度（有料会員のみ取得できる）
          const ls = lv ? LEVEL_STYLE[lv] : null;
          const bet = bets?.[r.race_id];
          const hit = !!bet?.hit;
          return (
            <li key={r.race_id} style={{ borderBottom: "1px solid var(--border)", borderLeft: ls && lv !== "pass" ? `3px solid ${ls.border}` : "3px solid transparent", background: hit ? "rgba(251,191,36,0.10)" : undefined }}>
              <PendingLink
                href={`/race/${r.race_id}`}
                warm={[`/api/race/${r.race_id}`, `/api/race/${r.race_id}/predictions`]}
                title={`${venue} ${r.round}R ${r.race_name}${v ? `\n${valueTitle(v)}` : ""}${bet ? `\n仮想購入 ${yen(bet.total_bet)} → ${yen(bet.total_return)}（回収率 ${roiText(bet.roi)}）${bet.hit ? " 的中" : " 不的中"}` : ""}`}
                style={{ display: "grid", gridTemplateColumns: "34px 44px 1fr auto", alignItems: "center", gap: 10, padding: "8px 14px", textDecoration: "none", color: "var(--text)", background: isG ? "rgba(251,191,36,0.06)" : undefined }}
              >
                <span style={{ background: lv === "hot" ? ls!.bg : "#3b82f6", boxShadow: lv === "hot" ? ls!.glow : undefined, color: "#fff", borderRadius: 6, textAlign: "center", fontWeight: 700, fontSize: 13, padding: "3px 0" }}>{r.round}</span>
                <span style={{ fontSize: 12, color: "var(--text-dim)", fontVariantNumeric: "tabular-nums" }}>{r.start_time ?? "—"}</span>
                <span style={{ minWidth: 0 }}>
                  <span style={{ fontSize: 14, fontWeight: isG ? 700 : 600, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", display: "inline-block", maxWidth: "100%", verticalAlign: "middle" }}>{r.race_name}</span>
                  {isG && <span style={{ marginLeft: 6 }}><GradeChip grade={r.grade!} /></span>}
                  {!isG && (r.grade === "OP" || r.grade === "L") && <span style={{ marginLeft: 6, fontSize: 10, color: "var(--text-dim)", border: "1px solid var(--border)", borderRadius: 4, padding: "0 5px" }}>{r.grade}</span>}
                  <span style={{ display: "block", fontSize: 11, color: "var(--text-dim)" }}>
                    {[course(r), r.entries_count ? `${r.entries_count}頭` : ""].filter(Boolean).join(" · ")}
                    {ls && lv !== "pass" && <span style={{ marginLeft: 8, color: ls.border, fontWeight: 700 }}>{lv === "hot" ? "★ " : ""}{v?.tier?.label ?? ls.label}</span>}
                  </span>
                </span>
                {(r.has_result || bet) && (
                  <span style={{ display: "flex", alignItems: "center", gap: 6, marginRight: 18 }}>
                    {hit && <span title={`推奨馬券が的中（仮想購入 ${yen(bet!.total_bet)} → ${yen(bet!.total_return)}）`} style={{ fontSize: 11, fontWeight: 800, color: "#1a1204", background: "linear-gradient(135deg,#fde68a,#f59e0b)", borderRadius: 4, padding: "1px 7px", whiteSpace: "nowrap" }}>的中 {roiText(bet!.roi)}</span>}
                    {r.has_result && <span style={{ fontSize: 10, color: "#4ade80", border: "1px solid rgba(74,222,128,0.4)", borderRadius: 4, padding: "1px 6px" }}>結果</span>}
                  </span>
                )}
              </PendingLink>
            </li>
          );
        })}
      </ul>
    </section>
  );
}

export default function RacesPage() {
  const [showModal, setShowModal] = useState(false);
  const [dates, setDates] = useState<string[]>([]);
  const [extraDates, setExtraDates] = useState<string[]>([]);
  const [date, setDate] = useState("");
  const [rows, setRows] = useState<Row[]>([]);
  const [loadingDates, setLoadingDates] = useState(true);
  const [loadingRaces, setLoadingRaces] = useState(false);
  const [error, setError] = useState("");
  const [futureNotice, setFutureNotice] = useState(false);
  const [sheetOpen, setSheetOpen] = useState(false);
  const [reloadKey, setReloadKey] = useState(0);
  useAutoRetry(!!error, () => { setError(""); setReloadKey((n) => n + 1); }, 8000);
  const { isMember, loading: authLoading } = useAuthStatus();
  const today = todayJst();
  const values = useRaceValues(date);          // 買い目の推奨度（有料会員のみ）
  const betsSummary = useBetsSummary(date);    // 推奨馬券の的中・日と週の回収率（有料会員のみ）

  /* 開催日の一覧（過去 30 日。今日が開催日なら今日を含む）。古い日はカレンダーから選ぶ。初期は『直近の開催日』（?date= があればそれ。未来なら直近に切り替える） */
  useEffect(() => {
    (async () => {
      try {
        let list: string[];
        if (USE_MOCK) {
          list = getMockRaceDates().filter((d) => d <= today);
        } else {
          const res = await fetch(`/api/scrape-dates?picker_past_days=${PAST_DAYS}&picker_future_days=0`, { cache: "no-store" });
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          list = ((await res.json()).dates ?? []).filter((d: string) => d <= today);
        }
        setDates(list);
        const r = resolveDate(new URLSearchParams(window.location.search).get("date") ?? "", latestOnOrBefore(list, today), today);
        setDate(r.date);
        setFutureNotice(r.future);
      } catch {
        setError("開催日の一覧を取得できませんでした。");
      } finally {
        setLoadingDates(false);
      }
    })();
  }, [today, reloadKey]);

  /* 選んだ開催日のレース一覧 */
  useEffect(() => {
    if (!date) return;
    let alive = true;
    (async () => {
      setLoadingRaces(true);
      setError("");
      try {
        if (USE_MOCK) {
          if (alive) setRows(MOCK_WEEKLY_RACES.map((r) => toRow({ ...r })));
          return;
        }
        const d = (await fetchRaceList(date)) as { races?: Record<string, unknown>[] } | Record<string, unknown>[];
        if (alive) setRows(((Array.isArray(d) ? d : d.races) ?? []).map((x: Record<string, unknown>) => toRow(x)));
      } catch (e) {
        if (alive) { setRows([]); setError(`レース一覧を取得できませんでした（${e instanceof Error ? e.message : String(e)}）`); }
      } finally {
        if (alive) setLoadingRaces(false);
      }
    })();
    return () => { alive = false; };
  }, [date, reloadKey]);

  // カレンダーで読み込んだ月の開催日を足す（古い開催でも、週ピッカーがその開催の全日を出せるように）
  const addDates = useCallback((ds: string[]) => setExtraDates((cur) => (ds.every((d) => cur.includes(d)) ? cur : [...new Set([...cur, ...ds])])), []);
  const allDates = useMemo(() => [...new Set([...dates, ...extraDates])].filter((d) => d <= today).sort(), [dates, extraDates, today]);
  const latest = latestOnOrBefore(allDates, today);

  function choose(d: string) {
    setDate(d);
    setFutureNotice(false);
    setSheetOpen(false);
    const u = new URL(window.location.href);
    u.searchParams.set("date", d);
    window.history.replaceState(null, "", u.toString());
  }

  // JRA の競馬場コード順（race_id の 5〜6 桁目: 札幌01 函館02 福島03 新潟04 東京05 中山06 中京07 京都08 阪神09 小倉10）
  const byVenue = useMemo(() => {
    const m = new Map<string, Row[]>();
    for (const r of rows) m.set(r.venue, [...(m.get(r.venue) ?? []), r]);
    return [...m.entries()].sort((a, b) => a[1][0].race_id.slice(4, 6).localeCompare(b[1][0].race_id.slice(4, 6)));
  }, [rows]);

  const wd = date ? WEEKDAY_JA[weekdayOf(date)] : "";

  return (
    <>
      {showModal && <MemberUpgradeModal onClose={() => setShowModal(false)} />}
      <PageShell title="レース一覧">
        {!authLoading && !isMember && (
          <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "10px 16px", borderRadius: 8, background: "rgba(59,130,246,0.08)", border: "1px solid rgba(59,130,246,0.25)", marginBottom: 16, fontSize: 13, color: "var(--text)", flexWrap: "wrap" }}>
            <span>🔒 AI予測データは会員限定です。</span>
            <button className="hit44" onClick={() => setShowModal(true)} style={{ background: "none", border: "none", color: "var(--accent)", cursor: "pointer", fontSize: 13, padding: 0, textDecoration: "underline" }}>
              会員登録はこちら →
            </button>
          </div>
        )}

        <div className="races-layout">
          {/* PC: 左にカレンダー（過去の開催日を年月から選ぶ） */}
          <aside className="races-cal" aria-label="開催日カレンダー">
            <RaceCalendar selectedDate={date} today={today} latestDate={latest} onSelect={choose} onMonthDays={addDates} />
          </aside>

          <div className="races-main">
            <div style={{ marginBottom: 14 }}>
              <WeekPicker dates={allDates} today={today} selectedDate={date} loading={loadingDates} onSelectDate={choose} pastOnly />
              {/* スマホ: カレンダーはボトムシートで開く（画面をふさがない） */}
              <button className="races-cal-open" onClick={() => setSheetOpen(true)} aria-haspopup="dialog">
                📅 カレンダーから過去の開催日を選ぶ
              </button>
            </div>

            {futureNotice && (
              <div role="status" style={{ marginBottom: 10, fontSize: 12, color: "var(--text-dim)" }}>
                未来の日付が指定されたため、直近の開催日を表示しています。
              </div>
            )}

            {date && (
              <div style={{ marginBottom: 12, fontSize: 13, color: "var(--text-dim)" }}>
                <span style={{ color: "var(--text)", fontWeight: 700 }}>{date.slice(0, 4)}年{formatMD(date)}（{wd}）</span> — {byVenue.map(([v, r]) => `${v} ${r.length}R`).join(" ／ ") || "開催なし"}
              </div>
            )}
            {betsSummary && <RoiBar b={betsSummary} />}
            {error && (
              <div role="alert" style={{ color: "var(--err)", fontSize: 13, padding: "10px 0" }}>
                ⚠️ {error}
                <span style={{ marginLeft: 10, color: "var(--text-dim)" }}>自動で取り直しています…</span>
              </div>
            )}
            {loadingRaces && <LoadingNotice what="レース一覧" padding="24px 0" onRetry={() => setReloadKey((n) => n + 1)} />}
            {!loadingRaces && !loadingDates && !error && rows.length === 0 && (
              <div style={{ color: "var(--text-dim)", fontSize: 13, padding: "24px 0" }}>表示できるレースがありません。</div>
            )}

            <div className="races-venues" style={{ ["--cols" as string]: Math.min(Math.max(byVenue.length, 1), 3) }}>
              {byVenue.map(([v, list]) => <VenueBlock key={v} venue={v} rows={list} values={values} bets={betsSummary?.races ?? null} />)}
            </div>
          </div>
        </div>

        {sheetOpen && <CalendarSheet onClose={() => setSheetOpen(false)}>
          <RaceCalendar selectedDate={date} today={today} latestDate={latest} onSelect={choose} onMonthDays={addDates} />
        </CalendarSheet>}
      </PageShell>
    </>
  );
}

/** スマホ用のボトムシート。背景を押す・Esc・閉じるボタンで閉じる */
function CalendarSheet({ onClose, children }: { onClose: () => void; children: React.ReactNode }) {
  useEffect(() => {
    const on = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", on);
    return () => window.removeEventListener("keydown", on);
  }, [onClose]);
  return (
    <div role="dialog" aria-modal="true" aria-label="開催日を選ぶ" style={{ position: "fixed", inset: 0, zIndex: 60, display: "flex", alignItems: "flex-end", background: "rgba(0,0,0,0.55)" }} onClick={onClose}>
      <div onClick={(e) => e.stopPropagation()} style={{ width: "100%", maxHeight: "88vh", overflowY: "auto", background: "var(--bg, #0d1117)", borderRadius: "16px 16px 0 0", padding: "12px 12px calc(16px + env(safe-area-inset-bottom))", borderTop: "1px solid var(--border)" }}>
        <div style={{ display: "flex", alignItems: "center", marginBottom: 8 }}>
          <strong style={{ fontSize: 15 }}>開催日を選ぶ</strong>
          <button onClick={onClose} aria-label="閉じる" style={{ marginLeft: "auto", minWidth: 44, minHeight: 44, borderRadius: 8, border: "1px solid var(--border)", background: "var(--surface)", color: "var(--text)", fontSize: 16, cursor: "pointer" }}>✕</button>
        </div>
        <div style={{ display: "flex", justifyContent: "center" }}>{children}</div>
      </div>
    </div>
  );
}
