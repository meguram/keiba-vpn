"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { PageShell } from "@/components/PageShell";
import { GradeChip, WeekPicker } from "@/components/WeekPicker";
import { USE_MOCK, MOCK_WEEKLY_RACES, getMockRaceDates } from "@/lib/mock";
import { useAuthStatus } from "@/lib/hooks/useAuthStatus";
import { MemberUpgradeModal } from "@/components/upgrade/MemberUpgradeModal";
import { firstDateOf, formatMD, pickTwoWeeks, todayJst, weekdayOf, WEEKDAY_JA } from "@/lib/weekDates";

/* レース一覧（開催日ごと・開催場ごと）。
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
const FUTURE_DAYS = 14;
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

function VenueBlock({ venue, rows }: { venue: string; rows: Row[] }) {
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
          return (
            <li key={r.race_id} style={{ borderBottom: "1px solid var(--border)" }}>
              <Link
                href={`/race/${r.race_id}`}
                title={`${venue} ${r.round}R ${r.race_name}`}
                style={{ display: "grid", gridTemplateColumns: "34px 44px 1fr auto", alignItems: "center", gap: 10, padding: "8px 14px", textDecoration: "none", color: "var(--text)", background: isG ? "rgba(251,191,36,0.06)" : undefined }}
              >
                <span style={{ background: "#3b82f6", color: "#fff", borderRadius: 6, textAlign: "center", fontWeight: 700, fontSize: 13, padding: "3px 0" }}>{r.round}</span>
                <span style={{ fontSize: 12, color: "var(--text-dim)", fontVariantNumeric: "tabular-nums" }}>{r.start_time ?? "—"}</span>
                <span style={{ minWidth: 0 }}>
                  <span style={{ fontSize: 14, fontWeight: isG ? 700 : 600, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", display: "inline-block", maxWidth: "100%", verticalAlign: "middle" }}>{r.race_name}</span>
                  {isG && <span style={{ marginLeft: 6 }}><GradeChip grade={r.grade!} /></span>}
                  {!isG && (r.grade === "OP" || r.grade === "L") && <span style={{ marginLeft: 6, fontSize: 10, color: "var(--text-dim)", border: "1px solid var(--border)", borderRadius: 4, padding: "0 5px" }}>{r.grade}</span>}
                  <span style={{ display: "block", fontSize: 11, color: "var(--text-dim)" }}>
                    {[course(r), r.entries_count ? `${r.entries_count}頭` : ""].filter(Boolean).join(" · ")}
                  </span>
                </span>
                {r.has_result && <span style={{ fontSize: 10, color: "#4ade80", border: "1px solid rgba(74,222,128,0.4)", borderRadius: 4, padding: "1px 6px" }}>結果</span>}
              </Link>
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
  const [date, setDate] = useState("");
  const [rows, setRows] = useState<Row[]>([]);
  const [loadingDates, setLoadingDates] = useState(true);
  const [loadingRaces, setLoadingRaces] = useState(false);
  const [error, setError] = useState("");
  const { isMember, loading: authLoading } = useAuthStatus();
  const today = todayJst();

  /* 開催日の一覧（過去 30 日 + 今日以降 14 日）。初期は『次回の開催』の最初の開催日（?date= があればそれ） */
  useEffect(() => {
    (async () => {
      try {
        let list: string[];
        if (USE_MOCK) {
          list = getMockRaceDates();
        } else {
          const res = await fetch(`/api/scrape-dates?picker_past_days=${PAST_DAYS}&picker_future_days=${FUTURE_DAYS}`, { cache: "no-store" });
          list = res.ok ? ((await res.json()).dates ?? []) : [];
        }
        setDates(list);
        const fromUrl = new URLSearchParams(window.location.search).get("date") ?? "";
        const preferred = process.env.NEXT_PUBLIC_DEFAULT_RACE_DATE;
        const next = pickTwoWeeks(list, today).find((p) => p.role === "next");
        const initial = list.includes(fromUrl) ? fromUrl : preferred && list.includes(preferred) ? preferred : next ? firstDateOf(next.week, today) : (list[0] ?? "");
        setDate(initial);
      } catch {
        setError("開催日の一覧を取得できませんでした。");
      } finally {
        setLoadingDates(false);
      }
    })();
  }, [today]);

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
        const res = await fetch(`/api/race-list/${date}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const d = await res.json();
        if (alive) setRows((d.races ?? d ?? []).map((x: Record<string, unknown>) => toRow(x)));
      } catch (e) {
        if (alive) { setRows([]); setError(`レース一覧を取得できませんでした（${e instanceof Error ? e.message : String(e)}）`); }
      } finally {
        if (alive) setLoadingRaces(false);
      }
    })();
    return () => { alive = false; };
  }, [date]);

  function choose(d: string) {
    setDate(d);
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

  // 現在の 2 つの開催に入らない過去の日付は、ここから選べる
  const shown = useMemo(() => new Set(pickTwoWeeks(dates, today).flatMap((p) => p.week.dates)), [dates, today]);
  const older = dates.filter((d) => !shown.has(d));
  const wd = date ? WEEKDAY_JA[weekdayOf(date)] : "";

  return (
    <>
      {showModal && <MemberUpgradeModal onClose={() => setShowModal(false)} />}
      <PageShell title="レース一覧">
        {!authLoading && !isMember && (
          <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "10px 16px", borderRadius: 8, background: "rgba(59,130,246,0.08)", border: "1px solid rgba(59,130,246,0.25)", marginBottom: 16, fontSize: 13, color: "var(--text)", flexWrap: "wrap" }}>
            <span>🔒 AI予測データは会員限定です。</span>
            <button onClick={() => setShowModal(true)} style={{ background: "none", border: "none", color: "var(--accent)", cursor: "pointer", fontSize: 13, padding: 0, textDecoration: "underline" }}>
              会員登録はこちら →
            </button>
          </div>
        )}

        <div style={{ marginBottom: 14 }}>
          <WeekPicker dates={dates} today={today} selectedDate={date} loading={loadingDates} onSelectDate={choose} />
          {older.length > 0 && (
            <label style={{ display: "inline-flex", alignItems: "center", gap: 8, marginTop: 10, fontSize: 12, color: "var(--text-dim)" }}>
              それ以前の開催日
              <select
                value={older.includes(date) ? date : ""}
                onChange={(e) => e.target.value && choose(e.target.value)}
                style={{ padding: "5px 8px", background: "var(--surface2)", border: "1px solid var(--border)", borderRadius: 6, color: "var(--text)", fontSize: 12 }}
              >
                <option value="">選ぶ…</option>
                {older.map((d) => <option key={d} value={d}>{formatMD(d)}（{WEEKDAY_JA[weekdayOf(d)]}）</option>)}
              </select>
            </label>
          )}
        </div>

        {date && (
          <div style={{ marginBottom: 12, fontSize: 13, color: "var(--text-dim)" }}>
            {formatMD(date)}（{wd}）— {byVenue.map(([v, r]) => `${v} ${r.length}R`).join(" ／ ") || "開催なし"}
          </div>
        )}
        {error && <div role="alert" style={{ color: "var(--err)", fontSize: 13, padding: "10px 0" }}>⚠️ {error}</div>}
        {loadingRaces && <div style={{ color: "var(--text-dim)", fontSize: 13, padding: "24px 0" }}>読み込み中…</div>}
        {!loadingRaces && !loadingDates && !error && rows.length === 0 && (
          <div style={{ color: "var(--text-dim)", fontSize: 13, padding: "24px 0" }}>表示できるレースがありません。</div>
        )}

        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(340px, 1fr))", gap: 14, alignItems: "start" }}>
          {byVenue.map(([v, list]) => <VenueBlock key={v} venue={v} rows={list} />)}
        </div>
      </PageShell>
    </>
  );
}
