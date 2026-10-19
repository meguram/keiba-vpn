"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { USE_MOCK, getMockRaceDates } from "@/lib/mock";
import { COLOR_SAT, COLOR_SUN, WEEKDAY_JA, dayKind, formatMDW } from "@/lib/weekDates";
import { CalDay, clampMonth, monthGrid, monthsOfYear, shiftMonth, yearsBetween, ymOf } from "@/lib/raceCalendar";
import { GradeChip } from "@/components/WeekPicker";

type MonthData = { days: CalDay[]; min: string | null; max: string | null };

// 月ごとの結果を共有（月を行き来するたびに取り直さない）。失敗は覚えない
const monthCache = new Map<string, Promise<MonthData>>();

function loadMonth(ym: string): Promise<MonthData> {
  let p = monthCache.get(ym);
  if (!p) {
    p = (async () => {
      if (USE_MOCK) {
        const days = getMockRaceDates().filter((d) => d.startsWith(ym)).map((date) => ({ date, n: 12, venues: [], grades: [] }));
        return { days, min: "20200101", max: null };
      }
      const res = await fetch(`/api/race-calendar?month=${ym}`, { cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const d = await res.json();
      return { days: d.days ?? [], min: d.range?.min ?? null, max: d.range?.max ?? null };
    })();
    p.catch(() => monthCache.delete(ym));
    monthCache.set(ym, p);
  }
  return p;
}

// 開催場の 1 文字略称（中山=中・中京=名で区別する）
const VENUE_SHORT: Record<string, string> = { 札幌: "札", 函館: "函", 福島: "福", 新潟: "新", 東京: "東", 中山: "中", 中京: "名", 京都: "京", 阪神: "阪", 小倉: "小" };

const GRADE_RANK: Record<string, number> = { G1: 0, "J.G1": 0, G2: 1, "J.G2": 1, G3: 2, "J.G3": 2 };

export type RaceCalendarProps = {
  selectedDate: string;                // YYYYMMDD（空なら未選択）
  today: string;                       // YYYYMMDD（JST）。これより後の日付は選べない
  latestDate: string;                  // 『直近の開催日』ボタンの行き先（今日以前で最新）
  onSelect: (date: string) => void;
  onMonthDays?: (dates: string[]) => void;   // 読み込んだ月の開催日（週ピッカーが古い開催も表示できるように）
};

/** 過去の開催日を選ぶ月カレンダー。開催日だけ押せる。年・月の直接ジャンプと月送り（今月より先へは進めない）つき */
export function RaceCalendar({ selectedDate, today, latestDate, onSelect, onMonthDays }: RaceCalendarProps) {
  const maxYm = ymOf(today);
  const [minYm, setMinYm] = useState("202001");
  const [viewYm, setViewYm] = useState(() => clampMonth(ymOf(selectedDate || latestDate || today), "202001", maxYm));
  const [data, setData] = useState<MonthData | null>(null);
  const [error, setError] = useState(false);
  const [reload, setReload] = useState(0);
  const [focusDate, setFocusDate] = useState("");
  const gridRef = useRef<HTMLDivElement>(null);

  // 外から選択日が変わったら（週ピッカー・URL）その月を表示する
  useEffect(() => {
    if (selectedDate) setViewYm((v) => (ymOf(selectedDate) === v ? v : clampMonth(ymOf(selectedDate), minYm, maxYm)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedDate]);

  useEffect(() => {
    let alive = true;
    setData(null);
    setError(false);
    loadMonth(viewYm).then(
      (d) => {
        if (!alive) return;
        setData(d);
        if (d.min) setMinYm(ymOf(d.min));
        onMonthDays?.(d.days.map((x) => x.date));
      },
      () => alive && setError(true),
    );
    return () => { alive = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [viewYm, reload]);

  const go = useCallback((ym: string) => setViewYm(clampMonth(ym, minYm, maxYm)), [minYm, maxYm]);
  const byDate = new Map((data?.days ?? []).map((d) => [d.date, d]));
  const year = Number(viewYm.slice(0, 4));
  const month = Number(viewYm.slice(4, 6));
  const focus = byDate.get(focusDate || selectedDate) ?? null;

  function moveFocus(from: string, step: number) {
    const buttons = Array.from(gridRef.current?.querySelectorAll<HTMLButtonElement>("button[data-date]") ?? []);
    const i = buttons.findIndex((b) => b.dataset.date === from);
    const next = buttons[Math.max(0, Math.min(buttons.length - 1, i + step))];
    next?.focus();
  }

  const nav: React.CSSProperties = {
    minWidth: 44, minHeight: 44, borderRadius: 8, border: "1px solid var(--border)", background: "var(--surface)", color: "var(--text)", fontSize: 16, cursor: "pointer",
  };
  const sel: React.CSSProperties = {
    minHeight: 44, padding: "0 8px", borderRadius: 8, border: "1px solid var(--border)", background: "var(--surface2)", color: "var(--text)", fontSize: 14,
  };

  return (
    <div style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 12, padding: 12, width: "100%", maxWidth: 380 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 8 }}>
        <button aria-label="前の月" disabled={viewYm <= minYm} onClick={() => go(shiftMonth(viewYm, -1))} style={{ ...nav, opacity: viewYm <= minYm ? 0.35 : 1 }}>◀</button>
        <select aria-label="年" value={year} onChange={(e) => go(`${e.target.value}${String(month).padStart(2, "0")}`)} style={{ ...sel, flex: 1 }}>
          {yearsBetween(minYm, maxYm).map((y) => <option key={y} value={y}>{y}年</option>)}
        </select>
        <select aria-label="月" value={month} onChange={(e) => go(`${year}${String(e.target.value).padStart(2, "0")}`)} style={{ ...sel, flex: 1 }}>
          {monthsOfYear(year, minYm, maxYm).map((m) => <option key={m} value={m}>{m}月</option>)}
        </select>
        <button aria-label="次の月" disabled={viewYm >= maxYm} onClick={() => go(shiftMonth(viewYm, 1))} style={{ ...nav, opacity: viewYm >= maxYm ? 0.35 : 1 }}>▶</button>
      </div>

      <button
        onClick={() => { if (latestDate) { go(ymOf(latestDate)); onSelect(latestDate); } }}
        disabled={!latestDate}
        style={{ width: "100%", minHeight: 44, marginBottom: 8, borderRadius: 8, border: "1px solid var(--accent)", background: "rgba(59,130,246,0.12)", color: "var(--accent)", fontSize: 13, fontWeight: 700, cursor: "pointer" }}
      >
        直近の開催日{latestDate ? `（${formatMDW(latestDate)}）` : ""}
      </button>

      <div role="grid" aria-label={`${year}年${month}月の開催日`} ref={gridRef}>
        <div role="row" style={{ display: "grid", gridTemplateColumns: "repeat(7, 1fr)", gap: 3, marginBottom: 3 }}>
          {WEEKDAY_JA.map((w, i) => (
            <div key={w} role="columnheader" style={{ textAlign: "center", fontSize: 11, fontWeight: 700, color: i === 0 ? COLOR_SUN : i === 6 ? COLOR_SAT : "var(--text-dim)" }}>{w}</div>
          ))}
        </div>

        {error ? (
          <div role="alert" style={{ padding: "18px 8px", textAlign: "center", fontSize: 13, color: "var(--err)" }}>
            カレンダーを取得できませんでした。
            <div><button onClick={() => setReload((n) => n + 1)} style={{ marginTop: 8, minHeight: 44, padding: "0 16px", borderRadius: 8, border: "1px solid var(--border)", background: "var(--surface2)", color: "var(--text)", cursor: "pointer" }}>再読み込み</button></div>
          </div>
        ) : (
          monthGrid(viewYm).map((week, wi) => (
            <div key={wi} role="row" style={{ display: "grid", gridTemplateColumns: "repeat(7, 1fr)", gap: 3, marginBottom: 3 }}>
              {week.map((d, ci) => {
                if (!d) return <div key={ci} role="gridcell" aria-hidden />;
                const day = byDate.get(d);
                const k = dayKind(d, today);
                const dayColor = ci === 0 ? COLOR_SUN : ci === 6 ? COLOR_SAT : day ? COLOR_SUN : "var(--text-dim)";   // 平日に開催がある日は祝日開催
                const on = d === selectedDate;
                const top = day?.grades.length ? [...day.grades].sort((a, b) => (GRADE_RANK[a.grade] ?? 9) - (GRADE_RANK[b.grade] ?? 9))[0] : null;
                const base: React.CSSProperties = {
                  minHeight: 52, borderRadius: 8, padding: "3px 0", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 1,
                  border: `1px solid ${on ? "var(--accent)" : "var(--border)"}`,
                };
                if (!day) {
                  return (
                    <div key={ci} role="gridcell" aria-disabled="true" aria-label={`${formatMDW(d)} 開催なし`} style={{ ...base, border: "1px solid transparent", opacity: data ? 0.3 : 0.15, color: "var(--text-dim)" }}>
                      <span style={{ fontSize: 13 }}>{Number(d.slice(6))}</span>
                    </div>
                  );
                }
                return (
                  <div key={ci} role="gridcell">
                    <button
                      data-date={d}
                      aria-pressed={on}
                      aria-label={`${formatMDW(d)} ${day.venues.join("・")} ${day.n}R${day.grades.length ? ` 重賞${day.grades.length}件` : ""}`}
                      onClick={() => onSelect(d)}
                      onFocus={() => setFocusDate(d)}
                      onMouseEnter={() => setFocusDate(d)}
                      onKeyDown={(e) => {
                        if (e.key === "ArrowRight" || e.key === "ArrowDown") { e.preventDefault(); moveFocus(d, 1); }
                        if (e.key === "ArrowLeft" || e.key === "ArrowUp") { e.preventDefault(); moveFocus(d, -1); }
                      }}
                      style={{
                        ...base, width: "100%", cursor: "pointer", color: "var(--text)",
                        background: on ? "rgba(59,130,246,0.22)" : "var(--surface2)",
                        boxShadow: on ? "0 0 0 1px var(--accent)" : k.isToday ? "0 0 0 1px #facc15" : undefined,
                      }}
                    >
                      <span style={{ fontSize: 14, fontWeight: 800, color: dayColor }}>{Number(d.slice(6))}</span>
                      <span style={{ fontSize: 9, color: "var(--text-dim)", lineHeight: 1 }}>{day.venues.map((v) => VENUE_SHORT[v] ?? v[0]).join("・")}</span>
                      {top ? <GradeChip grade={top.grade} /> : <span style={{ height: 14 }} aria-hidden />}
                    </button>
                  </div>
                );
              })}
            </div>
          ))
        )}
        {!data && !error && <div role="status" style={{ textAlign: "center", fontSize: 12, color: "var(--text-dim)", padding: "4px 0" }}>読み込み中…</div>}
      </div>

      {/* 日付の要約（スマホではホバーできないので、押した日・選択中の日をここに出す） */}
      <div aria-live="polite" style={{ marginTop: 8, minHeight: 40, fontSize: 12, color: "var(--text-dim)" }}>
        {focus ? (
          <>
            <div style={{ color: "var(--text)", fontWeight: 700 }}>{formatMDW(focus.date)} {focus.venues.join("・")} {focus.n}R</div>
            {focus.grades.length > 0
              ? <ul style={{ listStyle: "none", margin: "4px 0 0", padding: 0, display: "flex", flexDirection: "column", gap: 3 }}>
                  {focus.grades.slice(0, 4).map((g) => (
                    <li key={g.race_id} style={{ display: "flex", alignItems: "center", gap: 6 }}>
                      <GradeChip grade={g.grade} /><span style={{ color: "var(--text)" }}>{g.name}</span><span style={{ marginLeft: "auto", fontSize: 10 }}>{g.venue}{g.round}R</span>
                    </li>
                  ))}
                  {focus.grades.length > 4 && <li>ほか重賞 {focus.grades.length - 4} 件</li>}
                </ul>
              : <div>重賞なし</div>}
          </>
        ) : data && data.days.length === 0 ? <span>この月の開催日はありません。</span> : <span>色のついた日が開催日です。押すとその日のレースを表示します。</span>}
      </div>
    </div>
  );
}
