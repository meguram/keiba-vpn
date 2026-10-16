"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { USE_MOCK, MOCK_WEEKLY_RACES } from "@/lib/mock";
import {
  COLOR_SUN, GradeRace, PickedWeek, dayKind, firstDateOf, formatMD, formatRange, groupWeeks, meetingKeyOf, pickGradeRaces, pickTwoWeeks, venuesOf,
} from "@/lib/weekDates";

type RaceLite = { race_id: string; race_name?: string; grade?: string; venue?: string; round?: number | string };
type DaySummary = { n: number; venues: string[]; grades: GradeRace[] };

// 開催日ごとの要約（重賞・開催場）を共有。選び直すたびに取り直さない
const summaryCache = new Map<string, Promise<DaySummary>>();

function loadSummary(date: string): Promise<DaySummary> {
  let p = summaryCache.get(date);
  if (!p) {
    p = (async () => {
      let rows: RaceLite[];
      if (USE_MOCK) {
        rows = MOCK_WEEKLY_RACES as RaceLite[];
      } else {
        const res = await fetch(`/api/race-list/${date}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const d = await res.json();
        rows = d.races ?? d ?? [];
      }
      return { n: rows.length, venues: venuesOf(rows), grades: pickGradeRaces(rows) };
    })();
    p.catch(() => summaryCache.delete(date));     // 失敗は覚えない（次に開いたときにやり直す）
    summaryCache.set(date, p);
  }
  return p;
}

/** 週内の開催日の要約を並行取得する。失敗した日は 'error'（日付選択そのものは使える） */
function useDaySummaries(dates: string[]): Record<string, DaySummary | "loading" | "error"> {
  const [state, setState] = useState<Record<string, DaySummary | "loading" | "error">>({});
  const key = dates.join(",");
  useEffect(() => {
    let alive = true;
    for (const d of dates) {
      setState((s) => (s[d] && s[d] !== "error" ? s : { ...s, [d]: "loading" }));
      loadSummary(d).then((v) => alive && setState((s) => ({ ...s, [d]: v })), () => alive && setState((s) => ({ ...s, [d]: "error" })));
    }
    return () => { alive = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  return state;
}

export function GradeChip({ grade }: { grade: string }) {
  const g = grade.replace(".", "");
  const s =
    g === "G1" ? { bg: "rgba(239,68,68,0.18)", color: "#ef4444" } :
    g === "G2" ? { bg: "rgba(59,130,246,0.18)", color: "#60a5fa" } :
    g === "G3" ? { bg: "rgba(34,197,94,0.18)", color: "#22c55e" } :
    { bg: "rgba(167,139,250,0.18)", color: "#a78bfa" };
  return <span style={{ fontSize: 10, fontWeight: 700, padding: "1px 6px", borderRadius: 4, flexShrink: 0, ...s }}>{grade}</span>;
}

const REL_STYLE: Record<string, { bg: string; color: string }> = {
  今週: { bg: "rgba(59,130,246,0.2)", color: "#60a5fa" },
  来週: { bg: "rgba(34,197,94,0.15)", color: "#4ade80" },
  先週: { bg: "rgba(107,125,149,0.18)", color: "var(--text-dim)" },
};

export type WeekPickerProps = {
  dates: string[];                 // 開催日の候補（API から）
  today: string;                   // YYYYMMDD（JST）
  selectedDate: string;
  loading: boolean;
  onSelectDate: (date: string) => void;
};

/**
 * 開催日の選択（netkeiba のカレンダーを参考にした週単位の表示）。
 * 『前回の開催』と『次回の開催』の最大 2 つをタブで切り替え、選んだ開催の日付をカードで並べる。
 * 土=青・日と祝日開催=赤、重賞は G1/G2/G3 のチップとレース名で日付カードに出す。
 */
export function WeekPicker({ dates, today, selectedDate, loading, onSelectDate }: WeekPickerProps) {
  const picked: (PickedWeek & { extra: boolean })[] = useMemo(() => {
    const two = pickTwoWeeks(dates, today);
    // URL の ?date= が 2 つの開催に入らない過去の日付のときは、その開催も 1 つ足して選べるようにする
    if (selectedDate && dates.includes(selectedDate) && !two.some((p) => p.week.dates.includes(selectedDate))) {
      const w = groupWeeks(dates).find((x) => x.dates.includes(selectedDate));
      if (w) two.unshift({ week: w, role: "prev", rel: "" });
      return two.map((p, i) => ({ ...p, extra: i === 0 }));
    }
    return two.map((p) => ({ ...p, extra: false }));
  }, [dates, today, selectedDate]);

  const activeKey = selectedDate ? meetingKeyOf(selectedDate) : "";
  const activeIdx = Math.max(0, picked.findIndex((p) => p.week.dates.includes(selectedDate) || p.week.key === activeKey));
  const active = picked[activeIdx];
  const summaries = useDaySummaries(active ? active.week.dates : []);
  const tabRefs = useRef<(HTMLButtonElement | null)[]>([]);

  if (loading) {
    return <div role="status" aria-label="開催日を読み込み中" style={{ display: "flex", gap: 10 }}>{[0, 1].map((i) => <div key={i} style={{ width: 230, height: 54, borderRadius: 10, background: "var(--surface2)", opacity: 0.6 }} />)}</div>;
  }
  if (!picked.length) return <span style={{ fontSize: 13, color: "var(--text-dim)" }}>開催日のデータがありません</span>;

  function selectWeek(i: number) {
    const p = picked[i];
    if (p) onSelectDate(firstDateOf(p.week, today));
    tabRefs.current[i]?.focus();
  }

  return (
    <div style={{ width: "100%" }}>
      {/* 開催の切替（前回 / 次回） */}
      <div role="tablist" aria-label="開催週" style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 10 }}>
        {picked.map((p, i) => {
          const on = i === activeIdx;
          const title = p.extra ? "指定の開催" : p.role === "prev" ? "前回の開催" : "次回の開催";
          const rel = REL_STYLE[p.rel];
          return (
            <button
              key={p.week.key}
              ref={(el) => { tabRefs.current[i] = el; }}
              role="tab"
              aria-selected={on}
              tabIndex={on ? 0 : -1}
              onClick={() => selectWeek(i)}
              onKeyDown={(e) => {
                if (e.key === "ArrowRight") { e.preventDefault(); selectWeek(Math.min(picked.length - 1, i + 1)); }
                if (e.key === "ArrowLeft") { e.preventDefault(); selectWeek(Math.max(0, i - 1)); }
              }}
              style={{
                display: "flex", alignItems: "center", gap: 8, padding: "8px 14px", borderRadius: 10, cursor: "pointer", textAlign: "left",
                border: `1px solid ${on ? "var(--accent)" : "var(--border)"}`, background: on ? "rgba(59,130,246,0.15)" : "var(--surface)",
                color: on ? "var(--text)" : "var(--text-dim)",
              }}
            >
              {p.role === "prev" && i === 0 && <span aria-hidden>◀</span>}
              <span style={{ display: "flex", flexDirection: "column", lineHeight: 1.3 }}>
                <span style={{ fontSize: 11, display: "flex", alignItems: "center", gap: 6 }}>
                  {title}
                  {rel && <span style={{ fontSize: 10, fontWeight: 700, padding: "0 6px", borderRadius: 4, ...rel }}>{p.rel}</span>}
                </span>
                <span style={{ fontSize: 14, fontWeight: on ? 700 : 500 }}>{formatRange(p.week.dates)}</span>
              </span>
              {p.role === "next" && <span aria-hidden>▶</span>}
            </button>
          );
        })}
      </div>

      {/* 選んだ開催の日付カード */}
      <div role="group" aria-label="開催日" style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
        {active.week.dates.map((d) => {
          const k = dayKind(d, today);
          const on = d === selectedDate;
          const sm = summaries[d];
          const grades = sm && typeof sm === "object" ? sm.grades : [];
          return (
            <button
              key={d}
              aria-pressed={on}
              onClick={() => onSelectDate(d)}
              title={`${formatMD(d)}(${k.label})${k.holiday ? " 祝日開催" : ""}`}
              style={{
                flex: "1 1 230px", maxWidth: 360, minWidth: 200, textAlign: "left", cursor: "pointer", padding: "10px 12px 10px",
                borderRadius: 10, color: "var(--text)",
                borderStyle: "solid", borderWidth: "4px 1px 1px 1px",       // 省略形(border)と個別指定の併用は React が警告するので、まとめて個別指定にする
                borderColor: `${k.color} ${on ? "var(--accent)" : "var(--border)"} ${on ? "var(--accent)" : "var(--border)"} ${on ? "var(--accent)" : "var(--border)"}`,
                background: on ? "rgba(59,130,246,0.15)" : "var(--surface)",
                boxShadow: on ? "0 0 0 1px var(--accent)" : undefined,
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
                <span style={{ fontSize: 16, fontWeight: 800, color: k.color }}>{formatMD(d)}<span style={{ fontSize: 13, marginLeft: 2 }}>({k.label})</span></span>
                {k.holiday && <span style={{ fontSize: 10, fontWeight: 700, padding: "0 6px", borderRadius: 4, background: "rgba(248,113,113,0.18)", color: COLOR_SUN }}>祝</span>}
                {k.isToday && <span style={{ fontSize: 10, fontWeight: 700, padding: "0 6px", borderRadius: 4, background: "rgba(250,204,21,0.18)", color: "#facc15" }}>今日</span>}
                {on && <span aria-hidden style={{ marginLeft: "auto", fontSize: 11, color: "var(--accent)" }}>表示中</span>}
              </div>

              {sm === "loading" || sm === undefined ? (
                <div aria-hidden style={{ marginTop: 8 }}>
                  <div style={{ height: 11, width: "60%", borderRadius: 4, background: "var(--surface2)", marginBottom: 6 }} />
                  <div style={{ height: 11, width: "85%", borderRadius: 4, background: "var(--surface2)" }} />
                </div>
              ) : sm === "error" ? (
                <div style={{ marginTop: 6, fontSize: 11, color: "var(--text-dim)" }}>レース情報を取得できませんでした（選択は可能）</div>
              ) : (
                <>
                  <div style={{ marginTop: 4, fontSize: 12, color: "var(--text-dim)" }}>{sm.venues.join("・") || "—"} ／ {sm.n}R</div>
                  {grades.length > 0 ? (
                    <ul style={{ listStyle: "none", margin: "6px 0 0", padding: 0, display: "flex", flexDirection: "column", gap: 3 }}>
                      {grades.slice(0, 4).map((g) => (
                        <li key={g.race_id} title={`${g.venue} ${g.round}R ${g.fullName}`} style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12, minWidth: 0 }}>
                          <GradeChip grade={g.grade} />
                          <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", color: "var(--text)" }}>{g.name}</span>
                          <span style={{ marginLeft: "auto", fontSize: 10, color: "var(--text-dim)", flexShrink: 0 }}>{g.venue}{g.round}R</span>
                        </li>
                      ))}
                      {grades.length > 4 && <li style={{ fontSize: 11, color: "var(--text-dim)" }}>ほか重賞 {grades.length - 4} 件</li>}
                    </ul>
                  ) : (
                    <div style={{ marginTop: 6, fontSize: 11, color: "var(--text-dim)" }}>重賞なし</div>
                  )}
                </>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}

