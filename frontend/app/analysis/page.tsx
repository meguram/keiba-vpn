"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { fetchRaceList, getJson, prefetchJson } from "@/lib/httpCache";
import { LoadingNotice } from "@/components/LoadingNotice";
import { useAutoRetry } from "@/lib/hooks/useAutoRetry";
import { WeekPicker } from "@/components/WeekPicker";
import { RaceStrip } from "@/components/RaceStrip";
import { AnalysisTable } from "@/components/analysis/AnalysisTable";
import { RunningMap } from "@/components/analysis/RunningMap";
import { entriesUnavailable, formatComputedAt, formatPeriod } from "@/lib/raceAnalysis";
import type { RaceAnalysis } from "@/lib/raceAnalysis";
import { firstDateOf, meetingKeyOf, pickTwoWeeks, todayJst, venuesOf } from "@/lib/weekDates";

// 分析の表は列が多いので、広い画面では横幅を使い切って横スクロールなしで一面に収める
const PAGE_MAX = 1680;

type RaceItem = { race_id: string; race_name?: string; venue?: string; round?: number | string; date?: string; entries_phase?: string | null };

const NO_VALUES = new Map();
const analysisUrl = (raceId: string) => `/api/v1/races/${raceId}/analysis`;

type Load = { state: "idle" | "loading" | "ok" | "error"; data?: RaceAnalysis; status?: number; message?: string };

export default function AnalysisPage() {
  const today = useMemo(() => todayJst(), []);
  const [dates, setDates] = useState<string[]>([]);
  const [selectedDate, setSelectedDate] = useState("");
  const [races, setRaces] = useState<RaceItem[]>([]);
  const [selectedVenue, setSelectedVenue] = useState("");
  const [selectedRaceId, setSelectedRaceId] = useState("");
  const [selectedModule, setSelectedModule] = useState("");
  const [loadingDates, setLoadingDates] = useState(true);
  const [loadingRaces, setLoadingRaces] = useState(false);
  const [error, setError] = useState("");
  const [reload, setReload] = useState(0);
  const [load, setLoad] = useState<Load>({ state: "idle" });

  /** URL の ?date= ?venue= ?race= ?m= を書き換える（再読込・共有しても同じ表示に戻る） */
  function syncUrl(patch: Record<string, string | null>) {
    const u = new URL(window.location.href);
    for (const [k, v] of Object.entries(patch)) {
      if (v == null || v === "") u.searchParams.delete(k); else u.searchParams.set(k, v);
    }
    window.history.replaceState(null, "", u.toString());
  }

  const venues = useMemo(() => venuesOf(races), [races]);
  const activeVenue = venues.includes(selectedVenue) ? selectedVenue : (venues[0] ?? "");
  const visibleRaces = useMemo(
    () => races.filter((r) => !activeVenue || r.venue === activeVenue).sort((a, b) => Number(a.round ?? 0) - Number(b.round ?? 0)),
    [races, activeVenue],
  );
  /* 未来の開催日で出馬表がまだ取得できていないレースは選べない（JRA の公示前） */
  const disabledIds = useMemo(
    () => new Set(races.filter((r) => entriesUnavailable(r, selectedDate, today)).map((r) => r.race_id)),
    [races, selectedDate, today],
  );
  const selectable = useMemo(() => visibleRaces.filter((r) => !disabledIds.has(r.race_id)), [visibleRaces, disabledIds]);
  const activeRace = useMemo(() => selectable.find((r) => r.race_id === selectedRaceId) ?? selectable[0], [selectable, selectedRaceId]);
  const activeRaceId = activeRace?.race_id ?? "";

  useEffect(() => {
    const q = new URLSearchParams(window.location.search);
    if (q.get("venue")) setSelectedVenue(q.get("venue")!);
    if (q.get("race")) setSelectedRaceId(q.get("race")!);
    if (q.get("m")) setSelectedModule(q.get("m")!);
  }, []);

  /* 開催日一覧 */
  useEffect(() => {
    setError("");
    (async () => {
      try {
        const days = Number(process.env.NEXT_PUBLIC_PICKER_PAST_DAYS) || 30;
        const res = await fetch(`/api/scrape-dates?picker_past_days=${days}&picker_future_days=14`, { cache: "no-store" });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const d = await res.json();
        const list: string[] = d.dates ?? d ?? [];
        setDates(list);
        if (list.length) {
          const fromUrl = new URLSearchParams(window.location.search).get("date") ?? undefined;
          const next = pickTwoWeeks(list, today).find((w) => w.role === "next");
          setSelectedDate(fromUrl && list.includes(fromUrl) ? fromUrl : next ? firstDateOf(next.week, today) : list[0]);
        }
      } catch (e: unknown) {
        setError(`開催日の一覧を取得できませんでした（${e instanceof Error ? e.message : String(e)}）`);
      } finally {
        setLoadingDates(false);
      }
    })();
  }, [reload, today]);

  /* レース一覧（日付変更時） */
  const racesSeq = useRef(0);
  const loadRaces = useCallback(async (date: string) => {
    if (!date) return;
    const seq = ++racesSeq.current;      // 日付を素早く切り替えたとき、遅れて届いた古い一覧で上書きしない
    setLoadingRaces(true);
    setRaces([]);
    try {
      const d = (await fetchRaceList(date)) as { races?: RaceItem[] } | RaceItem[];
      if (seq !== racesSeq.current) return;
      setRaces((Array.isArray(d) ? d : d.races) ?? []);
    } catch (e: unknown) {
      if (seq === racesSeq.current) setError(`レース一覧を取得できませんでした（${e instanceof Error ? e.message : String(e)}）`);
    } finally {
      if (seq === racesSeq.current) setLoadingRaces(false);
    }
  }, []);
  useEffect(() => { if (selectedDate) void loadRaces(selectedDate); }, [selectedDate, loadRaces]);
  useAutoRetry(!!error, () => { setLoadingDates(true); setReload((n) => n + 1); }, 8000);
  useAutoRetry(load.state === "error" && load.status !== 401, () => setReload((n) => n + 1), 8000);

  /* 既定のレースが選ばれたときも URL に入れる（再読み込み・共有で同じレースが開く） */
  useEffect(() => {
    if (!activeRace) return;
    if (new URLSearchParams(window.location.search).get("race") !== activeRace.race_id) {
      syncUrl({ race: activeRace.race_id, ...(activeRace.venue ? { venue: activeRace.venue } : {}) });
    }
  }, [activeRace]);

  /* 選んだレースの分析。素早く切り替えたときは、最後に選んだレースの応答だけを使う */
  useEffect(() => {
    if (!activeRaceId) { setLoad({ state: "idle" }); return; }
    let cancelled = false;
    setLoad({ state: "loading" });
    getJson<RaceAnalysis>(analysisUrl(activeRaceId), { ttlMs: 60_000, persist: true })
      .then((data) => { if (!cancelled) setLoad({ state: "ok", data }); })
      .catch((e: Error & { status?: number }) => { if (!cancelled) setLoad({ state: "error", status: e.status, message: e.message }); });
    return () => { cancelled = true; };
  }, [activeRaceId, reload]);

  function chooseDate(d: string) {
    setSelectedDate(d);
    setSelectedRaceId("");
    syncUrl({ date: d, race: null });
  }
  function chooseVenue(v: string) {
    const first = races.filter((r) => r.venue === v && !disabledIds.has(r.race_id)).sort((a, b) => Number(a.round ?? 0) - Number(b.round ?? 0))[0];
    setSelectedVenue(v);
    setSelectedRaceId(first?.race_id ?? "");
    syncUrl({ venue: v, race: first?.race_id ?? null });
  }
  function chooseRace(raceId: string) {
    const r = races.find((x) => x.race_id === raceId);
    if (r?.venue && r.venue !== activeVenue) setSelectedVenue(r.venue);
    setSelectedRaceId(raceId);
    syncUrl({ race: raceId, ...(r?.venue ? { venue: r.venue } : {}) });
  }
  const warm = useCallback((id: string) => prefetchJson(analysisUrl(id), { ttlMs: 60_000, persist: true }), []);

  const data = load.state === "ok" ? load.data : undefined;
  const modules = data?.modules ?? [];
  const mod = modules.find((m) => m.id === selectedModule) ?? modules.find((m) => m.status === "ok") ?? modules[0];
  const race = data?.race;

  return (
    <div className="page-wide" style={{ minHeight: "100dvh", background: "var(--bg)", color: "var(--text)" }}>
      <div className="an-pad" style={{ background: "var(--surface)", borderBottom: "1px solid var(--border)", padding: "18px 24px" }}>
        <div style={{ maxWidth: PAGE_MAX, margin: "0 auto", display: "flex", alignItems: "center", gap: 12 }}>
          <h1 style={{ fontSize: 18, fontWeight: 700, color: "#f0f6fc", margin: 0 }}>🔎 めぐ分析</h1>
          <p className="hide-narrow" style={{ fontSize: 12, color: "var(--text-dim)", margin: 0 }}>
            開催日・競馬場・レースを選ぶと、出馬表の馬ごとに分析結果を表示します
          </p>
        </div>
      </div>

      <div className="an-pad" style={{ background: "var(--surface)", borderBottom: "1px solid var(--border)", padding: "14px 24px" }}>
        <div style={{ maxWidth: PAGE_MAX, margin: "0 auto" }}>
          <WeekPicker dates={dates} today={today} selectedDate={selectedDate} loading={loadingDates} onSelectDate={chooseDate} />
        </div>
      </div>

      <div className="an-pad" style={{ maxWidth: PAGE_MAX, margin: "0 auto", padding: "20px 24px" }}>
        {error && (
          <div role="alert" style={{ background: "rgba(239,68,68,0.1)", border: "1px solid rgba(239,68,68,0.3)", borderRadius: 8, padding: "14px 18px", color: "var(--err)", fontSize: 13, marginBottom: 16, display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
            <span>⚠️ {error}</span>
            <span style={{ marginLeft: "auto", color: "var(--text-dim)" }}>自動で取り直しています…</span>
          </div>
        )}

        {loadingRaces && <LoadingNotice what="レース一覧" size={36} padding="60px 0" onRetry={() => loadRaces(selectedDate)} />}

        {!loadingRaces && !error && races.length === 0 && selectedDate && (
          <div style={{ textAlign: "center", padding: "60px 0", color: "var(--text-dim)" }}>
            <div style={{ fontSize: 40, marginBottom: 12 }}>📭</div>
            <p>この日程のレース情報がありません</p>
          </div>
        )}

        {!loadingRaces && races.length > 0 && (
          <>
            <RaceStrip
              races={races}
              venue={activeVenue}
              onVenue={chooseVenue}
              values={NO_VALUES}
              selectedRaceId={activeRaceId}
              onSelectRace={chooseRace}
              onWarm={warm}
              showValue={false}
              disabledIds={disabledIds}
              disabledHint="出馬表がまだ公示されていません（JRA の公示後に選べます）"
            />

            {selectable.length === 0 && (
              <div role="note" className="card" style={{ color: "var(--text-dim)", fontSize: 13 }}>
                {activeVenue}のレースは、出馬表がまだ公示されていません。JRA の公示（特別登録・木曜の出走馬決定）の後に選べるようになります。
              </div>
            )}

            {load.state === "loading" && <LoadingNotice what="分析結果" size={32} padding="40px 0" />}

            {load.state === "error" && (
              <div role="alert" className="card" style={{ color: "var(--text-dim)", fontSize: 13 }}>
                {load.status === 401 ? (
                  <>めぐ分析は会員限定です。<Link href="/login" style={{ color: "var(--accent)" }}>ログイン</Link>してください。</>
                ) : (
                  <>分析結果を取得できませんでした（{load.message}）。<span style={{ marginLeft: 8 }}>自動で取り直しています…</span></>
                )}
              </div>
            )}

            {data && race && (
              <section>
                <div style={{ marginBottom: 10 }}>
                  <div style={{ fontSize: 16, fontWeight: 700, color: "#f0f6fc" }}>
                    {race.venue ?? activeRace?.venue} {race.round ?? activeRace?.round}R {race.race_name ?? activeRace?.race_name}
                  </div>
                  <div style={{ fontSize: 12, color: "var(--text-dim)", marginTop: 2 }}>
                    {[race.date, race.surface && race.distance ? `${race.surface}${race.distance}m` : null, race.grade, race.start_time ? `${race.start_time} 発走` : null, data.entries.length ? `${data.entries.length}頭` : null, formatComputedAt(data.computed_at) ? `分析の集計 ${formatComputedAt(data.computed_at)}` : null].filter(Boolean).join(" ・ ")}
                  </div>
                </div>

                {data.provisional_entries && (
                  <div role="note" style={{ marginBottom: 10, padding: "8px 12px", borderRadius: 8, border: "1px solid rgba(210,153,34,0.4)", background: "rgba(210,153,34,0.08)", color: "#d29922", fontSize: 12 }}>
                    暫定の出馬表です。馬番・枠順・騎手は確定前で、変わることがあります。
                  </div>
                )}

                {data.status === "pending" && (
                  <div className="card" style={{ color: "var(--text-dim)", fontSize: 13 }}>{data.message ?? "このレースの分析はまだ用意できていません。"}</div>
                )}

                {data.status === "no_entries" && (
                  <div className="card" style={{ color: "var(--text-dim)", fontSize: 13 }}>{data.message ?? "このレースの出馬表はまだ取得できていません。"}</div>
                )}

                {data.status === "ok" && modules.length > 0 && (
                  <>
                    {modules.length > 1 && (
                      <div role="tablist" aria-label="分析の種類" style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 8 }}>
                        {modules.map((m) => (
                          <button
                            key={m.id}
                            role="tab"
                            aria-selected={mod?.id === m.id}
                            onClick={() => { setSelectedModule(m.id); syncUrl({ m: m.id }); }}
                            style={{
                              minHeight: 36, padding: "0 14px", borderRadius: 999, fontSize: 13, cursor: "pointer",
                              border: `1px solid ${mod?.id === m.id ? "var(--accent)" : "var(--border)"}`,
                              background: mod?.id === m.id ? "rgba(59,130,246,0.15)" : "var(--surface)",
                              color: mod?.id === m.id ? "var(--accent)" : "var(--text-dim)",
                            }}
                          >
                            {m.label}
                          </button>
                        ))}
                      </div>
                    )}

                    {mod && (
                      <div className="card">
                        <div style={{ display: "flex", alignItems: "baseline", gap: 10, flexWrap: "wrap" }}>
                          <h2 style={{ fontSize: 15, fontWeight: 700, margin: 0 }}>{mod.label}</h2>
                          <span style={{ fontSize: 12, color: "var(--text-dim)" }}>{mod.description}</span>
                        </div>
                        {mod.status === "ok" && (mod.view === "map"
                          ? (mod.figure ? <RunningMap figure={mod.figure} entries={data.entries} /> : null)
                          : <AnalysisTable entries={data.entries} module={mod} />)}
                        {mod.status === "not_applicable" && <p style={{ fontSize: 13, color: "var(--text-dim)", marginTop: 8 }}>このレースでは表示できません（コース条件や元データが未確定です）。</p>}
                        {mod.status === "error" && <p style={{ fontSize: 13, color: "var(--err)", marginTop: 8 }}>この分析の計算に失敗しました。</p>}
                        {mod.status === "ok" && mod.source && formatPeriod(mod.source) && (
                          <p style={{ margin: "10px 0 0", fontSize: 12, color: "var(--text)" }}>{formatPeriod(mod.source)}</p>
                        )}
                        {mod.status === "ok" && mod.notes?.length ? (
                          <ul style={{ margin: "6px 0 0", paddingLeft: 18, fontSize: 11, color: "var(--text-dim)" }}>
                            {mod.notes?.map((n) => <li key={n}>{n}</li>)}
                          </ul>
                        ) : null}
                      </div>
                    )}
                  </>
                )}
              </section>
            )}
          </>
        )}
      </div>
    </div>
  );
}
