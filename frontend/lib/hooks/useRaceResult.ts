"use client";
import { useEffect, useState } from "react";
import { getJson } from "@/lib/httpCache";
import { todayJst } from "@/lib/weekDates";

export type ResultFinish = {
  pos: number; horse_number: number | null; bracket_number: number | null; horse_name: string; horse_id: string; sex_age: string;
  jockey_name: string; finish_time: string; margin: string; passing_order: string; last_3f: number | null;
  odds: number | null; popularity: number | null; weight: number | null; weight_change: number | string | null;
};
export type ResultPayoff = { bet_label: string; rows: { numbers: string; payout: number; popularity: number | null }[] };
export type RaceResultData = {
  available: boolean; race_id: string; status?: "final" | "provisional"; official?: boolean;
  race_name?: string; date?: string; venue?: string; surface?: string; distance?: number | null; weather?: string; track_condition?: string; start_time?: string;
  field_size?: number | null; finish?: ResultFinish[]; payoff?: ResultPayoff[]; fetched_at?: string;
  /** 確定のレースだけ・会員と開発者だけ。ホストで事前に計算した値（src/pipeline/race_time/levels.py） */
  race_level?: RaceLevel;
  /** 確定のレースだけ・会員と開発者だけ。その日・その場・その面のトラックバイアス（ホストで事前に計算した値。src/pipeline/track_bias/view.py） */
  track_bias?: TrackBias;
};
/** トラックバイアスの 1〜3 着の位置（番手は 番手/頭数 の分子。通過順が無ければ null） */
export type TrackBiasTop = { pos: number; horse_number: number; popularity: number | null; passing_order: string | null; early: number | null; corner4: number | null; field: number };
/** トラックバイアス（その日の結果の振り返り）。値が無ければ available=false で「算出前」 */
export type TrackBias = {
  available: boolean; note?: string; version?: string; built?: string; formula?: string;
  day?: { venue?: string; surf?: string; label?: string; front_label?: string; inner_label?: string; z_front: number | null; z_inner: number | null; n_races: number | null };
  race?: { front_se: number | null; front_zc: number | null; inner_se: number | null; inner_zc: number | null; front4c_se: number | null };
  top3?: TrackBiasTop[];
};
/** レースレベル・開催日レベル・メンバーレベル（秒。負なら基準より速い）。値が無ければ available=false で「算出なし」 */
export type RaceLevel = {
  available: boolean; note?: string; version?: string; built?: string; formula?: string;
  race?: { sec: number | null; raw_sec: number | null; word: string | null; pace: string | null };
  day?: { sec: number | null; word: string | null; venue?: string; surf?: string; n_races?: number | null };
  member?: { sec: number | null; word: string | null };
};

/** 速報は発走 15 分後に取得されるので、発走後 1 分おきに見に行く（確定の払戻まで揃ったら止める）。この時間を過ぎたら追いかけない。 */
const POLL_MS = 60_000;
const POLL_WINDOW_MS = 4 * 3600_000;

/** 発走時刻（JST）のミリ秒。日付か時刻が読めなければ null */
export function startMsOf(date?: string, startTime?: string): number | null {
  const d = (date ?? "").replace(/\D/g, "");
  const m = /^(\d{1,2}):(\d{2})/.exec(startTime ?? "");
  if (d.length !== 8 || !m) return null;
  const t = Date.parse(`${d.slice(0, 4)}-${d.slice(4, 6)}-${d.slice(6, 8)}T${m[1].padStart(2, "0")}:${m[2]}:00+09:00`);
  return Number.isNaN(t) ? null : t;
}

export function hasStarted(date: string | undefined, startTime: string | undefined, now = Date.now()): boolean {
  const ms = startMsOf(date, startTime);
  if (ms != null) return now >= ms;
  const d = (date ?? "").replace(/\D/g, "");
  return d.length === 8 && d < todayJst();
}

/** 発走後のレースの結果（着順・払戻）。結果が出ていなければ null。開いている間は結果が出るまで追いかける。
 *  poll=false は、開いたときに発走後なら 1 回だけ取る（周回しない。めぐ分析。CLAUDE.md: 15 分未満の間隔のタブの定期問い合わせを足さない）。 */
export function useRaceResult(raceId: string, date: string | undefined, startTime: string | undefined, enabled: boolean, poll = true): { result: RaceResultData | null; started: boolean } {
  const [result, setResult] = useState<RaceResultData | null>(null);
  const [started, setStarted] = useState(false);
  useEffect(() => {
    setResult(null);
    setStarted(hasStarted(date, startTime));
    if (!enabled) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const tick = async () => {
      const now = Date.now();
      const startMs = startMsOf(date, startTime);
      const st = hasStarted(date, startTime, now);
      if (!cancelled) setStarted(st);
      let done = false;
      if (st) {
        try {
          const r = await getJson<RaceResultData>(`/api/race/${raceId}/result`, { ttlMs: 20_000, credentials: "include" });
          if (cancelled) return;
          setResult(r.available ? r : null);
          done = !!(r.available && r.status === "final" && r.official);
        } catch {
          /* 取れなければ次の周回で再試行する */
        }
      }
      const within = startMs == null || now - startMs < POLL_WINDOW_MS;
      if (poll && !cancelled && !done && within) {
        const wait = st || startMs == null ? POLL_MS : Math.min(POLL_MS, Math.max(5_000, startMs - now));
        timer = setTimeout(tick, wait);
      }
    };
    tick();
    return () => { cancelled = true; if (timer) clearTimeout(timer); };
  }, [raceId, date, startTime, enabled, poll]);
  return { result, started };
}
