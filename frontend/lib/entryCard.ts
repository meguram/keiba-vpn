/* 出馬表の馬名・騎手・調教師をタップしたときの情報（GET /api/race/{race_id}/entry/{horse_id}/card）の型と取得。 */
import { USE_MOCK } from "@/lib/mock";

export type CardStat = { starts: number | null; wins: number | null; top3: number | null; win_rate: number | null; top3_rate: number | null; avg_finish?: number | null };
export type CondStat = CardStat & { label: string; has_data?: boolean };

export type PastRun = {
  date: string; race_id?: string | null; venue?: string | null; race_name?: string | null; surface?: string | null; distance?: number | null; going?: string | null;
  field?: number | null; post?: number | null; popularity?: number | null; finish?: number | string | null; time?: string | null; margin?: number | null;
  jockey?: string | null; jockey_weight?: number | null; passing?: string | null; last3f?: number | null; weight?: number | null; weight_change?: number | null;
};

export type TrainingInfo = { training_date?: string | null; course?: string | null; condition?: string | null; rider?: string | null; lap_times?: string | null; impression?: string | null; evaluation?: string | null; comment?: string | null };
export type TrainerComment = { comment: string; evaluation?: string | null; trainer?: string | null };

export type HorseCard = {
  status: "ok" | "no_data"; id: string; name?: string | null; sex_age?: string | null; weight?: number | null; weight_change?: number | null;
  profile: { sire?: string | null; dam?: string | null; dam_sire?: string | null; owner?: string | null; breeder?: string | null; birthday?: string | null; career?: string | null; earnings?: number | null } | null;
  recent: PastRun[];
  same_conditions: (CardStat & { label: string })[];
  changes: { layoff_days: number | null; distance_change: number | null; jockey_changed: boolean | null; previous_jockey?: string | null; last_run: { date: string; race_name?: string | null; finish?: number | string | null } | null };
  training: TrainingInfo | null;
  trainer_comment: TrainerComment | null;
  weights: { date: string; weight: number | null; change: number | null }[];
};

export type PersonProfile = {
  name?: string | null; affiliation?: string | null; debut_year?: string | null;
  career: { rides?: number | null; wins?: number | null; win_rate?: number | null; top2_rate?: number | null; top3_rate?: number | null } | null;
  this_year_text?: string | null; g1_wins_text?: string | null; graded_wins_text?: string | null;
} | null;

export type EntityCard = {
  status: "ok" | "no_data"; id: string; name?: string | null; profile: PersonProfile; conditions: CondStat[];
  recent: Record<string, CardStat & { label: string }>; career: (CardStat & { label: string }) | null; as_of: string | null; includes_after_race?: boolean;
  with_horse?: { rides_in_last5: number; wins: number; top3: number; checked: number; changed: boolean | null; previous_jockey?: string | null } | null;
  weight_today?: number | null; comment?: TrainerComment | null;
};

export type EntryCard = {
  status: "ok" | "not_found" | "bad_request"; race_id: string; horse_id: string; as_of?: string | null;
  race?: { name?: string | null; venue?: string | null; surface?: string | null; distance?: number | null; grade?: string | null };
  horse: HorseCard; jockey: EntityCard; trainer: EntityCard; sources?: string[];
};

const done = new Map<string, EntryCard>();
const inflight = new Map<string, Promise<EntryCard>>();

export function cachedEntryCard(raceId: string, horseId: string): EntryCard | undefined {
  return done.get(`${raceId}/${horseId}`);
}

/** 取得（同じ馬の再タップはキャッシュで即表示。同時の取得は 1 本にまとめる）。失敗は覚えない。 */
export function fetchEntryCard(raceId: string, horseId: string): Promise<EntryCard> {
  const key = `${raceId}/${horseId}`;
  const hit = done.get(key);
  if (hit) return Promise.resolve(hit);
  let p = inflight.get(key);
  if (!p) {
    p = (async () => {
      if (USE_MOCK) {
        const empty = { status: "no_data" as const, id: "", conditions: [], recent: {}, career: null, as_of: null, profile: null };
        return { status: "ok", race_id: raceId, horse_id: horseId, horse: { ...empty, recent: [], same_conditions: [], profile: null, changes: { layoff_days: null, distance_change: null, jockey_changed: null, last_run: null }, training: null, trainer_comment: null, weights: [] }, jockey: empty, trainer: empty } as unknown as EntryCard;
      }
      const res = await fetch(`/api/race/${encodeURIComponent(raceId)}/entry/${encodeURIComponent(horseId)}/card`);
      if (!res.ok) throw new Error(res.status === 404 ? "この馬の出馬表が見つかりません" : `HTTP ${res.status}`);
      const d: EntryCard = await res.json();
      done.set(key, d);
      return d;
    })();
    p.then(() => inflight.delete(key), () => inflight.delete(key));
    inflight.set(key, p);
  }
  return p;
}
