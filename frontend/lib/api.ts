const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

export async function fetchApi<T>(path: string, init?: RequestInit): Promise<T> {
  const url = `${API_BASE}${path}`;
  const res = await fetch(url, { ...init, cache: "no-store" });
  if (!res.ok) throw new Error(`${path}: ${res.status}`);
  return res.json() as Promise<T>;
}

export type RaceSummary = {
  race_id: string;
  race_name?: string;
  venue?: string;
  race_date?: string;
  start_time?: string;
  surface?: string;
  distance?: number;
};

export type PredictionHorse = {
  horse_id: string;
  post_no?: number;
  win_prob?: number;
  expected_win_roi?: number;
  expected_show_roi?: number;
  is_value_bet?: boolean;
  predicted_running_style?: string;
};

export type PredictionsResponse = {
  race_id: string;
  model_version: string;
  horses: PredictionHorse[];
  /** ゲスト閲覧時は true（TOP3 のみ返却）*/
  is_guest?: boolean;
  /** レース全頭数（ゲスト制限前の頭数）*/
  total_horses?: number;
  pace_prediction?: {
    pace_category?: string;
    lap_times?: { furlong_index: number; predicted_lap_sec: number }[];
  };
};
