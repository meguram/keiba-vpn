/**
 * 血統予想家の欄の型（めぐ分析。API は GET /api/v1/races/{race_id}/analysis が別のファイルから合わせて返す）。
 * 値はホストが発走前に作って置いたもの。画面では計算しない。点・閾値・人気の数字・成績は返らない（出さない）。
 */

export type PPPick = { horse_number: number; horse_name: string; reason: string };

export type PPFigure = {
  intro: string;
  /** 「回収率は確かめていない。成績は 2028 年ごろまで出さない」 */
  note: string;
  surface: string;
  /** 芝は「弱め」に固定。ダートは null（自信の段を出さない） */
  confidence: string | null;
  facts: string;
  /** pending: 発走の約 60 分前に出す / ready: 一覧あり / none: 当てはまる馬なし / unavailable: 出していない / not_applicable: 新馬戦など */
  picks_state: "pending" | "ready" | "none" | "unavailable" | "not_applicable";
  picks: PPPick[];
};

export function picksMessage(state: PPFigure["picks_state"]): string | null {
  switch (state) {
    case "pending":
      return "一覧は、発走の約 60 分前に出します。";
    case "none":
      return "このレースでは、当てはまる馬がいません。";
    case "unavailable":
      return "今回は一覧を出していません。";
    case "not_applicable":
      return "新馬戦などでは一覧を出しません。";
    default:
      return null;
  }
}

/** 発走の時刻（JST）を過ぎたか。過ぎたら欄を出さない（ブラウザに保存された前回の応答を先に出すときも、一覧と着順を並べない）。 */
export function pastStart(race: { date?: string | null; start_time?: string | null } | null | undefined, now: Date = new Date()): boolean {
  const d = race?.date ?? "";
  if (!/^\d{4}-\d{2}-\d{2}$/.test(d)) return true;
  const m = /^(\d{1,2}):(\d{2})/.exec(race?.start_time ?? "");
  const t = m ? `${m[1].padStart(2, "0")}:${m[2]}` : "23:59";
  return now.getTime() >= new Date(`${d}T${t}:00+09:00`).getTime();
}
