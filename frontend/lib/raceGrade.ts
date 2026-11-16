/** レースの格（重賞・リステッド・OP）の判定と色、レース番号の選び方（会場を切り替えたとき）。
 *  格はレース一覧の grade（DB）か、無ければ一覧の格のアイコン（list_grade_icon。netkeiba の Icon_GradeType<n>）から決める。
 *  新しい読み出しは足さない（今のレース一覧の項目だけを使う）。 */

export type GradeKey = "G1" | "G2" | "G3" | "J.G1" | "J.G2" | "J.G3" | "L" | "OP";

/** 一覧の格のアイコン番号 → 格（16〜18 は 3 勝〜1 勝クラスなので印を付けない） */
const ICON: Record<string, GradeKey> = { "1": "G1", "2": "G2", "3": "G3", "10": "J.G1", "11": "J.G2", "12": "J.G3", "15": "L", "5": "OP" };

const NORM: Record<string, GradeKey> = {
  G1: "G1", G2: "G2", G3: "G3", GI: "G1", GII: "G2", GIII: "G3", JG1: "J.G1", JG2: "J.G2", JG3: "J.G3", "J.G1": "J.G1", "J.G2": "J.G2", "J.G3": "J.G3",
  L: "L", Listed: "L", OP: "OP",
};

/** 格。①一覧の格のアイコン（netkeiba の一覧。重賞の印が確か）②レース名の「(G3)」③grade（DB・出馬表）が G・J.G のときだけ。
 *  DB・出馬表の grade は、ページの文字に「オープン」があると重賞でも OP になるので、OP・L はアイコンからしか付けない */
export function gradeOf(r: { grade?: string | null; list_grade_icon?: string | null; race_name?: string | null }): GradeKey | null {
  const m = /GradeType(\d+)/.exec(String(r.list_grade_icon ?? ""));
  if (m) return ICON[m[1]] ?? null;
  const n = /[（(]\s*(J\.?G[123]|G[123])\s*[）)]/.exec(String(r.race_name ?? ""));
  if (n) return NORM[n[1].replace("JG", "J.G")] ?? null;
  const g = r.grade ? NORM[String(r.grade).replace(/\s/g, "")] : undefined;
  return g && isGraded(g) ? g : null;
}

export function isGraded(g: GradeKey | null | undefined): boolean {
  return !!g && g !== "L" && g !== "OP";
}

/** 格の色（JRA の慣例に近い: G1 青・G2 赤・G3 緑。障害の重賞も同じ色）。札は白い文字で、ライト・ダークの両方で読める濃さにする。
 *  L・OP は暗い地に明るい文字の札（重賞の塗りと見分けるため。推奨度の色のチップの上でも読めるよう不透明にする） */
export const GRADE_STYLE: Record<GradeKey, { bg: string; fg: string; border: string }> = {
  G1: { bg: "#1d4ed8", fg: "#ffffff", border: "#3b82f6" },
  G2: { bg: "#b91c1c", fg: "#ffffff", border: "#ef4444" },
  G3: { bg: "#15803d", fg: "#ffffff", border: "#22c55e" },
  "J.G1": { bg: "#1d4ed8", fg: "#ffffff", border: "#3b82f6" },
  "J.G2": { bg: "#b91c1c", fg: "#ffffff", border: "#ef4444" },
  "J.G3": { bg: "#15803d", fg: "#ffffff", border: "#22c55e" },
  L: { bg: "#1e1b4b", fg: "#c4b5fd", border: "#8b5cf6" },
  OP: { bg: "#1f2937", fg: "#e5e7eb", border: "#6b7280" },
};

/** レース名の略称（「第12回」や末尾の「(G3)」を除き、長ければ詰める） */
export function shortName(name: string | null | undefined, max = 6): string {
  const s = String(name ?? "").replace(/^第\s*\d+回\s*/, "").replace(/[（(]\s*J?\.?G\d\s*[）)]\s*$/, "").trim();
  return s.length > max ? `${s.slice(0, max - 1)}…` : s;
}

/** 発走時刻（"15:40" の形だけ。それ以外は null） */
export function startTimeOf(r: { start_time?: string | null }): string | null {
  const m = /^(\d{1,2}):(\d{2})/.exec(String(r.start_time ?? ""));
  return m ? `${m[1].padStart(2, "0")}:${m[2]}` : null;
}

/** 会場を切り替えたときに選ぶレース: 今と同じレース番号があればそれ、無ければ一番近い番号（同じ距離なら小さい方）。
 *  選べないレース（disabled）は飛ばす。今のレースが無ければ、その会場の最初のレース */
export function sameRoundIn<T extends { race_id: string; venue?: string; round?: number | string }>(
  races: T[], venue: string, round: number | null | undefined, disabled?: Set<string>,
): T | undefined {
  const rn = (r: T) => {
    const x = Number(r.round);
    return Number.isFinite(x) && x > 0 ? x : Number(r.race_id.slice(-2));
  };
  const list = races.filter((r) => r.venue === venue && !disabled?.has(r.race_id)).sort((a, b) => rn(a) - rn(b));
  if (!list.length) return undefined;
  if (round == null || !Number.isFinite(round)) return list[0];
  return [...list].sort((a, b) => Math.abs(rn(a) - round) - Math.abs(rn(b) - round) || rn(a) - rn(b))[0];
}
