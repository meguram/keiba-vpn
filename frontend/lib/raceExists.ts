/** /api/race/<id> は存在しない ID でも 200 で全項目 null を返す。レース名・開催場・各カテゴリのどれも無ければ「存在しない」とみなす。 */
export function raceExists(rd: Record<string, unknown> | null | undefined): boolean {
  if (!rd) return false;
  if (rd.race_name || rd.venue) return true;
  return Object.entries(rd).some(([k, v]) => k.startsWith("race_") && k !== "race_id" && k !== "race_name" && !!v);
}
