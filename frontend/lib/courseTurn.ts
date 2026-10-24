/** コースの回り（右／左／直）。API の direction が無いとき（週間予測の一覧など）は競馬場から決める。
 *  JRA は競馬場ごとに回りが決まっている（右: 札幌・函館・福島・中山・京都・阪神・小倉、左: 新潟・東京・中京）。新潟の芝 1000m だけ直線。 */
const VENUE_TURN: Record<string, "右" | "左"> = {
  札幌: "右", 函館: "右", 福島: "右", 中山: "右", 京都: "右", 阪神: "右", 小倉: "右",
  新潟: "左", 東京: "左", 中京: "左",
};

export function turnOf(direction?: string | null, venue?: string | null, surface?: string | null, distance?: number | string | null): string | null {
  if (direction) return direction;
  const v = (venue ?? "").trim();
  const t = VENUE_TURN[v];
  if (!t) return null;
  if (v === "新潟" && (surface ?? "").startsWith("芝") && Number(distance) === 1000) return "直";
  return t;
}
