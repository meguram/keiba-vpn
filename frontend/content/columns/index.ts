/**
 * めぐコラムの記事一覧。記事を足すときは、この階層に <slug>.json を置いて下の ALL に 1 行足す。
 * 型と書式は lib/columns.ts の Column を参照。draft: true の記事は prod ビルドでは出ない（next dev と stg では出る）。
 */
import type { Column } from "@/lib/columns";
import { SHOW_DEV_ONLY } from "@/lib/features";
import upset_conditions from "./upset-conditions.json";
import upset_short_local from "./upset-short-local.json";
import summer_mare from "./summer-mare.json";
import first_blinkers from "./first-blinkers.json";
import gelding_return from "./gelding-return.json";
import transport_vs_stay from "./transport-vs-stay.json";
import early_maturing_bloodline from "./early-maturing-bloodline.json";
import first_crop_old_dam from "./first-crop-old-dam.json";
import front_runner from "./front-runner.json";
import first_dirt from "./first-dirt.json";
import jockey_long_distance from "./jockey-long-distance.json";

const ALL = [
  upset_conditions,
  upset_short_local,
  summer_mare,
  first_blinkers,
  gelding_return,
  transport_vs_stay,
  early_maturing_bloodline,
  first_crop_old_dam,
  front_runner,
  first_dirt,
  jockey_long_distance,
] as Column[];

export const COLUMNS: Column[] = ALL.filter((c) => SHOW_DEV_ONLY || !c.draft);

export function findColumn(slug: string): Column | undefined {
  return COLUMNS.find((c) => c.slug === slug);
}
