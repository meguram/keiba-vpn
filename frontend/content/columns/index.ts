/**
 * めぐコラムの記事一覧。記事を足すときは、この階層に <slug>.json を置いて下の ALL に 1 行足す。
 * 型と書式は lib/columns.ts の Column を参照。draft: true の記事は prod ビルドでは出ない（next dev と stg では出る）。
 */
import type { Column } from "@/lib/columns";
import { SHOW_DEV_ONLY } from "@/lib/features";
import upset_conditions from "./upset-conditions.json";
import summer_mare from "./summer-mare.json";
import first_blinkers from "./first-blinkers.json";
import gelding_return from "./gelding-return.json";
import transport_vs_stay from "./transport-vs-stay.json";
import early_maturing_bloodline from "./early-maturing-bloodline.json";
import first_crop_old_dam from "./first-crop-old-dam.json";
import front_runner from "./front-runner.json";
import first_dirt from "./first-dirt.json";
import jockey_long_distance from "./jockey-long-distance.json";
import jockey_change from "./jockey-change.json";
import pace_advantage from "./pace-advantage.json";
import weight_change from "./weight-change.json";
import distance_jockey from "./distance-jockey.json";
import jockey_roi from "./jockey-roi.json";
import pace_scenario from "./pace-scenario.json";
import sire_roi from "./sire-roi.json";

const ALL = [
  upset_conditions,
  summer_mare,
  first_blinkers,
  gelding_return,
  transport_vs_stay,
  early_maturing_bloodline,
  first_crop_old_dam,
  front_runner,
  first_dirt,
  jockey_long_distance,
  jockey_change,
  pace_advantage,
  weight_change,
  distance_jockey,
  jockey_roi,
  pace_scenario,
  sire_roi,
] as Column[];

export const COLUMNS: Column[] = ALL.filter((c) => SHOW_DEV_ONLY || !c.draft);

export function findColumn(slug: string): Column | undefined {
  return COLUMNS.find((c) => c.slug === slug);
}
