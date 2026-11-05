/**
 * 記事ごとの読み方ガイド（統計に詳しくない人向け）。型と書き方は lib/columns.ts の ColumnGuide。
 * 記事 JSON は集計コードが書き直すので、読み方の文章はここに置き、content/columns/index.ts で差し込む。
 * 共通の言葉は glossary.json（語 → やさしい説明）。記事の guides の match が記事に当たるかは tests/research/test_column_guides.py が確かめる。
 */
import type { ColumnGuide } from "@/lib/columns";
import glossary from "./glossary.json";
import distance_jockey from "./distance-jockey.json";
import early_maturing_bloodline from "./early-maturing-bloodline.json";
import first_blinkers from "./first-blinkers.json";
import first_crop_old_dam from "./first-crop-old-dam.json";
import first_dirt from "./first-dirt.json";
import front_runner from "./front-runner.json";
import gelding_return from "./gelding-return.json";
import jockey_change from "./jockey-change.json";
import jockey_class_venue_season from "./jockey-class-venue-season.json";
import jockey_long_distance from "./jockey-long-distance.json";
import jockey_roi from "./jockey-roi.json";
import pace_advantage from "./pace-advantage.json";
import pace_formation from "./pace-formation.json";
import pace_scenario from "./pace-scenario.json";
import sire_line_aptitude from "./sire-line-aptitude.json";
import sire_roi from "./sire-roi.json";
import summer_mare from "./summer-mare.json";
import transport_vs_stay from "./transport-vs-stay.json";
import upset_conditions from "./upset-conditions.json";
import weight_change from "./weight-change.json";
import wind_venue from "./wind-venue.json";

export const GLOSSARY = glossary as Record<string, string>;

export const GUIDES: Record<string, ColumnGuide> = {
  "distance-jockey": distance_jockey as ColumnGuide,
  "early-maturing-bloodline": early_maturing_bloodline as ColumnGuide,
  "first-blinkers": first_blinkers as ColumnGuide,
  "first-crop-old-dam": first_crop_old_dam as ColumnGuide,
  "first-dirt": first_dirt as ColumnGuide,
  "front-runner": front_runner as ColumnGuide,
  "gelding-return": gelding_return as ColumnGuide,
  "jockey-change": jockey_change as ColumnGuide,
  "jockey-class-venue-season": jockey_class_venue_season as ColumnGuide,
  "jockey-long-distance": jockey_long_distance as ColumnGuide,
  "jockey-roi": jockey_roi as ColumnGuide,
  "pace-advantage": pace_advantage as ColumnGuide,
  "pace-formation": pace_formation as ColumnGuide,
  "pace-scenario": pace_scenario as ColumnGuide,
  "sire-line-aptitude": sire_line_aptitude as ColumnGuide,
  "sire-roi": sire_roi as ColumnGuide,
  "summer-mare": summer_mare as ColumnGuide,
  "transport-vs-stay": transport_vs_stay as ColumnGuide,
  "upset-conditions": upset_conditions as ColumnGuide,
  "weight-change": weight_change as ColumnGuide,
  "wind-venue": wind_venue as ColumnGuide,
};
