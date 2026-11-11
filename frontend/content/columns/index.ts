/**
 * めぐコラムの記事一覧。記事を足すときは、この階層に <slug>.json を置いて下の ALL に 1 行足す。
 * 型と書式は lib/columns.ts の Column を参照。draft: true の記事は prod ビルドでは出ない（next dev と stg では出る）。
 */
import { applyGuide, type Column } from "@/lib/columns";
import { GLOSSARY, GUIDES } from "./guides";
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
import jockey_class_venue_season from "./jockey-class-venue-season.json";
import pace_advantage from "./pace-advantage.json";
import weight_change from "./weight-change.json";
import distance_jockey from "./distance-jockey.json";
import jockey_roi from "./jockey-roi.json";
import pace_scenario from "./pace-scenario.json";
import pace_formation from "./pace-formation.json";
import sire_roi from "./sire-roi.json";
import wind_venue from "./wind-venue.json";
import sire_line_aptitude from "./sire-line-aptitude.json";
import distance_extension from "./distance-extension.json";
import distance_shortening from "./distance-shortening.json";
import direction_switch from "./direction-switch.json";
import local_to_central from "./local-to-central.json";
import second_start_after_layoff from "./second-start-after-layoff.json";
import quick_turnaround from "./quick-turnaround.json";
import layoff_first_start from "./layoff-first-start.json";
import overseas_return from "./overseas-return.json";
import class_up_first from "./class-up-first.json";
import close_loss_last_time from "./close-loss-last-time.json";
import fastest_closer_loser from "./fastest-closer-loser.json";
import big_loss_rebound from "./big-loss-rebound.json";
import bad_going_excuse from "./bad-going-excuse.json";
import high_pace_survivor from "./high-pace-survivor.json";
import slow_pace_closer from "./slow-pace-closer.json";
import deep_closer_board from "./deep-closer-board.json";
import lone_front_runner from "./lone-front-runner.json";
import course_specialist from "./course-specialist.json";
import fastest_closing_count from "./fastest-closing-count.json";
import g1_step from "./g1-step.json";
import final_workout_jockey from "./final-workout-jockey.json";
import slope_acceleration from "./slope-acceleration.json";
import slope_vs_wood from "./slope-vs-wood.json";
import hard_workout_after_layoff from "./hard-workout-after-layoff.json";
import workout_volume_layoff from "./workout-volume-layoff.json";
import workout_rank from "./workout-rank.json";
import trainer_jockey_combo from "./trainer-jockey-combo.json";
import rare_top_jockey from "./rare-top-jockey.json";
import jockey_chose_other from "./jockey-chose-other.json";
import jockey_continuation from "./jockey-continuation.json";
import jockey_hot_day from "./jockey-hot-day.json";
import jockey_single_ride from "./jockey-single-ride.json";
import northern_farm_layoff from "./northern-farm-layoff.json";
import stable_two_runners from "./stable-two-runners.json";
import trainer_change from "./trainer-change.json";
import same_day_bias from "./same-day-bias.json";
import opening_week_inside from "./opening-week-inside.json";
import full_gate_outside from "./full-gate-outside.json";
import even_gate from "./even-gate.json";
import mudder from "./mudder.json";
import big_horse_heavy_going from "./big-horse-heavy-going.json";
import weight_drop_big from "./weight-drop-big.json";
import heavy_impost from "./heavy-impost.json";
import old_horse from "./old-horse.json";
import three_year_old_summer from "./three-year-old-summer.json";
import older_mare from "./older-mare.json";
import last_race_favorite from "./last-race-favorite.json";
import course_repeat from "./course-repeat.json";
import class_up_details from "./class-up-details.json";
import prev_race_excuses from "./prev-race-excuses.json";
import layoff_details from "./layoff-details.json";
import lap_type from "./lap-type.json";
import terrain_fit from "./terrain-fit.json";
import workout_speed from "./workout-speed.json";
import workout_pattern from "./workout-pattern.json";
import workout_notes from "./workout-notes.json";
import jockey_switch_quality from "./jockey-switch-quality.json";
import jockey_specialty from "./jockey-specialty.json";
import jockey_day from "./jockey-day.json";
import jockey_claim from "./jockey-claim.json";
import stable_form from "./stable-form.json";
import breeder_owner from "./breeder-owner.json";
import trip_region from "./trip-region.json";
import sire_course_going from "./sire-course-going.json";
import sire_timing from "./sire-timing.json";
import sire_line_dirt from "./sire-line-dirt.json";
import going_style from "./going-style.json";
import day_bias_more from "./day-bias-more.json";
import weather_wind_heat from "./weather-wind-heat.json";
import gate_more from "./gate-more.json";
import pace_setup from "./pace-setup.json";
import style_by_venue from "./style-by-venue.json";
import weight_impost_more from "./weight-impost-more.json";
import sex_age_more from "./sex-age-more.json";
import gear_gelding from "./gear-gelding.json";
import trainer_comment from "./trainer-comment.json";
import style_by_surface from "./style-by-surface.json";
import race_program from "./race-program.json";
import first_time from "./first-time.json";
import two_strong from "./two-strong.json";
import favorite_fade from "./favorite-fade.json";
import index_claims from "./index-claims.json";
import trainer_comment_wordcloud from "./trainer-comment-wordcloud.json";

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
  jockey_class_venue_season,
  pace_advantage,
  weight_change,
  distance_jockey,
  jockey_roi,
  pace_scenario,
  pace_formation,
  sire_roi,
  wind_venue,
  sire_line_aptitude,
  distance_extension,
  distance_shortening,
  direction_switch,
  local_to_central,
  second_start_after_layoff,
  quick_turnaround,
  layoff_first_start,
  overseas_return,
  class_up_first,
  close_loss_last_time,
  fastest_closer_loser,
  big_loss_rebound,
  bad_going_excuse,
  high_pace_survivor,
  slow_pace_closer,
  deep_closer_board,
  lone_front_runner,
  course_specialist,
  fastest_closing_count,
  g1_step,
  final_workout_jockey,
  slope_acceleration,
  slope_vs_wood,
  hard_workout_after_layoff,
  workout_volume_layoff,
  workout_rank,
  trainer_jockey_combo,
  rare_top_jockey,
  jockey_chose_other,
  jockey_continuation,
  jockey_hot_day,
  jockey_single_ride,
  northern_farm_layoff,
  stable_two_runners,
  trainer_change,
  same_day_bias,
  opening_week_inside,
  full_gate_outside,
  even_gate,
  mudder,
  big_horse_heavy_going,
  weight_drop_big,
  heavy_impost,
  old_horse,
  three_year_old_summer,
  older_mare,
  last_race_favorite,
  course_repeat,
  class_up_details,
  prev_race_excuses,
  layoff_details,
  lap_type,
  terrain_fit,
  workout_speed,
  workout_pattern,
  workout_notes,
  jockey_switch_quality,
  jockey_specialty,
  jockey_day,
  jockey_claim,
  stable_form,
  breeder_owner,
  trip_region,
  sire_course_going,
  sire_timing,
  sire_line_dirt,
  going_style,
  day_bias_more,
  weather_wind_heat,
  gate_more,
  pace_setup,
  style_by_venue,
  weight_impost_more,
  sex_age_more,
  gear_gelding,
  trainer_comment,
  style_by_surface,
  race_program,
  first_time,
  two_strong,
  favorite_fade,
  index_claims,
  trainer_comment_wordcloud,
] as Column[];

export const COLUMNS: Column[] = ALL.filter((c) => SHOW_DEV_ONLY || !c.draft)
  .map((c) => ({ ...c, body: applyGuide(c.body, GUIDES[c.slug], GLOSSARY) }));

export function findColumn(slug: string): Column | undefined {
  return COLUMNS.find((c) => c.slug === slug);
}
