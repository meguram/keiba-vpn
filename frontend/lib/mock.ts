export const USE_MOCK = process.env.NEXT_PUBLIC_MOCK === "true";

// ---------------------------------------------------------------------------
// ユーティリティ
// ---------------------------------------------------------------------------
/** JRA公式枠番計算: 1枠=1番、2枠=2-3番、…、8枠=14-16番 */
function calcBracket(horseNumber: number): number {
  return Math.min(Math.ceil((horseNumber + 1) / 2), 8);
}

// ---------------------------------------------------------------------------
// Races
// ---------------------------------------------------------------------------
export const MOCK_RACES = [
  { race_id: "r001", race_name: "東京優駿（日本ダービー）", venue: "東京", race_date: "2026-06-01", start_time: "15:40", surface: "芝", distance: 2400, field_size: 18, grade: "GI" },
  { race_id: "r002", race_name: "安田記念", venue: "東京", race_date: "2026-06-08", start_time: "15:40", surface: "芝", distance: 1600, field_size: 16, grade: "GI" },
  { race_id: "r003", race_name: "宝塚記念", venue: "阪神", race_date: "2026-06-22", start_time: "15:40", surface: "芝", distance: 2200, field_size: 15, grade: "GI" },
  { race_id: "r004", race_name: "中京記念", venue: "中京", race_date: "2026-07-20", start_time: "15:25", surface: "芝", distance: 1600, field_size: 16, grade: "GIII" },
  { race_id: "r005", race_name: "3歳未勝利", venue: "東京", race_date: "2026-07-05", start_time: "10:30", surface: "芝", distance: 1800, field_size: 14, grade: "未勝利" },
  { race_id: "r006", race_name: "3歳未勝利", venue: "阪神", race_date: "2026-07-05", start_time: "11:00", surface: "ダ", distance: 1400, field_size: 16, grade: "未勝利" },
  { race_id: "r007", race_name: "小倉2歳ステークス", venue: "阪神", race_date: "2026-08-17", start_time: "15:25", surface: "芝", distance: 1200, field_size: 12, grade: "GIII" },
  { race_id: "r008", race_name: "関屋記念", venue: "中京", race_date: "2026-08-10", start_time: "15:25", surface: "芝", distance: 1600, field_size: 14, grade: "GIII" },
];

// ---------------------------------------------------------------------------
// Horses
// ---------------------------------------------------------------------------
export const MOCK_HORSES = [
  { horse_id: "h01", horse_name: "アイアンウィル", post_no: 1, win_prob: 0.182, place_prob: 0.421, show_prob: 0.631, predicted_win_odds: 5.5, predicted_place_odds: 1.8, win_roi: 1.00, show_roi: 1.14, predicted_position: 1, predicted_running_style: "先行", is_value_bet: true },
  { horse_id: "h02", horse_name: "シルバーミスト", post_no: 2, win_prob: 0.145, place_prob: 0.372, show_prob: 0.592, predicted_win_odds: 6.9, predicted_place_odds: 2.1, win_roi: 1.00, show_roi: 1.24, predicted_position: 2, predicted_running_style: "差し", is_value_bet: false },
  { horse_id: "h03", horse_name: "ゴールドストーム", post_no: 3, win_prob: 0.231, place_prob: 0.481, show_prob: 0.712, predicted_win_odds: 4.3, predicted_place_odds: 1.6, win_roi: 0.99, show_roi: 1.14, predicted_position: 1, predicted_running_style: "逃げ", is_value_bet: true },
  { horse_id: "h04", horse_name: "クリムゾンドーン", post_no: 4, win_prob: 0.089, place_prob: 0.251, show_prob: 0.423, predicted_win_odds: 11.2, predicted_place_odds: 3.1, win_roi: 1.00, show_roi: 1.31, predicted_position: 5, predicted_running_style: "追い込み", is_value_bet: false },
  { horse_id: "h05", horse_name: "ブルーサンダー", post_no: 5, win_prob: 0.068, place_prob: 0.198, show_prob: 0.341, predicted_win_odds: 14.7, predicted_place_odds: 3.9, win_roi: 1.00, show_roi: 1.33, predicted_position: 7, predicted_running_style: "差し", is_value_bet: false },
  { horse_id: "h06", horse_name: "スターラッシュ", post_no: 6, win_prob: 0.121, place_prob: 0.312, show_prob: 0.521, predicted_win_odds: 8.3, predicted_place_odds: 2.4, win_roi: 1.00, show_roi: 1.25, predicted_position: 3, predicted_running_style: "先行", is_value_bet: false },
  { horse_id: "h07", horse_name: "ダークホライズン", post_no: 7, win_prob: 0.052, place_prob: 0.163, show_prob: 0.284, predicted_win_odds: 19.2, predicted_place_odds: 4.7, win_roi: 1.00, show_roi: 1.33, predicted_position: 9, predicted_running_style: "追い込み", is_value_bet: false },
  { horse_id: "h08", horse_name: "ルビーフレイム", post_no: 8, win_prob: 0.098, place_prob: 0.271, show_prob: 0.459, predicted_win_odds: 10.2, predicted_place_odds: 2.9, win_roi: 1.00, show_roi: 1.33, predicted_position: 4, predicted_running_style: "先行", is_value_bet: false },
  { horse_id: "h09", horse_name: "エメラルドウィング", post_no: 9, win_prob: 0.041, place_prob: 0.129, show_prob: 0.231, predicted_win_odds: 24.4, predicted_place_odds: 5.9, win_roi: 1.00, show_roi: 1.36, predicted_position: 11, predicted_running_style: "差し", is_value_bet: false },
  { horse_id: "h10", horse_name: "サンライズキング", post_no: 10, win_prob: 0.158, place_prob: 0.391, show_prob: 0.612, predicted_win_odds: 6.3, predicted_place_odds: 2.0, win_roi: 1.00, show_roi: 1.22, predicted_position: 2, predicted_running_style: "先行", is_value_bet: true },
  { horse_id: "h11", horse_name: "ムーンシャドウ", post_no: 11, win_prob: 0.033, place_prob: 0.101, show_prob: 0.189, predicted_win_odds: 30.3, predicted_place_odds: 7.4, win_roi: 1.00, show_roi: 1.40, predicted_position: 13, predicted_running_style: "逃げ", is_value_bet: false },
  { horse_id: "h12", horse_name: "ブロンズタイタン", post_no: 12, win_prob: 0.072, place_prob: 0.211, show_prob: 0.362, predicted_win_odds: 13.9, predicted_place_odds: 3.7, win_roi: 1.00, show_roi: 1.34, predicted_position: 6, predicted_running_style: "差し", is_value_bet: false },
  { horse_id: "h13", horse_name: "ホワイトライトニング", post_no: 13, win_prob: 0.029, place_prob: 0.089, show_prob: 0.172, predicted_win_odds: 34.5, predicted_place_odds: 8.4, win_roi: 1.00, show_roi: 1.45, predicted_position: 14, predicted_running_style: "追い込み", is_value_bet: false },
  { horse_id: "h14", horse_name: "クォーツクラウン", post_no: 14, win_prob: 0.044, place_prob: 0.141, show_prob: 0.253, predicted_win_odds: 22.7, predicted_place_odds: 5.3, win_roi: 1.00, show_roi: 1.34, predicted_position: 10, predicted_running_style: "差し", is_value_bet: false },
  { horse_id: "h15", horse_name: "スカーレットローズ", post_no: 15, win_prob: 0.113, place_prob: 0.293, show_prob: 0.487, predicted_win_odds: 8.8, predicted_place_odds: 2.6, win_roi: 1.00, show_roi: 1.26, predicted_position: 4, predicted_running_style: "先行", is_value_bet: false },
  { horse_id: "h16", horse_name: "サファイアストーム", post_no: 16, win_prob: 0.025, place_prob: 0.078, show_prob: 0.153, predicted_win_odds: 40.0, predicted_place_odds: 9.6, win_roi: 1.00, show_roi: 1.47, predicted_position: 15, predicted_running_style: "逃げ", is_value_bet: false },
];

// ---------------------------------------------------------------------------
// Lap data
// ---------------------------------------------------------------------------
export const MOCK_LAP_DATA = [
  { furlong_index: 1, predicted_lap_sec: 12.8 },
  { furlong_index: 2, predicted_lap_sec: 11.4 },
  { furlong_index: 3, predicted_lap_sec: 11.6 },
  { furlong_index: 4, predicted_lap_sec: 11.9 },
  { furlong_index: 5, predicted_lap_sec: 11.7 },
  { furlong_index: 6, predicted_lap_sec: 11.5 },
  { furlong_index: 7, predicted_lap_sec: 11.3 },
  { furlong_index: 8, predicted_lap_sec: 11.8 },
];

// ---------------------------------------------------------------------------
// Sires
// ---------------------------------------------------------------------------
export const MOCK_SIRES = [
  { sire_id: "s01", sire_name: "ディープインパクト", country: "JPN", win_rate: 0.182, place_rate: 0.472, avg_earnings: 18420000, turf_rate: 0.924, best_distance: "1600-2400", best_going: "良-稍重", sample_n: 1842 },
  { sire_id: "s02", sire_name: "キングカメハメハ", country: "JPN", win_rate: 0.168, place_rate: 0.451, avg_earnings: 16310000, turf_rate: 0.812, best_distance: "1800-2200", best_going: "良", sample_n: 1621 },
  { sire_id: "s03", sire_name: "ハーツクライ", country: "JPN", win_rate: 0.154, place_rate: 0.432, avg_earnings: 14820000, turf_rate: 0.891, best_distance: "2000-2500", best_going: "良-稍重", sample_n: 1398 },
  { sire_id: "s04", sire_name: "ロードカナロア", country: "JPN", win_rate: 0.196, place_rate: 0.492, avg_earnings: 12940000, turf_rate: 0.783, best_distance: "1200-1600", best_going: "良", sample_n: 1183 },
  { sire_id: "s05", sire_name: "エピファネイア", country: "JPN", win_rate: 0.171, place_rate: 0.458, avg_earnings: 15670000, turf_rate: 0.902, best_distance: "1800-2400", best_going: "良-稍重", sample_n: 987 },
  { sire_id: "s06", sire_name: "モーリス", country: "JPN", win_rate: 0.163, place_rate: 0.441, avg_earnings: 13210000, turf_rate: 0.855, best_distance: "1400-2000", best_going: "良", sample_n: 743 },
  { sire_id: "s07", sire_name: "ブリックスアンドモルタル", country: "USA", win_rate: 0.143, place_rate: 0.411, avg_earnings: 11830000, turf_rate: 0.921, best_distance: "1800-2200", best_going: "良", sample_n: 512 },
];

// ---------------------------------------------------------------------------
// Jockeys
// ---------------------------------------------------------------------------
export const MOCK_JOCKEYS = [
  { jockey_id: "j01", jockey_name: "川田将雅", win_rate: 0.241, place_rate: 0.551, top3_rate: 0.721, rides: 892, wins: 215 },
  { jockey_id: "j02", jockey_name: "クリストフ・ルメール", win_rate: 0.283, place_rate: 0.591, top3_rate: 0.762, rides: 812, wins: 230 },
  { jockey_id: "j03", jockey_name: "福永祐一", win_rate: 0.198, place_rate: 0.481, top3_rate: 0.659, rides: 841, wins: 167 },
  { jockey_id: "j04", jockey_name: "横山武史", win_rate: 0.214, place_rate: 0.511, top3_rate: 0.692, rides: 774, wins: 166 },
  { jockey_id: "j05", jockey_name: "松山弘平", win_rate: 0.187, place_rate: 0.463, top3_rate: 0.641, rides: 698, wins: 131 },
  { jockey_id: "j06", jockey_name: "武豊", win_rate: 0.221, place_rate: 0.532, top3_rate: 0.711, rides: 621, wins: 137 },
];

// ---------------------------------------------------------------------------
// Tracking difficulty
// ---------------------------------------------------------------------------
export const MOCK_TRACKING_DIFFICULTY = [
  { horse_id: "h03", horse_name: "ゴールドストーム", ease_score: 92, position_label: "逃げ", pace_sensitivity: 0.21, leader_gap_avg: 0.0 },
  { horse_id: "h01", horse_name: "アイアンウィル", ease_score: 87, position_label: "先行", pace_sensitivity: 0.35, leader_gap_avg: 1.2 },
  { horse_id: "h10", horse_name: "サンライズキング", ease_score: 83, position_label: "先行", pace_sensitivity: 0.38, leader_gap_avg: 1.5 },
  { horse_id: "h06", horse_name: "スターラッシュ", ease_score: 79, position_label: "先行", pace_sensitivity: 0.42, leader_gap_avg: 2.1 },
  { horse_id: "h15", horse_name: "スカーレットローズ", ease_score: 74, position_label: "先行", pace_sensitivity: 0.47, leader_gap_avg: 2.8 },
  { horse_id: "h08", horse_name: "ルビーフレイム", ease_score: 68, position_label: "中団", pace_sensitivity: 0.55, leader_gap_avg: 4.1 },
  { horse_id: "h02", horse_name: "シルバーミスト", ease_score: 63, position_label: "差し", pace_sensitivity: 0.61, leader_gap_avg: 5.3 },
  { horse_id: "h12", horse_name: "ブロンズタイタン", ease_score: 58, position_label: "差し", pace_sensitivity: 0.67, leader_gap_avg: 6.2 },
  { horse_id: "h04", horse_name: "クリムゾンドーン", ease_score: 51, position_label: "追い込み", pace_sensitivity: 0.78, leader_gap_avg: 8.1 },
  { horse_id: "h07", horse_name: "ダークホライズン", ease_score: 44, position_label: "追い込み", pace_sensitivity: 0.89, leader_gap_avg: 10.4 },
];

// ---------------------------------------------------------------------------
// Growth curve
// ---------------------------------------------------------------------------
export const MOCK_GROWTH_CURVE = [
  {
    horse_id: "h01", horse_name: "アイアンウィル",
    data: [
      { race_no: 1, time_index: 82, body_weight: 452 },
      { race_no: 2, time_index: 85, body_weight: 458 },
      { race_no: 3, time_index: 88, body_weight: 462 },
      { race_no: 4, time_index: 91, body_weight: 468 },
      { race_no: 5, time_index: 93, body_weight: 472 },
    ],
  },
  {
    horse_id: "h03", horse_name: "ゴールドストーム",
    data: [
      { race_no: 1, time_index: 78, body_weight: 480 },
      { race_no: 2, time_index: 82, body_weight: 484 },
      { race_no: 3, time_index: 87, body_weight: 486 },
      { race_no: 4, time_index: 90, body_weight: 490 },
      { race_no: 5, time_index: 95, body_weight: 492 },
    ],
  },
  {
    horse_id: "h10", horse_name: "サンライズキング",
    data: [
      { race_no: 1, time_index: 75, body_weight: 436 },
      { race_no: 2, time_index: 79, body_weight: 440 },
      { race_no: 3, time_index: 83, body_weight: 442 },
      { race_no: 4, time_index: 86, body_weight: 446 },
      { race_no: 5, time_index: 89, body_weight: 448 },
    ],
  },
  {
    horse_id: "h06", horse_name: "スターラッシュ",
    data: [
      { race_no: 1, time_index: 71, body_weight: 418 },
      { race_no: 2, time_index: 74, body_weight: 422 },
      { race_no: 3, time_index: 77, body_weight: 424 },
      { race_no: 4, time_index: 80, body_weight: 428 },
      { race_no: 5, time_index: 82, body_weight: 430 },
    ],
  },
  {
    horse_id: "h02", horse_name: "シルバーミスト",
    data: [
      { race_no: 1, time_index: 69, body_weight: 426 },
      { race_no: 2, time_index: 72, body_weight: 428 },
      { race_no: 3, time_index: 76, body_weight: 430 },
      { race_no: 4, time_index: 79, body_weight: 432 },
      { race_no: 5, time_index: 84, body_weight: 434 },
    ],
  },
];

// ---------------------------------------------------------------------------
// Track speed index
// ---------------------------------------------------------------------------
export const MOCK_TRACK_SPEED = [
  { date: "2026-06-01", venue: "東京", surface: "芝", going: "良", tsi: 101.2, moisture_pct: 12.1 },
  { date: "2026-06-08", venue: "東京", surface: "芝", going: "良", tsi: 100.8, moisture_pct: 11.8 },
  { date: "2026-06-15", venue: "東京", surface: "芝", going: "稍重", tsi: 98.4, moisture_pct: 18.3 },
  { date: "2026-06-22", venue: "阪神", surface: "芝", going: "良", tsi: 99.7, moisture_pct: 13.2 },
  { date: "2026-06-29", venue: "東京", surface: "芝", going: "重", tsi: 94.1, moisture_pct: 24.7 },
  { date: "2026-07-06", venue: "中京", surface: "芝", going: "良", tsi: 100.3, moisture_pct: 12.5 },
  { date: "2026-07-13", venue: "中京", surface: "芝", going: "稍重", tsi: 97.6, moisture_pct: 19.1 },
  { date: "2026-07-20", venue: "阪神", surface: "ダ", going: "良", tsi: 102.1, moisture_pct: 9.4 },
  { date: "2026-07-27", venue: "東京", surface: "芝", going: "良", tsi: 101.5, moisture_pct: 11.2 },
  { date: "2026-08-03", venue: "中京", surface: "ダ", going: "良", tsi: 103.4, moisture_pct: 8.8 },
];

// ---------------------------------------------------------------------------
// Bloodline clusters
// ---------------------------------------------------------------------------
export const MOCK_BLOODLINE_CLUSTERS = [
  { cluster_id: "L2-001", label: "スピード型芝短距離", horse_count: 312, best_courses: ["東京1400", "中山1600", "阪神1600"], running_style: "逃げ・先行", turf_affinity: 0.94, dirt_affinity: 0.21, distance_range: "1200-1600", key_sires: ["ロードカナロア", "グランアレグリア"] },
  { cluster_id: "L2-002", label: "クラシック型中距離", horse_count: 487, best_courses: ["東京2400", "京都2200", "阪神2000"], running_style: "先行・差し", turf_affinity: 0.91, dirt_affinity: 0.32, distance_range: "1800-2400", key_sires: ["ディープインパクト", "エピファネイア"] },
  { cluster_id: "L2-003", label: "ダート万能型", horse_count: 264, best_courses: ["東京ダ1600", "大井1800", "中山ダ1800"], running_style: "先行・逃げ", turf_affinity: 0.43, dirt_affinity: 0.89, distance_range: "1400-1800", key_sires: ["クロフネ", "ゴールドアリュール"] },
  { cluster_id: "L2-004", label: "長距離スタミナ型", horse_count: 198, best_courses: ["天皇賞春3200", "京都2400", "阪神2400"], running_style: "差し・追い込み", turf_affinity: 0.97, dirt_affinity: 0.12, distance_range: "2200-3200", key_sires: ["ハーツクライ", "オルフェーヴル"] },
  { cluster_id: "L2-005", label: "マイル特化万能型", horse_count: 341, best_courses: ["阪神1600", "東京1600", "中京1600"], running_style: "先行・差し", turf_affinity: 0.88, dirt_affinity: 0.48, distance_range: "1400-1800", key_sires: ["モーリス", "キングカメハメハ"] },
];

// ---------------------------------------------------------------------------
// Sire tree (for D3 pedigree map)
// ---------------------------------------------------------------------------
export type SireNode = { id: string; name: string; children?: SireNode[] };
export const MOCK_SIRE_TREE: SireNode = {
  id: "n01", name: "サンデーサイレンス",
  children: [
    {
      id: "n02", name: "ディープインパクト",
      children: [
        { id: "n05", name: "ジャスタウェイ" },
        { id: "n06", name: "サトノダイヤモンド" },
        { id: "n07", name: "マカヒキ" },
      ],
    },
    {
      id: "n03", name: "ステイゴールド",
      children: [
        { id: "n08", name: "オルフェーヴル" },
        { id: "n09", name: "ゴールドシップ" },
      ],
    },
    {
      id: "n04", name: "フジキセキ",
      children: [
        { id: "n10", name: "イスラボニータ" },
        { id: "n11", name: "カネヒキリ" },
      ],
    },
  ],
};

// ---------------------------------------------------------------------------
// Myostatin gene
// ---------------------------------------------------------------------------
export const MOCK_MYOSTATIN = [
  { genotype: "CC (speed type)", count: 124, pct: 31.2, best_distance: "1200-1600", stamina_index: 0.41, vo2max_est: 82.3 },
  { genotype: "CT (balanced)", count: 198, pct: 49.7, best_distance: "1600-2000", stamina_index: 0.67, vo2max_est: 78.1 },
  { genotype: "TT (stamina type)", count: 76, pct: 19.1, best_distance: "2000-3200", stamina_index: 0.91, vo2max_est: 71.4 },
];

// ---------------------------------------------------------------------------
// Kelly criterion result
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Race detail mock  (/race/[id])
// ---------------------------------------------------------------------------

export function getMockRaceDetail(raceId: string) {
  const race = MOCK_WEEKLY_RACES.find((r) => r.race_id === raceId) ?? MOCK_WEEKLY_RACES[0];
  const baseSec = race.surface === "芝" ? (race.distance as number) / 1000 * 60 + 58.0 : (race.distance as number) / 1000 * 60 + 62.0;
  return {
    race_id: raceId,
    race_name: race.race_name,
    venue: race.venue,
    round: race.round,
    date: new Date().toISOString().slice(0, 10),
    surface: race.surface,
    distance: race.distance,
    direction: "右",
    weather: "晴",
    track_condition: "良",
    start_time: "15:30",
    grade: race.grade,
    race_shutuba: {
      entries: MOCK_HORSES.slice(0, race.field_size ?? 16).map((h) => ({
        horse_number: h.post_no,
        bracket_number: calcBracket(h.post_no),
        horse_name: h.horse_name,
        horse_id: h.horse_id,
        sex_age: h.post_no % 2 === 0 ? "牡4" : "牝4",
        jockey_name: MOCK_JOCKEYS[h.post_no % MOCK_JOCKEYS.length].jockey_name,
        trainer_name: "友道康夫",
        jockey_weight: 55,
        win_odds: h.predicted_win_odds,
      })),
    },
    race_result: {
      entries: MOCK_HORSES.slice(0, race.field_size ?? 16).map((h, i) => {
        const mmm = Math.floor(baseSec / 60);
        const sss = (baseSec % 60 + i * 0.3).toFixed(1);
        return {
          horse_number: h.post_no,
          bracket_number: calcBracket(h.post_no),
          horse_name: h.horse_name,
          horse_id: h.horse_id,
          finish_position: i + 1,
          time: `${mmm}:${sss.padStart(4, "0")}`,
          jockey_name: MOCK_JOCKEYS[h.post_no % MOCK_JOCKEYS.length].jockey_name,
          win_odds: h.predicted_win_odds,
          passing_order: ["1-1-1", "3-3-3", "5-5-4", "8-8-7", "2-2-2"][i % 5],
          last3f: (33.8 + i * 0.25).toFixed(1),
        };
      }),
    },
  };
}

// ---------------------------------------------------------------------------
// Tracking difficulty mock  (/tracking-difficulty, /race/[id])
// ---------------------------------------------------------------------------

export function getMockTdData(raceId: string) {
  const race = MOCK_WEEKLY_RACES.find((r) => r.race_id === raceId) ?? MOCK_WEEKLY_RACES[0];
  const easeLabels = ["非常に楽", "楽", "普通", "やや困難", "困難"] as const;
  return {
    race_id: raceId,
    race_name: race.race_name,
    entries: MOCK_TRACKING_DIFFICULTY.map((td, i) => ({
      horse_number: i + 1,
      horse_id: td.horse_id,
      horse_name: td.horse_name,
      bracket_number: calcBracket(i + 1),
      tracking_difficulty: {
        ease_score: td.ease_score,
        ease_pct: td.ease_score,
        ease_label: easeLabels[Math.min(4, Math.floor((100 - td.ease_score) / 20))],
        flow_position: td.position_label,
        flow_sub: td.pace_sensitivity < 0.3 ? "主導権" : "",
        t1f_norm: td.pace_sensitivity,
        expected_last3f: {
          seconds: parseFloat((34.5 + (100 - td.ease_score) / 25).toFixed(1)),
          delta_sec: parseFloat(((td.ease_score - 70) / -50).toFixed(2)),
          rank: i + 1,
          label: td.ease_score >= 70 ? "速" : "普通",
        },
      },
      profile: {
        style: td.position_label,
        style_jra: td.position_label,
      },
    })),
    pace_prediction: {
      pace_type: "ミドルペース",
      pace_comment: "先行馬が多く中団前目が有利。差し馬は後半の末脚を要求される。",
    },
  };
}

// ---------------------------------------------------------------------------
// Race quality mock  (/race-quality)
// ---------------------------------------------------------------------------

const MOCK_RACE_AXES = [
  { label_ja: "スピード特化型" },
  { label_ja: "持久力均衡型" },
  { label_ja: "瞬発力炸裂型" },
  { label_ja: "ダート巧者型" },
  { label_ja: "長距離スタミナ型" },
  { label_ja: "先行有利型" },
  { label_ja: "差し追い込み型" },
  { label_ja: "逃げ主導型" },
  { label_ja: "ハンデ差崩し型" },
];

const BASE_PROBS = [0.22, 0.18, 0.15, 0.12, 0.10, 0.09, 0.07, 0.04, 0.03];

function shiftProbs(seed: number): number[] {
  return BASE_PROBS.map((p, i) => Math.max(0.01, p + ((seed + i) % 7 - 3) * 0.01));
}

export function getMockRaceQuality(raceId: string) {
  const race = MOCK_WEEKLY_RACES.find((r) => r.race_id === raceId) ?? MOCK_WEEKLY_RACES[0];
  const seed = raceId.charCodeAt(raceId.length - 1);
  return {
    race_id: raceId,
    race_name: race.race_name,
    axes: MOCK_RACE_AXES,
    probs: shiftProbs(seed),
    r2_fit: parseFloat((0.72 + (seed % 6) * 0.03).toFixed(2)),
    n_runners: race.field_size ?? 14,
  };
}

export function getMockRaceQualityDay(date: string) {
  const seed = date ? date.charCodeAt(date.length - 1) : 3;
  return {
    date,
    day_summary: {
      axes: MOCK_RACE_AXES,
      probs: shiftProbs(seed),
      n_races: MOCK_WEEKLY_RACES.length,
    },
    by_segment: {
      [`芝1600〜2000`]: {
        n_races: 3,
        axes: MOCK_RACE_AXES,
        probs: shiftProbs(seed + 1),
      },
      [`ダート1400〜1800`]: {
        n_races: 2,
        axes: MOCK_RACE_AXES,
        probs: shiftProbs(seed + 2),
      },
    },
    races: MOCK_WEEKLY_RACES.map((race, idx) => ({
      race_id: race.race_id,
      race_name: race.race_name,
      venue: race.venue,
      distance: race.distance,
      surface: race.surface,
      track_condition: "良",
      segment_key: `${race.surface}${race.distance}`,
      n_runners: race.field_size ?? 14,
      r2_fit: parseFloat((0.70 + idx * 0.02).toFixed(2)),
      probs: shiftProbs(seed + idx),
      axes: MOCK_RACE_AXES,
      pace_shape: { grind_index: 0.45, burst_index: 0.38, lap_evenness: 0.62 },
    })),
  };
}

// ---------------------------------------------------------------------------
// Weekly predictions mock
// ---------------------------------------------------------------------------

/** 今日を基準に直近3開催日（土日）の yyyyMMdd 形式リストを返す */
export function getMockRaceDates(): string[] {
  const result: string[] = [];
  const today = new Date();
  let cursor = new Date(today);
  while (result.length < 3) {
    const dow = cursor.getDay();
    if (dow === 0 || dow === 6) {
      const y = cursor.getFullYear();
      const m = String(cursor.getMonth() + 1).padStart(2, "0");
      const d = String(cursor.getDate()).padStart(2, "0");
      result.push(`${y}${m}${d}`);
    }
    cursor.setDate(cursor.getDate() + 1);
  }
  return result;
}

export const MOCK_WEEKLY_RACES = [
  { race_id: "m_r01", race_name: "東京メトロポリタンS", venue: "東京", round: 11, distance: 2000, surface: "芝", grade: "G3", field_size: 16 },
  { race_id: "m_r02", race_name: "阪神マイルCS", venue: "阪神", round: 10, distance: 1600, surface: "芝", grade: "G2", field_size: 18 },
  { race_id: "m_r03", race_name: "中京ダービー", venue: "中京", round: 9, distance: 1400, surface: "ダ", grade: "G3", field_size: 14 },
  { race_id: "m_r04", race_name: "3歳未勝利", venue: "東京", round: 3, distance: 1800, surface: "芝", grade: "", field_size: 14 },
  { race_id: "m_r05", race_name: "3歳未勝利", venue: "阪神", round: 4, distance: 1200, surface: "ダ", grade: "", field_size: 16 },
  { race_id: "m_r06", race_name: "古馬1勝クラス", venue: "中京", round: 6, distance: 1600, surface: "芝", grade: "", field_size: 15 },
  { race_id: "m_r07", race_name: "古馬2勝クラス", venue: "東京", round: 8, distance: 2200, surface: "芝", grade: "", field_size: 13 },
  { race_id: "m_r08", race_name: "古馬3勝クラス", venue: "阪神", round: 7, distance: 1800, surface: "ダ", grade: "", field_size: 12 },
];

/** モック予測データ（MOCK_HORSES を予測形式に変換） */
export function getMockPredictions(raceId: string) {
  const seed = raceId.charCodeAt(raceId.length - 1);
  return {
    status: "ok",
    has_prediction: true,
    model_description: "MockModel v1.0（開発確認用ダミーデータ）",
    total_horses: MOCK_HORSES.length,
    predictions: MOCK_HORSES.map((h, i) => {
      const rank = ((i + seed) % MOCK_HORSES.length) + 1;
      const marks = ["honmei", "pair", "anchor", "show_val", "star"];
      return {
        horse_number: h.post_no,
        horse_name: h.horse_name,
        horse_id: h.horse_id,
        mark_type: rank <= 5 ? marks[rank - 1] : "none",
        pred_rank: rank,
        composite_rank: rank,
        win_prob: h.win_prob,
        top2_prob: h.place_prob,
        top3_prob: h.show_prob,
        ev_win: h.win_roi,
        ev_place: h.show_roi,
        win_odds: h.predicted_win_odds,
        place_odds_min: h.predicted_place_odds - 0.3,
        place_odds_max: h.predicted_place_odds + 0.5,
        buy_tier: h.is_value_bet ? "A" : rank <= 3 ? "B" : "C",
      };
    }),
  };
}

// ---------------------------------------------------------------------------
// Megu index predicted mock  (/megu-index)
// ---------------------------------------------------------------------------

const MOCK_COND_SCENARIOS = [
  { type: "none",     label: null,          delta_mean: null,  delta_std: null,  transfer_sample_count: undefined as number | undefined },
  { type: "none",     label: null,          delta_mean: null,  delta_std: null,  transfer_sample_count: undefined as number | undefined },
  { type: "none",     label: null,          delta_mean: null,  delta_std: null,  transfer_sample_count: undefined as number | undefined },
  { type: "surface",  label: "芝→ダ初",     delta_mean: -5.2,  delta_std: 3.1,   transfer_sample_count: 42 as number | undefined },
  { type: "distance", label: "+700m",        delta_mean: -3.8,  delta_std: 2.4,   transfer_sample_count: 67 as number | undefined },
  { type: "none",     label: null,          delta_mean: null,  delta_std: null,  transfer_sample_count: undefined as number | undefined },
  { type: "both",     label: "芝→ダ+800m",  delta_mean: -8.1,  delta_std: 4.2,   transfer_sample_count: 18 as number | undefined },
  { type: "none",     label: null,          delta_mean: null,  delta_std: null,  transfer_sample_count: undefined as number | undefined },
];

function mockDistBand(d: number): string {
  if (d <= 1400) return "sprint";
  if (d <= 1800) return "mile";
  if (d <= 2200) return "middle";
  return "long";
}

export function getMockMeguPredicted(raceId: string) {
  const race = MOCK_WEEKLY_RACES.find(r => r.race_id === raceId) ?? MOCK_WEEKLY_RACES[0];
  const isTurf = race.surface === "芝";
  const parSec = isTurf
    ? (race.distance as number) / 1000 * 60 + 58.0
    : (race.distance as number) / 1000 * 62.0 + 4.0;

  const HIST_VENUES = ["東京", "阪神", "中京", "京都", "新潟"];
  const HIST_SURFACES = ["芝", "芝", "芝", "ダ", "芝"];
  const HIST_DATES = ["2026-06-28", "2026-06-07", "2026-05-18", "2026-04-27", "2026-03-29"];
  const HIST_DISTS = [1600, 2000, 1800, 1400, 2200];

  const MOCK_SEX_AGE = [
    { sex: "牡", age: 4 }, { sex: "牝", age: 3 }, { sex: "セ", age: 5 },
    { sex: "牡", age: 3 }, { sex: "牝", age: 4 }, { sex: "牡", age: 5 },
    { sex: "セ", age: 6 }, { sex: "牝", age: 3 }, { sex: "牡", age: 4 },
    { sex: "牝", age: 5 }, { sex: "セ", age: 4 }, { sex: "牡", age: 3 },
    { sex: "牝", age: 6 }, { sex: "牡", age: 4 }, { sex: "セ", age: 3 },
    { sex: "牝", age: 4 },
  ];

  const horses = MOCK_HORSES.slice(0, (race.field_size as number) ?? 16).map((h, i) => {
    const baseMegu = parseFloat((88 + h.win_prob * 80 + (i % 4) * 1.5).toFixed(1));
    const cc = MOCK_COND_SCENARIOS[i % MOCK_COND_SCENARIOS.length];
    const adjustedMegu = cc.delta_mean != null
      ? parseFloat((baseMegu + cc.delta_mean).toFixed(1))
      : baseMegu;
    const finishSec = parseFloat((parSec + i * 0.3).toFixed(1));
    const sa = MOCK_SEX_AGE[i % MOCK_SEX_AGE.length];

    const jockeyWeight = 55 + (i % 5 === 0 ? 2 : 0);
    const weightDelta = jockeyWeight > 55 ? -parseFloat(((jockeyWeight - 55) * 6.1).toFixed(1)) : 0;
    const meguFinal = parseFloat((adjustedMegu + weightDelta).toFixed(1));
    // モック: 1R目のみ結果確定済みとして実測値を付与（想定との差を再現）
    const actualMegu = raceId === "m_r01"
      ? parseFloat((meguFinal + (i % 3 === 0 ? 2.5 : i % 3 === 1 ? -1.8 : 0.6)).toFixed(1))
      : null;

    return {
      horse_id: h.horse_id,
      horse_name: h.horse_name,
      horse_number: h.post_no,
      bracket_number: calcBracket(h.post_no),
      sex_age: `${sa.sex}${sa.age}`,
      sex: sa.sex,
      age: sa.age,
      jockey_weight: jockeyWeight,
      finish_time_sec: raceId === "m_r01" ? finishSec : null,
      finish_pos: raceId === "m_r01" ? i + 1 : null,
      actual_megu: actualMegu,
      base_megu: baseMegu,
      megu_adjusted: adjustedMegu,
      weight_megu_delta: weightDelta,
      megu_final: meguFinal,
      condition_change: {
        type: cc.type,
        label: cc.label,
        delta_mean: cc.delta_mean,
        delta_std: cc.delta_std,
        transfer_sample_count: cc.transfer_sample_count,
      },
      history: HIST_DATES.map((date, j) => ({
        race_id: `hist_${h.horse_id}_${j}`,
        race_date: date,
        venue: HIST_VENUES[j],
        surface: HIST_SURFACES[j],
        distance: HIST_DISTS[j],
        megu_index: parseFloat((baseMegu - j * 1.8 + (i % 3) * 0.5).toFixed(1)),
        finish_pos: (j + i % 5) % 8 + 1,
      })),
    };
  });

  const mockLevelLabel =
    race.grade === "G1" ? "G1級"
    : race.grade === "G2" || race.grade === "G3" ? "重賞級"
    : race.grade === "OP" ? "3勝級"
    : race.grade === "2勝" ? "2勝級"
    : race.grade === "1勝" ? "1勝級"
    : "3勝級";

  return {
    race_id: raceId,
    race_info: {
      race_name: race.race_name,
      venue: race.venue,
      surface: race.surface,
      distance: race.distance as number,
      dist_band: mockDistBand(race.distance as number),
      track_condition: "良",
      grade: (race.grade as string) || null,
      race_date: "2026-07-12",
    },
    race_level: { label: mockLevelLabel, field_avg_megu: 102.4 },
    model_version: "mock-v1",
    horses,
  };
}

export const MOCK_KELLY = {
  race_id: "r001",
  bankroll: 100000,
  total_bet: 18200,
  expected_profit: 3640,
  kelly_fraction: 0.182,
  bets: [
    { horse_id: "h03", horse_name: "ゴールドストーム", bet_type: "単勝", stake: 8200, kelly_f: 0.082, edge: 0.093 },
    { horse_id: "h01", horse_name: "アイアンウィル", bet_type: "複勝", stake: 6000, kelly_f: 0.060, edge: 0.071 },
    { horse_id: "h10", horse_name: "サンライズキング", bet_type: "複勝", stake: 4000, kelly_f: 0.040, edge: 0.051 },
  ],
};

// ---------------------------------------------------------------------------
// Betting simulation mock params  (/betting-simulation)
// ---------------------------------------------------------------------------
export const MOCK_SIMULATION_PARAMS = {
  bankroll:       100_000,
  win_prob:       0.200,   // 20%
  win_odds:       5.5,     // 5.5倍
  kelly_fraction: 0.25,    // Quarter Kelly
  n_races:        100,
  n_trials:       1_000,
  ruin_threshold: 0.10,    // 初期軍資金の10%
};
