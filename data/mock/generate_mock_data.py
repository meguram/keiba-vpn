"""
netkeiba.com のデータ構造を模した競馬モックデータを生成する。

生成するデータ:
  - races.csv: レース情報
  - horses.csv: 出走馬情報（レースごとの成績を含む）
"""

import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

RACECOURSES = [
    ("東京", 1600, "芝"), ("中山", 2000, "芝"), ("阪神", 1800, "芝"),
    ("京都", 2400, "芝"), ("中京", 2000, "芝"), ("小倉", 1200, "芝"),
    ("新潟", 1800, "芝"), ("札幌", 1500, "芝"), ("函館", 1200, "芝"),
    ("福島", 2000, "芝"), ("東京", 1400, "ダート"), ("中山", 1800, "ダート"),
    ("阪神", 1400, "ダート"), ("京都", 1900, "ダート"),
]

HORSE_NAMES = [
    "サンデーブレイク", "ゴールドシップ二世", "ディープスカイ", "キタサンミラクル",
    "アーモンドフラッシュ", "コントレイルスター", "イクイノックスジュニア", "リバティアイランドⅡ",
    "ジャスティンロック", "ソールオリエンス", "タスティエーラ二世", "ドウデュースジュニア",
    "スターズオンアース", "エフフォーリアネクスト", "シャフリヤールⅡ", "タイトルホルダー二世",
    "パンサラッサネオ", "ジオグリフライト", "ナミュールスター", "スタニングローズ二世",
    "リバティゴールド", "ウインマリリン", "デアリングタクト二世", "クロノジェネシスⅡ",
    "レイパパレスター", "カフェファラオ二世", "メイケイエール", "ソダシスプリング",
    "シュネルマイスター二世", "グランアレグリアⅡ", "ファインルージュ", "アカイイト二世",
    "ジェラルディーナ", "ヴェラアズール二世", "ボルドグフーシュ", "アスクビクターモア二世",
    "ガイアフォース", "セリフォスライト", "ダノンベルーガⅡ", "プラダリアスター",
    "オニャンコポン二世", "ドゥラエレーデ", "キラーアビリティⅡ", "レッドベルオーブ",
    "アルファネクスト", "ブレイブジェネシス", "サクラトップラン", "マイティウインド",
    "トウショウファルコ", "キングスレイン",
]

JOCKEYS = [
    "武豊", "C.ルメール", "川田将雅", "横山武史", "戸崎圭太",
    "松山弘平", "M.デムーロ", "岩田望来", "坂井瑠星", "田辺裕信",
    "福永祐一", "池添謙一", "浜中俊", "石橋脩", "三浦皇成",
]

TRAINERS = [
    "矢作芳人", "国枝栄", "堀宣行", "木村哲也", "友道康夫",
    "藤原英昭", "中内田充正", "手塚貴久", "萩原清", "須貝尚介",
]

WEATHER_OPTIONS = ["晴", "曇", "小雨", "雨"]
TRACK_CONDITIONS = ["良", "稍重", "重", "不良"]
GRADE_OPTIONS = ["G1", "G2", "G3", "OP", "3勝", "2勝", "1勝", "未勝利"]


def _generate_ability(grade_weight: float) -> float:
    """馬の能力値を生成（グレードの高いレースほど高い能力を持つ馬が集まる）"""
    return np.clip(np.random.normal(50 + grade_weight * 10, 15), 10, 100)


def generate_races(n_races: int = 200) -> pd.DataFrame:
    races = []
    start_date = datetime(2025, 1, 4)
    for i in range(n_races):
        race_date = start_date + timedelta(days=random.randint(0, 365))
        course_info = random.choice(RACECOURSES)
        grade = random.choices(
            GRADE_OPTIONS, weights=[2, 3, 5, 8, 12, 15, 25, 30], k=1
        )[0]
        race_id = f"2025{race_date.strftime('%m%d')}{i+1:04d}"
        races.append({
            "race_id": race_id,
            "race_name": f"第{random.randint(1, 60)}回{course_info[0]}{'記念' if grade in ('G1', 'G2') else '特別' if grade in ('G3', 'OP') else 'ステークス'}",
            "date": race_date.strftime("%Y-%m-%d"),
            "racecourse": course_info[0],
            "distance": course_info[1] + random.choice([-200, 0, 0, 0, 200]),
            "surface": course_info[2],
            "weather": random.choices(WEATHER_OPTIONS, weights=[50, 30, 15, 5], k=1)[0],
            "track_condition": random.choices(TRACK_CONDITIONS, weights=[45, 30, 15, 10], k=1)[0],
            "grade": grade,
            "num_runners": random.randint(8, 18),
        })
    return pd.DataFrame(races)


def generate_horse_results(races_df: pd.DataFrame) -> pd.DataFrame:
    all_results = []
    grade_weight_map = {
        "G1": 4, "G2": 3, "G3": 2.5, "OP": 2,
        "3勝": 1.5, "2勝": 1, "1勝": 0.5, "未勝利": 0,
    }

    for _, race in races_df.iterrows():
        n_runners = race["num_runners"]
        selected_horses = random.sample(HORSE_NAMES, min(n_runners, len(HORSE_NAMES)))
        grade_w = grade_weight_map.get(race["grade"], 1)

        abilities = np.array([_generate_ability(grade_w) for _ in selected_horses])

        # 馬場状態による影響
        if race["track_condition"] in ("重", "不良"):
            abilities += np.random.normal(0, 8, len(abilities))

        order = np.argsort(-abilities)
        finish_positions = np.empty_like(order)
        finish_positions[order] = np.arange(1, len(order) + 1)

        for idx, horse_name in enumerate(selected_horses):
            age = random.randint(2, 7)
            weight = random.randint(430, 540)
            jockey = random.choice(JOCKEYS)
            trainer = random.choice(TRAINERS)
            post_position = idx + 1
            odds = max(1.1, np.random.exponential(10) + (finish_positions[idx] - 1) * 2.5)
            popularity = int(np.clip(finish_positions[idx] + random.randint(-3, 3), 1, n_runners))

            base_time = race["distance"] / 16.5
            finish_time = base_time + (finish_positions[idx] - 1) * 0.3 + random.uniform(-0.5, 0.5)

            weight_change = random.choice([-8, -6, -4, -2, 0, 0, 0, 2, 4, 6, 8])
            career_wins = random.randint(0, 10)
            career_starts = career_wins + random.randint(1, 30)

            all_results.append({
                "race_id": race["race_id"],
                "horse_name": horse_name,
                "horse_number": post_position,
                "jockey": jockey,
                "trainer": trainer,
                "age": age,
                "sex": random.choice(["牡", "牝", "セ"]),
                "weight": weight,
                "weight_change": weight_change,
                "jockey_weight": random.choice([54.0, 55.0, 56.0, 57.0, 58.0]),
                "post_position": post_position,
                "odds": round(odds, 1),
                "popularity": popularity,
                "finish_position": int(finish_positions[idx]),
                "finish_time": round(finish_time, 1),
                "margin": round(max(0, (finish_positions[idx] - 1) * 0.3 + random.uniform(-0.2, 0.2)), 1),
                "career_starts": career_starts,
                "career_wins": career_wins,
                "career_top3": career_wins + random.randint(0, career_starts - career_wins),
                "prize_money": round(random.uniform(0, 30000) * (1 + grade_w), 1),
            })

    return pd.DataFrame(all_results)


def generate_upcoming_races(n_races: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """未来のレース（予測対象）を生成"""
    races = []
    entries = []
    base_date = datetime(2026, 3, 22)

    for i in range(n_races):
        race_date = base_date + timedelta(days=i % 2)
        course_info = random.choice(RACECOURSES)
        grade = random.choices(GRADE_OPTIONS[:5], weights=[5, 8, 12, 20, 30], k=1)[0]
        race_id = f"2026{race_date.strftime('%m%d')}{i+1:04d}"
        n_runners = random.randint(10, 18)

        races.append({
            "race_id": race_id,
            "race_name": f"第{random.randint(1, 30)}回{course_info[0]}{'記念' if grade in ('G1','G2') else '特別'}",
            "date": race_date.strftime("%Y-%m-%d"),
            "racecourse": course_info[0],
            "distance": course_info[1],
            "surface": course_info[2],
            "weather": "晴",
            "track_condition": "良",
            "grade": grade,
            "num_runners": n_runners,
        })

        selected_horses = random.sample(HORSE_NAMES, min(n_runners, len(HORSE_NAMES)))
        for idx, horse_name in enumerate(selected_horses):
            entries.append({
                "race_id": race_id,
                "horse_name": horse_name,
                "horse_number": idx + 1,
                "jockey": random.choice(JOCKEYS),
                "trainer": random.choice(TRAINERS),
                "age": random.randint(3, 6),
                "sex": random.choice(["牡", "牝", "セ"]),
                "weight": random.randint(440, 530),
                "weight_change": random.choice([-4, -2, 0, 0, 2, 4]),
                "jockey_weight": random.choice([54.0, 55.0, 56.0, 57.0]),
                "post_position": idx + 1,
                "odds": round(max(1.1, np.random.exponential(12)), 1),
                "popularity": 0,
                "career_starts": random.randint(3, 30),
                "career_wins": random.randint(0, 8),
                "career_top3": random.randint(0, 15),
                "prize_money": round(random.uniform(100, 20000), 1),
            })

    return pd.DataFrame(races), pd.DataFrame(entries)


def main():
    print("モックデータを生成中...")

    races_df = generate_races(200)
    results_df = generate_horse_results(races_df)

    upcoming_races_df, upcoming_entries_df = generate_upcoming_races(5)

    races_df.to_csv(os.path.join(OUTPUT_DIR, "races.csv"), index=False, encoding="utf-8-sig")
    results_df.to_csv(os.path.join(OUTPUT_DIR, "results.csv"), index=False, encoding="utf-8-sig")
    upcoming_races_df.to_csv(os.path.join(OUTPUT_DIR, "upcoming_races.csv"), index=False, encoding="utf-8-sig")
    upcoming_entries_df.to_csv(os.path.join(OUTPUT_DIR, "upcoming_entries.csv"), index=False, encoding="utf-8-sig")

    print(f"  races.csv: {len(races_df)} レース")
    print(f"  results.csv: {len(results_df)} 出走結果")
    print(f"  upcoming_races.csv: {len(upcoming_races_df)} 予測対象レース")
    print(f"  upcoming_entries.csv: {len(upcoming_entries_df)} 出走エントリ")
    print("完了!")


if __name__ == "__main__":
    main()
