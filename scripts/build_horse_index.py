#!/usr/bin/env python3
"""
馬名インデックスファイルを生成するスクリプト

使用方法:
    python3 scripts/build_horse_index.py
"""

import json
import time
from pathlib import Path

# プロジェクトルート
BASE_DIR = Path(__file__).resolve().parent.parent


def build_horse_index():
    """ローカルキャッシュから馬名インデックスを構築"""

    horse_index = []
    cache_dir = BASE_DIR / "data" / "cache" / "horse_result"

    if not cache_dir.exists():
        print(f"キャッシュディレクトリが見つかりません: {cache_dir}")
        return

    total = 0
    errors = 0

    # 各年度のディレクトリを走査
    for year_dir in sorted(cache_dir.iterdir()):
        if not year_dir.is_dir():
            continue

        year = year_dir.name
        print(f"処理中: {year}年度...")

        count = 0
        for json_file in year_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                horse_id = data.get("horse_id")
                horse_name = data.get("horse_name")
                name_en = data.get("name_en", "")

                if horse_id and horse_name:
                    horse_index.append({
                        "horse_id": horse_id,
                        "horse_name": horse_name,
                        "name_en": name_en,
                        "year": year
                    })
                    count += 1
                    total += 1

            except Exception as e:
                errors += 1
                if errors <= 10:  # 最初の10個のエラーのみ表示
                    print(f"  エラー ({json_file.name}): {e}")

        print(f"  {count}頭を登録")

    # インデックスファイルを保存
    output_path = BASE_DIR / "data" / "knowledge" / "horse_name_index.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "version": "1.0",
            "total_horses": total,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "horses": horse_index
        }, f, ensure_ascii=False, indent=2)

    print(f"\n完了: {total}頭のインデックスを生成しました")
    print(f"出力先: {output_path}")
    print(f"エラー数: {errors}")


if __name__ == "__main__":
    build_horse_index()
