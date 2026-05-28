#!/usr/bin/env python3
"""
馬名検索用 horse_name_index.json を生成する。

正規の出力先: ``data/knowledge/horse_name_index.json``（API は常にこのローカルファイルのみ読む）。

- 既定: ローカルの ``data/cache/horse_result`` と ``data/local/mirror/horse_result`` だけを走査。
- ``--from-gcs``: メンテナンス用。GCS の ``horse_result`` を全走査して同じローカル JSON に書き出す
  （API サーバの高頻度パスでは使わないこと）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# プロジェクトルート（本ファイル: src/scripts/data/build_horse_index.py）
BASE_DIR = Path(__file__).resolve().parents[3]


def main() -> int:
    ap = argparse.ArgumentParser(description="馬名検索用 horse_name_index.json を生成")
    ap.add_argument(
        "--from-gcs",
        action="store_true",
        help="GCS の horse_result を全走査（ローカルキャッシュが無い場合の初回向け。時間がかかります）",
    )
    args = ap.parse_args()

    sys.path.insert(0, str(BASE_DIR))
    from src.utils.horse_name_index import (
        rebuild_horse_name_index_from_gcs_horse_result,
        rebuild_horse_name_index_from_horse_result_cache,
    )

    if args.from_gcs:
        n = rebuild_horse_name_index_from_gcs_horse_result(BASE_DIR)
    else:
        n = rebuild_horse_name_index_from_horse_result_cache(BASE_DIR)

    out = BASE_DIR / "data" / "knowledge" / "horse_name_index.json"
    print(f"完了: {n}頭のインデックスを {out} に書き出しました")
    if n == 0 and not args.from_gcs:
        print(
            "注意: 0 頭の場合は --from-gcs で GCS から再構築するか、"
            "horse_result のスクレイプ後に自動追記されるまでお待ちください。"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
