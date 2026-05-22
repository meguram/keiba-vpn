#!/usr/bin/env python3
"""
既存の data/page_reference/race_performance/races/ 内の JSON ファイルを
GCS にアップロードし、アップロード成功後にローカルファイルを削除するスクリプト。

使い方:
  # ドライラン（アップロードも削除もしない・確認のみ）
  python scripts/migrate_race_performance_to_gcs.py --dry-run

  # 実行（アップロード後にローカルファイルを削除）
  python scripts/migrate_race_performance_to_gcs.py --delete-after-upload

  # アップロードのみ（削除はしない）
  python scripts/migrate_race_performance_to_gcs.py
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description="race_performance JSON を GCS に移行する")
    ap.add_argument("--dry-run", action="store_true", help="確認のみ（実際の操作なし）")
    ap.add_argument(
        "--delete-after-upload",
        action="store_true",
        help="GCS アップロード成功後にローカルファイルを削除する",
    )
    ap.add_argument("--years", default="", help="対象年 (comma separated, 例: 2023,2024). 省略時は全年")
    args = ap.parse_args()

    from src.scraper.storage import HybridStorage

    storage = HybridStorage(base_dir=str(ROOT))
    if not storage.gcs_enabled:
        print("[ERROR] GCS が有効ではありません。GCS_BUCKET 環境変数を確認してください。", file=sys.stderr)
        return 1

    races_root = ROOT / "data" / "page_reference" / "race_performance" / "races"
    if not races_root.exists():
        print(f"[INFO] {races_root} は存在しません。移行対象なし。")
        return 0

    target_years: set[str] = set()
    if args.years.strip():
        target_years = {y.strip() for y in args.years.split(",") if y.strip()}

    all_json: list[Path] = []
    for year_dir in sorted(races_root.iterdir()):
        if not year_dir.is_dir():
            continue
        if target_years and year_dir.name not in target_years:
            continue
        all_json.extend(sorted(year_dir.glob("*.json")))

    total = len(all_json)
    if total == 0:
        print("[INFO] アップロード対象のファイルがありません。")
        return 0

    print(f"[INFO] 対象ファイル数: {total:,}")
    if args.dry_run:
        print("[DRY-RUN] 実際の操作はスキップします。")
        for p in all_json[:10]:
            print(f"  {p.relative_to(ROOT)}")
        if total > 10:
            print(f"  ... 他 {total - 10:,} ファイル")
        return 0

    ok = 0
    failed = 0
    deleted = 0
    t0 = time.monotonic()

    for i, path in enumerate(all_json, 1):
        race_id = path.stem
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] 読み込み失敗 [{race_id}]: {e}")
            failed += 1
            continue

        try:
            storage.save("race_performance", race_id, copy.deepcopy(payload))
            ok += 1
        except Exception as e:
            print(f"[WARN] GCS アップロード失敗 [{race_id}]: {e}")
            failed += 1
            continue

        if args.delete_after_upload:
            try:
                path.unlink()
                deleted += 1
            except Exception as e:
                print(f"[WARN] ローカル削除失敗 [{race_id}]: {e}")

        if i % 500 == 0 or i == total:
            elapsed = time.monotonic() - t0
            rate = i / elapsed if elapsed > 0 else 0
            eta = (total - i) / rate if rate > 0 else 0
            print(
                f"[{i:,}/{total:,}] ok={ok:,} failed={failed:,} deleted={deleted:,} "
                f"速度={rate:.1f}件/秒 ETA={eta:.0f}秒"
            )

    # アップロード後、races/ ディレクトリが空なら削除
    if args.delete_after_upload and deleted > 0:
        for year_dir in sorted(races_root.iterdir()):
            if year_dir.is_dir() and not any(year_dir.iterdir()):
                year_dir.rmdir()
                print(f"[INFO] 空ディレクトリ削除: {year_dir.relative_to(ROOT)}")
        if not any(races_root.iterdir()):
            races_root.rmdir()
            print(f"[INFO] races/ ディレクトリ削除完了: {races_root.relative_to(ROOT)}")

    elapsed_total = time.monotonic() - t0
    print(
        f"\n[完了] アップロード={ok:,} 失敗={failed:,} ローカル削除={deleted:,} "
        f"所要時間={elapsed_total:.1f}秒"
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
