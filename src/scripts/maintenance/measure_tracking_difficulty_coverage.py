#!/usr/bin/env python3
"""追走難度の事前計算カバレッジ（「未計算 (not_precomputed)」に当たる頻度）を計測する。

docs/git_management/todo/tracking-difficulty.md「共通TODO」の
「『未計算（not_precomputed）』に当たる頻度を計測する」に対応するスクリプト。

対象レースの定義:
  - race_shutuba が存在する race_id のうち、中央競馬 (JRA, venue code 01-10) で
    開催日が直近 N 日以内 (既定 365 日) のもの。
  - race_shutuba の有無は ``storage.list_keys("race_shutuba")`` で取得する。これは
    既存バッチ ``src.scripts.maintenance.batch_inference_all_races.collect_race_ids`` /
    ``precompute_tracking_difficulty_all.py`` と同じ「対象レース」定義。

計算済みの定義:
  - ``tracking_difficulty_store.exists_local(race_id)`` が True
    (= ``data/calculated_data/tracking_difficulty/{race_id}.json`` が存在し、
    cache_version が現行バージョンと一致)。

実行例:
  python3 -m src.scripts.maintenance.measure_tracking_difficulty_coverage
  python3 -m src.scripts.maintenance.measure_tracking_difficulty_coverage --days 180
  python3 -m src.scripts.maintenance.measure_tracking_difficulty_coverage --no-probe-dates

注意:
  事前計算ストア (``data/calculated_data/tracking_difficulty/``) は GCS ミラーが既定で
  無効 (``KEIBA_TRACKING_DIFFICULTY_GCS_MIRROR``) のため、このホストのローカルファイルのみを見る。
  race_shutuba データ・事前計算データが無いホスト（開発用の空チェックアウト等）で実行すると
  対象レース数が 0 件になり、未計算率は計測不可として報告される。
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(_ROOT / ".env")

from src.pipeline.inference.tracking_difficulty_store import (  # noqa: E402
    count_local,
    exists_local,
    index_meta,
    store_dir,
)
from src.scraper.storage import HybridStorage  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("MeasureTrackingDifficultyCoverage")

JRA_VENUE_CODES = {f"{i:02d}" for i in range(1, 11)}  # 01-10: 中央競馬 (netkeiba race_id[4:6])


def _is_jra_race_id(race_id: str) -> bool:
    return len(race_id) >= 6 and race_id[4:6] in JRA_VENUE_CODES


def _date_to_yyyymmdd(raw: str) -> str:
    """YYYY-MM-DD / YYYY/MM/DD / YYYYMMDD → YYYYMMDD。

    src.pipeline.models.tracking_difficulty._date_to_yyyymmdd と同じ正規化規則。
    """
    s = str(raw or "").strip()
    if not s:
        return ""
    if len(s) >= 8 and s[:8].isdigit():
        return s[:8]
    m = re.match(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})", s)
    if m:
        return f"{m.group(1)}{int(m.group(2)):02d}{int(m.group(3)):02d}"
    return ""


def collect_target_race_ids(
    storage: HybridStorage,
    *,
    since_yyyymmdd: str,
    until_yyyymmdd: str,
    jra_only: bool = True,
    probe_dates: bool = True,
) -> list[str]:
    """対象レースID一覧（race_shutuba が存在するレースのうち、条件に合うもの）。"""
    keys = storage.list_keys("race_shutuba") or []
    all_ids = sorted(set(k.replace(".json", "") for k in keys if k))

    candidates = []
    for rid in all_ids:
        if jra_only and not _is_jra_race_id(rid):
            continue
        year_prefix = rid[:4]
        # 粗い絞り込み: race_id の年が範囲に重ならなければ即除外（高速化）
        if year_prefix < since_yyyymmdd[:4] or year_prefix > until_yyyymmdd[:4]:
            continue
        candidates.append(rid)

    if not probe_dates:
        return candidates

    out = []
    for rid in candidates:
        shutuba = storage.load("race_shutuba", rid)
        if not shutuba:
            continue
        d = _date_to_yyyymmdd(str(shutuba.get("date", "")))
        if not d:
            # 実日付が取れない場合は年プレフィックスの粗い絞り込みのみで対象に含める
            out.append(rid)
            continue
        if since_yyyymmdd <= d <= until_yyyymmdd:
            out.append(rid)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=365, help="直近何日を対象にするか（既定365日）")
    parser.add_argument(
        "--no-jra-only", action="store_true", help="地方競馬も含める（既定は中央競馬のみ）"
    )
    parser.add_argument(
        "--no-probe-dates",
        action="store_true",
        help="各レースの実日付(date フィールド)を読まず、race_id の年プレフィックスのみで"
        "絞り込む（高速だが粗い）",
    )
    args = parser.parse_args(argv)

    now = datetime.now(timezone.utc)
    until = now.strftime("%Y%m%d")
    since = (now - timedelta(days=args.days)).strftime("%Y%m%d")

    storage = HybridStorage()
    target_ids = collect_target_race_ids(
        storage,
        since_yyyymmdd=since,
        until_yyyymmdd=until,
        jra_only=not args.no_jra_only,
        probe_dates=not args.no_probe_dates,
    )

    total = len(target_ids)
    computed = sum(1 for rid in target_ids if exists_local(rid))
    not_precomputed = total - computed
    rate = (not_precomputed / total * 100.0) if total else None

    print("=== 追走難度 事前計算カバレッジ計測 ===")
    print(f"期間: {since} 〜 {until} ({args.days}日間) / JRA限定: {not args.no_jra_only}")
    print(f"事前計算ストア: {store_dir()} (全件数: {count_local()})")
    print(f"index_meta: {index_meta()}")
    print(f"対象レース数 (race_shutuba あり): {total}")
    print(f"計算済み件数: {computed}")
    print(f"未計算 (not_precomputed) 件数: {not_precomputed}")
    if rate is None:
        print(
            "未計算率: 対象レースが0件のため計測不可"
            "（このホストに race_shutuba データが無い可能性がある。本番/VPSホストで再実行してください）"
        )
    else:
        print(f"未計算率: {rate:.2f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
