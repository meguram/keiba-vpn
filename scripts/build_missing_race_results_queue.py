#!/usr/bin/env python3
"""
2020年以降の全レースについて、GCS に以下のいずれかが無いものだけをキュー JSON に書き出す。

  - race_result        … 「DB結果」（db.netkeiba 確定成績）
  - race_result_on_time … 「速報結果」（race.netkeiba 当日系）

レース一覧は ``data/page_reference/tables/{年}/race_result_flat.parquet`` を優先し、
無ければ ``data/local/tables/{年}/race_result_flat.parquet`` を参照する。

GCS のキー集合は ``HybridStorage.batch_list_blobs`` により年単位で取得する
（ネットワークあり。キャッシュは ``data/cache/_blob_list/`` に書き込まれる）。

出力は ``scrape_queue.json`` と同じラッパ形式::

  {"jobs": [...], "updated_at": "..."}

別環境へ反映する例::

  # 生成（プロジェクトルートで）
  python3 scripts/build_missing_race_results_queue.py \\
    --output data/queue/scrape_queue_backfill_race_result_pair_2020plus.json

  # リモートの data/queue/scrape_queue.json をこのファイルで置き換え
  # （既存キューは消えるので、専用ワーカー／メンテ用VPS向け）
  scp data/queue/scrape_queue_backfill_race_result_pair_2020plus.json \\
      user@vps:/path/to/keiba-vpn/data/queue/scrape_queue.json

マージして既存ジョブを残したい場合は、生成後に jq 等で ``jobs`` 配列を結合するか、
手動で API / スクリプトから bulk 投入してください。
"""
from __future__ import annotations

import argparse
import json
import secrets
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _collect_race_ids_from_parquet(
    tables_root: Path,
    year_from: int,
    year_to: int,
) -> dict[str, set[str]]:
    """年 -> race_id の集合（12桁数字のみ）"""
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise SystemExit("pyarrow が必要です: pip install pyarrow") from e

    out: dict[str, set[str]] = {}
    for y in range(year_from, year_to + 1):
        p = tables_root / str(y) / "race_result_flat.parquet"
        if not p.is_file():
            continue
        t = pq.read_table(p, columns=["race_id"])
        s: set[str] = set()
        for r in t.column("race_id").to_pylist():
            rid = str(r or "").strip()
            if len(rid) == 12 and rid.isdigit():
                s.add(rid)
        if s:
            out[str(y)] = s
    return out


def _resolve_tables_root(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    from src.config.data_paths import TABLES_DIR

    if TABLES_DIR.is_dir() and any(TABLES_DIR.glob("*/*.parquet")):
        return TABLES_DIR
    legacy = ROOT / "data" / "local" / "tables"
    return legacy


def _make_job(race_id: str, *, seq: int) -> dict:
    from src.scraper.queue_tasks import build_job_label

    tasks = ["race_result", "race_result_on_time"]
    dedupe = f"race:{race_id}:{':'.join(sorted(set(tasks)))}"
    label = build_job_label("race", race_id, tasks)
    jid = f"q_bf_{int(time.time())}_{seq:08d}_{secrets.token_hex(4)}"
    now = datetime.now().isoformat()
    return {
        "job_id": jid,
        "dedupe_key": dedupe,
        "job_kind": "race",
        "target_id": race_id,
        "tasks": tasks,
        "job_label": label,
        "race_id": race_id,
        "date": "",
        "venue": "",
        "round": 0,
        "race_name": "",
        "types": list(tasks),
        "status": "precheck",
        "queued_at": now,
        "started_at": None,
        "completed_at": None,
        "error": None,
        "smart_skip": True,
        "overwrite": False,
        "priority": 0,
        "skip_local_mirror": False,
        "skip_pedigree": False,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="race_result + race_result_on_time が欠けているレースだけの scrape_queue 形式 JSON を生成する"
    )
    ap.add_argument("--year-from", type=int, default=2020)
    ap.add_argument("--year-to", type=int, default=datetime.now().year)
    ap.add_argument(
        "--tables-dir",
        default="",
        help="race_result_flat.parquet を含む tables ルート（省略時は page_reference/tables → local/tables）",
    )
    ap.add_argument(
        "--output",
        "-o",
        default=str(
            ROOT
            / "data"
            / "queue"
            / "scrape_queue_backfill_race_result_pair_2020plus.json"
        ),
        help="出力パス（scrape_queue.json と同じ {jobs, updated_at} 形式）",
    )
    ap.add_argument("--dry-run", action="store_true", help="件数だけ表示しファイルは書かない")
    ap.add_argument("--max-jobs", type=int, default=0, help="デバッグ用: 出力ジョブ数の上限（0=無制限）")
    args = ap.parse_args()

    if args.year_from > args.year_to:
        ap.error("--year-from は --year-to 以下である必要があります")

    tables_root = _resolve_tables_root(args.tables_dir.strip() or None)
    if not tables_root.is_dir():
        print(f"[ERROR] tables ディレクトリがありません: {tables_root}", file=sys.stderr)
        return 1

    by_year = _collect_race_ids_from_parquet(
        tables_root, args.year_from, args.year_to
    )
    if not by_year:
        print(f"[ERROR] race_result_flat.parquet が見つかりません: {tables_root}", file=sys.stderr)
        return 1

    from src.scraper.storage import HybridStorage

    storage = HybridStorage(base_dir=str(ROOT))
    if not storage.gcs_enabled:
        print(
            "[ERROR] GCS が無効です（GCS_BUCKET 等）。batch_list_blobs でキー集合を取れません。",
            file=sys.stderr,
        )
        return 1

    rr_by_year: dict[str, set[str]] = {}
    rot_by_year: dict[str, set[str]] = {}
    for y in sorted(by_year.keys(), key=int):
        print(f"[INFO] GCS blob 一覧取得: race_result / {y} …", flush=True)
        rr_by_year[y] = set(storage.batch_list_blobs("race_result", y).keys())
        print(f"[INFO] GCS blob 一覧取得: race_result_on_time / {y} …", flush=True)
        rot_by_year[y] = set(storage.batch_list_blobs("race_result_on_time", y).keys())
        print(
            f"       {y}: parquet {len(by_year[y]):,} レース / "
            f"GCS race_result {len(rr_by_year[y]):,} / on_time {len(rot_by_year[y]):,}",
            flush=True,
        )

    missing: list[str] = []
    for y, rids in sorted(by_year.items(), key=lambda x: int(x[0])):
        rr = rr_by_year.get(y, set())
        rot = rot_by_year.get(y, set())
        for rid in rids:
            if rid not in rr or rid not in rot:
                missing.append(rid)

    missing.sort()
    print(
        f"[INFO] 欠損（race_result または race_result_on_time のどちらか一方でも無い）: {len(missing):,} レース"
    )

    if args.dry_run:
        for rid in missing[:20]:
            yr = rid[:4]
            hr = "○" if rid in rr_by_year.get(yr, ()) else "×"
            ho = "○" if rid in rot_by_year.get(yr, ()) else "×"
            print(f"  {rid}  DB={hr} 速報={ho}")
        if len(missing) > 20:
            print(f"  … 他 {len(missing) - 20:,} 件")
        return 0

    jobs: list[dict] = []
    cap = args.max_jobs if args.max_jobs > 0 else len(missing)
    for i, rid in enumerate(missing[:cap]):
        jobs.append(_make_job(rid, seq=i))

    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"jobs": jobs, "updated_at": datetime.now().isoformat()}
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] {len(jobs):,} 件のジョブを書き込みました: {out_path}")
    print(
        "[NOTE] 別環境で data/queue/scrape_queue.json をこのファイルで上書きすると、"
        "その環境の既存キュー行はすべて置き換わります。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
