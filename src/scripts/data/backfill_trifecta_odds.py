"""3連単オッズ（race_trifecta_odds）の過去分の一括取得（ホストの stg で実行。再開可）。

python -m src.scripts.data.backfill_trifecta_odds --years 2026 --limit 20     # パイロット
python -m src.scripts.data.backfill_trifecta_odds --years 2026 2025 2024      # 新しい年から順に

1 レース 1 リクエスト（約 100KB）。確定後の type=8 は過去レースでも取得できる（2025 年の1レースで確認）。
- 保存済みのレースは取らない（既存は上書きしない）。読めなかったレースは進捗ファイルに記録して次回以降スキップ（--retry-failed で再試行）。
- リクエストは NetkeibaClient のプロセス横断スロットを使い、1 件ごとに 4.5〜7 秒、100 件ごとに 30〜60 秒休む。
- 修復・バックフィル・エクスポート等の他のスクレイピングが動いている間、およびホストの負荷が高い間は待つ（horse_race_remarks の
  取り込みとはスロットを共有して並走する）。連続 8 件の失敗でアクセス制限を疑って中断（終了コード 3）。
進捗: data/local/meta/trifecta_backfill_progress.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TABLES = ROOT / "data/page_reference/tables"
PROGRESS = ROOT / "data/local/meta/trifecta_backfill_progress.json"
OTHER_SCRAPERS = r"python[0-9.]* -m src\.(scripts\.maintenance\.repair_|scripts\.scraping\.(?!backfill_horse_race_remarks)|scraper\.(run|export_tables)|scripts\.data\.backfill_sanrenpuku_odds)"  # scraper.backfill（馬ページ・長時間稼働）は待たない。取得はプロセス横断スロットで 1 本に直列化される
MAX_LOAD = 18.0


def race_ids_for_year(year: int) -> list[str]:
    """確定結果のあるレース（race_result_flat）。新しい日付から。"""
    import polars as pl

    if year < 2020:  # 2016〜2019 年は tables が無い/一部のみ。保存済みの確定結果（race_result）のキーから取る
        from src.scraper.storage import HybridStorage

        return sorted(HybridStorage().list_keys("race_result", str(year)), reverse=True)
    f = TABLES / str(year) / "race_result_flat.parquet"
    if not f.exists():
        return []
    d = pl.read_parquet(f, columns=["race_id", "date"]).group_by("race_id").agg(pl.col("date").max().alias("d"))
    return [r for r, _ in sorted(d.iter_rows(), key=lambda x: (str(x[1]), x[0]), reverse=True) if r]


def busy() -> bool:
    out = subprocess.run(["bash", "-c", f"ps -eo args | grep -P '{OTHER_SCRAPERS}' | grep -v grep | wc -l"], capture_output=True, text=True).stdout.strip()
    return bool(out) and int(out) > 0


def load1() -> float:
    try:
        return os.getloadavg()[0]
    except OSError:
        return 0.0


def wait_ok(max_wait: int = 3600) -> None:
    t0 = time.time()
    while time.time() - t0 < max_wait and (busy() or load1() >= MAX_LOAD):
        time.sleep(30)


def load_progress() -> dict:
    if PROGRESS.exists():
        try:
            return json.loads(PROGRESS.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"no_data": {}, "saved": 0, "failed": 0}


def save_progress(p: dict) -> None:
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text(json.dumps(p, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2026, 2025, 2024])
    ap.add_argument("--limit", type=int, default=0, help="新規に取得するレース数の上限（パイロット用）")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="対象数だけ数える")
    a = ap.parse_args()

    from src.scraper.client import _GlobalSlot
    from src.scraper.run import ScraperRunner

    runner = ScraperRunner()
    prog = load_progress()
    todo: list[str] = []
    for y in a.years:
        for rid in race_ids_for_year(y):
            if (rid in prog["no_data"] and not a.retry_failed) or runner.storage.exists("race_trifecta_odds", rid):
                continue
            todo.append(rid)
    print(f"未取得 {len(todo)} レース（年: {a.years}）", flush=True)
    if a.dry_run:
        return

    t0, n, streak = time.time(), 0, 0
    for rid in todo:
        if a.limit and n >= a.limit:
            break
        wait_ok()
        with _GlobalSlot():
            d = runner.scrape_trifecta_odds(rid, skip_existing=True)
        n += 1
        if d:
            prog["saved"] += 1
            prog["no_data"].pop(rid, None)
            streak = 0
        else:
            prog["failed"] += 1
            prog["no_data"][rid] = time.strftime("%Y-%m-%dT%H:%M:%S")
            streak += 1
            if streak >= 8:
                prog["aborted"] = f"連続 {streak} 件失敗（アクセス制限の疑い）: {rid}"
                save_progress(prog)
                print(prog["aborted"], flush=True)
                raise SystemExit(3)
        if n % 25 == 0:
            prog.update({"last_race_id": rid, "seconds": round(time.time() - t0), "remaining": len(todo) - n})
            save_progress(prog)
            print(f"{n}/{len(todo)} 保存 {prog['saved']} 失敗 {prog['failed']} ({rid})", flush=True)
        time.sleep(random.uniform(30, 60) if n % 100 == 0 else random.uniform(4.5, 7.0))
    prog.update({"finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "seconds": round(time.time() - t0)})
    prog.pop("aborted", None)
    save_progress(prog)
    print(json.dumps({k: prog[k] for k in ("saved", "failed")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
