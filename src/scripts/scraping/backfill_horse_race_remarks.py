"""馬の戦績ページの 備考・厩舎ｺﾒﾝﾄ・ｽﾀｰﾄ指数・追走指数・上がり指数 を、新カテゴリ horse_race_remarks に取り込む（ホストの stg で実行）。

python -m src.scripts.scraping.backfill_horse_race_remarks --recent-years 3 --limit 30      # パイロット
python -m src.scripts.scraping.backfill_horse_race_remarks --recent-years 3                 # 直近 3 年に走った馬から全頭（再開可）
python -m src.scripts.scraping.backfill_horse_race_remarks --recent-years 6 --all-before   # さらに古い馬も

1 頭 1 リクエスト。ordering は直近に走った馬が先。他のセッション/定期ジョブのアクセスが止まるのを待ってから取得し（既存の NetkeibaClient の
レート制御・AccessGuard に従う）、アクセス制限の疑い・連続失敗で中断する。進捗は data/local/meta/horse_race_remarks_progress.json。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TABLES = ROOT / "data/page_reference/tables"
PROGRESS = ROOT / "data/local/meta/horse_race_remarks_progress.json"
PROGRESS_DAILY = ROOT / "data/local/meta/horse_race_remarks_daily.json"
OTHER_SCRAPERS = r"python[0-9.]* -m src\.(scripts\.maintenance\.repair_|scripts\.scraping\.(?!backfill_horse_race_remarks)|scraper\.(backfill(?!.* --phase past)|run|export_tables))"  # --phase past はこちらに譲るので待たない（待ち合いになる）


def horse_ids(recent_years: int, include_older: bool) -> list[str]:
    """直近に走った順（最終出走日の降順）。recent_years 年より前にだけ走った馬は include_older のときだけ後ろに足す。"""
    import polars as pl

    last: dict[str, str] = {}
    this_year = int(time.strftime("%Y"))
    for y in range(this_year, 2019, -1):
        f = TABLES / str(y) / "race_result_flat.parquet"
        if not f.exists():
            continue
        d = pl.read_parquet(f, columns=["horse_id", "date"]).group_by("horse_id").agg(pl.col("date").max().alias("d"))
        for hid, dt in d.iter_rows():
            if hid and (hid not in last or str(dt) > last[hid]):
                last[hid] = str(dt)
    cutoff = f"{this_year - recent_years + 1}-01-01"
    ids = sorted((h for h, d in last.items() if d >= cutoff), key=lambda h: last[h], reverse=True)
    if include_older:
        ids += sorted((h for h, d in last.items() if d < cutoff), key=lambda h: last[h], reverse=True)
    return ids


def recent_runs(days: int) -> dict[str, tuple[str, str]]:
    """直近 days 日に走った馬 → (最終出走の race_id, 日付)。日次の増分取得用。"""
    import datetime as dt

    import polars as pl

    since = (dt.date.today() - dt.timedelta(days=days)).isoformat()
    out: dict[str, tuple[str, str]] = {}
    this_year = int(time.strftime("%Y"))
    for y in (this_year, this_year - 1):
        f = TABLES / str(y) / "race_result_flat.parquet"
        if not f.exists():
            continue
        d = pl.read_parquet(f, columns=["horse_id", "date", "race_id"]).filter(pl.col("date").cast(pl.Utf8) >= since)
        for hid, dt_, rid in d.sort("date").iter_rows():
            if hid:
                out[hid] = (str(rid), str(dt_)[:10])
    return out


def busy() -> bool:
    """他のスクレイピング（修復・バックフィル・エクスポート）のプロセスが動いている。

    取得イベントの更新時刻は見ない（自分の取得でも更新されるので、毎回自分で待つことになる）。同時アクセスの調整は
    NetkeibaClient のレート制御とプロセス横断スロットに任せる。"""
    out = subprocess.run(["bash", "-c", f"ps -eo args | grep -P '{OTHER_SCRAPERS}' | grep -v grep | wc -l"], capture_output=True, text=True).stdout.strip()
    return bool(out) and int(out) > 0


def wait_idle(max_wait: int = 3600) -> None:
    t0, quiet = time.time(), 0
    while quiet < 3 and time.time() - t0 < max_wait:
        quiet = 0 if busy() else quiet + 1
        time.sleep(10)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recent-years", type=int, default=3)
    ap.add_argument("--all-before", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="新規に取得する頭数の上限（パイロット用）")
    ap.add_argument("--no-wait-idle", action="store_true")
    ap.add_argument("--ran-within-days", type=int, default=0, help="日次の増分: 直近 N 日に走った馬のうち、直近の出走の行が保存済みの戦績に無い馬だけ取り直す")
    a = ap.parse_args()
    from src.scraper.horse_race_remarks import run
    from src.scraper.run import ScraperRunner

    last_run = None
    if a.ran_within_days:
        last_run = recent_runs(a.ran_within_days)
        ids = sorted(last_run, key=lambda h: last_run[h][1], reverse=True)
        print(f"対象 {len(ids)} 頭（直近 {a.ran_within_days} 日に走った馬。直近の出走の行が無い馬だけ取り直す）", flush=True)
    else:
        ids = horse_ids(a.recent_years, a.all_before)
        print(f"対象 {len(ids)} 頭（直近 {a.recent_years} 年に走った馬から）", flush=True)
    t0 = time.time()
    st = run(ScraperRunner(), ids, limit=a.limit, last_run=last_run, wait_idle=None if a.no_wait_idle else (lambda: wait_idle() if busy() else None))
    st["seconds"] = round(time.time() - t0)
    st["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    prog = PROGRESS_DAILY if a.ran_within_days else PROGRESS
    prog.parent.mkdir(parents=True, exist_ok=True)
    prog.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(st, ensure_ascii=False, indent=1), flush=True)
    if st["aborted"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
