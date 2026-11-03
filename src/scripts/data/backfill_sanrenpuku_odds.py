"""3連複オッズ（race_pair_odds の sanrenpuku）の過去分の一括取得（ホストの stg で実行。再開可）。

python -m src.scripts.data.backfill_sanrenpuku_odds --years 2025 --limit 20     # パイロット
python -m src.scripts.data.backfill_sanrenpuku_odds --years 2025 2024 2023 2022 # 新しい年から順

2019〜2025 年の保存済み race_pair_odds は、3連複対応より前に取ったため sanrenpuku が空。type=7 だけを 1 レース 1 リクエストで取り、
既存文書の sanrenpuku キーだけを埋めて HybridStorage.save で保存し直す（馬連・ワイド・馬単など他のキーは変えない）。
- race_pair_odds が無いレース、sanrenpuku が既に入っているレースは対象外（上書きしない）。
- 読めなかったレースは進捗ファイルに記録して次回以降スキップ（--retry-failed で再試行）。連続 8 件の失敗でアクセス制限を疑って中断（終了コード 3）。
- リクエストは NetkeibaClient のプロセス横断スロット（_GlobalSlot）を使い、1 件ごとに 4.5〜7 秒、100 件ごとに 30〜60 秒休む。
- 修復系・取得系の他のスクレイピングが動いている間は待つ（backfill_trifecta_odds と同じ）。ただし長時間稼働する
  `src.scraper.backfill --phase past`（馬ページ）は待ち切れないので、待つのは --max-wait 秒までとし、以降はスロットで直列化して続ける。
進捗: data/local/meta/sanrenpuku_backfill_progress.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import time
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TABLES = ROOT / "data/page_reference/tables"
PROGRESS = ROOT / "data/local/meta/sanrenpuku_backfill_progress.json"
OTHER_SCRAPERS = r"python[0-9.]* -m src\.(scripts\.maintenance\.repair_|scripts\.scraping\.(?!backfill_horse_race_remarks|fetch_status)|scraper\.(run|export_tables)|scripts\.data\.backfill_trifecta_odds)"
LONG_RUNNING = r"python[0-9.]* -m src\.scraper\.backfill"
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


def _count(pattern: str) -> int:
    out = subprocess.run(["bash", "-c", f"ps -eo args | grep -P '{pattern}' | grep -v grep | wc -l"], capture_output=True, text=True).stdout.strip()
    return int(out) if out else 0


def busy() -> bool:
    return _count(OTHER_SCRAPERS) > 0


def load1() -> float:
    try:
        return os.getloadavg()[0]
    except OSError:
        return 0.0


def wait_ok(max_wait: int) -> None:
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


def fetch_sanrenpuku(session, parser_cls, race_id: str) -> list[dict] | None:
    """type=7 だけを取って [{"combo": [a,b,c], "odds": x, "popularity": n}, ...] にする。読めなければ None。"""
    try:
        resp = session.get(parser_cls.SANRENPUKU_API.format(race_id=race_id), timeout=20)
        if resp.status_code != 200:
            return None
        d = resp.json()
        if d.get("status") not in parser_cls.LIVE_STATUSES:
            return None
        raw = (d.get("data") or {}).get("odds", {}).get("7", {})
    except Exception:
        return None
    out = []
    for key, arr in raw.items():
        if not isinstance(arr, list) or not arr or len(key) != 6 or not key.isdigit():
            continue
        nums = [int(key[i:i + 2]) for i in range(0, 6, 2)]
        if 0 in nums or len(set(nums)) != 3:
            continue
        out.append({"combo": sorted(nums), "odds": parser_cls._safe_float(arr[0]), "popularity": parser_cls._safe_int(arr[2]) if len(arr) > 2 else 0})
    out.sort(key=lambda e: tuple(e["combo"]))
    if not out:
        return None
    n = len({x for e in out for x in e["combo"]})
    if len(out) != comb(n, 3):                      # 欠けた応答は保存しない（組合せ数が C(頭数,3) に合わない）
        return None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2025, 2024, 2023, 2022])
    ap.add_argument("--limit", type=int, default=0, help="新規に取得するレース数の上限（パイロット用）")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="対象数だけ数える")
    ap.add_argument("--max-wait", type=int, default=900, help="他の取得が動いている間に待つ最大秒数（1 回の待機あたり）")
    a = ap.parse_args()

    from src.scraper.client import _GlobalSlot
    from src.scraper.parsers import OddsParser
    from src.scraper.run import ScraperRunner

    runner = ScraperRunner()
    prog = load_progress()
    todo: list[str] = []
    for y in a.years:
        for rid in race_ids_for_year(y):
            if rid in prog["no_data"] and not a.retry_failed:
                continue
            todo.append(rid)
    print(f"候補 {len(todo)} レース（年: {a.years}）。race_pair_odds が無い・sanrenpuku が既にあるものは取得時に飛ばします", flush=True)
    if a.dry_run:
        return

    t0, n, streak, done = time.time(), 0, 0, 0
    for rid in todo:
        if a.limit and done >= a.limit:
            break
        doc = runner.storage.load("race_pair_odds", rid)
        if not isinstance(doc, dict):
            continue
        if doc.get("sanrenpuku"):
            continue
        wait_ok(a.max_wait)
        with _GlobalSlot():
            rows = fetch_sanrenpuku(runner.client._session, OddsParser, rid)
        n += 1
        if rows:
            doc["sanrenpuku"] = rows
            ok = runner.storage.save("race_pair_odds", rid, doc)
            if ok is False:
                prog["failed"] += 1
                prog["no_data"][rid] = time.strftime("%Y-%m-%dT%H:%M:%S") + " save_failed"
                streak += 1
            else:
                prog["saved"] += 1
                prog["no_data"].pop(rid, None)
                streak = 0
                done += 1
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
