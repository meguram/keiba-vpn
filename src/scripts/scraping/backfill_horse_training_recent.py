"""出走馬の調教ページ（horse_training）の穴埋め: 学習以外の運用予測・監視で『そのレース日の調教の行』が無い馬を、最新 1 ページの取り直しで埋める。

  python -m src.scripts.scraping.backfill_horse_training_recent --from 2026-04-01 --dry-run          # 対象の数だけ
  python -m src.scripts.scraping.backfill_horse_training_recent --from 2026-04-01 --limit 40         # パイロット
  python -m src.scripts.scraping.backfill_horse_training_recent --dates 2026-10-10 2026-10-11 --limit 700

馬別の調教ページは最新 10 走分を 1 ページに載せる。2026 年 4 月以降に 10 走を超えた馬はほぼ無いので、1 頭 1 リクエストで足りる。
保存は ``horse_training_recent.refresh_one``（同じ日付の行だけ入れ替え、既存の古い行は残す。読めなかった・0 件のページは保存しない）。
スクレイピング規則: 他の取得（開催日ランナー・定期取得・取得系の fork）が動いている間は待つ、1 頭ごとに約 5.5 秒（クライアントの間隔 + 上乗せ 0.5〜1.5 秒）、100 頭ごとに 30〜60 秒、
連続 8 件の失敗で中断。再開できる（進捗: data/local/meta/horse_training_backfill_progress.json）。
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
PROGRESS = ROOT / "data/local/meta/horse_training_backfill_progress.json"
OTHER_SCRAPERS = (
    r"python[0-9.]* -m src\.(scripts\.maintenance\.repair_|scripts\.scraping\.(?!backfill_horse_race_remarks|backfill_horse_training_recent|fetch_status)|"
    r"scripts\.data\.(backfill_trifecta_odds|acq_profile_pilot|acq_horse_lap_pilot)|scraper\.(backfill(?!.* --phase past)|run|export_tables|auto_scrape|raceday|queue))"
)  # 2016〜2019 年の補填（--phase past）はこちらに譲るので待たない（待つと待ち合いになる）。同時取得はプロセス横断スロットで 1 本になる
MAX_LOAD = 18.0
# クライアント自身が 1 リクエストあたり約 4.5〜5 秒空ける（実測: 前日夕方の取り直しは 450 頭 41 分 = 5.5 秒/頭）。ここの上乗せは小さくてよい
EXTRA_SLEEP = (float(os.environ.get("HORSE_TRAINING_BACKFILL_MIN_SEC", "0.5")), float(os.environ.get("HORSE_TRAINING_BACKFILL_MAX_SEC", "1.5")))
STATUSES = ("saved", "unreadable", "empty", "error")


def years_between(start: str, end: str) -> list[int]:
    return list(range(int(start[:4]), int(end[:4]) + 1))


def starts_between(start: str, end: str, dates: list[str] | None = None):
    """(horse_id, date[YYYY-MM-DD]) の出走行。確定結果の flat 表から。"""
    import polars as pl

    rows = []
    for y in years_between(start, end):
        f = TABLES / str(y) / "race_result_flat.parquet"
        if not f.exists():
            continue
        d = pl.read_parquet(f, columns=["race_id", "horse_id", "date"]).with_columns(pl.col("date").cast(pl.Utf8).str.slice(0, 10).alias("d"))
        d = d.filter((pl.col("d") >= start) & (pl.col("d") <= end))
        if dates:
            d = d.filter(pl.col("d").is_in(dates))
        rows += [(str(h), str(x)) for h, x in d.select(["horse_id", "d"]).iter_rows() if h]
    return rows


def missing_counts(starts: list[tuple[str, str]], load_many=None) -> dict[str, int]:
    """馬ごとに『その日の調教の行が保存記録に無い出走』の数。"""
    from src.scripts.data import drift_coverage as C

    horses = sorted({h for h, _ in starts})
    recs = (load_many or C.default_loader())(horses)
    blocks = C.block_dates(recs)
    out: dict[str, int] = {}
    for h, d in starts:
        if d not in blocks.get(h, set()):
            out[h] = out.get(h, 0) + 1
    return out


def busy() -> bool:
    out = subprocess.run(["bash", "-c", f"ps -eo args | grep -P '{OTHER_SCRAPERS}' | grep -v grep | wc -l"], capture_output=True, text=True).stdout.strip()
    return bool(out) and int(out) > 0


def load1() -> float:
    try:
        return os.getloadavg()[0]
    except OSError:
        return 0.0


def wait_ok(max_wait: int = 3600) -> bool:
    """他の取得が終わり、負荷が下がるまで待つ。待ちきれなければ False。"""
    t0 = time.time()
    while busy() or load1() >= MAX_LOAD:
        if time.time() - t0 > max_wait:
            return False
        time.sleep(30)
    return True


def load_progress() -> dict:
    if PROGRESS.exists():
        try:
            return json.loads(PROGRESS.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"done": {}, "counts": {s: 0 for s in STATUSES}}


def save_progress(p: dict) -> None:
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text(json.dumps(p, ensure_ascii=False, indent=1), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default="2026-04-01")
    ap.add_argument("--to", dest="end", default=time.strftime("%Y-%m-%d"))
    ap.add_argument("--dates", nargs="*", default=None, help="この開催日の出走馬だけ（検証用）")
    ap.add_argument("--horses-file", default=None, help="この馬 ID（1 行 1 頭）だけを対象にする（検証用）")
    ap.add_argument("--limit", type=int, default=0, help="取得する馬数の上限（パイロット・1 晩の上限）")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="対象数と所要時間の見積もりだけ")
    a = ap.parse_args(argv)

    starts = starts_between(a.start, a.end, a.dates)
    if a.horses_file:
        only = {x.strip() for x in Path(a.horses_file).read_text(encoding="utf-8").split() if x.strip()}
        starts = [(h, d) for h, d in starts if h in only]
    miss = missing_counts(starts)
    prog = load_progress()
    todo = [h for h, _ in sorted(miss.items(), key=lambda kv: (-kv[1], kv[0]))
            if (h not in prog["done"] or (a.retry_failed and prog["done"][h] != "saved"))]
    n = len(todo) if not a.limit else min(len(todo), a.limit)
    est_h = n * 6.0 / 3600
    print(f"出走 {len(starts)} 行 / 馬 {len({h for h, _ in starts})} 頭 / 調教の行が足りない馬 {len(miss)} 頭 / 未取得 {len(todo)} 頭 / 今回 {n} 頭（見積もり {est_h:.1f} 時間）", flush=True)
    if a.dry_run:
        return 0

    from src.scraper.access_guard import AccessRestrictionDetected
    from src.scraper.client import _GlobalSlot
    from src.scraper.horse_training_recent import refresh_one
    from src.scraper.run import ScraperRunner

    runner = ScraperRunner()
    t0, done, streak = time.time(), 0, 0
    for hid in todo[:n]:
        if not wait_ok():
            prog["aborted"] = "他の取得が終わらず待ちきれなかった"
            save_progress(prog)
            print(prog["aborted"], flush=True)
            return 4
        status = "error"
        try:
            with _GlobalSlot():
                status = refresh_one(runner, hid)["status"]
        except AccessRestrictionDetected as e:
            prog["aborted"] = f"アクセス制限の疑い: {e}"
            save_progress(prog)
            print(prog["aborted"], flush=True)
            return 3
        except Exception as e:  # noqa: BLE001
            print(f"失敗 {hid}: {e}", flush=True)
        done += 1
        prog["done"][hid] = status
        prog["counts"][status] = prog["counts"].get(status, 0) + 1
        streak = streak + 1 if status in ("unreadable", "error") else 0
        if streak >= 8:
            prog["aborted"] = f"連続 {streak} 件が読めない／失敗（アクセス制限またはページ構造の変化の疑い）: {hid}"
            save_progress(prog)
            print(prog["aborted"], flush=True)
            return 3
        if done % 25 == 0:
            prog.update({"seconds": round(time.time() - t0), "remaining": n - done})
            save_progress(prog)
            print(f"{done}/{n} {prog['counts']} ({hid})", flush=True)
        time.sleep(random.uniform(30, 60) if done % 100 == 0 else random.uniform(*EXTRA_SLEEP))
    prog.update({"finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "seconds": round(time.time() - t0)})
    prog.pop("aborted", None)
    save_progress(prog)
    print(json.dumps(prog["counts"], ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
