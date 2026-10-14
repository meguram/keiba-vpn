#!/usr/bin/env python3
"""race_index（タイム指数）・race_trainer_comment（調教師コメント）の取り直し。取得頭数が既存より増えるときだけ保存する。再開可能。

対象一覧は --ids-file（race_id を 1 行 1 件）。進捗は --progress に追記し、記録済みの race_id は再実行時に飛ばす。
既定の一覧・進捗ファイルは --category ごとに data/local/meta/rescrape/ 以下に決まる。

例::

  python -m src.scripts.scraping.rescrape_race_index --years 2025 2024 --interval 5
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

from src.utils.project_env import load_project_dotenv

RESCRAPE_DIR = Path("data/local/meta/rescrape")
CATEGORIES = {
    # category: (URL 属性, パーサー属性, 既定の対象一覧, 既定の進捗ファイル)
    "race_index": ("SPEED_INDEX_URL", "speed_parser", RESCRAPE_DIR / "race_index_rescrape_targets.txt", RESCRAPE_DIR / "race_index_progress.jsonl"),
    "race_trainer_comment": ("TRAINER_COMMENT_URL", "trainer_comment_parser", RESCRAPE_DIR / "race_trainer_comment_rescrape_targets.txt", RESCRAPE_DIR / "race_trainer_comment_progress.jsonl"),
}


def _done(progress: Path) -> set[str]:
    if not progress.exists():
        return set()
    return {json.loads(line)["race_id"] for line in progress.read_text(encoding="utf-8").splitlines() if line.strip()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="race_index の取り直し（件数が増えるときだけ保存）")
    ap.add_argument("--category", choices=sorted(CATEGORIES), default="race_index")
    ap.add_argument("--ids-file", type=Path)
    ap.add_argument("--progress", type=Path)
    ap.add_argument("--years", nargs="*", help="対象の開催年（例: 2025 2024）。指定順に処理する")
    ap.add_argument("--interval", type=float, default=5.0, help="レース間の待機秒（ゆらぎ ±30%%）")
    ap.add_argument("--limit", type=int, default=0, help="この回に処理する最大件数（0=無制限）")
    ap.add_argument("--stop-after-no-gain", type=int, default=40,
                    help="件数が増えない結果が連続したら停止（ログイン不全などの検知）")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    load_project_dotenv()
    cat = args.category
    url_attr, parser_attr, ids_default, progress_default = CATEGORIES[cat]
    args.ids_file = args.ids_file or ids_default
    args.progress = args.progress or progress_default
    ids = [x.strip() for x in args.ids_file.read_text(encoding="utf-8").splitlines() if x.strip()]
    years = args.years or sorted({i[:4] for i in ids}, reverse=True)
    done = _done(args.progress)
    todo = [i for y in years for i in ids if i.startswith(str(y)) and i not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[{cat}] 対象 {len(todo)} 件（記録済み {len(done)} 件を除く）", flush=True)
    if args.dry_run or not todo:
        return 0

    from src.scraper.run import ScraperRunner

    runner = ScraperRunner(interval=1.0, auto_login=False)
    args.progress.parent.mkdir(parents=True, exist_ok=True)
    saved = gain_rows = no_gain_streak = partial_streak = 0
    t0 = time.time()

    def fetch(rid: str, n_old: int):
        return runner._fetch_parse_save(
            cat, rid, getattr(runner, url_attr).format(race_id=rid), getattr(runner, parser_attr),
            {"race_id": rid}, skip_existing=False, need_login=True, min_entries=n_old + 1,
        )

    def relogin() -> bool:
        """同一アカウントの別ログインでセッションが無効になった疑い。入り直して会員状態を確認する。"""
        c = runner.client
        c._logged_in, c._premium_ok = False, None
        return bool(c.login()) and c.premium_ok is not False

    for n, rid in enumerate(todo, 1):
        old = runner.storage.load(cat, rid)
        n_old = len((old or {}).get("entries") or [])
        data = fetch(rid, n_old)
        if not data and n_old <= 3:
            partial_streak += 1
            if partial_streak >= 3:  # 部分的な状態(3頭以下)のまま増えない結果が続く → 再ログインして 1 回だけやり直す
                print(f"[{n}] 3 回連続で増えず。再ログインして再試行します", flush=True)
                if not relogin():
                    print("再ログイン後も会員状態が確認できないため停止します", file=sys.stderr)
                    return 3
                partial_streak = 0
                data = fetch(rid, n_old)
        else:
            partial_streak = 0
        n_new = len((data or {}).get("entries") or []) if data else None
        status = "saved" if data else "no_gain"
        saved += status == "saved"
        gain_rows += (n_new - n_old) if data else 0
        no_gain_streak = 0 if status == "saved" else no_gain_streak + 1
        with args.progress.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"race_id": rid, "old": n_old, "new": n_new, "status": status, "ts": int(time.time())}) + "\n")
        if n % 25 == 0:
            rate = n / (time.time() - t0) * 60
            print(f"[{n}/{len(todo)}] 保存 {saved} 件 / 追加 {gain_rows} 行 / {rate:.1f} 件/分", flush=True)
        if no_gain_streak >= args.stop_after_no_gain:
            print(f"件数が増えない結果が {no_gain_streak} 回連続したため停止します（ログイン状態を確認してください）", file=sys.stderr)
            return 3
        time.sleep(args.interval * random.uniform(0.7, 1.3))
    print(f"完了: 保存 {saved} 件 / 追加 {gain_rows} 行", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
