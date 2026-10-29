"""実行役（fetch_agent）が子プロセスとして呼ぶ取得の入口。最後に ``FETCH_RESULT {json}`` を 1 行出す。

  python -m src.ops.fetch_child auto <mode>                 # auto_scrape のモード
  python -m src.ops.fetch_child race <task> <race_id> <YYYYMMDD>   # race-odds-t60|t15|t10|t05|t02 / race-weight-check / race-result-t15
  python -m src.ops.fetch_child selfcheck                   # 取得に必要なモジュールを import できるか（VPS の venv 確認用）

auto_scrape.main() は例外を飲み込んで終了コード 0 を返すため、ここで状態ファイルを読んで error を終了コード 3 にする。
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

RESULT_PREFIX = "FETCH_RESULT "


def _shadow() -> bool:
    return os.environ.get("KEIBA_FETCH_SHADOW", "").strip() == "1"


def _emit(result: dict) -> None:
    if _shadow():
        result = {**result, "shadow": True}
    print(RESULT_PREFIX + json.dumps(result, ensure_ascii=False, default=str), flush=True)


def shadow_guard(task: str) -> int | None:
    """影モード（KEIBA_FETCH_SHADOW=1）のとき、対象外のタスクは実行せず、書き込みを shadow/ 以下に限る。問題があれば終了コード、無ければ None。"""
    from src.ops import fetch_shadow

    if task not in fetch_shadow.load_config().task_names():
        _emit({"errors": [f"shadow_not_allowed: {task}"], "access_restriction": False})
        return 4
    from src.ops import shadow_storage

    shadow_storage.install()
    return None


def race_task_list(task: str) -> list[str]:
    from src.ops.fetch_task import RACE_ODDS_TASKS, RACE_RESULT_TASK, RACE_WEIGHT_TASK
    from src.scraper import auto_scrape_queue as q
    from src.scraper.raceday_weight_watch import WEIGHT_WATCH_TASKS

    if task == RACE_WEIGHT_TASK:
        return list(WEIGHT_WATCH_TASKS)
    if task == RACE_RESULT_TASK:
        return ["race_result_on_time"]
    kind = RACE_ODDS_TASKS[task]
    if kind == "t60":
        return q.enabled_tasks(q.ODDS_T60_TASKS)
    if kind == "t15":
        return q.enabled_tasks(q.T15_RACE_TASKS)
    return [q.LATE_KIND_TASK[kind]]


def run_race(task: str, race_id: str, date: str) -> int:
    from src.ops.fetch_task import RACE_ODDS_TASKS, RACE_WEIGHT_TASK
    from src.scraper import auto_scrape_queue as q

    kind = RACE_ODDS_TASKS.get(task)
    if kind in q.LATE_KIND_TASK and not q.late_snapshots_enabled():
        _emit({"skipped": "late_snapshots_disabled", "errors": []})
        return 0
    tasks = race_task_list(task)
    from src.scraper.run import ScraperRunner

    runner = ScraperRunner()
    errors = q._run_race_tasks_direct(runner, race_id, date, tasks)
    result: dict = {"tasks": tasks, "errors": errors,
                    "access_restriction": any("access_restriction" in e for e in errors)}
    if task == RACE_WEIGHT_TASK and not errors:
        from src.scraper.raceday_weight_watch import weights_announced, weights_ratio

        card = runner.storage.load("race_shutuba", race_id)
        ratio = weights_ratio(card)
        result.update({"announced": bool(weights_announced(card, ratio)), "ratio": round(float(ratio), 3)})
    _emit(result)
    return 3 if errors else 0


def run_auto(mode: str) -> int:
    started = datetime.now(timezone.utc)
    from src.scraper import auto_scrape as a

    if mode == "shutuba-audit":
        os.environ.setdefault("KEIBA_SHUTUBA_AUDIT_DIRECT", "1")      # 実行役にはキューを消化する常駐が無いので、その場で取り直す
    a.script_basic_config()
    try:
        a.TASKS[mode]()
    except Exception as e:  # noqa: BLE001
        _emit({"errors": [f"{type(e).__name__}: {e}"], "access_restriction": "access_restriction" in str(e)})
        return 1
    st = a._load_status().get(mode) or {}
    last = st.get("last_run")
    fresh = True
    if last:
        try:
            fresh = datetime.fromisoformat(last).astimezone(timezone.utc) >= started
        except ValueError:
            fresh = True
    failed = fresh and st.get("status") == "error"
    _emit({"status": st.get("status"), "errors": [str(st.get("error"))] if failed else [],
           "reason": st.get("reason")})
    return 3 if failed else 0


def selfcheck() -> int:
    t0 = time.time()
    missing: list[str] = []
    for m in ("requests", "bs4", "google.cloud.storage", "src.scraper.run", "src.scraper.auto_scrape",
              "src.scraper.auto_scrape_queue", "src.scraper.raceday_weight_watch", "src.ops.fetch_task"):
        try:
            __import__(m)
        except Exception as e:  # noqa: BLE001
            missing.append(f"{m}: {type(e).__name__}: {e}")
    rss_mb = None
    try:
        import resource

        rss_mb = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    except Exception:  # noqa: BLE001
        pass
    _emit({"missing": missing, "import_rss_mb": rss_mb, "sec": round(time.time() - t0, 2)})
    return 1 if missing else 0


def main(argv: list[str] | None = None) -> int:
    a = list(sys.argv[1:] if argv is None else argv)
    if _shadow() and len(a) >= 2 and a[0] in ("auto", "race"):
        rc = shadow_guard(a[1])
        if rc is not None:
            return rc
    if len(a) == 2 and a[0] == "auto":
        return run_auto(a[1])
    if len(a) == 4 and a[0] == "race":
        return run_race(a[1], a[2], a[3])
    if a == ["selfcheck"]:
        return selfcheck()
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
