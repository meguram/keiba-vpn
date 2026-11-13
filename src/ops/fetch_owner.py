"""取得タスクの『持ち主』の切替。同じタスクをホストの cron と VPS の実行役が二重に取らないための排他。

``config/fetch_owner.json`` の ``agent_tasks`` に載っているタスクは VPS の実行役（fetch_agent）の担当で、
ホストの ``scripts/auto_scrape.sh`` は実行しない（``check`` が終了コード 10 を返す）。
VPS 上（``KEIBA_RUN_LOCATION=vps``）では常に実行してよい。ファイルが無い・壊れている場合は『誰も奪わない』（既存の cron が動く）。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

SKIP_EXIT = 10
DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "fetch_owner.json"


def agent_tasks(path: Path | None = None) -> set[str]:
    p = Path(os.environ.get("KEIBA_FETCH_OWNER_FILE") or path or DEFAULT_PATH)
    try:
        return {str(x) for x in (json.loads(p.read_text(encoding="utf-8")).get("agent_tasks") or [])}
    except (OSError, ValueError, AttributeError):
        return set()


# ホストの常駐ランナー（1 日居座るループ）は、VPS の実行役では個別タスクに置き換わる。置き換え先のどれかを VPS が担当したらランナーは止める。
RUNNER_REPLACED_BY: dict[str, frozenset[str]] = {
    "raceday-runner": frozenset({"race-odds-t60", "race-odds-t15", "race-odds-t10", "race-odds-t05", "race-odds-t02"}),
    "raceday-weight-runner": frozenset({"race-weight-check"}),
    "raceday-result-runner": frozenset({"race-result-t15"}),
}


# ランナーの取得をやめてよい条件: このタスクがすべて VPS の担当になったとき。一部だけ（例: 直前オッズだけ）では、
# ホストがやめると残りの取得（T-60・T-15）を誰も取らなくなるので、やめない。
RUNNER_CORE: dict[str, frozenset[str]] = {
    "raceday-runner": frozenset({"race-odds-t60", "race-odds-t15"}),
    "raceday-weight-runner": frozenset({"race-weight-check"}),
    "raceday-result-runner": frozenset({"race-result-t15"}),
}


def runner_fetch_moved(task: str, *, env: dict[str, str] | None = None, path: Path | None = None) -> bool:
    """ホストの常駐ランナー（raceday-runner など）の『取得』が VPS に移ったか。移っていればランナーは取得をせず、
    保存済みのデータ（GCS）で後処理（予測の起動・馬場速度・成績の集計）だけをする。``KEIBA_HOST_NO_FETCH=1`` でも同じ扱い。"""
    env = os.environ if env is None else env
    if env.get("KEIBA_RUN_LOCATION", "").strip().lower() == "vps":
        return False
    if env.get("KEIBA_HOST_NO_FETCH", "").strip() == "1":
        return True
    core = RUNNER_CORE.get(task)
    return bool(core) and core <= agent_tasks(path)


def host_must_skip(task: str, *, env: dict[str, str] | None = None, path: Path | None = None) -> bool:
    """このタスクをホストで実行しないか。常駐ランナーは取得が VPS に移っても見送らない（後処理と予測が中にあるため。
    ランナー自身が runner_fetch_moved で取得だけを飛ばす）。"""
    env = os.environ if env is None else env
    if env.get("KEIBA_RUN_LOCATION", "").strip().lower() == "vps":
        return False
    owned = agent_tasks(path)
    if task in owned:                     # ランナー名そのものが担当表にあれば、従来どおり丸ごと見送る
        return True
    return False


def main(argv: list[str] | None = None) -> int:
    a = list(sys.argv[1:] if argv is None else argv)
    if len(a) == 2 and a[0] == "check":
        if host_must_skip(a[1]):
            print(f"{a[1]} は VPS の取得の実行役の担当（config/fetch_owner.json）。このマシンでは実行しません")
            return SKIP_EXIT
        return 0
    print("usage: python -m src.ops.fetch_owner check <task>", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
