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
}


def host_must_skip(task: str, *, env: dict[str, str] | None = None, path: Path | None = None) -> bool:
    env = os.environ if env is None else env
    if env.get("KEIBA_RUN_LOCATION", "").strip().lower() == "vps":
        return False
    owned = agent_tasks(path)
    return task in owned or bool(RUNNER_REPLACED_BY.get(task, frozenset()) & owned)


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
