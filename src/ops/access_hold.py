"""取得元からアクセス制限の疑いを受けたときの『保留』。人が解除するまで、取得の実行役は新しいタスクを取らない。

方針（docs/operations/scraping_policy.md）: 制限を受けたら、別の User-Agent・別の IP・別の経路に切り替えて続行しない。
止めて、Slack で知らせ、人が原因を確認してから解除する。保留はファイルに残るので、再起動しても解除されない。

  python -m src.ops.access_hold status
  python -m src.ops.access_hold clear      # 原因を確認し、再開してよいと判断したあとに実行する
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def hold_path() -> Path:
    return Path(os.environ.get("KEIBA_FETCH_HOLD_FILE") or ROOT / "data" / "local" / "meta" / "fetch_access_hold.json")


def read_hold(path: Path | None = None) -> dict[str, Any] | None:
    p = path or hold_path()
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) and d.get("active") else None


def set_hold(reason: str, *, task_id: str = "", path: Path | None = None) -> dict[str, Any]:
    p = path or hold_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    d = {"active": True, "reason": (reason or "")[:2000], "task_id": task_id, "at": datetime.now().astimezone().isoformat(timespec="seconds")}
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)
    return d


def clear_hold(path: Path | None = None) -> bool:
    p = path or hold_path()
    try:
        p.unlink()
        return True
    except OSError:
        return False


def main(argv: list[str] | None = None) -> int:
    a = list(sys.argv[1:] if argv is None else argv)
    cmd = a[0] if a else "status"
    if cmd == "status":
        d = read_hold()
        print(json.dumps(d, ensure_ascii=False) if d else "保留なし")
        return 0
    if cmd == "clear":
        print("解除しました" if clear_hold() else "保留はありません")
        return 0
    print("使い方: python -m src.ops.access_hold status|clear", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
