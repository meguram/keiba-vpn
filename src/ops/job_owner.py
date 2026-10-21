"""ホストの cron ジョブの『持ち主』の切替。GCP / VPS が引き継いだジョブを、ホストの cron が二重に実行しないための排他。

``config/job_owner.json`` の ``owners``（``{cron_job_id: "gcp"|"vps"|"retire"}``）に載ったジョブは、ホストでは実行しない。
``scripts/cron/cron_run.sh`` が実行前に ``python -m src.ops.job_owner check <job_id>`` を呼び、終了コード 10 なら
コマンドを実行せず『担当は GCP』と出力して正常終了する（cron_run の記録には成功として残る＝監視ポータルが『止まった』と誤判定しない）。

- 切替の順序（二重実行を避ける）: ① GCP 側を並走で動かして一致を確認 → ② ここに載せる（ホストは何もしなくなる・crontab はそのまま）
  → ③ 観察期間の後に crontab の行をコメントアウト。戻すときは逆順（行を戻す → ここから外す）。
- 載せる・外すのは ``set`` / ``unset``（``can-stop`` は crontab の行を止めてよいかの確認＝担当が渡っていること）。ファイルが無い・壊れている場合は『誰も奪わない』（ホストの cron がこれまでどおり動く）。
- ``KEIBA_RUN_LOCATION`` が gcp / vps の環境では常に実行してよい（担当側そのもの）。ホストに残すジョブ（destination=host）は載せられない（テストで検出）。
- 取得系で VPS の実行役が担当するタスクの排他は ``fetch_owner.py``（auto_scrape.sh 経由）。こちらは cron_run.sh 経由の全ジョブ向け。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

SKIP_EXIT = 10
DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "job_owner.json"
OWNER_VALUES = ("gcp", "vps", "retire")


def owners(path: Path | None = None) -> dict[str, str]:
    p = Path(os.environ.get("KEIBA_JOB_OWNER_FILE") or path or DEFAULT_PATH)
    try:
        raw = json.loads(p.read_text(encoding="utf-8")).get("owners") or {}
        return {str(k): str(v) for k, v in raw.items()}
    except (OSError, ValueError, AttributeError):
        return {}


def host_must_skip(job_id: str, *, env: dict[str, str] | None = None, path: Path | None = None) -> bool:
    env = os.environ if env is None else env
    if env.get("KEIBA_RUN_LOCATION", "").strip().lower() in ("gcp", "vps"):
        return False
    return job_id in owners(path)


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"version": 1, "owners": {}}


def _write(path: Path, doc: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def can_switch(job_id: str) -> tuple[bool, str]:
    """担当を渡してよい cron_job_id か。job_map に載っていて、ホストに残すジョブ（destination=host）が一つも含まれないこと。"""
    from src.ops import job_map

    jobs = [j for j in job_map.jobs() if (j.get("origin") or {}).get("cron_job_id") == job_id]
    if not jobs:
        return False, f"{job_id} は config/gcp_job_map.json の origin.cron_job_id に無い"
    host = [j["id"] for j in jobs if j["destination"] == "host"]
    if host:
        return False, f"ホストに残すジョブを含む: {host}"
    return True, ""


def set_owner(job_id: str, owner: str, *, path: Path | None = None, today: str | None = None, force: bool = False) -> None:
    if owner not in OWNER_VALUES:
        raise ValueError(f"owner は {OWNER_VALUES}")
    ok, why = can_switch(job_id)
    if not ok and not force:
        raise ValueError(why)
    p = Path(os.environ.get("KEIBA_JOB_OWNER_FILE") or path or DEFAULT_PATH)
    doc = _read(p)
    doc.setdefault("owners", {})[job_id] = owner
    from datetime import date

    doc.setdefault("switched", {})[job_id] = today or date.today().isoformat()
    _write(p, doc)


def unset_owner(job_id: str, *, path: Path | None = None) -> None:
    p = Path(os.environ.get("KEIBA_JOB_OWNER_FILE") or path or DEFAULT_PATH)
    doc = _read(p)
    doc.setdefault("owners", {}).pop(job_id, None)
    doc.setdefault("switched", {}).pop(job_id, None)
    _write(p, doc)


def main(argv: list[str] | None = None) -> int:
    a = list(sys.argv[1:] if argv is None else argv)
    if len(a) == 2 and a[0] == "check":
        if host_must_skip(a[1]):
            print(f"{a[1]} は {owners().get(a[1])} の担当（config/job_owner.json）。このマシンでは実行しません")
            return SKIP_EXIT
        return 0
    if a == ["list"]:
        for k, v in sorted(owners().items()):
            print(f"{k}\t{v}")
        return 0
    if len(a) == 3 and a[0] == "set":
        try:
            set_owner(a[1], a[2])
        except ValueError as e:
            print(f"できません: {e}", file=sys.stderr)
            return 2
        print(f"担当を {a[2]} にしました: {a[1]}（ホストの cron はこのジョブを実行しなくなります）")
        return 0
    if len(a) == 2 and a[0] == "unset":
        unset_owner(a[1])
        print(f"担当を外しました: {a[1]}（ホストの cron がこのジョブを再び実行します）")
        return 0
    if len(a) == 2 and a[0] == "can-stop":
        # crontab の行をコメントアウトしてよいか: 担当が渡っていて（owners に載る）、ホストに残すジョブでないこと
        ok, why = can_switch(a[1])
        if not ok:
            print(why, file=sys.stderr)
            return 2
        if a[1] not in owners():
            print(f"{a[1]} はまだ担当が渡っていない（先に set。二重実行を避ける順序）", file=sys.stderr)
            return 3
        return 0
    print("usage: python -m src.ops.job_owner check <id> | list | set <id> <gcp|vps|retire> | unset <id> | can-stop <id>", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
