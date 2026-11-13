"""金曜の週次（weekly-update）の後処理を、ホストで行う（docs/operations/vps_only_fetch.md の段 4 後半）。

weekly-update の担当が VPS に移ると、VPS は取得だけをして、後処理（めぐ指数・PG 同期・予測の成績・追走難度の突き合わせ・ドリフト）を
しない。代わりに、取得とキュー待機を終えたときだけ、GCS の ops/mirror_vps/_weekly_finalize.json に目印（書いた日・時刻と対象の開催日）を書く。
このモジュールはその目印を名前指定で 1 件だけ読み（一覧しない）、まだ処理していない週なら finalize_weekly_update を動かす。
キュー待機が打ち切られた回（終了コード 7）は目印が書かれないので、ここも何もしない。

  python -m src.ops.weekly_finalize [--dry-run]

呼び出しは src/ops/host_postprocess.py の weekly-finalize（担当が VPS に移ったときだけ。土 00 時〜月 08 時 JST の毎時。
VPS のキュー待機が長引いて目印が遅れても拾う）。回の鍵は目印の written_at。済んだ回は logs/weekly_finalize_done.json に残し、二度動かさない。
失敗（partial・error・例外）は済みにせず、次の毎時の回で取り直す（3 回まで）。
費用: 目印 1 件（数百バイト）の読みが週 約 56 回（月 約 240 回）で 1 円未満。後処理がホストで GCS から読む分は、対象の開催日（多くて 2〜4 日 × 36 レース）の
確定結果などで 1 回 約 10〜20MB → 月 約 80MB 以下（転送 約 1.5 円）。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
JST = timezone(timedelta(hours=9))
DONE_FILE = ROOT / "logs" / "weekly_finalize_done.json"
MAX_AGE_DAYS = 6          # これより古い目印（前の週など）では動かさない
MAX_TRIES = 3             # 失敗（partial・error・例外）は次の回（毎時）で取り直す。この回数で諦める

logger = logging.getLogger("ops.weekly_finalize")


def _load_done(path: Path) -> dict[str, Any]:
    """{"done": [回の鍵], "tries": {回の鍵: 試した回数}, "gave_up": [諦めた回の鍵]}"""
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return {"done": [str(x) for x in d.get("done", [])], "tries": {str(k): int(v) for k, v in (d.get("tries") or {}).items()},
                "gave_up": [str(x) for x in d.get("gave_up", [])]}
    except Exception:  # noqa: BLE001
        return {"done": [], "tries": {}, "gave_up": []}


def _save_done(path: Path, rec: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    done = sorted(set(rec["done"]))[-20:]
    tries = dict(sorted((k, v) for k, v in rec["tries"].items() if k not in done)[-20:])       # 鍵は written_at（時刻順）。古いものは捨てる
    gave_up = sorted(set(rec.get("gave_up", [])))[-20:]
    tmp.write_text(json.dumps({"done": done, "tries": tries, "gave_up": gave_up}, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def run(*, backend: Any = None, finalize: Callable[[list[str], dict], dict] | None = None, now: datetime | None = None,
        done_file: Path = DONE_FILE, dry_run: bool = False) -> dict[str, Any]:
    from src.ops.gcs_mirror import read_weekly_handoff

    if backend is None:
        from src.ops.gcs_mirror import GcsBackend

        backend = GcsBackend()
    doc = read_weekly_handoff(backend)
    if not doc:
        return {"status": "skipped", "reason": "no-handoff"}
    try:
        written = float(doc.get("written_at"))
    except (TypeError, ValueError):
        return {"status": "error", "reason": f"bad-written_at:{doc.get('written_at')!r}"}
    key = f"{written:.0f}"                                   # 回の鍵（同じ日に週次を回し直しても別の回になる）
    dates = [str(d) for d in (doc.get("target_dates") or [])]
    base = {"run": key, "run_date": doc.get("run_date"), "target_dates": dates}
    today = (now or datetime.now(JST)).astimezone(JST)
    if (today - datetime.fromtimestamp(written, JST)).days > MAX_AGE_DAYS:
        return {"status": "skipped", "reason": "stale-handoff", **base}
    rec = _load_done(done_file)
    if key in rec["done"]:
        return {"status": "skipped", "reason": "already-done", **base}
    tries = int(rec["tries"].get(key, 0))
    if tries >= MAX_TRIES:
        if key in rec["gave_up"]:
            return {"status": "skipped", "reason": "gave-up", **base}         # 諦めた知らせは 1 回だけ（毎時 失敗を積まない）
        rec["gave_up"].append(key)
        _save_done(done_file, rec)
        return {"status": "error", "reason": f"gave-up-after-{tries}-tries", **base}
    if not dates:
        return {"status": "skipped", "reason": "no-target-dates", **base}
    if dry_run:
        return {"status": "dry-run", **base}
    if finalize is None:
        from src.scraper.auto_scrape import finalize_weekly_update as finalize
    rec["tries"][key] = tries + 1
    _save_done(done_file, rec)
    try:
        out = finalize(dates, {"status": "ok", "source": "vps-weekly-update", "run": key})
    except Exception as e:  # noqa: BLE001 — 次の回（毎時）で取り直す。MAX_TRIES 回で諦めて失敗を返し続ける
        logger.exception("週次の後処理に失敗: %s", e)
        out = {"status": "error", "reason": f"{type(e).__name__}: {e}"}
    status = str(out.get("status") or "ok")
    if status == "ok":
        rec["done"].append(key)
        rec["tries"].pop(key, None)
        _save_done(done_file, rec)
    try:
        from src.scraper.auto_scrape import _save_status

        _save_status("weekly-finalize", {**out, **base, "try": tries + 1})
    except Exception as e:  # noqa: BLE001
        logger.warning("状態の保存をスキップ: %s", e)
    return {"status": status, **base, "try": tries + 1}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="目印を読んで、動かすかどうかだけ表示する")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    from src.utils.project_env import load_project_dotenv

    load_project_dotenv()
    res = run(dry_run=a.dry_run)
    print(json.dumps(res, ensure_ascii=False))
    return 0 if res.get("status") in ("ok", "skipped", "dry-run") else 1     # partial・error は cron の記録で失敗にする


if __name__ == "__main__":
    sys.exit(main())
