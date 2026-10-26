"""v5/v6 の血統 ALL_BEST（PX）のリッジを、次の半年（1/1・7/1）に入る前に作っておく（Jobs ``px-ridge-window``。毎週月曜）。

    python -m src.ops.jobs.px_ridge [--today YYYY-MM-DD]

ホストの ``scripts/cron/px_ridge_window.sh`` と同じ判定を Python に移したもの: 新しい半年の初日 T0 の 30 日前以降で、まだ ``ridge_<T0>.npz`` が無いときだけ
``build_px_state --windows <T0>`` を実行する（それ以外は何もしない）。リッジは T0 より前の確定結果で学習する。祖先表・SVD は固定（state/px_v5 にある）。
無いまま新しい半年に入ると、最初の予測で自動学習して遅い／失敗しうる。入力: state/px_v5・履歴表の race_result_flat・学習表（モデル v4_train_table）。出力: state/px_v5/ridge_<T0>.npz。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[3]
LEAD_DAYS = 30


def next_window(today: date) -> tuple[str, int]:
    """(次の半年の初日 T0, あと何日)。"""
    t0 = date(today.year, 7, 1) if today.month <= 6 else date(today.year + 1, 1, 1)
    return t0.isoformat(), (t0 - today).days


def state_dir() -> Path:
    return Path(os.environ.get("KEIBA_INFERENCE_DIR") or ROOT / "data/local/inference") / "state" / "px_v5"


def run(today: date | None = None, sdir: Path | None = None, runner: Callable[[str], None] | None = None) -> dict[str, Any]:
    today = today or datetime.now(timezone(timedelta(hours=9))).date()
    sdir = Path(sdir or state_dir())
    t0, days = next_window(today)
    if days > LEAD_DAYS:
        return {"status": "skipped", "reason": f"まだ早い（T0={t0} まであと {days} 日。{LEAD_DAYS} 日より先）"}
    if (sdir / f"ridge_{t0}.npz").exists():
        return {"status": "skipped", "reason": f"ridge_{t0}.npz は既にある"}
    if not (sdir / "svd.npz").exists():
        return {"status": "error", "reason": "PX の状態（svd.npz）が無い。state/px_v5 の種まきを確認"}
    (runner or _build)(t0)
    if not (sdir / f"ridge_{t0}.npz").exists():
        return {"status": "error", "reason": f"ridge_{t0}.npz ができていない"}
    return {"status": "ok", "created": f"ridge_{t0}.npz", "t0": t0}


def _build(t0: str) -> None:
    subprocess.run([sys.executable, "-m", "src.scripts.data.build_px_state", "--windows", t0], check=True, cwd=str(ROOT),
                   env={**os.environ, "PYTHONPATH": str(ROOT), "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "4")})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--today", default=None, help="YYYY-MM-DD（動作確認用）")
    ns = ap.parse_args(argv)
    try:
        res = run(date.fromisoformat(ns.today) if ns.today else None)
    except subprocess.CalledProcessError as exc:
        res = {"status": "error", "reason": f"build_px_state が失敗（終了コード {exc.returncode}）"}
    print(json.dumps(res, ensure_ascii=False))
    return 1 if res["status"] == "error" else 0


if __name__ == "__main__":
    sys.exit(main())
